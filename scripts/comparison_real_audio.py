"""Fair, real-audio comparison: Mozhi vs Ollama (llama3.1:8b), two prompt
variants, scored with evaluate_accuracy.py's own WER / domain-aware
catch-rate / regression machinery on the same 32 independently-collected
volunteer recordings (data/volunteer_audio).

Why this script exists: comparison_benchmark.py's 34-sentence batch
(tests/batch_cases.py) is Mozhi's OWN regression fixture set, authored
against its dictionary - scoring Mozhi's accuracy on it and calling that
"vs Ollama" is circular. This script instead reuses the real-audio
evaluation used to produce reports/accuracy_evaluation_summary.txt, so
every corrector is scored against text nobody involved in building Mozhi
wrote to make Mozhi look good.

Method
------
- 32 pairs of (cached STT transcript, human-written expected text), same
  set evaluate_accuracy.py uses. Only cached transcripts are read - this
  script never runs Whisper.
- Three correctors per file: Mozhi's app.pipeline.correct_text (unchanged),
  and two Ollama prompts (frozen below, never edited after the run starts):
    naive           - the same instruction used in comparison_benchmark.py
    dictionary-aware - the same instruction plus the full 319-term list
                       from data/domain_terms.json
- Ollama is called with temperature 0 and a fixed seed (deterministic
  decoding as far as Ollama's backend supports - see the Caveats section
  of the generated report for what that does and doesn't guarantee), after
  one untimed warm-up call to move the model into memory before any timed
  call.
- Scoring reuses scripts/evaluate_accuracy.py's evaluate_pair() unchanged.
  Mozhi's corrector already returns real per-span matches. An Ollama
  corrector only returns a corrected string, so this script synthesizes
  match spans from a word-level diff between the STT input and the LLM's
  output (see synthetic_matches() below) and feeds evaluate_pair() the
  same Match-shaped objects Mozhi's pipeline would produce - this is what
  "same machinery" means here: identical region-finding, identical
  phonetic/domain-relevance categorization, identical WER math, applied to
  all three correctors alike.
- Latency: one wall-clock call per file per corrector (perf_counter around
  an in-process call for Mozhi, around the HTTP round trip for Ollama),
  over the same 32 full transcripts - not the short batch-case sentences.

Run:
    PYTHONPATH=. python3 scripts/comparison_real_audio.py

Needs Ollama reachable at OLLAMA_HOST with OLLAMA_MODEL pulled (see
comparison_benchmark.py); stops immediately, before touching anything, if
not. Writes reports/comparison_real_audio.txt and .csv.
"""
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import requests

from app.dictionary import get_term_strings
from app.pipeline import correct_text
from scripts.comparison_benchmark import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    call_ollama_inference,
    ollama_models_available,
)
from scripts.evaluate_accuracy import (
    AUDIO_DIR,
    CATEGORIES,
    PHONETIC,
    STT_DIR,
    Block,
    FileResult,
    Row,
    align,
    catch_rate,
    catch_stats,
    discover_pairs,
    evaluate_pair,
    to_words,
)

REPO_ROOT = Path(__file__).parent.parent
REPORT_TXT_PATH = REPO_ROOT / "reports" / "comparison_real_audio.txt"
REPORT_CSV_PATH = REPO_ROOT / "reports" / "comparison_real_audio.csv"

SEED = 42
OLLAMA_OPTIONS = {"temperature": 0, "seed": SEED}
WARMUP_PROMPT = "Reply with exactly one word: OK."

# --- Frozen prompts. Set once at import time; nothing below this point may edit
# these strings after the run starts (per the task: freeze both prompts before
# running). NAIVE_PROMPT_TEMPLATE is imported, not copied, so it is byte-for-byte
# the same template comparison_benchmark.py uses on the 34-sentence batch.
from scripts.comparison_benchmark import OLLAMA_PROMPT_TEMPLATE as NAIVE_PROMPT_TEMPLATE  # noqa: E402

_DOMAIN_TERMS = ", ".join(get_term_strings())
DICT_PROMPT_TEMPLATE = (
    "Correct any technical/domain-specific terminology transcription errors in the "
    "following text using ONLY this list of valid technical terms: " + _DOMAIN_TERMS + ". "
    "If a word or phrase in the text is a mis-transcription of one of these terms, "
    "replace it with the correct term. Do not introduce any other changes. Return "
    "only the corrected text, no explanation: {text}"
)

CORRECTOR_NAMES = ("mozhi", "ollama_naive", "ollama_dict")
CORRECTOR_LABELS = {
    "mozhi": "Mozhi",
    "ollama_naive": f"Ollama {OLLAMA_MODEL} (naive prompt)",
    "ollama_dict": f"Ollama {OLLAMA_MODEL} (dictionary-aware prompt)",
}


# --- synthetic match construction, so an LLM's plain corrected string can be
# scored through evaluate_accuracy.py's own region/regression machinery ------

def synthetic_matches(stt_words: list, corrected_text: str, method: str) -> list:
    """Match-shaped objects (start, end, original, replacement, method; char
    offsets into the STT text) derived from a word-level diff between the STT
    input and an LLM's corrected output, so build_groups()/find_regions() in
    evaluate_accuracy.py can treat an Ollama correction exactly like one of
    Mozhi's real per-span matches. Adjacent non-equal diff blocks are merged
    into one span (rapidfuzz commonly emits a delete+insert pair for a single
    substitution of different word count). A block with no STT-side span (the
    LLM inserted words with no counterpart anywhere in the input) has nothing
    to anchor a Match's char offsets to and is dropped - a documented
    approximation, noted in the report; it only affects the reconstructed
    "corrected" preview for a region, never the whole-file WER, which compares
    full token lists directly."""
    cor_words = to_words(corrected_text)
    blocks = align([w.norm for w in stt_words], [w.norm for w in cor_words])
    merged: list[Block] = []
    for b in blocks:
        if b.tag == "equal":
            continue
        if merged and merged[-1].hyp_end == b.hyp_start and merged[-1].ref_end == b.ref_start:
            prev = merged.pop()
            merged.append(Block("replace", prev.hyp_start, b.hyp_end, prev.ref_start, b.ref_end))
        else:
            merged.append(Block("replace", b.hyp_start, b.hyp_end, b.ref_start, b.ref_end))
    matches = []
    for b in merged:
        if b.hyp_start >= b.hyp_end:
            continue
        original = " ".join(w.text for w in stt_words[b.hyp_start : b.hyp_end])
        replacement = " ".join(w.text for w in cor_words[b.ref_start : b.ref_end])
        matches.append(SimpleNamespace(
            start=stt_words[b.hyp_start].start,
            end=stt_words[b.hyp_end - 1].end,
            original=original,
            replacement=replacement,
            method=method,
        ))
    return matches


def fixed_output_corrector(output_text: str, stt_words: list, method: str):
    """A corrector callable (evaluate_pair's expected shape) that always
    returns a pre-computed LLM output, so the (slow, already-timed) Ollama
    call happens once per file, not once inside evaluate_pair too."""
    matches = synthetic_matches(stt_words, output_text, method)
    return lambda _stt_text: SimpleNamespace(corrected_text=output_text, matches=matches)


# --- per-file record ---------------------------------------------------------

@dataclass
class FileRun:
    name: str
    stt_text: str
    expected_text: str
    stt_word_count: int
    results: dict  # corrector name -> FileResult
    latencies_ms: dict  # corrector name -> float


def run_file(name: str, stt_text: str, expected_text: str) -> FileRun:
    stt_words = to_words(stt_text)

    t0 = time.perf_counter()
    mozhi_result = evaluate_pair(name, stt_text, expected_text, corrector=correct_text)
    mozhi_ms = (time.perf_counter() - t0) * 1000

    naive_prompt = NAIVE_PROMPT_TEMPLATE.format(text=stt_text)
    naive_output, naive_ms = call_ollama_inference(
        OLLAMA_MODEL, naive_prompt, options=OLLAMA_OPTIONS, timeout=180
    )
    naive_result = evaluate_pair(
        name, stt_text, expected_text,
        corrector=fixed_output_corrector(naive_output, stt_words, "ollama_naive"),
    )

    dict_prompt = DICT_PROMPT_TEMPLATE.format(text=stt_text)
    dict_output, dict_ms = call_ollama_inference(
        OLLAMA_MODEL, dict_prompt, options=OLLAMA_OPTIONS, timeout=180
    )
    dict_result = evaluate_pair(
        name, stt_text, expected_text,
        corrector=fixed_output_corrector(dict_output, stt_words, "ollama_dict"),
    )

    return FileRun(
        name=name, stt_text=stt_text, expected_text=expected_text,
        stt_word_count=len(stt_words),
        results={"mozhi": mozhi_result, "ollama_naive": naive_result, "ollama_dict": dict_result},
        latencies_ms={"mozhi": mozhi_ms, "ollama_naive": naive_ms, "ollama_dict": dict_ms},
    )


# --- aggregation --------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, same convention as comparison_benchmark.py."""
    s = sorted(values)
    idx = min(len(s) - 1, math.ceil(p * len(s)) - 1)
    return s[idx]


def latency_stats(latencies: list[float]) -> dict:
    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": percentile(latencies, 0.95),
        "n": len(latencies),
    }


def aggregate(results: list[FileResult]) -> dict:
    rows: list[Row] = [r for res in results for r in res.rows]
    ref_words = sum(res.raw.ref_words for res in results)
    raw_err = sum(res.raw.errors for res in results)
    cor_err = sum(res.corrected.errors for res in results)
    stats = catch_stats(rows)
    stt_errors = [r for r in rows if r.row_type == "stt_error"]
    regressions = [r for r in rows if r.row_type == "regression"]
    over_editing_words = sum(r.n_expected for r in regressions)
    phon_fixed, phon_total = stats[PHONETIC]
    all_fixed = sum(f for f, _ in stats.values())
    return {
        "ref_words": ref_words,
        "raw_wer": raw_err / ref_words,
        "corrected_wer": cor_err / ref_words,
        "stt_error_count": len(stt_errors),
        "all_fixed": all_fixed,
        "regression_count": len(regressions),
        "over_editing_words": over_editing_words,
        "phon_fixed": phon_fixed,
        "phon_total": phon_total,
        "phon_catch_rate": catch_rate(phon_fixed, phon_total),
        "stats": stats,
    }


# --- report rendering ----------------------------------------------------------

def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_txt(runs: list[FileRun], aggregates: dict, machine: str) -> str:
    lines = [
        "Mozhi vs Ollama: real-audio comparison",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Machine: {machine} - ALL latency figures below (Mozhi in-process, both Ollama",
        "  variants over local HTTP) were measured on this machine, in this run.",
        f"Ollama: {OLLAMA_HOST}, model={OLLAMA_MODEL}, options={OLLAMA_OPTIONS}",
        "One untimed warm-up call was made before any timed call.",
        f"Files: {len(runs)} (data/volunteer_audio, cached STT transcripts only - no fresh transcription)",
        "",
        "Frozen prompts (verbatim, not edited after the run started):",
        f"  naive:            {NAIVE_PROMPT_TEMPLATE!r}",
        f"  dictionary-aware: instruction + the full 319-term list from data/domain_terms.json,",
        "                    then the same 'return only the corrected text' closing instruction",
        "                    and {text} - see DICT_PROMPT_TEMPLATE in this script for the exact string.",
        "",
        "=" * 78,
        "Per-file",
        "=" * 78,
    ]
    for run in runs:
        lines.append(f"\n=== {run.name} ===")
        lines.append(f"STT words: {run.stt_word_count}")
        lines.append("--- raw STT ---")
        lines.append(run.stt_text)
        lines.append("--- expected (human-written; includes a trailing acronym-tag line for")
        lines.append("    vol* files per data/volunteer_audio's convention - see Caveats) ---")
        lines.append(run.expected_text)
        for corrector in CORRECTOR_NAMES:
            res = run.results[corrector]
            ms = run.latencies_ms[corrector]
            lines.append(f"--- {CORRECTOR_LABELS[corrector]} corrected ({ms:.0f} ms) ---")
            lines.append(res.corrected_text)
        lines.append("scoring (this file only):")
        lines.append(f"  {'corrector':<47} {'corrected WER':>14} {'stt-err regions':>16} {'regressions':>12} {'over-edit words':>16} {'phon catch':>12}")
        for corrector in CORRECTOR_NAMES:
            res = run.results[corrector]
            rows = res.rows
            stt_errors = [r for r in rows if r.row_type == "stt_error"]
            regressions = [r for r in rows if r.row_type == "regression"]
            over_editing = sum(r.n_expected for r in regressions)
            stats = catch_stats(rows)
            pf, pt = stats[PHONETIC]
            lines.append(
                f"  {CORRECTOR_LABELS[corrector]:<47} {_pct(res.corrected.wer):>14} {len(stt_errors):>16} "
                f"{len(regressions):>12} {over_editing:>16} {f'{pf}/{pt}':>12}"
            )

    lines += ["", "=" * 78, f"AGGREGATE ({len(runs)} files, {aggregates['mozhi']['ref_words']} reference words)", "=" * 78]

    avg_stt_words = statistics.mean(r.stt_word_count for r in runs)
    lines += [
        "",
        "Latency - wall-clock, one call per file, same 32 full transcripts for all three",
        f"(avg transcript: {avg_stt_words:.1f} STT words; not the 34-sentence batch used elsewhere)",
        f"  {'corrector':<47} {'mean':>10} {'median':>10} {'p95':>10}",
    ]
    for corrector in CORRECTOR_NAMES:
        lat = [r.latencies_ms[corrector] for r in runs]
        st = latency_stats(lat)
        lines.append(f"  {CORRECTOR_LABELS[corrector]:<47} {st['mean']:>7.1f} ms {st['median']:>7.1f} ms {st['p95']:>7.1f} ms")

    lines += [
        "",
        "Word Error Rate (word-level, case/punctuation-insensitive, same alignment as evaluate_accuracy.py)",
        f"  raw STT WER: {_pct(aggregates['mozhi']['raw_wer'])}  (identical for all three - same STT input, same reference)",
    ]
    for corrector in CORRECTOR_NAMES:
        a = aggregates[corrector]
        lines.append(f"  {CORRECTOR_LABELS[corrector]:<47} corrected WER: {_pct(a['corrected_wer'])}")

    lines += [
        "",
        "Domain-aware catch rate (phonetic_substitution category: Metaphone-close AND the",
        "expected word(s) are/contain one of the 319 dictionary terms - same categorization,",
        "same threshold, same is_domain_relevant() as evaluate_accuracy.py, applied identically",
        "to every corrector's own STT-error regions)",
    ]
    for corrector in CORRECTOR_NAMES:
        a = aggregates[corrector]
        lines.append(
            f"  {CORRECTOR_LABELS[corrector]:<47} {a['phon_fixed']}/{a['phon_total']} = {_pct(a['phon_catch_rate'])}"
            f"   (all-category catch rate: {a['all_fixed']}/{a['stt_error_count']} = {_pct(catch_rate(a['all_fixed'], a['stt_error_count']))})"
        )

    lines += [
        "",
        "Regressions (STT already matched expected; corrector changed it anyway) and",
        "over-editing (total words touched by those regressions)",
        f"  {'corrector':<47} {'regressions':>12} {'over-edit words':>16}",
    ]
    for corrector in CORRECTOR_NAMES:
        a = aggregates[corrector]
        lines.append(f"  {CORRECTOR_LABELS[corrector]:<47} {a['regression_count']:>12} {a['over_editing_words']:>16}")

    lines += [
        "",
        "Caveats",
        "  - Expected text includes a trailing acronym-tag line for every 'vol*' file",
        "    (e.g. 'CRUD DNS SLA HSTS RAID VPN ORM'), a pre-existing convention in",
        "    data/volunteer_audio/*.expected.txt reused as-is from evaluate_accuracy.py's",
        "    own handling (expected.read_text().strip(), unmodified). Nobody spoke that",
        "    line, so it is an unrecoverable OMISSION region for every corrector alike -",
        "    it does not favor or penalize any one of the three.",
        "  - Ollama's per-region scoring (regions, regressions, catch rate) uses match",
        "    spans SYNTHESIZED from a word-level diff between its input and output, not",
        "    real per-span corrections like Mozhi's pipeline produces. Whole-file WER is",
        "    exact either way (direct token-list comparison); only the region/regression",
        "    breakdown is an approximation, and an LLM insertion with no counterpart",
        "    anywhere in the STT input is dropped from that breakdown (see",
        "    synthetic_matches() in this script).",
        "  - temperature=0 and a fixed seed make Ollama's decoding far more repeatable",
        "    than the default sampling used in comparison_benchmark.py, but llama.cpp-style",
        "    backends do not guarantee bit-exact reproducibility across runs, hosts, or",
        "    context-batching conditions - treat this run as highly-repeatable, not proven",
        "    identical on a re-run.",
        "  - This is 32 files, not a large sample. It answers whether Mozhi's non-generative",
        "    approach beats a same-hardware local LLM on real recordings, not a general claim",
        "    about llama3.1:8b or generative correction as a category.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, runs: list[FileRun]) -> None:
    fields = [
        "file", "stt_word_count",
        "stt_text", "expected_text",
        "mozhi_corrected", "ollama_naive_corrected", "ollama_dict_corrected",
        "mozhi_latency_ms", "ollama_naive_latency_ms", "ollama_dict_latency_ms",
        "mozhi_corrected_wer", "ollama_naive_corrected_wer", "ollama_dict_corrected_wer",
        "mozhi_stt_error_regions", "ollama_naive_stt_error_regions", "ollama_dict_stt_error_regions",
        "mozhi_regressions", "ollama_naive_regressions", "ollama_dict_regressions",
        "mozhi_over_editing_words", "ollama_naive_over_editing_words", "ollama_dict_over_editing_words",
        "mozhi_phon_fixed", "mozhi_phon_total",
        "ollama_naive_phon_fixed", "ollama_naive_phon_total",
        "ollama_dict_phon_fixed", "ollama_dict_phon_total",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "file": run.name, "stt_word_count": run.stt_word_count,
                "stt_text": run.stt_text, "expected_text": run.expected_text,
            }
            for corrector in CORRECTOR_NAMES:
                res = run.results[corrector]
                rows = res.rows
                stt_errors = [r for r in rows if r.row_type == "stt_error"]
                regressions = [r for r in rows if r.row_type == "regression"]
                over_editing = sum(r.n_expected for r in regressions)
                stats = catch_stats(rows)
                pf, pt = stats[PHONETIC]
                row[f"{corrector}_corrected"] = res.corrected_text
                row[f"{corrector}_latency_ms"] = f"{run.latencies_ms[corrector]:.1f}"
                row[f"{corrector}_corrected_wer"] = f"{res.corrected.wer:.4f}"
                row[f"{corrector}_stt_error_regions"] = len(stt_errors)
                row[f"{corrector}_regressions"] = len(regressions)
                row[f"{corrector}_over_editing_words"] = over_editing
                row[f"{corrector}_phon_fixed"] = pf
                row[f"{corrector}_phon_total"] = pt
            writer.writerow(row)


def machine_label() -> str:
    import platform
    try:
        import subprocess
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
        ).strip()
    except Exception:
        chip = platform.machine()
    return f"{platform.system()} {platform.machine()} ({chip})"


def main() -> int:
    try:
        available = ollama_models_available()
    except requests.RequestException as error:
        print(f"ERROR: Ollama not reachable at {OLLAMA_HOST}: {error}")
        return 1
    if OLLAMA_MODEL not in available:
        print(f"ERROR: model {OLLAMA_MODEL!r} not found in Ollama. Available: {available}")
        return 1

    if not AUDIO_DIR.is_dir():
        print(f"ERROR: {AUDIO_DIR} not found")
        return 1
    pairs, warnings = discover_pairs(AUDIO_DIR)
    for w in warnings:
        print(f"WARNING: {w}")

    file_texts: list[tuple[str, str, str]] = []
    for audio, expected in pairs:
        json_path = STT_DIR / f"{audio.stem}.json"
        if not json_path.exists():
            print(f"WARNING: no cached STT transcript for {audio.name}, skipped (this script does not transcribe)")
            continue
        stt_text = json.loads(json_path.read_text())["transcript"]
        expected_text = expected.read_text(encoding="utf-8").strip()
        file_texts.append((audio.stem, stt_text, expected_text))

    if not file_texts:
        print("No files with cached STT transcripts found.")
        return 1

    print(f"Backend: Ollama ({OLLAMA_HOST}, model={OLLAMA_MODEL}, options={OLLAMA_OPTIONS})")
    print(f"Files: {len(file_texts)}")
    print("Warming up (untimed)...")
    call_ollama_inference(OLLAMA_MODEL, WARMUP_PROMPT, options=OLLAMA_OPTIONS, timeout=180)

    runs: list[FileRun] = []
    for i, (name, stt_text, expected_text) in enumerate(file_texts, start=1):
        print(f"  [{i}/{len(file_texts)}] {name} ...", flush=True)
        try:
            runs.append(run_file(name, stt_text, expected_text))
        except requests.RequestException as error:
            print(f"    ERROR: Ollama request failed for {name}: {error}; skipped")
        except Exception as error:
            print(f"    ERROR: {name}: {error}; skipped")

    if not runs:
        print("No files evaluated successfully.")
        return 1

    aggregates = {
        corrector: aggregate([run.results[corrector] for run in runs])
        for corrector in CORRECTOR_NAMES
    }
    machine = machine_label()
    report = render_txt(runs, aggregates, machine)
    REPORT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_TXT_PATH.write_text(report, encoding="utf-8")
    write_csv(REPORT_CSV_PATH, runs)

    print()
    print(report[report.index("AGGREGATE") - 4:])
    print(f"Wrote {REPORT_TXT_PATH} and {REPORT_CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
