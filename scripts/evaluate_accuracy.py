"""
Repeatable accuracy evaluation: raw STT vs Mozhi-corrected, against human-written
transcripts.

Convention (see data/volunteer_audio/README.md): every recording is a pair,

    data/volunteer_audio/<name>.m4a
    data/volunteer_audio/<name>.expected.txt      what the speaker actually said

For each complete pair this script
  1. gets the raw STT transcript: cached data/stt_transcripts/<name>.json when it
     is newer than the audio, otherwise a fresh faster-whisper run using
     scripts/transcribe_local.py's transcribe_one (--retranscribe forces one);
  2. runs app.pipeline.correct_text() over it;
  3. word-aligns STT vs expected and corrected vs expected with rapidfuzz's
     Levenshtein opcodes (unit-cost edit distance; rapidfuzz is already a
     dependency), giving word-level WER = (S + D + I) / reference words;
  4. groups the STT-vs-expected errors into REGIONS, records whether Mozhi fixed
     each one, and attaches a SUGGESTED category - phonetic_substitution only when
     the expected word(s) are also domain-relevant (see is_domain_relevant()),
     otherwise out_of_scope, so generic STT noise unrelated to the 319-term
     dictionary does not dilute the addressable catch-rate.

Outputs: a CSV with one row per region (reports/accuracy_evaluation.csv) and a
text summary (reports/accuracy_evaluation_summary.txt). The `category_suggested`
column is a heuristic, not ground truth: review it and put corrections in
`category_reviewed`, which takes precedence in the summary and is preserved when
this script is re-run.

Headline number: the catch-rate WITHIN the phonetic_substitution category, i.e.
of the errors Mozhi's approach can address, the share it actually fixed.

Scoring is word-level, case- and punctuation-insensitive, using app.tokenizer
(so "X.509" and "two-factor" each count as more than one word, identically on
both sides). This script lives outside app/ and is not part of the correction
path.

Run:
    PYTHONPATH=. python3 scripts/evaluate_accuracy.py [--retranscribe] [--model base.en]
"""
import argparse
import csv
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from app.dictionary import get_term_strings
from app.pipeline import DEFAULT_THRESHOLD, FUZZY_STOPWORDS, correct_text
from app.scorer import metaphone_score
from app.tokenizer import normalize, tokenize

REPO_ROOT = Path(__file__).parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "volunteer_audio"
STT_DIR = REPO_ROOT / "data" / "stt_transcripts"
CSV_PATH = REPO_ROOT / "reports" / "accuracy_evaluation.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "accuracy_evaluation_summary.txt"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}  # mirrors transcribe_local.py
EXPECTED_SUFFIX = ".expected.txt"

# A region counts as a phonetic substitution when the Metaphone similarity
# (app.scorer.metaphone_score: Jaro-Winkler over sorted per-word Metaphone codes)
# between the STT words and the expected words reaches this. An unvalidated
# starting value, kept at module level so it can be swept; the score is written to
# the CSV so every suggestion can be checked by eye.
PHONETIC_CLOSE_THRESHOLD = 0.80

PHONETIC = "phonetic_substitution"
NON_PHONETIC = "non_phonetic_substitution"
OMISSION = "omission"
INSERTION = "insertion"  # extra STT words with no expected counterpart; not in the original 3-way taxonomy
# Phonetically close (>= PHONETIC_CLOSE_THRESHOLD) but the expected word(s) have
# nothing to do with the 319-term dictionary ("that" -> "the", "scheduled" ->
# "schedule"): generic STT noise Mozhi was never scoped to touch, not an
# addressable miss. Found necessary on a 32-file real-recording run: without this
# split, 180 of 190 PHONETIC-suggested regions turned out to be exactly this,
# diluting the catch-rate with errors outside Mozhi's domain entirely.
OUT_OF_SCOPE = "out_of_scope"
CATEGORIES = (PHONETIC, NON_PHONETIC, OMISSION, INSERTION, OUT_OF_SCOPE)

CSV_FIELDS = [
    "file", "region", "row_type", "expected", "stt", "corrected", "n_expected", "n_stt",
    "fixed", "outcome", "mozhi_edits", "category_suggested", "phonetic_score", "category_reviewed",
]


# --- words and alignment -------------------------------------------------------------------

@dataclass(frozen=True)
class Word:
    text: str   # surface form as written
    norm: str   # lowercased, used for all comparison
    start: int  # char offsets in the source string
    end: int


def to_words(text: str) -> list[Word]:
    """Tokenize with the pipeline's own tokenizer so evaluation counts words the
    way Mozhi sees them. Tokens that normalize to nothing (a lone apostrophe) are dropped."""
    words = [Word(t.text, normalize(t.text), t.start, t.end) for t in tokenize(text)]
    return [w for w in words if w.norm]


@dataclass(frozen=True)
class Block:
    """One aligned span. hyp = the hypothesis (STT or corrected), ref = expected.
    Tags follow rapidfuzz's opcodes, whose names read from the hypothesis's point of view:
      equal    same words
      replace  equal-length span of substituted words
      delete   hypothesis words with no reference counterpart (extra words: a WER insertion)
      insert   reference words missing from the hypothesis (dropped words: a WER deletion)"""
    tag: str
    hyp_start: int
    hyp_end: int
    ref_start: int
    ref_end: int


def align(hyp: list[str], ref: list[str]) -> list[Block]:
    """Word-level minimum edit-distance alignment. Where several alignments are
    equally cheap, rapidfuzz picks one deterministically but arbitrarily."""
    return [
        Block(o.tag, o.src_start, o.src_end, o.dest_start, o.dest_end)
        for o in Levenshtein.opcodes(hyp, ref)
    ]


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int   # reference words missing from the hypothesis
    insertions: int  # extra hypothesis words
    ref_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.ref_words


def error_counts(hyp: list[str], ref: list[str]) -> ErrorCounts:
    """Word-level S/D/I counts and WER = (S + D + I) / len(ref)."""
    if not ref:
        raise ValueError("expected transcript has no words; WER is undefined")
    blocks = align(hyp, ref)
    return ErrorCounts(
        substitutions=sum(b.ref_end - b.ref_start for b in blocks if b.tag == "replace"),
        deletions=sum(b.ref_end - b.ref_start for b in blocks if b.tag == "insert"),
        insertions=sum(b.hyp_end - b.hyp_start for b in blocks if b.tag == "delete"),
        ref_words=len(ref),
    )


# --- regions ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Group:
    """STT tokens [start, end) that Mozhi treated as one unit, and what they became.
    A span Mozhi matched is one group ("one link" -> "WAN link"); any other token
    is a group of one that maps to itself."""
    start: int
    end: int
    out: tuple[Word, ...]
    edit: str  # "method: 'original' -> 'replacement'", or "" if Mozhi left it alone


def build_groups(stt: list[Word], matches: list) -> list[Group]:
    """Partition the STT tokens into groups using the pipeline's Match objects
    (anything with start, end, original, replacement, method; char offsets in the STT text)."""
    ordered = sorted(matches, key=lambda m: m.start)
    groups: list[Group] = []
    i = mi = 0
    while i < len(stt):
        while mi < len(ordered) and ordered[mi].end <= stt[i].start:
            mi += 1
        m = ordered[mi] if mi < len(ordered) else None
        if m is not None and m.start <= stt[i].start and stt[i].end <= m.end:
            j = i
            while j < len(stt) and stt[j].start >= m.start and stt[j].end <= m.end:
                j += 1
            groups.append(Group(i, j, tuple(to_words(m.replacement)),
                                f"{m.method}: {m.original!r} -> {m.replacement!r}"))
            i, mi = j, mi + 1
        else:
            groups.append(Group(i, i + 1, (stt[i],), ""))
            i += 1
    return groups


@dataclass(frozen=True)
class Region:
    stt_start: int  # STT tokens [stt_start, stt_end); may be empty (words the STT dropped)
    stt_end: int
    exp_start: int  # expected words [exp_start, exp_end); may be empty (extra STT words)
    exp_end: int


def find_regions(stt: list[Word], exp: list[Word], groups: list[Group]) -> tuple[list[Region], list[bool]]:
    """Maximal misaligned regions, plus which groups are dirty.

    A token is a CLEAN ANCHOR if the STT-vs-expected alignment calls it `equal` and
    its group neither contains an error token nor changed anything. Each stretch
    between consecutive anchors that holds any STT token or any expected word is
    one region. Because a group is always wholly inside one stretch, a Mozhi match
    that straddles an error and a correct neighbour ("one link" -> "WAN link",
    where only "one" was wrong) expands the region to the whole match, and
    "did Mozhi fix it?" is well defined."""
    exp_of: dict[int, int] = {}
    for b in align([w.norm for w in stt], [w.norm for w in exp]):
        if b.tag == "equal":
            for k in range(b.hyp_end - b.hyp_start):
                exp_of[b.hyp_start + k] = b.ref_start + k

    dirty: list[bool] = []
    for g in groups:
        tokens = range(g.start, g.end)
        has_error_token = any(t not in exp_of for t in tokens)
        changed = [w.norm for w in g.out] != [stt[t].norm for t in tokens]
        dirty.append(has_error_token or changed)

    anchors = [
        (t, exp_of[t])
        for g, is_dirty in zip(groups, dirty) if not is_dirty
        for t in range(g.start, g.end)
    ]
    anchors.sort()
    anchors.append((len(stt), len(exp)))

    regions: list[Region] = []
    prev_s = prev_e = -1
    for s, e in anchors:
        if s > prev_s + 1 or e > prev_e + 1:
            regions.append(Region(prev_s + 1, s, prev_e + 1, e))
        prev_s, prev_e = s, e
    return regions, dirty


def _domain_words() -> frozenset[str]:
    """Every normalized word appearing in ANY dictionary term, single- or
    multi-word ("two-factor authentication" contributes "two", "factor" and
    "authentication" individually, via app.tokenizer - the same word split
    used everywhere else in this script), EXCLUDING app.pipeline.FUZZY_STOPWORDS.
    Some phrase terms contain an ordinary filler word ("living off THE land",
    "pass-THE-hash", "pass-THE-ticket"); without excluding it, "the" or "off"
    alone in a region's expected text would count as domain-relevant, which
    defeats the point of this check - reuses the pipeline's own guard for
    exactly this "a short common word isn't a meaningful signal" problem
    rather than inventing a second list. Built once; get_term_strings() itself
    is already lru_cache'd in app.dictionary, this just avoids re-tokenizing
    all 319 terms on every call."""
    words: set[str] = set()
    for term in get_term_strings():
        words.update(w.norm for w in to_words(term))
    return frozenset(words) - FUZZY_STOPWORDS


_DOMAIN_WORDS: frozenset[str] | None = None


def is_domain_relevant(exp_words: list[str]) -> bool:
    """True if the expected word(s) for a region ARE, or contain, an actual
    dictionary term - i.e. this is something Mozhi's mechanism could
    plausibly have addressed, not generic STT noise unrelated to the
    dictionary. ANY overlapping word counts ("part of" a term), not every
    word: a region can mix a real domain word with ordinary surrounding
    words the STT also mis-heard, and the domain word alone is enough to
    make the region addressable in principle. Empty input is not relevant."""
    global _DOMAIN_WORDS
    if _DOMAIN_WORDS is None:
        _DOMAIN_WORDS = _domain_words()
    return any(w in _DOMAIN_WORDS for w in exp_words)


def suggest_category(
    stt_words: list[str], exp_words: list[str], threshold: float = PHONETIC_CLOSE_THRESHOLD
) -> tuple[str, float | None]:
    """SUGGESTED category for a region, and the Metaphone score behind it. A
    heuristic for a human to review, not ground truth.
      omission                   STT has no word aligned to the expected word(s)
      insertion                  STT has extra word(s) with no expected counterpart
      phonetic_substitution      Metaphone similarity >= threshold AND the expected
                                  word(s) are/contain an actual dictionary term
      out_of_scope               Metaphone similarity >= threshold but NOT domain-relevant
      non_phonetic_substitution  Metaphone similarity < threshold"""
    if not stt_words:
        return OMISSION, None
    if not exp_words:
        return INSERTION, None
    score = metaphone_score(" ".join(stt_words), " ".join(exp_words))
    if score < threshold:
        return NON_PHONETIC, score
    return (PHONETIC if is_domain_relevant(exp_words) else OUT_OF_SCOPE), score


@dataclass
class Row:
    file: str
    region: int
    row_type: str  # stt_error: the STT was wrong here | regression: the STT was right and Mozhi broke it
    expected: str
    stt: str
    corrected: str
    n_expected: int
    n_stt: int
    fixed: bool
    outcome: str  # fixed | missed (Mozhi left it) | changed_wrong (Mozhi changed it, still wrong) | regression
    mozhi_edits: str
    category_suggested: str  # blank for regressions
    phonetic_score: float | None
    category_reviewed: str = ""

    @property
    def category(self) -> str:
        """Reviewed category if a human set one, else the suggestion."""
        return self.category_reviewed or self.category_suggested


def _surface(words) -> str:
    return " ".join(w.text for w in words)


@dataclass
class FileResult:
    name: str
    stt_source: str  # cached | fresh | provided
    stt_text: str
    expected_text: str
    corrected_text: str
    raw: ErrorCounts
    corrected: ErrorCounts
    rows: list[Row] = field(default_factory=list)


def evaluate_pair(
    name: str,
    stt_text: str,
    expected_text: str,
    corrector: Callable = correct_text,
    stt_source: str = "provided",
) -> FileResult:
    """Score one recording. `corrector(text)` must return an object with
    .corrected_text and .matches (defaults to the real pipeline; tests inject fakes)."""
    exp = to_words(expected_text)
    if not exp:
        raise ValueError(f"{name}: expected transcript has no words")
    stt = to_words(stt_text)
    result = corrector(stt_text)
    cor = to_words(result.corrected_text)

    raw = error_counts([w.norm for w in stt], [w.norm for w in exp])
    corrected = error_counts([w.norm for w in cor], [w.norm for w in exp])

    groups = build_groups(stt, result.matches)
    regions, dirty = find_regions(stt, exp, groups)
    rows: list[Row] = []
    for n, r in enumerate(regions, start=1):
        stt_w = stt[r.stt_start : r.stt_end]
        exp_w = exp[r.exp_start : r.exp_end]
        in_region = [g for g, d in zip(groups, dirty) if d and g.start >= r.stt_start and g.end <= r.stt_end]
        cor_w = [w for g in in_region for w in g.out]
        stt_n, exp_n, cor_n = ([w.norm for w in ws] for ws in (stt_w, exp_w, cor_w))
        fixed = cor_n == exp_n
        if stt_n == exp_n:
            row_type, outcome, category, score = "regression", "regression", "", None
        else:
            row_type = "stt_error"
            outcome = "fixed" if fixed else ("missed" if cor_n == stt_n else "changed_wrong")
            category, score = suggest_category(stt_n, exp_n)
        rows.append(Row(
            file=name, region=n, row_type=row_type,
            expected=_surface(exp_w), stt=_surface(stt_w), corrected=_surface(cor_w),
            n_expected=len(exp_w), n_stt=len(stt_w), fixed=fixed, outcome=outcome,
            mozhi_edits="; ".join(g.edit for g in in_region if g.edit),
            category_suggested=category, phonetic_score=score,
        ))
    return FileResult(name, stt_source, stt_text, expected_text, result.corrected_text, raw, corrected, rows)


# --- aggregation ------------------------------------------------------------------------------

def catch_stats(rows: list[Row]) -> dict[str, tuple[int, int]]:
    """{category: (fixed, total)} over stt_error rows, using reviewed-else-suggested."""
    stats = {c: [0, 0] for c in CATEGORIES}
    for row in rows:
        if row.row_type != "stt_error":
            continue
        cell = stats.setdefault(row.category, [0, 0])
        cell[1] += 1
        cell[0] += row.fixed
    return {c: (f, t) for c, (f, t) in stats.items()}


def catch_rate(fixed: int, total: int) -> float | None:
    return fixed / total if total else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _counts(c: ErrorCounts) -> str:
    return f"{c.errors} error{'' if c.errors == 1 else 's'} (S{c.substitutions}/D{c.deletions}/I{c.insertions})"


def summarize(results: list[FileResult], threshold: float = DEFAULT_THRESHOLD) -> str:
    rows = [r for res in results for r in res.rows]
    ref_words = sum(res.raw.ref_words for res in results)
    raw_err = sum(res.raw.errors for res in results)
    cor_err = sum(res.corrected.errors for res in results)
    raw_wer, cor_wer = raw_err / ref_words, cor_err / ref_words
    stats = catch_stats(rows)
    stt_errors = [r for r in rows if r.row_type == "stt_error"]
    regressions = [r for r in rows if r.row_type == "regression"]
    reviewed = sum(1 for r in stt_errors if r.category_reviewed)
    phon_fixed, phon_total = stats[PHONETIC]

    lines = [
        "Mozhi accuracy evaluation",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Files evaluated: {len(results)}   Reference words: {ref_words}",
        f"Corrector: app.pipeline.correct_text (threshold {threshold}, {len(get_term_strings())}-term dictionary)",
        "Scoring: WORD-level, case and punctuation ignored, app.tokenizer word boundaries",
        "Alignment: rapidfuzz Levenshtein opcodes (unit cost); equally-cheap alignments are broken deterministically by the library",
        "",
        "Per file",
    ]
    for res in results:
        lines.append(
            f"  {res.name:<24} ref={res.raw.ref_words:>3}  raw WER {_pct(res.raw.wer):>6} {_counts(res.raw):<28}"
            f"  corrected WER {_pct(res.corrected.wer):>6} {_counts(res.corrected)}   [STT: {res.stt_source}]"
        )
    lines += [
        "",
        "Aggregate (micro-averaged over all reference words)",
        f"  raw WER:        {_pct(raw_wer)}  ({raw_err} errors / {ref_words} words)",
        f"  corrected WER:  {_pct(cor_wer)}  ({cor_err} errors / {ref_words} words)",
        "  WER reduction:  " + f"{(raw_wer - cor_wer) * 100:.1f} percentage points (positive = Mozhi helped)"
        + ("" if raw_wer == 0 else f", relative {(raw_wer - cor_wer) / raw_wer * 100:.1f}%"),
        "",
        f"Error regions in the raw STT: {len(stt_errors)}   (a region is a maximal run of misaligned words, usually one)",
        "  category                     regions   fixed   catch-rate",
    ]
    for c in CATEGORIES:
        f, t = stats[c]
        lines.append(f"  {c:<28} {t:>7} {f:>7}   {_pct(catch_rate(f, t)):>8}")
    all_fixed = sum(f for f, _ in stats.values())
    lines += [
        f"  {'all categories':<28} {len(stt_errors):>7} {all_fixed:>7}   {_pct(catch_rate(all_fixed, len(stt_errors))):>8}",
        "",
        f"PHONETIC-SUBSTITUTION CATCH RATE: {phon_fixed}/{phon_total} = {_pct(catch_rate(phon_fixed, phon_total))}"
        "   <- the addressable accuracy number",
        f"  phonetic share of STT error regions: {phon_total}/{len(stt_errors)} = {_pct(catch_rate(phon_total, len(stt_errors)))}",
        f"  changed but still wrong: {sum(1 for r in stt_errors if r.outcome == 'changed_wrong')}",
        f"  regressions (STT right, Mozhi broke it): {len(regressions)}",
        "",
        "Caveats",
        f"  - Categories are SUGGESTIONS (Metaphone similarity >= {PHONETIC_CLOSE_THRESHOLD}, an unvalidated starting value),"
        f" not ground truth. {reviewed} of {len(stt_errors)} STT-error regions carry a hand-set category_reviewed.",
        "  - 'insertion' (extra STT words) and 'out_of_scope' (phonetically close but the expected word(s) are not"
        " in the 319-term dictionary) are two labels beyond the original phonetic/omission/non-phonetic taxonomy.",
        "  - WER is per word; catch-rate is per region. A region can span several words.",
        f"  - Sample: {len(results)} file(s), {len(stt_errors)} error region(s). Small samples say whether the pipeline works,"
        " not how accurate Mozhi is.",
    ]
    return "\n".join(lines) + "\n"


# --- CSV, with hand-set categories preserved across re-runs ---------------------------------------

def _key(file: str, expected: str, stt: str) -> tuple[str, str, str]:
    return (file, expected, stt)


def load_reviews(path: Path) -> dict[tuple[str, str, str], str]:
    """category_reviewed values from a previous CSV, keyed by (file, expected, stt),
    so regenerating the CSV never discards manual review."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {
            _key(r["file"], r["expected"], r["stt"]): r["category_reviewed"].strip()
            for r in csv.DictReader(f)
            if r.get("category_reviewed", "").strip()
        }


def apply_reviews(rows: list[Row], reviews: dict[tuple[str, str, str], str]) -> int:
    """Copy preserved reviews onto matching rows; returns how many were applied."""
    n = 0
    for row in rows:
        review = reviews.get(_key(row.file, row.expected, row.stt))
        if review:
            row.category_reviewed = review
            n += 1
    return n


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "file": r.file, "region": r.region, "row_type": r.row_type,
                "expected": r.expected, "stt": r.stt, "corrected": r.corrected,
                "n_expected": r.n_expected, "n_stt": r.n_stt,
                "fixed": "yes" if r.fixed else "no", "outcome": r.outcome,
                "mozhi_edits": r.mozhi_edits, "category_suggested": r.category_suggested,
                "phonetic_score": "" if r.phonetic_score is None else f"{r.phonetic_score:.3f}",
                "category_reviewed": r.category_reviewed,
            })


# --- discovery and transcription ------------------------------------------------------------------

def expected_path_for(audio: Path) -> Path:
    return audio.with_name(audio.stem + EXPECTED_SUFFIX)


def discover_pairs(audio_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Complete (audio, expected) pairs, sorted by name, plus a warning for every
    audio file without expected text and every expected text without audio."""
    audio = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    pairs = [(a, expected_path_for(a)) for a in audio if expected_path_for(a).exists()]
    warnings = [f"no {expected_path_for(a).name} for {a.name}: skipped" for a in audio if not expected_path_for(a).exists()]
    stems = {a.stem for a in audio}
    for e in sorted(audio_dir.glob(f"*{EXPECTED_SUFFIX}")):
        if e.name[: -len(EXPECTED_SUFFIX)] not in stems:
            warnings.append(f"{e.name} has no matching audio file: skipped")
    return pairs, warnings


class Transcriber:
    """Runs scripts/transcribe_local.py's transcribe_one, loading the Whisper model once, lazily."""

    def __init__(self, model_name: str = "base.en"):
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        """Load the Whisper model now (idempotent), so a long-running caller such as
        the Streamlit demo can pay the load cost at startup, not on the first request."""
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"Loading Whisper model '{self.model_name}' (CPU, int8)...")
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def __call__(self, audio: Path) -> dict:
        from scripts.transcribe_local import transcribe_one
        self.load()
        return transcribe_one(self._model, audio)


def get_transcript(
    audio: Path, stt_dir: Path, transcriber: Callable[[Path], dict], retranscribe: bool = False
) -> tuple[str, str]:
    """(transcript, 'cached' | 'fresh'). Cached JSON is reused only if newer than the audio."""
    json_path = stt_dir / f"{audio.stem}.json"
    stale = (not json_path.exists()) or json_path.stat().st_mtime < audio.stat().st_mtime
    if retranscribe or stale:
        result = transcriber(audio)
        stt_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2) + "\n")  # same format as transcribe_local.py
        return result["transcript"], "fresh"
    return json.loads(json_path.read_text())["transcript"], "cached"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--stt-dir", type=Path, default=STT_DIR)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--model", default="base.en", help="Whisper model size (default: base.en)")
    parser.add_argument("--retranscribe", action="store_true", help="ignore cached transcripts and run Whisper again")
    args = parser.parse_args(argv)

    if not args.audio_dir.is_dir():
        print(f"ERROR: {args.audio_dir} not found")
        return 1
    pairs, warnings = discover_pairs(args.audio_dir)
    for w in warnings:
        print(f"WARNING: {w}")
    if not pairs:
        print(f"No complete <name>.<audio> + <name>{EXPECTED_SUFFIX} pairs in {args.audio_dir}")
        return 1

    transcriber = Transcriber(args.model)
    results: list[FileResult] = []
    for audio, expected in pairs:
        try:
            stt_text, source = get_transcript(audio, args.stt_dir, transcriber, args.retranscribe)
            results.append(evaluate_pair(audio.stem, stt_text, expected.read_text(encoding="utf-8").strip(), stt_source=source))
        except ImportError:
            print("ERROR: faster-whisper is not installed. Run: pip install -r requirements-benchmark.txt")
            return 1
        except Exception as error:
            print(f"WARNING: {audio.name}: {error}; skipped")
    if not results:
        print("No files evaluated successfully.")
        return 1

    rows = [r for res in results for r in res.rows]
    kept = apply_reviews(rows, load_reviews(args.csv))
    write_csv(args.csv, rows)
    summary = summarize(results)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote {args.csv} ({len(rows)} rows, {kept} hand-set categories preserved) and {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
