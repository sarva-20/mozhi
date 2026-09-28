"""Benchmark a generative correction baseline against Mozhi's non-generative
pipeline.

Active backend: local Ollama. Setup::

    ollama pull llama3.1:8b   # if not already present
    PYTHONPATH=. python3 scripts/comparison_benchmark.py

Set ``OLLAMA_MODEL`` / ``OLLAMA_HOST`` to override the model or server
address. The script checks ``/api/tags`` first and stops with a clear error
if Ollama isn't reachable or the model isn't pulled, rather than proceeding
blind. Output is also saved to reports/comparison_benchmark_ollama.txt.

------------------------------------------------------------------------------
ABANDONED: Hugging Face free-tier backend (kept below for the record; not
called by main() anymore). The free serverless Inference API could not
reliably serve any of the fallback models on demand - 404/503, gated,
unavailable to the provider, or cold-starting (see
reports/comparison_benchmark_blocker.md for the setup notes from when this
was the active path). Superseded by the Ollama backend, which runs a real
local model with no third-party availability gating. To retry the HF path
anyway: pip install -r requirements-benchmark.txt, export HF_TOKEN=hf_...,
then call call_hf_inference() directly - it is not wired into main().
------------------------------------------------------------------------------
"""
import io
import math
import os
import statistics
import sys
import time
from pathlib import Path

import requests
from contextlib import redirect_stdout

from app.pipeline import correct_text
from tests.batch_cases import BATCH_CASES

REPORT_PATH = Path(__file__).parent.parent / "reports" / "comparison_benchmark_ollama.txt"

# Mozhi's own measured numbers, quoted rather than recomputed here (per
# CLAUDE.md: every latency/accuracy claim must come from a reports/
# artifact). Latency: reports/latency_benchmark_letter_composition.txt
# (34-sentence batch, 20 runs/sentence, in-process time.perf_counter).
# Domain-aware catch rate: reports/accuracy_evaluation_summary.txt
# (phonetic-substitution catch rate over 32 real volunteer recordings).
MOZHI_LATENCY_MEAN_MS = 24.656
MOZHI_LATENCY_P99_MS = 38.248
MOZHI_REAL_AUDIO_CATCH_RATE = "5/35 = 14.3%"

# === ABANDONED: Hugging Face free-tier backend ==============================
# Dead end - see module docstring above. Left in place as documented history;
# not imported or called anywhere in main().
HF_API_URL = "https://router.huggingface.co/hf-inference/models/{model}"
MODEL_FALLBACKS = [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "HuggingFaceH4/zephyr-7b-beta",
]
DEFAULT_MODEL = MODEL_FALLBACKS[0]

GEC_PROMPT_TEMPLATE = """Correct any speech-to-text mistranscriptions of technical terms in this \
tech support sentence. Return ONLY the corrected sentence.

Sentence: {text}"""


def call_hf_inference(model: str, prompt: str, token: str, timeout: int = 30) -> tuple[str, float]:
    start = time.perf_counter()
    response = requests.post(
        HF_API_URL.format(model=model),
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 60}},
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        output = data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        output = data["generated_text"]
    else:
        output = str(data)
    return output.strip(), elapsed_ms
# === end abandoned HF backend ================================================


# === Ollama backend (active) =================================================
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

OLLAMA_PROMPT_TEMPLATE = (
    "Correct any technical/domain-specific terminology transcription errors "
    "in the following text. Return only the corrected text, no explanation: {text}"
)


def ollama_models_available(host: str = OLLAMA_HOST, timeout: int = 5) -> list[str]:
    """Names of models Ollama currently has pulled, via GET /api/tags."""
    response = requests.get(f"{host}/api/tags", timeout=timeout)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


def call_ollama_inference(
    model: str, prompt: str, host: str = OLLAMA_HOST, timeout: int = 120,
    options: dict | None = None,
) -> tuple[str, float]:
    """Call POST /api/generate (non-streaming) and return (output, elapsed_ms).
    `options` is passed through verbatim (e.g. {"temperature": 0, "seed": 42} for
    reproducible runs); omitted by default so this script's own behavior is unchanged."""
    body = {"model": model, "prompt": prompt, "stream": False}
    if options:
        body["options"] = options
    start = time.perf_counter()
    response = requests.post(f"{host}/api/generate", json=body, timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    output = response.json()["response"]
    return output.strip(), elapsed_ms
# === end Ollama backend =======================================================


def _expected_domain_terms(text: str) -> list[str]:
    """Ground-truth domain term(s) for this sentence, reusing Mozhi's own
    dictionary-matching pipeline instead of building separate scoring
    machinery for the generative baseline."""
    result = correct_text(text)
    return sorted({m.replacement for m in result.matches})


def _is_hit(output: str, input_text: str, expected_terms: list[str]) -> bool:
    """A positive case hits if every expected domain term appears in the
    model's output (case-insensitive substring). A negative case (no domain
    term expected) hits if the model left the sentence alone."""
    if expected_terms:
        output_cf = output.casefold()
        return all(term.casefold() in output_cf for term in expected_terms)
    return output.strip().casefold() == input_text.strip().casefold()


def main():
    try:
        available = ollama_models_available()
    except requests.RequestException as error:
        print(f"ERROR: Ollama not reachable at {OLLAMA_HOST}: {error}")
        print("Start it with `ollama serve` and retry.")
        return 1
    if OLLAMA_MODEL not in available:
        print(f"ERROR: model {OLLAMA_MODEL!r} not found in Ollama. Available: {available}")
        print(f"Pull it with: ollama pull {OLLAMA_MODEL}")
        return 1

    sentences = [text for text, _ in BATCH_CASES]
    print(f"Backend: Ollama ({OLLAMA_HOST}, model={OLLAMA_MODEL})")
    print(f"Sentences: {len(sentences)}")
    print(f"Prompt template: {OLLAMA_PROMPT_TEMPLATE!r}")
    print()

    results = []
    hits = 0
    errors = 0
    for index, (text, _expected_text) in enumerate(BATCH_CASES, start=1):
        expected_terms = _expected_domain_terms(text)
        try:
            output, elapsed_ms = call_ollama_inference(
                OLLAMA_MODEL, OLLAMA_PROMPT_TEMPLATE.format(text=text)
            )
        except requests.RequestException as error:
            errors += 1
            print(f"  [{index}/{len(sentences)}] REQUEST ERROR: {error}")
            continue
        hit = _is_hit(output, text, expected_terms)
        hits += hit
        results.append({"input": text, "output": output, "elapsed_ms": elapsed_ms})
        marker = "OK" if hit else "MISS"
        expected_label = ", ".join(expected_terms) if expected_terms else "unchanged"
        print(
            f"  [{index}/{len(sentences)}] [{elapsed_ms:7.1f}ms] [{marker:4}] "
            f"{text!r} -> {output!r}  (expected: {expected_label})"
        )

    if not results:
        print("\nNo successful calls.")
        return 1

    latencies = [r["elapsed_ms"] for r in results]
    accuracy = hits / len(sentences)  # errored calls count against the full corpus
    sorted_lat = sorted(latencies)
    p99_index = min(len(sorted_lat) - 1, math.ceil(0.99 * len(sorted_lat)) - 1)
    ollama_p99 = sorted_lat[p99_index]

    print()
    print(f"Successful calls: {len(results)}/{len(sentences)}")
    if errors:
        print(f"Failed calls: {errors}")
    print(f"  mean:   {statistics.mean(latencies):.1f} ms")
    print(f"  median: {statistics.median(latencies):.1f} ms")
    print(f"  min:    {min(latencies):.1f} ms")
    print(f"  max:    {max(latencies):.1f} ms")
    print(f"  p99:    {ollama_p99:.1f} ms  (n={len(sorted_lat)}, nearest-rank)")
    print(f"  Domain-term accuracy on this batch: {hits}/{len(sentences)} = {accuracy * 100:.1f}%")
    print("  NOTE: this batch accuracy is NOT compared against Mozhi below - see Notes.")
    print()
    print("=" * 78)
    print("Side-by-side: Mozhi (non-generative) vs Ollama (generative baseline)")
    print("=" * 78)
    col = f"{{:32}}{{:>18}}{{:>26}}"
    print(col.format("", "Mozhi", f"Ollama {OLLAMA_MODEL}"))
    print(col.format("Latency mean", f"{MOZHI_LATENCY_MEAN_MS:.3f} ms", f"{statistics.mean(latencies):.1f} ms"))
    print(col.format("Latency p99", f"{MOZHI_LATENCY_P99_MS:.3f} ms", f"{ollama_p99:.1f} ms"))
    print("Accuracy, this 34-case batch        not comparable - see Notes below")
    print(col.format("Domain-aware catch rate, real audio", MOZHI_REAL_AUDIO_CATCH_RATE, "n/a"))
    print("  (Ollama's real-audio catch rate is in reports/comparison_real_audio.txt,")
    print("   not this row - it needs the same 32-recording machinery, not this script)")
    print()
    print("Notes:")
    print("  - The accuracy row above is deliberately NOT a Mozhi-vs-Ollama comparison.")
    print("    These 34 sentences are Mozhi's own regression fixtures (tests/batch_cases.py),")
    print("    authored against its dictionary, so Mozhi's accuracy on them is circular by")
    print("    construction (previously misreported here as '34/34 = 100% vs Ollama's X%',")
    print("    which is not a valid comparison and has been removed). Ollama's accuracy on")
    print("    this same batch, printed above, is a real number for THIS SCRIPT's own")
    print("    narrow purpose (can a generic LLM do this task at all on short synthetic")
    print("    sentences) - it is not a fair fight against a system tuned on the very")
    print("    sentences being scored, and is not placed next to a Mozhi number here.")
    print("    The one honest accuracy comparison is on real, independently-collected")
    print("    audio: see reports/comparison_real_audio.txt (evaluate_accuracy.py's own")
    print("    WER/catch-rate machinery, run against both Mozhi and Ollama on the same")
    print("    32 volunteer recordings).")
    print("  - Latency is not like-for-like: Mozhi's figure is in-process")
    print("    time.perf_counter over 20 runs/sentence (reports/latency_benchmark_")
    print("    letter_composition.txt). Ollama's is wall-clock over one HTTP round trip")
    print("    per sentence to a local server, including model decode time.")
    print("  - With n=34, nearest-rank p99 always equals the batch max, so a single")
    print("    slow call dominates the reported p99. The first call in a run often")
    print("    includes Ollama's cold-start cost of loading the model into memory if")
    print("    the server was recently (re)started; this is a real wall-clock cost for")
    print("    that scenario, not excluded here, but it is not steady-state latency.")
    print()
    return 0


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main()
    output = buf.getvalue()
    print(output)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Host: {os.uname().sysname} {os.uname().machine}\n\n")
        f.write(output)
    print(f"Saved to {REPORT_PATH}")
    sys.exit(exit_code)
