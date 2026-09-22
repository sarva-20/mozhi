"""Benchmark a generative correction baseline using Hugging Face's free API.

Setup::

    pip install -r requirements-benchmark.txt
    export HF_TOKEN=hf_...
    PYTHONPATH=. python3 scripts/comparison_benchmark.py

Set ``HF_MODEL`` to try another model when the free serverless tier returns
404 or 503 for the default model. These failures are common because models
may be gated, unavailable to the provider, or cold-starting.
Create ``HF_TOKEN`` as a fine-grained token with the ``Make calls to Inference
Providers`` permission; older token types may fail with a permissions error.
See https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained.
"""
import os
import statistics
import time

import requests

from tests.batch_cases import BATCH_CASES

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


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN before running.")
        return 1

    model = os.environ.get("HF_MODEL", DEFAULT_MODEL)
    sentences = [text for text, _ in BATCH_CASES]
    print(f"Model: {model}")
    print(f"Sentences: {len(sentences)}")
    print("(The first call may be slow while the model cold-starts on HF.)")

    results = []
    errors = 0
    for index, text in enumerate(sentences, start=1):
        try:
            output, elapsed_ms = call_hf_inference(
                model, GEC_PROMPT_TEMPLATE.format(text=text), token
            )
            results.append({"input": text, "output": output, "elapsed_ms": elapsed_ms})
            print(f"  [{index}/{len(sentences)}] [{elapsed_ms:.0f}ms] {text!r} -> {output!r}")
        except requests.HTTPError as error:
            errors += 1
            print(f"  [{index}/{len(sentences)}] HTTP ERROR: {error}")
        except requests.RequestException as error:
            errors += 1
            print(f"  [{index}/{len(sentences)}] REQUEST ERROR: {error}")

    if not results:
        print("\nNo successful calls. Try another model with HF_MODEL:")
        print("  export HF_MODEL=" + MODEL_FALLBACKS[1])
        return 1

    latencies = [r["elapsed_ms"] for r in results]
    print()
    print(f"Successful calls: {len(results)}/{len(sentences)}")
    if errors:
        print(f"Failed calls: {errors}")
    print(f"  mean:   {statistics.mean(latencies):.1f} ms")
    print(f"  median: {statistics.median(latencies):.1f} ms")
    print(f"  min:    {min(latencies):.1f} ms")
    print(f"  max:    {max(latencies):.1f} ms")
    print()
    print("Compare against Mozhi's measured pipeline latency: mean 8.4ms, p99 13.6ms")
    print("(see reports/no_model_inference_proof.txt and scripts/latency_benchmark.py)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
