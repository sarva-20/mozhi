# Comparison benchmark: free-tier setup notes

The paid OpenAI and Anthropic clients were removed. The generative baseline now
uses Hugging Face's free serverless Inference API:

```bash
pip install -r requirements-benchmark.txt
export HF_TOKEN=hf_...
PYTHONPATH=. python3 scripts/comparison_benchmark.py
```

The default model is `google/flan-t5-base`. Set `HF_MODEL` to one of the
fallbacks when the free tier returns 404 or 503:

```bash
export HF_MODEL=google/flan-t5-large
```

The benchmark calls the current router endpoint,
`https://router.huggingface.co/hf-inference/models/{model}`. Create `HF_TOKEN`
as a fine-grained token with the **Make calls to Inference Providers**
permission; older token types may fail with a permissions error. Token setup:
https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained

The free serverless tier does not reliably host every model on demand. Models
may be gated, unavailable to the provider, or cold-starting. These failures
are reported per sentence rather than hiding the limitation. The benchmark
records wall-clock latency and compares successful calls with Mozhi's measured
pipeline latency of 24.66 ms mean and 38.25 ms p99 (measured by
`scripts/latency_benchmark.py`; full output in
`reports/latency_benchmark_letter_composition.txt`). This reflects the current
319-term dictionary, up from the original 76-term baseline, with the
letter-composition module included. That module adds negligible overhead
(~0.007 ms per sentence for the whole pre-pass including punctuation handling,
measured directly; see the addendum in the same report).

## Local STT baseline

`scripts/transcribe_local.py` uses `faster-whisper` on CPU with the `base.en`
model and `int8` compute type. It accepts an audio file or directory, reports
real-time factor per file, and writes transcripts to
`data/stt_transcripts/<filename>.json` for later ground-truth annotation.
ffmpeg must be available on PATH. Model weights are downloaded on first use,
so transcription needs normal internet access.

Both scripts use dependencies in `requirements-benchmark.txt`; these remain
separate from the deployed FastAPI service's `requirements.txt`.
