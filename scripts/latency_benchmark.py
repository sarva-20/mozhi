"""
Latency benchmark: runs the full batch of hand-written test sentences
through correct_text() and reports timing stats. This is early latency
evidence for Review 1's "prove latency, measured not estimated"
requirement - each call's elapsed_ms comes from time.perf_counter() inside
the pipeline itself (see app/pipeline.py), not a separate wall-clock
wrapper, so per-call timing reflects only the correction logic.

Run: PYTHONPATH=. python3 scripts/latency_benchmark.py
"""
import statistics

from app.pipeline import correct_text
from tests.batch_cases import BATCH_CASES

WARMUP_RUNS = 3
MEASURED_RUNS = 20


def main():
    sentences = [text for text, _ in BATCH_CASES]

    # Warm up (first calls pay for dictionary loading via lru_cache).
    for _ in range(WARMUP_RUNS):
        for s in sentences:
            correct_text(s)

    all_latencies = []
    for _ in range(MEASURED_RUNS):
        for s in sentences:
            result = correct_text(s)
            all_latencies.append(result.elapsed_ms)

    all_latencies.sort()
    n = len(all_latencies)
    mean = statistics.mean(all_latencies)
    median = statistics.median(all_latencies)
    p95 = all_latencies[int(n * 0.95)]
    p99 = all_latencies[int(n * 0.99)]
    worst = all_latencies[-1]
    best = all_latencies[0]

    print(f"Sentences: {len(sentences)}   Runs per sentence: {MEASURED_RUNS}   Total calls: {n}")
    print(f"  min:    {best:.3f} ms")
    print(f"  mean:   {mean:.3f} ms")
    print(f"  median: {median:.3f} ms")
    print(f"  p95:    {p95:.3f} ms")
    print(f"  p99:    {p99:.3f} ms")
    print(f"  max:    {worst:.3f} ms")


if __name__ == "__main__":
    main()
