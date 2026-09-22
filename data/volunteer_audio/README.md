# Volunteer audio for accuracy evaluation

Each recording is a pair, sharing a stem:

    <name>.m4a             the recording (any extension scripts/transcribe_local.py accepts)
    <name>.expected.txt    what the speaker actually said, written by a human

`scripts/evaluate_accuracy.py` evaluates every complete pair. A recording with no
`.expected.txt` (or the reverse) is reported and skipped, never guessed at.

Writing the expected text:
- Write what was SAID, exactly, including domain terms in their canonical form
  ("SLA", "WAN link"), not what you think the STT should have produced.
- Case and punctuation are ignored when scoring. Hyphens and dots split a word
  ("two-factor" and "X.509" each count as more than one word), identically on
  both sides, so scoring stays consistent.
- One utterance per file keeps the per-file numbers interpretable.

These are recordings of real people. `.dockerignore` keeps this directory out of
the Docker image; keep it out of anything published.
