"""
Correctness test over tests/batch_cases.py. Every case runs and asserts; nothing is xfailed.

History: two cases used to be skipped as "known word/domain-term collisions".
  - "this is a great and simple idea" ("idea" vs "IDE", 0.944) was fixed with
    ("idea", "ide") in pipeline.NEVER_MATCH_PAIRS.
  - "restart the dokker container" ("restart" vs "restore", 0.873) had already been fixed
    when the fuzzy threshold was raised from 0.80 to 0.88, but the imperative
    pytest.xfail() (called inside the test, so the case never ran) hid that. It now runs.

DEFAULT_THRESHOLD stays 0.88 (a lowering to 0.83 was tried and reverted - see
tests/test_threshold_investigation.py and the DEFAULT_THRESHOLD comment in
app/pipeline.py), but "restart" -> "restore" now has TWO independent layers of
protection instead of one: the threshold margin (0.873 < 0.88, as before) AND a
permanent NEVER_MATCH_PAIRS entry that does not depend on the threshold value at
all. Both are verified below.
"""
import pytest
from app.pipeline import DEFAULT_THRESHOLD, NEVER_MATCH_PAIRS, correct_text
from app.scorer import combined_score
from tests.batch_cases import BATCH_CASES


@pytest.mark.parametrize("text,expected", BATCH_CASES)
def test_batch_case(text, expected):
    result = correct_text(text)
    assert result.corrected_text == expected


# --- "restart" vs "restore": now double-protected - threshold margin AND a guard -----------------------

RESTART_CASE = "restart the dokker container"


def test_threshold_is_still_0_88():
    assert DEFAULT_THRESHOLD == 0.88


def test_restart_does_not_become_restore():
    # Both are real, common words and "restore" is in the dictionary. The scorer
    # rates them 0.873, only 0.007 under DEFAULT_THRESHOLD: rejected by margin alone,
    # same as always - AND, independently, by the guard verified below.
    assert combined_score("restart", "restore")["total"] < DEFAULT_THRESHOLD
    result = correct_text(RESTART_CASE)
    assert result.corrected_text == "restart the Docker container"
    assert "restore" not in result.corrected_text.lower()
    assert [(m.replacement, m.method) for m in result.matches if m.replacement.lower() == "restore"] == []


def test_the_pair_is_registered():
    assert ("restart", "restore") in NEVER_MATCH_PAIRS


def test_the_guard_alone_is_sufficient_even_if_the_threshold_were_lower():
    # At a threshold low enough to clear 0.873 on its own (0.87, the old fragile
    # margin), the sentence must still resolve correctly - proving the guard, not
    # just threshold margin, now protects this pair. This is what makes a future
    # threshold change (if one is ever revisited) safe for THIS pair specifically,
    # even though the broader 0.83 attempt was not safe overall.
    assert combined_score("restart", "restore")["total"] > 0.87
    assert correct_text(RESTART_CASE, threshold=0.87).corrected_text == "restart the Docker container"


def test_control_without_the_guard_it_would_misfire_at_a_lower_threshold(monkeypatch):
    # Isolates the claim above: remove only this one pair (not all of
    # NEVER_MATCH_PAIRS) and confirm the old, pre-guard behavior returns.
    import app.pipeline as pipeline_module
    monkeypatch.setattr(
        pipeline_module, "NEVER_MATCH_PAIRS", frozenset(p for p in NEVER_MATCH_PAIRS if p != ("restart", "restore"))
    )
    assert correct_text(RESTART_CASE, threshold=0.87).corrected_text == "restore the Docker container"


def test_the_docker_correction_itself_is_far_above_the_threshold():
    # The half of the case that must succeed is not borderline.
    (docker,) = [m for m in correct_text(RESTART_CASE).matches if m.replacement == "Docker"]
    assert (docker.method, docker.score) == ("fuzzy", pytest.approx(0.927, abs=1e-3))
    assert docker.score - DEFAULT_THRESHOLD > 0.04


# --- "lad then" vs "latency": still rejected by the threshold, unguarded (and that's fine) -----------------

def test_lad_then_still_rejected_by_the_threshold_alone():
    # The other original false positive. No guard exists for it, and none was ever
    # needed at 0.88: 0.822 stays comfortably below. Part of why the 0.83 attempt
    # failed was NOT this pair (it stayed safely below 0.83 too) but the wide sweep
    # of OTHER, previously-unexamined collisions it exposed - see
    # tests/test_threshold_investigation.py.
    score = combined_score("lad then", "latency")["total"]
    assert score == pytest.approx(0.822, abs=1e-3)
    assert score < DEFAULT_THRESHOLD
    result = correct_text("check the lad then before you deploy")
    assert result.corrected_text == "check the lad then before you deploy"
    assert result.matches == []
