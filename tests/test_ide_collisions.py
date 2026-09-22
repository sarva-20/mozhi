"""
IDE false positives: "idea" -> IDE (0.944), "I die" -> IDE (0.941), "id" -> IDE (0.919)
and "idle" -> IDE (0.905).

All four are the same weakness: IDE is a three-letter term whose Metaphone code ("IT") is
shared by ordinary words, so they score above DEFAULT_THRESHOLD against it. Raising the
threshold would break real matches ("catch" -> "cache" at 0.913), so they are blocked
word by word in NEVER_MATCH_PAIRS, the same mechanism as drive/driver.

"I die" surfaced when Whisper transcribed an MP3 re-encoding of test_audio1 as
"...then move away if I die speed outage" and the live demo showed "if IDE speed outage".
"id" and "idle" are very common in tech-support speech ("your ticket id", "the server is
idle"). Deliberately not blocked: "I dye" (0.905), not realistic tech-support vocabulary.
"""
import pytest

from app import pipeline
from app.pipeline import DEFAULT_THRESHOLD, NEVER_MATCH_PAIRS, _best_fuzzy_match, correct_text
from app.scorer import combined_score

# What Whisper base.en produced for the MP3 of test_audio1 (the m4a gave "...and verify the ISP outage.").
MP3_TRANSCRIPT = "Check if there is a leg for you escalate, then move away if I die speed outage."


def ide_matches(text):
    return [m for m in correct_text(text).matches if m.replacement == "IDE"]


# --- the threshold is untouched, and the scorer still rates these high -----------------------------------

def test_threshold_is_still_the_locked_value():
    assert DEFAULT_THRESHOLD == 0.88


@pytest.mark.parametrize("candidate,expected", [("i die", 0.941), ("idea", 0.944), ("id", 0.919), ("idle", 0.905)])
def test_scorer_still_scores_the_collisions_above_threshold(candidate, expected):
    # The fix is the guard, not a change to scoring: these must remain above the threshold.
    score = combined_score(candidate, "IDE")["total"]
    assert score == pytest.approx(expected, abs=1e-3) and score >= DEFAULT_THRESHOLD


def test_the_pairs_are_registered_alongside_drive_driver():
    assert {("drive", "driver"), ("idea", "ide"), ("die", "ide"), ("idle", "ide"), ("id", "ide")} <= NEVER_MATCH_PAIRS
    assert ("dye", "ide") not in NEVER_MATCH_PAIRS  # deliberately left out, see the module docstring
    assert all(term == term.casefold() and word == word.casefold() for word, term in NEVER_MATCH_PAIRS)


# --- the guard is what closes them (mutation control) -------------------------------------------------------

@pytest.mark.parametrize("candidate", ["idea", "i die", "if i die", "idle", "id", "your id", "the idle"])
def test_fuzzy_match_is_blocked(candidate):
    assert _best_fuzzy_match(candidate, ["IDE"], DEFAULT_THRESHOLD) == (None, 0.0, None)


@pytest.mark.parametrize("candidate", ["idea", "i die", "idle", "id"])
def test_control_without_the_guard_they_match_ide(monkeypatch, candidate):
    monkeypatch.setattr(pipeline, "NEVER_MATCH_PAIRS", frozenset())
    term, score, _ = _best_fuzzy_match(candidate, ["IDE"], DEFAULT_THRESHOLD)
    assert term == "IDE" and score >= DEFAULT_THRESHOLD


def test_drive_driver_is_still_blocked():
    assert _best_fuzzy_match("drive", ["driver"], DEFAULT_THRESHOLD)[0] is None


# --- through the pipeline ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "this is a great and simple idea",
    "that is a good idea",
    "I have no idea what happened",
    "if I die speed outage",
    "I die a little every time the build fails",
    MP3_TRANSCRIPT,
    "the server is idle",
    "the server is idle right now",
    "your ticket id",
    "please send me your ticket id",
    "what is your user id",
    "she said the cpu is idle again",
    "the machine sat idle all night",
    "check the id card at the desk",
])
def test_no_ide_misfire_in_ordinary_sentences(text):
    assert ide_matches(text) == []
    assert "IDE" not in correct_text(text).corrected_text


def test_the_mp3_sentence_from_the_demo_is_left_alone():
    result = correct_text(MP3_TRANSCRIPT)
    assert result.corrected_text == MP3_TRANSCRIPT and result.matches == []


def test_sla_and_isp_stay_honestly_unfixed_in_that_sentence():
    # "a leg for you" (SLA) and "speed" (ISP) are STT failures Mozhi does not attempt to
    # fix: no match, no forced correction. This pins the current, intended behaviour.
    corrected = correct_text(MP3_TRANSCRIPT).corrected_text
    assert "a leg for you" in corrected and "speed" in corrected
    assert "SLA" not in corrected and "ISP" not in corrected


# --- the exact paths still reach IDE ---------------------------------------------------------------------------------

def test_acronym_map_still_resolves_spoken_ide():
    result = correct_text("the eye dee ee is open")
    assert result.corrected_text == "the IDE is open"
    assert [(m.replacement, m.method) for m in result.matches] == [("IDE", "acronym_map")]


def test_letter_composition_still_resolves_spoken_ide():
    result = correct_text("please install the eye dee e plugin")
    assert result.corrected_text == "please install the IDE plugin"
    assert [(m.replacement, m.method) for m in result.matches] == [("IDE", "letter_composed")]


def test_a_blocked_word_elsewhere_does_not_stop_a_real_ide():
    result = correct_text("the idea is to open the eye dee ee")
    assert result.corrected_text == "the idea is to open the IDE"


@pytest.mark.parametrize("text,expected,method", [
    ("the ticket id is open in the eye dee ee", "the ticket id is open in the IDE", "acronym_map"),
    ("the server is idle while the eye dee e loads", "the server is idle while the IDE loads", "letter_composed"),
    ("the eye dee ee is idle and the user id is missing", "the IDE is idle and the user id is missing", "acronym_map"),
    ("your ticket id shows the eye dee e is open", "your ticket id shows the IDE is open", "letter_composed"),
])
def test_real_ide_resolves_beside_the_blocked_words_id_and_idle(text, expected, method):
    result = correct_text(text)
    assert result.corrected_text == expected
    assert [(m.replacement, m.method) for m in result.matches if m.replacement == "IDE"] == [("IDE", method)]
