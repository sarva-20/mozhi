"""
Hand-written mis-transcription test cases.

These are illustrative examples authored to exercise each pipeline
component (acronym fallback, single-word fuzzy match, multi-word fuzzy
match, and known-failure cases) BEFORE real STT failure recordings are
collected from the team + volunteers. Once that dataset exists, this file
should be supplemented (not replaced) with cases mined from actual
failures, since hand-written cases can't be trusted alone for the
reviewer's "measured accuracy" requirement.
"""
import pytest
from app.pipeline import correct_text


ACRONYM_FALLBACK_CASES = [
    ("please check the sea pu usage", "CPU"),
    ("increase the ess ess dee capacity", "SSD"),
    ("the vee pea en keeps dropping", "VPN"),
    ("check the aitch dee dee health", "HDD"),
]


@pytest.mark.parametrize("text,expected_term", ACRONYM_FALLBACK_CASES)
def test_acronym_fallback_matches(text, expected_term):
    result = correct_text(text)
    assert expected_term in result.corrected_text
    assert any(m.method == "acronym_map" and m.replacement == expected_term for m in result.matches)


SINGLE_WORD_FUZZY_CASES = [
    ("check the ram usage", "RAM"),
    ("the kubernetees cluster is down", "Kubernetes"),
    ("the fire wall blocked it", "firewall"),
]


@pytest.mark.parametrize("text,expected_term", SINGLE_WORD_FUZZY_CASES)
def test_fuzzy_matches_domain_term(text, expected_term):
    result = correct_text(text)
    assert expected_term in result.corrected_text


@pytest.mark.parametrize(
    "text",
    [
        "the drive failed",
        "check drive D for space",
        "the hard drive is full",
        "eject the drive safely",
    ],
)
def test_drive_is_not_corrected_to_driver(text):
    result = correct_text(text)
    assert "driver" not in result.corrected_text


def test_driver_is_preserved_when_already_correct():
    result = correct_text("the driver failed")
    assert result.corrected_text == "the driver failed"


def test_no_false_positive_on_ordinary_words():
    """'configure' and 'connection' sound close enough to 'container' to
    false-positive at a lower threshold (0.72); this locks in the current
    default (0.80) so a future threshold change gets caught by a failing
    test rather than silently regressing precision."""
    result = correct_text("please configure the connection settings")
    assert "container" not in result.corrected_text.lower()


def test_no_false_positive_on_short_stopwords():
    """Short acronyms (2-3 letters) saturate Jaro-Winkler against common
    short function words - 'is' scored 0.88 against 'ISP' purely from the
    shared prefix, before the stopword guard was added."""
    result = correct_text("the weather today is quite pleasant")
    assert result.corrected_text == "the weather today is quite pleasant"
    assert result.matches == []


def test_no_false_positive_on_multiword_stopword_window():
    """All-stopword windows from real volunteer-style audio must not match
    technical acronyms through fuzzy scoring."""
    result = correct_text("check if there is a leg for you to escalate")
    assert "ISP" not in result.corrected_text


def test_long_acronym_beats_nested_short_acronym():
    result = correct_text("we saw a dee dee oh ess attack")
    assert result.corrected_text == "we saw a DDoS attack"


def test_does_not_touch_unrelated_text():
    text = "the weather today is quite pleasant"
    result = correct_text(text)
    assert result.corrected_text == text
    assert result.matches == []


def test_latency_is_measured_and_small():
    """Not a strict perf assertion (hardware-dependent) - just confirms
    elapsed_ms is populated and pipeline runs sub-second on a short
    sentence, per Review 1's 'prove latency, measured not estimated'."""
    result = correct_text("check the ram and sea pu usage")
    assert result.elapsed_ms > 0
    assert result.elapsed_ms < 1000


# --- Reordered multi-word term. This used to be an open issue: greedy
# left-to-right matching picked a suboptimal start position when a filler
# word ('the') sat directly before a term whose FIRST word was itself
# reordered by the STT error. It is fixed by the DP over all start positions
# (see the docstring below), so this is an ordinary asserting test with no
# xfail marker.
def test_reordered_multiword_term():
    """Fixed by the DP/exhaustive-start-position matching strategy: the
    greedy left-to-right scan used to consume 'the address' as a unit
    before 'address IP' (one token over) was ever tried, producing a
    duplicated stray 'IP'. The DP considers every start position and picks
    the globally best-scoring non-overlapping set."""
    result = correct_text("configure the address IP for the router")
    assert result.corrected_text == "configure the IP address for the router"
