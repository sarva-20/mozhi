"""
DP fragmentation defect: six correctly-spoken multi-word terms did not survive
correct_text() unchanged. The full-dictionary coverage test
(scripts/dictionary_coverage.py) found all six as "canonical form altered".

Root cause: the DP over token positions (app/pipeline.py step 2) maximized the
raw SUM of per-span scores. Score is per-MATCH, not per-token, so a 1-token
exact match and a 5-token exact match both contribute exactly 1.0. Whenever a
correctly-spoken multi-word term has a sub-word that is ALSO a standalone
dictionary term, splitting the span into {sub-word -> itself, 1.0} +
{remainder -> a genuine partial-phrase fuzzy match against the full term,
<1.0 but >= threshold} always scores higher than {whole span -> the one
correct exact match, 1.0} - and splices back to a literal duplicated word:

    SQL injection                        -> SQL SQL injection
    TCP handshake                        -> TCP TCP handshake
    configuration management database    -> configuration management database database
    web application firewall             -> web application firewall firewall

"living off the land" is the same shape but the leftover word matches a
DIFFERENT term (LAN) instead of itself:

    living off the land                  -> living off the land LAN

Fix: the DP now compares (exact_tokens_covered, score_sum) lexicographically
instead of score_sum alone, where exact_tokens_covered counts tokens under a
match that is exact by construction (acronym_map / phrase_alias /
letter_composed) or a fuzzy match scoring exactly 1.0 (verbatim, not a
heuristic guess). A single span that verbatim covers N tokens can then never
lose to a fragmentation that verifies fewer than N of those tokens, no matter
how high the fragmentation's raw score sums.

"server-side request forgery" is the sixth and does NOT go through the DP fix:
its canonical form is hyphenated, so the correct full-phrase candidate never
scores a full 1.0 (candidate has a space, term has a hyphen) - no exactness
preference can favor a 0.965 span over a fragmentation that includes an
exact 1.0 piece. It is fixed the other way: "side" -> SSID (0.885, the most
frequent bad replacement in the coverage run - 11 occurrences, unrelated to
this specific phrase) is blocked in NEVER_MATCH_PAIRS, the same mechanism as
drive/driver and the IDE collisions.
"""
import pytest

from app import pipeline
from app.pipeline import DEFAULT_THRESHOLD, NEVER_MATCH_PAIRS, _best_fuzzy_match, correct_text
from app.scorer import combined_score

CARRIER = "please check the {} now"

# term -> the sub-word that is also independently a valid match, and its score
# against the FULL term (this is the fragment the DP used to prefer).
FRAGMENTATION_CASES = [
    ("SQL injection", "injection", 0.887),
    ("TCP handshake", "handshake", 0.883),
    ("configuration management database", "configuration management", 0.920),
    ("web application firewall", "web application", 0.888),
    ("living off the land", "living off the", 0.919),
]

ALL_SIX_TERMS = [
    "SQL injection", "TCP handshake", "configuration management database",
    "web application firewall", "living off the land", "server-side request forgery",
]


# --- the scorer still rates the fragments above threshold: the fix is the DP objective, not the scorer ---

@pytest.mark.parametrize("term,fragment,expected", FRAGMENTATION_CASES)
def test_the_partial_phrase_fragment_still_scores_above_threshold(term, fragment, expected):
    score = combined_score(fragment, term)["total"]
    assert score == pytest.approx(expected, abs=1e-3) and score >= DEFAULT_THRESHOLD


def test_side_still_scores_above_threshold_against_ssid():
    score = combined_score("side", "SSID")["total"]
    assert score == pytest.approx(0.885, abs=1e-3) and score >= DEFAULT_THRESHOLD


# --- correctly-spoken input: all six must now come back unchanged --------------------------------------

@pytest.mark.parametrize("term", ALL_SIX_TERMS)
def test_correctly_spoken_term_survives_unchanged(term):
    text = CARRIER.format(term)
    result = correct_text(text)
    assert result.corrected_text == text, result.corrected_text


@pytest.mark.parametrize("term", ALL_SIX_TERMS)
def test_correctly_spoken_term_produces_no_duplicated_word(term):
    words = correct_text(CARRIER.format(term)).corrected_text.split()
    dupes = [w for w, nxt in zip(words, words[1:]) if w.casefold() == nxt.casefold()]
    assert dupes == [], f"duplicated word(s) {dupes} in {correct_text(CARRIER.format(term)).corrected_text!r}"


# --- the DP objective is what fixes the first five; a control proves it ------------------------------

@pytest.mark.parametrize("term,fragment,fragment_score", FRAGMENTATION_CASES)
def test_control_the_fragment_sum_genuinely_exceeds_the_exact_match(term, fragment, fragment_score):
    """Not a near-tie: the fragmented total is comfortably higher than the
    single correct match (1.0), which is why a near-tie bonus could not have
    fixed this - only preferring verified token coverage over raw score does."""
    leftover_words = len(term.split()) - len(fragment.split())
    assert leftover_words >= 1
    # the leftover sub-word matching itself contributes another 1.0
    assert fragment_score + 1.0 * leftover_words > 1.0 + 0.5  # comfortable margin, not a tie


# --- genuine mis-transcriptions of these terms must still recover (the fix must not be one-directional) ---

# Picked so the corruption does NOT leave an intact sub-word that is itself a
# standalone dictionary term (a corrupted "SQL" -> "sequel" is a common real
# STT substitution and doesn't collide with anything); that is the scenario
# this fix targets. Confirmed against the real pipeline, and cross-checked
# against the coverage rerun's per-term CSV showing zero regressions across
# all 319 terms (2 of these two are its own "recovered" synthetic variants).
GENUINE_MISTRANSCRIPTIONS = [
    ("please check the sequel injection now", "SQL injection"),
    ("please check the lyving off the land now", "living off the land"),
    ("please check the servur side request forgery now", "server-side request forgery"),
]


@pytest.mark.parametrize("text,expected_term", GENUINE_MISTRANSCRIPTIONS)
def test_genuine_mistranscription_of_the_same_terms_still_recovers(text, expected_term):
    result = correct_text(text)
    assert any(m.replacement == expected_term for m in result.matches), result.corrected_text


def test_three_of_the_six_have_no_synthetic_recovery_path_at_all_pre_existing():
    """TCP handshake, configuration management database and web application
    firewall are not reachable by ANY single-word synthetic corruption
    (scripts/dictionary_coverage.py: 0/10 each, before and after this fix,
    confirmed identical by hand against the old scoring-only DP) - because if
    the leading word survives the corruption intact it re-triggers this exact
    defect's shape (it self-matches and wins over the corrupted full phrase),
    and if it doesn't survive intact, the DP falls back to comparing raw
    score, which this fix does not change (both sides are non-exact, so it is
    the same fragmentation defect this fix does not reach). This is a
    separate, pre-existing gap in fuzzy recovery, not something this fix was
    scoped to close (the task's own priority framing: correctly-spoken input
    first). Pinned so a future attempt to close it is a deliberate, visible
    change, not an accidental side effect."""
    for text, term in [
        ("please check the TCP handshak now", "TCP handshake"),
        ("please check the configuration management databaze now", "configuration management database"),
        ("please check the web application firewal now", "web application firewall"),
    ]:
        result = correct_text(text)
        assert result.corrected_text != CARRIER.format(term)


# --- side -> SSID: registered, and a mutation control proves the guard (not luck) is what closes it ----

def test_side_ssid_is_registered_in_never_match_pairs():
    assert ("side", "ssid") in NEVER_MATCH_PAIRS


def test_side_alone_does_not_match_ssid():
    assert _best_fuzzy_match("side", ["SSID"], DEFAULT_THRESHOLD) == (None, 0.0, None)


def test_control_without_the_guard_side_matches_ssid():
    import app.pipeline as pl
    pairs_without_side = frozenset(p for p in NEVER_MATCH_PAIRS if p != ("side", "ssid"))
    orig = pl.NEVER_MATCH_PAIRS
    try:
        pl.NEVER_MATCH_PAIRS = pairs_without_side
        term, score, _ = _best_fuzzy_match("side", ["SSID"], DEFAULT_THRESHOLD)
        assert term == "SSID" and score >= DEFAULT_THRESHOLD
    finally:
        pl.NEVER_MATCH_PAIRS = orig


@pytest.mark.parametrize("text", [
    "please check the server-side request forgery now",
    "the client-side rendering is slow",
    "check the server side logs",
    "the outside temperature is fine",
])
def test_side_never_misfires_to_ssid_in_ordinary_sentences(text):
    result = correct_text(text)
    assert not any(m.replacement == "SSID" for m in result.matches)


def test_real_ssid_recovery_is_unaffected_by_the_side_guard():
    # SSID reached through its own dedicated fuzzy/acronym path, not through "side".
    result = correct_text("check the ess ess eye dee network name")
    assert any(m.replacement == "SSID" for m in result.matches), result.corrected_text


# --- test_long_acronym_beats_nested_short_acronym-style check: exact_tokens comparison picks the LONGER exact span ---

def test_longer_exact_span_still_beats_a_nested_shorter_exact_span():
    result = correct_text("we saw a dee dee oh ess attack")
    assert result.corrected_text == "we saw a DDoS attack"
