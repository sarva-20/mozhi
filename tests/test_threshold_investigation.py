"""
Investigation: can DEFAULT_THRESHOLD be safely lowered below 0.88 now that
NEVER_MATCH_PAIRS and app.morphology handle several precision risks explicitly?

Answer: no. Tried 0.83 (guarding "restart"/"restore" first, since it was the
one blocker for the two ORIGINAL false positives that justified 0.88 - see
tests/test_batch_cases.py for that pair's own dedicated tests). The narrow
two-case check passed. But a systematic sweep of every one of the 319
dictionary terms against a 10,000-word common-English reference list
(data/common_english_words.txt, the same list scripts/collision_check.py
uses) found 257 NEW word/term pairs clear 0.83 while staying below 0.88 -
compared to 129 pairs already above 0.88 (pre-existing, unrelated to this
investigation, and mostly protected in practice by other mechanisms:
self-matching plurals like "printers"/"printer" recover correctly rather
than misfiring, since the plural itself isn't a different concept - it is
the ones below that are the risk).

Reproducing the sweep (not run as part of the suite - ~8s, 3.19M scorer
calls, and it is one-time evidence, not a thing that needs re-checking on
every test run):

    PYTHONPATH=. python3 -c "
    from app.scorer import combined_score
    from app.dictionary import get_term_strings
    from scripts.collision_check import load_common_words
    for term in get_term_strings():
        for word in load_common_words():
            s = combined_score(word, term)['total']
            if 0.83 <= s < 0.88:
                print(word, term, s)
    "

Many of the 257 are short strings unlikely to stand alone in real speech
("io"/IOC, "ra"/RAM) and would need individual verification against real
recordings before blocking - exactly the standard every other guard in
NEVER_MATCH_PAIRS was held to (never pad speculatively). But a large
fraction are ordinary, highly plausible tech-support vocabulary: "app",
"access", "pass", "launch", "export", "gate", "service", "local", "demo",
"IP". Guarding all of them individually is not a small follow-up to one
guard, it is a different scale of undertaking, and doing it from a synthetic
word-list sweep alone (not real STT failures) would abandon this project's
own standard for when a guard is justified. DEFAULT_THRESHOLD therefore
stays at 0.88; the tests below pin a representative sample of the 257 as
regression evidence, so a future casual re-attempt at lowering the threshold
fails loudly here first, instead of silently reopening 257 false positives.
"""
import pytest

from app.pipeline import DEFAULT_THRESHOLD, correct_text
from app.scorer import combined_score

# --- final decision ---------------------------------------------------------------------------------

def test_default_threshold_is_0_88():
    assert DEFAULT_THRESHOLD == 0.88


# --- the two ORIGINAL false positives: still safe (see tests/test_batch_cases.py for the dedicated,
# dual-layer-protection tests for "restart"/"restore" specifically) --------------------------------

def test_the_two_original_false_positives_still_score_below_0_88():
    assert combined_score("lad then", "latency")["total"] < DEFAULT_THRESHOLD
    assert combined_score("restart", "restore")["total"] < DEFAULT_THRESHOLD


# --- representative sample of the 257 new collisions found at 0.83, none guarded -------------------
# Each pair scores in [0.83, 0.88): safe today, would misfire if the threshold were lowered to 0.83
# without addressing it. None of these went through the "verified against real speech" bar every
# other NEVER_MATCH_PAIRS/morphology entry did, which is exactly why none are guarded.

REPRESENTATIVE_NEW_COLLISIONS = [
    ("app", "API"), ("access", "access port"), ("pass", "passkey"), ("launch", "LAN"),
    ("IP", "IPv4"), ("IP", "IPv6"), ("is open", "ISP"), ("escalate", "escalation"),
    ("crashes on", "crash"), ("backed up the", "backup"), ("containerized", "containerization"),
]


@pytest.mark.parametrize("candidate,term", REPRESENTATIVE_NEW_COLLISIONS)
def test_representative_new_collision_scores_in_the_danger_band(candidate, term):
    score = combined_score(candidate, term)["total"]
    assert 0.83 <= score < DEFAULT_THRESHOLD, f"{candidate!r} vs {term!r} scored {score:.4f}, expected [0.83, 0.88)"


# --- proof through the real pipeline: safe today, would misfire at 0.83 ----------------------------

REPRESENTATIVE_SENTENCES = [
    ("the new employee still can't launch the app before lunch", "LAN", "API"),
    ("can you confirm the client has access to the shared folder", "access port", "access port"),
    ("we need to whitelist that IPS before the integration test will pass", "passkey", "passkey"),
    ("please check the client IP for that request", "IPv4", "IPv4"),
]


@pytest.mark.parametrize("text,term_a,term_b", REPRESENTATIVE_SENTENCES)
def test_representative_sentence_is_clean_at_0_88_but_would_misfire_at_0_83(text, term_a, term_b):
    clean = correct_text(text, threshold=0.88)
    assert not any(m.replacement in (term_a, term_b) for m in clean.matches), clean.corrected_text

    risky = correct_text(text, threshold=0.83)
    assert any(m.replacement in (term_a, term_b) for m in risky.matches), (
        f"expected a misfire to {term_a!r}/{term_b!r} at threshold 0.83 to demonstrate the risk, "
        f"got {risky.corrected_text!r} - if this now fails, the sweep's finding may be stale"
    )


# --- van -> WAN: a permanent, accepted limitation, not something still being worked on --------------

def test_van_wan_is_permanently_out_of_reach():
    """Documented limitation, not an open item. van/WAN (0.706) is far below any
    threshold this investigation found even provisionally workable (0.83 itself was
    rejected; van/WAN would need something closer to 0.70, which reopens vastly more
    than the 257 pairs already found unsafe at 0.83). No further attempt should be
    made to recover this without an entirely different mechanism than a global
    threshold change - not scoped here, and not a near-term goal."""
    score = combined_score("van", "WAN")["total"]
    assert score == pytest.approx(0.706, abs=1e-3)
    assert score < 0.83  # below even the rejected 0.83, let alone the actual 0.88
    result = correct_text("check the van settings before the trip")
    assert not any(m.replacement == "WAN" for m in result.matches)
