"""
Tests for app/morphology.py and the NEVER_MATCH_PAIRS entries added alongside
it, both from Task 1 of the same investigation: a 32-file real-recording
accuracy run found 26 regressions (STT already correct, Mozhi broke it) across
10 patterns. Root cause split into two mechanisms, tested separately here:

  Class I  (app/morphology.is_ordinary_inflection): candidate is not a
           phonetic corruption of the term at all - it IS the term, just
           ordinarily inflected ("client's", "crashes", "containerized",
           "ticketing", "backed up"). A general rule, not a pair list.
  Class II (NEVER_MATCH_PAIRS): candidate is a genuinely different word or
           acronym that happens to sound like the term ("print"/"printer",
           "IPS"/"IPsec", "to"+"access port"/"access point"). No morphology
           connects these, so the general guard can't and shouldn't reach
           them; same targeted-pair mechanism as "idea"/"IDE".

Two of the ten original patterns ("catch"/"cache", "read"/"RAID") are
DELIBERATELY left unfixed: both have a confirmed true positive elsewhere
(tests/batch_cases.py's "clear the browser catch" -> "...cache", and
test_audio2's "read" -> "RAID"), so fixing the false positive would break a
confirmed true positive. Pinned here as a known, accepted limitation, the
same way "I dye" -> "IDE" is pinned in test_ide_collisions.py.
"""
import pytest

from app import pipeline
from app.morphology import is_ordinary_inflection, strip_inflection
from app.pipeline import NEVER_MATCH_PAIRS, _best_fuzzy_match, DEFAULT_THRESHOLD, correct_text


# --- strip_inflection: the building block -----------------------------------------------------------

@pytest.mark.parametrize("word,stem", [
    ("client's", "client"), ("crashes", "crash"), ("crashed", "crash"),
    ("ticketing", "ticket"), ("containerized", "container"), ("backed", "back"),
])
def test_strip_inflection_removes_one_ordinary_ending(word, stem):
    assert strip_inflection(word) == stem


@pytest.mark.parametrize("word", ["ram", "cache", "raid", "lan", "fire", "mother", "printer", "access"])
def test_strip_inflection_leaves_words_with_no_recognized_ending_unchanged(word):
    assert strip_inflection(word) == word


def test_strip_inflection_refuses_to_strip_down_to_a_trivial_stem():
    assert strip_inflection("is") == "is"  # would otherwise strip "s" -> "i"


def test_strip_inflection_has_no_bare_trailing_s_rule():
    """Deliberately absent (see the module docstring): a bare "s" strip reads
    "address"/"access"/"loss"/"CVSS"/"continuous"/"analytics" as if their
    final "s" were a plural marker, when it is just part of the spelling."""
    for word in ("address", "access", "loss", "cvss", "continuous", "analytics"):
        assert strip_inflection(word) == word


# --- is_ordinary_inflection: the guard, matching the module's own __main__ demo -----------------------

CLASS_I_TRUE = [
    ("client's", "client"), ("crashes", "crash"), ("crashed", "crash"),
    ("containerized", "container"), ("ticketing", "ticket"), ("backed up", "backup"),
]


@pytest.mark.parametrize("candidate,term", CLASS_I_TRUE)
def test_is_ordinary_inflection_true_for_the_ten_confirmed_regressions(candidate, term):
    assert is_ordinary_inflection(candidate, term) is True


MUST_STAY_FALSE = [
    ("ram", "RAM"),                        # case-only: the case-fix path must stay open
    ("fire wall", "firewall"),             # genuine STT segmentation fix, no stripping needed
    ("mother board", "motherboard"),       # same
    ("catch", "cache"),                    # different root entirely (Class II)
    ("read", "RAID"),                      # different root entirely (Class II)
    ("land", "LAN"),                       # different root entirely - recall must be preserved
    ("print", "printer"),                  # derivational, not inflectional (Class II)
    ("credential", "credentials"),         # bare "s" deliberately not stripped, see module docstring
    ("addres", "address"),                 # "address"'s own spelling, not "addres" + inflection
    ("cvs", "CVSS"),                       # "CVSS"'s own spelling, not "cvs" + inflection
    ("acces", "access"),                   # same
]


@pytest.mark.parametrize("candidate,term", MUST_STAY_FALSE)
def test_is_ordinary_inflection_false_where_it_must_not_fire(candidate, term):
    assert is_ordinary_inflection(candidate, term) is False


def test_fuses_to_requires_the_strip_to_be_non_trivial():
    # "fire" needs no stripping, so the phrasal-fusion rule must not fire even
    # though "fire" + "wall" concatenates to exactly "firewall".
    assert is_ordinary_inflection("fire wall", "firewall") is False
    # "backed" DOES need stripping ("backed" -> "back"), so this one fires.
    assert is_ordinary_inflection("backed up", "backup") is True


def test_is_ordinary_inflection_word_count_mismatch_outside_the_fuses_shape_is_false():
    assert is_ordinary_inflection("address IP", "IP address") is False  # reordering, not inflection
    assert is_ordinary_inflection("mac address table", "MAC address") is False  # 3 words vs 2, not the 2-vs-1 shape


# --- through the pipeline: the 10 confirmed regressions no longer fire -----------------------------

# (sentence, the spoken word/phrase that must survive verbatim, the term it
# used to get wrongly "corrected" to)
REGRESSION_SENTENCES = [
    ("the client's firewall is dropping our health check pings", "client's", "client"),
    ("the new employee still can't print from the third floor", "print", "printer"),
    ("we backed up the database before the migration started", "backed up", "backup"),
    ("that fix works locally but breaks in the containerized environment", "containerized", "container"),
    ("the mobile app crashes on launch for anyone on the older os version", "crashes", "crash"),
    ("that patch fixed the crash but the ticket got reopened after it crashed again", "crashed", "crash"),
    ("someone needs to close out that incident ticketing from last friday", "ticketing", "ticket"),
]


@pytest.mark.parametrize("text,spoken,wrongly_matched_term", REGRESSION_SENTENCES)
def test_regression_sentence_is_left_untouched(text, spoken, wrongly_matched_term):
    # Substring, not full-sentence equality: some of these sentences also contain
    # an unrelated, legitimate correction ("os" -> "OS"), which must still fire.
    # And check by ORIGINAL span, not just replacement term: "crash" legitimately
    # self-matches elsewhere in one of these sentences - only the SPOKEN word
    # ("crashed") must not be the thing that got matched to the term.
    result = correct_text(text)
    assert spoken in result.corrected_text
    assert not any(
        m.original.casefold() == spoken.casefold() and m.replacement == wrongly_matched_term
        for m in result.matches
    )


def test_whitelist_ips_is_not_corrected_to_ipsec():
    result = correct_text("we need to whitelist that IPS before the integration test will pass")
    assert result.corrected_text == "we need to whitelist that IPS before the integration test will pass"


@pytest.mark.parametrize("text", [
    "we need read access to the shared drive",
    "can you confirm the client has access to the shared folder",
])
def test_access_to_is_not_corrected_to_access_port_or_access_point(text):
    result = correct_text(text)
    assert not any(m.replacement in ("access port", "access point") for m in result.matches)


# --- recall preserved: real corrections and real recoveries are untouched --------------------------

RECALL_MUST_SURVIVE = [
    ("check the ram usage", "RAM"),
    ("the fire wall blocked it", "firewall"),
    ("the mother board for damage", "motherboard"),
    ("we need read access to the shared raid", "RAID"),
]


@pytest.mark.parametrize("text,expected_term", RECALL_MUST_SURVIVE)
def test_genuine_corrections_still_fire(text, expected_term):
    result = correct_text(text)
    assert any(m.replacement == expected_term for m in result.matches), result.corrected_text


def test_land_still_corrects_to_lan():
    result = correct_text("the land is fine but the connection keeps dropping")
    assert any(m.replacement == "LAN" for m in result.matches)


# --- deliberately NOT fixed: catch/cache and read/RAID are irreducibly ambiguous -------------------

def test_catch_cache_is_not_in_never_match_pairs():
    assert ("catch", "cache") not in NEVER_MATCH_PAIRS


def test_catch_still_misfires_to_cache_a_known_accepted_limitation():
    # Confirmed true positive elsewhere: tests/test_batch_cases.py's
    # "clear the browser catch" -> "clear the browser cache". Blocking "catch"
    # globally would fix this sentence but break that one; not attempted.
    result = correct_text("that is a good catch, I hadn't thought about the edge case")
    assert any(m.replacement == "cache" for m in result.matches)


def test_read_still_misfires_to_raid_a_known_accepted_limitation():
    # Confirmed true positive elsewhere: test_audio2's "read" -> "RAID".
    result = correct_text("we need read access to the shared drive")
    assert any(m.replacement == "RAID" for m in result.matches)


# --- the four new NEVER_MATCH_PAIRS entries: registered, and a mutation control ---------------------

NEW_CLASS_II_PAIRS = {("print", "printer"), ("ips", "ipsec"), ("to", "access port"), ("to", "access point")}


def test_new_pairs_are_registered():
    assert NEW_CLASS_II_PAIRS <= NEVER_MATCH_PAIRS


@pytest.mark.parametrize("candidate,term", [
    ("print", "printer"), ("ips", "IPsec"), ("access to", "access port"), ("access to", "access point"),
])
def test_class_ii_pairs_are_blocked(candidate, term):
    assert _best_fuzzy_match(candidate, [term], DEFAULT_THRESHOLD) == (None, 0.0, None)


@pytest.mark.parametrize("candidate,term", [
    ("print", "printer"), ("ips", "IPsec"), ("access to", "access port"), ("access to", "access point"),
])
def test_control_without_the_guard_they_would_misfire(monkeypatch, candidate, term):
    monkeypatch.setattr(pipeline, "NEVER_MATCH_PAIRS", frozenset())
    matched_term, score, _ = _best_fuzzy_match(candidate, [term], DEFAULT_THRESHOLD)
    assert matched_term == term and score >= DEFAULT_THRESHOLD


def test_blocking_access_port_alone_does_not_leave_access_point_exposed():
    """Found during verification: blocking only ("to", "access port") left
    "access to" -> "access point" (the dictionary's only other "access "
    term) as the new best score. Both must be blocked together."""
    assert _best_fuzzy_match("access to", ["access point", "access port"], DEFAULT_THRESHOLD) == (None, 0.0, None)
