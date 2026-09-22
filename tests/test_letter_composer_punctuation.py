"""
Punctuation handling (gate G5) in letter composition.

The risk this file exists to pin down: a rule that ends a letter-sound run at
punctuation closes false positives like "Go see Em, Dee, be nice" -> CMDB, but
must NOT also break legitimate spelled-out acronyms that carry commas, because
Whisper may well emit "ess, ess, dee" or "vee-ell-ay-en". The rule is therefore
"hard break at sentence-terminal marks, and the separators between consecutive
letters must be uniform", not "any comma breaks the run".

Legitimate-spelling tests use composer-only acronyms (VLAN, XSS, SQLi, WAN) as
well as SSD: SSD alone would prove nothing, because acronym_map recovers it
whatever the composer does.
"""
import pytest

from app.letter_composer import (
    HARD_BREAK_CHARS,
    compose_letter_acronym,
    letter_run_length,
)
from app.pipeline import correct_text
from app.tokenizer import separators, tokenize


# --- tokenizer.separators -----------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("vee ell ay en", ["", "", ""]),
        ("vee, ell, ay, en", [",", ",", ","]),
        ("vee-ell-ay-en", ["-", "-", "-"]),
        ("Go see Em, Dee, be nice", ["", "", ",", ",", ""]),
        ("thank you. Are you okay?", ["", ".", "", ""]),
        ("one  ,   two", [","]),  # whitespace around punctuation is stripped
        ("solo", []),
    ],
)
def test_separators(text, expected):
    assert separators(text, tokenize(text)) == expected


# --- legitimate comma / hyphen-carrying spellings must still compose ----------------------

# (spoken tokens, gaps between them, expected term)
LEGIT_SEPARATED_SPELLINGS = [
    ("ess ess dee", [",", ","], "SSD"),
    ("vee ell ay en", [",", ",", ","], "VLAN"),
    ("ex ess ess", [",", ","], "XSS"),
    ("ess cue ell eye", [",", ",", ","], "SQLi"),
    ("vee ell ay en", ["-", "-", "-"], "VLAN"),
    ("vee ell ay en", ["", "", ""], "VLAN"),  # no separators at all
    # W is two tokens; the gap INSIDE "double you" is not an inter-letter separator.
    ("double you ay en", ["", ",", ","], "WAN"),
    ("double you ay en", ["", "", ""], "WAN"),
    ("double you ay en", ["", "-", "-"], "WAN"),
]


@pytest.mark.parametrize("spoken,gaps,term", LEGIT_SEPARATED_SPELLINGS)
def test_composer_accepts_uniformly_separated_spellings(spoken, gaps, term):
    tokens = spoken.split()
    assert compose_letter_acronym(tokens, gaps=gaps) == (term, len(tokens))


# Whisper-plausible sentences, through the whole pipeline.
LEGIT_SEPARATED_SENTENCES = [
    ("the vee, ell, ay, en is down", "the VLAN is down", "VLAN", "letter_composed"),
    ("the vee, ell, ay, en, is down", "the VLAN, is down", "VLAN", "letter_composed"),  # trailing comma
    ("we saw the ex, ess, ess attack", "we saw the XSS attack", "XSS", "letter_composed"),
    ("open the ess, cue, ell, eye report", "open the SQLi report", "SQLi", "letter_composed"),
    ("the vee-ell-ay-en is down", "the VLAN is down", "VLAN", "letter_composed"),
    ("Ess, ess, dee is full", "SSD is full", "SSD", "acronym_map"),  # sentence-initial capital
    ("the ess, ess, dee is full", "the SSD is full", "SSD", "acronym_map"),
    ("check the double you, ay, en link", "check the WAN link", "WAN", "letter_composed"),
]


@pytest.mark.parametrize("text,expected,term,method", LEGIT_SEPARATED_SENTENCES)
def test_pipeline_recovers_separated_spellings(text, expected, term, method):
    result = correct_text(text)
    assert result.corrected_text == expected
    assert [(m.replacement, m.method) for m in result.matches] == [(term, method)]


# --- the leaks G5 closes, each with a control proving G5 is what closes it -------------------

# (tokens, gaps as in the sentence, term it spelled before G5)
CLOSED_LEAKS = [
    (["see", "em", "dee", "be"], ["", ",", ","], "CMDB"),  # "Go see Em, Dee, be nice"
    (["are", "dee", "pee"], ["", ","], "RDP"),  # "How are Dee, Pee and Jay"
]


@pytest.mark.parametrize("tokens,gaps,term", CLOSED_LEAKS)
def test_control_without_gap_signal_the_leak_composes(tokens, gaps, term):
    assert compose_letter_acronym(tokens) == (term, len(tokens))


@pytest.mark.parametrize("tokens,gaps,term", CLOSED_LEAKS)
def test_with_gap_signal_the_leak_is_closed(tokens, gaps, term):
    assert compose_letter_acronym(tokens, gaps=gaps) is None


@pytest.mark.parametrize(
    "text",
    ["Go see Em, Dee, be nice about it", "How are Dee, Pee and Jay doing"],
)
def test_closed_leaks_in_the_pipeline(text):
    assert not any(m.method == "letter_composed" for m in correct_text(text).matches)


# --- gap rules -------------------------------------------------------------------------------

@pytest.mark.parametrize("mark", sorted(HARD_BREAK_CHARS))
def test_every_hard_break_char_ends_the_run(mark):
    assert compose_letter_acronym(["vee", "ell", "ay", "en"], gaps=["", mark, ""]) is None
    assert letter_run_length(["vee", "ell", "ay", "en"], gaps=["", mark, ""]) == 2


def test_hard_break_inside_the_first_gap_leaves_a_run_of_one():
    assert letter_run_length(["vee", "ell", "ay"], gaps=[".", ""]) == 1


def test_hard_break_inside_a_multi_token_sound_is_rejected():
    assert compose_letter_acronym(["double", "you", "ay", "en"], gaps=[".", "", ""]) is None


def test_separator_change_ends_the_run_but_earlier_hits_survive():
    # "ess ess dee, ex": uniform through SSD, then the separator changes.
    index = {"ssd": "SSD", "ssdx": "SSDX"}
    tokens = ["ess", "ess", "dee", "ex"]
    assert compose_letter_acronym(tokens, index, gaps=["", "", ","]) == ("SSD", 3)
    assert compose_letter_acronym(tokens, index, gaps=["", "", ""]) == ("SSDX", 4)


def test_gaps_none_means_no_punctuation_information():
    assert compose_letter_acronym(["see", "em", "dee", "be"]) == ("CMDB", 4)


def test_gaps_too_short_is_an_error_not_a_silent_pass():
    with pytest.raises(ValueError):
        compose_letter_acronym(["vee", "ell", "ay", "en"], gaps=[","])


# --- documented limitations (pinned so they cannot change silently) ------------------------------

def test_limitation_mixed_separators_between_letters_stop_a_composer_only_acronym():
    # "vee ell, ay en": a real spelling with inconsistent punctuation. Not recovered
    # (composer-only acronym); the price of uniformity in exchange for closing CMDB/RDP.
    assert compose_letter_acronym(["vee", "ell", "ay", "en"], gaps=["", ",", ""]) is None


def test_limitation_periods_between_letters_stop_a_composer_only_acronym():
    assert compose_letter_acronym(["vee", "ell", "ay", "en"], gaps=[".", ".", "."]) is None
    assert correct_text("the vee. ell. ay. en. is down").matches == []


def test_limitation_periods_do_not_stop_a_hand_mapped_acronym():
    # acronym_map matches on normalized tokens and ignores punctuation entirely.
    result = correct_text("the ess. ess. dee. is full")
    assert [(m.replacement, m.method) for m in result.matches] == [("SSD", "acronym_map")]
