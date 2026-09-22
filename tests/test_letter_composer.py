"""
Tests for letter-by-letter acronym composition (app/letter_composer.py).

Each gate (G1-G4) is tested with a positive control that flips it, so a
"returns None" assertion can't pass merely because a different gate fired.
Composer-level tests pass their own acronym_index; pipeline-level tests use
the real dictionary.
"""
import pytest

from app import letter_composer
from app.dictionary import get_acronym_index
from app.letter_composer import (
    COMPOSITION_EXCLUDED,
    LETTER_SOUNDS,
    MAX_LETTER_WINDOW,
    MIN_ACRONYM_LETTERS,
    REQUIRE_UNAMBIGUOUS,
    compose_letter_acronym,
    compose_letter_acronym_spans,
)
from app.pipeline import correct_text
from app.tokenizer import normalize


# --- constants locked by the design -----------------------------------------

def test_design_constants():
    assert MIN_ACRONYM_LETTERS == 3
    assert MAX_LETTER_WINDOW == 6
    assert REQUIRE_UNAMBIGUOUS is True
    assert COMPOSITION_EXCLUDED == frozenset({"SATA", "STP", "DDoS", "IDS", "PAT", "HSTS", "IPS"})


# --- letter-sound table -------------------------------------------------------

def test_table_covers_all_26_letters():
    assert {letter for letter, _ in LETTER_SOUNDS.values()} == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_table_keys_are_already_normalized():
    for key in LETTER_SOUNDS:
        assert all(normalize(tok) == tok for tok in key.split()), key


def test_only_w_has_multi_token_keys():
    multi = {key: letter for key, (letter, _) in LETTER_SOUNDS.items() if " " in key}
    assert multi and set(multi.values()) == {"W"}


@pytest.mark.parametrize(
    "sound,letter,unambiguous",
    [
        ("ess", "S", True),
        ("dee", "D", True),
        ("em", "M", True),
        ("vee", "V", True),
        ("double you", "W", True),
        ("see", "C", False),
        ("sea", "C", False),
        ("are", "R", False),
        ("you", "U", False),
        ("owe", "O", False),
        ("why", "Y", False),
    ],
)
def test_table_entries_and_ambiguity_class(sound, letter, unambiguous):
    assert LETTER_SOUNDS[sound] == (letter, unambiguous)


# --- happy path ---------------------------------------------------------------

@pytest.mark.parametrize(
    "spoken,term,consumed",
    [
        ("vee ell ay en", "VLAN", 4),
        ("ess cue ell eye", "SQLi", 4),  # canonical casing, not the composed uppercase
        ("ex ess ess", "XSS", 3),
        ("em dee are", "MDR", 3),
        ("vee ex ell ay en", "VXLAN", 5),
        ("double you ay en", "WAN", 4),  # two-token W
        ("dub ay en", "WAN", 3),
    ],
)
def test_composes_dictionary_acronyms(spoken, term, consumed):
    assert compose_letter_acronym(spoken.split()) == (term, consumed)


def test_double_alone_is_not_a_letter():
    assert compose_letter_acronym(["double", "ay", "en"]) is None


def test_walk_stops_at_non_letter_token():
    assert compose_letter_acronym(["vee", "ell", "banana", "ay", "en"]) is None


def test_spans_returns_every_hit_and_wrapper_returns_longest():
    index = {"ssd": "SSD", "ssdx": "SSDX"}
    tokens = ["ess", "ess", "dee", "ex"]
    assert compose_letter_acronym_spans(tokens, index) == [("SSD", 3), ("SSDX", 4)]
    assert compose_letter_acronym(tokens, index) == ("SSDX", 4)


# --- G1: minimum length --------------------------------------------------------

def test_g1_two_letter_run_rejected():
    index = {"ss": "SS"}
    assert compose_letter_acronym(["ess", "ess"], index) is None


def test_g1_control_lowering_minimum_admits_it(monkeypatch):
    monkeypatch.setattr(letter_composer, "MIN_ACRONYM_LETTERS", 2)
    assert compose_letter_acronym(["ess", "ess"], {"ss": "SS"}) == ("SS", 2)


# --- G2: exact dictionary membership --------------------------------------------

def test_g2_composed_string_not_in_index_rejected():
    assert compose_letter_acronym(["ess", "ess", "ay"]) is None  # "SSA" is not a dictionary acronym


def test_g2_control_same_run_resolves_when_present():
    assert compose_letter_acronym(["ess", "ess", "ay"], {"ssa": "SSA"}) == ("SSA", 3)


def test_g2_empty_index_resolves_nothing():
    assert compose_letter_acronym(["ess", "ess", "dee"], {}) is None


# --- G3: ambiguity gate -----------------------------------------------------------

@pytest.mark.parametrize(
    "spoken,term",
    [
        ("i owe a", "IOA"),      # the natural-speech collision the gate exists for
        ("see pea you", "CPU"),  # all-ambiguous, also hand-mapped in acronym_map
    ],
)
def test_g3_all_ambiguous_run_rejected(spoken, term):
    assert compose_letter_acronym(spoken.split()) is None


@pytest.mark.parametrize("spoken,term", [("i owe a", "IOA"), ("see pea you", "CPU")])
def test_g3_control_disabling_gate_makes_them_reachable(monkeypatch, spoken, term):
    monkeypatch.setattr(letter_composer, "REQUIRE_UNAMBIGUOUS", False)
    assert compose_letter_acronym(spoken.split()) == (term, 3)


def test_g3_one_unambiguous_letter_is_enough():
    # "em" (M) is unambiguous; "dee" (D) too. Mixed runs pass.
    assert compose_letter_acronym(["em", "dee", "are"]) == ("MDR", 3)


# --- G4: exclusion list ------------------------------------------------------------

# Spoken forms of the 7 hand-fixed acronyms, as they appear in acronym_map.
EXCLUDED_SPOKEN = {
    "SATA": "ess ay tee ay",
    "STP": "ess tee pee",
    "DDoS": "dee dee oh ess",
    "IDS": "eye dee ess",
    "PAT": "pea ay tee",
    "HSTS": "aitch ess tee ess",
    "IPS": "eye pea ess",
}


def test_excluded_spoken_forms_cover_the_whole_list():
    assert set(EXCLUDED_SPOKEN) == set(COMPOSITION_EXCLUDED)


@pytest.mark.parametrize("term,spoken", EXCLUDED_SPOKEN.items())
def test_g4_excluded_terms_unreachable_even_when_injected(monkeypatch, term, spoken):
    # Injected into the index AND with G3 off, so nothing but G4 can block them.
    monkeypatch.setattr(letter_composer, "REQUIRE_UNAMBIGUOUS", False)
    index = {**get_acronym_index(), term.casefold(): term}
    assert compose_letter_acronym(spoken.split(), index) is None
    assert compose_letter_acronym_spans(spoken.split(), index) == []


@pytest.mark.parametrize("term,spoken", EXCLUDED_SPOKEN.items())
def test_g4_control_without_exclusion_they_would_resolve(monkeypatch, term, spoken):
    monkeypatch.setattr(letter_composer, "REQUIRE_UNAMBIGUOUS", False)
    monkeypatch.setattr(letter_composer, "_EXCLUDED_FOLDED", frozenset())
    index = {**get_acronym_index(), term.casefold(): term}
    assert compose_letter_acronym(spoken.split(), index) == (term, len(spoken.split()))


@pytest.mark.parametrize("term,spoken", EXCLUDED_SPOKEN.items())
def test_excluded_terms_still_resolve_via_acronym_map_in_pipeline(term, spoken):
    result = correct_text(f"check the {spoken} now")
    assert any(m.replacement == term and m.method == "acronym_map" for m in result.matches)
    assert not any(m.method == "letter_composed" for m in result.matches)


# --- window bound -------------------------------------------------------------------

def test_max_letter_window_bounds_the_walk():
    index_seven = {"s" * 7: "S7"}
    index_six = {"s" * 6: "S6"}
    assert compose_letter_acronym(["ess"] * 7, index_seven) is None
    assert compose_letter_acronym(["ess"] * 6, index_six) == ("S6", 6)


# --- pipeline integration ------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected,term",
    [
        ("the vee ell ay en is misconfigured", "the VLAN is misconfigured", "VLAN"),
        ("check the double you ay en link", "check the WAN link", "WAN"),
        ("open the ess cue ell eye report", "open the SQLi report", "SQLi"),
        ("we saw the ex ess ess attack", "we saw the XSS attack", "XSS"),
        ("run the em dee are check", "run the MDR check", "MDR"),
    ],
)
def test_pipeline_composes_letter_runs(text, expected, term):
    result = correct_text(text)
    assert result.corrected_text == expected
    assert [(m.replacement, m.method) for m in result.matches] == [(term, "letter_composed")]


def test_pipeline_i_owe_a_hundred_dollars_does_not_produce_ioa():
    result = correct_text("i owe a hundred dollars")
    assert "IOA" not in result.corrected_text
    assert result.matches == []
    assert result.corrected_text == "i owe a hundred dollars"


def test_pipeline_ordinary_speech_with_ambiguous_letter_words_untouched():
    text = "we need to see you tomorrow"
    assert correct_text(text).corrected_text == text


def test_pipeline_acronym_map_outranks_letter_composed():
    # "ess ess dee" composes to SSD too; the hand-verified map must win.
    result = correct_text("the ess ess dee is full")
    assert [(m.replacement, m.method) for m in result.matches] == [("SSD", "acronym_map")]


def test_pipeline_composed_span_wider_than_max_window_is_still_found():
    # VXLAN is 5 tokens; max_window=4 means the scoring loop never visits it,
    # so this exercises the post-loop insertion path.
    result = correct_text("the vee ex ell ay en", max_window=4)
    assert result.corrected_text == "the VXLAN"
    assert [(m.replacement, m.method) for m in result.matches] == [("VXLAN", "letter_composed")]
