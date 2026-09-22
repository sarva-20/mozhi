"""
Adversarial tests for letter-by-letter acronym composition.

1. COVERAGE: every acronym the composer is supposed to reach is spelled out
   with EVERY combination of its letter-sound variants and must be recovered
   exactly; every acronym G3 is supposed to block must stay unreachable under
   every spelling. Both sets are derived from data/domain_terms.json, and one
   test pins their sizes so dictionary growth forces a conscious update.
2. NEGATIVE CORPUS, three groups, each attacking a different guard. Meta-tests
   assert every sentence really contains the run it claims (measured with the
   composer's own punctuation-aware walk), because a "clean" result is only
   meaningful if the attack was real:
     A  all-ambiguous runs (G3 is the guard)
     B  runs containing a real-word unambiguous token such as kay/jay/zed/dub
        (G3 satisfied, so G2's exact dictionary match is the guard)
     P  runs that exist only if punctuation is ignored (G5 is the guard)
3. KNOWN NAME COLLISION: "Em, Dee, are you coming" still spells MDR. Marked
   with a strict @pytest.mark.xfail (record the limitation, don't hide or patch
   it unilaterally); it fails loudly once fixed. Never use an imperative
   pytest.xfail() call for this: it stops the test from ever running.

Root cause of the name collisions: "em", "en" and "dee" are classed unambiguous
(G3 satisfied) because nobody says them when they mean a plain word - but they
are also common first names and typographic terms, and tokenizer.py drops
punctuation. G5 (uniform separators, hard break at sentence-terminal marks)
closes the mixed-separator cases; a uniformly comma-separated run of names is
structurally identical to a comma-spelled acronym and stays open.
"""
import itertools
import re

import pytest

from app.dictionary import get_terms
from app.letter_composer import (
    COMPOSITION_EXCLUDED,
    LETTER_SOUNDS,
    compose_letter_acronym,
    compose_letter_acronym_spans,
    letter_run_length,
)
from app.pipeline import correct_text
from app.tokenizer import normalize, separators, tokenize

_BY_LETTER: dict[str, list[tuple[str, bool]]] = {}
for _sound, (_letter, _unamb) in LETTER_SOUNDS.items():
    _BY_LETTER.setdefault(_letter, []).append((_sound, _unamb))

_EXCLUDED = {t.casefold() for t in COMPOSITION_EXCLUDED}
_LETTERS_ONLY = [
    t["term"]
    for t in get_terms()
    if t.get("is_acronym") and re.fullmatch(r"[A-Za-z]+", t["term"]) and len(t["term"]) >= 3
    and t["term"].casefold() not in _EXCLUDED
]


def _can_be_unambiguous(acronym: str) -> bool:
    return any(unamb for letter in acronym.upper() for _, unamb in _BY_LETTER[letter])


COMPOSABLE = [a for a in _LETTERS_ONLY if _can_be_unambiguous(a)]
G3_BLOCKED = [a for a in _LETTERS_ONLY if not _can_be_unambiguous(a)]


def _spellings(acronym: str) -> list[list[str]]:
    """Every spelling of `acronym` using every combination of table variants."""
    options = [[sound for sound, _ in _BY_LETTER[letter]] for letter in acronym.upper()]
    return [" ".join(combo).split() for combo in itertools.product(*options)]


# --- 1. coverage -----------------------------------------------------------------

def test_coverage_set_sizes_are_pinned():
    # Update these deliberately if data/domain_terms.json gains/loses acronyms.
    assert len(COMPOSABLE) == 61
    assert len(G3_BLOCKED) == 15


@pytest.mark.parametrize("acronym", COMPOSABLE)
def test_every_spelling_of_every_composable_acronym_is_recovered(acronym):
    misses = []
    for tokens in _spellings(acronym):
        got = compose_letter_acronym(tokens)
        if got != (acronym, len(tokens)):
            misses.append((" ".join(tokens), got))
    assert not misses, f"{acronym}: {len(misses)} spelling(s) not recovered, e.g. {misses[:3]}"


@pytest.mark.parametrize("acronym", G3_BLOCKED)
def test_every_spelling_of_every_all_ambiguous_acronym_stays_unreachable(acronym):
    reachable = [" ".join(t) for t in _spellings(acronym) if compose_letter_acronym(t)]
    assert not reachable, f"{acronym}: reachable via {reachable[:3]}"


@pytest.mark.parametrize("acronym", COMPOSABLE)
def test_end_to_end_canonical_spelling_is_recovered_by_the_pipeline(acronym):
    # First variant of each letter, inside a sentence. Either the hand-mapped
    # acronym_map or the composer may win (acronym_map outranks it); what
    # matters is that the acronym is recovered.
    spoken = " ".join(_BY_LETTER[letter][0][0] for letter in acronym.upper())
    result = correct_text(f"check the {spoken} now")
    assert any(m.replacement == acronym for m in result.matches), result.corrected_text


# --- 2. negative corpus ------------------------------------------------------------

# Group A: runs of >=3 letter-sound words that are ALL ambiguous (ordinary words).
# G3 is the guard being attacked. Commas appear only where the run survives them
# (uniform), so each sentence stays a real attack under the punctuation rules.
NEGATIVE_AMBIGUOUS = [
    "Are you a tea drinker or a coffee drinker",
    "Oh I see you are here already",
    "I owe you a tea for helping me",
    "Why are you a bee keeper",
    "Be a sea captain or be a bee keeper",
    "You are a pea in a pod",
    "Oh I owe a bee a tea",  # contains the IOA collision ("i owe a") mid-run
    "Tea, sea, pea are all words that rhyme",  # TCP prefix, uniform commas
    "Look at the sea pea you see growing there",  # CPU prefix
    "The pirate found a pea eye patch and a hook",  # API prefix
    "Gee, you, I never knew that",  # GUI, uniform commas
    "I owe a hundred dollars",
    "If you are a pea or you are a bee, be a good sport",
]

# Group B: runs of >=3 that DO contain a real-word unambiguous token (ex/jay/kay/
# dub/zed/em...), so G3 is satisfied and G2 (exact dictionary match) has to be
# the guard.
NEGATIVE_WITH_UNAMBIGUOUS_WORDS = [
    "Kay you are late again",
    "Jay are you a tea drinker",
    "Jay, Kay, Dee, are you ready",
    "Zed are you a bee or a sea",
    "Why dub a bee movie",
    "In the en dash, I see em dashes too",
    "Ex I see you are here",
    "Kay oh I owe you a tea",
]

# Group P: runs that exist only if punctuation is ignored. They end at a
# sentence-terminal mark or where the separator changes (G5). The last two were
# real false positives (CMDB, RDP) before G5 existed.
NEGATIVE_PUNCTUATION_PROTECTED = [
    "Oh, I see, you are here already",
    "Thank you. Are you okay? Yes I am",
    "Say oh. Why? See a doctor if you are unwell",
    "Kay, you are late again",
    "Ex, I see you are here",
    "The pirate found a pea, eye patch and a hook",
    "Go see Em, Dee, be nice about it",  # was CMDB
    "How are Dee, Pee and Jay doing",  # was RDP
]

NEGATIVE_CORPUS = NEGATIVE_AMBIGUOUS + NEGATIVE_WITH_UNAMBIGUOUS_WORDS + NEGATIVE_PUNCTUATION_PROTECTED


def _tokens_and_gaps(sentence: str) -> tuple[list[str], list[str]]:
    toks = tokenize(sentence)
    return [normalize(t.text) for t in toks], separators(sentence, toks)


def _longest_run(sentence: str, *, punctuation_aware: bool, require_unambiguous: bool = False) -> int:
    """Longest run of letter-sound tokens at any start position, measured with the
    composer's own walk. With punctuation_aware=False the gap signal is withheld."""
    norm, gaps = _tokens_and_gaps(sentence)
    best = 0
    for i in range(len(norm)):
        n = letter_run_length(norm[i:], gaps[i:] if punctuation_aware else None)
        if require_unambiguous and not any(LETTER_SOUNDS.get(t, ("", False))[1] for t in norm[i : i + n]):
            continue
        best = max(best, n)
    return best


@pytest.mark.parametrize("sentence", NEGATIVE_AMBIGUOUS)
def test_corpus_a_really_contains_a_run_of_three_or_more_ambiguous_letter_words(sentence):
    assert _longest_run(sentence, punctuation_aware=True) >= 3
    norm, _ = _tokens_and_gaps(sentence)
    assert not any(LETTER_SOUNDS[t][1] for t in norm if t in LETTER_SOUNDS)


@pytest.mark.parametrize("sentence", NEGATIVE_WITH_UNAMBIGUOUS_WORDS)
def test_corpus_b_really_contains_a_run_of_three_or_more_including_an_unambiguous_word(sentence):
    assert _longest_run(sentence, punctuation_aware=True, require_unambiguous=True) >= 3


@pytest.mark.parametrize("sentence", NEGATIVE_PUNCTUATION_PROTECTED)
def test_corpus_p_run_exists_only_when_punctuation_is_ignored(sentence):
    blind = _longest_run(sentence, punctuation_aware=False)
    aware = _longest_run(sentence, punctuation_aware=True)
    assert blind >= 3
    assert aware < blind, "punctuation rule did not shorten any run; sentence does not exercise G5"


@pytest.mark.parametrize("sentence", NEGATIVE_CORPUS)
def test_negative_corpus_pipeline_produces_no_composed_match(sentence):
    result = correct_text(sentence)
    composed = [(m.original, m.replacement) for m in result.matches if m.method == "letter_composed"]
    assert composed == []


@pytest.mark.parametrize("sentence", NEGATIVE_CORPUS)
def test_negative_corpus_no_window_composes_at_composer_level(sentence):
    # Stronger than the pipeline check: independent of the DP, no start position
    # in the sentence yields any composed span at all (punctuation signal included).
    norm, gaps = _tokens_and_gaps(sentence)
    spans = [(i, compose_letter_acronym_spans(norm[i:], gaps=gaps[i:])) for i in range(len(norm))]
    assert [(i, s) for i, s in spans if s] == []


# --- 3. known name collision (a real false positive today) ----------------------------

# "Em, Dee, are" is a uniformly comma-separated run, which is structurally identical
# to a comma-spelled M-D-R. G5 cannot tell them apart; only capitalisation or
# semantics could. Kept as a strict xfail so the limitation is visible and a future
# fix flips it loudly.
KNOWN_NAME_COLLISIONS = [
    pytest.param(
        "Em, Dee, are you coming to dinner",
        "MDR",
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "known limitation: uniformly comma-separated names ('Em, Dee, are') are "
                "structurally identical to a comma-spelled acronym; G5 cannot distinguish them"
            ),
        ),
    ),
]


@pytest.mark.parametrize("sentence,leaked_term", KNOWN_NAME_COLLISIONS)
def test_names_spelled_like_letters_do_not_compose(sentence, leaked_term):
    result = correct_text(sentence)
    assert not any(
        m.method == "letter_composed" and m.replacement == leaked_term for m in result.matches
    )
