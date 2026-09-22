"""
Letter-by-letter acronym composition.

Third exact-lookup mechanism, alongside acronym_map (hand-enumerated spoken
forms) and phrase_alias (hand-verified mishearings). Instead of enumerating
each acronym's spoken form, this module keeps a small per-letter sound table
and composes runs of consecutive letter-sounds ("vee ell ay en") into a
candidate string ("VLAN"), which is accepted ONLY on exact match against an
acronym already in the domain dictionary. Never fuzzy: no scorer call, no
threshold, no similarity estimate - just dict lookups, so the
zero-model-inference property is unchanged.

Priority in the pipeline is acronym_map > phrase_alias > letter_composed >
fuzzy. Hand-verified entries always override anything derived here.

Every composed candidate must pass these gates:
  G1  at least MIN_ACRONYM_LETTERS letters. Runs of 1-2 letter-sound words are
      pervasive in ordinary English ("are you", "I owe"); the only 2-letter
      dictionary acronyms are already hand-mapped or contain a digit.
  G2  the composed string exactly matches a dictionary acronym
      (dictionary.get_acronym_index), returning the canonical casing.
  G3  (REQUIRE_UNAMBIGUOUS) at least one letter was heard via an unambiguous
      variant - one that is essentially never an ordinary word ("ess", "dee",
      "em"). Variants like "see", "are", "you", "owe" are also common words, so
      a run made only of them ("I owe a" -> IOA) is far more likely to be
      plain speech than spelling.
  G4  the term is not in COMPOSITION_EXCLUDED. These were hand-fixed through
      acronym_map because they collide with common English words; none are in
      domain_terms.json today, so G2 already blocks them, but that protection
      would vanish silently if one were added to the file. G4 makes it explicit.
  G5  punctuation (only when the caller supplies `gaps`, see below). The
      tokenizer drops punctuation, so "Em, Dee, are you coming" would reach the
      composer as "em dee are you" and spell MDR. A run therefore ends at any
      sentence-terminal mark (HARD_BREAK_CHARS), and the separators between
      consecutive letters must be UNIFORM: "ess, ess, dee", "vee-ell-ay-en" and
      "vee ell ay en" each use one separator throughout and compose, while
      "see Em, Dee, be" (space, comma, comma) does not. Uniformity, rather than
      "any comma breaks the run", is deliberate: Whisper may well write a
      spelled acronym with commas between the letters, and a blanket comma rule
      would break those. Known residual: a uniformly comma-separated run of
      names ("Em, Dee, are") is structurally identical to a comma-spelled
      acronym, so no punctuation rule can tell them apart.

Note for pipeline.py: many letter-sound variants ("a", "be", "i", "you", "are")
are in FUZZY_STOPWORDS. That guard exists to protect FUZZY scoring on short
strings and does not apply here, because composition does no fuzzy scoring.
The composed-match branch runs before the stopword check on purpose.
"""
from collections.abc import Iterator

from app.dictionary import get_acronym_index

MIN_ACRONYM_LETTERS = 3
# Longest reachable acronym is 5 letters (VXLAN, BSSID); one extra token allows
# for a "double you" spelling of W. Independent of pipeline max_window so the
# fuzzy loop is not widened (each extra window level costs ~31 ms there).
MAX_LETTER_WINDOW = 6
REQUIRE_UNAMBIGUOUS = True  # G3; kept as a module constant so it can be swept for ablation
COMPOSITION_EXCLUDED = frozenset({"SATA", "STP", "DDoS", "IDS", "PAT", "HSTS", "IPS"})
_EXCLUDED_FOLDED = frozenset(t.casefold() for t in COMPOSITION_EXCLUDED)
# G5: a gap containing any of these ends the run outright. Commas and hyphens are
# deliberately NOT here - they are legitimate letter separators when uniform.
HARD_BREAK_CHARS = frozenset(".!?;:…")

# (letter, is_unambiguous, spoken variants). Variants are included only where
# the pronunciation is genuinely common or a known STT spelling of the same
# sound - not padded speculatively. "Unambiguous" means the variant is
# essentially never an ordinary standalone word; ambiguous ones (see, are, you,
# tea, owe, why...) are.
_LETTER_SPEC: list[tuple[str, bool, tuple[str, ...]]] = [
    ("A", False, ("ay", "a", "eh")),
    ("B", False, ("bee", "be")),
    ("C", False, ("see", "sea", "cee")),
    ("D", True, ("dee", "di")),
    ("E", False, ("ee", "e")),
    ("F", True, ("ef", "eff")),
    ("G", False, ("gee", "jee")),
    ("H", True, ("aitch", "haitch", "aich")),
    ("I", False, ("eye", "i", "aye")),
    ("J", True, ("jay", "jae")),
    ("K", True, ("kay", "kae")),
    ("L", True, ("el", "ell")),
    ("M", True, ("em", "emm")),
    ("N", True, ("en", "enn")),
    ("O", False, ("oh", "owe", "o")),
    ("P", False, ("pea", "pee", "pe")),
    ("Q", False, ("queue", "cue", "kyu")),
    ("R", False, ("are", "ar", "arr")),
    ("S", True, ("ess", "es")),
    ("T", False, ("tea", "tee", "te")),
    ("U", False, ("you", "ewe", "u", "yu")),
    ("V", True, ("vee", "ve")),
    ("W", True, ("double you", "double u", "dub", "dubya")),  # only multi-token entries
    ("X", True, ("ex", "ecks", "eks")),
    ("Y", False, ("why", "wye")),
    ("Z", True, ("zee", "zed")),
]

# normalized spoken token(s) -> (LETTER, is_unambiguous)
LETTER_SOUNDS: dict[str, tuple[str, bool]] = {
    sound: (letter, unambiguous)
    for letter, unambiguous, sounds in _LETTER_SPEC
    for sound in sounds
}
# A duplicated spoken key would silently map one sound to two letters.
assert len(LETTER_SOUNDS) == sum(len(sounds) for _, _, sounds in _LETTER_SPEC), (
    "duplicate spoken variant in _LETTER_SPEC"
)

_MAX_KEY_TOKENS = max(len(sound.split()) for sound in LETTER_SOUNDS)


def _is_hard_break(gap: str) -> bool:
    return any(ch in HARD_BREAK_CHARS for ch in gap)


def _iter_letter_prefixes(
    tokens: list[str],
    gaps: list[str] | None,
) -> Iterator[tuple[list[str], bool, int]]:
    """Walk forward from tokens[0], yielding (letters, saw_unambiguous,
    tokens_consumed) after each letter-sound consumed. Ends at the first token
    that is not a letter-sound, at a hard-break gap, at a gap that differs from
    the run's first inter-letter separator (G5), or at MAX_LETTER_WINDOW tokens.
    `gaps[k]` is the separator between tokens[k] and tokens[k+1] (see
    tokenizer.separators); None means no punctuation information. The gap inside
    a multi-token sound ("double you") only has to not be a hard break - what
    must be uniform is the separator BETWEEN letters."""
    window = tokens[:MAX_LETTER_WINDOW]
    if gaps is not None and len(gaps) < len(window) - 1:
        raise ValueError(f"gaps must have len(tokens) - 1 entries, got {len(gaps)} for {len(window)} tokens")
    letters: list[str] = []
    saw_unambiguous = False
    separator: str | None = None  # the run's first inter-letter gap
    pos = 0
    while pos < len(window):
        if pos > 0 and gaps is not None:
            gap = gaps[pos - 1]
            if _is_hard_break(gap):
                return
            if separator is None:
                separator = gap
            elif gap != separator:
                return
        # Longest table key first, so "double you" wins over anything shorter.
        entry = None
        for key_len in range(min(_MAX_KEY_TOKENS, len(window) - pos), 0, -1):
            if gaps is not None and any(_is_hard_break(g) for g in gaps[pos : pos + key_len - 1]):
                continue
            entry = LETTER_SOUNDS.get(" ".join(window[pos : pos + key_len]))
            if entry is not None:
                break
        if entry is None:
            return  # next token is not a letter-sound: the run ends here
        letter, unambiguous = entry
        letters.append(letter)
        saw_unambiguous = saw_unambiguous or unambiguous
        pos += key_len
        yield letters, saw_unambiguous, pos


def letter_run_length(tokens: list[str], gaps: list[str] | None = None) -> int:
    """Number of tokens in the run of letter-sounds starting at tokens[0], under
    the same walk (and G5 punctuation rules) the composer uses; 0 if tokens[0] is
    not a letter-sound. No dictionary lookup, no gates G1-G4."""
    consumed = 0
    for _, _, consumed in _iter_letter_prefixes(tokens, gaps):
        pass
    return consumed


def compose_letter_acronym_spans(
    tokens: list[str],
    acronym_index: dict[str, str] | None = None,
    gaps: list[str] | None = None,
) -> list[tuple[str, int]]:
    """Walk forward from tokens[0], composing consecutive letter-sounds, and
    return EVERY (term, tokens_consumed) hit that passes gates G1-G5, shortest
    first. `tokens` must already be normalized (tokenizer.normalize). The walk
    stops at the first token that is not a letter-sound; it never skips over
    one. Returning all hits, not just the longest, lets the pipeline's DP
    choose between a short and a longer exact span. `acronym_index` defaults
    to the dictionary's; tests pass their own to inject terms. `gaps` is the
    optional punctuation signal for G5 (see _iter_letter_prefixes)."""
    index = get_acronym_index() if acronym_index is None else acronym_index
    hits: list[tuple[str, int]] = []
    for letters, saw_unambiguous, consumed in _iter_letter_prefixes(tokens, gaps):
        if len(letters) < MIN_ACRONYM_LETTERS:  # G1
            continue
        if REQUIRE_UNAMBIGUOUS and not saw_unambiguous:  # G3
            continue
        term = index.get("".join(letters).casefold())  # G2: exact match only
        if term is None or term.casefold() in _EXCLUDED_FOLDED:  # G4
            continue
        hits.append((term, consumed))
    return hits


def compose_letter_acronym(
    tokens: list[str],
    acronym_index: dict[str, str] | None = None,
    gaps: list[str] | None = None,
) -> tuple[str, int] | None:
    """Return the longest (term, tokens_consumed) composed from a run of
    letter-sounds starting at tokens[0], or None if no run passes the gates."""
    hits = compose_letter_acronym_spans(tokens, acronym_index, gaps)
    return hits[-1] if hits else None


if __name__ == "__main__":
    # (spoken, gaps) - gaps=None means "no punctuation info"
    samples: list[tuple[str, list[str] | None]] = [
        ("vee ell ay en", None),                       # VLAN
        ("double you ay en", None),                    # WAN via multi-token W
        ("ess cue ell eye", None),                     # SQLi (case-insensitive exact match)
        ("eye owe ay hundred", None),                  # IOA: all-ambiguous, blocked by G3
        ("ess tee pee", None),                         # STP: excluded / not in dictionary
        ("see pea you", None),                         # CPU: all-ambiguous, blocked by G3
        ("ess ess", None),                             # too short, blocked by G1
        ("vee ell banana ay en", None),                # run ends at a non-letter token
        ("vee ell ay en", [",", ",", ","]),            # "vee, ell, ay, en": uniform commas still compose
        ("see em dee be", ["", ",", ","]),             # "see Em, Dee, be": mixed separators end the run (G5)
        ("vee ell ay en", [".", ".", "."]),            # sentence-terminal marks end the run (G5)
    ]
    for spoken, gaps in samples:
        note = "" if gaps is None else f"   gaps={gaps}"
        print(f"{spoken!r:26} -> {compose_letter_acronym(spoken.split(), gaps=gaps)}{note}")
