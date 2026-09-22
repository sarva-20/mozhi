"""
Ordinary-English-inflection guard for fuzzy matching.

Found via a 32-file real-recording accuracy run (n=2 hand-picked audio never
exercised this): the fuzzy scorer (Metaphone + Jaro-Winkler + token-sort)
treats a correctly-transcribed, ordinarily-inflected form of a domain term's
OWN word(s) as if it were a phonetic corruption of a *different* concept, and
"corrects" it back to the bare term - discarding grammatical information the
speaker said correctly and actively raising WER. Examples pulled straight
from that run: "client's" -> "client", "crashes"/"crashed" -> "crash",
"containerized" -> "container", "ticketing" -> "ticket", "backed up" ->
"backup".

Note on scope: an earlier version of this module also stripped a bare
trailing "s" (which would additionally catch "credential" -> "credentials"),
but the full-dictionary coverage test showed that rule cost more than it was
worth - see the comment on _INFLECTIONAL_SUFFIXES below - so it was removed.
None of the confirmed regressions this module fixes need it.

This module answers one narrow question: is `candidate` just `term` (or
`term` just `candidate`), plus one ordinary English inflectional ending? If
so, this is NOT a phonetic corruption of a different word - it is the same
word, said with different grammar Mozhi's dictionary does not carry a
separate entry for - and the match must be refused so the sentence is left
as the speaker actually said it.

Deliberately narrow, on purpose:
  - INFLECTIONAL suffixes only ('s, plural/3rd-person -s, -ed, -ing, plus the
    -ize/-ise verbalizing family - see below), never other DERIVATIONAL ones
    (-er, -or, -tion...). "print" -> "printer" is a *different word* (a verb
    vs. the machine that does it), not "print" with different grammar - that
    stays a same-mechanism problem as any other sound-alike collision
    ("catch"/"cache", "idea"/"IDE") and is handled case by case in
    NEVER_MATCH_PAIRS in app/pipeline.py, not here.
  - The -ize/-ise family (containerize, virtualize, serialize, sanitize,
    tokenize, ...) is included even though it is derivational, not
    inflectional in the strict grammatical sense: it is a deliberate,
    documented scope call, not an oversight. It is extremely common and
    *productive* in exactly this tech-support/DevOps/security domain, so the
    next batch of volunteer data is likely to hit it again the same way
    "containerized" -> "container" did in this one.
  - Same word count only, with ONE exception (_fuses_to) for a phrasal verb
    fusing into a closed compound noun ("backed up" -> "backup"), the one
    multi-word shape the accuracy run surfaced. It fires only when stripping
    an inflection from the FIRST word was actually necessary, which "fire"
    (in "fire wall" -> "firewall") or "mother" (in "mother board" ->
    "motherboard") never satisfies - so it cannot interfere with those
    genuine STT segmentation fixes, which need zero stripping to already
    equal the term after de-spacing.
  - Never fires when candidate already equals term case-insensitively (e.g.
    "ram" vs "RAM"): that is not a correction being wrongly applied, it is
    the ordinary case-fix path, and must stay open.
  - Never touches word reordering ("address IP" -> "IP address") or any
    word-count change outside the one fuses_to shape, so it cannot interfere
    with those either.
"""
from __future__ import annotations

# Checked longest-first (by length, not list order) so a more specific ending
# is preferred over a shorter one it would otherwise be swallowed by (e.g.
# "containerized" must match "ized", not just trailing "ed").
#
# Deliberately NO bare trailing "s". A plain "strip one trailing s" rule
# looks appealing (it would also catch "credential"/"credentials"), but the
# full-dictionary coverage test showed it costs more than it's worth: many
# ordinary words end in "s" as part of their spelling, not as a plural
# marker ("address", "access", "loss", "continuous", "CVSS", "analytics"),
# and a genuine STT corruption that drops one letter ("addres" from
# "address", "cvs" from "CVSS", "acces" from "access") then coincidentally
# matches the stripped form and gets wrongly refused - the opposite failure
# mode from the one this module exists to fix. None of the 10 confirmed
# regressions this module was built for need it (verified below); it is
# left out on that evidence, not included speculatively for a case that
# only showed up once, incidentally, and was never a required fix.
_INFLECTIONAL_SUFFIXES = (
    "izing", "ising",              # containerizing, virtualizing
    "ized", "ised", "izes", "ises",  # containerized, serialized, tokenizes
    "'s", "s'", "ing", "es", "ed",  # possessive, gerund, plural/past
    "ize", "ise",                   # bare present tense (containerize)
)
_SUFFIXES_BY_LENGTH = tuple(sorted(_INFLECTIONAL_SUFFIXES, key=len, reverse=True))

_MIN_STEM_LEN = 3  # refuse to strip down to a stem too short to be a real word ("is" -> "i")


def strip_inflection(word: str) -> str:
    """Remove ONE trailing ordinary-English inflectional ending, if present.
    Returns `word` casefolded and otherwise unchanged if no ending applies or
    the remaining stem would be shorter than _MIN_STEM_LEN."""
    lower = word.casefold()
    for suffix in _SUFFIXES_BY_LENGTH:
        if lower.endswith(suffix) and len(lower) - len(suffix) >= _MIN_STEM_LEN:
            return lower[: -len(suffix)]
    return lower


def _same_word(a: str, b: str) -> bool:
    """a and b are the same underlying word: identical, or one is the other
    plus a single ordinary inflectional ending."""
    a, b = a.casefold(), b.casefold()
    return a == b or strip_inflection(a) == b or a == strip_inflection(b)


def _fuses_to(word1: str, word2: str, term: str) -> bool:
    """word1 word2 (two candidate tokens) fuses to `term` (one term word)
    because word1 needed an inflectional ending stripped first - the
    phrasal-verb-to-compound-noun pattern ("backed" + "up" -> "backup").
    Requires the strip to be non-trivial, so a plain concatenation match
    ("fire" + "wall" -> "firewall", no stripping needed at all) is left
    alone: that is a genuine STT segmentation error the fuzzy scorer is
    meant to fix, not an inflection to guard against."""
    stripped = strip_inflection(word1)
    if stripped == word1.casefold():
        return False
    return stripped + word2.casefold() == term.casefold()


def is_ordinary_inflection(candidate: str, term: str) -> bool:
    """True if `candidate` is not a phonetic corruption of `term` at all, but
    an ordinarily-inflected form of the SAME word(s) the term already names -
    i.e. accepting this "correction" would discard grammatical information
    the speaker said correctly, not fix an STT error. Used to refuse a fuzzy
    match candidate, not to accept one: it is a guard, not a matcher."""
    if candidate.casefold() == term.casefold():
        return False  # already the term verbatim (modulo case): let the case-fix path through
    cand_words, term_words = candidate.split(), term.split()
    if len(cand_words) == len(term_words):
        return all(_same_word(c, t) for c, t in zip(cand_words, term_words))
    if len(cand_words) == 2 and len(term_words) == 1:
        return _fuses_to(cand_words[0], cand_words[1], term_words[0])
    return False


if __name__ == "__main__":
    examples = [
        # the 10 confirmed regressions this module fixes (all True)
        ("client's", "client"), ("crashes", "crash"), ("crashed", "crash"),
        ("containerized", "container"), ("ticketing", "ticket"), ("backed up", "backup"),
        # must stay False: case-only, genuine STT fixes, or a different word/root entirely
        ("ram", "RAM"), ("fire wall", "firewall"), ("mother board", "motherboard"),
        ("catch", "cache"), ("read", "RAID"), ("land", "LAN"), ("print", "printer"),
        # bare trailing "s" is deliberately NOT stripped: "credential"/"credentials" stays
        # False (see the module docstring), and so does a genuine corruption of a word
        # that just happens to end in "s" ("addres" from "address", "cvs" from "CVSS")
        ("credential", "credentials"), ("addres", "address"), ("cvs", "CVSS"),
    ]
    for candidate, term in examples:
        print(f"{candidate!r:16} vs {term!r:14} -> {is_ordinary_inflection(candidate, term)}")
