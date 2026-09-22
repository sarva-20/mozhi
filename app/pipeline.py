"""
Core Mozhi correction pipeline.

    tokenize -> exhaustive sliding-window matching across all start
                positions -> acronym HashMap fallback -> weighted fuzzy
                scoring -> optimal non-overlapping replacement selection

Deliberately has zero model inference anywhere: every step is a
deterministic tokenization, dict lookup, or string/phonetic-code
computation (Metaphone, Jaro-Winkler, token sort ratio). This module is
also the thing to `profile` for Review 2's "prove no-model-inference"
requirement - there is nothing here that loads weights, calls an API, or
touches a GPU, and that's verifiable just by reading it top to bottom.

Matching strategy: for every (start position, window length) pair up to
max_window, we score the best acronym/fuzzy candidate for that exact span.
A single greedy left-to-right scan then breaks on STT reorderings, because
once a span is consumed the scan jumps past it and never considers a
better-aligned span starting one token later (e.g. "the address IP" ->
consuming "the address" as a unit before "address IP" is ever tried as its
own span, one token over). Instead we run a DP over token positions that
chooses the set of NON-OVERLAPPING spans maximizing total match score
across the whole sentence - equivalent to weighted interval scheduling.
This is what "exhaustive sliding-window matching across all start
positions" refers to in the paper: every start position is a candidate,
and the final selection is globally optimal given the per-span scores,
not just locally greedy.
"""
import time
from dataclasses import dataclass, field

from app.tokenizer import tokenize, normalize, separators, Token
from app.dictionary import get_term_strings, get_max_term_tokens
from app.scorer import combined_score
from app.acronym_map import ACRONYM_PHONETIC_MAP, lookup_acronym, lookup_phrase_alias
from app.letter_composer import MAX_LETTER_WINDOW, compose_letter_acronym_spans
from app.morphology import is_ordinary_inflection

# History:
#   0.72 -> 0.80 after hand-written test cases showed ordinary false positives.
#   0.80 -> 0.88 after two more real false positives surfaced: "lad, then" ->
#     "latency" at 0.822 and "restart" -> "restore" at 0.873. Raising the
#     threshold was the only lever available at the time - neither
#     NEVER_MATCH_PAIRS nor app.morphology existed yet.
#   0.88 -> 0.83 -> back to 0.88: tried once NEVER_MATCH_PAIRS/app.morphology
#     existed, on the theory that both original false positives were now
#     individually guardable (restart/restore was, and got a permanent
#     NEVER_MATCH_PAIRS entry below - it stays regardless of threshold, see
#     that entry's comment). But the narrow two-case check does not generalize:
#     a systematic sweep of every domain term against a 10k-word common-English
#     list found 257 NEW word/term pairs clear 0.83 that stay below 0.88 (vs
#     129 already above 0.88) - "app"/API, "access"/"access port", "pass"/
#     "passkey", "export"/exploit, "gate"/gateway, "service"/"service desk",
#     "IP"/IPv4, "is open"/ISP, "escalate"/escalation among many plausible
#     ones. Guarding all of them individually would mean blocking pairs never
#     verified against real speech, the opposite of how every other guard in
#     this file was justified. 0.88 stays the threshold; NEVER_MATCH_PAIRS and
#     app.morphology handle specific, evidence-backed collisions on top of it
#     rather than replacing it. tests/test_threshold_investigation.py has the
#     full investigation and the sweep.
DEFAULT_THRESHOLD = 0.88

# Jaro-Winkler saturates high on short strings almost regardless of content
# (common prefixes dominate), so short acronyms (2-3 letters) are prone to
# false-positiving against common short English function words - e.g. "is"
# scored 0.88 against "ISP" purely from the shared "is" prefix. Rather than
# raising the global threshold (which would hurt genuine longer-term
# matches), single-token candidates that are common stopwords are excluded
# from fuzzy scoring outright: they are overwhelmingly likely to be real
# words, not STT phonetic errors, in ordinary sentences. This does not
# affect the acronym HashMap fallback (exact spelled-out-letter lookups
# never collide with stopwords).
FUZZY_STOPWORDS = frozenset(
    """a an the is are was were be been being to of in on at by for with
    from as it he she we you they i and or but if then so than that this
    these those do does did not no yes up down out off over under again
    my your his her its our their""".split()
)

# Some common words are valid domain terms but must never be treated as
# phonetic variants of another common word. These pairs are structurally
# ambiguous to the scorer but not credible STT corrections.
#
# Each pair is (candidate word, casefolded domain term): a fuzzy window that
# CONTAINS the word is never matched to that term. It applies to fuzzy scoring
# only; the exact paths (acronym_map, phrase_alias, letter_composed) are
# untouched, so "eye dee ee" still resolves to IDE. IDE is a three-letter term
# whose Metaphone code ("IT") is shared by ordinary words, so it scores above the
# threshold against them: "idea" 0.944, "I die" 0.941 (the two-word window scores
# high because per-word Metaphone codes are sorted and joined), "id" 0.919 and
# "idle" 0.905. Entries are per word because the block is per word. "I dye" (0.905)
# is deliberately not listed: not realistic tech-support vocabulary.
#
# "side" -> SSID (0.885) is the same shape: SSID's Metaphone code is a plain
# suffix match against "side" alone, unrelated to any specific phrase. It
# surfaced from the full-dictionary coverage test as the most frequent bad
# replacement (11 occurrences) and as the one contributor to the "canonical
# form altered" defects (see the DP fix below) that this fix alone doesn't
# reach: "server-side request forgery" never scores a full 1.0 exact match
# (its canonical form uses a hyphen, the spoken candidate a space), so no
# exactness-based DP preference can save it - only removing the "side"->SSID
# candidate does.
#
# The next three surfaced from a 32-file real-recording accuracy run and are
# NOT inflections of their term (that class is handled generally by
# app.morphology.is_ordinary_inflection below, not by listing pairs here) -
# each is a genuinely different word/acronym that just sounds like the term:
# "print" -> "printer" 0.922 (print is a real, different word - a verb - not
# printer missing "-er"), "IPS" -> "IPsec" 0.882 (a different acronym,
# Intrusion Prevention System, not in this dictionary), and "to" ->
# "access port" 0.909 for the whole phrase "access to" (the single word "to"
# alone scores 0.04; the block is structural - it fires on any window
# containing "to" scored against this term, same as "drive" against
# "driver"). Blocking only "access port" left "access to" -> "access point"
# (0.895, the dictionary's only other "access "-prefixed term) exposed as
# the new best score once "access port" was removed from consideration - the
# metaphone signal barely discriminates past the shared "access" word, so
# BOTH are blocked, not just the one the accuracy run happened to surface
# first.
#
# "catch" -> "cache" (0.913) surfaced too (3 occurrences, 0 true positives in
# that sample) but is deliberately NOT added: tests/batch_cases.py has an
# existing, deliberately-authored true positive for it ("clear the browser
# catch" -> "...cache"), so this is genuinely context-dependent, irreducible
# ambiguity (the same shape as "read" -> "RAID", also never blocked, also a
# confirmed true positive in test_audio2), not a bug this mechanism can fix.
# Blocking it would trade one accuracy run's false positives for a
# different, already-confirmed sample's false negative.
#
# "restart" -> "restore" (0.873) is one of the two original false positives
# that justified raising DEFAULT_THRESHOLD to 0.88 in the first place (see
# the comment above it). An investigation into lowering the threshold now
# that targeted guards exist found it was the sole blocker of the two: at
# every tested value below 0.873 it reopened completely unguarded, since
# raising the threshold - not a targeted guard - was originally what stopped
# it. Added here so it is guarded independently of the threshold value, same
# as every other entry in this set; DEFAULT_THRESHOLD ultimately stayed at
# 0.88 anyway (the broader threshold-lowering attempt was not safe - see the
# comment above DEFAULT_THRESHOLD), but this entry is harmless and correct
# regardless, and gives "restart"/"restore" the same defense-in-depth every
# other collision here has instead of relying on threshold margin alone.
# "restart" is not a dictionary term.
NEVER_MATCH_PAIRS = frozenset({
    ("drive", "driver"),
    ("idea", "ide"),
    ("die", "ide"),
    ("idle", "ide"),
    ("id", "ide"),
    ("side", "ssid"),
    ("print", "printer"),
    ("ips", "ipsec"),
    ("to", "access port"),
    ("to", "access point"),
    ("restart", "restore"),
})


def _max_acronym_map_tokens() -> int:
    if not ACRONYM_PHONETIC_MAP:
        return 1
    return max(len(k.split()) for k in ACRONYM_PHONETIC_MAP)


@dataclass
class Match:
    original: str          # original surface text of the matched span
    replacement: str       # canonical domain term substituted in
    start: int              # char offset in source text
    end: int                # char offset (exclusive) in source text
    method: str             # "acronym_map" | "phrase_alias" | "letter_composed" | "fuzzy"
    score: float
    score_breakdown: dict = field(default_factory=dict)


@dataclass
class CorrectionResult:
    original_text: str
    corrected_text: str
    matches: list[Match]
    elapsed_ms: float


def _best_fuzzy_match(candidate: str, domain_terms: list[str], threshold: float):
    best_term, best_score, best_breakdown = None, 0.0, None
    for term in domain_terms:
        candidate_words = candidate.casefold().split()
        blocked = any(
            term.casefold() == blocked_term
            and blocked_candidate in candidate_words
            for blocked_candidate, blocked_term in NEVER_MATCH_PAIRS
        )
        # General guard (app/morphology.py): candidate is just this SPECIFIC
        # term, ordinarily inflected ("client's" for "client"), not a
        # phonetic corruption of anything - refuse only against this one
        # term, so scoring against every OTHER term is unaffected.
        if blocked or is_ordinary_inflection(candidate, term):
            continue
        breakdown = combined_score(candidate, term)
        if breakdown["total"] > best_score:
            best_term, best_score, best_breakdown = term, breakdown["total"], breakdown
    if best_term is not None and best_score >= threshold:
        return best_term, best_score, best_breakdown
    return None, 0.0, None


def correct_text(
    text: str,
    domain_terms: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_window: int | None = None,
) -> CorrectionResult:
    """Run the full sliding-window correction pipeline over `text`."""
    start_time = time.perf_counter()

    domain_terms = domain_terms if domain_terms is not None else get_term_strings()
    max_window = (
        max_window
        if max_window is not None
        else max(get_max_term_tokens(), _max_acronym_map_tokens())
    )

    tokens: list[Token] = tokenize(text)
    n = len(tokens)

    # Letter-composition pre-pass: exact-match spans built from runs of spoken
    # letter-sounds. Uses its own MAX_LETTER_WINDOW rather than widening
    # max_window, which would also widen the (expensive) fuzzy loop below.
    # tokenize() drops punctuation, so the composer is also handed the separator
    # between each pair of tokens: a sentence-terminal mark or a change of
    # separator ends a run (G5), so "Go see Em, Dee, be nice" cannot spell CMDB.
    normalized = [normalize(t.text) for t in tokens]
    gap_after = separators(text, tokens)
    composed: dict[tuple[int, int], str] = {}
    for i in range(n):
        spans = compose_letter_acronym_spans(
            normalized[i : i + MAX_LETTER_WINDOW],
            gaps=gap_after[i : i + MAX_LETTER_WINDOW - 1],
        )
        for term, consumed in spans:
            composed[(i, consumed)] = term

    # Step 1: score every (start position, window length) span independently.
    # candidates[i][window_len] = (replacement, method, score, breakdown) or None
    candidates: list[dict[int, tuple]] = [dict() for _ in range(n)]
    for i in range(n):
        for window_len in range(1, min(max_window, n - i) + 1):
            window_tokens = tokens[i : i + window_len]
            candidate_str = " ".join(normalize(t.text) for t in window_tokens)

            acr = lookup_acronym(candidate_str)
            if acr:
                candidates[i][window_len] = (acr, "acronym_map", 1.0, {})
                continue

            alias = lookup_phrase_alias(candidate_str)
            if alias:
                candidates[i][window_len] = (alias, "phrase_alias", 1.0, {})
                continue

            # Deliberately BEFORE the FUZZY_STOPWORDS check: many letter-sound
            # variants ("a", "be", "i", "you", "are") are stopwords, but that
            # guard only protects fuzzy scoring, which composition never does.
            composed_term = composed.get((i, window_len))
            if composed_term:
                candidates[i][window_len] = (composed_term, "letter_composed", 1.0, {})
                continue

            window_words = candidate_str.split()
            if window_words and all(word in FUZZY_STOPWORDS for word in window_words):
                continue

            term, score, breakdown = _best_fuzzy_match(candidate_str, domain_terms, threshold)
            if term:
                candidates[i][window_len] = (term, "fuzzy", score, breakdown)

    # Composed spans wider than max_window are never visited by the loop above.
    # The length guard keeps this from overwriting an acronym_map / phrase_alias
    # result that already won a span the loop did visit.
    for (i, window_len), composed_term in composed.items():
        if window_len > max_window:
            candidates[i][window_len] = (composed_term, "letter_composed", 1.0, {})

    # Step 2: DP over token positions (weighted interval scheduling) to pick
    # the set of non-overlapping spans maximizing total match score.
    #
    # The objective is lexicographic: (exact_tokens_covered, score_sum), not
    # score_sum alone. Score is per-MATCH, not per-token, so a 1-token exact
    # match and a 5-token exact match both contribute 1.0 - fragmenting a
    # correctly-spoken span can therefore only ever raise the raw score sum,
    # never lower it, whenever some sub-word of a multi-word term is ALSO a
    # standalone dictionary term. E.g. for "SQL injection" spoken correctly:
    # {"SQL"->SQL (1.0)} + {"injection"->"SQL injection" (0.887, a genuine
    # partial-phrase match that exists to recover a dropped leading word)}
    # sums to 1.887, beating the single correct exact match at 1.0 - and
    # splices back to the literal duplicate "SQL SQL injection". The
    # full-dictionary coverage test (scripts/dictionary_coverage.py) found six
    # such cases, all sharing this shape.
    #
    # exact_tokens_covered counts tokens under a match that is exact by
    # construction (acronym_map / phrase_alias / letter_composed) or a fuzzy
    # match whose score is 1.0 - i.e. the candidate string is verbatim the
    # term, not a heuristic guess. Comparing this first means a single span
    # that verbatim covers N tokens can never lose to a fragmentation that
    # verifies fewer than N of those tokens, regardless of the fragmentation's
    # raw score sum - which is exactly what fixes the six cases above,
    # because in every one only ONE side of the split is genuinely exact (the
    # other is a <1.0 partial-phrase match, which must stay possible so a
    # genuinely dropped word can still be recovered). This generalizes the
    # previous exact_match_tie rule (which only broke a literal numeric tie)
    # to the general case where the fragmented total is strictly higher.
    #
    # Not handled here: "server-side request forgery" never scores a full 1.0
    # (hyphen in the term vs. space in speech), so this rule can't prefer it
    # over "server"(1.0)+"side"->SSID(0.885); that one is fixed by blocking
    # "side"->SSID directly in NEVER_MATCH_PAIRS above. Also out of scope: a
    # term whose full span AND every one of its individual words are all
    # simultaneously exact matches (both sides tie on exact_tokens); the
    # score tie-break would then favor the finer split. This does not occur
    # for any term in the current dictionary; the tie-break exists only to
    # decide between exact spans of different reach, as it already did.
    EXACT_METHODS = ("acronym_map", "phrase_alias", "letter_composed")
    dp_exact = [0] * (n + 1)
    dp_score = [0.0] * (n + 1)
    choice: list[int | None] = [None] * n
    for i in range(n - 1, -1, -1):
        best_key = (dp_exact[i + 1], dp_score[i + 1])  # skip token i
        best_len = None
        for window_len, (_, method, score, _) in candidates[i].items():
            is_exact = method in EXACT_METHODS or score >= 1.0 - 1e-9
            key = (
                dp_exact[i + window_len] + (window_len if is_exact else 0),
                dp_score[i + window_len] + score,
            )
            better = key > best_key or (
                key == best_key and (best_len is None or window_len > best_len)
            )
            if better:
                best_key = key
                best_len = window_len
        dp_exact[i], dp_score[i] = best_key
        choice[i] = best_len

    # Step 3: reconstruct the chosen matches by walking the DP's choices.
    matches: list[Match] = []
    i = 0
    while i < n:
        window_len = choice[i]
        if window_len is None:
            i += 1
            continue
        replacement, method, score, breakdown = candidates[i][window_len]
        window_tokens = tokens[i : i + window_len]
        matches.append(
            Match(
                original=text[window_tokens[0].start : window_tokens[-1].end],
                replacement=replacement,
                start=window_tokens[0].start,
                end=window_tokens[-1].end,
                method=method,
                score=score,
                score_breakdown=breakdown,
            )
        )
        i += window_len

    # Rebuild corrected text by splicing replacements into the original
    # string, right-to-left so earlier char offsets stay valid.
    corrected = text
    for m in sorted(matches, key=lambda m: m.start, reverse=True):
        corrected = corrected[: m.start] + m.replacement + corrected[m.end :]

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return CorrectionResult(
        original_text=text,
        corrected_text=corrected,
        matches=sorted(matches, key=lambda m: m.start),
        elapsed_ms=elapsed_ms,
    )


if __name__ == "__main__":
    samples = [
        "please check the sea pu usage before you reboot",
        "can you increase the ram and check the ess ess dee",
        "we need to configure the address IP for the router",
        "the fire wall is blocking the vee pea en connection",
        "restart the kubernetees container and check the aitch dee dee",
    ]
    for s in samples:
        result = correct_text(s)
        print(f"IN:  {s}")
        print(f"OUT: {result.corrected_text}")
        for m in result.matches:
            print(f"     - '{m.original}' -> '{m.replacement}' ({m.method}, score={m.score:.2f})")
        print(f"     [{result.elapsed_ms:.3f} ms]")
        print()
