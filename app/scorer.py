"""
Weighted phonetic/lexical scorer.

Combines three signals, each catching a different class of STT error:

  - Metaphone similarity: catches phonetic substitutions (consonant-level
    sound-alikes), e.g. "sea pu" -> "CPU". We compare Metaphone codes with
    Jaro-Winkler rather than requiring exact code equality, because partial
    phonetic overlap (e.g. one extra/missing sound) should still count.
  - Jaro-Winkler on raw surface strings: catches close spelling/sound
    edit-distance errors and rewards matching prefixes (useful for acronyms
    and compound terms).
  - Token sort ratio: catches word-order scrambling in multi-word terms,
    e.g. "address IP" said/heard out of order relative to "IP address".

No model inference occurs anywhere in this module: every function is a
deterministic string/phonetic-code computation.
"""
import jellyfish
from rapidfuzz import fuzz

# Tunable weights - must sum to 1.0. Exposed at module level (not hidden
# inside a function) so they can be swept during benchmarking/ablation for
# the paper's "why these weights" section.
WEIGHT_METAPHONE = 0.40
WEIGHT_JARO_WINKLER = 0.35
WEIGHT_TOKEN_SORT = 0.25

assert abs((WEIGHT_METAPHONE + WEIGHT_JARO_WINKLER + WEIGHT_TOKEN_SORT) - 1.0) < 1e-9


def _sorted_metaphone_code(s: str) -> str:
    """Per-word Metaphone codes, sorted, then joined. Sorting makes this
    order-invariant across words - matching how token_sort_ratio already
    normalizes word order for the surface-string signal. Without this,
    a genuine STT reordering (e.g. 'address IP' heard for 'IP address')
    scores poorly on the phonetic signal even though the sounds present
    are identical, just reordered."""
    codes = [jellyfish.metaphone(w) for w in s.split()]
    return " ".join(sorted(c for c in codes if c))


def metaphone_score(candidate: str, term: str) -> float:
    """Phonetic similarity via Jaro-Winkler distance between (sorted,
    per-word) Metaphone codes. Returns 0.0 if either string produces no
    non-empty Metaphone codes (e.g. pure digits/symbols)."""
    c_code = _sorted_metaphone_code(candidate)
    t_code = _sorted_metaphone_code(term)
    if not c_code or not t_code:
        return 0.0
    return jellyfish.jaro_winkler_similarity(c_code, t_code)


def surface_score(candidate: str, term: str) -> float:
    """Jaro-Winkler similarity on the raw (lowercased) surface strings."""
    return jellyfish.jaro_winkler_similarity(candidate.lower(), term.lower())


def token_sort_score(candidate: str, term: str) -> float:
    """Token sort ratio, normalized to 0.0-1.0 (rapidfuzz returns 0-100)."""
    return fuzz.token_sort_ratio(candidate.lower(), term.lower()) / 100.0


def _score_once(candidate: str, term: str) -> dict:
    m = metaphone_score(candidate, term)
    s = surface_score(candidate, term)
    t = token_sort_score(candidate, term)
    total = WEIGHT_METAPHONE * m + WEIGHT_JARO_WINKLER * s + WEIGHT_TOKEN_SORT * t
    return {"metaphone": m, "surface": s, "token_sort": t, "total": total}


def combined_score(candidate: str, term: str) -> dict:
    """Returns the individual component scores plus the weighted total, so
    callers/tests/benchmarks can inspect why a match did or didn't fire.

    When the term is a single fused word (e.g. "motherboard") but the STT
    candidate is space-split across tokens (e.g. "mother board"), every
    component score is penalized by the literal space character even
    though the sounds are identical - badly enough that a truncated
    single-token candidate ("mother") can outscore the correctly-bounded
    two-token one. To catch this, we also score the space-removed
    candidate against such terms and keep whichever score is higher."""
    base = _score_once(candidate, term)
    if " " in candidate and " " not in term:
        fused = _score_once(candidate.replace(" ", ""), term)
        if fused["total"] > base["total"]:
            return fused
    return base


if __name__ == "__main__":
    examples = [
        ("sea pu", "CPU"),
        ("ram", "RAM"),
        ("address IP", "IP address"),
        ("fire wall", "firewall"),
        ("kubernetees", "Kubernetes"),
        ("banana", "SSH"),
    ]
    for candidate, term in examples:
        print(candidate, "->", term, combined_score(candidate, term))
