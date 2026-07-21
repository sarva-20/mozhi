import jellyfish
from rapidfuzz import fuzz
from core.phonetic import phonetic_match_score


# Default weights for single-word terms
PHONETIC_WEIGHT = 0.5
JARO_WINKLER_WEIGHT = 0.3
TOKEN_SORT_WEIGHT = 0.2

# Default weights for multi-word terms
MULTIWORD_PHONETIC_WEIGHT = 0.1
MULTIWORD_JARO_WINKLER_WEIGHT = 0.0
MULTIWORD_TOKEN_SORT_WEIGHT = 0.9

# Minimum threshold to consider a match
MATCH_THRESHOLD = 0.75


def jaro_winkler_score(word: str, term: str) -> float:
    """
    Measures string similarity with prefix bias.
    Higher score for words that share the same starting characters.

    Example:
        "magistrate" vs "maji-strayt" → high score (similar prefix)
    """
    word = word.lower().strip().replace("-", " ")
    term = term.lower().strip()
    return jellyfish.jaro_winkler_similarity(word, term)


def token_sort_score(word: str, term: str) -> float:
    """
    Compares words regardless of token ordering.
    Useful when STT drops or reorders words in a phrase.

    Example:
        "court high" vs "high court" → 1.0
        
    Returns float 0.0 to 1.0 (rapidfuzz returns 0-100, we normalize).
    """
    word = word.lower().strip()
    term = term.lower().strip()
    return fuzz.token_sort_ratio(word, term) / 100.0


def compute_score(
    incoming: str,
    term: str,
    phonetic_index: dict,
    weights: dict = None
) -> float:
    """
    Combines all three signals into one confidence score.
    Dynamically adjusts weights based on whether the term
    is single-word or multi-word.

    Single-word  → phonetic dominates (0.5 / 0.3 / 0.2)
    Multi-word   → token sort dominates (0.2 / 0.2 / 0.6)

    Args:
        incoming: raw STT word or phrase
        term: ontology term to compare against
        phonetic_index: precomputed index from build_phonetic_index()
        weights: optional dict to override all defaults
            e.g. {"phonetic": 0.6, "jaro": 0.2, "token": 0.2}

    Returns:
        float between 0.0 and 1.0
    """
    is_multiword = len(term.strip().split()) > 1

    if weights:
        # User provided explicit weights — respect them
        w_phonetic = weights.get("phonetic", PHONETIC_WEIGHT)
        w_jaro = weights.get("jaro", JARO_WINKLER_WEIGHT)
        w_token = weights.get("token", TOKEN_SORT_WEIGHT)
    elif is_multiword:
        # Multi-word term — token sort dominates
        w_phonetic = MULTIWORD_PHONETIC_WEIGHT
        w_jaro = MULTIWORD_JARO_WINKLER_WEIGHT
        w_token = MULTIWORD_TOKEN_SORT_WEIGHT
    else:
        # Single-word term — phonetic dominates
        w_phonetic = PHONETIC_WEIGHT
        w_jaro = JARO_WINKLER_WEIGHT
        w_token = TOKEN_SORT_WEIGHT

    # Compute individual signals
    p_score = phonetic_match_score(incoming, term, phonetic_index)
    j_score = jaro_winkler_score(incoming, term)
    t_score = token_sort_score(incoming, term)

    # Weighted combination
    final = (w_phonetic * p_score) + (w_jaro * j_score) + (w_token * t_score)

    return round(final, 4)


def is_match(score: float, threshold: float = MATCH_THRESHOLD) -> bool:
    """
    Simple threshold gate.
    Sliding window uses this to decide whether to replace a token.
    """
    return score >= threshold