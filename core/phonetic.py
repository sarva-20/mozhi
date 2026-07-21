import jellyfish
from metaphone import doublemetaphone


def get_phonetic_code(word: str) -> tuple[str, str]:
    """
    Returns Double Metaphone code for a word.
    Strips hyphens before encoding so misheared hyphenated
    words are treated as single phonetic units.
    """
    word = word.lower().strip().replace("-", "")
    primary, secondary = doublemetaphone(word)
    return primary, secondary


def build_phonetic_index(terms: list[str]) -> dict[str, tuple[str, str]]:
    """
    Precomputes phonetic codes for all ontology terms at startup.
    """
    index = {}
    for term in terms:
        cleaned = term.lower().strip()
        if cleaned:
            index[cleaned] = get_phonetic_code(cleaned)
    return index


def _code_similarity(code_a: str, code_b: str) -> float:
    """
    Measures similarity between two phonetic code strings
    using Jaro-Winkler on the codes themselves.
    Returns 0.0 if either code is empty.
    """
    if not code_a or not code_b:
        return 0.0
    return jellyfish.jaro_winkler_similarity(code_a, code_b)


def phonetic_match_score(word: str, term: str, phonetic_index: dict) -> float:
    """
    Compares phonetic similarity between an incoming word
    and a precomputed ontology term.

    Checks all code combinations (primary/secondary) and
    returns the highest similarity score found.

    Returns float between 0.0 and 1.0.
    """
    incoming_primary, incoming_secondary = get_phonetic_code(word)
    term_primary, term_secondary = phonetic_index.get(term, ("", ""))

    if not incoming_primary or not term_primary:
        return 0.0

    scores = [
        _code_similarity(incoming_primary, term_primary),
    ]

    if term_secondary:
        scores.append(_code_similarity(incoming_primary, term_secondary))

    if incoming_secondary:
        scores.append(_code_similarity(incoming_secondary, term_primary))

    if incoming_secondary and term_secondary:
        scores.append(_code_similarity(incoming_secondary, term_secondary))

    return max(scores)