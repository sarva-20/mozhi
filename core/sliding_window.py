from core.stopwords import load_stopwords, is_stopword_only
from core.phonetic import build_phonetic_index
from core.scorer import compute_score, is_match
from config.loader import load_all


# Load config once at module startup — not on every call
_config = load_all()
_ONTOLOGY = _config["ontology"]
_ACRONYMS = _config["acronyms"]
_WEIGHTS = _config["weights"]
_STOPWORDS = load_stopwords()
_PHONETIC_INDEX = build_phonetic_index(_ONTOLOGY)


def correct_transcript(
    transcript: str,
    ontology_terms: list[str] = None,
    acronym_map: dict[str, str] = None,
    max_window: int = 5,
    weights: dict = None,
    stopwords: frozenset = None,
    phonetic_index: dict = None
) -> dict:
    """
    Main correction function. Takes a raw STT transcript and returns
    a corrected version by scanning with a sliding window.

    All arguments are now optional — Mozhi loads from config/
    by default. Pass arguments explicitly to override config
    for testing or custom use cases.

    Args:
        transcript     : raw STT string
        ontology_terms : overrides config/ontology.json terms
        acronym_map    : overrides config/acronyms.json
        max_window     : maximum window size (default 5 tokens)
        weights        : overrides scorer weights
        stopwords      : overrides stopword set
        phonetic_index : overrides precomputed phonetic index

    Returns:
        dict with:
            "original"     : raw input transcript
            "corrected"    : corrected transcript
            "replacements" : list of (original_phrase, replacement) tuples
    """

    # Use config defaults unless overrides are passed
    _terms = ontology_terms if ontology_terms is not None else _ONTOLOGY
    _acronyms = acronym_map if acronym_map is not None else _ACRONYMS
    _weights = weights if weights is not None else _WEIGHTS
    _sw = stopwords if stopwords is not None else _STOPWORDS
    _index = phonetic_index if phonetic_index is not None else (
        _PHONETIC_INDEX if ontology_terms is None
        else build_phonetic_index(ontology_terms)
    )

    # Normalize acronym map keys to lowercase
    _acronyms = {k.lower(): v for k, v in _acronyms.items()}

    # Tokenize transcript on whitespace
    tokens = transcript.strip().split()
    corrected_tokens = []
    replacements = []

    i = 0
    while i < len(tokens):
        matched = False

        for window_size in range(min(max_window, len(tokens) - i), 0, -1):
            window_tokens = tokens[i:i + window_size]
            window_phrase = " ".join(window_tokens).lower()

            # Check 1 — stopword only window
            if is_stopword_only(window_tokens, _sw):
                corrected_tokens.append(tokens[i])
                i += 1
                matched = True
                break

            # Check 2 — acronym HashMap direct lookup
            if window_phrase in _acronyms:
                replacement = _acronyms[window_phrase]
                corrected_tokens.append(replacement)
                replacements.append((window_phrase, replacement))
                i += window_size
                matched = True
                break

            # Check 3 — phonetic + scorer match
            best_score = 0.0
            best_term = None

            for term in _terms:
                term_token_count = len(term.split())
                if term_token_count != window_size:
                    continue

                score = compute_score(
                    window_phrase,
                    term,
                    _index,
                    weights=_weights
                )

                if score > best_score:
                    best_score = score
                    best_term = term

            if best_term and is_match(best_score):
                corrected_tokens.append(best_term)
                replacements.append((window_phrase, best_term))
                i += window_size
                matched = True
                break

        if not matched:
            corrected_tokens.append(tokens[i])
            i += 1

    return {
        "original": transcript,
        "corrected": " ".join(corrected_tokens),
        "replacements": replacements
    }