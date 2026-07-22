from core.stopwords import load_stopwords, is_stopword_only
from core.phonetic import build_phonetic_index
from core.scorer import compute_score, is_match


def correct_transcript(
    transcript: str,
    ontology_terms: list[str],
    acronym_map: dict[str, str] = None,
    max_window: int = 5,
    weights: dict = None,
    stopwords: frozenset = None,
    phonetic_index: dict = None
) -> dict:
    """
    Main correction function. Takes a raw STT transcript and returns
    a corrected version by scanning with a sliding window.

    Args:
        transcript     : raw STT string
        ontology_terms : list of domain terms to match against
        acronym_map    : dict of enunciated → acronym e.g. {"eye pee see": "IPC"}
        max_window     : maximum window size (default 5 tokens)
        weights        : optional scorer weight override
        stopwords      : preloaded frozenset (loaded fresh if None)
        phonetic_index : precomputed index (built fresh if None)

    Returns:
        dict with:
            "original"  : raw input transcript
            "corrected" : corrected transcript
            "replacements": list of (original_phrase, replacement) tuples
    """

    # Load dependencies if not precomputed
    if stopwords is None:
        stopwords = load_stopwords()

    if phonetic_index is None:
        phonetic_index = build_phonetic_index(ontology_terms)

    if acronym_map is None:
        acronym_map = {}

    # Normalize acronym map keys to lowercase
    acronym_map = {k.lower(): v for k, v in acronym_map.items()}

    # Tokenize transcript on whitespace
    tokens = transcript.strip().split()
    corrected_tokens = []
    replacements = []

    i = 0
    while i < len(tokens):
        matched = False

        # Try windows from largest to smallest
        for window_size in range(min(max_window, len(tokens) - i), 0, -1):
            window_tokens = tokens[i:i + window_size]
            window_phrase = " ".join(window_tokens).lower()

            # Check 1 — stopword only window, skip immediately
            if is_stopword_only(window_tokens, stopwords):
                corrected_tokens.append(tokens[i])
                i += 1
                matched = True
                break

            # Check 2 — acronym HashMap direct lookup
            if window_phrase in acronym_map:
                replacement = acronym_map[window_phrase]
                corrected_tokens.append(replacement)
                replacements.append((window_phrase, replacement))
                i += window_size
                matched = True
                break

            # Check 3 — phonetic + scorer match against ontology
            best_score = 0.0
            best_term = None

            for term in ontology_terms:
                # Skip terms longer than current window (length filter)
                term_token_count = len(term.split())
                if term_token_count != window_size:
                    continue

                score = compute_score(
                    window_phrase,
                    term,
                    phonetic_index,
                    weights=weights
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

        # No match at any window size — keep original token
        if not matched:
            corrected_tokens.append(tokens[i])
            i += 1

    return {
        "original": transcript,
        "corrected": " ".join(corrected_tokens),
        "replacements": replacements
    }