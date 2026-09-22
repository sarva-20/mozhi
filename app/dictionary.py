"""
Domain term dictionary loader.

Loads the JSON dictionary of domain-specific terms and exposes:
  - DOMAIN_TERMS: list[str] of canonical terms
  - ACRONYM_TERMS: list[str] subset that are acronyms
  - MAX_TERM_TOKENS: longest term in tokens (bounds sliding-window size)
  - get_acronym_index(): casefolded acronym -> canonical term, for the exact
    match gate in letter_composer
"""
import json
from pathlib import Path
from functools import lru_cache

DEFAULT_DICT_PATH = Path(__file__).parent.parent / "data" / "domain_terms.json"


@lru_cache(maxsize=1)
def load_dictionary(path: str | Path = DEFAULT_DICT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_terms(path: str | Path = DEFAULT_DICT_PATH) -> list[dict]:
    return load_dictionary(path)["terms"]


def get_term_strings(path: str | Path = DEFAULT_DICT_PATH) -> list[str]:
    return [t["term"] for t in get_terms(path)]


@lru_cache(maxsize=1)
def get_acronym_index(path: str | Path = DEFAULT_DICT_PATH) -> dict[str, str]:
    """Casefolded acronym -> canonical term, built from the `is_acronym: true`
    entries (e.g. 'sqli' -> 'SQLi'). Used by letter_composer for its exact-match
    gate, so composed candidates can only ever resolve to acronyms already in
    the dictionary."""
    return {t["term"].casefold(): t["term"] for t in get_terms(path) if t.get("is_acronym")}


def get_max_term_tokens(path: str | Path = DEFAULT_DICT_PATH) -> int:
    """Longest domain term, measured in whitespace tokens (e.g. 'two-factor
    authentication' = 2). Bounds how wide the sliding window needs to be."""
    return max(len(t["term"].split()) for t in get_terms(path))


if __name__ == "__main__":
    terms = get_terms()
    print(f"Loaded {len(terms)} domain terms")
    print(f"Max term length (tokens): {get_max_term_tokens()}")
