import json
import nltk
from pathlib import Path

# Download NLTK stopwords if not already present
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_stopwords


def load_stopwords(config_path: str = "config/stopwords.json") -> frozenset:
    """
    Builds a frozen stopword set by:
    1. Starting with NLTK's English stopwords as base

    // For domain-specific applications, users can provide a JSON config file @ config/stopwords.json with the following structure: 
    2. Adding domain-specific words from config
    3. Removing any words the user wants to keep

    Returns a frozenset for O(1) lookup and immutability at runtime.
    """

    # Step 1 — base set from NLTK
    base = set(nltk_stopwords.words("english"))

    # Step 2 — load user config
    config_file = Path(config_path)

    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)

        # Step 3 — apply additions
        additions = set(word.lower() for word in config.get("add", []))
        base.update(additions)

        # Step 4 — apply removals
        removals = set(word.lower() for word in config.get("remove", []))
        base.difference_update(removals)

    # Step 5 — freeze it
    return frozenset(base)


def is_stopword_only(tokens: list[str], stopwords: frozenset) -> bool:
    """
    Returns True if every token in the list is a stopword.
    Used by the sliding window to skip meaningless groupings.
    """
    return all(token.lower() in stopwords for token in tokens)