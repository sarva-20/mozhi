import json
from pathlib import Path


# Default config directory
CONFIG_DIR = Path(__file__).parent


def _load_json(filename: str) -> dict:
    """
    Loads a JSON file from the config directory.
    Raises a clear error if the file is missing or malformed.
    """
    filepath = CONFIG_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Config file not found: {filepath}\n"
            f"Make sure {filename} exists in the config/ directory."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")


def load_ontology() -> list[str]:
    """
    Loads domain ontology terms from ontology.json.
    Returns a list of strings.

    Expected format:
        { "terms": ["term1", "term2", ...] }
    """
    data = _load_json("ontology.json")

    if "terms" not in data:
        raise ValueError("ontology.json must have a 'terms' key.")

    if not isinstance(data["terms"], list):
        raise ValueError("'terms' in ontology.json must be a list.")

    # Normalize to lowercase, strip whitespace
    return [term.lower().strip() for term in data["terms"] if term.strip()]


def load_acronyms() -> dict[str, str]:
    """
    Loads acronym HashMap from acronyms.json.
    Returns dict mapping enunciated form → acronym.

    Expected format:
        { "eye pee see": "IPC", ... }
    """
    data = _load_json("acronyms.json")

    if not isinstance(data, dict):
        raise ValueError("acronyms.json must be a flat JSON object.")

    # Normalize keys to lowercase
    return {k.lower().strip(): v for k, v in data.items()}


def load_stopwords_config() -> dict:
    """
    Loads stopword additions and removals from stopwords.json.
    Returns raw dict — stopwords.py handles the merging logic.

    Expected format:
        { "add": [...], "remove": [...] }
    """
    data = _load_json("stopwords.json")

    if not isinstance(data, dict):
        raise ValueError("stopwords.json must be a JSON object.")

    return data


def load_scorer_weights() -> dict | None:
    """
    Optionally loads scorer weight overrides from ontology.json.
    Returns None if no weights defined — scorer uses its defaults.

    Expected format in ontology.json:
        {
          "terms": [...],
          "weights": {
            "phonetic": 0.5,
            "jaro": 0.3,
            "token": 0.2
          }
        }
    """
    data = _load_json("ontology.json")
    return data.get("weights", None)


def load_all() -> dict:
    """
    Loads all config in one call.
    Returns a single dict with all Mozhi configuration.

    Usage:
        from config.loader import load_all
        config = load_all()

        ontology  = config["ontology"]
        acronyms  = config["acronyms"]
        weights   = config["weights"]
    """
    return {
        "ontology": load_ontology(),
        "acronyms": load_acronyms(),
        "stopwords": load_stopwords_config(),
        "weights": load_scorer_weights()
    }