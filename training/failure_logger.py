import json
from pathlib import Path
from datetime import datetime


# Paths
TRAINING_DIR = Path(__file__).parent
FAILURES_FILE = TRAINING_DIR / "failures.json"
CONFIG_DIR = TRAINING_DIR.parent / "config"
ACRONYMS_FILE = CONFIG_DIR / "acronyms.json"


def _load_failures() -> list[dict]:
    """
    Loads existing failure log.
    Returns empty list if no failures logged yet.
    """
    if not FAILURES_FILE.exists():
        return []

    with open(FAILURES_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_failures(failures: list[dict]) -> None:
    """
    Saves failure log back to failures.json.
    """
    with open(FAILURES_FILE, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)


def log_failure(stt_heard: str, correct_term: str) -> dict:
    """
    Logs a single transcription failure.

    Args:
        stt_heard    : what the STT produced e.g. "en dee pee es"
        correct_term : what it should have been e.g. "NDPS"

    Returns:
        The logged entry as a dict.

    Example:
        log_failure("en dee pee es", "NDPS")
    """
    failures = _load_failures()

    # Check if this exact pair already exists
    for entry in failures:
        if (entry["stt_heard"].lower() == stt_heard.lower() and
                entry["correct_term"] == correct_term):
            print(f"Already logged: '{stt_heard}' → '{correct_term}'")
            return entry

    entry = {
        "stt_heard": stt_heard.lower().strip(),
        "correct_term": correct_term.strip(),
        "logged_at": datetime.now().isoformat(),
        "promoted": False
    }

    failures.append(entry)
    _save_failures(failures)

    print(f"Logged failure: '{stt_heard}' → '{correct_term}'")
    return entry


def flush_to_config(auto_promote: bool = False) -> dict:
    """
    Promotes unpromotoed failures into config/acronyms.json.

    Args:
        auto_promote: if True, promotes all unreviewed failures.
                      if False (default), only promotes entries
                      that were manually marked as confirmed.

    Returns:
        dict with promoted count and skipped count.

    Usage:
        # Promote everything (trust all logged failures)
        flush_to_config(auto_promote=True)
    """
    failures = _load_failures()

    if not failures:
        print("No failures logged yet.")
        return {"promoted": 0, "skipped": 0}

    # Load current acronyms
    with open(ACRONYMS_FILE, "r", encoding="utf-8") as f:
        acronyms = json.load(f)

    promoted = 0
    skipped = 0

    for entry in failures:
        if entry.get("promoted"):
            skipped += 1
            continue

        if not auto_promote and not entry.get("confirmed", False):
            skipped += 1
            continue

        key = entry["stt_heard"].lower().strip()
        value = entry["correct_term"].strip()

        if key not in acronyms:
            acronyms[key] = value
            entry["promoted"] = True
            promoted += 1
        else:
            # Already exists in acronyms — mark promoted, skip
            entry["promoted"] = True
            skipped += 1

    # Save updated acronyms and failure log
    with open(ACRONYMS_FILE, "w", encoding="utf-8") as f:
        json.dump(acronyms, f, indent=2, ensure_ascii=False)

    _save_failures(failures)

    print(f"Promoted: {promoted} | Skipped: {skipped}")
    return {"promoted": promoted, "skipped": skipped}


def get_pending_failures() -> list[dict]:
    """
    Returns all failures not yet promoted to config.
    Useful for building a review UI later.
    """
    failures = _load_failures()
    return [f for f in failures if not f.get("promoted")]


def clear_failures() -> None:
    """
    Wipes the failure log entirely.
    Use with caution — unpromotoed entries will be lost.
    """
    _save_failures([])
    print("Failure log cleared.")