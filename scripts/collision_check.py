"""Report candidate domain-term collisions without changing the dictionary.

Usage::

    PYTHONPATH=. python3 scripts/collision_check.py path/to/candidates.json

The common-English reference list is committed in ``data/`` and is never
downloaded at runtime. Exact duplicates are separated before fuzzy scoring;
all remaining candidates are compared with every reference word.
"""
import argparse
import json
from pathlib import Path

from app.pipeline import DEFAULT_THRESHOLD
from app.scorer import combined_score

REPO_ROOT = Path(__file__).parent.parent
COMMON_WORDS_PATH = REPO_ROOT / "data" / "common_english_words.txt"
DOMAIN_TERMS_PATH = REPO_ROOT / "data" / "domain_terms.json"


def load_terms(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    terms = data.get("terms")
    if not isinstance(terms, list):
        raise ValueError(f"{path} must contain a 'terms' list")
    if any(not isinstance(item, dict) or not item.get("term") for item in terms):
        raise ValueError(f"Every entry in {path} must have a non-empty 'term'")
    return terms


def load_common_words() -> list[str]:
    if not COMMON_WORDS_PATH.is_file():
        raise FileNotFoundError(f"Missing static reference file: {COMMON_WORDS_PATH}")
    return [line.strip() for line in COMMON_WORDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def duplicate_reason(term: str, existing: dict[str, str], seen: dict[str, str]) -> str | None:
    key = term.casefold()
    if key in existing:
        return f"already exists as {existing[key]!r}"
    if key in seen:
        return f"duplicates candidate {seen[key]!r}"
    return None


def build_report(candidate_path: Path) -> str:
    candidates = load_terms(candidate_path)
    existing_terms = load_terms(DOMAIN_TERMS_PATH)
    common_words = load_common_words()
    existing = {item["term"].casefold(): item["term"] for item in existing_terms}
    seen: dict[str, str] = {}
    clean: list[str] = []
    duplicate: list[tuple[str, str]] = []
    flagged: list[tuple[str, list[tuple[str, float]]]] = []

    for item in candidates:
        term = item["term"].strip()
        reason = duplicate_reason(term, existing, seen)
        if reason:
            duplicate.append((term, reason))
            continue
        seen[term.casefold()] = term

        collisions = []
        for word in common_words:
            score = combined_score(term, word)["total"]
            if score >= DEFAULT_THRESHOLD:
                collisions.append((word, score))
        collisions.sort(key=lambda collision: (-collision[1], collision[0]))
        if collisions:
            flagged.append((term, collisions))
        else:
            clean.append(term)

    lines = [
        f"Collision check: {candidate_path.name}",
        f"Existing dictionary: {DOMAIN_TERMS_PATH.name}",
        f"Common-word reference: {COMMON_WORDS_PATH.name} ({len(common_words)} words)",
        f"Threshold: {DEFAULT_THRESHOLD:.2f}",
        f"Candidates: {len(candidates)}",
        "",
        f"CLEAN ({len(clean)})",
        "=" * 80,
    ]
    lines.extend(f"  {term}" for term in clean)
    lines.extend(["", f"DUPLICATE ({len(duplicate)})", "=" * 80])
    lines.extend(f"  {term!r} — {reason} — skip" for term, reason in duplicate)
    lines.extend(["", f"FLAGGED ({len(flagged)})", "=" * 80])
    for term, collisions in flagged:
        lines.append(f"  {term!r}")
        lines.extend(f"    collides with {word!r} at {score:.3f}" for word, score in collisions)
    lines.extend(
        [
            "",
            "No terms were merged automatically.",
            "Review CLEAN and FLAGGED entries before manually updating data/domain_terms.json.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_file", type=Path, help="Candidate terms JSON file")
    args = parser.parse_args()
    if not args.candidate_file.is_file():
        parser.error(f"candidate file not found: {args.candidate_file}")

    report = build_report(args.candidate_file)
    output_path = REPO_ROOT / "reports" / f"collision_check_{args.candidate_file.stem}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())