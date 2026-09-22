"""
Minimal tokenizer for STT output.

STT transcripts are lowercase, punctuation-light, and often lack the
capitalization/hyphenation of the canonical domain terms (e.g. "two factor
authentication" instead of "two-factor authentication"). The tokenizer
normalizes for matching purposes while preserving the original text so we
can do exact-span replacement later.
"""
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class Token:
    text: str          # original surface form, as it appeared in the input
    start: int         # char offset in the original string
    end: int           # char offset (exclusive) in the original string


def tokenize(text: str) -> list[Token]:
    """Split text into word tokens, keeping original character offsets so
    a matched window can be sliced back out of the source string for
    replacement."""
    return [Token(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def separators(text: str, tokens: list[Token]) -> list[str]:
    """Punctuation sitting between each pair of adjacent tokens, whitespace
    removed: result[k] is what lies between tokens[k] and tokens[k+1] ("" for a
    plain space, "," for ", ", "." for ". ", "-" for a hyphen). tokenize() drops
    punctuation, so this recovers it from the char offsets; letter_composer uses
    it to end a run of letter-sounds at a sentence boundary or a change of
    separator. Length is len(tokens) - 1."""
    return ["".join(text[a.end : b.start].split()) for a, b in zip(tokens, tokens[1:])]


def normalize(token_text: str) -> str:
    """Lowercase, strip trailing possessive/apostrophes for matching."""
    return token_text.lower().strip("'")


if __name__ == "__main__":
    sample = "please check the sea pu usage and the ram before you reboot"
    for t in tokenize(sample):
        print(t)
