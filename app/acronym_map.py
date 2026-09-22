"""
Acronym HashMap fallback.

Acronyms are the hardest case for the fuzzy scorer: an STT engine often
transcribes an acronym as its *spelled-out sound* ("sea pu", "ess ess dee")
rather than anything visually or phonetically close to the acronym string
itself ("CPU", "SSD"). Metaphone/Jaro-Winkler on "sea pu" vs "CPU" score
reasonably well because Metaphone strips vowels, but some acronyms
(e.g. "aitch dee dee" -> "HDD") score too low to clear the fuzzy threshold
safely without risking false positives elsewhere.

This is a plain O(1) dict lookup - zero inference, zero scoring - checked
BEFORE the fuzzy scorer runs. It only needs to cover the acronyms in the
domain dictionary, and only the letter-name pronunciations that don't
already clear the fuzzy threshold on their own (kept lean deliberately,
not padded, so it stays traceable in the paper as "structural evidence
of no-model-inference").

Entries are added/verified against the actual STT failure recordings
collected from the team + volunteers, not guessed - see data/domain_terms.json
for the acronym list this must stay in sync with.
"""

# candidate (normalized, lowercase, space-joined letter names) -> canonical term
ACRONYM_PHONETIC_MAP: dict[str, str] = {
    "sea pu": "CPU",
    "see pu": "CPU",
    "sea pew": "CPU",
    "ess ess dee": "SSD",
    "aitch dee dee": "HDD",
    "aitch dee em eye": "HDMI",
    "you ess bee": "USB",
    "bye os": "BIOS",
    "bio's": "BIOS",
    "ay pea eye": "API",
    "ay p i": "API",
    "ess dee kay": "SDK",
    "eye dee ee": "IDE",
    "gooey": "GUI",
    "goo ey": "GUI",
    "see ell eye": "CLI",
    "you are ell": "URL",
    "dee en ess": "DNS",
    "dee aitch see pea": "DHCP",
    "vee pea en": "VPN",
    "ess ess aitch": "SSH",
    "tee see pea": "TCP",
    "you dee pea": "UDP",
    "ess ess ell": "SSL",
    "tee ell ess": "TLS",
    "oh auth": "OAuth",
    "oh authe": "OAuth",
    "ess ell ay": "SLA",
    "ess ess oh": "SSO",
    "oh ess": "OS",
    "eye ess pea": "ISP",
    "ess cue ell": "SQL",
    "em ay see": "MAC address",
    # These short acronyms (SATA, STP, DDoS, IDS, PAT, HSTS, and IPS) scored
    # above the fuzzy threshold against common English words (for example,
    # STP vs "step"/"stop" at 0.94). IPS was missed in the original exclusion
    # list because it also collides with "IP", already present in the core
    # dictionary through "IP address" - a direct in-domain collision caught
    # only when the full test suite ran after the merge. Because Jaro-Winkler
    # inflates similarity for short strings, handle them only through exact
    # deterministic lookup, never general fuzzy scoring.
    "ess ay tee ay": "SATA",
    "ess tee pee": "STP",
    "dee dee oh ess": "DDoS",
    "dee dos": "DDoS",
    "eye dee ess": "IDS",
    "pea ay tee": "PAT",
    "aitch ess tee ess": "HSTS",
    "eye pea ess": "IPS",
    # LAN, WAN, RAID, NAS are conventionally pronounced as words rather than
    # spelled out letter-by-letter, so they're left to the fuzzy scorer
    # (surface/Metaphone similarity on "lan"/"wan"/"raid"/"nas") rather than
    # added here - adding them as letter-spellings would be phonetically
    # inaccurate, not just redundant.
}


def lookup_acronym(candidate: str) -> str | None:
    """O(1) exact lookup on the normalized candidate string. Returns the
    canonical term, or None if not a known spelled-out acronym."""
    return ACRONYM_PHONETIC_MAP.get(candidate.lower().strip())


# Known STT phrase mishearings that don't score high enough on fuzzy matching
# but are verifiable from real failure recordings. These are exact-match lookups,
# not fuzzy - they bypass the scorer and are checked at the same priority as
# acronym_map, before fuzzy matching. Keep these lean and only add entries
# backed by actual STT failures, never speculative.
PHRASE_ALIASES: dict[str, str] = {
    "one link": "WAN link",
}


def lookup_phrase_alias(candidate: str) -> str | None:
    """O(1) exact lookup on the normalized candidate string. Returns the
    canonical term, or None if not a known phrase alias."""
    return PHRASE_ALIASES.get(candidate.lower().strip())


if __name__ == "__main__":
    for k in ["sea pu", "ess ess dee", "banana"]:
        print(k, "->", lookup_acronym(k))
    print()
    for k in ["one link", "WAN link", "banana"]:
        print(k, "->", lookup_phrase_alias(k))
