"""
Tests for scripts/evaluate_accuracy.py: word alignment, region finding, category
suggestion, aggregation, CSV review preservation, discovery/caching, and an
end-to-end run of main() on temp directories (no Whisper needed: transcripts are
supplied as cached JSON or via a fake transcriber).

Unit tests inject a fake corrector so they exercise the evaluation logic alone;
the fixture tests at the bottom use the real pipeline on the two saved STT
transcripts (test_audio1 / test_audio2) and pin the numbers from the first run.
"""
import csv
import json
import os
import random
from types import SimpleNamespace

import pytest

from scripts import evaluate_accuracy as ev


# --- helpers -------------------------------------------------------------------------------------

def naive_levenshtein(a: list[str], b: list[str]) -> int:
    """Independent textbook DP, used only to cross-check the rapidfuzz-based code."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def fake_corrector(edits: dict[str, str]):
    """A corrector that replaces each key (a substring of the input) with its value
    and reports Match-like objects with correct char offsets."""
    def corrector(text: str):
        matches, out, cursor = [], [], 0
        spans = sorted((text.find(k), k) for k in edits if k in text)
        for start, key in spans:
            end = start + len(key)
            out.append(text[cursor:start])
            out.append(edits[key])
            matches.append(SimpleNamespace(
                start=start, end=end, original=key, replacement=edits[key], method="fake", score=1.0))
            cursor = end
        out.append(text[cursor:])
        return SimpleNamespace(corrected_text="".join(out), matches=matches)
    return corrector


def rows_of(stt, expected, edits=None):
    return ev.evaluate_pair("t", stt, expected, corrector=fake_corrector(edits or {})).rows


def norm(text):
    return [w.norm for w in ev.to_words(text)]


# --- words ---------------------------------------------------------------------------------------

def test_to_words_lowercases_and_drops_punctuation():
    assert norm("Check the SLA, then Verify it.") == ["check", "the", "sla", "then", "verify", "it"]


def test_to_words_keeps_surface_form_and_offsets():
    (w,) = ev.to_words("  SLA")
    assert (w.text, w.norm, w.start, w.end) == ("SLA", "sla", 2, 5)


def test_to_words_splits_hyphens_and_dots_identically_on_both_sides():
    assert norm("two-factor X.509") == ["two", "factor", "x", "509"]


def test_to_words_drops_tokens_that_normalize_to_nothing():
    assert norm("it ' works") == ["it", "works"]


# --- align / error_counts ------------------------------------------------------------------------

def test_identical_sequences_have_zero_wer():
    c = ev.error_counts(["a", "b", "c"], ["a", "b", "c"])
    assert (c.errors, c.wer) == (0, 0.0)


def test_substitution_counts():
    c = ev.error_counts(["a", "x", "c"], ["a", "b", "c"])
    assert (c.substitutions, c.deletions, c.insertions) == (1, 0, 0)
    assert c.wer == pytest.approx(1 / 3)


def test_word_missing_from_hypothesis_is_a_deletion():
    c = ev.error_counts(["a", "c"], ["a", "b", "c"])
    assert (c.substitutions, c.deletions, c.insertions) == (0, 1, 0)


def test_extra_hypothesis_word_is_an_insertion():
    c = ev.error_counts(["a", "x", "c"], ["a", "c"])
    assert (c.substitutions, c.deletions, c.insertions) == (0, 0, 1)
    assert c.wer == 0.5  # errors are divided by REFERENCE length


def test_wer_can_exceed_one():
    assert ev.error_counts(["x"] * 5, ["a"]).wer == 5.0


def test_empty_reference_is_rejected():
    with pytest.raises(ValueError):
        ev.error_counts(["a"], [])


def test_align_tag_semantics():
    (eq1, ins, eq2) = ev.align(["a", "c"], ["a", "b", "c"])
    assert ins.tag == "insert" and (ins.ref_start, ins.ref_end) == (1, 2) and ins.hyp_start == ins.hyp_end
    (_, dele, _) = ev.align(["a", "x", "c"], ["a", "c"])
    assert dele.tag == "delete" and (dele.hyp_start, dele.hyp_end) == (1, 2) and dele.ref_start == dele.ref_end


def test_error_counts_agree_with_an_independent_dp_on_random_inputs():
    rng = random.Random(1234)
    vocab = ["a", "b", "c", "d", "e"]
    for _ in range(400):
        hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 9))]
        ref = [rng.choice(vocab) for _ in range(rng.randint(1, 9))]
        c = ev.error_counts(hyp, ref)
        assert c.errors == naive_levenshtein(hyp, ref), (hyp, ref)
        # every reference word is accounted for exactly once as equal, substituted, or deleted
        blocks = ev.align(hyp, ref)
        assert sum(b.ref_end - b.ref_start for b in blocks) == len(ref)
        assert sum(b.hyp_end - b.hyp_start for b in blocks) == len(hyp)


# --- build_groups --------------------------------------------------------------------------------

def match(text, original, replacement, method="fuzzy"):
    start = text.index(original)
    return SimpleNamespace(start=start, end=start + len(original), original=original,
                           replacement=replacement, method=method)


def test_build_groups_without_matches_is_one_group_per_token():
    stt = ev.to_words("the cat sat")
    groups = ev.build_groups(stt, [])
    assert [(g.start, g.end, g.edit) for g in groups] == [(0, 1, ""), (1, 2, ""), (2, 3, "")]
    assert [g.out[0].norm for g in groups] == ["the", "cat", "sat"]


def test_build_groups_multi_token_match_is_one_group():
    text = "check the one link now"
    groups = ev.build_groups(ev.to_words(text), [match(text, "one link", "WAN link", "phrase_alias")])
    assert [(g.start, g.end) for g in groups] == [(0, 1), (1, 2), (2, 4), (4, 5)]
    g = groups[2]
    assert [w.text for w in g.out] == ["WAN", "link"]
    assert g.edit == "phrase_alias: 'one link' -> 'WAN link'"


def test_build_groups_replacement_with_a_different_word_count():
    text = "the sea pu is hot"
    groups = ev.build_groups(ev.to_words(text), [match(text, "sea pu", "CPU")])
    g = next(g for g in groups if g.edit)
    assert (g.start, g.end) == (1, 3) and [w.text for w in g.out] == ["CPU"]


def test_build_groups_two_matches_and_unordered_input():
    text = "read the land"
    m1, m2 = match(text, "read", "RAID"), match(text, "land", "LAN")
    groups = ev.build_groups(ev.to_words(text), [m2, m1])  # deliberately out of order
    assert [g.edit != "" for g in groups] == [True, False, True]


# --- regions (through evaluate_pair with a fake corrector) -----------------------------------------

def test_no_errors_no_rows():
    assert rows_of("the router is down", "the router is down") == []


def test_case_only_difference_is_not_an_error():
    assert rows_of("check the ram", "check the RAM", {"ram": "RAM"}) == []


def test_single_substitution_fixed():
    (row,) = rows_of("check the read array", "check the RAID array", {"read": "RAID"})
    assert (row.row_type, row.expected, row.stt, row.corrected) == ("stt_error", "RAID", "read", "RAID")
    assert (row.fixed, row.outcome, row.category_suggested) == (True, "fixed", ev.PHONETIC)


def test_single_substitution_missed_when_corrector_does_nothing():
    (row,) = rows_of("check the read array", "check the RAID array")
    assert (row.fixed, row.outcome, row.corrected) == (False, "missed", "read")
    assert row.mozhi_edits == ""


def test_changed_but_still_wrong():
    (row,) = rows_of("check the read array", "check the RAID array", {"read": "RAM"})
    assert (row.fixed, row.outcome, row.corrected) == (False, "changed_wrong", "RAM")


def test_regression_when_stt_was_right_and_mozhi_broke_it():
    (row,) = rows_of("this is a simple idea", "this is a simple idea", {"idea": "IDE"})
    assert (row.row_type, row.outcome, row.fixed) == ("regression", "regression", False)
    assert (row.expected, row.stt, row.corrected) == ("idea", "idea", "IDE")
    assert row.category_suggested == "" and row.phonetic_score is None


def test_match_straddling_an_error_and_a_correct_word_expands_the_region():
    # Only "one" is wrong, but Mozhi matched "one link" as a unit, so the region is the whole match.
    (row,) = rows_of("the one link is down", "the WAN link is down", {"one link": "WAN link"})
    assert (row.expected, row.stt, row.corrected) == ("WAN link", "one link", "WAN link")
    assert (row.n_expected, row.n_stt, row.fixed) == (2, 2, True)


def test_match_collapsing_two_stt_words_into_one_expected_word():
    (row,) = rows_of("the sea pu is hot", "the CPU is hot", {"sea pu": "CPU"})
    assert (row.expected, row.stt, row.corrected) == ("CPU", "sea pu", "CPU")
    assert (row.n_expected, row.n_stt, row.fixed) == (1, 2, True)


def test_pure_omission_is_a_region_with_no_stt_words_and_is_never_fixable():
    (row,) = rows_of("check the array", "check the RAID array")
    assert (row.expected, row.stt, row.n_stt) == ("RAID", "", 0)
    assert (row.category_suggested, row.fixed, row.outcome) == (ev.OMISSION, False, "missed")


def test_pure_insertion_extra_stt_words():
    (row,) = rows_of("check the move away array", "check the array")
    assert (row.expected, row.stt, row.n_expected) == ("", "move away", 0)
    assert (row.category_suggested, row.fixed) == (ev.INSERTION, False)


def test_adjacent_errors_form_one_region():
    (row,) = rows_of("check the aa bb array", "check the xx yy array")
    assert (row.expected, row.stt) == ("xx yy", "aa bb")


def test_separated_errors_form_separate_regions_numbered_in_order():
    rows = rows_of("read the land now", "RAID the LAN now", {"read": "RAID", "land": "LAN"})
    assert [(r.region, r.stt, r.fixed) for r in rows] == [(1, "read", True), (2, "land", True)]


def test_match_that_swallows_two_separate_errors_merges_them_into_one_region():
    # "aa ok bb" is matched as one unit; the equal word between the two errors is dragged in.
    (row,) = rows_of("x aa ok bb y", "x XX ok YY y", {"aa ok bb": "XX ok YY"})
    assert (row.expected, row.stt, row.corrected, row.fixed) == ("XX ok YY", "aa ok bb", "XX ok YY", True)


def test_empty_expected_transcript_is_rejected():
    with pytest.raises(ValueError):
        ev.evaluate_pair("t", "hello", "  ...  ", corrector=fake_corrector({}))


def test_rows_and_wer_are_consistent():
    # Every STT-vs-expected error lies in some region, so a clean STT has no rows and vice versa.
    r = ev.evaluate_pair("t", "check the read array", "check the RAID array", corrector=fake_corrector({"read": "RAID"}))
    assert (r.raw.errors, r.corrected.errors) == (1, 0)
    assert (r.raw.wer, r.corrected.wer) == (0.25, 0.0)


# --- suggest_category ------------------------------------------------------------------------------

def test_suggest_omission_when_stt_has_no_word():
    assert ev.suggest_category([], ["raid"]) == (ev.OMISSION, None)


def test_suggest_insertion_when_expected_has_no_word():
    assert ev.suggest_category(["move", "away"], []) == (ev.INSERTION, None)


@pytest.mark.parametrize("stt,exp", [("read", "raid"), ("land", "lan"), ("one link", "wan link")])
def test_suggest_phonetic_for_sound_alikes(stt, exp):
    category, score = ev.suggest_category(stt.split(), exp.split())
    assert category == ev.PHONETIC and score >= ev.PHONETIC_CLOSE_THRESHOLD


@pytest.mark.parametrize("stt,exp", [("is", "keeps"), ("if there is a leg for", "the sla before")])
def test_suggest_non_phonetic_for_unrelated_words(stt, exp):
    category, score = ev.suggest_category(stt.split(), exp.split())
    assert category == ev.NON_PHONETIC and score < ev.PHONETIC_CLOSE_THRESHOLD


def test_suggest_threshold_is_a_parameter_and_boundary_inclusive():
    _, score = ev.suggest_category(["land"], ["lan"])
    assert ev.suggest_category(["land"], ["lan"], threshold=score)[0] == ev.PHONETIC
    assert ev.suggest_category(["land"], ["lan"], threshold=score + 0.01)[0] == ev.NON_PHONETIC


def test_suggest_digits_only_words_have_no_phonetic_signal():
    assert ev.suggest_category(["802"], ["509"]) == (ev.NON_PHONETIC, 0.0)


# --- domain relevance (Task 2: category_suggested must not dilute with non-domain STT noise) ------

def test_domain_words_include_single_and_multi_word_terms():
    words = ev._domain_words()
    assert "raid" in words and "lan" in words            # single-word terms
    assert {"wan", "link"} <= words                       # each word of a multi-word term
    assert {"two", "factor", "authentication"} <= words   # hyphenated term, split like everywhere else


def test_domain_words_exclude_ordinary_words_never_in_any_term():
    words = ev._domain_words()
    assert "scheduled" not in words and "approval" not in words and "that" not in words


@pytest.mark.parametrize("exp_words", [["raid"], ["wan", "link"], ["the", "wan", "dropped"]])
def test_is_domain_relevant_true_when_any_word_is_a_domain_word(exp_words):
    assert ev.is_domain_relevant(exp_words) is True


@pytest.mark.parametrize("exp_words", [["scheduled"], ["that", "the"], []])
def test_is_domain_relevant_false_when_no_word_is_a_domain_word(exp_words):
    assert ev.is_domain_relevant(exp_words) is False


@pytest.mark.parametrize("stt,exp", [
    ("schedule", "scheduled"), ("passed", "past"), ("the", "that"), ("catching", "caching"),
])
def test_suggest_out_of_scope_for_phonetically_close_non_domain_words(stt, exp):
    """These are real examples from the 32-file accuracy run: phonetically close
    (>= threshold) but nothing to do with the dictionary - must NOT be counted
    as an addressable phonetic_substitution miss."""
    category, score = ev.suggest_category([stt], [exp])
    assert category == ev.OUT_OF_SCOPE and score >= ev.PHONETIC_CLOSE_THRESHOLD


@pytest.mark.parametrize("stt,exp", [("read", "raid"), ("land", "lan"), ("one link", "wan link")])
def test_suggest_still_phonetic_for_domain_relevant_sound_alikes(stt, exp):
    """Regression guard: the domain-relevance gate must not touch the genuine
    addressable cases suggest_phonetic_for_sound_alikes above already pins."""
    category, score = ev.suggest_category(stt.split(), exp.split())
    assert category == ev.PHONETIC and score >= ev.PHONETIC_CLOSE_THRESHOLD


def test_suggest_out_of_scope_beats_non_phonetic_by_construction():
    # out_of_scope is a NARROWING of what used to be phonetic_substitution, not
    # an alternative to non_phonetic_substitution: it never fires below threshold.
    category, score = ev.suggest_category(["is"], ["keeps"])
    assert category == ev.NON_PHONETIC
    assert score < ev.PHONETIC_CLOSE_THRESHOLD


def test_out_of_scope_is_in_categories_and_summarized():
    assert ev.OUT_OF_SCOPE in ev.CATEGORIES
    stats = ev.catch_stats([make_row(category_suggested=ev.OUT_OF_SCOPE)])
    assert stats[ev.OUT_OF_SCOPE] == (0, 1)


# --- catch_stats / catch_rate / review precedence -----------------------------------------------------

def make_row(**kw):
    base = dict(file="f", region=1, row_type="stt_error", expected="a", stt="b", corrected="b",
                n_expected=1, n_stt=1, fixed=False, outcome="missed", mozhi_edits="",
                category_suggested=ev.PHONETIC, phonetic_score=0.9)
    base.update(kw)
    return ev.Row(**base)


def test_catch_rate_handles_zero_total():
    assert ev.catch_rate(0, 0) is None
    assert ev.catch_rate(1, 4) == 0.25


def test_catch_stats_counts_fixed_over_total_per_category_and_skips_regressions():
    rows = [
        make_row(fixed=True, outcome="fixed"),
        make_row(),
        make_row(category_suggested=ev.NON_PHONETIC),
        make_row(row_type="regression", outcome="regression", category_suggested=""),
    ]
    stats = ev.catch_stats(rows)
    assert stats[ev.PHONETIC] == (1, 2)
    assert stats[ev.NON_PHONETIC] == (0, 1)
    assert stats[ev.OMISSION] == (0, 0)


def test_reviewed_category_takes_precedence_over_suggestion():
    row = make_row(category_reviewed=ev.NON_PHONETIC)
    assert row.category == ev.NON_PHONETIC
    assert ev.catch_stats([row])[ev.PHONETIC] == (0, 0)
    assert make_row().category == ev.PHONETIC


# --- CSV writing and review preservation ------------------------------------------------------------------

def test_write_csv_header_and_formatting(tmp_path):
    path = tmp_path / "out.csv"
    ev.write_csv(path, [make_row(fixed=True, outcome="fixed"), make_row(category_suggested=ev.OMISSION, phonetic_score=None)])
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ev.CSV_FIELDS
    assert (rows[0]["fixed"], rows[0]["phonetic_score"]) == ("yes", "0.900")
    assert (rows[1]["fixed"], rows[1]["phonetic_score"]) == ("no", "")
    assert rows[0]["category_reviewed"] == ""


def test_load_reviews_missing_file_is_empty(tmp_path):
    assert ev.load_reviews(tmp_path / "nope.csv") == {}


def hand_review(path, key_row, category):
    """Simulate a human filling in category_reviewed in the CSV."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if (r["file"], r["expected"], r["stt"]) == (key_row.file, key_row.expected, key_row.stt):
            r["category_reviewed"] = category
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev.CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_review_round_trip_survives_regeneration(tmp_path):
    path = tmp_path / "eval.csv"
    first = [make_row(expected="RAID", stt="read"), make_row(expected="keeps", stt="is", category_suggested=ev.NON_PHONETIC)]
    ev.write_csv(path, first)
    hand_review(path, first[1], ev.PHONETIC)  # the human disagrees with the suggestion

    reviews = ev.load_reviews(path)
    assert reviews == {("f", "keeps", "is"): ev.PHONETIC}  # blank reviews are not loaded

    regenerated = [make_row(expected="RAID", stt="read"), make_row(expected="keeps", stt="is", category_suggested=ev.NON_PHONETIC)]
    assert ev.apply_reviews(regenerated, reviews) == 1
    assert regenerated[0].category_reviewed == ""
    assert regenerated[1].category_reviewed == ev.PHONETIC
    assert regenerated[1].category == ev.PHONETIC and regenerated[1].category_suggested == ev.NON_PHONETIC

    ev.write_csv(path, regenerated)  # writing again keeps the review
    assert ev.load_reviews(path) == reviews


def test_review_is_dropped_when_the_row_no_longer_exists(tmp_path):
    path = tmp_path / "eval.csv"
    old = make_row(expected="RAID", stt="read")
    ev.write_csv(path, [old])
    hand_review(path, old, ev.OMISSION)
    fresh = [make_row(expected="RAID", stt="red")]  # STT changed, so it is a different error
    assert ev.apply_reviews(fresh, ev.load_reviews(path)) == 0
    assert fresh[0].category_reviewed == ""


# --- discovery and transcript caching -----------------------------------------------------------------------

def test_discover_pairs_reports_orphans_in_both_directions(tmp_path):
    (tmp_path / "a.m4a").write_bytes(b"")
    (tmp_path / "a.expected.txt").write_text("hello")
    (tmp_path / "b.m4a").write_bytes(b"")               # audio, no expected text
    (tmp_path / "c.expected.txt").write_text("hello")   # expected text, no audio
    (tmp_path / "notes.txt").write_text("ignored")
    pairs, warnings = ev.discover_pairs(tmp_path)
    assert [(a.name, e.name) for a, e in pairs] == [("a.m4a", "a.expected.txt")]
    assert any("b.m4a" in w for w in warnings) and any("c.expected.txt" in w for w in warnings)
    assert len(warnings) == 2


def test_discover_pairs_sorted_and_multiple_extensions(tmp_path):
    for name in ("z.m4a", "m.wav", "a.mp3"):
        (tmp_path / name).write_bytes(b"")
        (tmp_path / (name.split(".")[0] + ".expected.txt")).write_text("x")
    pairs, warnings = ev.discover_pairs(tmp_path)
    assert [a.name for a, _ in pairs] == ["a.mp3", "m.wav", "z.m4a"] and warnings == []


def make_audio_and_cache(tmp_path, transcript="cached text", cache_newer=True):
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"")
    stt_dir = tmp_path / "stt"
    stt_dir.mkdir()
    cache = stt_dir / "clip.json"
    cache.write_text(json.dumps({"transcript": transcript}))
    now = os.stat(audio).st_mtime
    os.utime(cache, (now + (10 if cache_newer else -10),) * 2)
    return audio, stt_dir


class CountingTranscriber:
    def __init__(self, text="fresh text"):
        self.calls, self.text = 0, text

    def __call__(self, audio):
        self.calls += 1
        return {"file": audio.name, "transcript": self.text}


def test_get_transcript_uses_a_newer_cache_without_transcribing(tmp_path):
    audio, stt_dir = make_audio_and_cache(tmp_path)
    t = CountingTranscriber()
    assert ev.get_transcript(audio, stt_dir, t) == ("cached text", "cached") and t.calls == 0


def test_get_transcript_retranscribes_a_stale_cache_and_rewrites_it(tmp_path):
    audio, stt_dir = make_audio_and_cache(tmp_path, cache_newer=False)
    t = CountingTranscriber()
    assert ev.get_transcript(audio, stt_dir, t) == ("fresh text", "fresh") and t.calls == 1
    assert json.loads((stt_dir / "clip.json").read_text())["transcript"] == "fresh text"


def test_get_transcript_retranscribe_flag_overrides_a_fresh_cache(tmp_path):
    audio, stt_dir = make_audio_and_cache(tmp_path)
    t = CountingTranscriber()
    assert ev.get_transcript(audio, stt_dir, t, retranscribe=True)[1] == "fresh" and t.calls == 1


def test_get_transcript_transcribes_when_there_is_no_cache(tmp_path):
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"")
    t = CountingTranscriber()
    assert ev.get_transcript(audio, tmp_path / "new_stt_dir", t) == ("fresh text", "fresh")
    assert (tmp_path / "new_stt_dir" / "clip.json").exists()


# --- the two recorded files, through the real pipeline (numbers from the first run) ----------------------------------

AUDIO1_STT = "Check if there is a leg for you escalate, then move away and verify the ISP outage."
AUDIO1_EXPECTED = "check the SLA before you escalate, then verify the ISP outage"
AUDIO2_STT = "The land is fine, but the one link to the read array is dropping."
AUDIO2_EXPECTED = "the LAN is fine but the WAN link to the RAID array keeps dropping"


def test_fixture_test_audio2_numbers():
    r = ev.evaluate_pair("test_audio2", AUDIO2_STT, AUDIO2_EXPECTED)
    assert (r.raw.ref_words, r.raw.substitutions, r.raw.deletions, r.raw.insertions) == (14, 4, 0, 0)
    assert (r.corrected.substitutions, r.corrected.deletions, r.corrected.insertions) == (1, 0, 0)
    assert r.raw.wer == pytest.approx(4 / 14) and r.corrected.wer == pytest.approx(1 / 14)
    assert r.corrected_text == "The LAN is fine, but the WAN link to the RAID array is dropping."
    got = [(x.expected, x.stt, x.corrected, x.outcome, x.category_suggested) for x in r.rows]
    assert got == [
        ("LAN", "land", "LAN", "fixed", ev.PHONETIC),
        ("WAN link", "one link", "WAN link", "fixed", ev.PHONETIC),
        ("RAID", "read", "RAID", "fixed", ev.PHONETIC),
        ("keeps", "is", "is", "missed", ev.NON_PHONETIC),
    ]
    assert r.rows[1].mozhi_edits == "phrase_alias: 'one link' -> 'WAN link'"
    assert r.rows[0].phonetic_score == pytest.approx(0.911, abs=1e-3)
    assert r.rows[3].phonetic_score == 0.0


def test_fixture_test_audio1_numbers():
    r = ev.evaluate_pair("test_audio1", AUDIO1_STT, AUDIO1_EXPECTED)
    assert (r.raw.ref_words, r.raw.substitutions, r.raw.deletions, r.raw.insertions) == (11, 3, 0, 6)
    assert r.corrected == r.raw and r.corrected_text == AUDIO1_STT  # Mozhi changes nothing here
    got = [(x.expected, x.stt, x.corrected, x.outcome, x.category_suggested) for x in r.rows]
    assert got == [
        ("the SLA before", "if there is a leg for", "if there is a leg for", "missed", ev.NON_PHONETIC),
        ("", "move away and", "move away and", "missed", ev.INSERTION),
    ]
    assert r.rows[0].phonetic_score == pytest.approx(0.676, abs=1e-3)


def test_fixture_summary_numbers():
    results = [ev.evaluate_pair("test_audio1", AUDIO1_STT, AUDIO1_EXPECTED),
               ev.evaluate_pair("test_audio2", AUDIO2_STT, AUDIO2_EXPECTED)]
    summary = ev.summarize(results)
    assert "Files evaluated: 2   Reference words: 25" in summary
    assert "raw WER:        52.0%  (13 errors / 25 words)" in summary       # (9 + 4) / 25
    assert "corrected WER:  40.0%  (10 errors / 25 words)" in summary       # (9 + 1) / 25
    assert "PHONETIC-SUBSTITUTION CATCH RATE: 3/3 = 100.0%" in summary
    assert "regressions (STT right, Mozhi broke it): 0" in summary
    assert "0 of 6 STT-error regions carry a hand-set category_reviewed" in summary


def test_summary_with_no_phonetic_regions_reports_na_not_a_crash():
    r = ev.evaluate_pair("x", "a b c", "a b d", corrector=fake_corrector({}))
    r.rows[0].category_suggested = ev.NON_PHONETIC
    assert "PHONETIC-SUBSTITUTION CATCH RATE: 0/0 = n/a" in ev.summarize([r])


def test_summary_uses_reviewed_categories():
    r = ev.evaluate_pair("x", "check the read array", "check the RAID array", corrector=fake_corrector({"read": "RAID"}))
    assert "PHONETIC-SUBSTITUTION CATCH RATE: 1/1 = 100.0%" in ev.summarize([r])
    r.rows[0].category_reviewed = ev.NON_PHONETIC
    summary = ev.summarize([r])
    assert "PHONETIC-SUBSTITUTION CATCH RATE: 0/0 = n/a" in summary
    assert "1 of 1 STT-error regions carry a hand-set category_reviewed" in summary


# --- main() end to end on temp directories ---------------------------------------------------------------------------------

def run_main(tmp_path, *extra):
    return ev.main([
        "--audio-dir", str(tmp_path / "audio"), "--stt-dir", str(tmp_path / "stt"),
        "--csv", str(tmp_path / "out" / "eval.csv"), "--summary", str(tmp_path / "out" / "summary.txt"), *extra,
    ])


def setup_project(tmp_path):
    audio, stt = tmp_path / "audio", tmp_path / "stt"
    audio.mkdir()
    stt.mkdir()
    (audio / "test_audio2.m4a").write_bytes(b"")
    (audio / "test_audio2.expected.txt").write_text(AUDIO2_EXPECTED + "\n")
    cache = stt / "test_audio2.json"
    cache.write_text(json.dumps({"transcript": AUDIO2_STT}))
    now = os.stat(audio / "test_audio2.m4a").st_mtime
    os.utime(cache, (now + 10, now + 10))


def test_main_end_to_end_writes_csv_and_summary(tmp_path, capsys):
    setup_project(tmp_path)
    assert run_main(tmp_path) == 0
    with open(tmp_path / "out" / "eval.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["expected"], r["stt"], r["fixed"]) for r in rows] == [
        ("LAN", "land", "yes"), ("WAN link", "one link", "yes"), ("RAID", "read", "yes"), ("keeps", "is", "no")]
    summary = (tmp_path / "out" / "summary.txt").read_text()
    assert "raw WER:        28.6%" in summary and "corrected WER:  7.1%" in summary
    assert "PHONETIC-SUBSTITUTION CATCH RATE: 3/3 = 100.0%" in summary
    assert "[STT: cached]" in summary
    assert "Wrote" in capsys.readouterr().out


def test_main_preserves_hand_set_categories_on_rerun(tmp_path):
    setup_project(tmp_path)
    assert run_main(tmp_path) == 0
    csv_path = tmp_path / "out" / "eval.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["stt"] == "is":
            r["category_reviewed"] = "omission"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev.CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    assert run_main(tmp_path) == 0
    with open(csv_path, newline="") as f:
        again = {r["stt"]: r for r in csv.DictReader(f)}
    assert again["is"]["category_reviewed"] == "omission"
    assert again["is"]["category_suggested"] == ev.NON_PHONETIC  # suggestion untouched
    assert "1 of 4 STT-error regions carry a hand-set category_reviewed" in (tmp_path / "out" / "summary.txt").read_text()


def test_main_skips_unpaired_files_and_fails_when_nothing_is_evaluable(tmp_path, capsys):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "lonely.m4a").write_bytes(b"")
    assert run_main(tmp_path) == 1
    out = capsys.readouterr().out
    assert "no lonely.expected.txt for lonely.m4a" in out and "No complete" in out


def test_main_missing_audio_dir_is_an_error(tmp_path):
    assert run_main(tmp_path) == 1


def test_main_skips_a_file_with_an_empty_expected_transcript(tmp_path, capsys):
    setup_project(tmp_path)
    (tmp_path / "audio" / "test_audio2.expected.txt").write_text("   \n")
    assert run_main(tmp_path) == 1
    assert "skipped" in capsys.readouterr().out
