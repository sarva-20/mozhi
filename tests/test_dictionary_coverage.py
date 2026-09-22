"""
Tests for scripts/dictionary_coverage.py: the corruption operators, variant generation,
classification, aggregation, reports, and a real-pipeline guard on the canonical forms.

The full 319-term run (~20 s with workers) is a script, not a test. What is tested here is the
LOGIC, so a change to an operator or to the scoring convention fails loudly instead of silently
moving the headline number.
"""
import csv
import random
from types import SimpleNamespace

import pytest

from app.dictionary import get_terms
from app.pipeline import correct_text
from scripts import dictionary_coverage as dc

ALL_TERMS = [t["term"] for t in get_terms()]
ENTRIES = {t["term"]: t for t in get_terms()}


def variants_of(term, all_terms=None):
    return dc.generate_variants(term, all_terms if all_terms is not None else ALL_TERMS)


# --- word-level operators ----------------------------------------------------------------------------

def test_split_words_splits_on_spaces_and_hyphens():
    assert dc.split_words("two-factor authentication") == ["two", "factor", "authentication"]
    assert dc.split_words("Wi-Fi") == ["Wi", "Fi"]


def test_vowel_swap_replaces_each_vowel_with_each_other_vowel():
    assert dc.op_vowel_swap("cat") == ["cet", "cit", "cot", "cut"]


def test_consonant_swap_uses_the_confusion_table_one_letter_at_a_time():
    assert dc.op_consonant_swap("bad") == ["pad", "bat"]
    assert dc.op_consonant_swap("router")  # r<->l and t<->d apply
    assert "louter" in dc.op_consonant_swap("router")


def test_double_consonant_doubles_inner_consonants_and_undoubles_pairs():
    assert dc.op_double_consonant("router") == ["routter"]
    assert dc.op_double_consonant("dokker") == ["doker"]  # undouble; never a triple like "dokkker"
    assert all(v[0] == "r" and v[1] != "r" for v in dc.op_double_consonant("router"))  # never doubles a vowel or the first letter


def test_drop_last_only_for_words_of_five_or_more_letters():
    assert dc.op_drop_last("cache") == ["cach"] and dc.op_drop_last("cat") == []


def test_add_suffix_adds_a_stray_consonant_to_short_words_only_and_never_a_plural():
    assert dc.op_add_suffix("lan") == ["land", "lant"]
    assert dc.op_add_suffix("router") == [] and dc.op_add_suffix("cache") == []
    assert not any(v.endswith("s") for v in dc.op_add_suffix("lan") + dc.op_add_suffix("router"))


def test_respell_applies_orthographic_rules_at_every_occurrence():
    assert "fishing" in dc.op_respell("phishing")
    assert "rayd" in dc.op_respell("raid")


# --- phrase-level operators --------------------------------------------------------------------------------

def test_split_word_cuts_long_words_leaving_at_least_three_letters_each_side():
    out = dc.op_split_word(["firewall"])
    assert "fire wall" in out
    assert all(len(p) >= 3 for v in out for p in v.split())
    assert dc.op_split_word(["router"]) == [] and dc.op_split_word(["ab12cdefgh"]) == []


def test_merge_swap_and_drop_filler():
    assert dc.op_merge_words(["load", "balancer"]) == ["loadbalancer"] and dc.op_merge_words(["router"]) == []
    assert dc.op_swap_words(["ip", "address"]) == ["address ip"] and dc.op_swap_words(["a", "a"]) == []
    assert dc.op_drop_filler(["living", "off", "the", "land"]) == ["living off land"]
    assert dc.op_drop_filler(["pass", "the"]) == []  # a 2-word phrase keeps its filler


# --- spelling and spoken digits -----------------------------------------------------------------------------

def test_spell_letters_canonical_and_multi_token_w():
    assert dc.spell_letters("SSD") == "ess ess dee"
    assert dc.spell_letters("wan") == "double you ay en"
    assert dc.spell_letters("A1") == ""  # a non-letter cannot be spelled


def test_spell_letters_alt_always_differs_and_is_deterministic():
    for word in ("SSD", "VPN", "OS", "NVMe", "API"):
        canonical, alt = dc.spell_letters(word, word), dc.spell_letters(word, word, alt=True)
        assert alt and alt != canonical
        assert alt == dc.spell_letters(word, word, alt=True)


@pytest.mark.parametrize("term,plain,spelled", [
    ("IPv4", "ipv four", "eye pea vee four"),
    ("WPA2", "wpa two", "double you pea ay two"),
    ("C2", "c two", "see two"),
    ("X.509", "x dot five oh nine", "ex dot five oh nine"),
    ("802.11ax", "eight oh two dot one one ax", "eight oh two dot one one ay ex"),
])
def test_spoken_digits(term, plain, spelled):
    assert dc.spoken_digits(term, spelled=False) == plain
    assert dc.spoken_digits(term, spelled=True) == spelled


# --- generate_variants -------------------------------------------------------------------------------------------------

def test_canonical_control_is_always_first_and_unchanged():
    for term in ("router", "IP address", "Wi-Fi", "SSD"):
        first = variants_of(term)[0]
        assert (first.family, first.operator, first.text) == (dc.CANONICAL, "canonical", term)


def test_generation_is_deterministic_and_independent_of_term_order():
    shuffled = ALL_TERMS[:]
    random.Random(7).shuffle(shuffled)
    for term in ("router", "two-factor authentication", "LAN", "IPv4"):
        assert variants_of(term) == variants_of(term)
        assert variants_of(term) == dc.generate_variants(term, shuffled)


def test_no_variant_equals_the_canonical_or_another_dictionary_term_or_a_duplicate():
    taken = {t.casefold() for t in ALL_TERMS}
    for term in ALL_TERMS[:120]:
        vs = variants_of(term)[1:]
        norms = [" ".join(dc.split_words(v.text)).casefold() for v in vs]
        assert len(norms) == len(set(norms)), term
        assert dc.norm_words(term) not in [dc.norm_words(v.text) for v in vs], term
        assert not any(v.text.casefold() in taken or n in taken for v, n in zip(vs, norms)), term


def test_a_variant_that_is_another_term_is_dropped():
    # "core" is vowel_swap("care"); with "core" in the dictionary it must not be offered as a corruption of "care".
    assert "core" not in [v.text for v in dc.generate_variants("care", ["care", "core"])]
    assert "core" in [v.text for v in dc.generate_variants("care", ["care"])]


def test_synthetic_variants_are_capped_and_spread_over_operators():
    vs = [v for v in variants_of("living off the land") if v.family == dc.SYNTHETIC]
    assert len(vs) == dc.SYNTHETIC_CAP
    assert len({v.operator for v in vs}) >= 5


def test_no_variant_keeps_a_hyphen_since_stt_output_has_none():
    for term in ("two-factor authentication", "Wi-Fi", "pass-the-hash", "use-after-free"):
        assert not any("-" in v.text for v in variants_of(term)[1:])


def test_families_for_a_letters_only_acronym():
    fams = {(v.family, v.operator): v.text for v in variants_of("SSD")}
    assert fams[(dc.ACRONYM_MAP, "acronym_map")] == "ess ess dee"           # from ACRONYM_PHONETIC_MAP
    assert fams[(dc.SPELLED, "spelled_alt")] != "ess ess dee"               # canonical spelling is the map entry, deduplicated
    assert (dc.SPELLED, "spelled") not in fams


def test_phrase_alias_family_comes_from_the_alias_table():
    assert (dc.PHRASE_ALIAS, "phrase_alias", "one link") in [(v.family, v.operator, v.text) for v in variants_of("WAN link")]


def test_spell_token_family_spells_the_acronym_inside_a_phrase():
    tokens = {v.text for v in variants_of("IP address") if v.family == dc.SPELLED_TOKEN}
    assert tokens == {"eye pea address"}


def test_digit_terms_get_spoken_digit_variants():
    ops = {v.operator: v.text for v in variants_of("IPv4") if v.family == dc.SYNTHETIC}
    assert ops == {"spoken_digits": "ipv four", "spoken_digits_spelled": "eye pea vee four"}


def test_wordlike_acronyms_get_word_corruptions_and_letter_name_acronyms_do_not():
    lan = {v.text for v in variants_of("LAN") if v.operator == "add_suffix"}
    assert "land" in lan
    assert not [v for v in variants_of("SSD") if v.family == dc.SYNTHETIC]


def test_every_dictionary_term_gets_at_least_one_evaluated_variant():
    # A term with only its control would be silently untested.
    for term in ALL_TERMS:
        assert [v for v in variants_of(term) if v.family not in dc.TAUTOLOGICAL], term


def test_total_variant_count_is_in_the_expected_range():
    n = sum(len(variants_of(t)) for t in ALL_TERMS)
    assert 3000 <= n <= 3400  # ~3.2k; a big swing means an operator changed


# --- classify --------------------------------------------------------------------------------------------------------------

CARRIER_TERM = "please check the {} now"


@pytest.mark.parametrize("term,variant,output,outcome", [
    ("two-factor authentication", "two factor uuthentication", "please check the two factor authentication now", "recovered"),
    ("LAN", "land", "please check the LAN now", "recovered"),
    ("Kubernetes", "kubernetees", "please check the kubernetes now", "recovered"),  # case differs only
    ("router", "louter", "please check the louter now", "missed"),
    ("router", "louter", "please check the Louter now", "missed"),                # a case-only change is not a fix
    ("web application firewall", "web applicatoin firewall", "please check the web application firewall firewall now", "garbled"),
    ("SQL injection", "sql injectoin", "please check the SQL SQL injection now", "garbled"),
    ("subnet mask", "subnet nask", "please check the subnet NAS now", "wrong"),
    ("router", "louter", "please check the switch now", "wrong"),
])
def test_classify(term, variant, output, outcome):
    sent = CARRIER_TERM.format(variant)
    assert dc.classify(CARRIER_TERM.format(term), sent, output, term) == outcome


def test_a_swallowed_carrier_word_is_not_a_recovery():
    # Mozhi merged the carrier's "now" into the match. The term is present but the sentence is not
    # right (collateral damage), so it is "garbled", never "recovered".
    assert dc.classify(CARRIER_TERM.format("authentication"), CARRIER_TERM.format("authenticatiom"),
                       "please check the authentication", "authentication") == "garbled"


# --- evaluate (fake corrector) --------------------------------------------------------------------------------------------------

def fake(mapping):
    def corrector(sentence):
        out = sentence
        edits = []
        for original, repl in mapping.items():
            if original in out:
                out = out.replace(original, repl)
                edits.append((original, repl))
        return out, edits
    return corrector


def test_evaluate_builds_results_with_outcome_exact_flag_and_edits():
    vs = [dc.Variant("LAN", dc.SYNTHETIC, "add_suffix", "land"),
          dc.Variant("two-factor authentication", dc.SYNTHETIC, "vowel_swap", "two factor uuthentication"),
          dc.Variant("router", dc.SYNTHETIC, "consonant_swap", "louter")]
    results = dc.evaluate(vs, ENTRIES, corrector=fake({"land": "LAN", "uuthentication": "authentication"}))
    lan, twofa, router = results
    assert (lan.outcome, lan.exact, lan.edits) == ("recovered", True, "land -> LAN")
    assert (twofa.outcome, twofa.exact) == ("recovered", False)  # right words, hyphen missing: recovered but not byte-exact
    assert (router.outcome, router.exact, router.edits) == ("missed", False, "")
    assert (lan.category, lan.shape, lan.n_words) == (ENTRIES["LAN"]["category"], "acronym", 1)
    assert twofa.shape == "phrase" and twofa.n_words == 3


def test_evaluate_uses_the_given_carrier():
    seen = []
    dc.evaluate([dc.Variant("router", dc.SYNTHETIC, "x", "louter")], ENTRIES, carrier="say {} twice",
                corrector=lambda s: (seen.append(s) or s, []))
    assert seen == ["say louter twice"]


# --- aggregation and reports ---------------------------------------------------------------------------------------------------

def result(term, family, operator, outcome, edits="", exact=None, variant="v", category="general", shape="single_word", n_words=1):
    return dc.Result(term, category, shape, n_words, family, operator, variant, "out", outcome, edits,
                     exact if exact is not None else outcome == "recovered")


def sample_results():
    return [
        result("alpha", dc.CANONICAL, "canonical", "recovered"),
        result("alpha", dc.ACRONYM_MAP, "acronym_map", "recovered"),
        result("alpha", dc.SYNTHETIC, "vowel_swap", "recovered"),
        result("alpha", dc.SYNTHETIC, "consonant_swap", "missed"),
        result("alpha", dc.SPELLED, "spelled", "recovered"),
        result("beta", dc.CANONICAL, "canonical", "garbled", "x -> beta beta"),
        result("beta", dc.SYNTHETIC, "vowel_swap", "wrong", "side -> SSID"),
        result("beta", dc.SYNTHETIC, "consonant_swap", "wrong", "side -> SSID"),
        result("beta", dc.SYNTHETIC, "respell", "garbled", "injection -> SQL injection"),
    ]


def test_term_rows_exclude_tautological_families_from_the_evaluated_rate():
    rows = {t.term: t for t in dc.term_rows(sample_results())}
    a, b = rows["alpha"], rows["beta"]
    assert (a.evaluated_n, a.evaluated_caught) == (3, 2)   # 2 synthetic + 1 spelled; the control and acronym_map are excluded
    assert (a.synthetic_n, a.synthetic_caught, a.synthetic_rate) == (2, 1, 0.5)
    assert a.canonical_ok and not b.canonical_ok
    assert (b.synthetic_rate, b.evaluated_rate) == (0.0, 0.0)
    assert a.failing_operators == "consonant_swap" and b.failing_operators == "consonant_swap, respell, vowel_swap"


def test_a_term_with_no_synthetic_variants_has_no_synthetic_rate():
    (row,) = dc.term_rows([result("SSD", dc.CANONICAL, "canonical", "recovered"), result("SSD", dc.SPELLED, "spelled", "missed")])
    assert row.synthetic_rate is None and row.evaluated_rate == 0.0


def test_top_replacements_counts_pairs_and_skips_the_intended_term():
    top = dc.top_replacements(sample_results())
    assert top[0] == (("side", "SSID"), 2, 1)
    assert (("injection", "SQL injection"), 1, 1) in top
    assert ("x", "beta beta") in [k for k, _, _ in top]
    # an edit that produces the intended term is not a "bad replacement"
    assert dc.top_replacements([result("LAN", dc.SYNTHETIC, "add_suffix", "garbled", "land -> LAN")]) == []


def test_summary_states_the_synthetic_caveat_and_the_headline_numbers():
    summary = dc.summarize(sample_results(), n_terms=2, carrier=dc.CARRIER, elapsed_s=1.0)
    assert "SYNTHETIC" in summary and "NOT real STT errors" not in summary  # the title says it; the body explains it
    assert "not real-world" in summary and "EXCLUDED" in summary
    assert "synthetic corruptions, micro:  1/5 = 20.0%" in summary          # alpha 1/2 + beta 0/3
    assert "CANONICAL FORM ALTERED BY MOZHI (1 terms)" in summary and "'beta'" in summary
    assert "NEVER RECOVERED in any synthetic variant (1 terms)" in summary
    assert "'side' -> 'SSID'" in summary


def test_hyphen_report_compares_hyphenated_and_plain_phrases_and_estimates_the_fix():
    rs = [
        result("power-on self-test", dc.SYNTHETIC, "vowel_swap", "missed", variant="power on solf test", shape="phrase", n_words=3),
        result("power-on self-test", dc.SYNTHETIC, "respell", "recovered", variant="power on self tesst", shape="phrase", n_words=3),
        result("memory leak", dc.SYNTHETIC, "vowel_swap", "recovered", variant="memory lek", shape="phrase", n_words=2),
        result("memory leak", dc.SYNTHETIC, "respell", "missed", variant="memory leek", shape="phrase", n_words=2),
    ]
    text = "\n".join(dc.hyphen_report(rs))
    assert "hyphenated (1 terms): 1/2 = 50.0%" in text and "plain (1 terms): 1/2 = 50.0%" in text
    assert "missed hyphenated variants: 1;" in text and "of 1 (scoring only" in text
    assert dc.hyphen_report([result("a", dc.SYNTHETIC, "x", "missed", shape="phrase")])[-1] == "  not enough data"


def test_write_csvs_columns_and_exact_flag(tmp_path):
    dc.write_csvs(sample_results(), tmp_path / "v.csv", tmp_path / "t.csv")
    with open(tmp_path / "v.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == dc.VARIANT_FIELDS and len(rows) == 9
    assert {r["exact"] for r in rows} == {"yes", "no"}
    with open(tmp_path / "t.csv", newline="") as f:
        terms = list(csv.DictReader(f))
    assert list(terms[0]) == dc.TERM_FIELDS
    assert [t["term"] for t in terms] == ["beta", "alpha"]  # worst synthetic rate first
    assert {t["term"]: t["canonical_ok"] for t in terms} == {"alpha": "yes", "beta": "NO"}


# --- the real pipeline ---------------------------------------------------------------------------------------------------------------

def test_carrier_is_inert():
    for word in dc.CARRIER.replace("{}", "zzqx").split():
        assert correct_text(word).matches == [], word
    assert correct_text(dc.CARRIER.format("zzqx")).matches == []


def test_real_pipeline_examples_through_evaluate():
    vs = [dc.Variant("LAN", dc.SYNTHETIC, "add_suffix", "land"),
          dc.Variant("Kubernetes", dc.SYNTHETIC, "vowel_swap", "kubernetees"),
          dc.Variant("two-factor authentication", dc.SYNTHETIC, "vowel_swap", "two factor uuthentication"),
          dc.Variant("subnet mask", dc.SYNTHETIC, "swap_words", "mask subnet"),
          dc.Variant("SSD", dc.ACRONYM_MAP, "acronym_map", "ess ess dee"),
          dc.Variant("VLAN", dc.SPELLED, "spelled", "vee ell ay en")]
    got = {(r.term, r.variant): (r.outcome, r.exact) for r in dc.evaluate(vs, ENTRIES)}
    assert got == {
        ("LAN", "land"): ("recovered", True),
        ("Kubernetes", "kubernetees"): ("recovered", True),
        ("two-factor authentication", "two factor uuthentication"): ("recovered", False),  # hyphen lost: the word-level scoring decision
        ("subnet mask", "mask subnet"): ("missed", False),
        ("SSD", "ess ess dee"): ("recovered", True),
        ("VLAN", "vee ell ay en"): ("recovered", True),
    }


def test_all_canonical_forms_survive():
    """Regression guard for a DP fragmentation defect the coverage run found and
    app/pipeline.py has since fixed: six correctly-spoken multi-word terms
    ("SQL injection", "TCP handshake", "configuration management database",
    "living off the land", "server-side request forgery", "web application
    firewall") got a sub-word split off into its own match whenever that
    sub-word was ALSO a standalone term ("SQL", "database", "firewall", ...)
    or, for the hyphenated one, an unrelated term ("side" -> SSID) - because
    the DP maximized raw score sum, and two matches summing above the one
    correct exact match always won, producing literal duplicate words
    ("SQL SQL injection"). Fixed by comparing (exact_tokens_covered, score)
    lexicographically in the DP, plus blocking "side" -> SSID in
    NEVER_MATCH_PAIRS for the hyphenated case, which never scores a full 1.0
    exact match and so needed the direct block. If this ever fails again,
    check both mechanisms before assuming it is the same defect."""
    controls = [v for t in ALL_TERMS for v in variants_of(t)[:1]]
    results = dc.evaluate(controls, ENTRIES, workers=4)
    broken = {r.term for r in results if r.outcome != "recovered"}
    assert broken == set(), f"canonical defects: {sorted(broken)}"


# --- the script end to end ---------------------------------------------------------------------------------------------------------------

def test_main_writes_three_reports_for_selected_terms(tmp_path, capsys):
    assert dc.main(["--terms", "router,LAN", "--workers", "1", "--out-dir", str(tmp_path)]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "dictionary_coverage_variants.csv", "dictionary_coverage_terms.csv", "dictionary_coverage_summary.txt"}
    with open(tmp_path / "dictionary_coverage_terms.csv", newline="") as f:
        assert {r["term"] for r in csv.DictReader(f)} == {"router", "LAN"}
    assert "SYNTHETIC" in (tmp_path / "dictionary_coverage_summary.txt").read_text()
    assert "Wrote" in capsys.readouterr().out


def test_main_rejects_unknown_terms(tmp_path, capsys):
    assert dc.main(["--terms", "not-a-term", "--out-dir", str(tmp_path)]) == 1
    assert "not in the dictionary" in capsys.readouterr().out and not list(tmp_path.iterdir())
