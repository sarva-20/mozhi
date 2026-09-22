"""
Tests for streamlit_app.py through Streamlit's own script runner (AppTest), which
executes the real script top to bottom without a browser. Whisper is replaced by a
fake Transcriber so the suite stays fast; one opt-in test runs the real model:

    MOZHI_REAL_WHISPER=1 pytest tests/test_streamlit_app.py -k real_whisper

Not covered here: the st.audio_input microphone widget, which AppTest cannot drive.
The recording path differs from Upload only in where the bytes come from (both end in
demo.core.run_pipeline, covered in tests/test_demo_core.py).
"""
import os
import re

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from scripts import evaluate_accuracy

RECORD, UPLOAD, SAMPLES, TYPE = "🎤 Record", "⬆️ Upload", "🎧 Samples", "⌨️ Type text"


class FakeTranscriber:
    """Stands in for evaluate_accuracy.Transcriber. Class attributes are set per test."""
    transcript = "The land is fine, but the one link to the read array is dropping."
    load_error: Exception | None = None
    call_error: Exception | None = None
    calls = 0

    def __init__(self, model_name="base.en"):
        self.model_name = model_name

    def load(self):
        if FakeTranscriber.load_error:
            raise FakeTranscriber.load_error

    def __call__(self, audio_path):
        FakeTranscriber.calls += 1
        if FakeTranscriber.call_error:
            raise FakeTranscriber.call_error
        return {"transcript": FakeTranscriber.transcript, "transcription_time_s": 0.4, "audio_duration_s": 5.9}


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch):
    FakeTranscriber.transcript = "The land is fine, but the one link to the read array is dropping."
    FakeTranscriber.load_error = FakeTranscriber.call_error = None
    FakeTranscriber.calls = 0
    st.cache_resource.clear()  # the model/pipeline caches are process-wide; isolate every test
    monkeypatch.setattr(evaluate_accuracy, "Transcriber", FakeTranscriber)
    yield
    st.cache_resource.clear()


def new_app() -> AppTest:
    return AppTest.from_file("streamlit_app.py", default_timeout=60).run()


def switch(at: AppTest, mode: str) -> AppTest:
    return at.button_group[0].set_value(mode).run()


def click(at: AppTest, label: str) -> AppTest:
    return next(b for b in at.button if b.label == label).click().run()


def html_of(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def hero_ms(at: AppTest) -> float:
    return float(re.search(r'mz-hero-value">([\d.]+)<span>', html_of(at)).group(1))


def box(at: AppTest, title: str) -> str:
    return re.search(re.escape(title) + r"</div>(.*?)</div>", html_of(at), re.S).group(1)


def no_exceptions(at: AppTest):
    assert list(at.exception) == [], [e.value for e in at.exception]


# --- first load ------------------------------------------------------------------------------------------

def test_app_loads_cleanly_in_record_mode():
    at = new_app()
    no_exceptions(at)
    assert not list(at.error) and not list(at.warning)
    assert 'class="mz-title">Mozhi' in html_of(at)
    assert at.button_group[0].value == RECORD
    assert "as soon as the audio is in" in " ".join(c.value for c in at.caption)


def test_sidebar_states_the_facts_a_reviewer_would_ask_about():
    sidebar = " ".join(m.value for m in new_app().sidebar.markdown)
    assert "319 terms" in sidebar and "0.88" in sidebar and "No model inference" in sidebar
    assert "base.en" in sidebar


def test_all_four_input_modes_are_offered():
    # AppTest reports the display labels; the browser renders the emoji prefixes as icons.
    assert list(new_app().button_group[0].options) == ["Record", "Upload", "Samples", "Type text"]


# --- type text -------------------------------------------------------------------------------------------------

def test_type_text_flow_shows_latency_highlight_and_match_table():
    at = switch(new_app(), TYPE)
    at = click(at, "Correct text")
    no_exceptions(at)
    assert hero_ms(at) > 0
    assert 'mz-removed">sea pu</mark>' in box(at, "Raw STT transcript")
    corrected = box(at, "Corrected by Mozhi")
    assert 'mz-added">CPU</mark>' in corrected and 'mz-added">Kubernetes</mark>' in corrected
    table = at.dataframe[0].value
    assert list(table.columns) == ["#", "Original", "Replacement", "Method", "Score", "Changed"]
    assert table.loc[table["Original"] == "sea pu", "Method"].item() == "acronym_map"


def test_text_mode_shows_word_counts_not_meaningless_whisper_tiles():
    at = click(switch(new_app(), TYPE), "Correct text")
    labels = [m.label for m in at.metric]
    assert labels == ["Words in transcript", "Matches found", "Words corrected"]


def test_empty_text_is_a_friendly_warning_not_an_error():
    at = switch(new_app(), TYPE)
    at.text_area[0].set_value("   ").run()
    at = click(at, "Correct text")
    no_exceptions(at)
    assert any("Type or paste" in w.value for w in at.warning) and not list(at.error)


def test_typed_html_is_escaped_in_both_transcript_boxes():
    at = switch(new_app(), TYPE)
    at.text_area[0].set_value("<script>alert(1)</script> check the sea pu").run()
    at = click(at, "Correct text")
    raw, corrected = box(at, "Raw STT transcript"), box(at, "Corrected by Mozhi")
    assert "<script>" not in raw + corrected and "&lt;script&gt;" in raw and "&lt;script&gt;" in corrected


def test_no_matches_shows_an_info_message_and_no_table():
    at = switch(new_app(), TYPE)
    at.text_area[0].set_value("the weather today is quite pleasant").run()
    at = click(at, "Correct text")
    assert any("No domain terms needed correcting" in i.value for i in at.info)
    assert len(at.dataframe) == 0
    assert "Mozhi made no changes" in " ".join(c.value for c in at.caption)


def test_latency_benchmark_button_reports_the_spread():
    at = click(switch(new_app(), TYPE), "Correct text")
    at = click(at, "Run 50 corrections")
    no_exceptions(at)
    stats = {m.label: m.value for m in at.metric if m.label in ("min", "median", "mean", "p95", "p99")}
    assert set(stats) == {"min", "median", "mean", "p95", "p99"} and all(v.endswith(" ms") for v in stats.values())


# --- samples (fake Whisper) --------------------------------------------------------------------------------------

def test_samples_flow_transcribes_corrects_and_shows_what_was_said():
    at = switch(new_app(), SAMPLES)
    # data/volunteer_audio now also holds the 30 staged volunteer recordings
    # (vol01_crud .. vol30_json); this only pins the two hand-curated ones
    # this test actually exercises, not the full, growing set.
    options = list(at.selectbox[0].options)
    assert options[:2] == ["test_audio1", "test_audio2"] and len(options) == 32
    at.selectbox[0].select("test_audio2").run()
    at = click(at, "Transcribe and correct")
    no_exceptions(at)
    assert FakeTranscriber.calls == 1
    assert "The <mark" in box(at, "Raw STT transcript") and hero_ms(at) > 0
    assert "LAN</mark> is fine" in box(at, "Corrected by Mozhi") and "WAN link</mark>" in box(at, "Corrected by Mozhi")
    assert {m.label for m in at.metric} == {"Whisper transcription", "Audio length", "Words corrected"}
    assert "What was actually said" in " ".join(c.value for c in at.caption)
    assert len(at.dataframe[0].value) == 3


def test_selecting_an_mp3_sample_does_not_crash_the_audio_player():
    """Regression guard: st.audio(data, format=None) crashed
    (AttributeError: 'NoneType' object has no attribute 'encode') the first time a
    non-.m4a sample was picked - every sample used to be .m4a, so this path was
    never exercised until the volunteer_audio staging added .mp3 recordings
    (vol21_ids .. vol30_json). Selecting one must render the audio player cleanly,
    not just avoid crashing on "Transcribe and correct"."""
    at = switch(new_app(), SAMPLES)
    assert "vol21_ids" in list(at.selectbox[0].options)
    at = at.selectbox[0].select("vol21_ids").run()
    no_exceptions(at)  # AppTest has no element type for st.audio; absence of a crash IS the check


def test_a_finished_result_is_not_recomputed_by_unrelated_reruns():
    at = click(switch(new_app(), SAMPLES), "Transcribe and correct")
    first = FakeTranscriber.calls
    at.run()  # any widget interaction reruns the script
    assert FakeTranscriber.calls == first


def test_switching_mode_hides_the_previous_modes_result():
    at = click(switch(new_app(), SAMPLES), "Transcribe and correct")
    assert 'mz-hero-value">' in html_of(at)  # the element, not the stylesheet rule of the same name
    at = switch(at, TYPE)
    assert 'mz-hero-value">' not in html_of(at)


# --- failures never surface as tracebacks -----------------------------------------------------------------------------

def test_stt_failure_shows_a_friendly_error_with_collapsed_details():
    FakeTranscriber.call_error = RuntimeError("decoder exploded")
    at = click(switch(new_app(), SAMPLES), "Transcribe and correct")
    no_exceptions(at)
    assert any("Transcription failed" in e.value for e in at.error)
    assert any(e.label == "Technical details" for e in at.expander)
    assert "RuntimeError: decoder exploded" in " ".join(c.value for c in at.code)
    assert "Traceback" not in html_of(at)


def test_empty_transcript_is_a_no_speech_warning():
    FakeTranscriber.transcript = "   "
    at = click(switch(new_app(), SAMPLES), "Transcribe and correct")
    no_exceptions(at)
    assert any("No speech was detected" in w.value for w in at.warning) and not list(at.error)


def test_model_load_failure_warns_in_audio_modes_but_type_text_still_works():
    FakeTranscriber.load_error = OSError("no network")
    at = new_app()
    no_exceptions(at)
    assert any("Speech recognition is unavailable" in w.value and "OSError" in w.value for w in at.warning)
    assert any(b.label == "Retry loading the speech model" for b in at.button)
    at = switch(at, SAMPLES)
    assert next(b for b in at.button if b.label == "Transcribe and correct").disabled
    at = switch(at, TYPE)
    assert not list(at.warning)
    at = click(at, "Correct text")
    no_exceptions(at)
    assert hero_ms(at) > 0


def test_missing_faster_whisper_gives_an_install_hint_and_no_crash():
    FakeTranscriber.load_error = ImportError("No module named 'faster_whisper'")
    at = new_app()
    no_exceptions(at)
    assert any("pip install -r requirements-demo.txt" in w.value for w in at.warning)


def test_retry_button_recovers_once_the_model_can_load():
    FakeTranscriber.load_error = OSError("offline")
    at = new_app()
    FakeTranscriber.load_error = None  # network is back
    at = click(at, "Retry loading the speech model")
    no_exceptions(at)
    assert not list(at.warning)


# --- opt-in: the real Whisper model through the real script ---------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("MOZHI_REAL_WHISPER") != "1", reason="set MOZHI_REAL_WHISPER=1 to run the real Whisper model")
def test_real_whisper_end_to_end_on_a_recorded_sample(monkeypatch):
    monkeypatch.undo()  # drop the FakeTranscriber patch: use the real one
    st.cache_resource.clear()
    at = AppTest.from_file("streamlit_app.py", default_timeout=180).run()
    at = switch(at, SAMPLES)
    at.selectbox[0].select("test_audio2").run()
    at = click(at, "Transcribe and correct")
    no_exceptions(at)
    assert not list(at.error) and not list(at.warning)
    assert "LAN</mark>" in box(at, "Corrected by Mozhi") and "WAN link</mark>" in box(at, "Corrected by Mozhi")
    assert hero_ms(at) > 0
