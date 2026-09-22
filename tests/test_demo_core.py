"""
Tests for demo/core.py, the non-UI logic behind streamlit_app.py: highlight diff,
HTML escaping, match tables, latency stats, silence detection, and the audio ->
transcript -> correction flow. Transcribers and correctors are injected fakes, so
nothing here needs Whisper or Streamlit.
"""
import io
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.pipeline import correct_text
from demo import core


# --- helpers ---------------------------------------------------------------------------------------

def make_wav(samples, rate=16000, width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}{'h' if width == 2 else 'B'}", *samples))
    return buf.getvalue()


SPEECH_LIKE = make_wav([int(12000 * (1 if i % 40 < 20 else -1)) for i in range(16000)])  # loud square wave
SILENCE = make_wav([0] * 16000)


def match(text, original, replacement, method="fuzzy", score=0.9, breakdown=None):
    start = text.index(original)
    return SimpleNamespace(start=start, end=start + len(original), original=original, replacement=replacement,
                           method=method, score=score, score_breakdown=breakdown or {})


class FakeTranscriber:
    """Records the temp file it was handed so tests can check its suffix, bytes and cleanup."""
    def __init__(self, transcript="the sea pu is hot", raises=None, payload=None):
        self.transcript, self.raises, self.payload = transcript, raises, payload
        self.path = None
        self.bytes_seen = None

    def __call__(self, audio_path: Path) -> dict:
        self.path = audio_path
        self.bytes_seen = audio_path.read_bytes()
        if self.raises:
            raise self.raises
        if self.payload is not None:
            return self.payload
        return {"transcript": self.transcript, "transcription_time_s": 0.42, "audio_duration_s": 3.5}


# --- highlight_segments ----------------------------------------------------------------------------

def test_no_matches_is_one_unchanged_segment_on_each_side():
    raw, corrected = core.highlight_segments("nothing to fix", [])
    assert raw == corrected == [core.Segment("nothing to fix", False)]


def test_changed_span_is_flagged_on_both_sides_with_the_right_text():
    text = "check the sea pu now"
    raw, corrected = core.highlight_segments(text, [match(text, "sea pu", "CPU", "acronym_map")])
    assert raw == [core.Segment("check the ", False), core.Segment("sea pu", True), core.Segment(" now", False)]
    assert corrected == [core.Segment("check the ", False), core.Segment("CPU", True), core.Segment(" now", False)]


def test_match_that_changes_nothing_is_not_highlighted():
    text = "verify the ISP outage"
    raw, corrected = core.highlight_segments(text, [match(text, "ISP", "ISP", score=1.0)])
    assert not any(s.changed for s in raw + corrected)


def test_case_only_change_is_highlighted():
    text = "check the ram"
    _, corrected = core.highlight_segments(text, [match(text, "ram", "RAM")])
    assert core.Segment("RAM", True) in corrected


def test_adjacent_and_edge_matches_leave_no_empty_segments():
    text = "read land"
    raw, corrected = core.highlight_segments(text, [match(text, "read", "RAID"), match(text, "land", "LAN")])
    assert [s.text for s in corrected] == ["RAID", " ", "LAN"]
    assert all(s.text for s in raw + corrected)


def test_matches_are_processed_in_reading_order_regardless_of_input_order():
    text = "read the land"
    m1, m2 = match(text, "read", "RAID"), match(text, "land", "LAN")
    assert core.highlight_segments(text, [m2, m1]) == core.highlight_segments(text, [m1, m2])


def test_overlapping_matches_are_skipped_defensively_not_duplicated():
    text = "one link here"
    first = match(text, "one link", "WAN link")
    overlapping = SimpleNamespace(start=4, end=8, original="link", replacement="LINK", method="fuzzy", score=0.9, score_breakdown={})
    raw, corrected = core.highlight_segments(text, [first, overlapping])
    assert "".join(s.text for s in raw) == text
    assert "".join(s.text for s in corrected) == "WAN link here"


@pytest.mark.parametrize("text", [
    "The land is fine, but the one link to the read array is dropping.",
    "please check the sea pu usage and restart the kubernetees container",
    "we saw the ex ess ess attack on the vee ell ay en",
    "Check if there is a leg for you escalate, then move away and verify the ISP outage.",
    "",
])
def test_segments_reproduce_both_texts_for_the_real_pipeline(text):
    result = correct_text(text)
    raw, corrected = core.highlight_segments(result.original_text, result.matches)
    assert "".join(s.text for s in raw) == text
    assert "".join(s.text for s in corrected) == result.corrected_text


# --- segments_to_html --------------------------------------------------------------------------------

def test_html_wraps_changed_segments_in_mark_with_the_class():
    html_out = core.segments_to_html([core.Segment("a ", False), core.Segment("CPU", True)], "mz-added")
    assert html_out == 'a <mark class="mz-added">CPU</mark>'


def test_html_escapes_arbitrary_stt_text_in_both_kinds_of_segment():
    hostile = "<script>alert(1)</script> & <b>"
    out = core.segments_to_html([core.Segment(hostile, False), core.Segment(hostile, True)], "mz-added")
    assert "<script>" not in out and "<b>" not in out
    assert out.count("&lt;script&gt;alert(1)&lt;/script&gt; &amp; &lt;b&gt;") == 2


def test_html_escapes_the_class_name_too():
    assert '"><' not in core.segments_to_html([core.Segment("x", True)], '"><img src=x>')


def test_html_empty_segment_list_is_empty_string():
    assert core.segments_to_html([], "c") == ""


# --- match_rows / breakdown_rows -------------------------------------------------------------------------------

def test_match_rows_order_labels_and_rounding():
    text = "read the ISP and the land"
    matches = [match(text, "land", "LAN", score=0.90812), match(text, "ISP", "ISP", score=1.0), match(text, "read", "RAID", score=0.885)]
    rows = core.match_rows(text, matches)
    assert [r["#"] for r in rows] == [1, 2, 3]
    assert [(r["Original"], r["Replacement"]) for r in rows] == [("read", "RAID"), ("ISP", "ISP"), ("land", "LAN")]
    assert [r["Changed"] for r in rows] == ["yes", "no (already correct)", "yes"]
    assert rows[2]["Score"] == 0.908 and rows[0]["Method"] == "fuzzy"


def test_match_rows_for_the_real_pipeline_include_method_and_score():
    result = correct_text("The land is fine, but the one link to the read array is dropping.")
    rows = core.match_rows(result.original_text, result.matches)
    assert [(r["Original"], r["Replacement"], r["Method"]) for r in rows] == [
        ("land", "LAN", "fuzzy"), ("one link", "WAN link", "phrase_alias"), ("read", "RAID", "fuzzy")]
    assert rows[1]["Score"] == 1.0
    assert set(core.METHOD_DESCRIPTIONS) >= {r["Method"] for r in rows}


def test_breakdown_rows_only_for_fuzzy_matches_with_a_breakdown():
    text = "the land and sea pu"
    breakdown = {"metaphone": 0.9111, "surface": 0.9422, "token_sort": 0.8571, "total": 0.9083}
    rows = core.breakdown_rows(text, [match(text, "land", "LAN", "fuzzy", breakdown=breakdown),
                                      match(text, "sea pu", "CPU", "acronym_map", 1.0)])
    assert rows == [{"Original": "land", "Replacement": "LAN", "Metaphone": 0.911, "Surface (JW)": 0.942,
                     "Token sort": 0.857, "Weighted total": 0.908}]


# --- latency_stats --------------------------------------------------------------------------------------------------

def test_latency_stats_summarizes_each_calls_own_elapsed_ms_after_warmup():
    calls = []
    timings = iter(range(1, 1000))

    def fake(text):
        calls.append(text)
        return SimpleNamespace(elapsed_ms=float(next(timings)))

    stats = core.latency_stats("hello", runs=10, warmup=3, corrector=fake)
    assert len(calls) == 13 and set(calls) == {"hello"}
    # warm-up consumed timings 1..3; measured runs are 4..13
    assert (stats["runs"], stats["min"], stats["max"]) == (10, 4.0, 13.0)
    assert stats["mean"] == pytest.approx(8.5) and stats["median"] == pytest.approx(8.5)
    assert stats["p95"] == 13.0 and stats["p99"] == 13.0


def test_latency_stats_single_run_and_validation():
    stats = core.latency_stats("x", runs=1, warmup=0, corrector=lambda t: SimpleNamespace(elapsed_ms=2.5))
    assert stats["min"] == stats["max"] == stats["p99"] == 2.5
    with pytest.raises(ValueError):
        core.latency_stats("x", runs=0)


def test_latency_stats_real_pipeline_is_ordered_and_positive():
    s = core.latency_stats("the sea pu is hot", runs=15)
    assert 0 < s["min"] <= s["median"] <= s["p95"] <= s["max"]


# --- silence detection ------------------------------------------------------------------------------------------------

def test_peak_fraction_of_silence_and_loud_audio():
    assert core.wav_peak_fraction(SILENCE) == 0.0
    assert core.wav_peak_fraction(SPEECH_LIKE) == pytest.approx(12000 / 32768)


def test_peak_fraction_uses_the_negative_peak_too():
    assert core.wav_peak_fraction(make_wav([0, -32768, 0])) == 1.0


def test_is_silent_threshold_boundary():
    quiet = make_wav([int(core.SILENCE_PEAK_FRACTION * 32768) - 5, 0] * 50)   # just under
    audible = make_wav([int(core.SILENCE_PEAK_FRACTION * 32768) + 5, 0] * 50)  # just over
    assert core.is_silent(quiet) is True and core.is_silent(audible) is False


def test_non_wav_and_unsupported_wav_cannot_be_judged():
    assert core.wav_peak_fraction(b"not a wav file") is None
    assert core.is_silent(b"\x00\x01\x02" * 100) is None
    assert core.is_silent(make_wav([128] * 100, width=1)) is None  # 8-bit PCM is not inspected


def test_wav_with_no_frames_counts_as_silent():
    assert core.wav_peak_fraction(make_wav([])) == 0.0 and core.is_silent(make_wav([])) is True


@pytest.mark.parametrize("name,mime,expected", [
    ("clip.M4A", None, ".m4a"), ("voice.mp3", "audio/mpeg", ".mp3"), (None, "audio/wav", ".wav"),
    ("audio.bin", "audio/ogg", ".ogg"), (None, None, ".wav"), ("noext", "weird/type", ".wav"),
])
def test_audio_suffix(name, mime, expected):
    assert core.audio_suffix(name, mime) == expected


# --- mimetype_for_suffix -------------------------------------------------------------------------------------------------
# Regression guard: st.audio(data, format=None) crashes deep in Streamlit's media file
# manager (AttributeError on None.encode()) instead of falling back to a guess. This was
# only ever exercised for .m4a samples (all samples were .m4a) until the volunteer_audio
# staging added .mp3 recordings, which crashed the live "Samples" flow the first time one
# was picked. Every extension st.audio() might be asked to play must resolve to a real
# mimetype string, never None.

@pytest.mark.parametrize("suffix,expected", [
    (".wav", "audio/wav"), (".mp3", "audio/mpeg"), (".m4a", "audio/mp4"),
    (".ogg", "audio/ogg"), (".flac", "audio/flac"), (".aac", "audio/aac"),
    (".M4A", "audio/mp4"),  # case-insensitive, matches audio_suffix()'s own convention
])
def test_mimetype_for_suffix(suffix, expected):
    assert core.mimetype_for_suffix(suffix) == expected


def test_mimetype_for_suffix_never_returns_none():
    for ext in core.AUDIO_EXTENSIONS:
        assert isinstance(core.mimetype_for_suffix(ext), str) and core.mimetype_for_suffix(ext)


def test_mimetype_for_suffix_falls_back_to_wav_for_unknown_extensions():
    assert core.mimetype_for_suffix(".xyz") == "audio/wav"


# --- samples -----------------------------------------------------------------------------------------------------------

def test_list_samples_returns_paired_recordings_only(tmp_path):
    (tmp_path / "a.m4a").write_bytes(b"x")
    (tmp_path / "a.expected.txt").write_text("hello world\n")
    (tmp_path / "orphan.m4a").write_bytes(b"x")
    samples = core.list_samples(tmp_path)
    assert [(s.name, s.expected, s.audio_path.name) for s in samples] == [("a", "hello world", "a.m4a")]


def test_list_samples_missing_directory_is_empty(tmp_path):
    assert core.list_samples(tmp_path / "nope") == []


def test_list_samples_finds_the_repo_recordings():
    assert {"test_audio1", "test_audio2"} <= {s.name for s in core.list_samples()}


# --- run_pipeline: the happy path ------------------------------------------------------------------------------------------

def test_run_pipeline_success_flows_audio_to_transcript_to_correction():
    t = FakeTranscriber("the sea pu is hot")
    out = core.run_pipeline(SPEECH_LIKE, ".wav", t)
    assert out.ok and out.stage == "done"
    assert out.stt_text == "the sea pu is hot"
    assert out.result.corrected_text == "the CPU is hot"
    assert (out.stt_seconds, out.audio_seconds) == (0.42, 3.5)
    assert out.result.elapsed_ms > 0


def test_run_pipeline_hands_the_transcriber_a_temp_file_with_the_bytes_and_suffix_then_deletes_it():
    t = FakeTranscriber()
    core.run_pipeline(SPEECH_LIKE, ".m4a", t)
    assert t.path.suffix == ".m4a" and t.bytes_seen == SPEECH_LIKE
    assert not t.path.exists()


def test_run_pipeline_strips_transcript_whitespace():
    out = core.run_pipeline(SPEECH_LIKE, ".wav", FakeTranscriber("  the sea pu is hot \n"))
    assert out.stt_text == "the sea pu is hot"


def test_run_pipeline_skips_the_silence_check_for_compressed_audio():
    t = FakeTranscriber()
    out = core.run_pipeline(b"\x00" * 5000, ".m4a", t)  # undecodable here, so not judged silent; the transcriber decides
    assert out.ok and t.path is not None


# --- run_pipeline: failures never raise and never show a traceback -------------------------------------------------------------

@pytest.mark.parametrize("audio", [b"", None, b"tiny"])
def test_no_audio_is_a_warning_and_the_transcriber_is_not_called(audio):
    t = FakeTranscriber()
    out = core.run_pipeline(audio, ".wav", t)
    assert (out.ok, out.stage, out.kind) == (False, "input", "warning")
    assert "No audio" in out.message and t.path is None


def test_oversized_audio_is_rejected_before_transcribing():
    t = FakeTranscriber()
    out = core.run_pipeline(b"x" * (core.MAX_AUDIO_BYTES + 1), ".wav", t)
    assert (out.ok, out.kind) == (False, "warning") and "25 MB" in out.message and t.path is None


def test_silent_recording_is_a_warning_and_the_transcriber_is_not_called():
    t = FakeTranscriber()
    out = core.run_pipeline(SILENCE, ".wav", t)
    assert (out.ok, out.stage, out.kind) == (False, "input", "warning")
    assert "silent" in out.message and t.path is None


def test_empty_transcript_means_no_speech_detected():
    for empty in ("", "   ", None):
        out = core.run_pipeline(SPEECH_LIKE, ".wav", FakeTranscriber(empty))
        assert (out.ok, out.stage, out.kind) == (False, "transcribe", "warning")
        assert "No speech" in out.message
        assert out.stt_seconds == 0.42  # the time spent is still reported


def test_transcriber_returning_a_non_dict_is_handled():
    out = core.run_pipeline(SPEECH_LIKE, ".wav", FakeTranscriber(payload=["not", "a", "dict"]))
    assert (out.ok, out.kind) == (False, "warning") and "No speech" in out.message


def test_missing_speech_library_gives_an_install_hint():
    out = core.run_pipeline(SPEECH_LIKE, ".wav", FakeTranscriber(raises=ImportError("No module named 'faster_whisper'")))
    assert (out.ok, out.stage, out.kind) == (False, "transcribe", "error")
    assert "pip install -r requirements-demo.txt" in out.message and "faster_whisper" in out.detail


def test_undecodable_audio_says_so():
    class InvalidDataError(Exception):
        pass
    out = core.run_pipeline(SPEECH_LIKE, ".m4a", FakeTranscriber(raises=InvalidDataError("Invalid data found when processing input")))
    assert "couldn't be decoded" in out.message and out.detail.startswith("InvalidDataError")


def test_any_other_transcription_failure_gets_a_generic_message_and_one_line_detail():
    out = core.run_pipeline(SPEECH_LIKE, ".wav", FakeTranscriber(raises=RuntimeError("boom " * 200)))
    assert (out.ok, out.stage, out.kind) == (False, "transcribe", "error")
    assert "Transcription failed" in out.message
    assert out.detail.startswith("RuntimeError: boom") and len(out.detail) < 260 and "\n" not in out.detail


@pytest.mark.parametrize("error", [ImportError("x"), RuntimeError("x"), OSError("x")])
def test_temp_file_is_deleted_even_when_the_transcriber_fails(error):
    t = FakeTranscriber(raises=error)
    core.run_pipeline(SPEECH_LIKE, ".wav", t)
    assert t.path is not None and not t.path.exists()


def test_corrector_failure_is_an_error_that_keeps_the_transcript():
    def broken(text):
        raise ValueError("scorer exploded")
    out = core.run_pipeline(SPEECH_LIKE, ".wav", FakeTranscriber("hello there"), corrector=broken)
    assert (out.ok, out.stage) == (False, "correct")
    assert out.stt_text == "hello there" and "ValueError: scorer exploded" in out.detail


def test_no_failure_message_ever_contains_a_traceback():
    cases = [
        core.run_pipeline(b"", ".wav", FakeTranscriber()),
        core.run_pipeline(SILENCE, ".wav", FakeTranscriber()),
        core.run_pipeline(SPEECH_LIKE, ".wav", FakeTranscriber(raises=RuntimeError("x"))),
        core.run_pipeline(SPEECH_LIKE, ".wav", FakeTranscriber(raises=ImportError("x"))),
        core.run_pipeline(SPEECH_LIKE, ".wav", FakeTranscriber(""), ),
    ]
    for out in cases:
        assert not out.ok
        assert "Traceback" not in out.message + out.detail and 'File "' not in out.message + out.detail


# --- run_text -------------------------------------------------------------------------------------------------------------------

def test_run_text_success():
    out = core.run_text("  the sea pu is hot  ")
    assert out.ok and out.stt_text == "the sea pu is hot" and out.result.corrected_text == "the CPU is hot"
    assert out.stt_seconds is None and out.audio_seconds is None


@pytest.mark.parametrize("text", ["", "   \n ", None])
def test_run_text_empty_is_a_warning(text):
    out = core.run_text(text)
    assert (out.ok, out.stage, out.kind) == (False, "input", "warning") and "Type or paste" in out.message


def test_run_text_corrector_failure_is_handled():
    def broken(text):
        raise KeyError("nope")
    out = core.run_text("hello", corrector=broken)
    assert (out.ok, out.stage) == (False, "correct") and "KeyError" in out.detail
