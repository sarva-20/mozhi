"""
Non-UI logic for the Streamlit demo: highlight diff, match tables, latency stats,
silence detection, and the audio -> transcript -> correction flow with every failure
turned into a user-facing message (the UI never shows a traceback).

Latency honesty: the headline latency is the correction layer's own elapsed_ms,
measured inside app.pipeline.correct_text with time.perf_counter. Whisper time is
tracked separately (stt_seconds) so the two are never conflated.
"""
import html
import io
import logging
import statistics
import sys
import tempfile
import wave
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.pipeline import correct_text
from scripts.evaluate_accuracy import AUDIO_DIR, discover_pairs

log = logging.getLogger("mozhi.demo")

MIN_AUDIO_BYTES = 256                  # a WAV header alone is 44 bytes; anything this small has no audio in it
MAX_AUDIO_BYTES = 25 * 1024 * 1024     # a CPU Whisper run on a huge upload would freeze a live demo
SILENCE_PEAK_FRACTION = 0.01           # peak below 1% of full scale counts as silence (16-bit WAV only)
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac")

# Legend for the matches table: what each method id means.
METHOD_DESCRIPTIONS = {
    "acronym_map": "hand-mapped spoken form of an acronym (exact lookup)",
    "phrase_alias": "verified multi-word mishearing (exact lookup)",
    "letter_composed": "spelled-out letters composed into a dictionary acronym (exact match)",
    "fuzzy": "Metaphone + Jaro-Winkler + token-sort score above the threshold",
}


# --- highlight diff --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    text: str
    changed: bool


def highlight_segments(text: str, matches: list) -> tuple[list[Segment], list[Segment]]:
    """Split `text` into (raw_side, corrected_side) segment lists. Matched spans become
    a changed segment on both sides (original words on the raw side, replacement on the
    corrected side); everything between them is shared, unchanged text. A match whose
    replacement equals the original ("ISP" -> "ISP") is not a visible change and is not
    flagged. Joining the corrected side reproduces CorrectionResult.corrected_text."""
    raw: list[Segment] = []
    corrected: list[Segment] = []
    cursor = 0
    for m in sorted(matches, key=lambda m: m.start):
        if m.start < cursor:  # overlapping spans never come out of the DP; skip defensively
            continue
        gap = text[cursor : m.start]
        if gap:
            raw.append(Segment(gap, False))
            corrected.append(Segment(gap, False))
        original = text[m.start : m.end]
        changed = original != m.replacement
        raw.append(Segment(original, changed))
        corrected.append(Segment(m.replacement, changed))
        cursor = m.end
    tail = text[cursor:]
    if tail:
        raw.append(Segment(tail, False))
        corrected.append(Segment(tail, False))
    return raw, corrected


def segments_to_html(segments: list[Segment], css_class: str) -> str:
    """HTML for a segment list: every piece of text is escaped (STT output is
    arbitrary), changed segments are wrapped in <mark class=css_class>."""
    parts = []
    for s in segments:
        escaped = html.escape(s.text)
        parts.append(f'<mark class="{html.escape(css_class)}">{escaped}</mark>' if s.changed else escaped)
    return "".join(parts)


def match_rows(text: str, matches: list) -> list[dict]:
    """Rows for the matches table, in reading order."""
    rows = []
    for i, m in enumerate(sorted(matches, key=lambda m: m.start), start=1):
        original = text[m.start : m.end]
        rows.append({
            "#": i,
            "Original": original,
            "Replacement": m.replacement,
            "Method": m.method,
            "Score": round(float(m.score), 3),
            "Changed": "yes" if original != m.replacement else "no (already correct)",
        })
    return rows


def breakdown_rows(text: str, matches: list) -> list[dict]:
    """Per-signal scores for fuzzy matches (the only method that has them)."""
    rows = []
    for m in sorted(matches, key=lambda m: m.start):
        b = m.score_breakdown
        if m.method == "fuzzy" and b:
            rows.append({
                "Original": text[m.start : m.end], "Replacement": m.replacement,
                "Metaphone": round(b["metaphone"], 3), "Surface (JW)": round(b["surface"], 3),
                "Token sort": round(b["token_sort"], 3), "Weighted total": round(b["total"], 3),
            })
    return rows


# --- latency ---------------------------------------------------------------------------------------------

def latency_stats(text: str, runs: int = 50, warmup: int = 3, corrector: Callable = correct_text) -> dict:
    """Re-run the correction `runs` times on `text` and summarize each call's own elapsed_ms.
    One call is noisy; this is the live version of scripts/latency_benchmark.py."""
    if runs < 1:
        raise ValueError("runs must be >= 1")
    for _ in range(warmup):
        corrector(text)
    xs = sorted(corrector(text).elapsed_ms for _ in range(runs))
    n = len(xs)
    return {
        "runs": n, "min": xs[0], "mean": statistics.mean(xs), "median": statistics.median(xs),
        "p95": xs[min(n - 1, int(n * 0.95))], "p99": xs[min(n - 1, int(n * 0.99))], "max": xs[-1],
    }


# --- audio checks ------------------------------------------------------------------------------------------

def wav_peak_fraction(data: bytes) -> float | None:
    """Peak amplitude as a fraction of full scale for a 16-bit PCM WAV, or None if the
    bytes are not one (compressed uploads can't be inspected without decoding them)."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2:
                return None
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return None
    samples = array("h")
    samples.frombytes(raw[: len(raw) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0
    return max(abs(min(samples)), abs(max(samples))) / 32768


def is_silent(data: bytes) -> bool | None:
    """True/False for a 16-bit WAV; None when it can't be judged."""
    peak = wav_peak_fraction(data)
    return None if peak is None else peak < SILENCE_PEAK_FRACTION


def audio_suffix(name: str | None, mime: str | None = None) -> str:
    """File extension for the temp file handed to the transcriber."""
    if name and Path(name).suffix.lower() in AUDIO_EXTENSIONS:
        return Path(name).suffix.lower()
    by_mime = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/mpeg": ".mp3",
               "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/ogg": ".ogg", "audio/flac": ".flac"}
    return by_mime.get((mime or "").lower(), ".wav")


# The reverse of audio_suffix's by_mime: st.audio() needs an explicit mimetype for
# raw bytes read from disk (the sample-picker flow) - it cannot sniff one from the
# suffix itself, and passing format=None crashes deep in Streamlit's media file
# manager (AttributeError: 'NoneType' object has no attribute 'encode') rather than
# falling back to a guess. This surfaced only once data/volunteer_audio held a
# non-.m4a sample (the staged .mp3 volunteer recordings): every sample before that
# was .m4a, so the missing case was never exercised.
_MIMETYPE_BY_SUFFIX = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac",
}


def mimetype_for_suffix(suffix: str) -> str:
    """Best-effort mimetype for a file extension, for st.audio()'s `format`
    argument. Falls back to audio/wav (as audio_suffix() itself does) rather
    than None, which st.audio() cannot handle."""
    return _MIMETYPE_BY_SUFFIX.get(suffix.lower(), "audio/wav")


# --- samples ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    name: str
    audio_path: Path
    expected: str


def list_samples(audio_dir: Path = AUDIO_DIR) -> list[Sample]:
    """Recordings in data/volunteer_audio that have a paired .expected.txt."""
    if not audio_dir.is_dir():
        return []
    pairs, _ = discover_pairs(audio_dir)
    return [Sample(a.stem, a, e.read_text(encoding="utf-8").strip()) for a, e in pairs]


# --- the flow -------------------------------------------------------------------------------------------------------

@dataclass
class Outcome:
    ok: bool
    stage: str = "done"          # input | transcribe | correct | done
    kind: str = "error"          # error | warning   (only meaningful when ok is False)
    message: str = ""            # user-facing explanation when not ok
    detail: str = ""             # one-line technical detail, shown collapsed
    stt_text: str = ""
    stt_seconds: float | None = None
    audio_seconds: float | None = None
    result: object | None = None  # app.pipeline.CorrectionResult


def _fail(stage: str, message: str, detail: str = "", kind: str = "error", **kw) -> Outcome:
    return Outcome(ok=False, stage=stage, kind=kind, message=message, detail=detail, **kw)


def _is_decode_error(error: Exception) -> bool:
    """faster-whisper decodes through PyAV, which raises InvalidDataError on a file it can't parse."""
    return type(error).__name__ == "InvalidDataError" or "invalid data found" in str(error).lower()


def _correct(text: str, corrector: Callable, **kw) -> Outcome:
    try:
        result = corrector(text)
    except Exception as error:  # noqa: BLE001 - the UI must never show a traceback
        log.exception("correction failed")
        return _fail("correct", "The correction step hit an unexpected error.",
                     f"{type(error).__name__}: {str(error)[:200]}", stt_text=text, **kw)
    return Outcome(ok=True, stt_text=text, result=result, **kw)


def run_text(text: str, corrector: Callable = correct_text) -> Outcome:
    """Correction-only flow for typed text (no audio, no STT)."""
    text = (text or "").strip()
    if not text:
        return _fail("input", "Type or paste some text to correct.", kind="warning")
    return _correct(text, corrector)


def run_pipeline(
    audio: bytes,
    suffix: str,
    transcriber: Callable[[Path], dict],
    corrector: Callable = correct_text,
) -> Outcome:
    """audio bytes -> temp file -> transcriber -> corrector. Never raises: empty, oversized
    or silent audio, a missing speech model, an undecodable file, an empty transcript and a
    corrector failure each come back as an Outcome with a plain-language message."""
    if not audio or len(audio) < MIN_AUDIO_BYTES:
        return _fail("input", "No audio was captured. Record again, upload a file, or pick a sample.", kind="warning")
    if len(audio) > MAX_AUDIO_BYTES:
        return _fail("input", f"That file is larger than {MAX_AUDIO_BYTES // (1024 * 1024)} MB. "
                              "Please use a shorter recording.", kind="warning")
    if is_silent(audio):
        return _fail("input", "That recording is silent. Check the microphone and try again.", kind="warning")

    with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as tmp:
        tmp.write(audio)
        path = Path(tmp.name)
    try:
        raw = transcriber(path)
    except ImportError as error:
        log.exception("speech recognition unavailable")
        return _fail("transcribe", "Speech recognition isn't installed. Run: pip install -r requirements-demo.txt",
                     f"{type(error).__name__}: {error}")
    except Exception as error:  # noqa: BLE001
        log.exception("transcription failed")
        if _is_decode_error(error):
            message = "That audio file couldn't be decoded. It may be corrupted or in an unsupported format."
        else:
            message = ("Transcription failed. The speech model may not have loaded (it downloads once on first use "
                       "and needs network access), or the audio could not be processed.")
        return _fail("transcribe", message, f"{type(error).__name__}: {str(error)[:200]}")
    finally:
        path.unlink(missing_ok=True)

    text = (raw.get("transcript") or "").strip() if isinstance(raw, dict) else ""
    seconds = {"stt_seconds": raw.get("transcription_time_s"), "audio_seconds": raw.get("audio_duration_s")} \
        if isinstance(raw, dict) else {}
    if not text:
        return _fail("transcribe", "No speech was detected in that audio.", kind="warning", **seconds)
    return _correct(text, corrector, **seconds)
