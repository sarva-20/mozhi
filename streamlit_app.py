"""
Mozhi live demo: microphone or audio file -> Whisper -> Mozhi correction, with the
correction latency front and centre.

Run:
    pip install -r requirements-demo.txt
    streamlit run streamlit_app.py

Reuses scripts/evaluate_accuracy.Transcriber (the same faster-whisper wrapper the
accuracy evaluation uses) and app.pipeline.correct_text. All non-UI logic lives in
demo/core.py so it is unit-tested; this file is layout and wiring only. The
correction path itself still does zero model inference: only the STT step, which
sits in front of Mozhi, uses a model.
"""
import hashlib

import streamlit as st

from app.dictionary import get_term_strings
from app.pipeline import DEFAULT_THRESHOLD, correct_text
from app.tokenizer import tokenize
from demo import core
from scripts import evaluate_accuracy

WHISPER_MODEL = "base.en"
MODES = ["🎤 Record", "⬆️ Upload", "🎧 Samples", "⌨️ Type text"]
RECORD, UPLOAD, SAMPLES, TYPE = MODES
DEFAULT_TEXT = "please check the sea pu usage and restart the kubernetees container"

st.set_page_config(page_title="Mozhi · live demo", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
.block-container { padding-top: 2.4rem; max-width: 1180px; }
.mz-title { font-size: 2.3rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; line-height: 1.1; }
.mz-tag { opacity: .72; margin: .3rem 0 .8rem; font-size: 1.02rem; }
.mz-pill { display: inline-block; padding: .14rem .65rem; border-radius: 999px; font-size: .78rem;
           border: 1px solid rgba(128,128,128,.38); margin: 0 .4rem .3rem 0; }
.mz-hero { border-radius: 14px; padding: 1.05rem 1.4rem 1rem; border-left: 6px solid var(--primary-color, #0f766e);
           background: var(--secondary-background-color, rgba(128,128,128,.10)); }
.mz-hero-label { text-transform: uppercase; letter-spacing: .07em; font-size: .74rem; opacity: .7; font-weight: 600; }
.mz-hero-value { font-size: 3.6rem; font-weight: 700; line-height: 1.05; font-variant-numeric: tabular-nums; }
.mz-hero-value span { font-size: 1.4rem; font-weight: 500; opacity: .65; margin-left: .3rem; }
.mz-hero-sub { font-size: .8rem; opacity: .68; margin-top: .35rem; }
.mz-box { border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: .85rem 1.1rem 1rem;
          line-height: 1.7; font-size: 1.1rem; min-height: 5.2rem; }
.mz-box-title { text-transform: uppercase; letter-spacing: .07em; font-size: .72rem; opacity: .65;
                font-weight: 600; margin-bottom: .3rem; }
mark.mz-removed { background: rgba(239,68,68,.26); color: inherit; border-radius: 4px; padding: 0 .22em; }
mark.mz-added { background: rgba(34,197,94,.34); color: inherit; border-radius: 4px; padding: 0 .22em; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# --- cached resources -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_speech_model():
    """(transcriber, error). The Whisper model is loaded once per server process; a failure is
    returned, not raised, so the app can still offer the text mode."""
    transcriber = evaluate_accuracy.Transcriber(WHISPER_MODEL)
    try:
        transcriber.load()
    except ImportError:
        return None, "faster-whisper isn't installed. Run: pip install -r requirements-demo.txt"
    except Exception as error:  # noqa: BLE001
        return None, f"The speech model couldn't be loaded ({type(error).__name__}). It downloads once on first use and needs network access."
    return transcriber, None


@st.cache_resource(show_spinner=False)
def warm_pipeline() -> bool:
    """Run the correction a few times so the first live latency figure isn't a cold-start outlier."""
    for _ in range(5):
        correct_text("warm up the sea pu")
    return True


with st.spinner(f"Loading speech model ({WHISPER_MODEL})…"):
    transcriber, load_error = load_speech_model()
warm_pipeline()
n_terms = len(get_term_strings())


# --- sidebar ----------------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Mozhi")
    st.caption("A non-generative phonetic correction layer between a speech recogniser and the app that consumes its text.")
    st.markdown(
        f"""
- **Correction path:** dictionary lookups and phonetic scoring only. No model inference, no GPU, no network.
- **Domain dictionary:** {n_terms} terms
- **Fuzzy threshold:** {DEFAULT_THRESHOLD} (fixed)
- **Speech recogniser:** Whisper `{WHISPER_MODEL}`, CPU, int8. This is the only model in the demo, and it sits *in front of* Mozhi.
"""
    )
    st.markdown("**Match methods**")
    for method, description in core.METHOD_DESCRIPTIONS.items():
        st.caption(f"`{method}`: {description}")


# --- header --------------------------------------------------------------------------------------------

st.markdown('<div class="mz-title">Mozhi</div>', unsafe_allow_html=True)
st.markdown('<div class="mz-tag">Say a sentence with technical terms in it. Watch the speech recogniser get them wrong, and Mozhi put them right.</div>', unsafe_allow_html=True)
st.markdown(
    '<span class="mz-pill">zero model inference in the correction path</span>'
    '<span class="mz-pill">CPU only</span><span class="mz-pill">deterministic</span>',
    unsafe_allow_html=True,
)

mode = st.segmented_control("Input", MODES, default=RECORD, label_visibility="collapsed") or RECORD
st.write("")

if mode != TYPE and load_error:
    st.warning(f"Speech recognition is unavailable. {load_error} You can still use **Type text**.")
    if st.button("Retry loading the speech model"):
        load_speech_model.clear()
        st.rerun()


# --- running the pipeline -----------------------------------------------------------------------------

def store(outcome: core.Outcome, key: str, source: str, expected: str = "") -> None:
    st.session_state.update(outcome=outcome, outcome_key=key, outcome_mode=mode, outcome_source=source, outcome_expected=expected)


def process_audio(data: bytes, suffix: str, source: str, key: str, expected: str = "", force: bool = False) -> None:
    if not force and st.session_state.get("outcome_key") == key and st.session_state.get("outcome_mode") == mode:
        return  # same audio, already processed: don't re-transcribe on every widget interaction
    with st.spinner("Transcribing with Whisper, then correcting…"):
        outcome = core.run_pipeline(data, suffix, transcriber)
    store(outcome, key, source, expected)


def audio_key(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


if mode == RECORD:
    st.markdown("**Record a sentence.** Try: *“please check the sea pu usage before you restart the vee pea en”*")
    recording = st.audio_input("Microphone", sample_rate=16000, label_visibility="collapsed")
    if recording is not None and transcriber is not None:
        data = recording.getvalue()
        process_audio(data, core.audio_suffix(recording.name, recording.type), "Microphone recording", audio_key(data))

elif mode == UPLOAD:
    st.markdown("**Upload a recording** (WAV, MP3, M4A, OGG, FLAC or AAC).")
    upload = st.file_uploader("Audio file", type=[e.lstrip(".") for e in core.AUDIO_EXTENSIONS], label_visibility="collapsed")
    if upload is not None and transcriber is not None:
        data = upload.getvalue()
        process_audio(data, core.audio_suffix(upload.name, upload.type), upload.name, audio_key(data))

elif mode == SAMPLES:
    samples = core.list_samples()
    if not samples:
        st.info("No sample recordings found in data/volunteer_audio.")
    else:
        st.markdown("**Run the pipeline on a recorded sample.** These are the volunteer recordings used for the accuracy evaluation.")
        by_name = {s.name: s for s in samples}
        choice = by_name[st.selectbox("Sample", list(by_name), label_visibility="collapsed")]
        st.audio(choice.audio_path.read_bytes(), format=core.mimetype_for_suffix(choice.audio_path.suffix))
        if st.button("Transcribe and correct", type="primary", disabled=transcriber is None):
            data = choice.audio_path.read_bytes()
            process_audio(data, choice.audio_path.suffix, choice.name, f"sample:{choice.name}", expected=choice.expected, force=True)

else:
    st.markdown("**Paste a transcript** to see the correction on its own, without the speech step.")
    with st.form("text_form", border=False):
        typed = st.text_area("Transcript", value=DEFAULT_TEXT, height=110, label_visibility="collapsed")
        submitted = st.form_submit_button("Correct text", type="primary")
    if submitted:
        store(core.run_text(typed), f"text:{hashlib.sha1(typed.encode()).hexdigest()}", "Typed text")


# --- rendering the result --------------------------------------------------------------------------------------

def render_result(outcome: core.Outcome, source: str, expected: str) -> None:
    if not outcome.ok:
        (st.warning if outcome.kind == "warning" else st.error)(outcome.message)
        if outcome.detail:
            with st.expander("Technical details"):
                st.code(outcome.detail, language=None)
        return

    result = outcome.result
    st.divider()
    ms = result.elapsed_ms
    n_words = len(tokenize(result.original_text))
    changed = sum(1 for m in result.matches if result.original_text[m.start : m.end] != m.replacement)
    hero, metric_a, metric_b, metric_c = st.columns([2.3, 1, 1, 1])
    hero.markdown(
        f'<div class="mz-hero"><div class="mz-hero-label">Mozhi correction latency</div>'
        f'<div class="mz-hero-value">{ms:.1f}<span>ms</span></div>'
        f'<div class="mz-hero-sub">one call · {n_words}-word transcript · timed inside correct_text()<br>'
        f'grows with transcript length · no model inference, no network</div></div>',
        unsafe_allow_html=True,
    )
    if outcome.stt_seconds is not None:  # audio modes: show the STT step, clearly separate from Mozhi's latency
        metric_a.metric("Whisper transcription", f"{outcome.stt_seconds:.2f} s",
                        help="Speech-to-text time. This step happens BEFORE Mozhi and is not part of the correction latency.")
        metric_b.metric("Audio length", f"{outcome.audio_seconds:.1f} s" if outcome.audio_seconds is not None else "n/a")
    else:
        metric_a.metric("Words in transcript", n_words)
        metric_b.metric("Matches found", len(result.matches), help="Including any that confirmed a term as already correct.")
    metric_c.metric("Words corrected", changed, help=f"{len(result.matches)} match(es) in total, including any that confirmed a term as already correct.")
    st.write("")

    raw_segments, corrected_segments = core.highlight_segments(result.original_text, result.matches)
    left, right = st.columns(2)
    left.markdown(
        f'<div class="mz-box"><div class="mz-box-title">Raw STT transcript</div>{core.segments_to_html(raw_segments, "mz-removed")}</div>',
        unsafe_allow_html=True,
    )
    right.markdown(
        f'<div class="mz-box"><div class="mz-box-title">Corrected by Mozhi</div>{core.segments_to_html(corrected_segments, "mz-added")}</div>',
        unsafe_allow_html=True,
    )
    legend = f"Source: {source}. " + ("Red: what the recogniser heard. Green: Mozhi's replacement." if changed else "Mozhi made no changes to this transcript.")
    st.caption(legend)
    if expected:
        st.caption(f"What was actually said: *{expected}*")

    st.markdown("#### Matches")
    rows = core.match_rows(result.original_text, result.matches)
    if rows:
        st.dataframe(
            rows, hide_index=True, width="stretch",
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Score": st.column_config.NumberColumn(format="%.3f", help="1.000 for exact-lookup methods; the weighted fuzzy score otherwise."),
            },
        )
        fuzzy = core.breakdown_rows(result.original_text, result.matches)
        if fuzzy:
            with st.expander("Score breakdown for fuzzy matches"):
                st.dataframe(fuzzy, hide_index=True, width="stretch")
    else:
        st.info("No domain terms needed correcting in this transcript.")

    with st.expander("Measure latency over 50 runs"):
        st.caption("One call is a noisy sample. This re-runs the correction on this transcript and reports the spread.")
        if st.button("Run 50 corrections"):
            with st.spinner("Timing…"):
                stats = core.latency_stats(result.original_text, runs=50)
            cols = st.columns(5)
            for col, name in zip(cols, ["min", "median", "mean", "p95", "p99"]):
                col.metric(name, f"{stats[name]:.2f} ms")
            st.caption(f"{stats['runs']} runs after warm-up; max {stats['max']:.2f} ms.")


outcome = st.session_state.get("outcome")
if outcome is not None and st.session_state.get("outcome_mode") == mode:
    render_result(outcome, st.session_state.get("outcome_source", ""), st.session_state.get("outcome_expected", ""))
elif mode in (RECORD, UPLOAD) and not load_error:
    st.caption("The result appears here as soon as the audio is in.")
