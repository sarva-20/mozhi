"""Transcribe local audio with faster-whisper on CPU and save JSON results.

Usage::

    pip install -r requirements-benchmark.txt
    PYTHONPATH=. python3 scripts/transcribe_local.py path/to/audio_or_dir/

Audio files are written to ``data/stt_transcripts/<filename>.json``. ffmpeg
must be installed and available on PATH.
"""
import argparse
import json
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "stt_transcripts"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}


def transcribe_one(model, audio_path: Path) -> dict:
    start = time.perf_counter()
    segments, info = model.transcribe(str(audio_path), beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed_s = time.perf_counter() - start
    duration_s = info.duration

    return {
        "file": audio_path.name,
        "transcript": text,
        "audio_duration_s": round(duration_s, 3),
        "transcription_time_s": round(elapsed_s, 3),
        "real_time_factor": round(elapsed_s / duration_s, 3) if duration_s else None,
        "detected_language": info.language,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", help="Audio file or directory of audio files")
    parser.add_argument("--model", default="base.en", help="Whisper model size (default: base.en)")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper is not installed. Run: pip install -r requirements-benchmark.txt")
        return 1

    path = Path(args.audio_path)
    if path.is_dir():
        audio_files = sorted(
            file_path for file_path in path.iterdir() if file_path.suffix.lower() in AUDIO_EXTENSIONS
        )
    elif path.is_file():
        audio_files = [path]
    else:
        print(f"ERROR: {path} not found")
        return 1

    if not audio_files:
        print(f"No audio files found in {path} (looked for {sorted(AUDIO_EXTENSIONS)})")
        return 1

    print(f"Loading Whisper model '{args.model}' (CPU, int8)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for audio_path in audio_files:
        print(f"\nTranscribing {audio_path.name}...")
        try:
            result = transcribe_one(model, audio_path)
        except Exception as error:
            print(f"  ERROR: {error}")
            continue

        results.append(result)
        print(f"  transcript: {result['transcript']!r}")
        print(f"  RTF: {result['real_time_factor']}")
        output_path = OUTPUT_DIR / f"{audio_path.stem}.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  saved: {output_path}")

    if not results:
        print("\nNo files transcribed successfully.")
        return 1

    rtfs = [result["real_time_factor"] for result in results if result["real_time_factor"] is not None]
    print(f"\n=== Summary ({len(results)} files) ===")
    print(f"  Model: {args.model} (CPU, int8)")
    if rtfs:
        print(f"  RTF mean: {sum(rtfs) / len(rtfs):.2f}x")
    print(f"  Transcripts saved to: {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())