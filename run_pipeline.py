"""Generate full narration audio from script.txt, then transcribe it with timestamps."""

import argparse
import os

from pipeline import config, transcribe, tts


def _format_ts(seconds):
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _write_plaintext(path, segments):
    with open(path, "w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(
                f"[{segment['start']:.3f} --> {segment['end']:.3f}] {segment['text']}\n"
            )


def _write_srt(path, segments):
    with open(path, "w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(f"{segment['index']}\n")
            handle.write(
                f"{_format_ts(segment['start'])} --> {_format_ts(segment['end'])}\n"
            )
            handle.write(f"{segment['text']}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Turn script.txt into full.wav and timestamped transcript files."
    )
    parser.add_argument(
        "--script",
        default=config.SCRIPT_PATH,
        help="Path to the input script text file.",
    )
    parser.add_argument(
        "--audio",
        default=config.FULL_AUDIO,
        help="Path to write the generated wav file.",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Reuse an existing audio file and only run transcription.",
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Alias for --skip-tts. Reuse existing audio and regenerate timestamps only.",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=config.TTS_FINAL_ATEMPO,
        help=(
            "Final narration tempo multiplier. Use values below 1.0 to slow the audio "
            "slightly, or above 1.0 to speed it up."
        ),
    )
    args = parser.parse_args()

    config.ensure_dirs()

    if args.skip_tts or args.transcript_only:
        if not os.path.exists(args.audio):
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        audio_path = args.audio
        print(f"Using existing audio: {audio_path}")
    else:
        script_text = config.read_script(args.script)
        if not script_text:
            raise ValueError(f"Script file is empty: {args.script}")
        audio_path, duration = tts.synthesize_full(
            script_text,
            args.audio,
            final_atempo=args.pace,
        )
        print(f"Wrote audio: {audio_path} ({duration:.1f}s)")

    segments = transcribe.transcribe(audio_path)
    config.write_json(config.TRANSCRIPT_JSON, segments)
    _write_plaintext(config.TRANSCRIPT_TXT, segments)
    _write_srt(config.TRANSCRIPT_SRT, segments)
    print(f"Wrote transcript JSON: {config.TRANSCRIPT_JSON}")
    print(f"Wrote transcript text: {config.TRANSCRIPT_TXT}")
    print(f"Wrote transcript SRT: {config.TRANSCRIPT_SRT}")


if __name__ == "__main__":
    main()
