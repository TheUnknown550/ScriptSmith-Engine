"""Regenerate transcript/timestamps from an existing audio file only."""

import argparse

from pipeline import config, transcribe
from run_pipeline import _write_plaintext, _write_srt


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an existing narration audio file without regenerating TTS."
    )
    parser.add_argument(
        "--audio",
        default=config.FULL_AUDIO,
        help="Existing narration audio path.",
    )
    parser.add_argument(
        "--json",
        default=config.TRANSCRIPT_JSON,
        help="Path to write transcript JSON.",
    )
    parser.add_argument(
        "--txt",
        default=config.TRANSCRIPT_TXT,
        help="Path to write transcript text.",
    )
    parser.add_argument(
        "--srt",
        default=config.TRANSCRIPT_SRT,
        help="Path to write transcript SRT.",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    segments = transcribe.transcribe(args.audio)
    config.write_json(args.json, segments)
    _write_plaintext(args.txt, segments)
    _write_srt(args.srt, segments)
    print(f"Wrote transcript JSON: {args.json}")
    print(f"Wrote transcript text: {args.txt}")
    print(f"Wrote transcript SRT: {args.srt}")


if __name__ == "__main__":
    main()
