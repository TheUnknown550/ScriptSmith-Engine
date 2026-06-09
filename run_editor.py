"""Build a video from scene images and narration audio."""

import argparse
import json
import os

from pipeline import config, editor


def _load_segments(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(
        description="Stitch scene images into a video synced to narration audio."
    )
    parser.add_argument(
        "--images",
        default=config.DEFAULT_IMAGE_DIR,
        help=f"Folder containing scene images. Defaults to {config.DEFAULT_IMAGE_DIR}.",
    )
    parser.add_argument(
        "--audio",
        default=config.FULL_AUDIO,
        help="Narration audio path.",
    )
    parser.add_argument(
        "--segments",
        default=config.TRANSCRIPT_JSON,
        help="Transcript JSON from run_pipeline.py.",
    )
    parser.add_argument(
        "--output",
        default=config.FINAL_VIDEO,
        help="Path to the final mp4 output.",
    )
    parser.add_argument(
        "--use-filename-timestamps",
        action="store_true",
        help="Use image filenames like 1-29.png as timeline anchors instead of transcript JSON.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if not os.path.isdir(args.images):
        raise NotADirectoryError(f"Image folder not found: {args.images}")

    segments = None
    if not args.use_filename_timestamps:
        if not os.path.exists(args.segments):
            raise FileNotFoundError(
                f"Transcript JSON not found: {args.segments}. "
                "Use run_pipeline.py first or pass --use-filename-timestamps."
            )
        segments = _load_segments(args.segments)

    editor.stitch_images_to_video(
        image_dir=args.images,
        audio_path=args.audio,
        output_path=args.output,
        segments=segments,
        use_filename_timestamps=args.use_filename_timestamps,
    )


if __name__ == "__main__":
    main()
