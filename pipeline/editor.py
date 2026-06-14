"""Stitch timestamped images into a video synced to narration audio."""

from __future__ import annotations

import os
import re
import subprocess

from . import config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _timestamp_from_name(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    match = re.match(r"^(\d+(?:-\d+){1,3})", stem)
    if not match:
        return None
    parts = match.group(1).split("-")
    if not parts or any(not part.isdigit() for part in parts):
        return None

    if len(parts) == 4:
        hours, minutes, seconds, millis = (int(part) for part in parts)
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

    if len(parts) == 3:
        minutes, seconds, millis = (int(part) for part in parts)
        return minutes * 60 + seconds + millis / 1000.0

    if len(parts) == 2:
        minutes, seconds = (int(part) for part in parts)
        return minutes * 60 + seconds

    return None


def _audio_duration_seconds(audio_path):
    ffprobe = config.find_ffprobe()
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())

    import soundfile as sf

    with sf.SoundFile(audio_path) as handle:
        return len(handle) / handle.samplerate


def _sorted_images(image_dir):
    files = []
    for entry in os.listdir(image_dir):
        path = os.path.join(image_dir, entry)
        if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in IMAGE_EXTS:
            files.append(path)
    files.sort(key=lambda path: _natural_key(os.path.basename(path)))
    if not files:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return files


def images_have_filename_timestamps(image_dir):
    images = _sorted_images(image_dir)
    return all(_timestamp_from_name(os.path.basename(path)) is not None for path in images)


def load_timeline_from_segments(segments, image_dir):
    images = _sorted_images(image_dir)
    if len(images) < len(segments):
        raise ValueError(
            f"Not enough images for transcript segments: {len(images)} images for {len(segments)} segments."
        )

    timeline = []
    for image_path, segment in zip(images, segments):
        duration = max(0.6, float(segment["end"]) - float(segment["start"]))
        timeline.append(
            {
                "image_path": image_path,
                "start": float(segment["start"]),
                "end": float(segment["start"]) + duration,
                "duration": duration,
                "label": segment.get("text", ""),
            }
        )
    return timeline


def load_timeline_from_filenames(image_dir, audio_path):
    images = _sorted_images(image_dir)
    points = []
    for image_path in images:
        stamp = _timestamp_from_name(os.path.basename(image_path))
        if stamp is None:
            raise ValueError(
                "Image filename must start with a timestamp like "
                f"M-SS, MM-SS-MMM, or HH-MM-SS-MMM: {os.path.basename(image_path)}"
            )
        points.append((stamp, image_path))

    audio_duration = _audio_duration_seconds(audio_path)
    timeline = []
    for index, (start, image_path) in enumerate(points):
        next_start = points[index + 1][0] if index + 1 < len(points) else audio_duration
        duration = max(0.6, float(next_start) - float(start))
        timeline.append(
            {
                "image_path": image_path,
                "start": float(start),
                "end": float(start) + duration,
                "duration": duration,
                "label": os.path.basename(image_path),
            }
        )
    return timeline


def _render_scene_clip(ffmpeg_path, item, index, output_path):
    fps = config.VIDEO_FPS
    video_encoder = config.find_h264_encoder()
    vf = (
        f"scale={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"format=yuv420p"
    )

    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-loop",
            "1",
            "-i",
            item["image_path"],
            "-t",
            f"{item['duration']:.3f}",
            "-vf",
            vf,
            "-r",
            str(fps),
            "-an",
            "-c:v",
            video_encoder,
            "-pix_fmt",
            "yuv420p",
            output_path,
        ],
        check=True,
    )
    print(
        f"[editor] clip {index + 1}: {os.path.basename(item['image_path'])} "
        f"({item['duration']:.2f}s, encoder={video_encoder})"
    )


def _xfade_duration(previous_item, next_item):
    max_allowed = min(previous_item["duration"], next_item["duration"])
    return min(config.FADE_DURATION, max(0.0, max_allowed - 0.01))


def _final_video_duration(timeline):
    return max(0.0, sum(item["duration"] for item in timeline))


def stitch_images_to_video(
    image_dir,
    audio_path,
    output_path=None,
    segments=None,
    use_filename_timestamps=False,
):
    ffmpeg_path = config.find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg not found on PATH.")

    config.ensure_dirs()
    output_path = output_path or config.FINAL_VIDEO
    timeline = (
        load_timeline_from_filenames(image_dir, audio_path)
        if use_filename_timestamps
        else load_timeline_from_segments(segments or [], image_dir)
    )
    if not timeline:
        raise ValueError("No timeline items available for stitching.")

    clips_dir = os.path.join(config.TEMP_DIR, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    clip_paths = []
    for index, item in enumerate(timeline):
        clip_path = os.path.join(clips_dir, f"clip_{index:04d}.mp4")
        render_item = dict(item)
        if index > 0:
            render_item["duration"] = item["duration"] + config.FADE_DURATION
        _render_scene_clip(ffmpeg_path, render_item, index, clip_path)
        clip_paths.append(clip_path)

    filter_parts = []
    for index in range(len(clip_paths)):
        filter_parts.append(f"[{index}:v]setpts=PTS-STARTPTS[v{index}]")

    current_label = "v0"
    current_duration = timeline[0]["duration"]
    for index in range(1, len(clip_paths)):
        fade = _xfade_duration(timeline[index - 1], timeline[index])
        if fade <= 0:
            raise ValueError(
                f"Unable to build crossfade for clips {index} and {index + 1}: duration too short."
            )
        next_label = f"v{index}"
        output_label = f"x{index}"
        offset = max(0.0, current_duration - fade)
        filter_parts.append(
            f"[{current_label}][{next_label}]xfade=transition=fade:duration={fade:.3f}:offset={offset:.3f}[{output_label}]"
        )
        current_label = output_label
        current_duration += timeline[index]["duration"]

    edge_fade = min(config.FADE_DURATION, max(0.0, _final_video_duration(timeline) / 2.0 - 0.01))
    if edge_fade > 0:
        faded_label = "vfinal"
        fade_out_start = max(0.0, _final_video_duration(timeline) - edge_fade)
        filter_parts.append(
            f"[{current_label}]fade=t=in:st=0:d={edge_fade:.3f},"
            f"fade=t=out:st={fade_out_start:.3f}:d={edge_fade:.3f}[{faded_label}]"
        )
        current_label = faded_label

    filter_script = os.path.join(config.TEMP_DIR, "xfade_filter.txt")
    with open(filter_script, "w", encoding="utf-8") as handle:
        handle.write(";".join(filter_parts))
    video_encoder = config.find_h264_encoder()
    audio_input_index = len(clip_paths)
    ffmpeg_inputs = []
    for clip_path in clip_paths:
        ffmpeg_inputs.extend(["-i", clip_path])
    ffmpeg_inputs.extend(["-i", audio_path])

    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            *ffmpeg_inputs,
            "-filter_complex_script",
            filter_script,
            "-map",
            f"[{current_label}]",
            "-map",
            f"{audio_input_index}:a:0",
            "-c:v",
            video_encoder,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            output_path,
        ],
        check=True,
    )
    print(f"[editor] wrote final video: {output_path}")
    return output_path
