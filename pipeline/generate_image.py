"""Scene planning and continuity-aware image generation for narrated videos."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import urllib.error
import urllib.request
from typing import Any
from uuid import uuid4

from runware import IImageInference, IInputReference, IInputs, IOpenAIProviderSettings, Runware

from . import config


def _load_segments(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Transcript JSON must contain a list of segments: {path}")
    return rows


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "scene"


def _timestamp_slug(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    if hours > 0:
        return f"{hours:02d}-{minutes:02d}-{secs:02d}-{millis:03d}"
    return f"{minutes:02d}-{secs:02d}-{millis:03d}"


def _scene_schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scene_number": {"type": "integer"},
            "start": {"type": "number"},
            "end": {"type": "number"},
            "transcript": {"type": "string"},
            "summary": {"type": "string"},
            "scene_change": {"type": "string", "enum": ["hard", "soft"]},
            "continue_from_previous": {"type": "boolean"},
            "reference_mode": {"type": "string", "enum": ["none", "soft", "strong"]},
            "prompt": {"type": "string"},
        },
        "required": [
            "scene_number",
            "start",
            "end",
            "transcript",
            "summary",
            "scene_change",
            "continue_from_previous",
            "reference_mode",
            "prompt",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"scenes": {"type": "array", "items": item}},
        "required": ["scenes"],
    }


def _build_planner_prompt(segments: list[dict[str, Any]]) -> str:
    transcript_lines = []
    for row in segments:
        transcript_lines.append(
            f"Segment {int(row.get('index', len(transcript_lines) + 1))} "
            f"[{float(row['start']):.2f}-{float(row['end']):.2f}] "
            f"{_clean_text(row.get('text', ''))}"
        )
    transcript_block = "\n".join(transcript_lines)
    return "\n".join(
        [
            "Turn this narrated transcript into an image plan for a cinematic video.",
            "Create exactly one scene per transcript segment.",
            "Do not merge segments.",
            "Do not split segments.",
            "Return exactly the same number of scenes as the transcript segments provided.",
            "Each scene_number must match the transcript segment index.",
            "Each scene start and end must exactly match the transcript segment timestamps.",
            "Use continue_from_previous=true only when the next image should visibly preserve the same subject, location, and style continuity.",
            "Use scene_change=hard and reference_mode=none for major visual resets like new place, new time, new action, or a big reveal.",
            "Use scene_change=soft with reference_mode=soft or strong only for neighboring shots that should feel connected.",
            "Write prompts for still-image generation, not for animation.",
            "Every prompt must include clear subject, environment, camera framing, lighting, and mood.",
            f"Assume this persistent global style is always added separately: {config.IMAGE_PROMPT_STYLE}",
            "Transcript:",
            transcript_block,
        ]
    )


def _extract_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    raise RuntimeError("Planner response did not contain text output.")


def _extract_json_block(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Planner output did not contain a JSON object.")
    return text[start : end + 1]


def _plan_with_minimax(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not config.MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY not configured.")

    body = {
        "model": config.MINIMAX_SCENE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a scene planner for an automated video pipeline. "
                    "Return valid JSON only. Do not wrap it in markdown. "
                    "Do not include explanation text before or after the JSON."
                ),
            },
            {"role": "user", "content": _build_planner_prompt(segments)},
        ],
        "max_completion_tokens": 4000,
    }
    request = urllib.request.Request(
        f"{config.MINIMAX_API_BASE.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiniMax planner request failed: {exc.code} {detail}") from exc

    text = _extract_response_text(payload)
    parsed = json.loads(_extract_json_block(text))
    return parsed["scenes"]


def _heuristic_scene_change(current_text: str, next_text: str) -> bool:
    current_text = current_text.lower()
    next_text = next_text.lower()
    change_cues = (
        "suddenly",
        "then",
        "meanwhile",
        "later",
        "now",
        "instead",
        "but",
        "however",
        "across",
        "outside",
        "inside",
    )
    if any(cue in next_text for cue in change_cues):
        return True
    current_subject = set(re.findall(r"[a-z]{4,}", current_text))
    next_subject = set(re.findall(r"[a-z]{4,}", next_text))
    overlap = len(current_subject & next_subject)
    return overlap <= 1


def _plan_with_heuristics(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    for row in segments:
        transcript = _clean_text(row.get("text", ""))
        summary = _clean_text(transcript[:160])
        previous = scenes[-1] if scenes else None
        scene_change = "hard"
        continue_from_previous = False
        reference_mode = "none"
        if previous is not None and not _heuristic_scene_change(previous["transcript"], transcript):
            scene_change = "soft"
            continue_from_previous = True
            reference_mode = "soft"
        scenes.append(
            {
                "scene_number": int(row.get("index", len(scenes) + 1)),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "transcript": transcript,
                "summary": summary,
                "scene_change": scene_change,
                "continue_from_previous": continue_from_previous,
                "reference_mode": reference_mode,
                "prompt": (
                    f"Create a cinematic still image for this narration beat: {transcript}. "
                    "Choose a clear focal subject, a believable environment, dynamic framing, "
                    "story-driven lighting, and a mood that matches the line."
                ),
            }
        )
    return scenes


def plan_scenes(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        scenes = _plan_with_minimax(segments)
        print(f"[image] planned {len(scenes)} scenes with MiniMax M3.")
    except Exception as exc:  # noqa: BLE001
        print(f"[image] MiniMax planner unavailable, using heuristic fallback: {exc}")
        scenes = _plan_with_heuristics(segments)
        print(f"[image] planned {len(scenes)} scenes with heuristic fallback.")
    return _normalize_scenes(scenes)


def _normalize_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, scene in enumerate(scenes, 1):
        start = float(scene["start"])
        end = max(start, float(scene["end"]))
        transcript = _clean_text(scene.get("transcript", ""))
        summary = _clean_text(scene.get("summary", transcript[:140]))
        scene_change = scene.get("scene_change", "hard")
        reference_mode = scene.get("reference_mode", "none")
        continue_from_previous = bool(scene.get("continue_from_previous", False))
        if scene_change == "hard":
            continue_from_previous = False
            reference_mode = "none"
        reference_strength = 0.0
        if reference_mode == "soft":
            reference_strength = config.REFERENCE_SOFT_STRENGTH
        elif reference_mode == "strong":
            reference_strength = config.REFERENCE_STRONG_STRENGTH
        timestamp_prefix = _timestamp_slug(start)
        image_name = f"{timestamp_prefix}_{index:04d}_{_slugify(summary)}.png"
        scene_number = int(scene.get("scene_number", index))
        normalized.append(
            {
                "scene_number": scene_number,
                "start": start,
                "end": end,
                "duration": max(0.1, end - start),
                "transcript": transcript,
                "summary": summary,
                "scene_change": scene_change,
                "continue_from_previous": continue_from_previous,
                "reference_mode": reference_mode,
                "reference_strength": reference_strength,
                "prompt": _clean_text(scene.get("prompt", transcript)),
                "image_name": image_name,
            }
        )
    return normalized


def _compose_prompt(scene: dict[str, Any]) -> str:
    scene_bits = [
        config.IMAGE_PROMPT_STYLE,
        scene["prompt"],
    ]
    if scene["continue_from_previous"]:
        scene_bits.append(
            "Preserve the same core art direction and subject continuity as the previous image, "
            "but allow composition changes that fit this shot."
        )
    else:
        scene_bits.append(
            "Treat this as a fresh scene with a clear new composition and no obligation to preserve the previous shot."
        )
    return " ".join(bit.strip() for bit in scene_bits if bit.strip())


def write_scene_plan(path: str, scenes: list[dict[str, Any]]) -> None:
    config.ensure_dirs()
    config.write_json(path, scenes)
    lines = []
    for scene in scenes:
        lines.extend(
            [
                f"Scene {scene['scene_number']} [{scene['start']:.2f}-{scene['end']:.2f}]",
                f"Change: {scene['scene_change']}, reference: {scene['reference_mode']}",
                f"Transcript: {scene['transcript']}",
                f"Prompt: {scene['prompt']}",
                "",
            ]
        )
    with open(config.SCENE_PLAN_TXT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")


def load_scene_plan(path: str | None = None) -> list[dict[str, Any]]:
    scene_plan_path = path or config.SCENE_PLAN_JSON
    with open(scene_plan_path, "r", encoding="utf-8-sig") as handle:
        scenes = json.load(handle)
    if not isinstance(scenes, list):
        raise ValueError(f"Scene plan JSON must contain a list of scenes: {scene_plan_path}")
    return _normalize_scenes(scenes)


def _find_previous_generated_image(scene_number: int, scenes: list[dict[str, Any]], image_dir: str) -> str | None:
    for previous_scene in reversed(scenes):
        if int(previous_scene["scene_number"]) >= int(scene_number):
            continue
        candidate = os.path.join(image_dir, previous_scene["image_name"])
        if os.path.exists(candidate):
            return candidate
    return None


async def _generate_one_image(
    runware: Runware,
    scene: dict[str, Any],
    image_dir: str,
    previous_image_path: str | None = None,
) -> str:
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, scene["image_name"])
    if os.path.exists(output_path):
        print(f"[image] skipping existing scene {scene['scene_number']}: {output_path}")
        return output_path

    inputs = None
    if scene["continue_from_previous"] and previous_image_path and os.path.exists(previous_image_path):
        inputs = IInputs(
            referenceImages=[
                IInputReference(
                    image=previous_image_path,
                    role="style",
                    tag="previous_scene",
                    strength=scene["reference_strength"],
                )
            ]
        )

    request = IImageInference(
        model=config.IMAGE_MODEL,
        positivePrompt=_compose_prompt(scene),
        negativePrompt=config.IMAGE_NEGATIVE_PROMPT,
        width=config.IMAGE_WIDTH,
        height=config.IMAGE_HEIGHT,
        outputFormat="PNG",
        outputType="URL",
        numberResults=1,
        includeCost=True,
        providerSettings=IOpenAIProviderSettings(
            quality=config.IMAGE_PROVIDER_QUALITY,
        ),
        inputs=inputs,
        taskUUID=str(uuid4()),
    )

    result = await runware.imageInference(request)
    if not result:
        raise RuntimeError(f"No image returned for scene {scene['scene_number']}")
    image = result[0]
    image_url = getattr(image, "imageURL", None)
    if not image_url:
        raise RuntimeError(f"Image URL missing for scene {scene['scene_number']}")

    with urllib.request.urlopen(image_url, timeout=120) as response:
        data = response.read()
    with open(output_path, "wb") as handle:
        handle.write(data)

    print(
        f"[image] wrote scene {scene['scene_number']} to {output_path} "
        f"(reference={scene['reference_mode']})"
    )
    return output_path


async def generate_images(
    scenes: list[dict[str, Any]],
    image_dir: str | None = None,
    scene_numbers: list[int] | None = None,
    random_test: bool = False,
) -> list[str]:
    if not config.RUNWARE_API_KEY:
        raise RuntimeError("RUNWARE_API_KEY not set in .env")

    config.ensure_dirs()
    target_dir = image_dir or config.IMAGE_DIR
    selected_scenes = list(scenes)
    if scene_numbers:
        selected = {int(number) for number in scene_numbers}
        selected_scenes = [scene for scene in scenes if int(scene["scene_number"]) in selected]
    if random_test:
        if not selected_scenes:
            raise ValueError("No scenes available for test generation.")
        selected_scenes = [random.choice(selected_scenes)]

    runware = Runware(api_key=config.RUNWARE_API_KEY)
    await runware.connect()

    outputs = []
    for scene in selected_scenes:
        previous_image_path = None
        if scene["continue_from_previous"]:
            previous_image_path = _find_previous_generated_image(
                int(scene["scene_number"]),
                scenes,
                target_dir,
            )
        output_path = await _generate_one_image(
            runware,
            scene,
            target_dir,
            previous_image_path=previous_image_path,
        )
        outputs.append(output_path)
    return outputs


def create_scene_plan(transcript_path: str | None = None) -> list[dict[str, Any]]:
    transcript_file = transcript_path or config.TRANSCRIPT_JSON
    segments = _load_segments(transcript_file)
    scenes = plan_scenes(segments)
    if len(scenes) != len(segments):
        raise ValueError(
            f"Scene planner returned {len(scenes)} scenes for {len(segments)} transcript segments."
        )
    write_scene_plan(config.SCENE_PLAN_JSON, scenes)
    print(f"[image] wrote scene plan: {config.SCENE_PLAN_JSON}")
    return scenes


def run_image_pipeline(
    transcript_path: str | None = None,
    image_dir: str | None = None,
    plan_only: bool = False,
    generate_only: bool = False,
    test: bool = False,
    scene_plan_path: str | None = None,
) -> dict[str, Any]:
    if generate_only:
        scenes = load_scene_plan(path=scene_plan_path)
        print(f"[image] loaded existing scene plan: {scene_plan_path or config.SCENE_PLAN_JSON}")
    else:
        scenes = create_scene_plan(transcript_path=transcript_path)
    image_paths: list[str] = []
    if not plan_only:
        image_paths = asyncio.run(
            generate_images(
                scenes,
                image_dir=image_dir,
                random_test=test,
            )
        )
    return {
        "scene_plan_path": scene_plan_path or config.SCENE_PLAN_JSON,
        "scene_count": len(scenes),
        "image_dir": image_dir or config.IMAGE_DIR,
        "images": image_paths,
        "test_mode": test,
    }
