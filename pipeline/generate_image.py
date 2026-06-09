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
            "You are going to generate images for a YouTube script.",
            "Your job is to read the script carefully and create a separate image plan for each timestamp.",
            "Each image must visually illustrate what the narrator is saying at that exact moment.",
            "Do not create random images. Every image should feel like a simple visual explanation of the current line.",
            "Turn this narrated transcript into an image plan for a YouTube video.",
            "Create exactly one scene per transcript segment.",
            "Do not merge segments.",
            "Do not split segments.",
            "Return exactly the same number of scenes as the transcript segments provided.",
            "Each scene_number must match the transcript segment index.",
            "The generated image prompts should be written for ChatGPT Image 2 style image generation.",
            "The prompts must preserve consistency across scenes while still making each image different.",
            "Use continue_from_previous=true only when the next image should visibly preserve the same subject, location, and style continuity.",
            "Use scene_change=hard and reference_mode=none for major visual resets like new place, new time, new action, or a big reveal.",
            "Use scene_change=soft with reference_mode=soft or strong only for neighboring shots that should feel connected.",
            "Write prompts for still-image generation, not for animation or video.",
            "Every prompt must include a clear subject, the simple environment, the emotion, and the idea being explained.",
            "Style requirements for every prompt:",
            "- extremely simple beginner drawings made in MS Paint",
            "- white background",
            "- thick uneven black outlines",
            "- wobbly hand-drawn lines",
            "- stick figure humans with round heads and line bodies",
            "- simple dot eyes or circle eyes",
            "- very basic facial expressions",
            "- flat colors only",
            "- mostly empty white space",
            "- occasional flat colors like green, brown, gray, red, yellow, orange, and blue",
            "- red arrows or red question marks only when helpful",
            "- handwritten text only when it helps explain the idea",
            "- if text appears, it must be short, spelled correctly, and easy to read",
            "- 16:9 horizontal YouTube frame",
            "Things to avoid in every prompt:",
            "- realistic shading",
            "- 3D",
            "- cinematic lighting",
            "- polished illustration",
            "- anime or Disney style",
            "- realistic humans",
            "- detailed backgrounds",
            "- complex textures",
            "- glossy modern design",
            "The drawings should feel amateur, funny, simple, and intentionally bad, like a beginner drew them quickly in Paint.",
            f"Assume this persistent global style is always added separately too: {config.IMAGE_PROMPT_STYLE}",
            "For each scene, return a prompt that is specific enough to generate a different image for that timestamp.",
            "Output format rules:",
            "- Return plain text only, no markdown fences.",
            "- Return exactly one line per segment.",
            "- Use this exact delimiter between fields: |||",
            "- Each line must follow this exact format (replace each field with its real value):",
            "  1|||hard|||false|||none|||Man walks into hospital brain scanner|||MS Paint drawing of stick figure sitting in a grey oval scanner, white background, thick black lines",
            "- scene_change: use the word hard or soft (not the label 'scene_change')",
            "- continue_from_previous: use the word true or false (not the label 'continue_from_previous')",
            "- reference_mode: use the word none, soft, or strong (not the label 'reference_mode')",
            "- Do not output a header row. Do not use field names as values.",
            "- Do not include the delimiter sequence ||| inside summary or prompt",
            "- Do not include any extra commentary before or after the lines",
            "Transcript:",
            transcript_block,
        ]
    )


def _chunk_segments(segments: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [segments[index : index + batch_size] for index in range(0, len(segments), batch_size)]


def _extract_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        text_parts.append(str(item["text"]))
                    elif item.get("text"):
                        text_parts.append(str(item["text"]))
                elif isinstance(item, str) and item.strip():
                    text_parts.append(item)
            joined = "\n".join(part for part in text_parts if part.strip()).strip()
            if joined:
                return joined
        reasoning_details = message.get("reasoning_details", [])
        if isinstance(reasoning_details, list):
            text_parts = []
            for item in reasoning_details:
                if isinstance(item, dict) and item.get("text"):
                    text_parts.append(str(item["text"]))
            joined = "\n".join(part for part in text_parts if part.strip()).strip()
            if joined:
                return joined
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    raise RuntimeError("Planner response did not contain text output.")


def _strip_model_wrappers(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _write_raw_minimax_response(text: str) -> None:
    config.ensure_dirs()
    with open(config.MINIMAX_RAW_RESPONSE, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_raw_minimax_payload(payload: dict[str, Any]) -> None:
    config.ensure_dirs()
    with open(config.MINIMAX_RAW_PAYLOAD, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _append_raw_minimax_payload(payload: dict[str, Any], batch_label: str) -> None:
    config.ensure_dirs()
    with open(config.MINIMAX_RAW_PAYLOAD, "a", encoding="utf-8") as handle:
        handle.write(f"\n\n=== {batch_label} ===\n")
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _parse_minimax_scene_lines(text: str, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = _strip_model_wrappers(text)
    _write_raw_minimax_response(cleaned)
    all_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    lines = [
        line for line in all_lines
        if line.count("|||") == 5 and line.split("|||")[0].strip().lstrip("-").isdigit()
    ]
    if len(lines) != len(segments):
        raise RuntimeError(
            f"MiniMax returned {len(lines)} scene lines for {len(segments)} transcript segments. "
            f"Raw response saved to {config.MINIMAX_RAW_RESPONSE}"
        )

    scenes = []
    for row, line in zip(segments, lines):
        parts = [part.strip() for part in line.split("|||")]
        if len(parts) != 6:
            raise RuntimeError(
                f"MiniMax scene line did not have 6 fields: {line}\n"
                f"Raw response saved to {config.MINIMAX_RAW_RESPONSE}"
            )
        scene_number_text, scene_change, continue_text, reference_mode, summary, prompt = parts
        scene_number = int(scene_number_text)
        continue_from_previous = continue_text.lower() == "true"
        if scene_change not in {"hard", "soft"}:
            raise RuntimeError(f"Invalid scene_change from MiniMax: {scene_change}")
        if reference_mode not in {"none", "soft", "strong"}:
            raise RuntimeError(f"Invalid reference_mode from MiniMax: {reference_mode}")
        scenes.append(
            {
                "scene_number": scene_number,
                "start": float(row["start"]),
                "end": float(row["end"]),
                "transcript": _clean_text(row.get("text", "")),
                "summary": _clean_text(summary),
                "scene_change": scene_change,
                "continue_from_previous": continue_from_previous,
                "reference_mode": reference_mode,
                "prompt": _clean_text(prompt),
            }
        )
    return scenes


def _plan_with_minimax_batch(
    segments: list[dict[str, Any]],
    batch_index: int,
    batch_total: int,
) -> list[dict[str, Any]]:
    if not config.MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY not configured.")

    body = {
        "model": config.MINIMAX_SCENE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a scene planner for an automated video pipeline. "
                    "Return plain text only in the exact requested line format. "
                    "Do not wrap it in markdown. "
                    "Do not include explanation text before or after the output."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        f"This is batch {batch_index} of {batch_total}.",
                        "Only return lines for the segments included in this batch.",
                        _build_planner_prompt(segments),
                    ]
                ),
            },
        ],
        "max_completion_tokens": 8000,
        "temperature": 0.2,
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
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiniMax planner request failed: {exc.code} {detail}") from exc

    _append_raw_minimax_payload(payload, f"batch {batch_index}/{batch_total}")
    text = _extract_response_text(payload)
    return _parse_minimax_scene_lines(text, segments)


_MINIMAX_BATCH_RETRIES = 3


def _plan_with_minimax(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config.ensure_dirs()
    with open(config.MINIMAX_RAW_PAYLOAD, "w", encoding="utf-8") as handle:
        handle.write("")
    with open(config.MINIMAX_RAW_RESPONSE, "w", encoding="utf-8") as handle:
        handle.write("")

    all_scenes = []
    batches = _chunk_segments(segments, config.MINIMAX_PLANNER_BATCH_SIZE)
    batch_total = len(batches)
    for batch_offset, batch_segments in enumerate(batches, 1):
        last_exc: Exception | None = None
        for attempt in range(1, _MINIMAX_BATCH_RETRIES + 1):
            try:
                batch_scenes = _plan_with_minimax_batch(
                    batch_segments,
                    batch_index=batch_offset,
                    batch_total=batch_total,
                )
                if attempt > 1:
                    print(f"[image] batch {batch_offset}/{batch_total} succeeded on attempt {attempt}.")
                all_scenes.extend(batch_scenes)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"[image] batch {batch_offset}/{batch_total} attempt {attempt} failed: {exc}")
        if last_exc is not None:
            raise last_exc
    return all_scenes


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


def _visual_thread_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z]{4,}", text.lower()))
    stop_words = {
        "about",
        "after",
        "almost",
        "because",
        "being",
        "brain",
        "could",
        "every",
        "first",
        "from",
        "have",
        "just",
        "like",
        "more",
        "most",
        "only",
        "other",
        "people",
        "right",
        "same",
        "some",
        "that",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "until",
        "very",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
    }
    return {token for token in tokens if token not in stop_words}


def _same_visual_thread(previous_scene: dict[str, Any], transcript: str) -> bool:
    previous_tokens = _visual_thread_tokens(previous_scene["transcript"])
    current_tokens = _visual_thread_tokens(transcript)
    shared = previous_tokens & current_tokens
    if len(shared) >= 2:
        return True

    anchor_terms = {
        "brain",
        "hospital",
        "scanner",
        "scientists",
        "myth",
        "city",
        "streetlight",
        "memory",
        "neurons",
        "attention",
        "stress",
        "practice",
        "sleep",
        "reading",
        "scrolling",
        "writer",
        "pianist",
        "surgeon",
    }
    if (previous_tokens & anchor_terms) and (current_tokens & anchor_terms):
        return True

    previous_summary = previous_scene.get("summary", "").lower()
    current_text = transcript.lower()
    if any(
        phrase in previous_summary and phrase in current_text
        for phrase in (
            "10%",
            "brain",
            "city",
            "scientists",
            "memory",
            "attention",
            "practice",
        )
    ):
        return True

    return False


def _plan_with_heuristics(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    for row in segments:
        transcript = _clean_text(row.get("text", ""))
        summary = _clean_text(transcript[:160])
        previous = scenes[-1] if scenes else None
        scene_change = "hard"
        continue_from_previous = False
        reference_mode = "none"
        if previous is not None and (
            _same_visual_thread(previous, transcript)
            or not _heuristic_scene_change(previous["transcript"], transcript)
        ):
            scene_change = "soft"
            continue_from_previous = True
            reference_mode = "strong" if _same_visual_thread(previous, transcript) else "soft"
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
        image_name = f"{timestamp_prefix}.png"
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
