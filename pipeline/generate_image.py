"""Scene planning and continuity-aware image generation for narrated videos."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import urllib.error
import urllib.request
from typing import Any
from uuid import uuid4

from runware import IImageInference, IInputs, IOpenAIProviderSettings, Runware

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
            "Every prompt must include a clear subject, a detailed setting/background, the emotion, and the idea being explained.",
            "Style requirements for every prompt:",
            "- very simple, childlike 2D doodle/whiteboard-explainer style, like a hand-drawn sketch video",
            "- stick-figure characters with round heads, simple dot or oval eyes, and simple eyebrows/mouths that clearly show emotion (happy, scared, sad, confused, excited, embarrassed, etc.)",
            "- thin black line arms and legs, no detailed clothing or accessories unless essential to the idea",
            "- background color: use a single flat soft, chill, pastel/muted color that gently fits the scene's mood, place, or time of day (e.g. soft pale blue, soft muted lavender, soft warm cream, soft dusty pink, soft pale grey-green); plain white is one valid option and should be the default only when the line has no specific place, time, or mood; never use harsh, strong, neon, or highly saturated colors",
            "- when this scene shares mood/location with the neighboring scene(s), shift the background to a nearby soft tone rather than an unrelated color, so the palette flows smoothly across the video instead of jumping abruptly",
            "- add a small handful of simple doodle objects in the background and/or foreground that fit the scene and keep it visually engaging (e.g. an outdoor scene gets a tree, a bird, a road, a house, a cloud; an indoor scene gets a bed, a window, a lamp) — pick 2-5 objects that make sense for this specific line",
            "- every object must stay as plain and simple as the main character: flat single-color shapes, no gradients, no shading, no fine detail, no busy or fully rendered scenery",
            "- every character and object must be filled with a flat color that clearly contrasts with the background so it never blends in (e.g. on a soft light blue background, draw the stick figure in white or another contrasting color, never a similar blue)",
            "- bold, thick, slightly uneven black outlines around characters and the simple objects present",
            "- keep character designs consistent (same look for the same recurring character) and keep the overall art style consistent and simple across scenes",
            "- 16:9 horizontal framing, with the character(s) as the clear focal point and the simple doodle objects placed around them",
            "- handwritten-style text, speech bubbles, or simple symbols (arrows, question marks, exclamation marks) only when they help explain the idea, and only if short and spelled correctly",
            "- make it funny and entertaining where it fits the line: exaggerated goofy facial expressions, silly poses, and occasional deliberately bad/wonky doodle details (lopsided objects, wobbly proportions, a silly sweat drop or motion line) — humor should never break the simple doodle art style or make the scene confusing",
            "Things to avoid in every prompt:",
            "- photorealism",
            "- 3D rendering",
            "- anime or Disney-style detailed faces",
            "- fully rendered, detailed, or busy backgrounds",
            "- glossy modern design or photographic lighting",
            "- realistic human anatomy",
            "- text, watermark, or logos unless explicitly part of the scene",
            "- harsh, strong, neon, or oversaturated colors",
            "- characters or objects whose color blends into the background",
            "- naming or describing any real, copyrighted, or trademarked character, franchise, movie, brand, or celebrity (e.g. Darth Vader, Mickey Mouse, Batman, a specific named actor) — the image generator will reject these even when only described, not named, if the description still matches that character's well-known design (e.g. 'a yellow creature with red cheeks, pointy ears, and a lightning-bolt tail' is instantly recognizable as Pikachu even with no name used). When a transcript line is itself about a famous misremembered visual detail of a copyrighted character (e.g. 'people think Pikachu's tail has a black tip'), the rejection risk comes from the signature color too, not just the extra accessories — change the creature's base color to something that character is not known for (e.g. orange or teal instead of yellow) in addition to dropping cheeks/lightning/other iconic accessories, while still keeping the one plot-relevant marking (the tail or ear tip color) so the joke still reads",
            "- for unmistakably shaped copyrighted items (e.g. a dark helmet with the exact silhouette of a famous movie villain, a glowing sword/blade tied to one franchise) — recoloring or renaming is not enough, because the rendered shape itself still reads as that character; for these, skip drawing the character or item at all and represent the joke abstractly instead, e.g. a speech-bubble doodle with the misremembered word crossed out by an X, a thought bubble, or simple unlabeled shapes with question marks",
            "- poses that could be misread as self-harm, even innocently (e.g. gripping or pulling at one's own head/hair with both hands, choking gestures) — for shock, disbelief, or fear use clearly comedic alternatives instead: hands raised near the face, jaw-drop, wide eyes, a single sweat drop, or a startled jump",
            "The drawings should feel like a simple, charming hand-drawn doodle explainer that's also funny and entertaining: easy to read at a glance, with just enough simple background/foreground detail and goofy humor to stay engaging without becoming busy or confusing.",
            f"Assume this persistent global style is always added separately too: {config.IMAGE_PROMPT_STYLE}",
            "For each scene, return a prompt that is specific enough to generate a different image for that timestamp.",
            "Output format rules:",
            "- Return plain text only, no markdown fences.",
            "- Return exactly one line per segment.",
            "- Use this exact delimiter between fields: |||",
            "- Each line must follow this exact format (replace each field with its real value):",
            "  1|||hard|||false|||none|||Man walks into hospital brain scanner|||Simple childlike doodle illustration of a white stick-figure man with a worried expression, walking past a simple scanner shape, a window, and a wall clock, soft muted pale blue background, thick black outlines",
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
_IMAGE_GENERATION_RETRIES = 4
_IMAGE_RETRY_BACKOFF_SECONDS = 3


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
                    f"Create a very simple, childlike 2D doodle illustration for this narration beat: {transcript}. "
                    "Show stick-figure character(s) with clear, exaggerated, goofy expressions, filled with a "
                    "color that contrasts clearly against the background, on a soft, chill, pastel/muted "
                    "background color that gently fits the scene's mood (plain white only if the line has no "
                    "specific place, time, or mood — never harsh or vibrant colors), with a small handful of "
                    "simple doodle objects around them that fit the scene, funny and entertaining but still "
                    "in the style of a hand-drawn whiteboard explainer video."
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
        normalized.append(
            {
                "scene_number": index,
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
        print(f"[image] scene {scene['scene_number']}: already exists, skipping")
        return output_path

    print(f"[image] scene {scene['scene_number']}: generating (mode={scene['reference_mode']})...")
    inputs = None
    if scene["continue_from_previous"] and previous_image_path and os.path.exists(previous_image_path):
        with open(previous_image_path, "rb") as _fh:
            _b64 = base64.b64encode(_fh.read()).decode("utf-8")
        inputs = IInputs(referenceImages=[f"data:image/png;base64,{_b64}"])

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

    # Build a sorted list of all scene numbers for prerequisite lookups
    all_nums_sorted = sorted(int(s["scene_number"]) for s in scenes)

    def _prerequisite_num(scene_number: int) -> int | None:
        try:
            idx = all_nums_sorted.index(scene_number)
        except ValueError:
            return None
        return all_nums_sorted[idx - 1] if idx > 0 else None

    # One asyncio.Event per scene — fires when that scene's image is on disk.
    # Pre-set for images that already exist so dependent scenes don't wait needlessly.
    done: dict[int, asyncio.Event] = {}
    for scene in scenes:
        ev = asyncio.Event()
        if os.path.exists(os.path.join(target_dir, scene["image_name"])):
            ev.set()
        done[int(scene["scene_number"])] = ev

    semaphore = asyncio.Semaphore(config.IMAGE_GENERATION_CONCURRENCY)
    failures: list[tuple[int, str]] = []

    async def _generate_scene(scene: dict[str, Any]) -> str | None:
        scene_num = int(scene["scene_number"])

        # Dependent scenes wait until their reference image is written before grabbing the semaphore
        if scene["continue_from_previous"]:
            prereq = _prerequisite_num(scene_num)
            if prereq is not None and prereq in done:
                await done[prereq].wait()

        prev_path = None
        if scene["continue_from_previous"]:
            prev_path = _find_previous_generated_image(scene_num, scenes, target_dir)

        result = None
        last_exc: Exception | None = None
        for attempt in range(1, _IMAGE_GENERATION_RETRIES + 1):
            try:
                async with semaphore:
                    result = await _generate_one_image(
                        runware, scene, target_dir, previous_image_path=prev_path
                    )
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"[image] scene {scene_num}: attempt {attempt} failed: {exc}")
                if attempt < _IMAGE_GENERATION_RETRIES:
                    await asyncio.sleep(_IMAGE_RETRY_BACKOFF_SECONDS * attempt)

        # Always release the event so dependent scenes don't wait forever on a failed
        # prerequisite — they'll fall back to generating without a reference image.
        done[scene_num].set()

        if last_exc is not None:
            print(
                f"[image] scene {scene_num}: giving up after {_IMAGE_GENERATION_RETRIES} attempt(s), "
                f"skipping. Reason: {last_exc}"
            )
            failures.append((scene_num, str(last_exc)))
            return None
        return result

    n_ind = sum(1 for s in selected_scenes if not s["continue_from_previous"])
    n_dep = sum(1 for s in selected_scenes if s["continue_from_previous"])
    print(
        f"[image] queuing {len(selected_scenes)} scene(s): "
        f"{n_ind} independent, {n_dep} dependent (each waits only for its direct reference) "
        f"— up to {config.IMAGE_GENERATION_CONCURRENCY} running at once..."
    )

    await asyncio.gather(*[_generate_scene(s) for s in selected_scenes])

    if failures:
        config.ensure_dirs()
        with open(config.FAILED_SCENES_LOG, "w", encoding="utf-8") as handle:
            for scene_num, reason in failures:
                scene = next(s for s in selected_scenes if int(s["scene_number"]) == scene_num)
                handle.write(f"Scene {scene_num}: {reason}\nPrompt: {_compose_prompt(scene)}\n\n")
        print(
            f"[image] {len(failures)} scene(s) failed and were skipped: "
            f"{[num for num, _ in failures]}. Details written to {config.FAILED_SCENES_LOG}"
        )

    return [
        os.path.join(target_dir, s["image_name"])
        for s in selected_scenes
        if os.path.exists(os.path.join(target_dir, s["image_name"]))
    ]


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
