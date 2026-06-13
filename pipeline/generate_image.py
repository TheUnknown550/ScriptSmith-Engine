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


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_model_wrappers(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise RuntimeError("Planner response did not contain a JSON object.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise RuntimeError("Planner JSON response was not an object.")
    return parsed


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
            "Every prompt must include a visible background or setting, even if the background is simple.",
            "The setting should help the viewer immediately understand where the story is happening.",
            "Style requirements for every prompt:",
            "- extremely simple beginner drawings made in MS Paint",
            "- thick uneven black outlines",
            "- wobbly hand-drawn lines",
            "- stick figure humans with round heads and line bodies",
            "- simple dot eyes or circle eyes",
            "- very basic facial expressions",
            "- flat colors only",
            "- simple hand-drawn backgrounds with only the most important shapes and props",
            "- backgrounds should be clear but not detailed",
            "- keep lots of open space, but not an empty white void",
            "- occasional flat colors like green, brown, gray, red, yellow, orange, and blue",
            "- red arrows or red question marks only when helpful",
            "- handwritten text only when it helps explain the idea",
            "- if text appears, it must be short, spelled correctly, and easy to read",
            "- 16:9 horizontal YouTube frame",
            "Things to avoid in every prompt:",
            "- blank white background",
            "- realistic shading",
            "- 3D",
            "- cinematic lighting",
            "- polished illustration",
            "- anime or Disney style",
            "- realistic humans",
            "- highly detailed backgrounds",
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
            "  1|||hard|||false|||none|||Man walks into hospital brain scanner|||MS Paint drawing of stick figure sitting in a grey oval scanner room with a wall, floor line, machine cables, and simple hospital background, thick black lines",
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


def _build_background_prompt(segments: list[dict[str, Any]]) -> str:
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
            "You are designing one consistent visual world for a YouTube story.",
            "Read the full transcript and decide the clearest recurring setting or place language that should unify the scene images.",
            "The style is extremely simple beginner MS Paint art, not polished illustration.",
            "The background must help viewers understand the story, but it should still be simple, sparse, and easy to redraw across many scenes.",
            "Return exactly one JSON object and nothing else.",
            "Use this schema exactly:",
            '{'
            '"setting_name":"short place label",'
            '"setting_summary":"1-2 sentences describing the recurring world and why it fits the story",'
            '"background_prompt":"A single detailed image prompt for generating one reusable background reference image in simple MS Paint style",'
            '"recurring_elements":["item 1","item 2","item 3"],'
            '"palette":["color 1","color 2","color 3"]'
            '}',
            "Rules for background_prompt:",
            "- include a clear location with walls, ground, horizon, furniture, props, or landmarks when appropriate",
            "- keep it simple, flat, and hand-drawn",
            "- no white seamless studio background",
            "- no realistic shading or cinematic rendering",
            "- allow empty space for characters to be placed later",
            "- 16:9 horizontal frame",
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


def _normalize_background_plan(plan: dict[str, Any]) -> dict[str, Any]:
    recurring_elements = plan.get("recurring_elements", [])
    if not isinstance(recurring_elements, list):
        recurring_elements = []
    palette = plan.get("palette", [])
    if not isinstance(palette, list):
        palette = []
    return {
        "setting_name": _clean_text(plan.get("setting_name", "story world")),
        "setting_summary": _clean_text(plan.get("setting_summary", "")),
        "background_prompt": _clean_text(plan.get("background_prompt", "")),
        "recurring_elements": [_clean_text(item) for item in recurring_elements if _clean_text(str(item))],
        "palette": [_clean_text(item) for item in palette if _clean_text(str(item))],
    }


def _heuristic_background_plan(segments: list[dict[str, Any]]) -> dict[str, Any]:
    joined = " ".join(_clean_text(row.get("text", "")) for row in segments).lower()
    if any(term in joined for term in ("mountain", "summit", "peak", "climb", "cloud")):
        setting_name = "mountain summit world"
        setting_summary = (
            "A simple hand-drawn mountain world with cliffs, clouds, warning signs, and a steep path. "
            "It matches a story about struggle, danger, and losing ground."
        )
        recurring = ["steep mountain path", "clouds", "warning sign", "cliff edge"]
        palette = ["light blue", "gray", "green", "red"]
    elif any(term in joined for term in ("city", "street", "building", "office", "computer")):
        setting_name = "simple city story world"
        setting_summary = (
            "A flat hand-drawn city backdrop with a few buildings, windows, roads, and simple office or home props. "
            "It gives the story a recognizable real-world place without adding detail."
        )
        recurring = ["boxy buildings", "road", "desk or computer", "simple sky"]
        palette = ["light blue", "gray", "brown", "green"]
    else:
        setting_name = "simple story backdrop"
        setting_summary = (
            "A flexible hand-drawn setting with a floor line, horizon, a few props, and enough empty space for characters. "
            "It keeps the world readable while staying very simple."
        )
        recurring = ["floor line", "horizon line", "simple prop", "open space"]
        palette = ["light blue", "green", "gray", "brown"]

    background_prompt = (
        f"MS Paint beginner drawing of a {setting_name}, with {', '.join(recurring)}, "
        "thick uneven black outlines, wobbly lines, flat colors, simple readable background shapes, "
        "lots of open space for characters, 16:9 horizontal frame, intentionally amateur."
    )
    return {
        "setting_name": setting_name,
        "setting_summary": setting_summary,
        "background_prompt": background_prompt,
        "recurring_elements": recurring,
        "palette": palette,
    }


def plan_background(segments: list[dict[str, Any]]) -> dict[str, Any]:
    if not config.MINIMAX_API_KEY:
        return _heuristic_background_plan(segments)

    body = {
        "model": config.MINIMAX_SCENE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a visual world planner for an automated video pipeline. "
                    "Return exactly one JSON object with no markdown fences or extra text."
                ),
            },
            {
                "role": "user",
                "content": _build_background_prompt(segments),
            },
        ],
        "max_completion_tokens": 4000,
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
        print(f"[image] background planner unavailable, using heuristic fallback: {exc.code} {detail}")
        return _heuristic_background_plan(segments)
    except Exception as exc:  # noqa: BLE001
        print(f"[image] background planner unavailable, using heuristic fallback: {exc}")
        return _heuristic_background_plan(segments)

    _append_raw_minimax_payload(payload, "background plan")
    text = _extract_response_text(payload)
    try:
        return _normalize_background_plan(_extract_json_object(text))
    except Exception as exc:  # noqa: BLE001
        print(f"[image] background planner returned invalid JSON, using heuristic fallback: {exc}")
        return _heuristic_background_plan(segments)


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
                    f"Create an extremely simple MS Paint-style still image for this narration beat: {transcript}. "
                    "Choose a clear focal subject, a simple readable background, a few useful props, "
                    "and a mood that matches the line."
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


def _normalize_scenes(scenes: list[dict[str, Any]], background_plan: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    normalized = []
    background_plan = background_plan or {}
    recurring_elements = background_plan.get("recurring_elements", [])
    palette = background_plan.get("palette", [])
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
                "setting_name": _clean_text(scene.get("setting_name", background_plan.get("setting_name", ""))),
                "setting_summary": _clean_text(
                    scene.get("setting_summary", background_plan.get("setting_summary", ""))
                ),
                "background_elements": list(
                    scene.get("background_elements", recurring_elements)
                    if isinstance(scene.get("background_elements", recurring_elements), list)
                    else recurring_elements
                ),
                "background_palette": list(
                    scene.get("background_palette", palette)
                    if isinstance(scene.get("background_palette", palette), list)
                    else palette
                ),
                "image_name": image_name,
            }
        )
    return normalized


def _compose_prompt(scene: dict[str, Any], background_plan: dict[str, Any] | None = None) -> str:
    background_plan = background_plan or {}
    setting_name = _clean_text(scene.get("setting_name", background_plan.get("setting_name", "")))
    setting_summary = _clean_text(scene.get("setting_summary", background_plan.get("setting_summary", "")))
    recurring_elements = scene.get("background_elements", background_plan.get("recurring_elements", []))
    palette = scene.get("background_palette", background_plan.get("palette", []))
    scene_bits = [
        config.IMAGE_PROMPT_STYLE,
        scene["prompt"],
        (
            f"The scene takes place in the recurring setting '{setting_name}'. {setting_summary}"
            if setting_name or setting_summary
            else ""
        ),
        (
            "Keep a simple visible background with these recurring elements: "
            + ", ".join(_clean_text(str(item)) for item in recurring_elements if _clean_text(str(item)))
            if recurring_elements
            else "Keep a simple visible background that clearly establishes the place."
        ),
        (
            "Use this simple palette where helpful: "
            + ", ".join(_clean_text(str(item)) for item in palette if _clean_text(str(item)))
            if palette
            else ""
        ),
        "Do not use a blank white background. Include walls, ground, horizon, props, or landmarks as needed.",
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
                f"Setting: {scene.get('setting_name', '')}",
                f"Transcript: {scene['transcript']}",
                f"Prompt: {scene['prompt']}",
                "",
            ]
        )
    with open(config.SCENE_PLAN_TXT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")


def write_background_plan(path: str, background_plan: dict[str, Any]) -> None:
    config.ensure_dirs()
    config.write_json(path, background_plan)


def load_background_plan(path: str | None = None) -> dict[str, Any]:
    background_plan_path = path or config.BACKGROUND_PLAN_JSON
    with open(background_plan_path, "r", encoding="utf-8-sig") as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict):
        raise ValueError(f"Background plan JSON must contain an object: {background_plan_path}")
    return _normalize_background_plan(plan)


def load_scene_plan(path: str | None = None) -> list[dict[str, Any]]:
    scene_plan_path = path or config.SCENE_PLAN_JSON
    with open(scene_plan_path, "r", encoding="utf-8-sig") as handle:
        scenes = json.load(handle)
    if not isinstance(scenes, list):
        raise ValueError(f"Scene plan JSON must contain a list of scenes: {scene_plan_path}")
    background_plan = None
    if os.path.exists(config.BACKGROUND_PLAN_JSON):
        try:
            background_plan = load_background_plan()
        except Exception:  # noqa: BLE001
            background_plan = None
    return _normalize_scenes(scenes, background_plan=background_plan)


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
    background_plan: dict[str, Any] | None = None,
    background_image_path: str | None = None,
    previous_image_path: str | None = None,
) -> str:
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, scene["image_name"])
    if os.path.exists(output_path):
        print(f"[image] scene {scene['scene_number']}: already exists, skipping")
        return output_path

    print(f"[image] scene {scene['scene_number']}: generating (mode={scene['reference_mode']})...")
    reference_images: list[str] = []
    if background_image_path and os.path.exists(background_image_path):
        with open(background_image_path, "rb") as _fh:
            _b64 = base64.b64encode(_fh.read()).decode("utf-8")
        reference_images.append(f"data:image/png;base64,{_b64}")
    if scene["continue_from_previous"] and previous_image_path and os.path.exists(previous_image_path):
        with open(previous_image_path, "rb") as _fh:
            _b64 = base64.b64encode(_fh.read()).decode("utf-8")
        reference_images.append(f"data:image/png;base64,{_b64}")
    inputs = IInputs(referenceImages=reference_images) if reference_images else None

    request = IImageInference(
        model=config.IMAGE_MODEL,
        positivePrompt=_compose_prompt(scene, background_plan=background_plan),
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


async def _generate_background_image(
    runware: Runware,
    background_plan: dict[str, Any],
    output_path: str | None = None,
) -> str | None:
    if not background_plan.get("background_prompt"):
        return None

    background_path = output_path or config.BACKGROUND_IMAGE
    os.makedirs(os.path.dirname(background_path), exist_ok=True)
    if os.path.exists(background_path):
        print(f"[image] background reference already exists, skipping: {background_path}")
        return background_path

    prompt = " ".join(
        bit.strip()
        for bit in (
            config.IMAGE_PROMPT_STYLE,
            background_plan.get("background_prompt", ""),
            "Create only the reusable background environment with no dominant foreground character.",
            "Keep the place simple, readable, flat, and consistent with later scene images.",
            "Do not use a blank white background.",
        )
        if bit and str(bit).strip()
    )
    print("[image] generating shared background reference...")
    request = IImageInference(
        model=config.IMAGE_MODEL,
        positivePrompt=prompt,
        width=config.IMAGE_WIDTH,
        height=config.IMAGE_HEIGHT,
        outputFormat="PNG",
        outputType="URL",
        numberResults=1,
        includeCost=True,
        providerSettings=IOpenAIProviderSettings(
            quality=config.IMAGE_PROVIDER_QUALITY,
        ),
        taskUUID=str(uuid4()),
    )

    result = await runware.imageInference(request)
    if not result:
        raise RuntimeError("No image returned for background reference")
    image = result[0]
    image_url = getattr(image, "imageURL", None)
    if not image_url:
        raise RuntimeError("Image URL missing for background reference")

    with urllib.request.urlopen(image_url, timeout=120) as response:
        data = response.read()
    with open(background_path, "wb") as handle:
        handle.write(data)
    print(f"[image] wrote background reference to {background_path}")
    return background_path


async def generate_images(
    scenes: list[dict[str, Any]],
    image_dir: str | None = None,
    background_plan: dict[str, Any] | None = None,
    background_image_path: str | None = None,
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
    background_reference = await _generate_background_image(
        runware,
        background_plan or {},
        output_path=background_image_path or config.BACKGROUND_IMAGE,
    )

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

    async def _generate_scene(scene: dict[str, Any]) -> str:
        scene_num = int(scene["scene_number"])

        # Dependent scenes wait until their reference image is written before grabbing the semaphore
        if scene["continue_from_previous"]:
            prereq = _prerequisite_num(scene_num)
            if prereq is not None and prereq in done:
                await done[prereq].wait()

        prev_path = None
        if scene["continue_from_previous"]:
            prev_path = _find_previous_generated_image(scene_num, scenes, target_dir)

        async with semaphore:
            result = await _generate_one_image(
                runware,
                scene,
                target_dir,
                background_plan=background_plan,
                background_image_path=background_reference,
                previous_image_path=prev_path,
            )

        done[scene_num].set()
        return result

    n_ind = sum(1 for s in selected_scenes if not s["continue_from_previous"])
    n_dep = sum(1 for s in selected_scenes if s["continue_from_previous"])
    print(
        f"[image] queuing {len(selected_scenes)} scene(s): "
        f"{n_ind} independent, {n_dep} dependent (each waits only for its direct reference) "
        f"— up to {config.IMAGE_GENERATION_CONCURRENCY} running at once..."
    )

    await asyncio.gather(*[_generate_scene(s) for s in selected_scenes])

    return [
        os.path.join(target_dir, s["image_name"])
        for s in selected_scenes
        if os.path.exists(os.path.join(target_dir, s["image_name"]))
    ]


def create_scene_plan(transcript_path: str | None = None) -> list[dict[str, Any]]:
    transcript_file = transcript_path or config.TRANSCRIPT_JSON
    segments = _load_segments(transcript_file)
    background_plan = plan_background(segments)
    write_background_plan(config.BACKGROUND_PLAN_JSON, background_plan)
    print(f"[image] wrote background plan: {config.BACKGROUND_PLAN_JSON}")
    scenes = _normalize_scenes(plan_scenes(segments), background_plan=background_plan)
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
        background_plan = None
        if os.path.exists(config.BACKGROUND_PLAN_JSON):
            background_plan = load_background_plan()
        scenes = load_scene_plan(path=scene_plan_path)
        print(f"[image] loaded existing scene plan: {scene_plan_path or config.SCENE_PLAN_JSON}")
    else:
        background_plan = load_background_plan() if os.path.exists(config.BACKGROUND_PLAN_JSON) else None
        scenes = create_scene_plan(transcript_path=transcript_path)
        if os.path.exists(config.BACKGROUND_PLAN_JSON):
            background_plan = load_background_plan()
    image_paths: list[str] = []
    if not plan_only:
        image_paths = asyncio.run(
            generate_images(
                scenes,
                image_dir=image_dir,
                background_plan=background_plan,
                random_test=test,
            )
        )
    return {
        "scene_plan_path": scene_plan_path or config.SCENE_PLAN_JSON,
        "background_plan_path": config.BACKGROUND_PLAN_JSON,
        "scene_count": len(scenes),
        "image_dir": image_dir or config.IMAGE_DIR,
        "images": image_paths,
        "test_mode": test,
    }
