"""AI-driven SFX planning and mixing for narrated videos."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from . import config

_FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/text/"
_MINIMAX_RETRIES = 3


# ── Freesound search & download ───────────────────────────────────────────────

def _search_freesound(query: str, max_duration: float) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "token": config.FREESOUND_API_KEY,
        "fields": "id,name,previews,duration",
        "filter": f"duration:[0 TO {max_duration}]",
        "sort": "rating_desc",
        "page_size": 5,
    }
    url = _FREESOUND_SEARCH_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "MythozAutoEditor/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8")).get("results", [])


def _download_preview(preview_url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    req = urllib.request.Request(preview_url, headers={"User-Agent": "MythozAutoEditor/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(dest_path, "wb") as handle:
        handle.write(data)


def _simplify_query(query: str) -> str:
    """Strip adjectives and keep the last 1-2 meaningful words as a fallback."""
    words = query.split()
    drop = {"ambient", "distant", "deep", "ominous", "cinematic", "heavy", "soft", "low", "high"}
    simplified = [w for w in words if w.lower() not in drop]
    return " ".join(simplified[-2:]) if simplified else words[-1]


def _fetch_sfx(query: str, dest_path: str) -> dict[str, Any] | None:
    last_word = query.strip().split()[-1]
    for attempt_query in [query, _simplify_query(query), last_word]:
        results = _search_freesound(attempt_query, config.SFX_MAX_DURATION_SECONDS)
        if results:
            break
    if not results:
        return None
    sound = results[0]
    preview_url = (
        sound.get("previews", {}).get("preview-hq-mp3")
        or sound.get("previews", {}).get("preview-lq-mp3")
    )
    if not preview_url:
        return None
    _download_preview(preview_url, dest_path)
    return {
        "freesound_id": sound["id"],
        "name": sound["name"],
        "duration": float(sound["duration"]),
        "preview_url": preview_url,
        "local_path": dest_path,
    }


# ── MiniMax SFX planner ───────────────────────────────────────────────────────

def _build_sfx_prompt(scenes: list[dict[str, Any]]) -> str:
    total_duration = max((s["end"] for s in scenes), default=0.0)
    budget = max(2, int(total_duration / config.SFX_MIN_INTERVAL_SECONDS))

    scene_lines = [
        f"Scene {s['scene_number']} [{s['start']:.2f}-{s['end']:.2f}s] "
        f"change={s['scene_change']} | {s['transcript'][:120]}"
        for s in scenes
    ]
    return "\n".join([
        "You are a sound designer for a YouTube narration video drawn in MS Paint style.",
        f"Total video duration: {total_duration:.1f} seconds.",
        f"SFX budget: assign sound effects to AT MOST {budget} scenes. Be selective — less is more.",
        "Priority targets: the first scene, the last scene, major reveals, punchlines, hard scene changes.",
        "Skip scenes that are mid-sentence, transitional, or quiet/reflective.",
        "For each SFX you assign, write a short specific Freesound search query (2-4 words, real sounds only).",
        "sfx_type choices: 'hit' = short punchy one-shot, 'ambient' = low background texture.",
        "volume: float 0.15 (very subtle) to 0.55 (prominent). Narration always stays on top.",
        "Output format — one line per scene, no extra text:",
        "  WITH sfx:    scene_number|||sfx_query|||sfx_type|||volume",
        "  WITHOUT sfx: scene_number|||skip",
        "Example:",
        "  1|||cinematic impact hit|||hit|||0.45",
        "  2|||skip",
        "  3|||cartoon boing|||hit|||0.30",
        "Scenes:",
        *scene_lines,
    ])


def _call_minimax(prompt: str) -> str:
    if not config.MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY not set in .env")
    body = {
        "model": config.MINIMAX_SCENE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a sound designer. Return plain text only in the exact requested "
                    "line format. No markdown, no commentary, no header row."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 4000,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        f"{config.MINIMAX_API_BASE.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choices = payload.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content
    raise RuntimeError("No content in MiniMax SFX planner response.")


def _parse_sfx_response(text: str, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scene_map = {s["scene_number"]: s for s in scenes}
    assignments: list[dict[str, Any]] = []
    last_sfx_time = -config.SFX_MIN_INTERVAL_SECONDS

    for line in text.splitlines():
        line = line.strip()
        if not line or "|||" not in line:
            continue
        parts = [p.strip() for p in line.split("|||")]
        if not parts[0].lstrip("-").isdigit():
            continue
        scene_number = int(parts[0])
        if len(parts) < 2 or parts[1].lower() == "skip":
            continue
        if len(parts) < 4:
            continue

        scene = scene_map.get(scene_number)
        if not scene:
            continue

        # Enforce minimum interval between SFX
        if scene["start"] - last_sfx_time < config.SFX_MIN_INTERVAL_SECONDS:
            continue

        sfx_query = parts[1]
        sfx_type = parts[2] if parts[2] in ("hit", "ambient") else "hit"
        try:
            volume = max(0.10, min(0.60, float(parts[3])))
        except ValueError:
            volume = config.SFX_HIT_VOLUME if sfx_type == "hit" else config.SFX_AMBIENT_VOLUME

        assignments.append({
            "scene_number": scene_number,
            "start": scene["start"],
            "sfx_query": sfx_query,
            "sfx_type": sfx_type,
            "volume": volume,
        })
        last_sfx_time = scene["start"]

    return assignments


def plan_sfx(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt = _build_sfx_prompt(scenes)
    last_exc: Exception | None = None
    for attempt in range(1, _MINIMAX_RETRIES + 1):
        try:
            text = _call_minimax(prompt)
            assignments = _parse_sfx_response(text, scenes)
            print(f"[sfx] planned {len(assignments)} sound effect(s).")
            return assignments
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[sfx] planner attempt {attempt} failed: {exc}")
    raise last_exc  # type: ignore[misc]


# ── Download ──────────────────────────────────────────────────────────────────

def download_sfx_for_plan(
    assignments: list[dict[str, Any]],
    sfx_dir: str,
) -> list[dict[str, Any]]:
    if not config.FREESOUND_API_KEY:
        raise RuntimeError("FREESOUND_API_KEY not set in .env")

    enriched: list[dict[str, Any]] = []
    for item in assignments:
        scene_num = item["scene_number"]
        safe_query = re.sub(r"[^a-z0-9]+", "_", item["sfx_query"].lower())[:40].strip("_")
        dest = os.path.join(sfx_dir, f"scene_{scene_num:04d}_{safe_query}.mp3")

        if os.path.exists(dest):
            print(f"[sfx] scene {scene_num}: reusing cached {os.path.basename(dest)}")
            enriched.append({**item, "local_path": dest})
            continue

        print(f"[sfx] scene {scene_num}: searching '{item['sfx_query']}'...")
        info = _fetch_sfx(item["sfx_query"], dest)
        if info is None:
            print(f"[sfx] scene {scene_num}: no results for '{item['sfx_query']}', skipping.")
            continue

        print(f"[sfx] scene {scene_num}: downloaded '{info['name']}' ({info['duration']:.1f}s)")
        enriched.append({**item, **info})

    return enriched


# ── FFmpeg mix ────────────────────────────────────────────────────────────────

def mix_sfx_into_audio(
    narration_path: str,
    sfx_plan: list[dict[str, Any]],
    output_path: str,
) -> str:
    ffmpeg = config.find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found.")

    ready = [s for s in sfx_plan if s.get("local_path") and os.path.exists(s["local_path"])]
    if not ready:
        print("[sfx] no SFX to mix — copying narration unchanged.")
        import shutil
        shutil.copy2(narration_path, output_path)
        return output_path

    inputs: list[str] = ["-i", narration_path]
    for item in ready:
        inputs += ["-i", item["local_path"]]

    filter_parts: list[str] = []
    sfx_labels: list[str] = []
    for idx, item in enumerate(ready, 1):
        delay_ms = int(item["start"] * 1000)
        label = f"sfx{idx}"
        filter_parts.append(
            f"[{idx}:a]adelay={delay_ms}|{delay_ms},volume={item['volume']}[{label}]"
        )
        sfx_labels.append(f"[{label}]")

    n_inputs = 1 + len(ready)
    mix_inputs = "[0:a]" + "".join(sfx_labels)
    filter_parts.append(
        f"{mix_inputs}amix=inputs={n_inputs}:duration=first:normalize=0[out]"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-y",
            *inputs,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    print(f"[sfx] mixed {len(ready)} SFX into: {output_path}")
    return output_path


# ── Top-level pipeline ────────────────────────────────────────────────────────

def run_sfx_pipeline(
    scene_plan_path: str | None = None,
    narration_path: str | None = None,
    output_path: str | None = None,
    sfx_dir: str | None = None,
    sfx_plan_path: str | None = None,
    plan_only: bool = False,
    mix_only: bool = False,
) -> dict[str, Any]:
    scene_plan_file = scene_plan_path or config.SCENE_PLAN_JSON
    narration_file = narration_path or config.FULL_AUDIO
    out_file = output_path or config.FULL_AUDIO_SFX
    sfx_folder = sfx_dir or config.SFX_DIR
    sfx_plan_file = sfx_plan_path or config.SFX_PLAN_JSON

    config.ensure_dirs()

    with open(scene_plan_file, "r", encoding="utf-8-sig") as handle:
        scenes = json.load(handle)

    if mix_only:
        with open(sfx_plan_file, "r", encoding="utf-8-sig") as handle:
            sfx_plan = json.load(handle)
        print(f"[sfx] loaded existing SFX plan: {sfx_plan_file}")
    else:
        assignments = plan_sfx(scenes)
        sfx_plan = download_sfx_for_plan(assignments, sfx_folder)
        config.write_json(sfx_plan_file, sfx_plan)
        print(f"[sfx] wrote SFX plan: {sfx_plan_file}")

    if plan_only:
        return {
            "sfx_plan_path": sfx_plan_file,
            "sfx_count": len(sfx_plan),
            "output_path": None,
        }

    mix_sfx_into_audio(narration_file, sfx_plan, out_file)
    return {
        "sfx_plan_path": sfx_plan_file,
        "sfx_count": len(sfx_plan),
        "output_path": out_file,
    }
