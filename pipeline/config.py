"""Shared configuration for the local audio and editor pipeline."""

import json
import os
import glob
import shutil
import subprocess

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(ROOT, "script.txt")
OUTPUT_DIR = os.path.join(ROOT, "output")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "transcripts")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
IMAGE_PLAN_DIR = os.path.join(OUTPUT_DIR, "image_plan")
SFX_DIR = os.path.join(OUTPUT_DIR, "sfx")
SFX_PLAN_DIR = os.path.join(OUTPUT_DIR, "sfx_plan")

TTS_SAMPLE_RATE = 24000
WORDS_PER_SECOND = 2.5

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com/v1")
MINIMAX_SCENE_MODEL = os.environ.get("MINIMAX_SCENE_MODEL", "MiniMax-M3")
RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY", "")
FREESOUND_API_KEY = os.environ.get("FREESOUND_API_KEY", "")
FREESOUND_CLIENT_ID = os.environ.get("FREESOUND_CLIENT_ID", "")
FREESOUND_CLIENT_SECRET = os.environ.get("FREESOUND_CLIENT_SECRET", "")
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_VOICE = "Orus"
GEMINI_TTS_STYLE = (
    "You are a confident, engaging YouTube narrator for a documentary-style video. "
    "Deliver the narration with consistent energy — warm, authoritative, and clear throughout. "
    "Speak at a steady, natural pace: not too fast, not too slow. "
    "Emphasise key words naturally without overdoing it. "
    "Pause briefly at commas, more clearly at full stops, and give paragraph breaks a full breath. "
    "Never rush. Never trail off. Keep the same voice, tone, and mic presence from first word to last. "
    "Do not add any intro, outro, or commentary — read only the text given: "
)
TTS_MAX_INPUT_CHARS = 24000
TTS_MAX_OUTPUT_TOKENS = 16384
TTS_OUTPUT_TOKENS_PER_SECOND = 32
TTS_REQUEST_MARGIN = 0.95
TTS_MAX_CHUNK_CHARS = 2800
TTS_MAX_CHUNK_WORDS = 220
TTS_JOIN_CROSSFADE_MS = 80
TTS_SHORT_PAUSE_MS = 150
TTS_MEDIUM_PAUSE_MS = 300
TTS_LONG_PAUSE_MS = 560
TTS_TARGET_RMS = 0.28
TTS_PEAK_LIMIT = 0.98
TTS_FINAL_ATEMPO = 1
TTS_API_RETRIES = 4
TTS_RETRY_BASE_DELAY_SECONDS = 2.0

WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"
TRANSCRIPT_MIN_SEGMENT_SECONDS = 1.0
TRANSCRIPT_TARGET_SEGMENT_SECONDS = 2.5
TRANSCRIPT_MAX_SEGMENT_SECONDS = 3.0

FULL_AUDIO = os.path.join(AUDIO_DIR, "full.wav")
FULL_AUDIO_SFX = os.path.join(AUDIO_DIR, "full_with_sfx.wav")
SFX_PLAN_JSON = os.path.join(SFX_PLAN_DIR, "sfx_plan.json")
SFX_MAX_DURATION_SECONDS = 8.0
SFX_MIN_INTERVAL_SECONDS = 20.0
SFX_HIT_VOLUME = 0.45
SFX_AMBIENT_VOLUME = 0.20
TRANSCRIPT_JSON = os.path.join(TRANSCRIPTS_DIR, "segments.json")
TRANSCRIPT_TXT = os.path.join(TRANSCRIPTS_DIR, "segments.txt")
TRANSCRIPT_SRT = os.path.join(TRANSCRIPTS_DIR, "segments.srt")
FINAL_VIDEO = os.path.join(VIDEO_DIR, "final_video.mp4")
SCENE_PLAN_JSON = os.path.join(IMAGE_PLAN_DIR, "scene_plan.json")
SCENE_PLAN_TXT = os.path.join(IMAGE_PLAN_DIR, "scene_prompts.txt")
MINIMAX_RAW_RESPONSE = os.path.join(IMAGE_PLAN_DIR, "minimax_raw_response.txt")
MINIMAX_RAW_PAYLOAD = os.path.join(IMAGE_PLAN_DIR, "minimax_raw_payload.json")
FAILED_SCENES_LOG = os.path.join(IMAGE_PLAN_DIR, "failed_scenes.txt")
IMAGE_PROMPT_STYLE = (
    "Very simple, childlike 2D doodle illustration, like a hand-drawn whiteboard "
    "explainer video sketch: stick-figure characters with round heads, plain dot "
    "or oval eyes, simple eyebrows and a simple mouth for emotion, thin black "
    "line limbs, no detailed clothing. Background is a single flat soft, chill, "
    "pastel/muted color chosen to gently match the scene's mood and setting "
    "(e.g. soft pale blue for calm or daytime, soft muted lavender or navy-grey "
    "for night, soft warm cream for cozy, soft pale grey-green for outdoors, "
    "soft dusty pink or peach for happy/embarrassed) — plain white is one valid "
    "option and should be the default only when the line has no specific "
    "place, time, or mood. Never use harsh, strong, or highly saturated colors; "
    "every background color stays soft, clean, and low-saturation. Include a "
    "small handful of simple doodle objects in the background and/or "
    "foreground that fit the scene (e.g. a tree, a bird, a house, a road, a "
    "cloud, a lamp) to keep the image engaging, but keep every object just as "
    "plain and simple as the main character — flat shapes, no gradients, no "
    "shading, no fine detail. Every character and object must be filled with a "
    "color that visibly contrasts with the background so nothing blends into "
    "it (e.g. if the background is soft light blue, draw the stick figure in "
    "white or another contrasting flat color, never a similar blue). Bold "
    "uneven black outlines, minimal flat colors, 16:9 framing, "
    "consistent simple visual identity across the full video. When moving "
    "between neighboring scenes that share the same mood or location, shift "
    "the background color smoothly to a nearby soft tone rather than jumping "
    "to an unrelated color, so the video's palette feels like a gentle, "
    "continuous flow rather than abrupt jumps. Tone should be "
    "funny and entertaining: exaggerated goofy expressions, silly poses, and "
    "occasional deliberately bad/wonky doodle details (e.g. lopsided objects, "
    "wobbly proportions, a silly sweat drop or motion line) where it fits the "
    "moment, without breaking the simple doodle art style."
)
IMAGE_NEGATIVE_PROMPT = (
    "photorealistic, 3D render, anime, Disney style, fully rendered detailed "
    "background, busy background, gradients, blurry, low detail, "
    "deformed anatomy, duplicated subjects, extra limbs, text, watermark, "
    "logo, muddy colors, collage layout, split screen, intricate details, "
    "harsh vibrant neon colors, oversaturated colors, bright primary colors, "
    "characters or objects blending into the background color"
)
IMAGE_MODEL = os.environ.get("RUNWARE_IMAGE_MODEL", "openai:gpt-image@2")
IMAGE_PROVIDER_QUALITY = os.environ.get("RUNWARE_IMAGE_QUALITY", "low")
IMAGE_PROVIDER_MODERATION = os.environ.get("RUNWARE_IMAGE_MODERATION", "auto")
IMAGE_WIDTH = 1792
IMAGE_HEIGHT = 1024
SCENE_TARGET_SECONDS = 7.5
SCENE_MAX_SECONDS = 11.0
SCENE_MIN_SECONDS = 3.0
REFERENCE_SOFT_STRENGTH = 0.35
REFERENCE_STRONG_STRENGTH = 0.60
MINIMAX_PLANNER_BATCH_SIZE = 20
IMAGE_GENERATION_CONCURRENCY = 5

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
FADE_DURATION = 0.2


def ensure_dirs():
    for path in (
        OUTPUT_DIR,
        AUDIO_DIR,
        TRANSCRIPTS_DIR,
        VIDEO_DIR,
        TEMP_DIR,
        IMAGE_DIR,
        IMAGE_PLAN_DIR,
        SFX_DIR,
        SFX_PLAN_DIR,
    ):
        os.makedirs(path, exist_ok=True)


def read_script(path=None):
    script_path = path or SCRIPT_PATH
    with open(script_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _find_binary(name):
    found = shutil.which(name)
    if found:
        return found
    candidates = []
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        candidates.append(os.path.join(local, "Microsoft", "WinGet", "Links", f"{name}.exe"))
        candidates += glob.glob(
            os.path.join(
                local,
                "Microsoft",
                "WinGet",
                "Packages",
                "Gyan.FFmpeg*",
                "**",
                "bin",
                f"{name}.exe",
            ),
            recursive=True,
        )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def find_ffmpeg():
    return _find_binary("ffmpeg")


def find_ffprobe():
    return _find_binary("ffprobe")


def find_h264_encoder():
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return "libx264"

    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # noqa: BLE001
        return "libx264"

    encoders = result.stdout
    if "h264_nvenc" in encoders:
        return "h264_nvenc"
    if "h264_amf" in encoders:
        return "h264_amf"
    if "h264_qsv" in encoders:
        return "h264_qsv"
    return "libx264"
