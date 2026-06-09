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
DEFAULT_IMAGE_DIR = r"D:\mattc\Documents\Creative\Mythoz\vdos\What If The Moon Disappeared Tomorrow\img"

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
GEMINI_TTS_VOICE = "Achird"
GEMINI_TTS_STYLE = (
    "Read this as cinematic story narration in a dry, deadpan, slightly sarcastic "
    "documentary voice. Keep the pacing steady, controlled, and easy to follow. "
    "Speak slightly slower than normal conversation. Pause briefly at commas, pause "
    "more clearly at sentence endings, and let major paragraph transitions breathe. "
    "Do not rush key lines. Maintain the same delivery style, speaker identity, tone, "
    "mic distance, and tempo throughout the full narration: "
)
TTS_MAX_INPUT_CHARS = 24000
TTS_MAX_OUTPUT_TOKENS = 16384
TTS_OUTPUT_TOKENS_PER_SECOND = 32
TTS_REQUEST_MARGIN = 0.95
TTS_MAX_CHUNK_CHARS = 2800
TTS_MAX_CHUNK_WORDS = 220
TTS_JOIN_CROSSFADE_MS = 0
TTS_SHORT_PAUSE_MS = 120
TTS_MEDIUM_PAUSE_MS = 220
TTS_LONG_PAUSE_MS = 420
TTS_TARGET_RMS = 0.11
TTS_PEAK_LIMIT = 0.98
TTS_FINAL_ATEMPO = 1
TTS_API_RETRIES = 4
TTS_RETRY_BASE_DELAY_SECONDS = 2.0

WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"
TRANSCRIPT_MIN_SEGMENT_SECONDS = 1.0
TRANSCRIPT_TARGET_SEGMENT_SECONDS = 2.8
TRANSCRIPT_MAX_SEGMENT_SECONDS = 4.0

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
IMAGE_PROMPT_STYLE = (
    "Cinematic digital illustration with strong composition, coherent anatomy, "
    "high detail, dramatic but natural lighting, storybook realism, crisp focus, "
    "clean silhouettes, and a consistent visual identity across the full video."
)
IMAGE_NEGATIVE_PROMPT = (
    "blurry, low detail, deformed anatomy, duplicated subjects, extra limbs, "
    "text, watermark, logo, flat lighting, muddy colors, collage layout, split screen"
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
