# Auto Editor

This project now has three separate parts:

1. `run_pipeline.py`
   Turns `script.txt` into narration audio and timestamp files.
2. `run_image_generate.py`
   Turns transcript segments into scene prompts and generated images.
3. `run_editor.py`
   Turns your scene images into a synced video using the narration audio.

## Setup

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Part 1: Audio + Timestamps

```powershell
python run_pipeline.py
```

For slightly slower story pacing, you can also run:

```powershell
python run_pipeline.py --pace 0.90
```

Outputs:

- `output/audio/full.wav`
- `output/transcripts/segments.json`
- `output/transcripts/segments.txt`
- `output/transcripts/segments.srt`

## Part 2: Scene Planning + Image Generation

Set these keys in `.env`:

- `MINIMAX_API_KEY` for scene planning with `MiniMax-M3`
- `RUNWARE_API_KEY` for image generation

Build only the scene plan:

```powershell
python run_image_generate.py --plan-only
```

Build the scene plan and generate images:

```powershell
python run_image_generate.py
```

Reuse the existing scene plan and generate images only:

```powershell
python run_image_generator.py --generate-only
```

Generate one random scene to test the image stack:

```powershell
python run_image_generator.py --generate-only --test
```

Outputs:

- `output/image_plan/scene_plan.json`
- `output/image_plan/scene_prompts.txt`
- `output/images/*.png`

The planner uses MiniMax M3 to group transcript rows into larger visual scenes and decides when to carry the previous image forward as a soft reference. If `MINIMAX_API_KEY` is missing, it falls back to a local heuristic planner.
Generated image filenames now begin with transcript timestamps like `00-12-340_0003_scene-name.png`, so you can use `run_editor.py --use-filename-timestamps` directly.

## Part 3: Editor

If your images are one-per-transcript-segment, run:

```powershell
python run_editor.py --images "D:\path\to\img"
```

If your image filenames already contain timestamps like `0-06.png`, `1-29.png`, run:

```powershell
python run_editor.py --images "D:\path\to\img" --use-filename-timestamps
```

Output:

- `output/video/final_video.mp4`

The editor now keeps each image static and adds only quick fade in/out transitions so the scene changes feel cleaner than a hard image swap.
If your FFmpeg build supports it, the editor will use NVIDIA GPU encoding (`h264_nvenc`) automatically and fall back to CPU encoding otherwise.
