"""Plan and mix AI-selected sound effects into the narration audio."""

import argparse

from pipeline import config, sfx


def main():
    parser = argparse.ArgumentParser(
        description="Use MiniMax to pick SFX, search Freesound, and mix into narration audio."
    )
    parser.add_argument(
        "--scene-plan",
        default=config.SCENE_PLAN_JSON,
        help="Scene plan JSON produced by run_image_generate.py.",
    )
    parser.add_argument(
        "--narration",
        default=config.FULL_AUDIO,
        help="Narration WAV produced by run_pipeline.py.",
    )
    parser.add_argument(
        "--output",
        default=config.FULL_AUDIO_SFX,
        help="Output WAV path with SFX mixed in.",
    )
    parser.add_argument(
        "--sfx-dir",
        default=config.SFX_DIR,
        help="Folder to cache downloaded SFX previews.",
    )
    parser.add_argument(
        "--sfx-plan",
        default=config.SFX_PLAN_JSON,
        help="SFX plan JSON path.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan and download SFX without mixing into audio.",
    )
    parser.add_argument(
        "--mix-only",
        action="store_true",
        help="Skip planning — reuse existing SFX plan and mix only.",
    )
    args = parser.parse_args()

    result = sfx.run_sfx_pipeline(
        scene_plan_path=args.scene_plan,
        narration_path=args.narration,
        output_path=args.output,
        sfx_dir=args.sfx_dir,
        sfx_plan_path=args.sfx_plan,
        plan_only=args.plan_only,
        mix_only=args.mix_only,
    )

    print(f"SFX count:    {result['sfx_count']}")
    print(f"SFX plan:     {result['sfx_plan_path']}")
    if result["output_path"]:
        print(f"Mixed audio:  {result['output_path']}")


if __name__ == "__main__":
    main()
