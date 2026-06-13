"""Plan scenes from a transcript and generate one image per scene."""

import argparse

from pipeline import config, generate_image


def main():
    parser = argparse.ArgumentParser(
        description="Turn transcript segments into scene prompts and generated images."
    )
    parser.add_argument(
        "--transcript",
        default=config.TRANSCRIPT_JSON,
        help="Transcript JSON from run_pipeline.py or run_transcript.py.",
    )
    parser.add_argument(
        "--images",
        default=config.IMAGE_DIR,
        help="Output folder for generated scene images.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only build the scene plan and prompts without generating images.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Reuse an existing scene plan and generate images without rebuilding prompts.",
    )
    parser.add_argument(
        "--scene-plan",
        default=config.SCENE_PLAN_JSON,
        help="Scene plan JSON to reuse with --generate-only.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate just 1 random scene for a quick image-generation test.",
    )
    args = parser.parse_args()

    result = generate_image.run_image_pipeline(
        transcript_path=args.transcript,
        image_dir=args.images,
        plan_only=args.plan_only,
        generate_only=args.generate_only,
        test=args.test,
        scene_plan_path=args.scene_plan,
    )
    print(f"Wrote scene plan: {result['scene_plan_path']}")
    print(f"Wrote background plan: {result['background_plan_path']}")
    print(f"Planned scenes: {result['scene_count']}")
    if not args.plan_only:
        print(f"Wrote images to: {result['image_dir']}")
    if args.test:
        print("Generated test output for 1 random scene.")


if __name__ == "__main__":
    main()
