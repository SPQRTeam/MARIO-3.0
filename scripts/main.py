#!/usr/bin/env python3
"""Main entry point for MARIO video processing."""

import argparse
from pathlib import Path
import sys
import logging
from ultralytics import YOLO
from collections import namedtuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils as utils
from src.core.tracker import Tracking
from src.ml.models.ollama_wrapper import OllamaWrapper
from src.ml.models.openai_chat_wrapper import commentary_llm_from_config

from src.utils.coherence import check_coherence_with_saved_or_update
from src.utils.side_hint import SideHint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MARIO - Multi-Agent RoboCup Interactive Observation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config.yaml
  python scripts/main.py

  # Run with custom config
  python scripts/main.py --config myconfig.yaml

  # Enable debug logging
  python scripts/main.py --verbose
        """
    )

    parser.add_argument(
        "--sections",
        "-s",
        type=str,
        nargs='+',
        help="Sections to work with. ONLY THE LETTERS. Defaults to all."
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)"
    )

    parser.add_argument(
        "--game-name",
        "--game",
        "-g",
        type=str,
        default=None,
        help="Name of the game to work with (a directory in data/). If unspecified, will use the last one used in this project."
    )

    parser.add_argument(
        "--field-type",
        "--field",
        "-f",
        type=str,
        default=None,
        choices=utils.FIELD_TYPE_TO_FILE.keys(),
        help="The field this game was played on. Only specify the first time you work with this game."
    )

    parser.add_argument(
        "--vision-type",
        "--vision",
        "-v",
        type=str,
        default=None,
        choices=utils.VISION_TYPE_TO_FILE.keys(),
        help="The set of vision models to load, typically determined by which robot(s) played. Only specify the first time you work with this game."
    )

    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Force opening the Calibration Studio to modify radial/homography"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Override output CSV path"
    )

    parser.add_argument(
        "--streaming",
        type=str,
        nargs=2,
        default=None,
        help="Provide a YouTube URL followed by a time range to activate STREAMING MODE. Example: --streaming https://www.youtube.com/watch?v=... 51:27-1:12:32"
    )

    parser.add_argument(
        "--side-hint",
        "-sh",
        type=str,
        nargs=2,
        default=None,
        help="Specify the number (int) OR field-player color (name) of EITHER the left or right team. Only specify this argument if doing exactly 1 section, but the hint is saved so future runs (even multi-section ones) will remember. Example: --side-team left blue",
    )

    parser.add_argument(
        "--calibration-timestamp",
        "-ct",
        type=float,
        default=20.0,
        help="The point in time in the video (in seconds) to use for calibration. Use this to find a frame without obstructions. Will not be saved.",
    )

    parser.add_argument(
        "--halp",
        action="store_true",
        help='''Display a "panic help" message useful if you've been away for too long, then exit.'''
    )

    return parser.parse_args()


def process_section(section_name, models, args, config, logger):
    logger.info(f"Now doing section: {section_name}")
    logger.info(f"Game directory: {config.game_dir}")
    logger.info(f"Video file: {config.video_path(section_name)}")
    logger.info(f"Features:")
    logger.info(f"  - Write CSV: {config.features.write_csv}")
    logger.info(f"  - Write videos: {config.features.write_videos}")
    logger.info(
        f"  - CNN colore: sotto {config.features.cnn_abstain_below_confidence:.0%} confidenza max → unknown (tracker)"
    )

    # Check video exists
    if not config.is_streaming and not config.video_path(section_name).exists():
        logger.error(f"Video file not found: {config.video_path(section_name)}")
        logger.error("Please check your config.yaml settings")
        return 1
    
    if args.side_hint:
        sh_tmp = SideHint.get_serialization(*args.side_hint)
    else:
        sh_tmp = None
    sh_tmp = check_coherence_with_saved_or_update(config.section_dir(section_name) / "side_hint.txt", sh_tmp, "side hint", "-sh", required=False)
    if sh_tmp:
        the_side_hint = SideHint.loads(sh_tmp)
    else:
        the_side_hint = None

    # Determine output path
    output_csv = args.output if args.output else config.mario_csv_path(section_name)
    logger.info(f"Output CSV: {output_csv}")

    # Initialize and run tracking
    logger.info("Initializing tracker...")
    print(logger.level)
    tracker = Tracking(config, section_name, models, the_side_hint, args.calibration_timestamp, logging.getLevelName(logger.level))

    logger.info("Starting video processing...")
    try:
        tracker.track_image(output_csv, args.recalibrate)
    finally:
        # Ensure writers/threads are flushed and released even on Ctrl+C.
        tracker._cleanup_runtime_outputs()

    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_csv}")

    if config.features.write_videos:
        logger.info(f"Annotated video: {config.mario_annotated_video_path(section_name)}")
        logger.info(f"Plan-view video: {config.mario_planview_path(section_name)}")


def main():
    """Main execution function."""
    args = parse_args()

    if args.halp:
        print("So, this is what you really gotta do:")
        print("")
        print("python3 main.py -g <GAME_NAME>")
        print("")
        print("For STREAMING MODE:")
        print("python3 main.py --stream <URL> <HH:MM:SS-HH:MM:SS> -g <GAME_NAME> -s <SECTION>")
        print("")
        print("Extra stuff:")
        print("- Omit -g to use the last game used in this project (including post-processing stuff such as sincrolog)")
        print("- Use -s to only do specific sections (A, B, ...)")
        print("- Use --recalibrate to force the calibration studio")
        exit(0)

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = utils.setup_logger(__name__, level=log_level)

    logger.info("=" * 60)
    logger.info("MARIO v2.0 - Starting video analysis")
    logger.info("=" * 60)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config or 'config.yaml'}")
    config = utils.MarioConfig.from_yaml(args)

    if config.is_streaming:
        if not args.sections:
            args.sections = ["S"]

    if len(args.sections) != 1 and args.side_hint is not None:
        raise ValueError("Only specify --side-hint with 1 section. The hint will then be saved for future runs (even multi-section ones)")

    # load models — commentary text LLM: OpenAI API or local Ollama (see models.commentary_llm_backend)
    _llm_backend = config.models.commentary_llm_backend.lower().strip()
    if _llm_backend == "openai":
        llm = commentary_llm_from_config(config.models)
        logger.info(
            "Commentary LLM: OpenAI (model=%s)",
            config.models.openai_model,
        )
    else:
        llm = OllamaWrapper(config.models.llm_model)
        logger.info("Commentary LLM: Ollama (model=%s)", config.models.llm_model)

    match_tts = None
    tts_on = config.tts.enabled
    geom_llm = config.commentating.geometry_llm_commentary_enabled
    periodic_llm = config.commentating.periodic_llm_commentary_enabled
    commentate = config.features.commentate
    # Commentary LLM runs inside MatchCommentaryTTS.maybe_llm_commentary, which needs match_tts set.
    # Previously match_tts existed only with tts.enabled, so geometry LLM never ran with audio off.
    if tts_on or (commentate and (geom_llm or periodic_llm)):
        from src.ml.models.match_commentary_tts import MatchCommentaryTTS

        match_tts = MatchCommentaryTTS(config)
        if tts_on:
            logger.info(
                "TTS commentary queue enabled (install: pip install edge-tts emoji; ffplay for playback)."
            )
        else:
            logger.info(
                "MatchCommentaryTTS: text/LLM only (tts.enabled=false); no speech synthesis."
            )

    Models = namedtuple("Models", ["yolo_v8", "yolo_v12", "ocr_vlm", "llm", "match_tts"])
    models = Models(
        yolo_v8 = YOLO(str(config.root_dir / config.vision_config.yolo_v8)),
        yolo_v12 = YOLO(str(config.root_dir / config.vision_config.yolo_v12)),
        ocr_vlm = OllamaWrapper(config.models.ocr_vlm_name),
        llm = llm,
        match_tts=match_tts,
    )

    sections_to_do = utils.get_sections_to_do(config.game_dir, config.dir_names.mario_section_prefix, args.sections)

    if config.is_streaming:
        for section_name in sections_to_do:
            config.section_dir(section_name).mkdir(parents=True, exist_ok=True)

    try:
        for section_name in sections_to_do:
            process_section(section_name, models, args, config, logger)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        logger.error("\nProcessing failed. Check the error message above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
