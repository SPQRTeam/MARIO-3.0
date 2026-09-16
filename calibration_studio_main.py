import argparse
import cv2
from pathlib import Path

from src.vision.calibration_studio import CalibrationStudio
import src.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("--section", "-s", type=str, required=True, help="Section to work with. ONLY ONE. Only the letter.")
parser.add_argument("--config", "-c", type=Path, default=None, help="Path to config.yaml (default: ./config.yaml)")
parser.add_argument("--game-name", "--game", "-g", type=str, default=None, help="Name of the game to work with (a directory in data/). If unspecified, will use the last one used in this project.")
parser.add_argument("--field-type", "--field", "-f", type=str, default=None, choices=utils.FIELD_TYPE_TO_FILE.keys(), help="The field this game was played on. Only specify the first time you work with this game.")
parser.add_argument("--recalibrate", action="store_true", help="Force opening the Calibration Studio to modify radial/homography")
parser.add_argument("--calibration-timestamp", "-ct", type=float, default=20.0, help="The point in time in the video (in seconds) to use for calibration. Use this to find a frame without obstructions. Will not be saved.")
parser.add_argument("--calibration-frame", "-cf", type=str, default=None, help="Path to the image to use as source view (video frame). Avoid this if you have access to the video, use -ct instead. Use this only if you don't, e.g. if coming from streaming mode.")
args = parser.parse_args()
args.streaming = False  # for MarioConfig compatibility
args.vision_type = None  # for MarioConfig compatibility

config = utils.MarioConfig.from_yaml(args, update_game_history=False)
section_name = config.dir_names.mario_section_prefix + args.section
paths = config.get_paths(mario_section_name=section_name)

if args.calibration_frame is not None:
    raise NotImplementedError("This is easy to implement but it's best to use -ct if possible")
elif args.calibration_timestamp is not None:
    video = cv2.VideoCapture(str(paths.source_video))
    if not video.isOpened():
        raise ValueError(f"Cannot open video file: {paths.source_video}")
    video.set(cv2.CAP_PROP_POS_MSEC, args.calibration_timestamp * 1000)
    _, calibration_frame = video.read()
else:
    raise AssertionError("Specify only one of -ct or -cf")

calibrator = CalibrationStudio(calibration_frame, config)

_ = calibrator.try_load(paths.camera_calibration_npz)
calibrator.go()
if calibrator.result_ok:
    calibrator.save(paths.camera_calibration_npz)
    exit(0)
else:
    exit(1)
