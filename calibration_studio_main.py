import argparse
import cv2
import numpy as np
from pathlib import Path

from src.vision.calibration_studio import CalibrationStudio
import src.utils as utils

def median_frame(video, n_frames, margin=0.02):
    """Per-pixel median of n_frames frames spread evenly over the video (skipping the first and last `margin` of it)."""
    n_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, int(n_total * (margin + (1 - 2 * margin) * i / max(1, n_frames - 1))))
        ok, frame = video.read()
        if ok:
            frames.append(frame)
    if not frames:
        raise ValueError("Could not read any frame for the median")
    stack = np.stack(frames)  # (n, H, W, 3) uint8
    out = np.empty_like(frames[0])
    for y0 in range(0, out.shape[0], 128):  # by strips: np.median would make a float64 copy of the whole stack
        strip = np.sort(stack[:, y0:y0 + 128], axis=0)
        out[y0:y0 + 128] = strip[len(frames) // 2]
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--section", "-s", type=str, required=True, help="Section to work with. ONLY ONE. Only the letter.")
parser.add_argument("--config", "-c", type=Path, default=None, help="Path to config.yaml (default: ./config.yaml)")
parser.add_argument("--game-name", "--game", "-g", type=str, default=None, help="Name of the game to work with (a directory in data/). If unspecified, will use the last one used in this project.")
parser.add_argument("--field-type", "--field", "-f", type=str, default=None, choices=utils.FIELD_TYPE_TO_FILE.keys(), help="The field this game was played on. Only specify the first time you work with this game.")
parser.add_argument("--recalibrate", action="store_true", help="Force opening the Calibration Studio to modify radial/homography")
parser.add_argument("--calibration-timestamp", "-ct", type=float, default=None, help="The point in time in the video (in seconds) to use for calibration (default: 20). Use this to find a frame without obstructions. Not compatible with -cm or -cf. Will not be saved.")
parser.add_argument("--calibration-median-frames", "-cm", type=int, default=1, help="Use the per-pixel median of this many frames, spread evenly over the video, instead of the single frame at -ct. Not compatible with -ct or -cf. People and robots standing on the lines disappear (the camera must not move during the video). 25 is a good value. Will not be saved.")
parser.add_argument("--calibration-frame", "-cf", type=str, default=None, help="Path to the image to use as source view (video frame). Avoid this if you have access to the video, use -ct or -cm instead. Not compatible with -ct or -cm. Use this only if you don't, e.g. if coming from streaming mode.")
args = parser.parse_args()
args.streaming = False  # for MarioConfig compatibility
args.vision_type = None  # for MarioConfig compatibility

config = utils.MarioConfig.from_yaml(args, update_game_history=False)
section_name = config.dir_names.mario_section_prefix + args.section
paths = config.get_paths(mario_section_name=section_name)

n_sources = (args.calibration_frame is not None) + (args.calibration_timestamp is not None) + (args.calibration_median_frames > 1)
if n_sources > 1:
    parser.error("use only one of -cf, -ct, -cm")

if args.calibration_frame is not None:
    calibration_frame = cv2.imread(str(args.calibration_frame))
    if calibration_frame is None:
        raise ValueError(f"Cannot read calibration frame: {args.calibration_frame}")
else:
    video = cv2.VideoCapture(str(paths.source_video))
    if not video.isOpened():
        raise ValueError(f"Cannot open video file: {paths.source_video}")
    if args.calibration_median_frames > 1:
        calibration_frame = median_frame(video, args.calibration_median_frames)
    else:
        video.set(cv2.CAP_PROP_POS_MSEC, (args.calibration_timestamp or 20.0) * 1000)
        _, calibration_frame = video.read()

calibrator = CalibrationStudio(calibration_frame, config)

_ = calibrator.try_load(paths.camera_calibration_npz)
calibrator.go()
if calibrator.result_ok:
    calibrator.save(paths.camera_calibration_npz)
    exit(0)
else:
    exit(1)
