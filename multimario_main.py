import argparse
from collections import defaultdict
import shutil
import subprocess

import src.utils as utils
from src.utils.config_loader import ROOT_DIR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--game-names",
    "--games",
    "-g",
    type=str,
    nargs='+',
    help="Names of the games to work with (directories in data/)."
)
parser.add_argument(
    "--reparse-gc",
    action="store_true",
    help="Force parsing the gc.yaml log even if it's alreaddy been processed previously (as evidenced by the existence of gameinfo.yaml)"
)
parser.add_argument(
    "--recalibrate",
    action="store_true",
    help="Force opening the Calibration Studio for each game"
)
parser.add_argument(
    "--skip-video-analysis",
    action="store_true",
    help="Skip the video analysis, the heaviest part of the process. Useful if you only need to regenerate some post-processing stuff"
)
parser.add_argument(
    "--field-type",
    "--field",
    "-f",
    type=str,
    default=None,
    choices=utils.FIELD_TYPE_TO_FILE.keys(),
    help="Specify the field type for ALL games. Will try to override any saved ones. If blank, follow saved, or be asked separately for each unavailable one."
)
parser.add_argument(
    "--vision-type",
    "--vision",
    "-v",
    type=str,
    default=None,
    choices=utils.VISION_TYPE_TO_FILE.keys(),
    help="Specify the vision type for ALL games. Will try to override any saved ones. If blank, follow saved, or be asked separately for each unavailable one."
)
args = parser.parse_args()

# elaborate GC logs

def _arg_saved_or_ask(cli_arg, saved_path, input_question):
    if cli_arg is not None:
        return cli_arg
    elif saved_path.exists():
        return saved_path.read_text()
    else:
        return input(input_question)

for game_name in args.game_names:
    game_dir = ROOT_DIR / "data" / "games" / game_name
    field_type = _arg_saved_or_ask(args.field_type, game_dir / "field_type.txt", f"{game_name} field type? ")
    vision_type = _arg_saved_or_ask(args.vision_type, game_dir / "vision_type.txt", f"{game_name} vision type? ")  # it's way too early to care, but it's the cleanest way to write this
    if not (game_dir / "gameinfo.yaml").exists() or args.reparse_gc:
        subprocess.run([
            "python3", "sumgen_main.py",
            "-g", game_name,
            "-t", "gc",
            "-f", field_type,
            "-v", vision_type,
        ])


# calibrate everything first

ask_calibration_timestamp = True
side_hints = dict()

for game_name in args.game_names:
    game_dir = ROOT_DIR / "data" / "games" / game_name
    # field type is already handled by the gc processing
    section_names_to_do = utils.get_section_names_to_do(game_dir, "mario_", None)
    for section_name in section_names_to_do:
        section_letter = section_name[6:]
        section_dir = game_dir / section_name
        if not (section_dir / "camera_calbration.npz").exists() or args.recalibrate:
            if section_name != section_names_to_do[0]:
                print("Copy calibration from the first section?")
                print("Y: yes and check in studio (default)")
                print("s: yes and skip studio")
                print("n: no, calibrate manually")
                resp = input("Copy? ")
                if not "n" in resp.lower():  # then for sure we copy
                    shutil.copy(game_dir / section_names_to_do[0] / "camera_calbration.npz", section_dir / "camera_calbration.npz")
                    if "s" in resp.lower():  # then skip studio for this section
                        continue
                    # and y is the default, copy happened but go on and open studio to check
            if ask_calibration_timestamp:
                print("(enter nothing in the following to never be asked again)")
                calibration_timestamp_str = input(f"{game_name}/{section_name} calibration timestamp? ")
                if calibration_timestamp_str == "":
                    calibration_timestamp_str = None
                    ask_calibration_timestamp = False
                # else it's good
            else:
                calibration_timestamp_str = None
            while True:
                _cmd = [
                    "python3", "calibration_studio_main.py",
                    "-g", game_name,
                    "-s", section_letter,
                ]
                if calibration_timestamp_str is not None:
                    _cmd += ["-ct", calibration_timestamp_str]
                completed_process = subprocess.run(_cmd)
                if completed_process.returncode == 0:
                    break
                else:
                    print("Calibration Studio error (see above).")
                    resp = input("Retry? [y/N] ")
                    if not "y" in resp.lower():
                        break
        if not (section_dir / "side_hint.txt").exists() or args.recalibrate:
            print("Side hint help: [left-or-right] [color-or-number]. Example: left red. Leave empty to ignore.")
            _sh = input(f"{game_name}/{section_name} side hint? ")
            if _sh:
                side_hints[(game_name, section_name)] = _sh.split(" ")

# mario everything later

succeeded = defaultdict(list)
failed = []

for game_name in args.game_names:
    game_dir = ROOT_DIR / "data" / "games" / game_name
    # field_type is already set up by calibration studio by now
    section_names_to_do = utils.get_section_names_to_do(game_dir, "mario_", None)
    for section_name in section_names_to_do:
        if args.skip_video_analysis:
            succeeded[game_name].append(section_name)
        else:
            section_letter = section_name[6:]
            section_dir = game_dir / section_name
            _cmd = [
                "python3", "scripts/main.py",
                "-g", game_name,
                "-s", section_letter,
            ]
            if (game_name, section_name) in side_hints:
                _cmd += ["-sh"] + side_hints[(game_name, section_name)]
            completed_process = subprocess.run(_cmd)
            if completed_process.returncode == 0:
                succeeded[game_name].append(section_name)
            else:
                failed.append(f"{game_name}/{section_name}")

for game_name in succeeded:
    section_letters = [section_name[6:] for section_name in sorted(succeeded[game_name])]
    subprocess.run([
        "python3", "sumgen_main.py",
        "-g", game_name,
        "-s", *section_letters,
        "-t", "ap", "st", "ls"
    ])

if failed:
    print("Some analyses failed:")
    for f in failed:
        print(f" - {f}")
