import argparse
import subprocess

import scripts.multimario as multimario_base
from src.utils.config.paths import ROOT_DIR
from src.utils.coherence import arg_saved_or_ask

parser = argparse.ArgumentParser()
multimario_base.add_multimario_arguments(parser)
args = parser.parse_args()

succeeded, failed = multimario_base.go(args)

for game_name in succeeded:
    robot_type = arg_saved_or_ask(args.robot_type, ROOT_DIR / "data" / "games" / game_name / "robot_type.txt", f"{game_name} field type? ")
    subprocess.run([
        "python3", "sumgen_main.py",
        "-g", game_name,
        "-t", "ap",  # still need this to fill the section_params
    ])
    subprocess.run([
        "python3", "sincrolog_main.py",
        "-g", game_name,
        "-r", robot_type,
        "-t", "1", "2", "3", "4",
    ])

if failed:
    print("Some MARIOs failed:")
    for f in failed:
        print(f" - {f}")
