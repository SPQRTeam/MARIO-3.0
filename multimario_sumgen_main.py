import argparse
import subprocess

import scripts.multimario as multimario_base

parser = argparse.ArgumentParser()
multimario_base.add_multimario_arguments(parser)
args = parser.parse_args()

succeeded, failed = multimario_base.go(args)

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
