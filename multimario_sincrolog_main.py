import argparse
import subprocess
import yaml

import scripts.multimario as multimario_base
from src.utils.config.paths import ROOT_DIR, PathsConfig
from src.utils.coherence import arg_saved_or_ask
from src.utils.sectioning import get_section_names_to_do
from src.utils.side_hint import SideHint
from src.utils.team_info import make_team_mapping_lr

# take the backlink as proof that a section has already been auto-param'd
# also memo, autoparams must be done for all sections at once or the check for automation kinda breaks
def autoparams_needed(game_name):
    paths = PathsConfig.from_yaml(game_name)
    for mario_section_name in get_section_names_to_do(paths.game_dir, "mario_", None):
        paths.MARIO_SECTION = mario_section_name
        if not paths.gcsec_backlink.exists():
            return True
    return False

def sections_to_sincrolog(game_name):
    PREFIX = "gc_section_"
    result = []
    paths = PathsConfig.from_yaml(game_name)
    for gc_section_name in get_section_names_to_do(paths.game_dir, PREFIX, None):
        paths.GC_SECTION = gc_section_name
        if not paths.merged_csv.exists():
            result.append(gc_section_name[len(PREFIX):])
    return result

def make_team_mappings(game_name):
    paths = PathsConfig.from_yaml(game_name)
    with open(paths.gameinfo, 'r') as f:
        gameinfo = yaml.safe_load(f)
    for mario_section_name in get_section_names_to_do(paths.game_dir, "mario_", None):
        paths.MARIO_SECTION = mario_section_name
        side_hint = SideHint.loads(paths.side_hint_txt.read_text())
        make_team_mapping_lr(gameinfo, side_hint, mario_section_name, dict()).dump(
            paths.team_mapping_lr,
            indent=4,
        )

def do_sincrolog(game_name, robot_type, gc_sections=None):
    subprocess.run([
        "python3", "sincrolog_main.py",
        "-g", game_name,
        "-r", robot_type,
        "-t", "2",
    ])
    cmd = [
        "python3", "sincrolog_main.py",
        "-g", game_name,
        "-r", robot_type,
        "-t", "3", "4",
    ]
    if gc_sections is not None:
        cmd += ["-s"] + gc_sections
    subprocess.run(cmd)



def main():
    parser = argparse.ArgumentParser()
    multimario_base.add_multimario_arguments(parser)
    parser.add_argument(
        "--resincrolog",
        action="store_true",
        help="Force repeat of sincrolog even if it had already been performed"
    )
    args = parser.parse_args()

    multimario_base.go_prelims(args)

    # we still need this to fill the section_params
    # do this early as it's still potential user interaction
    for game_name in args.game_names:
        if args.recalibrate or autoparams_needed(game_name):
            print(game_name)
            # sumgen needs this json to work. this also helps us check for any typos in the side hints.
            make_team_mappings(game_name)
            subprocess.run([
                "python3", "sumgen_main.py",
                "-g", game_name,
                "-t", "ap",
            ])

    succeeded, failed = multimario_base.go_mario(args)

    for game_name in succeeded:
        robot_type = arg_saved_or_ask(args.robot_type, ROOT_DIR / "data" / "games" / game_name / "robot_type.txt", f"{game_name} field type? ")
        sects = sections_to_sincrolog(game_name)
        if args.resincrolog:
            print(f"{game_name}: doing all sincrolog due to resincrolog request")
            do_sincrolog(game_name, robot_type)
        elif sects:
            print(f"{game_name}: sincrolog-ing these sections: {sects}")
            do_sincrolog(game_name, robot_type, sects)
        else:
            print(f"{game_name}: sincrolog already done")

    if failed:
        print("Some MARIOs failed:")
        for f in failed:
            print(f" - {f}")

main()
