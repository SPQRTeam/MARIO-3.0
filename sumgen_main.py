import argparse
from pathlib import Path
import src.utils as utils

STEP_AUTOPARAMS = "ap"
STEP_GCEXTR = "gc"
STEP_STATS = "st"
STEP_LLMSUMMARY = "ls"

all_steps = [STEP_AUTOPARAMS, STEP_GCEXTR, STEP_STATS, STEP_LLMSUMMARY]

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=Path, default=None, help="Path to config.yaml (default: ./config.yaml)")
parser.add_argument('--game-name', '--game', '-g', type=str, default=None, help="Name of the game to work with (a directory in data/). If unspecified, will use the last one used in this project.")
parser.add_argument('--sections', '-s', nargs='+', help="Sections to work with. ONLY THE LETTERS. Defaults to all.")
parser.add_argument('--steps', '-t', type=str, nargs="+", choices=all_steps, help="Steps to perform, in order. May default to all in the future, if I decide it's a good idea.")
parser.add_argument("--field-type", "--field", "-f", type=str, default=None, choices=utils.FIELD_TYPE_TO_FILE.keys(), help="The field this game was played on. Only specify the first time you work with this game.")
parser.add_argument("--vision-type", "--vision", "-v", type=str, default=None, choices=utils.VISION_TYPE_TO_FILE.keys(), help="The set of vision models for this game. This stage doesn't care, so specify only if you want to update the saved one.")
parser.add_argument("--halp", action="store_true", help='''Display a "panic help" message useful if you've been away for too long, then exit.''')
args = parser.parse_args()
args.streaming = False  # for MarioConfig compatibility

if args.halp:
    print("So, this is what you really gotta do:")
    print("")
    print("python3 sincrolog_main.py -g <GAME_NAME> -t <STEPS>")
    print("")
    print("See the first lines in this script to know which steps are available.")
    print("Typically do gc before mario, all the others after mario in order.")
    print("")
    print("Extra stuff:")
    print("- Omit -g to use the last game used in this project (including MARIO!)")
    print("- Use -s to only do specific sections")
    exit(0)

config = utils.MarioConfig.from_yaml(args)
paths = config.get_paths()
section_names = utils.get_section_names_to_do(paths.game_dir, config.dir_names.mario_section_prefix, args.sections)
section_names.sort()

for sn in section_names:
    config.get_paths(mario_section_name=sn).sumgen_output_dir.mkdir(exist_ok=True)

if STEP_GCEXTR in args.steps:
    import src.post_process.summary_generation.gc_extraction as step_gc
    step_gc.main(config)

if STEP_AUTOPARAMS in args.steps:
    import src.post_process.summary_generation.autocalc_section_params_berlin as step_ap
    step_ap.main(section_names, config)

if STEP_STATS in args.steps:
    import src.post_process.summary_generation.numeric_stats as step_st
    for sn in section_names:
        step_st.main(sn, config)

if STEP_LLMSUMMARY in args.steps:
    import src.post_process.summary_generation.llm_summary as step_ls
    step_ls.main(section_names, config)
