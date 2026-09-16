import argparse
from pathlib import Path
import src.utils as utils

all_steps = [1,2,3,4,9]

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=Path, default=None, help="Path to config.yaml (default: ./config.yaml)")
parser.add_argument('--game-name', '--game', '-g', type=str, default=None, help="Name of the game to work with (a directory in data/). If unspecified, will use the last one used in this project.")
parser.add_argument('--steps', '-t', type=int, nargs="+", choices=all_steps, default=all_steps, help="Steps to perform. Defaults to all.")
parser.add_argument('--sections', '-s', nargs='+', help="Sections to work with. ONLY THE NUMBERS OR LETTERS. Defaults to all.")
parser.add_argument("--field-type", "--field", "-f", type=str, default=None, choices=utils.FIELD_TYPE_TO_FILE.keys(), help="The field this game was played on. Only specify the first time you work with this game.")
parser.add_argument("--profile", action="store_true", help="Enable profiling")
parser.add_argument("--time-limit", "-l", type=int, default=-1, help="Seconds of video to process. Default is all video.")
parser.add_argument("--halp", action="store_true", help='''Display a "panic help" message useful if you've been away for too long, then exit.''')
args = parser.parse_args()
args.streaming = False  # for MarioConfig compatibility

if args.halp:
    print("So, this is what you really gotta do:")
    print("")
    print("python3 sincrolog_main.py -g <GAME_NAME>")
    print("")
    print("Extra stuff:")
    print("- Omit -g to use the last game used in this project (including MARIO!)")
    print("- Use -t to only do specific steps")
    print("- Use -s to only do specific sections (numbers for steps 1,3,4; letters for step 2)")
    exit(0)

config = utils.MarioConfig.from_yaml(args)

if 1 in args.steps:
    import src.post_process.sincrolog.step1_yaml2csv as step1
    step1.main(config)

if 2 in args.steps:
    import src.post_process.sincrolog.step2_filtermario as step2
    section_names = utils.get_section_names_to_do(config.game_dir, config.dir_names.mario_section_prefix, args.sections)
    for sn in section_names:
        step2.main(sn, config)

if 3 in args.steps:
    import src.post_process.sincrolog.step3_penalties as step3
    section_names = utils.get_section_names_to_do(config.game_dir, config.dir_names.gc_section_prefix, args.sections)
    for sn in section_names:
        step3.main(sn, config)

if 4 in args.steps:
    import src.post_process.sincrolog.step4_match as step4

    if args.profile:
        raise NotImplementedError("Config was reloaded, this doesn't work anymore as is")
        # if we need profiling again we'll have to update it to use the new config
        prof = cProfile.Profile()
        prof.run('main(mario_csv_name, config.MARIO_START_TIME, gc_penalties_csv_name)')
        prof.dump_stats('output.prof')

        stream = open('output.txt', 'w')
        stats = pstats.Stats('output.prof', stream=stream)
        stats.sort_stats('cumtime')
        stats.print_stats()
    else:
        section_names = utils.get_section_names_to_do(config.game_dir, config.dir_names.gc_section_prefix, args.sections)
        for sn in section_names:
            print(sn)
            step4.main(sn, args.time_limit, config)

if 9 in args.steps:
    import src.post_process.sincrolog.stepx_error as stepx
    section_names = utils.get_section_names_to_do(config.game_dir, config.dir_names.gc_section_prefix, args.sections)
    for sn in section_names:
        stepx.main(sn, args.time_limit, config)
