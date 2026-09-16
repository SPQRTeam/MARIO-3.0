"""
Only works with videos that are begin exactly when the GC clicked ready, such as Berlin United's.
"""

import json
import pandas as pd

from src.post_process.sincrolog.step3_penalties import find_gc_start_time
import src.utils as utils

def process_section(mario_section_name, config, predetermined_gc_section_name=None):
    if predetermined_gc_section_name is None:
        print(f"Auto-section-params for {config.game_name}/{mario_section_name}")
        _gc_section_num = input(f"Enter the corresponding GC section > ")
        gc_section_name = config.dir_names.gc_section_prefix + _gc_section_num
    else:
        print(f"Auto-assigned {mario_section_name} <-> {predetermined_gc_section_name}")
        gc_section_name = predetermined_gc_section_name
    gc_csv_path = config.game_dir / gc_section_name / "gc_collective.csv"
    gc_df = pd.read_csv(gc_csv_path)

    gc_start_time = float(find_gc_start_time(gc_df, 0))

    with open(config.team_mapping_lr_path(mario_section_name)) as f:
        team_mapping_lr = json.load(f)

    section_params = {
        "manual": {
            "mario_half_name": mario_section_name,
            "gc_validity_start_time": gc_start_time,
            "mario_start_time": gc_start_time,  # the video is assumed to start at the beginning of ready, and the gctime does too, so they're aligned.
            "gc_flip_team": team_mapping_lr["right_number"],  # the right team needs to have its poses flipped to go on the other side of the field
        },
        "calculated_gc_start_time": gc_start_time,
    }

    with open(config.section_params_path(gc_section_name), "w") as f:
        json.dump(section_params, f, indent=4)

    # print(f"Binding {mario_section_name} <-> {gc_section_name} done.")
    # print("Are there any more GC sections to bind to this MARIO section (b/c of timeouts or stuff)?")
    # print("If so, more code will have to be implemented.")

    # under our assumption, it shouldn't be possible to have more than one gc section for the same mario section or viceversa,
    # since they both start on ready
    with open(config.game_dir / mario_section_name / "gc_section_backlink.json", "w") as f:
        json.dump({"gc_section_name": gc_section_name}, f, indent=4)

def main(mario_section_names, config):
    assert (config.game_dir / "gc_section_0").exists(), "This requires gc_extraction first"

    # shortcut for the sake of automation: if there is an equal number of video and GC sections, assume they match sequentially and don't prompt the user.
    # a way for turning this assumption off is not implemented at the moment, this is left as an "exercise" if necessary.
    gc_section_names = utils.get_section_names_to_do(config.game_dir, config.dir_names.gc_section_prefix, None)
    print(f"{len(mario_section_names)} {len(gc_section_names)}")
    if len(gc_section_names) == len(mario_section_names):
        for (mario_sn, gc_sn) in zip(mario_section_names, gc_section_names):
            process_section(mario_sn, config, predetermined_gc_section_name=gc_sn)
    else:
        # plain :(
        for sn in mario_section_names:
            process_section(sn, config)
