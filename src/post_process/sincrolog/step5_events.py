from collections import namedtuple, defaultdict
import json
import pandas as pd
import yaml

import src.utils as utils
from src.ml import load_llm, load_tts_maybe
from src.post_process.sincrolog.mario_compat import merged_to_mario
from scripts.events_posthoc import process_events_posthoc

def process_section(config, mario_section_name, gc_section_names, models_for_commentary=None):
    _merged_dfs = []
    for gc_sec_name in gc_section_names:
        paths = config.get_paths(gc_section_name=gc_sec_name)
        _merged_dfs.append(pd.read_csv(paths.merged_csv))
    print(len(_merged_dfs))
            
    paths = config.get_paths(mario_section_name=mario_section_name)
    with open(paths.gameinfo, 'r') as f:
        gameinfo = yaml.safe_load(f)

    # this is meant to work mainly on the merged data resulting from sincrolog,
    # but sincrolog only merges the robots (at the moment),
    # so we need the mario df for the ball still.
    merged_df_all_sections = merged_to_mario(pd.concat(_merged_dfs), gameinfo)
    mario_df = pd.read_csv(paths.mario_csv)
    final_df = pd.concat([
        mario_df[mario_df.type == "ball"],
        merged_df_all_sections[merged_df_all_sections.type == "robot"],  # type here is redundant but symmetry
    ])

    process_events_posthoc(config, mario_section_name, models_for_commentary, custom_df=final_df)

def main(mario_section_names, config, use_tts=False):
    # associate each mario section to all gc sections that insist on it
    section_association = defaultdict(list)
    for gc_section_name in sorted(utils.get_section_names_to_do(config.get_paths().game_dir, config.dir_names.gc_section_prefix, None)):
        with open(config.get_paths(gc_section_name=gc_section_name).section_params) as f:
            section_params = json.load(f)
        mario_section_name = section_params["manual"]["mario_half_name"]
        section_association[mario_section_name].append(gc_section_name)

    # load models maybe
    if use_tts:
        Models = namedtuple("Models", ["llm", "match_tts"])
        models = Models(
            llm=load_llm(config),
            match_tts=load_tts_maybe(config),
        )
    else:
        models=None

    for mario_section_name in mario_section_names:
        gc_section_names = section_association[mario_section_name]
        process_section(config, mario_section_name, gc_section_names, models)
