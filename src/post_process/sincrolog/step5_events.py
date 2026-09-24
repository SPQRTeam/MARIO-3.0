from collections import namedtuple, defaultdict
import json

import src.utils as utils
from src.post_process.sincrolog.s5_src.events_post import SincrologPosthocEventer
from src.ml import load_llm, load_tts_maybe

def main(mario_section_names, config, use_tts=False):
    # associate each mario section to all gc sections that insist on it
    section_association = defaultdict(list)
    for gc_section_name in sorted(utils.get_section_names_to_do(config.get_paths().game_dir, config.dir_names.gc_section_prefix, None)):
        with open(config.get_paths(gc_section_name=gc_section_name).section_params) as f:
            section_params = json.load(f)
        mario_section_name = section_params["manual"]["mario_half_name"]
        section_association[mario_section_name].append(gc_section_name)

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
        SincrologPosthocEventer(config, mario_section_name, gc_section_names, models).go()
