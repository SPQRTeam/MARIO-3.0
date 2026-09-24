from collections import namedtuple, defaultdict
import json

import src.utils as utils
from src.post_process.sincrolog.s5_src.events_post import SincrologPosthocEventer
from src.ml.models.ollama_wrapper import OllamaWrapper
from src.ml.models.openai_chat_wrapper import commentary_llm_from_config

def main(mario_section_names, config, use_tts=False):
    # associate each mario section to all gc sections that insist on it
    section_association = defaultdict(list)
    for gc_section_name in sorted(utils.get_section_names_to_do(config.get_paths().game_dir, config.dir_names.gc_section_prefix, None)):
        with open(config.get_paths(gc_section_name=gc_section_name).section_params) as f:
            section_params = json.load(f)
        mario_section_name = section_params["manual"]["mario_half_name"]
        section_association[mario_section_name].append(gc_section_name)

    if use_tts:
        # load models — commentary text LLM: OpenAI API or local Ollama (see models.commentary_llm_backend)
        _llm_backend = config.models.commentary_llm_backend.lower().strip()
        if _llm_backend == "openai":
            llm = commentary_llm_from_config(config.models)
            print(f"Commentary LLM: OpenAI (model={config.models.openai_model})")
        else:
            llm = OllamaWrapper(config.models.llm_model)
            print(f"Commentary LLM: Ollama (model={config.models.llm_model})")

        match_tts = None
        tts_on = config.tts.enabled
        geom_llm = config.commentating.geometry_llm_commentary_enabled
        periodic_llm = config.commentating.periodic_llm_commentary_enabled
        commentate = config.features.commentate
        # Commentary LLM runs inside MatchCommentaryTTS.maybe_llm_commentary, which needs match_tts set.
        # Previously match_tts existed only with tts.enabled, so geometry LLM never ran with audio off.
        if tts_on or (commentate and (geom_llm or periodic_llm)):
            from src.ml.models.match_commentary_tts import MatchCommentaryTTS

            match_tts = MatchCommentaryTTS(config)
            if tts_on:
                print("TTS commentary queue enabled (install: pip install edge-tts emoji; ffplay for playback).")

        Models = namedtuple("Models", ["llm", "match_tts"])
        models = Models(
            llm = llm,
            match_tts=match_tts,
        )
    else:
        models=None

    for mario_section_name in mario_section_names:
        gc_section_names = section_association[mario_section_name]
        SincrologPosthocEventer(config, mario_section_name, gc_section_names, models).go()
