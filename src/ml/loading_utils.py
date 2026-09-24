from src.ml.models.ollama_wrapper import OllamaWrapper
from src.ml.models.openai_chat_wrapper import commentary_llm_from_config
from src.ml.models.match_commentary_tts import MatchCommentaryTTS

def load_llm(config, logger=None):
    _llm_backend = config.models.commentary_llm_backend.lower().strip()
    if _llm_backend == "openai":
        llm = commentary_llm_from_config(config.models)
        if logger is not None:
            logger.info(f"Commentary LLM: OpenAI (model={config.models.openai_model})")
    else:
        llm = OllamaWrapper(config.models.llm_model)
        if logger is not None:
            logger.info(f"Commentary LLM: Ollama (model={config.models.llm_model})")
    return llm

def load_tts_maybe(config, logger=None):
    match_tts = None
    tts_on = config.tts.enabled
    geom_llm = config.commentating.geometry_llm_commentary_enabled
    periodic_llm = config.commentating.periodic_llm_commentary_enabled
    commentate = config.features.commentate
    if tts_on or (commentate and (geom_llm or periodic_llm)):
        match_tts = MatchCommentaryTTS(config)
        if tts_on:
            if logger is not None:
                logger.info("TTS commentary queue enabled (install: pip install edge-tts emoji; ffplay for playback).")
        else:
            if logger is not None:
                logger.info("MatchCommentaryTTS: text/LLM only (tts.enabled=false); no speech synthesis.")
    return match_tts
