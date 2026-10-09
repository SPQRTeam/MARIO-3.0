import argparse
from collections import namedtuple
import pandas as pd
from pathlib import Path
import tqdm
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.commentator import Commentator
from src.core.realtime_display import RealtimeDisplayState
from src.core.event_processor import EventProcessor
from src.utils.team_info import TeamMappingLR
import src.utils as utils
from src.ml import load_llm, load_tts_maybe
import src.core.files_handling as files_module

"""
Run event detection code (and optionally commentary generation) stand-alone,
without having to run all the tracking as well.

Works both with normal mario-only output (execute this script directly)
and with the sincrolog merged output (execute the appropriate step of sincrolog_main).
"""
# custom_df is for usage with sincrolog. User is responsible for passing one for the correct game and section, as the other paths are still from config.
def process_events_posthoc(config, mario_section_name, models_for_commentary=None, *, custom_df=None):
    paths = config.get_paths(mario_section_name=mario_section_name)

    if custom_df is not None:  # sincrolog mode
        df = custom_df
        tqdm_desc = f"step5 for {mario_section_name}"
    else:  # mario-only mode
        df = pd.read_csv(paths.mario_csv)
        tqdm_desc = f"standalone events for {mario_section_name}"

    working_files_tracker = files_module.TempWorkingFilesManager(paths.mario_section_dir)

    # event processing
    team_map = TeamMappingLR.load(paths.team_mapping_lr)
    eventproc = EventProcessor(
        config,
        working_files_tracker.register_and_get_twf(
            paths.symbolic_events_jsonl
        ),
        team_map
    )

    if models_for_commentary is None:
        commentary_enabled = False
    else:
        commentary_enabled = True
        models = models_for_commentary
        commentator = Commentator(config, team_map, eventproc)
        _display = RealtimeDisplayState(config)

    eventproc.begin_section()

    frame_ids = pd.unique(df.frame)
    print(frame_ids)
    for frame in tqdm.tqdm(frame_ids, desc=tqdm_desc):
        frame_df = df[df.frame == frame]
        events = eventproc.process_frame(frame_df)  # does all as internal side-effect, including writing the file

        if commentary_enabled:
            _, _, _ = _display.commentary_labels(
                events=events,
                commentate=bool(config.features.commentate),
                eventproc=eventproc,
                commentator=commentator,
                match_tts=models.match_tts,
                llm=models.llm,
                show_window=config.features.show_commentary_window,
            )

    working_files_tracker.finalize_and_cleanup()

# standalone execution: detects events in the normal mario output
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument('--game-name', '--game', '-g', type=str, default=None, help="Name of the game to work with (a directory in data/). If unspecified, will use the last one used in this project.")
    parser.add_argument('--sections', '-s', nargs='+', help="Sections to work with. ONLY THE LETTERS. Defaults to all.")
    parser.add_argument('--commentary', action="store_true", help="Enable commentary.")
    args = parser.parse_args()
    args.streaming = False  # for MarioConfig compatibility
    args.field_type = None  # for MarioConfig compatibility

    config = utils.MarioConfig.from_yaml(args)
    paths = config.get_paths()

    if args.commentary:
        Models = namedtuple("Models", ["llm", "match_tts"])
        models = Models(
            llm=load_llm(config),
            match_tts=load_tts_maybe(config),
        )
    else:
        models = None

    section_names = utils.get_section_names_to_do(paths.game_dir, config.dir_names.mario_section_prefix, args.sections)
    for sn in section_names:
        process_events_posthoc(config, sn, models)
