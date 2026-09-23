"""Main tracking and processing module for MARIO."""

import numpy as np
import pandas as pd
import tqdm
import yaml

from ..utils.config.config_loader import MarioConfig
from ..utils.logger import setup_logger
from ..commentator import Commentator
from .realtime_display import RealtimeDisplayState
from .event_processor import EventProcessor

import src.core.files_handling as files_module

from src.utils.team_info import TeamMappingLR


class Funyarinpa:
    """
    Main tracking class for processing video frames with detection and tracking.
    """

    def __init__(self, config: MarioConfig, section_name: str, models: any, log_level="INFO"):
        """
        Initialize tracking with configuration.

        Args:
            config: MARIO configuration object.
        """
        self.logger = setup_logger(__name__, level=log_level)
        self.config = config
        self.paths = config.get_paths(mario_section_name=section_name, gc_section_name="gc_section_0")  # TODO infilare GC section(s)

        with open(self.paths.gameinfo, 'r') as f:
            self.gameinfo = yaml.safe_load(f)

        self.working_files_tracker = files_module.TempWorkingFilesManager(self.paths.mario_section_dir)

        # event processing
        self.team_map = TeamMappingLR.load(self.paths.team_mapping_lr)
        self.eventproc = EventProcessor(
            config,
            self.working_files_tracker.register_and_get_twf(
                self.paths.symbolic_events_jsonl
            ),
            self.team_map
        )


        # # these are only useful for the commentary
        # self.models = models
        # if self.models.match_tts is not None:
        #     self.logger.info("Match commentary TTS enabled (LLM lines → speech).")
        # self.commentator = Commentator(config, self.team_map, self.eventproc)
        # self._display = RealtimeDisplayState(config)

        self.logger.info("init ok")

    def merged_to_mario(self, df):
        df = df.rename(columns={"frame_n": "frame", "robot_x": "field_x", "robot_y": "field_y"})
        # filter out penalized robots, they're not on the field
        df = df[df.penalized == False]
        df["id"] = df.team * 1000 + df.player
        df["type"] = "robot"
        df["color"] = np.where(df.team == self.gameinfo["teams"]["home"]["number"], self.gameinfo["teams"]["home"]["fieldPlayerColor"], self.gameinfo["teams"]["away"]["fieldPlayerColor"])
        df["score_left"] = -1  # TEMP?
        df["score_right"] = -1  # TEMP?
        return df[["frame", "id", "type", "color", "bounding_box_in_image_space", "field_x", "field_y", "score_left", "score_right"]]

    def track_image(self):
        """
        Main tracking loop - process video and save results.

        Args:
            out_data: Output CSV file path (unused when ``write_csv`` is false).
        """
        self.logger.info("Pippoooooooooooooooooooooooooooooo")
        self.eventproc.begin_section()

        # this is meant to work mainly on the merged data resulting from sincrolog,
        # but we need the mario df for the ball still.
        merged_df = self.merged_to_mario(pd.read_csv(self.paths.merged_csv))
        mario_df = pd.read_csv(self.paths.mario_csv)
        final_df = pd.concat([
            mario_df[mario_df.type == "ball"],
            merged_df[merged_df.type == "robot"],  # type here is redundant but symmetry
        ])
        frame_ids = pd.unique(final_df.frame)
        print(frame_ids)
        for frame in tqdm.tqdm(frame_ids):
            frame_df = final_df[final_df.frame == frame]
            events = self.eventproc.process_frame(frame_df)  # does all as internal side-effect, including writing the file

            # commentary_event_label = None
            # commentary_text = None
            # ball_motion_debug = None
            # t_comm = time.perf_counter()
            # commentary_event_label, commentary_text, ball_motion_debug = self._display.commentary_labels(
            #     events=events,
            #     commentate=bool(self.config.features.commentate),
            #     eventproc=self.eventproc,
            #     commentator=self.commentator,
            #     match_tts=self.models.match_tts,
            #     llm=self.models.llm,
            #     show_window=self.config.features.show_commentary_window,
            # )
            # commentate_ms = (time.perf_counter() - t_comm) * 1000.0

        # self.working_files_tracker.finalize_and_cleanup()
        self.logger.info("kay")
