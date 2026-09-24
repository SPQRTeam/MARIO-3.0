"""Main tracking and processing module for MARIO."""

import numpy as np
import pandas as pd
import tqdm
import yaml

from src.commentator import Commentator
from src.core.realtime_display import RealtimeDisplayState
from src.core.event_processor import EventProcessor
from src.utils.team_info import TeamMappingLR

import src.core.files_handling as files_module

class SincrologPosthocEventer:
    """
    Main tracking class for processing video frames with detection and tracking.
    """

    def __init__(self, config, mario_section_name, gc_section_names, models_for_commentary=None):
        """
        Initialize tracking with configuration.

        Args:
            config: MARIO configuration object.
        """
        self.config = config

        _merged_dfs = []
        for gc_sec_name in gc_section_names:
            paths = config.get_paths(gc_section_name=gc_sec_name)
            _merged_dfs.append(pd.read_csv(paths.merged_csv))
        print(len(_merged_dfs))
                
        paths = config.get_paths(mario_section_name=mario_section_name)
        with open(paths.gameinfo, 'r') as f:
            self.gameinfo = yaml.safe_load(f)

        # this is meant to work mainly on the merged data resulting from sincrolog,
        # but we need the mario df for the ball still.
        merged_df_all_sections = self.merged_to_mario(pd.concat(_merged_dfs))
        mario_df = pd.read_csv(paths.mario_csv)
        self.final_df = pd.concat([
            mario_df[mario_df.type == "ball"],
            merged_df_all_sections[merged_df_all_sections.type == "robot"],  # type here is redundant but symmetry
        ])

        self.working_files_tracker = files_module.TempWorkingFilesManager(paths.mario_section_dir)

        # event processing
        self.team_map = TeamMappingLR.load(paths.team_mapping_lr)
        self.eventproc = EventProcessor(
            config,
            self.working_files_tracker.register_and_get_twf(
                paths.symbolic_events_jsonl
            ),
            self.team_map
        )

        if models_for_commentary is None:
            self.commentary_enabled = False
        else:
            self.commentary_enabled = True
            self.models = models_for_commentary
            self.commentator = Commentator(config, self.team_map, self.eventproc)
            self._display = RealtimeDisplayState(config)

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

    def go(self):
        """
        Main tracking loop - process video and save results.

        Args:
            out_data: Output CSV file path (unused when ``write_csv`` is false).
        """
        self.eventproc.begin_section()

        frame_ids = pd.unique(self.final_df.frame)
        print(frame_ids)
        for frame in tqdm.tqdm(frame_ids):
            frame_df = self.final_df[self.final_df.frame == frame]
            events = self.eventproc.process_frame(frame_df)  # does all as internal side-effect, including writing the file

            if self.commentary_enabled:
                _, _, _ = self._display.commentary_labels(
                    events=events,
                    commentate=bool(self.config.features.commentate),
                    eventproc=self.eventproc,
                    commentator=self.commentator,
                    match_tts=self.models.match_tts,
                    llm=self.models.llm,
                    show_window=self.config.features.show_commentary_window,
                )

        self.working_files_tracker.finalize_and_cleanup()
