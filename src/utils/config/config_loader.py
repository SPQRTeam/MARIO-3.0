"""Configuration loader for MARIO project."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from munch import Munch
from src.utils.transforms import TransformsCalculator
import numpy as np
from datetime import datetime

from ..coherence import check_coherence_with_saved_or_update
from src.vision.color_cnn import ColorCNNBooster, ColorCNNNao

from .paths import PathsConfig, ROOT_DIR

FIELD_TYPE_TO_FILE = {
    "SPL": "fieldSPL.yaml",
    "HSL-m-go26": "fieldHSL_M_2026.yaml",
    "HSL-s-go26": "fieldHSL_S_go2026.yaml",
    "RCAP": "fieldRCAP.yaml",
}

VISION_TYPE_TO_FILE = {
    "nao": "nao.yaml",
    "booster": "booster.yaml",
    "k1": "booster.yaml",
    "t1": "booster.yaml",
}

VISION_TYPE_TO_CNN = {
    "nao": ColorCNNNao,
    "booster": ColorCNNBooster,
    "k1": ColorCNNBooster,
    "t1": ColorCNNBooster,
}

def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (mutates ``base``)."""
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base

class MarioConfig(Munch):
    """Main MARIO configuration."""

    def __init__(self, data, args, update_game_history=True):
        super().__init__(data)

        game_history_path = ROOT_DIR / self.dir_names.game_history_fname
        if game_history_path.exists():
            with open(game_history_path, 'r') as f:
                game_history = yaml.safe_load(f)
        else:
            game_history = []

        record = None
        if args.streaming:
            # this is STREAMING MODE, will create a new game_name if one was not provided
            assert len(args.streaming) == 2, "Malformed argument --streaming"
            record = {
                "streaming_url": args.streaming[0],
                "time_range": args.streaming[1],
                "game_name": args.game_name or f"streaming_{datetime.now().isoformat(sep='_')}",
            }
        elif args.game_name:
            # this is a local file, no streaming
            record = {"game_name": args.game_name}
        # else: record is still None, see next

        if record is None:
            # nothing was provided, fetch the last element of game_history
            assert game_history, "You must provide a game name on your first run!"
            record = game_history[-1]
        else:
            # if something was provided, add it to the history
            try:
                game_history.remove(record)
            except ValueError:  # record not in game_history
                pass
            game_history.append(record)

        self.game_name = record["game_name"]
        paths_path = ROOT_DIR / "config" / "paths" / "paths.yaml"
        with open(paths_path, 'r') as f:
            self.paths_data = yaml.safe_load(f)
        # paths don't get a field in self because they're heavily dependent on the section
        paths = self.get_paths()  # that said, we do need some section-agnostic paths right now

        self.is_streaming = "streaming_url" in record
        if self.is_streaming:
            paths.game_dir.mkdir(exist_ok=True)
            self.streaming_url = record["streaming_url"]
            self.time_range = record["time_range"]

        self.field_type = check_coherence_with_saved_or_update(paths.field_type_txt, args.field_type, "field type", "-f", required=True)
        # load field dimensions
        field_path = ROOT_DIR / "config" / "fields" / FIELD_TYPE_TO_FILE[self.field_type]
        with open(field_path, 'r') as f:
            data = yaml.safe_load(f)
        # circus yaml is in metres, convert to mm
        self.field_config = Munch({k: v * 1000 for k, v in data.items()})
        # precalculate and cache a bunch of derived measures, too
        self.field_config.xlimit = self.field_config.width / 2
        self.field_config.ylimit = self.field_config.height / 2
        self.field_config.goal_area_ylimit = self.field_config.goal_area_height / 2
        self.field_config.penalty_area_ylimit = self.field_config.penalty_area_height / 2
        self.field_config.penalty_mark_x = self.field_config.xlimit - self.field_config.penalty_mark_distance
        # note: goal_width is correct below, goal measures have a different nomenclature than field area measures
        self.field_config.goalpost_y = self.field_config.goal_width / 2

        # set up lazy robot vision config (it only matters for the tracker, so i don't want stuff like GC extraction to care)
        self.vision_type_from_cli = args.vision_type
        self._vision_type = None
        self._vision_config = None
        # be eager if the vision type was given explicitly in the CLI so any script can update the saved one
        if self.vision_type_from_cli is not None:
            self._load_vision_config()

        if update_game_history:
            with open(game_history_path, 'w') as f:
                yaml.dump(game_history, f)

    def get_paths(self, *, mario_section_name=None, gc_section_name=None):
        return PathsConfig(self.paths_data, self.game_name, mario_section_name=mario_section_name, gc_section_name=gc_section_name)

    # lazy vision config
    def _load_vision_config(self):
        self._vision_type = check_coherence_with_saved_or_update(self.get_paths().vision_type_txt, self.vision_type_from_cli, "vision_type", "-v", required=True)
        vision_path = ROOT_DIR / "config" / "vision_types" / VISION_TYPE_TO_FILE[self._vision_type]
        with open(vision_path, 'r') as f:
            data = yaml.safe_load(f)
        self._vision_config = Munch(data)
    @property
    def vision_type(self):
        if self._vision_type is None:
            self._load_vision_config()
        return self._vision_type
    @property
    def vision_config(self):
        if self._vision_config is None:
            self._load_vision_config()
        return self._vision_config


    @property
    def min_robot_persistence_frames(self):
        return self.processing.fps * self.mario_postprocessing.min_robot_persistence_secs
    
    @property
    def max_frames_for_moving_ball_interp(self):
        return self.processing.fps * self.mario_postprocessing.max_secs_for_moving_ball_interp
    
    @property
    def max_frames_for_still_ball_interp(self):
        return self.processing.fps * self.mario_postprocessing.max_secs_for_still_ball_interp
    
    @property
    def trackcolor_window_length(self):
        return self.processing.fps * self.mario_postprocessing.trackcolor_window_length_secs
    
    @property
    def neotrack_unambiguous_length(self):
        return self.processing.fps * self.mario_postprocessing.neotrack_unambiguous_length_secs


    def make_transforms(self):
        from ..drawing import render_field_image, MARGIN_PX
        img, _ = render_field_image(self.field_config)
        img_h, img_w = img.shape[:2]
        return TransformsCalculator(
            MARIO_POS_MIN = np.array((MARGIN_PX, MARGIN_PX)),
            MARIO_POS_MAX = np.array((img_w - MARGIN_PX, img_h - MARGIN_PX)),
            FIELD_POS_MIN = np.array((-self.field_config.xlimit, -self.field_config.ylimit)),
            FIELD_POS_MAX = np.array((self.field_config.xlimit, self.field_config.ylimit)),
        )

    @classmethod
    def from_yaml(cls, args, update_game_history=True):
        config_path = args.config
        if config_path is None:
            config_path = ROOT_DIR / "config.yaml"
        else:
            config_path = Path(config_path)

        config_path = config_path.resolve()
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        merge_list = data.pop("_merge_files", None)
        if merge_list:
            base_dir = config_path.parent
            for rel in merge_list:
                frag_path = Path(rel)
                if not frag_path.is_absolute():
                    frag_path = (base_dir / frag_path).resolve()
                with open(frag_path, 'r') as f:
                    frag = yaml.safe_load(f) or {}
                _deep_merge(data, frag)

        data = munchify_all(data)
        return cls(data, args, update_game_history)


def munchify_all(diz):
    for key in diz:
        if isinstance(diz[key], dict):
            diz[key] = munchify_all(diz[key])
    return Munch(diz)
