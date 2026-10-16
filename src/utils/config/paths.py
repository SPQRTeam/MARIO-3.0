from munch import Munch
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
PATHS_CONFIG_PATH = ROOT_DIR / "config" / "paths" / "paths.yaml"  # by its very nature, this can't be in paths.yaml :)

class PathsConfig(Munch):
    def __init__(self, data, game_name, *, mario_section_name=None, gc_section_name=None):
        super().__init__(data)
        self.ROOT = ROOT_DIR
        self.GAME = game_name
        self.MARIO_SECTION = mario_section_name
        self.GC_SECTION = gc_section_name

    def __getattr__(self, k):
        raw = super().__getattr__(k)

        # special error message
        if "SECTION" in k and raw is None:
            raise ValueError(f"Attempted requesting a path specific to a section of type {k} from a PathsConfig that doesn't have it")

        final_path = Path()
        for part in Path(raw).parts:
            if part.startswith("$"):
                part = self.__getattr__(part[1:])
            final_path /= part
        return final_path

    # load as standalone, w/o all the overhead of MarioConfig
    @classmethod
    def from_yaml(cls, game_name, *, mario_section_name=None, gc_section_name=None):
        with open(PATHS_CONFIG_PATH) as f:
            return cls(yaml.safe_load(f), game_name, mario_section_name=mario_section_name, gc_section_name=gc_section_name)
