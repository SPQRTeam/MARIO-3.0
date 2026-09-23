import yaml
import json
from dataclasses import dataclass, asdict
from .side_hint import SideHint


class NoTagLoader(yaml.SafeLoader):
    pass

def ignore_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    else:
        return loader.construct_scalar(node)

NoTagLoader.add_multi_constructor('!', ignore_unknown)

def load_team_config(team_config_path):
    with open(team_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class TeamMapping:
    home: int
    away: int
    home_color: str
    away_color: str
    home_goalkeeper_color: str
    away_goalkeeper_color: str

def extract_team_mapping_from_yaml(yaml_path):
    if yaml_path is None:
        raise FileNotFoundError("yaml_path non fornito")
    
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=NoTagLoader)
    teams = data["teams"]

    home_num = teams["home"]["number"]
    away_num = teams["away"]["number"]
    home_color = teams["home"].get("fieldPlayerColor")
    away_color = teams["away"].get("fieldPlayerColor")
    home_goalkeeper_color = teams["home"].get("goalkeeperColor")
    away_goalkeeper_color = teams["away"].get("goalkeeperColor")  
                   

    return TeamMapping(
        int(home_num),
        int(away_num),
        home_color,
        away_color,
        home_goalkeeper_color,
        away_goalkeeper_color,
    )


@dataclass
class TeamMappingLR:
    left_number: int
    right_number: int
    left_color_name: str
    right_color_name: str
    left_goalkeeper_color_name: str
    right_goalkeeper_color_name: str
    left_short_name: str
    right_short_name: str
    left_full_name: str
    right_full_name: str

    def dump(self, path, **kwargs):
        with open(path, "w") as f:
            json.dump(asdict(self), f, **kwargs)

    @classmethod
    def load(cls, path, **kwargs):
        with open(path, "r") as f:
            data = json.load(f, **kwargs)
        # Unpack the dictionary directly into the dataclass constructor
        return cls(**data)

def make_team_mapping_lr(gameinfo, side_hint, section_name, team_names):
    if side_hint:
        key_to_read = side_hint.hint_type

        if side_hint.hint == gameinfo["teams"]["home"][key_to_read]:
            hinted_homeness = "home"
            other_homeness = "away"
        elif side_hint.hint == gameinfo["teams"]["away"][key_to_read]:
            hinted_homeness = "away"
            other_homeness = "home"
        else:
            raise KeyError(f"Hint {side_hint.hint} ({side_hint.hint_type}) not found in this game. (Available: {gameinfo['teams']['home'][key_to_read]}, {gameinfo['teams']['away'][key_to_read]})\nHave you tried extracting the gameinfo from the GC log?")

        if side_hint.side == SideHint.LEFT:
            left_team_homeness = hinted_homeness
            right_team_homeness = other_homeness
        else:
            right_team_homeness = hinted_homeness
            left_team_homeness = other_homeness

    else:
        # This is a hack, it's usually good but breaks the "S section" from streaming mode
        # A single OCR call on the relevant spot of the overlay would be another possible idea
        # TODO with the commentator and sumgen, there's now an expectation that the gc extraction
        #      is performed before mario now, so maybe we could request that mario-gc section linking
        #      is also done beforehand and refer to the gc log of the section, which does say 1st-2nd half
        is_second_half = "_A" not in section_name.name
        # This is another hack. Back in the day Charlotte and I noted the sidemapping field isn't always reliable.
        # It is for the more recent games we're using right now though.
        # Maybe more OCR for the future?
        left_team_homeness, right_team_homeness = "home", "away"
        if gameinfo["sideMapping"] == "homeDefendsRightGoal":
            left_team_homeness, right_team_homeness = right_team_homeness, left_team_homeness
        if is_second_half:
            left_team_homeness, right_team_homeness = right_team_homeness, left_team_homeness
    _left_no = int(gameinfo["teams"][left_team_homeness]["number"])
    _right_no = int(gameinfo["teams"][right_team_homeness]["number"])
    _left_col = gameinfo["teams"][left_team_homeness]["fieldPlayerColor"]
    _right_col = gameinfo["teams"][right_team_homeness]["fieldPlayerColor"]
    return TeamMappingLR(
        left_number=_left_no,
        right_number=_right_no,
        left_color_name=_left_col,
        right_color_name=_right_col,
        left_goalkeeper_color_name=gameinfo["teams"][left_team_homeness]["goalkeeperColor"],
        right_goalkeeper_color_name=gameinfo["teams"][right_team_homeness]["goalkeeperColor"],
        left_full_name=team_names.get(_left_no, None),
        right_full_name=team_names.get(_right_no, None),
        left_short_name=team_names.get(_left_no, _left_col),
        right_short_name=team_names.get(_right_no, _right_col),
    )
