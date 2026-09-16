import ast
import tqdm
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import scipy.stats as stats
import json
import yaml
import src.utils as utils

TEAMS_WITH_LOCALIZATION = {
    5,  # B-Human
    8,  # Whirlwind
}

# due to the current limitations of our color CNN and goalkeepers wearing a different color jersey,
# some teams have their goalie consistently not recognized as part of the team,
# so it's unfair to count the goalie for the error
TEAMS_WITHOUT_GOALIE = {
    5,  # B-Human
}

MARGIN = 1000

"""
Questo si mette al posto di step4
"""

def calculate_errors(gc_fixed_penalties, mario_df, time_limit, team_map, config):
    robot_error_by_frame = []

    colors_with_localization = set()
    if team_map.home in TEAMS_WITH_LOCALIZATION:
        colors_with_localization.add(team_map.home_color)
    if team_map.away in TEAMS_WITH_LOCALIZATION:
        colors_with_localization.add(team_map.away_color)

    frame_indices = sorted(mario_df['frame'].unique())
    frame_indices_tqdm = tqdm.tqdm(enumerate(frame_indices), total=len(frame_indices))
    for i, frame in frame_indices_tqdm:
        mario_row = mario_df[mario_df['frame'] == frame].iloc[0]
        timestamp = mario_row['gametime']
        frame_indices_tqdm.set_postfix_str(f"gametime: {timestamp/1000:.2f}s")
        if time_limit > 0 and timestamp > time_limit*1000:
           break
        if timestamp < 0 or timestamp > gc_fixed_penalties.gametime.max():
            continue


        gc_state = utils.get_gc_state_at_time(gc_fixed_penalties, timestamp)
        gc_robots = gc_state[(gc_state["team"].isin(TEAMS_WITH_LOCALIZATION)) & (~gc_state["penalized"])]
        goalie_condition = (gc_robots["team"].isin(TEAMS_WITHOUT_GOALIE)) & (gc_robots["player"] == 1)
        gc_robots = gc_robots[~goalie_condition]
        gc_robot_coords = gc_robots[["x", "y"]].to_numpy()

        mario_state = utils.get_mario_state_at_time(mario_df, timestamp)
        mario_robots_mid = mario_state[mario_state.type == "robot"]
        color_filter = mario_robots_mid["color"].apply(lambda x: not x.isdisjoint(colors_with_localization))
        position_filter = (abs(mario_robots_mid["field_x"]) < config.field_config.xlimit + MARGIN) & (abs(mario_robots_mid["field_y"]) < config.field_config.ylimit + MARGIN)
        mario_robots = mario_robots_mid[color_filter & position_filter]
        mario_robot_coords = mario_robots[["field_x", "field_y"]].to_numpy()

        # if this is satisfied for a number of frames, it's okay, doesn't have to be 100%
        assert len(gc_robots) < len(gc_state)
        assert len(mario_robots) < len(mario_robots_mid)

        if len(mario_robot_coords) >= len(gc_robot_coords):
            larger_coords = mario_robot_coords
            smaller_coords = gc_robot_coords
        else:
            larger_coords = gc_robot_coords
            smaller_coords = mario_robot_coords

        # Query a cKDTree for the nearest neighbor of each point in the larger table
        # k=1 means we only want the single closest point.
        # The query returns an array of the minimum Euclidean distances and their indices.
        tree = cKDTree(smaller_coords)
        min_distances, closest_indices = tree.query(larger_coords, k=1)

        total_distance = min_distances.mean()

        robot_error_by_frame.append(total_distance)

    return np.array(robot_error_by_frame)



# it's a class just so I can get the __dict__ for saving
class GlobalStats:
    def __init__(self, errors):
        self.global_mean = errors.mean().item()
        self.global_std = errors.std(ddof=1).item()
        def _getci (conf):
            a,b = stats.t.interval(conf, df=len(errors)-1, loc=self.global_mean, scale=self.global_std / np.sqrt(len(errors)))
            return a.item(), b.item()
        self.global_confidence_interval_80 = _getci(0.80)
        self.global_confidence_interval_90 = _getci(0.90)
        self.global_confidence_interval_95 = _getci(0.95)
        self.global_confidence_interval_99 = _getci(0.99)
        self.max = errors.max().item()
        self.min = errors.min().item()
        self.quantiles = np.quantile(errors, [0.25, 0.50, 0.75]).tolist()




def main(section_name, time_limit, config):

    paths = config.get_paths(gc_section_name=section_name)

    paths.sincrolog_output_dir.mkdir(parents=True, exist_ok=True)

    with open(paths.section_params, 'r') as f:
        params = json.load(f)

    paths = config.get_paths(gc_section_name=section_name, mario_section_name=params["manual"]["mario_half_name"])

    team_map = utils.extract_team_mapping_from_yaml(paths.gameinfo)

    mario_df = pd.read_csv(
        paths.mario_post_step2_csv,
        converters={
            'bounding_box_in_image_space': ast.literal_eval,
            'color': ast.literal_eval,
        },
    )
    
    mario_df["videotime"] = (mario_df.frame / config.processing.fps) * 1000
    mario_df["gametime"] = mario_df.videotime - params["manual"]["mario_start_time"]

    gc_fixed_penalties = pd.read_csv(paths.gc_post_step3_csv)

    errors = calculate_errors(gc_fixed_penalties, mario_df, time_limit, team_map, config)

    error_dir = paths.mario_section_dir / "error"
    error_dir.mkdir(exist_ok=True)
    rebf_array = np.array(errors)
    np.savetxt(error_dir / f"by_frame_robotsonly_{section_name[-2:]}.txt", rebf_array)
    np.save(error_dir / f"by_frame_robotsonly_{section_name[-2:]}.npy", rebf_array)

    with open(error_dir / f"global_stats_robotsonly_{section_name[-2:]}.yaml", "w") as f:
        yaml.dump(vars(GlobalStats(errors)), f)

    print(errors)
