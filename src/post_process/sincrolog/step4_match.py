import tqdm
import ast
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment 
import src.post_process.sincrolog.s4_src.match_ball as match_ball
import src.post_process.sincrolog.s4_src.match_robot as match_robot
import json
import src.utils as utils

def calculate_assignments_at_start_of_play(gc_fixed_penalties, mario_df, time, config):

    # Usa il dataset corretto per le assegnazioni
    final_gc_state = utils.get_gc_state_at_time(gc_fixed_penalties, time)
    gc_on_field = final_gc_state[(final_gc_state.penalized == False) & (final_gc_state.x.notna())].copy()
    
    mario_frame = mario_df[(mario_df.gametime >= time - 100) & (mario_df.gametime <= time + 2000)]
    mario_robots = mario_frame[mario_frame.type == 'robot'].groupby('id').last().reset_index()
    
    assignments = {}
    gc_only_robots = set()
    
    if not gc_on_field.empty and not mario_robots.empty:
        # LOGICA HUNGARIAN
        gc_positions = gc_on_field[['x', 'y']].values
        mario_positions = mario_robots[['field_x', 'field_y']].values
        distance_matrix = cdist(gc_positions, mario_positions)
        
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        assigned_gc_indices = set()

        for r, c in zip(row_ind, col_ind):
            if distance_matrix[r, c] < config.sincrolog.distance_threshold_initial:
                gc_robot = gc_on_field.iloc[r]
                mario_robot = mario_robots.iloc[c]
                assignments[(int(gc_robot.team), int(gc_robot.player))] = int(mario_robot.id)
                assigned_gc_indices.add(r)
        
        unassigned_gc_indices = set(range(len(gc_on_field))) - assigned_gc_indices
        for r_idx in unassigned_gc_indices:
            gc_robot = gc_on_field.iloc[r_idx]
            gc_only_robots.add((int(gc_robot.team), int(gc_robot.player)))
    else:
        gc_only_robots = {(r.team, r.player) for _, r in gc_on_field.iterrows()}
    
    mario_balls = mario_frame[mario_frame.type == 'ball']
    if len(mario_balls) == 1:
        initial_ball = (mario_balls.iloc[0].field_x, mario_balls.iloc[0].field_y, mario_balls.iloc[0].gametime)
        ball_source = 'mario'
    elif len(mario_balls) > 1:

        initial_ball = (mario_balls.iloc[0].field_x, mario_balls.iloc[0].field_y, mario_balls.iloc[0].gametime)
        ball_source = 'mario_multi'
    else:
        # Se non c'è palla MARIO, usa la logica GC (
        gc_state = utils.get_gc_state_at_time(gc_fixed_penalties, time)
        ball_x, ball_y, ball_source = match_ball.choose_ball_position(mario_frame, gc_state, None, None, time, config.sincrolog.max_ball_age, config.sincrolog.max_ball_jump)
        initial_ball = (ball_x, ball_y, time)

    return {
        'assignments': assignments,
        'gc_only': gc_only_robots,
        'corrected_gc_df': gc_fixed_penalties,
        'initial_ball': initial_ball,
        'ball_source': ball_source
    }


def local_to_global(robot_x, robot_y, robot_theta, local_x, local_y):
    
    global_x = robot_x + local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
    global_y = robot_y + local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)
    return global_x, global_y

def merge_datasets_with_manual_reassignment(gc_fixed_penalties, mario_df, assignment_data, time_limit, team_map, config):
    """
    Merge GC and MARIO datasets using manual reassignment data.
    """
    # DEBUG
    debug_robots = {(13, 1), (13, 2), (13, 3), (13, 6)}
    results = []
    initial_assignments = assignment_data['assignments']
    last_associated_mario_pos = {}
    for robot_key, mario_id in initial_assignments.items():
        mario_row = mario_df[(mario_df.id == mario_id) & (mario_df.gametime <= 0)]
        if not mario_row.empty:
            last_associated_mario_pos[robot_key] = mario_row.iloc[-1][['field_x', 'field_y']].values
        else:
            last_associated_mario_pos[robot_key] = None
    initial_gc_only = assignment_data['gc_only']

    current_assignments = initial_assignments.copy()
    current_gc_only = initial_gc_only.copy()
    robot_loss_times = {}
    gc_flipped_state = {} 
    timestamps = sorted(mario_df['gametime'].unique())
    frame_indices = sorted(mario_df['frame'].unique())
    max_gc_time = gc_fixed_penalties.gametime.max()
    timestamps = [t for t in timestamps if 0 <= t <= max_gc_time]
    ball_x = None
    ball_y = None


    print(f"--- Inizio Merge Manuale ({len(timestamps)} frames) ---")
    association_frame_count = {}
    break_events = []
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
        mario_state = utils.get_mario_state_at_time(mario_df, timestamp)

        current_assignments, current_gc_only, robot_loss_times, last_associated_mario_pos, gc_flipped_state, break_events = match_robot.update_assignments_manually(
            gc_fixed_penalties, mario_df, current_assignments, current_gc_only,
            robot_loss_times, timestamp, last_associated_mario_pos, gc_state, mario_state,
            association_frame_count, gc_flipped_state, frame, break_events, team_map, config,
            ##### passo come argomento gc_state e mario_state per evitare di ricalcolarlo
            ##### idem per team_map
        )

        if gc_state.empty: 
            continue
        
        global_playing = gc_state.playing.any()
        
 
        mario_balls = mario_state[mario_state.type == 'ball']
        if not mario_balls.empty:
            ball_x, ball_y = mario_balls.iloc[0].field_x, mario_balls.iloc[0].field_y
        
        for _, gc_robot in gc_state.iterrows():
            
            robot_key = (int(gc_robot.team), int(gc_robot.player))
            final_x, final_y, position_source = gc_robot.x, gc_robot.y, 'gc'
            bounding_box_in_image_space = "()"
           
            mario_id = current_assignments.get(robot_key)
            if mario_id is not None:
                mario_robot = mario_state[mario_state.id == mario_id]
                if not mario_robot.empty:
                    final_x, final_y = mario_robot.iloc[0].field_x, mario_robot.iloc[0].field_y
                    bounding_box_in_image_space = mario_robot.iloc[0].bounding_box_in_image_space
                    position_source = 'mario_dynamic'
                else:
                    position_source = 'gc_mario_gap'

            flipped_flag = gc_flipped_state.get(robot_key, False)

            results.append({
                'timestamp_ms': timestamp, 'frame_n': frame, 'team': gc_robot.team, 'player': gc_robot.player, 'mario_id': mario_id,
                'robot_x': final_x, 'robot_y': final_y, 'robot_theta': gc_robot.theta,
                'ball_x': ball_x, 'ball_y': ball_y, 'ball_age': gc_robot.ballage,
                'robot_ball_x': gc_robot.ballx, 'robot_ball_y': gc_robot.bally,
                'fallen': gc_robot.fallen, 'penalized': gc_robot.penalized, 'playing': global_playing,
                'position_source': position_source,
                # Questo mi serve per la visualizzazione su video, se ti va puoi rendere l'inserimento condizionale con una flag
                'bounding_box_in_image_space': bounding_box_in_image_space,
                'flipped': flipped_flag
            })
        
    print("--- Merge Manuale Completato ---")
    return pd.DataFrame(results), pd.DataFrame(break_events)



def flip_gc_robot_origin(gc_df, robots_to_flip, intervals):
    """
    Flippa i robot specificati rispetto all'origine (0,0) negli intervalli di tempo dati.
    """
    print(f"[DEBUG] Flippando robot {robots_to_flip} negli intervalli {intervals}")
    
    for robot_key in robots_to_flip:
        team, player = robot_key
        robot_mask = (gc_df.team == team) & (gc_df.player == player)
        
        for start_time, end_time in intervals:
            time_mask = (gc_df.gametime >= start_time) & (gc_df.gametime <= end_time)
            combined_mask = robot_mask & time_mask
            
            if combined_mask.sum() > 0:
                print(f"  Flippando robot ({team}, {player}) da {start_time}ms a {end_time}ms ({combined_mask.sum()} frame)")
                
                # Flippa le coordinate rispetto all'origine
                gc_df.loc[combined_mask, 'x'] = -gc_df.loc[combined_mask, 'x']
                gc_df.loc[combined_mask, 'y'] = -gc_df.loc[combined_mask, 'y']
                
                # Flippa anche l'orientazione
                gc_df.loc[combined_mask, 'theta'] = (gc_df.loc[combined_mask, 'theta'] + np.pi) % (2 * np.pi)

    return gc_df

def main(section_name, time_limit, config):

    output_dir = config.sincrolog_output_path(section_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    section_params_path = config.section_params_path(section_name)
    with open(section_params_path, 'r') as f:
        params = json.load(f)

    team_map = utils.extract_team_mapping_from_yaml(config.gameinfo_path)

    mario_csv_path = config.mario_post_step2_path(params["manual"]["mario_half_name"])
    mario_df = pd.read_csv(
        mario_csv_path,
        converters={'color': ast.literal_eval},
    )
    
    mario_df["videotime"] = (mario_df.frame / config.processing.fps) * 1000
    mario_df["gametime"] = mario_df.videotime - params["manual"]["mario_start_time"]

    input_gc_path = config.gc_csv_post_step3_path(section_name)
    try:
        gc_fixed_penalties = pd.read_csv(input_gc_path)
    except pd.errors.EmptyDataError:
        print("This section has no GC data, there is nothing to merge")
        print("Maybe it was such a small section that it had no playing? Who knows.")
        print("Anyways, bye")
        merged_df = pd.DataFrame()
        output_file = config.merged_csv_path(section_name)
        merged_df.to_csv(output_file, index=False)
        return

    assignment_data = calculate_assignments_at_start_of_play(gc_fixed_penalties, mario_df, 0, config)
    print("\n[DEBUG] ASSEGNAZIONI INIZIALI:")
    for robot_key, mario_id in assignment_data['assignments'].items():
        print(f"  Robot {robot_key} → MARIO ID {mario_id}")
    print(f"GC only (senza assegnazione): {assignment_data['gc_only']}\n")
    # Costruisci il dataset della palla prima
    initial_ball = assignment_data['initial_ball']
    ball_source = assignment_data['ball_source']

    merged_df, break_df = merge_datasets_with_manual_reassignment(gc_fixed_penalties, mario_df, assignment_data, time_limit, team_map, config)
    # ball_df = match_ball.build_ball_dataset(mario_df, gc_fixed_penalties, initial_ball, ball_source, config)
    # ball_df.to_csv(output_dir / "ball_dataset.csv", index=False)

    output_file = config.merged_csv_path(section_name)
    print(f"[CONFIG] Salvando output in: {output_file}")
    merged_df.to_csv(output_file, index=False)

    if not break_df.empty:
        # Ordina per timestamp per cronologia
        break_df = break_df.sort_values('timestamp_ms').reset_index(drop=True)
        
        break_csv_path = config.breaks_csv_path(section_name)
        break_df.to_csv(break_csv_path, index=False)
