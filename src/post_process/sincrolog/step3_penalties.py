from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
import json

def find_gc_start_time(gc_df, gc_validity_start_time):
    # returns a column with the time in the csv for which it is greater than or equal to that of the initial whistle and the condition if
    # the robots are actually starting to play
    valid = (gc_df.gctime >= gc_validity_start_time) & (gc_df.playing)
    if not valid.any():
        return None
    return gc_df[valid].iloc[0].gctime

def crash_condition(gc_df, max_absence_time=20000):
    """
    If a robot is not detected for more than 20 seconds, consider it penalized from the very first moment it stops being detected.
    """
    corrected_gc_df = gc_df.copy()
    gc_df_filtered = gc_df[gc_df.gametime >= 0].copy()

    print(f"\n[CRASHED ROBOT] Robot absent for {max_absence_time/1000:.1f} seconds...")

    all_robots = gc_df_filtered[['team', 'player']].drop_duplicates()
    
    for _, robot_info in all_robots.iterrows():
        team, player = robot_info.team, robot_info.player
        robot_key = (team, player)
        
        # robot history
        robot_history = gc_df_filtered[
            (gc_df_filtered.team == team) & 
            (gc_df_filtered.player == player)
        ].sort_values('gametime').copy()
        
        if len(robot_history) < 2:
            continue
        
        # Find long absence time
        for i in range(1, len(robot_history)):
            prev_time = robot_history.iloc[i-1].gametime
            curr_time = robot_history.iloc[i].gametime
            absence_duration = curr_time - prev_time

            # If the absence is too long, consider the robot crashed from the previous moment
            if absence_duration > max_absence_time:
                crash_start_time = prev_time
                crash_end_time = curr_time

                print(f"Robot {robot_key}: CRASHED from {crash_start_time/1000:.1f}s to {crash_end_time/1000:.1f}s (absence of {absence_duration/1000:.1f}s)")

                # Mark all records from the moment of the crash until the return as penalized
                robot_mask = (corrected_gc_df.team == team) & (corrected_gc_df.player == player)
                crash_mask = robot_mask & (corrected_gc_df.gametime >= crash_start_time) & (corrected_gc_df.gametime < crash_end_time)
                
                records_penalized = crash_mask.sum()
                print(f"{records_penalized} records marked as penalized during the absence")
                
                corrected_gc_df.loc[crash_mask, 'penalized'] = True

        # Check if the robot disappears permanently at the end
        if len(robot_history) > 0:
            last_seen_time = robot_history.iloc[-1].gametime
            game_end_time = gc_df_filtered.gametime.max()
            final_absence = game_end_time - last_seen_time
            
            if final_absence > max_absence_time:
                print(f"Robot {robot_key}: CRASHED DEFINITIVELY from {last_seen_time/1000:.1f}s to the end")

                # Mark all records as penalized from the final crash moment onward.
                robot_mask = (corrected_gc_df.team == team) & (corrected_gc_df.player == player)
                crash_mask = robot_mask & (corrected_gc_df.gametime >= last_seen_time)
                
                records_penalized = crash_mask.sum()
                print(f" {records_penalized} records marked as penalized from the final crash")
                
                corrected_gc_df.loc[crash_mask, 'penalized'] = True
    
    return corrected_gc_df

def exclude_robots_from_gc(gc_df, exclusions):
    """
    Esclude robot specifici dal dataset GC in determinati intervalli di tempo.
    
    Args:
        gc_df: DataFrame con i dati GC
        exclusions: Lista di dict con formato:
                   [{'team_number': int, 'player_number': int, 'start_time': ms, 'end_time': ms, 'reason': str}]
    
    Returns:
        DataFrame: gc_df filtrato senza i robot esclusi negli intervalli specificati
    """
    if not exclusions:
        return gc_df
    
    gc_df = gc_df.copy()
    
    print(f"\n[ROBOT EXCLUSIONS] Applying {len(exclusions)} exclusions...")
    
    for exclusion in exclusions:
        team = exclusion['team_number']
        player = exclusion['player_number']
        start_time = exclusion['start_time']
        end_time = exclusion['end_time']
        reason = exclusion.get('reason', 'No reason provided')
        
        # Maschera per identificare le righe da escludere
        exclusion_mask = (
            (gc_df.team == team) & 
            (gc_df.player == player) & 
            (gc_df.gametime >= start_time) & 
            (gc_df.gametime <= end_time)
        )
        
        excluded_count = exclusion_mask.sum()
        print(f"  → Excluding robot (team={team}, player={player}) from {start_time/1000:.2f}s to {end_time/1000:.2f}s")
        print(f"    Reason: {reason}")
        print(f"    Records excluded: {excluded_count}")
        
        # Rimuovi le righe che corrispondono ai criteri di esclusione
        gc_df = gc_df[~exclusion_mask]
    
    return gc_df

def filter_gc_after_penalty(gc_df, min_movement=500):
    """
    Removes GC records of robots that exit penalty but have not yet relocalized.
    A robot is considered "non-relocalized" if its position does not change significantly
    compared to the last position during the penalty.
    
    Args:
        gc_df: gc DataFrame
        min_movement: Minimum distance in mm to consider a robot "relocalized"

    Returns:
        DataFrame: gc_df filtered without the records of non-relocalized robots post-penalty
    """
    gc_df = gc_df.copy()
    gc_df_filtered = gc_df[gc_df.gametime >= 0].copy()
    
    print(f"\n[NON-RELOCALIZED ROBOTS] Detecting robots not relocalized after penalty...")
    
    all_robots = gc_df_filtered[['team', 'player']].drop_duplicates()
    
    records_to_remove = []
    
    for _, robot_info in all_robots.iterrows():
        team, player = robot_info.team, robot_info.player
        robot_key = (team, player)
        
        robot_history = gc_df_filtered[
            (gc_df_filtered.team == team) & 
            (gc_df_filtered.player == player)
        ].sort_values('gametime').copy()
        
        if len(robot_history) < 2:
            continue
        
        penalty_position = None
        penalty_end_time = None
        
        for i in range(len(robot_history)):
            current_row = robot_history.iloc[i]
            is_penalized = current_row.penalized
            current_pos = np.array([current_row.x, current_row.y])
            current_time = current_row.gametime
            
            if is_penalized:
                # Save the last position during penalty
                penalty_position = current_pos
                penalty_end_time = None  # Reset
            elif penalty_position is not None and penalty_end_time is None:
                # The robot has just exited the penalty
                penalty_end_time = current_time
                print(f"Robot {robot_key}: exited penalty at {penalty_end_time/1000:.2f}s")
                print(f"Last penalty position: ({penalty_position[0]:.1f}, {penalty_position[1]:.1f})")

            # If the robot has exited the penalty but has not yet moved
            if penalty_end_time is not None and not is_penalized:
                movement = np.linalg.norm(current_pos - penalty_position)
                
                if movement < min_movement:
                    # The robot has not yet moved enough - remove this record
                    records_to_remove.append(current_row.name)
                else:
                    # The robot has moved enough - it is re-localized
                    print(f"Robot {robot_key} relocalized at {current_time/1000:.2f}s (moved {movement:.1f}mm)")
                    penalty_position = None
                    penalty_end_time = None
    
    if records_to_remove:
        print(f"Removing {len(records_to_remove)} non-relocalized records")
        gc_df = gc_df.drop(records_to_remove)
    else:
        print(f"No non-relocalized records found")
    
    return gc_df

def convert_row(row):
    if (
        not np.isnan(row.ballx) and not np.isnan(row.bally)
        and not np.isnan(row.x) and not np.isnan(row.y) and not np.isnan(row.theta)
        and ('ballage' not in row or row.ballage > 0)
    ):
        global_x, global_y = local_to_global(
            row.x, row.y, row.theta, row.ballx, row.bally
        )
        row.ballx = global_x
        row.bally = global_y
    return row

def local_to_global(robot_x, robot_y, robot_theta, local_x, local_y):
    
    global_x = robot_x + local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)
    global_y = robot_y + local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)
    return global_x, global_y

def main(section_name, config):

    gc_csv_path = config.gc_csv_raw_path(section_name)
    gc_df = pd.read_csv(gc_csv_path)

    section_params_path = config.section_params_path(section_name)

    if not section_params_path.exists():
        with open(section_params_path, 'w') as f:
            f.write("{\n\n}")
        print(f"{section_params_path.relative_to(config.game_dir)} does not exist! Created it for convenience, but do fill it!")
        return

    with open(section_params_path, 'r') as f:
        params = json.load(f)
    gc_flip_team = params["manual"]["gc_flip_team"]

    mario_csv_path = config.mario_post_step2_path(params["manual"]["mario_half_name"])
    mario_df = pd.read_csv(mario_csv_path)

    mario_df["videotime"] = (mario_df.frame / config.processing.fps) * 1000
    # gc_validity_start_time is read from the configuration json file
    gc_start_time = find_gc_start_time(gc_df, params["manual"]["gc_validity_start_time"])
    print(f"[DEBUG] gc_start_time: {gc_start_time}")

    if gc_start_time is None:
        final_corrected_gc_df = pd.DataFrame([])
    else:
        # save the new gc_start_time in a new field
        gc_df["gametime"] = gc_df.gctime - gc_start_time
        mario_df["gametime"] = mario_df.videotime - params["manual"]["mario_start_time"]
        print("[DEBUG] Primi valori GC:")
        print(gc_df[["gctime", "gametime", "playing"]].head(10))
        # COORDINATE TRANSFORMATION
        if gc_flip_team is not None:
            gc_df.loc[gc_df['team'] == gc_flip_team, ['x', 'y']] *= -1
        
        params['calculated_gc_start_time'] = float(gc_start_time)
        with open(section_params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

        # APPLY ROBOT EXCLUSIONS (BEFORE crash detection)
        robot_exclusions = params["manual"].get("robot_exclusions", [])
        if robot_exclusions:
            gc_df = exclude_robots_from_gc(gc_df, robot_exclusions)
            
        # CONSIDER ROBOT CRASHES
        final_corrected_gc_df = crash_condition(gc_df, max_absence_time=20000)
        final_corrected_gc_df = filter_gc_after_penalty(final_corrected_gc_df, min_movement=500)
        final_corrected_gc_df.apply(convert_row, axis = 1)
    gc_penalties_path = config.gc_csv_post_step3_path(section_name)
    print(f"[CONFIG] Saving output to: {gc_penalties_path}")
    final_corrected_gc_df.to_csv(gc_penalties_path, index=False)
