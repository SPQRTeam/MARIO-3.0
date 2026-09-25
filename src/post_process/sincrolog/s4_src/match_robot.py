from scipy.optimize import linear_sum_assignment 
import numpy as np
from collections import namedtuple

Candidate = namedtuple('Candidate', ['cost', 'robot_key', 'mario_id', 'distance'])

def get_gc_robot_color(team, player, team_map):
    """
    Determine the color of a GC robot based on team and player using the config.
    
    Args:
        team: Team number 
        player: Player number 
    
    Returns:
        str: robot color
    """
    # Check if it's the goalkeeper (assuming player 1 = goalkeeper)
    is_goalkeeper = (player == 1)
    
    if team == team_map.home:
        if is_goalkeeper:
            return team_map.home_goalkeeper_color
        else:
            return team_map.home_color
    elif team == team_map.away:
        if is_goalkeeper:
            return team_map.away_goalkeeper_color
        else:
            return team_map.away_color
    else:
        print(f"[COLOR] Unknown team {team}")
        return 'unknown'

# TEMP pending new vision model
def normalize_color_for_matching(color):
    """
    Normalize colors to handle common detection model errors.
    """
    color_same = {
        'red': {'red'},
        'blue': {'blue', 'green'},
        'black': {'black', 'green'},
        'white': {'white'},
        'yellow': {'yellow', 'red', 'green'},
        'gray': {'gray', 'grey', 'green'},
        'grey': {'gray', 'grey', 'green'},
    }
    
    return color_same.get(color.lower(), {color.lower()})    
def is_robot_flipped_origin(gc_x, gc_y, mario_x, mario_y, flip_threshold):
    """
    Returns True if the flipped position of the GC is significantly closer to MARIO than the normal one.
    """
    dist = np.linalg.norm([gc_x - mario_x, gc_y - mario_y])
    flipped_x, flipped_y = -gc_x, -gc_y
    dist_flipped = np.linalg.norm([flipped_x - mario_x, flipped_y - mario_y])
    return dist > flip_threshold and dist_flipped < flip_threshold

def verify_mario_id_loss(mario_df, mario_id, current_time, future_window_ms=2000):
    """
    Checks if a MARIO ID reappears within a future window.
    """
    future_frames = mario_df[
        (mario_df.gametime > current_time) & 
        (mario_df.gametime <= current_time + future_window_ms)
    ]
    return mario_id not in future_frames['id'].unique()

def calculate_assignment_cost(future_positions, gc_trajectory, current_time, current_distance, s_config):
    
    total_cost = current_distance * s_config.current_distance_weight

    future_cost_sum = 0
    distances = []
    distances.append(current_distance)
    if not future_positions.empty:

        for i in range(0, len(future_positions), s_config.future_sample_rate):
            future_pos = future_positions.iloc[i]
            gc_at_time = gc_trajectory[gc_trajectory.gametime <= future_pos.gametime]
            if gc_at_time.empty:
                continue
            
            gc_future_pos = gc_at_time.iloc[-1][['x', 'y']].values
            mario_future_pos = future_pos[['field_x', 'field_y']].values
            future_distance = np.linalg.norm(gc_future_pos - mario_future_pos)
            distances.append(future_distance)
            
            time_offset = future_pos.gametime - current_time
            weight = max(0.1, 1.0 - (time_offset / s_config.future_analysis_window_ms))
            weighted_cost = future_distance * weight
            future_cost_sum += weighted_cost
            

        avg_future_cost = future_cost_sum / max(1, len(distances))
        future_contribution = avg_future_cost
        total_cost += future_contribution

    
    return total_cost


def update_gc_flipped_state(current_assignments, gc_state_all, mario_robots, last_associated_mario_pos, gc_flipped_state, current_time, s_config):
    # debug
    debug_robot = (3, 5)  
    debug_interval = (540000, 542000) 
    
    for robot_key in gc_state_all.apply(lambda r: (r.team, r.player), axis=1):
        gc_robot_data = gc_state_all[(gc_state_all.team == robot_key[0]) & (gc_state_all.player == robot_key[1])]
        if gc_robot_data.empty:
            continue

        is_penalized = gc_robot_data.iloc[0].penalized
        if is_penalized:
            was_flipped = gc_flipped_state.get(robot_key, False)
            gc_flipped_state[robot_key] = False
            
            if robot_key in last_associated_mario_pos:
                del last_associated_mario_pos[robot_key]
            
            if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                print(f"\n[DEBUG FLIP] t={current_time/1000:.2f}s Robot {robot_key}:")
                print(f"  Robot is PENALIZED - RESETTING flip state")
                if was_flipped:
                    print(f"CHANGED FLIP STATE: {was_flipped} → False (due to penalty)")
            continue  # skip the rest of the logic for penalized robots
        
        gc_pos_raw = gc_robot_data.iloc[0][['x', 'y']].values
        gc_pos = np.array(gc_pos_raw, dtype=np.float64).reshape(-1)
        
        mario_id = current_assignments.get(robot_key)
        was_flipped = gc_flipped_state.get(robot_key, False)
        is_flipped = was_flipped  # default to previous state
        
        # define mario robots not assigned
        mario_robots_no_asg = mario_robots[~mario_robots.id.isin(current_assignments.values())]

        # DEBUG 
        if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
            print(f"\n[DEBUG FLIP] t={current_time/1000:.2f}s Robot {robot_key}:")
            print(f"  GC pos: ({gc_pos[0]:.1f}, {gc_pos[1]:.1f})")
            print(f"  Flipped pos would be: ({-gc_pos[0]:.1f}, {-gc_pos[1]:.1f})")
            print(f"  Mario ID assigned: {mario_id}")
            print(f"  Was flipped: {was_flipped}")
            print(f"  Current assignments: {current_assignments}")
            print(f"  Available mario robots: {len(mario_robots_no_asg)}")

        if mario_id is not None:
            # Case 1: robot has an assigned MARIO
            mario_robot = mario_robots[mario_robots.id == mario_id]
            if not mario_robot.empty:
                mario_pos = np.array(mario_robot.iloc[0][['field_x', 'field_y']].values, dtype=np.float64)
                dist = np.linalg.norm(gc_pos - mario_pos)
                flipped_pos = np.array([-gc_pos[0], -gc_pos[1]], dtype=np.float64)
                dist_flipped = np.linalg.norm(flipped_pos - mario_pos)

                if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                    print(f"  CASE 1: Robot has an assigned MARIO")
                    print(f"  Mario pos: ({mario_pos[0]:.1f}, {mario_pos[1]:.1f})")
                    print(f"  Dist normal: {dist:.1f}")
                    print(f"  Dist flipped: {dist_flipped:.1f}")
                    print(f"  Flip threshold: {s_config.flip_dist_threshold}")
                    print(f"  Condition dist > threshold: {dist} > {s_config.flip_dist_threshold} = {dist > s_config.flip_dist_threshold}")
                    print(f"  Condition flipped < threshold: {dist_flipped} < {s_config.flip_dist_threshold} = {dist_flipped < s_config.flip_dist_threshold}")

                if dist > s_config.flip_dist_threshold and dist_flipped < s_config.flip_dist_threshold:
                    is_flipped = True
                    if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                        print(f"  SETTING FLIPPED (dist too high, flipped close)")
                elif dist < s_config.flip_dist_threshold:
                    is_flipped = False
                    if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                        print(f"  UNSETTING FLIPPED (normal dist OK)")

        else:
            # Case 2: robot does NOT have an assigned MARIO
            last_mario_pos = last_associated_mario_pos.get(robot_key)
            if last_mario_pos is None:
                if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                    print(f"  CASE 2: Robot does NOT have an assigned MARIO")
                    print(f"  NO last_mario_pos (just came out of penalty?)")
                    print(f"  Keeping current flip state: {is_flipped}")
                continue
            if last_mario_pos is not None:
                if np.isscalar(last_mario_pos) or len(last_mario_pos) != 2:
                    if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                        print(f"  Invalid last_mario_pos format")
                    continue
                
                last_mario_pos = np.array(last_mario_pos, dtype=np.float64)
                search_radius = s_config.max_search_radius
                mario_positions_array = mario_robots_no_asg[['field_x', 'field_y']].values.astype(np.float64)
                
                if len(mario_positions_array) > 0:
                    distances_to_last = np.linalg.norm(mario_positions_array - last_mario_pos.reshape(1, -1), axis=1)
                    mario_positions = mario_positions_array[distances_to_last < search_radius]
                    
                    if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                        print(f"  CASE 2: Robot does NOT have an assigned MARIO")
                        print(f"  Last MARIO pos: ({last_mario_pos[0]:.1f}, {last_mario_pos[1]:.1f})")
                        print(f"  Search radius: {search_radius:.1f}")
                        print(f"  Distances to last pos: {distances_to_last}")
                        print(f"  Candidates in range: {len(mario_positions)}")
                        if len(mario_positions) > 0:
                            print(f"  Candidate positions: {mario_positions}")
                    
                    if len(mario_positions) > 0:
                        flipped_pos = np.array([-gc_pos[0], -gc_pos[1]], dtype=np.float64)
                        dists_flipped = np.linalg.norm(mario_positions - flipped_pos, axis=1)
                        min_dist_flipped = dists_flipped.min()
                        
                        dists_normal = np.linalg.norm(mario_positions - gc_pos, axis=1)
                        min_dist_normal = dists_normal.min()
                        
                        if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                            print(f"  All dists normal: {dists_normal}")
                            print(f"  All dists flipped: {dists_flipped}")
                            print(f"  Min dist normal: {min_dist_normal:.1f}")
                            print(f"  Min dist flipped: {min_dist_flipped:.1f}")
                            print(f"  Condition normal > threshold: {min_dist_normal} > {s_config.flip_dist_threshold} = {min_dist_normal > s_config.flip_dist_threshold}")
                            print(f"  Condition flipped < threshold*1.5: {min_dist_flipped} < {s_config.flip_dist_threshold * 1.5} = {min_dist_flipped < s_config.flip_dist_threshold * 1.5}")
                        
                        if min_dist_normal > s_config.flip_dist_threshold and min_dist_flipped < s_config.flip_dist_threshold * 1.5:
                            is_flipped = True
                            if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                                print(f" SETTING FLIPPED (unassigned, normal too far)")
                        elif min_dist_normal < s_config.flip_dist_threshold:
                            is_flipped = False
                            if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                                print(f" UNSETTING FLIPPED (unassigned, normal close)")
                        else:
                            if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
                                print(f" NO CHANGE (unassigned conditions not met)")

        if robot_key == debug_robot and debug_interval[0] <= current_time <= debug_interval[1]:
            print(f"  Final flipped state: {is_flipped}")
            if was_flipped != is_flipped:
                print(f" CHANGED FLIP STATE: {was_flipped} → {is_flipped}")

        gc_flipped_state[robot_key] = is_flipped
        

def calc_dist(mario_robot, gc_pos, robot_key, gc_flipped_state, current_time):
    if not mario_robot.empty:
        mario_pos = mario_robot.iloc[0][['field_x', 'field_y']].values
        gc_pos_to_use = gc_pos
        is_using_flipped = gc_flipped_state.get(robot_key, False)
        
        if is_using_flipped:
            gc_pos_to_use = np.array([-gc_pos[0], -gc_pos[1]])
            
        distance = np.linalg.norm(gc_pos_to_use - mario_pos)

        
        return distance
    else:
        return float('inf')
    
def mindist(mario_robots, gc_pos, distance):
    # Find the closest mario robot among all visible ones
    candidates_min = np.linalg.norm(mario_robots[['field_x', 'field_y']].values.astype(np.float32) - gc_pos.astype(np.float32), axis=1).min()
    return min(distance, candidates_min)
  
def remove_lost_assignments(updated_assignments, gc_state_all, mario_robots, mario_ids_visible, mario_df, current_time, updated_gc_only, updated_loss_times, association_frame_count, gc_flipped_state, current_frame, break_events, s_config):
    if break_events is None:
        break_events = []

    for robot_key, mario_id in list(updated_assignments.items()):
        # Find the current position of the GC-robot
        robot_gc_data_all = gc_state_all[(gc_state_all.team == robot_key[0]) & (gc_state_all.player == robot_key[1])]
        if robot_gc_data_all.empty:
            continue

        is_penalized = robot_gc_data_all.iloc[0].penalized
        gc_pos = robot_gc_data_all.iloc[0][['x', 'y']].values

        # Find the current position of the associated MARIO-robot
        mario_robot = mario_robots[mario_robots.id == mario_id]

        distance = calc_dist(mario_robot, gc_pos, robot_key, gc_flipped_state, current_time)


        dynamic_threshold = s_config.distance_break_threshold_max 

        break_due_to_distance = (
            distance > dynamic_threshold and
            mindist(mario_robots, gc_pos, distance) < distance * s_config.distance_better_factor
        )
        if break_due_to_distance:
            break_events.append({
                    'timestamp_ms': current_time,
                    'frame_n': current_frame,
                    'team': robot_key[0],
                    'player': robot_key[1],
                    'mario_id': mario_id,
                })

        if mario_id not in mario_ids_visible or break_due_to_distance or is_penalized:
            if verify_mario_id_loss(mario_df, mario_id, current_time) or break_due_to_distance or is_penalized:
                updated_assignments.pop(robot_key)
                if not robot_gc_data_all.empty and not is_penalized:
                    updated_gc_only.add(robot_key)
                    updated_loss_times[robot_key] = current_time

    return break_events

def update_reentered_robots(gc_state_active, updated_assignments, updated_gc_only, updated_loss_times, current_time):
    for _, robot_row in gc_state_active.iterrows():
        robot_key = (robot_row.team, robot_row.player)
        if (robot_key not in updated_assignments and 
            robot_key not in updated_gc_only):
            updated_gc_only.add(robot_key)
            updated_loss_times[robot_key] = current_time

def get_robots_to_assign(gc_state_active, updated_gc_only, current_time):
    robots_to_assign = []
    
    for key in updated_gc_only:
        robot_gc_data_active = gc_state_active[(gc_state_active.team == key[0]) & (gc_state_active.player == key[1])]
        

        if not robot_gc_data_active.empty:
            robots_to_assign.append(key)

    return robots_to_assign

def compute_candidates(robots_to_assign, gc_state_active, updated_loss_times, current_time, available_mario_robots, mario_df, gc_df, last_associated_mario_pos, gc_flipped_state, team_map, s_config):
    all_candidates = []
    
    # Debug 
    debug_time_start = 0 
    debug_time_end = 25 * 1000    
    debug_robots = [(13, 2)]  
    
    for robot_key in robots_to_assign:
        gc_robot_data = gc_state_active[(gc_state_active.team == robot_key[0]) & (gc_state_active.player == robot_key[1])]
        if gc_robot_data.empty: 
            continue
        gc_pos = gc_robot_data.iloc[0][['x', 'y']].values

        # Use the flipped position if the robot is flipped
        is_flipped = gc_flipped_state.get(robot_key, False)
        if is_flipped:
            gc_pos_for_search = np.array([-gc_pos[0], -gc_pos[1]])
        else:
            gc_pos_for_search = gc_pos
        
        time_since_loss = current_time - updated_loss_times.get(robot_key, current_time)
        search_radius = max(
            s_config.max_robot_speed, 
            min(s_config.max_robot_speed + (s_config.max_robot_speed) * time_since_loss, 
                s_config.max_search_radius)
        )
        
        gc_trajectory = gc_df[
            (gc_df.team == robot_key[0]) & (gc_df.player == robot_key[1]) &
            (gc_df.gametime >= current_time) & 
            (gc_df.gametime <= current_time + s_config.future_analysis_window_ms)
        ].sort_values('gametime')
        
        gc_robot_color = get_gc_robot_color(robot_key[0], robot_key[1], team_map)
        gc_color_same = normalize_color_for_matching(gc_robot_color)

        if (robot_key in debug_robots and 
            debug_time_start <= current_time <= debug_time_end):
            print(f"\n[COLOR DEBUG] t={current_time/1000:.2f}s Robot {robot_key}:")
            print(f"  GC robot color: {gc_robot_color}")
            print(f"  GC color equivalents: {gc_color_same}")
            print(f"  Available MARIO robots: {len(available_mario_robots)}")

        for _, mario_robot in available_mario_robots.iterrows():
            mario_pos = mario_robot[['field_x', 'field_y']].values
            mario_id = mario_robot.id

            # Color filtering con debug esteso
            if 'color' in mario_df.columns:
                # take colors of this mario_id
                mario_color_data = mario_df[mario_df.id == mario_id]
                if not mario_color_data.empty:
                    mario_colors = mario_color_data.iloc[0]['color']  # color list 
                    assert isinstance(mario_colors, (set, list, tuple))  # i.e. not a string or some other scalar
                    
                    # Debug per robot specifici
                    if (robot_key in debug_robots and 
                        debug_time_start <= current_time <= debug_time_end):
                        print(f"    Checking MARIO ID {mario_id}:")
                        print(f"      MARIO colors: {mario_colors}")
                    
                    # check if at least one of the MARIO colors matches the GC robot color
                    color_match = False
                    for mario_color in mario_colors:
                        mario_color_same = normalize_color_for_matching(mario_color)
                        if gc_color_same & mario_color_same:  
                            color_match = True
                            if (robot_key in debug_robots and 
                                debug_time_start <= current_time <= debug_time_end):
                                print(f"        COLOR MATCH: {mario_color} -> {mario_color_same}")
                            break
                        elif (robot_key in debug_robots and 
                                debug_time_start <= current_time <= debug_time_end):
                            print(f"        No match: {mario_color} -> {mario_color_same}")
                    
                    if not color_match:
                        # Debug esteso
                        if (robot_key in debug_robots and 
                            debug_time_start <= current_time <= debug_time_end):
                            print(f"      REJECTED for color mismatch")
                        continue  # Skip this candidate
                    else:
                        if (robot_key in debug_robots and 
                            debug_time_start <= current_time <= debug_time_end):
                            print(f"      ACCEPTED for color match")

            # Use the flipped position if the robot is flipped
            distance = np.linalg.norm(gc_pos_for_search - mario_pos)

            if distance <= search_radius:
                future_positions = mario_df[
                    (mario_df.id == mario_id) &
                    (mario_df.gametime > current_time) &
                    (mario_df.gametime <= current_time + s_config.future_analysis_window_ms)
                ].sort_values('gametime')
                cost = calculate_assignment_cost(
                    future_positions, gc_trajectory, current_time, distance, s_config
                )
                
                if (robot_key in debug_robots and 
                    debug_time_start <= current_time <= debug_time_end):
                    print(f"      CANDIDATE: distance={distance:.1f}, cost={cost:.1f}")
                
                all_candidates.append(Candidate(cost, robot_key, mario_robot.id, distance))
                
    return all_candidates

def assign_with_hungarian(all_candidates, updated_assignments, updated_gc_only, updated_loss_times, mario_df, current_time, last_associated_mario_pos, association_frame_count, s_config):
    if not all_candidates:
        return
    unassigned_gc_keys = sorted(list(set(candidate.robot_key for candidate in all_candidates)))
    available_mario_ids_list = sorted(list(set(candidate.mario_id for candidate in all_candidates)))
    if not unassigned_gc_keys or not available_mario_ids_list:
        return
    gc_map = {key: i for i, key in enumerate(unassigned_gc_keys)}
    mario_map = {id: i for i, id in enumerate(available_mario_ids_list)}
    cost_matrix = np.full((len(unassigned_gc_keys), len(available_mario_ids_list)), 100000)

    for candidate in all_candidates:
      
        if candidate.cost <= s_config.max_assignment_cost:
            gc_idx = gc_map[candidate.robot_key]
            mario_idx = mario_map[candidate.mario_id]
            cost_matrix[gc_idx, mario_idx] = candidate.cost
    if not np.any(np.isfinite(cost_matrix)):
        return
    

    row_ind, col_ind = linear_sum_assignment(cost_matrix)


    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost <= s_config.max_assignment_cost:
            robot_key = unassigned_gc_keys[r]
            mario_id = available_mario_ids_list[c]
            if mario_id not in updated_assignments.values():
                updated_assignments[robot_key] = mario_id
                updated_gc_only.discard(robot_key)
                if robot_key in updated_loss_times:
                    del updated_loss_times[robot_key]
    
            mario_row = mario_df[(mario_df.id == mario_id) & (mario_df.gametime == current_time)]
            if not mario_row.empty:
                last_associated_mario_pos[robot_key] = mario_row.iloc[-1][['field_x', 'field_y']].values



# TODO this could be turned into an object i think
def update_assignments_manually(
        gc_df, mario_df, current_assignments, gc_only, loss_times, current_time,
        last_associated_mario_pos, gc_state_raw, mario_state,
        association_frame_count, gc_flipped_state, current_frame, break_events,
        team_map, s_config):
    """
    Manual reassignment algorithm with dynamic search radius.
    """
    if break_events is None:
        break_events = []
    gc_state_all = gc_state_raw[gc_state_raw.x.notna()].copy()
    gc_state_active = gc_state_raw[(gc_state_raw.penalized == False) & (gc_state_raw.x.notna())].copy()
    mario_robots = mario_state[mario_state.type == 'robot'].copy()
    mario_ids_visible = set(mario_robots.id.unique())
    
    update_gc_flipped_state(current_assignments, gc_state_all, mario_robots, last_associated_mario_pos, gc_flipped_state, current_time, s_config)
    # 1. Remove lost assignments
    break_events = remove_lost_assignments(
        current_assignments, gc_state_all, mario_robots, mario_ids_visible, mario_df, current_time,
        gc_only, loss_times, association_frame_count, gc_flipped_state, current_frame, break_events, s_config
    )
    # 2. Handle re-entered robots
    update_reentered_robots(gc_state_active, current_assignments, gc_only, loss_times, current_time)

    # 3. Find robots to assign
    robots_to_assign = get_robots_to_assign(gc_state_active, gc_only, current_time)

    # 4. Compute candidates and assign
    if robots_to_assign and not mario_robots.empty:
        available_mario_ids = set(mario_robots.id.unique()) - set(current_assignments.values())
        available_mario_robots = mario_robots[mario_robots.id.isin(available_mario_ids)]

        all_candidates = compute_candidates(
            robots_to_assign, gc_state_active, loss_times, current_time, available_mario_robots, mario_df, gc_df, last_associated_mario_pos, gc_flipped_state, team_map, s_config
        )
        assign_with_hungarian(
            all_candidates, current_assignments, gc_only, loss_times, mario_df, current_time, last_associated_mario_pos, association_frame_count, s_config
        )

    # 5. Update last_associated_mario_pos for each gc robot, if it has not received an assignment, keep the last detected mario position    
    for _, robot_row in gc_state_all.iterrows():
        robot_key = (robot_row.team, robot_row.player)
        mario_id = current_assignments.get(robot_key)
        if mario_id is not None:
            association_frame_count[robot_key] = association_frame_count.get(robot_key, 0) + 1
            mario_row = mario_df[(mario_df.id == mario_id) & (mario_df.gametime == current_time)]
            last_associated_mario_pos[robot_key] = mario_row.iloc[-1][['field_x', 'field_y']].values
        else:
            association_frame_count[robot_key] = 0
            last_associated_mario_pos[robot_key] = last_associated_mario_pos.get(robot_key)
    

    return current_assignments, gc_only, loss_times, last_associated_mario_pos, gc_flipped_state, break_events