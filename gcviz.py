import cv2 as cv
import numpy as np
import argparse
import math
import tqdm
from pathlib import Path

from src.utils.opencv_qt_fonts import ensure_opencv_qt_fonts
ensure_opencv_qt_fonts()

import src.utils as utils
from src.utils.drawing import Drawer, TEAM_COLORS
from src.post_process.game_controller.data_transmutation import DataTransmuter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, default=None, help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--yaml", "-y", type=Path, required=True, help="Path to the original GC yaml log")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=960)   # Halved default width
    parser.add_argument('--height', type=int, default=540)  # Halved default height
    parser.add_argument("--field-type", "--field", "-f", type=str, required=True, choices=utils.FIELD_TYPE_TO_FILE.keys(), help="The field this game was played on.")
    args = parser.parse_args()
    # compat with MarioConfig
    args.streaming = False
    args.game_name = None

    # explicitly do NOT pass the field_type to MarioConfig...
    _field_type = args.field_type
    args.field_type = None
    config = utils.MarioConfig.from_yaml(args)
    # ...then override the cached_property so it doesn't try to check or update the txt saved in the game (which we don't have here)
    config.field_type = _field_type

    # 1. Parse YAML using DataTransmuter
    metadata, data = DataTransmuter(
        gc_log_path=args.yaml,
        section_prefix=config.dir_names.gc_section_prefix,
    ).stir()

    # 2. Initialize Drawer
    draw = Drawer.NullDrawer(config)

    # 3. Process each Section
    for section_name, section in data.items():
        process_section(section_name, section, metadata.params.game, draw, args)


def process_section(section_name, section, gameinfo, draw, args):
    gc_df = section.players_collective_records

    if gc_df.empty:
        print(f"No GC data found for {section_name}.")
        return
    
    phase = section.phase
    if phase not in {"firstHalf", "secondHalf"}:
        print(f"Warning: Unexpected phase '{phase}' in section {section_name}. Defaulting to firstHalf.")
        phase = "firstHalf"

    # Determine which team defends the right goal and store their team number
    home_defends_right = (gameinfo.sideMapping == "homeDefendsRightGoal")
    if phase == "secondHalf":
        home_defends_right = not home_defends_right
        
    flip_team = gameinfo.teams.home.number if home_defends_right else gameinfo.teams.away.number

    gc_records = gc_df.sort_values("gctime").to_dict('records')

    output_filename = f"{section_name}_IDEAL_VIZ.mp4"
    out = cv.VideoWriter(output_filename, cv.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))

    base_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    draw.warped_field(base_frame)

    max_time_ms = gc_records[-1]['gctime']
    frame_count = math.ceil((max_time_ms / 1000.0) * args.fps)

    robot_state = {}
    event_idx = 0
    total_events = len(gc_records)
    
    # Initialize game_state directly from the first record
    game_state = {
        'gamestate': gc_records[0]['gamestate'],
        'secsremaining': gc_records[0]['secsremaining'],
        'gamestopped': gc_records[0]['gamestopped']
    }

    progressbar = tqdm.tqdm(total=frame_count, desc=f"gcviz {section_name}")

    scale_factor = args.width / 1920.0
    radius = max(5, int(20 * scale_factor))
    txt_scale = max(0.4, 1.0 * scale_factor)
    txt_thick = max(1, int(2 * scale_factor))
    pen_scale = max(0.6, 2.0 * scale_factor)
    pen_thick = max(1, int(3 * scale_factor))

    for frame_idx in range(frame_count):
        timestamp = (frame_idx / args.fps) * 1000

        while event_idx < total_events and gc_records[event_idx]['gctime'] <= timestamp:
            ev = gc_records[event_idx]
            robot_state[(ev['team'], ev['player'])] = ev
            
            game_state['gamestate'] = ev['gamestate']
            game_state['secsremaining'] = ev['secsremaining']
            game_state['gamestopped'] = ev['gamestopped']
                
            event_idx += 1

        frame = base_frame.copy()
        penalized_home = set()
        penalized_away = set()

        for (team, player), row in robot_state.items():
            is_home = (team == gameinfo.teams.home.number)
            
            if row['penalized']:
                if is_home:
                    penalized_home.add(player)
                else:
                    penalized_away.add(player)
                continue

            color = TEAM_COLORS[gameinfo.teams.home.fieldPlayerColor] if is_home else TEAM_COLORS[gameinfo.teams.away.fieldPlayerColor]
            
            x, y = row['x'], row['y']
            
            # Flip coordinates if the robot belongs to the team defending the right goal
            if team == flip_team:
                x, y = -x, -y

            pos = draw.position_to_image_pipeline(x, y)

            cv.circle(frame, pos, radius, color, -1)
            cv.putText(frame, str(player), (pos[0] - radius//2, pos[1] + radius//2),
                       cv.FONT_HERSHEY_SIMPLEX, txt_scale, (255, 255, 255), txt_thick)

        if penalized_home:
            cv.putText(frame, "Pen. " + " ".join(str(p) for p in sorted(penalized_home)),
                       (20, args.height - max(10, args.height // 40)), cv.FONT_HERSHEY_SIMPLEX, pen_scale, TEAM_COLORS[gameinfo.teams.home.fieldPlayerColor], pen_thick)
        if penalized_away:
            cv.putText(frame, "Pen. " + " ".join(str(p) for p in sorted(penalized_away)),
                       (args.width // 2, args.height - max(10, args.height // 40)), cv.FONT_HERSHEY_SIMPLEX, pen_scale, TEAM_COLORS[gameinfo.teams.away.fieldPlayerColor], pen_thick)

        # Draw Right-Side UI text (State, Time, Stop)
        ui_x = int(args.width * 0.82)
        ui_y_state = max(25, int(50 * scale_factor))
        ui_y_time = ui_y_state + max(25, int(40 * scale_factor))
        ui_y_stop = ui_y_time + max(25, int(40 * scale_factor))

        cv.putText(frame, game_state['gamestate'].capitalize(), (ui_x, ui_y_state), 
                   cv.FONT_HERSHEY_SIMPLEX, pen_scale, (255, 255, 255), pen_thick)

        time_sign = "-" if game_state['secsremaining'] < 0 else ""
        abs_secs = abs(game_state['secsremaining'])
        time_str = f"{time_sign}{abs_secs // 60:02d}:{abs_secs % 60:02d}"
        
        cv.putText(frame, time_str, (ui_x, ui_y_time), 
                   cv.FONT_HERSHEY_SIMPLEX, pen_scale, (255, 255, 255), pen_thick)

        if game_state['gamestopped']:
            cv.putText(frame, "STOP", (ui_x, ui_y_stop), 
                       cv.FONT_HERSHEY_SIMPLEX, pen_scale, (0, 0, 255), pen_thick)

        out.write(frame)
        progressbar.update(1)

    out.release()

if __name__ == "__main__":
    main()