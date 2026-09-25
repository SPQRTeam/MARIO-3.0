import cv2 as cv

from src.utils.opencv_qt_fonts import ensure_opencv_qt_fonts

ensure_opencv_qt_fonts()

import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import pandas as pd
import ast
import json
import argparse
import math
import traceback
from collections import defaultdict
import math
import src.utils as utils
import tqdm
from pathlib import Path
from src.utils.drawing import Drawer, TEAM_COLORS
from src.vision.calibration import Calibration
import src.core.files_handling as files_module


# sorry but multiprocessing w/o globals is a major pain
def init(section_name):
    global MARIO_START_TIME, FLIP_TEAM, team_map, cap, width, height, \
        frame_count, fps, out, gc_df, mario_df, sincrolog_df, draw, \
        working_files_tracker

    paths = config.get_paths(gc_section_name=section_name)
    working_files_tracker = files_module.TempWorkingFilesManager(paths.gc_section_dir)

    team_map = utils.extract_team_mapping_from_yaml(paths.gameinfo)

    # Carica i parametri dal file di configurazione JSON
    with open(paths.section_params, 'r', encoding='utf-8') as f:
        params = json.load(f)

    MARIO_START_TIME = params["manual"]["mario_start_time"]
    FLIP_TEAM = params["manual"]["gc_flip_team"]
    mario_half_name = params["manual"]["mario_half_name"]

    paths = config.get_paths(gc_section_name=section_name, mario_section_name=mario_half_name)

    cap = cv.VideoCapture(str(paths.source_video))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # center loaded below
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    assert round(fps)==30, f"fps expected to be 30 but actually is {fps}"

    if args.fused:
        output_filename = "FUSIONVIZ.mp4"
    else:
        output_filename = "MARIOVIZ.mp4"
    out = cv.VideoWriter(
        str(working_files_tracker.register_and_get_twf(paths.sincrolog_output_dir / output_filename)),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )  

    gc_df = pd.read_csv(paths.gc_post_step3_csv)
    mario_df = pd.read_csv(
        paths.mario_post_step2_csv,
        converters={
            'bounding_box_in_image_space': ast.literal_eval,
            'color': ast.literal_eval,
        },
    )
    if args.fused:
        sincrolog_df = pd.read_csv(
            paths.merged_csv,
            converters={'bounding_box_in_image_space': ast.literal_eval},
        )
        #sincrolog_ball_df = pd.read_csv(output_dir / "ball_dataset.csv")
    else:
        sincrolog_df = None

    calibration = Calibration.load(paths.camera_calibration_npz)
    draw = Drawer(config, calibration)

def end():
    working_files_tracker.finalize_and_cleanup()



# SIDE-EFFECT
def process_frame_fused(frame, frame_idx):
    timestamp = frame_idx / fps * 1000 - MARIO_START_TIME

    if args.show_gc_field:
        draw.warped_field(frame)

    # assuming the info from the dataset is dense enough, otherwise this would need to be a persistent state
    is_flipped = defaultdict(lambda: False)

    # MARIO-robots
    x = sincrolog_df[sincrolog_df.timestamp_ms <= timestamp].drop_duplicates(("player","team"), keep="last")
    for _, row in x.iterrows():
        is_flipped[(row.team, row.player)] = row.flipped
        if row.position_source != "mario_dynamic":
            assert row.position_source == "gc_mario_gap" or "mario" not in row.position_source
            continue
        pos = draw.position_to_image_pipeline(row.robot_x, row.robot_y)
        draw.centered_rect(frame, pos, (20,20), (0,255,255), -1)
        bbx, bby, bbw, bbh = tuple(int(a) for a in row.bounding_box_in_image_space)
        cv.rectangle(frame, (bbx, bby), (bbx+bbw, bby+bbh), (0,255,255), 2)
        text_pos = (pos[0]-10, pos[1]+10)
        text_color = TEAM_COLORS[team_map.home_color] if row.team == team_map.home else TEAM_COLORS[team_map.away_color]
        cv.putText(frame, str(row.player), text_pos, cv.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv.fillConvexPoly(frame, (np.array([[bbx+bbw-20, bby+10], [bbx+bbw+21, bby+2], [bbx+bbw+6, bby-19]])), (0,255,255))
        cv.putText(frame, str(row.player), (bbx+bbw-5, bby+5), cv.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
    
    penalized_nonflip_team = set()
    penalized_flip_team = set()

    # GC-robots
    x = gc_df[gc_df.gametime <= timestamp].drop_duplicates(("player","team"), keep="last")
    for _, row in x.iterrows():
        if row.penalized:
            if row.team == FLIP_TEAM:
                penalized_flip_team.add(row.player)
            else:
                penalized_nonflip_team.add(row.player)
            continue
        color = TEAM_COLORS[team_map.home_color] if row.team == team_map.home else TEAM_COLORS[team_map.away_color]
        pos = draw.position_to_image_pipeline(row.x, row.y, is_flipped[(row.team, row.player)])
        cv.circle(frame, pos, 20, color, -1)
        text_pos = (pos[0]-10, pos[1]+10)
        cv.putText(frame, str(row.player), text_pos, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if is_flipped[(row.team, row.player)]:
            draw.flip_border(frame, pos, 20, 4)

    if penalized_nonflip_team:
        color = TEAM_COLORS[team_map.home_color] if team_map.home != FLIP_TEAM else TEAM_COLORS[team_map.away_color]
        cv.putText(frame, "Pen. "+" ".join(str(p) for p in sorted(penalized_nonflip_team)), (0, height-height//40), cv.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    if penalized_flip_team:
        color = TEAM_COLORS[team_map.home_color] if team_map.home == FLIP_TEAM else TEAM_COLORS[team_map.away_color]
        cv.putText(frame, "Pen. "+" ".join(str(p) for p in sorted(penalized_flip_team)), (width//2, height-height//40), cv.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    # ball
    # balls = sincrolog_ball_df[sincrolog_ball_df.timestamp_ms <= timestamp]
    # if not balls.empty:
    #         ball_row = balls.iloc[-1]
    #         if getattr(ball_row, "is_playing", True):
    #             if not math.isnan(ball_row.chosen_ball_x) and not math.isnan(ball_row.chosen_ball_y):
    #                 pos = draw.position_to_image_pipeline(ball_row.chosen_ball_x, ball_row.chosen_ball_y)
    #                 cv2_x(frame, pos, (20,20), (0, 255, 0), 2)  
    #             if not math.isnan(ball_row.mario_ball_x):
    #                 pos = draw.position_to_image_pipeline(ball_row.mario_ball_x, ball_row.mario_ball_y)
    #                 draw.centered_rect(frame, pos, (20,20), (255,255,255), 2)
    #             if not math.isnan(ball_row.gc_ball_x):
    #                 pos = draw.position_to_image_pipeline(ball_row.gc_ball_x, ball_row.gc_ball_y)
    #                 cv.circle(frame, pos, 20, (255,255,255), 2)

def process_frame_mario(frame, frame_idx):
    timestamp = frame_idx / fps * 1000 - MARIO_START_TIME

    if args.show_gc_field:
        draw.warped_field(frame)

    # MARIO-robots
    x = mario_df[mario_df.frame == frame_idx]
    for _, row in x.iterrows():
        bbx, bby, bbw, bbh = tuple(int(a) for a in row.bounding_box_in_image_space)
        draw.multicolor_rect(frame, (bbx, bby), (bbx+bbw, bby+bbh), row.color, 1)
        if row.type == "robot":
            pos = draw.position_to_image_pipeline(row.field_x, row.field_y)
            draw.centered_rect(frame, pos, (5,5), (0,0,0), 1)
            cv.putText(frame, str(row.id), (bbx, bby-5), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 3)
            cv.putText(frame, str(row.id), (bbx, bby-5), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

    # GC-robots
    x = gc_df[gc_df.gametime <= timestamp].drop_duplicates(("player","team"), keep="last")
    for _, row in x.iterrows():
        if row.penalized:
            continue
        color = TEAM_COLORS[team_map.home_color] if row.team == team_map.home else TEAM_COLORS[team_map.away_color]
        pos = draw.position_to_image_pipeline(row.x, row.y)
        cv.circle(frame, pos, 4, color, -1)

def process_frame_parallel(frame_idx):
    try:
        shm = SharedMemory(name="mariofield_batch", create=False)
        batch = np.ndarray((args.batch_size, height, width, 3), dtype=np.uint8, buffer=shm.buf)
        if args.fused:
            process_frame_fused(batch[frame_idx % args.batch_size], frame_idx)
        else:
            process_frame_mario(batch[frame_idx % args.batch_size], frame_idx)
        shm.close()
    except Exception:
        print(traceback.format_exc())


# used to be temporary, but it's still somewhat useful as a simpler test bed
# and for compatibility with certain Windows systems.
# not guaranteed to have the latest features though.
def main_legacy():
    frame_idx = math.ceil(MARIO_START_TIME / 1000 * fps)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    progressbar = tqdm.tqdm(total=frame_count - frame_idx)

    while cap.isOpened():

        _ret, frame = cap.read()

        if not _ret:
            print("Could not read (stream finished?)")
            break
        
        if args.fused:
            process_frame_fused(frame, frame_idx)
        else:
            process_frame_mario(frame, frame_idx)
        frame_idx += 1

        out.write(frame)

        progressbar.update(args.batch_size)

        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord('q'):
            break

        if args.time_limit >= 0 and frame_idx >= args.time_limit * fps:
            print("Time limit reached")
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()


def main_parallel():
    process_number = cv.getNumberOfCPUs()
    shm = SharedMemory(name="mariofield_batch", create=True, size=args.batch_size*height*width*3)
    batch = np.ndarray((args.batch_size, height, width, 3), dtype=np.uint8, buffer=shm.buf)

    frame_idx = math.ceil(MARIO_START_TIME / 1000 * fps)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)

    progressbar = tqdm.tqdm(total=frame_count - frame_idx)

    while cap.isOpened():

        idxs = []

        while len(idxs) < args.batch_size:
            _ret, frame = cap.read()
            if _ret:
                batch[frame_idx % args.batch_size] = frame
                idxs.append(frame_idx)
                frame_idx += 1
            else:
                break

        pool = multiprocessing.Pool(process_number)
        pool.imap_unordered(process_frame_parallel, idxs, chunksize=1)
        pool.close()
        pool.join()

        for i in idxs:
            out.write(batch[i % args.batch_size])
        
        progressbar.update(len(idxs))

        cv.imshow("frame", batch[0])
        if cv.waitKey(1) == ord('q'):
            break

        if not _ret:
            print("Could not read (stream finished?)")
            break

        if args.time_limit >= 0 and frame_idx >= args.time_limit * fps:
            print("Time limit reached")
            break

    shm.close()
    shm.unlink()
    cap.release()
    out.release()
    cv.destroyAllWindows()



parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=Path, default=None, help="Path to config.yaml (default: ./config.yaml)")
parser.add_argument('--mario', '-m', action="store_true", help="Make MARIO visualization")
parser.add_argument('--fused', '-f', action="store_true", help="Make FUSED visualization")
parser.add_argument('--game-name', '--game', '-g', type=str, default=None, help="Name of the game to work with (a directory in data/). If unspecified, will use the last one used in this project.")
parser.add_argument('--sections', '-s', nargs='+', type=str, help="Sections to work with. ONLY THE NUMBERS. Defaults to all.")
parser.add_argument("--time-limit", "-l", type=int, default=-1)
parser.add_argument("--batch-size", "-b", type=int, default=300)
parser.add_argument("--no-parallel", action="store_false", dest="parallel")
parser.add_argument("--show-gc-field", action="store_true")
args = parser.parse_args()
args.streaming = False  # for MarioConfig compatibility
args.field_type = None  # for MarioConfig compatibility
args.vision_type = None  # for MarioConfig compatibility

if args.mario + args.fused != 1:
    parser.error("Please select exactly one visualization mode (mario or fused).")

config = utils.MarioConfig.from_yaml(args)
paths = config.get_paths()

section_names = utils.get_section_names_to_do(paths.game_dir, config.dir_names.gc_section_prefix, args.sections)

for sn in section_names:
    try:
        init(sn)
    except pd.errors.EmptyDataError:
        print(f"Empty data for {sn}, it's probably bad")
        continue

    if args.parallel:
        main_parallel()
    else:
        main_legacy()

    end()
    