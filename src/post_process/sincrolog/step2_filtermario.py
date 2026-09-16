import ast
import numpy as np
import pandas as pd

import src.post_process.sincrolog.s2_src.interpolate_ball as interpolation
import src.post_process.sincrolog.s2_src.process_colors as colors

def main(section_name, config):
    df = pd.read_csv(
        config.mario_csv_path(section_name),
        dtype={"frame": np.int64, "id": np.int32},
        converters={"bounding_box_in_image_space": ast.literal_eval},
    )

    # erase all track ids for the ball: there is only one ball, after all
    # TODO this line is still untested, I just added it after noticing,
    # while restoring nao vision, that there's a fallback from volov8 to v12
    # that may introduce ids different than -1 for the ball.
    # the test is postponed to the next use of sincrolog.
    df[df.type == "ball"].id = -1


    # ROBOT PERSISTENCE CRITERION
    # (while ball tracking is already pretty weak on its own in this version of MARIO)
    for the_id in df.id.unique():
        partial_df = df[df.id == the_id]
        if partial_df.iloc[0].type == "robot" and len(partial_df) < config.min_robot_persistence_frames:
            df.drop(index=partial_df.index, inplace=True)

    df = colors.go(df, config)
    df = interpolation.go(df, config)

    reordered_columns = list(df.columns)
    reordered_columns.remove("bounding_box_in_image_space")
    reordered_columns.append("bounding_box_in_image_space")
    df = df.reindex(columns=reordered_columns)

    mario_csv_path = config.mario_post_step2_path(section_name)
    df.to_csv(mario_csv_path, header=True, index=False)
