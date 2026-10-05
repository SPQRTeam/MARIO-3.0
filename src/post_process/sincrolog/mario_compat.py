import numpy as np

"""
Convert a sincrolog merged dataframe to the format of the original output of MARIO,
fo it can be injected back into MARIO-related features (e.g. the event processor).
"""
def merged_to_mario(df, gameinfo):
    df = df.rename(columns={"frame_n": "frame", "robot_x": "field_x", "robot_y": "field_y"})
    # filter out penalized robots, they're not on the field
    df = df[df.penalized == False]
    df["id"] = df.team * 1000 + df.player
    df["type"] = "robot"
    df["color"] = np.where(df.team == gameinfo["teams"]["home"]["number"], gameinfo["teams"]["home"]["fieldPlayerColor"], gameinfo["teams"]["away"]["fieldPlayerColor"])
    df["score_left"] = -1  # TEMP?
    df["score_right"] = -1  # TEMP?
    return df[["frame", "id", "type", "color", "bounding_box_in_image_space", "field_x", "field_y", "score_left", "score_right"]]
