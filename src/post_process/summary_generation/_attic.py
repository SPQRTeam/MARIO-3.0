"""Where old, less effective ideas are stashed just in case."""

import collections
import pandas as pd


"""
Get the (almost) raw position data from MARIO. Turns out they aren't digested too well on their own.
"""

def mario_csv_data(config, mario_section_name, home_team, away_team):
    mario_df = pd.read_csv(config.mario_csv_path(mario_section_name))
    mario_df = mario_df.drop(["bounding_box_in_image_space", "score_left", "score_right"], axis=1)

    mario_df["frame"] = mario_df["frame"] * 1000 / 30
    mario_df = mario_df.rename(columns={"frame": "time_ms"})

    color_to_name = {"color": {home_team.color: home_team.name, away_team.color: away_team.name}}
    mario_df = mario_df.replace(color_to_name)
    mario_df = mario_df.rename(columns={"color": "team_name"})

    return mario_df.to_csv(index=False, float_format="%.2f")



"""
Try a different hierarchy order to put individual differences of first and second half closer together.
Looked worse, but maybe can still be tweaked, you never know with
"""

def build_user_prompt_for_phase_compare_alt(main_data, data_by_phase, prev_output):
    homename = main_data.home_team.name
    awayname = main_data.away_team.name

    es1 = data_by_phase["firstHalf"].event_stats
    es2 = data_by_phase["secondHalf"].event_stats
    event_stats_by_half = {}
    for evname in es1:
        # yes, the normal dict also maintains insertion order since 3.6/3.7, but still, I especially care about this here.
        event_stats_by_half[evname] = collections.OrderedDict()
        event_stats_by_half[evname][homename + " (first half)"] = es1[evname][homename]
        event_stats_by_half[evname][awayname + " (first half)"] = es1[evname][awayname]
        event_stats_by_half[evname][homename + " (second half)"] = 0
        event_stats_by_half[evname][awayname + " (second half)"] = 0
    for evname in es2:
        if evname not in event_stats_by_half:
            event_stats_by_half[evname] = collections.OrderedDict()
            event_stats_by_half[evname][homename + " (first half)"] = 0
            event_stats_by_half[evname][awayname + " (first half)"] = 0
        event_stats_by_half[evname][homename + " (second half)"] = es2[evname][homename]
        event_stats_by_half[evname][awayname + " (second half)"] = es2[evname][awayname]

    return f"""
You have already produced the following summary of the game as a whole:

```
{prev_output}
```

Now enhance this by adding a shorter paragraph in the same style comparing the statistics of the first and the second half, possibly describing how the situation evolved during the course of the game.
Focus ONLY on this, refrain from making general statements about the game since that's already been done previously.

The following is the JSON data regarding the game statistics, broken down by first and second half.

{event_stats_by_half} 
"""
