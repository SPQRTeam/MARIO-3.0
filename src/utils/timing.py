def get_gc_state_at_time(gc_df, time, future_offset=500):
    valid_messages = gc_df[gc_df.gametime <= time + future_offset]
    latest_states = valid_messages.groupby(['team', 'player']).last().reset_index()
    return latest_states

def get_mario_state_at_time(mario_df, time, tolerance=0):
    return mario_df[(mario_df.gametime >= time - tolerance) & (mario_df.gametime <= time + tolerance)]
