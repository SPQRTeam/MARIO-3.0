import numpy as np
import scipy
import pandas as pd

def _pos(ball):  # readability
    return (ball[["field_x", "field_y"]]).to_numpy()

def _get_ball(balldf, frame):
    ball = balldf[balldf['frame'] == frame]
    if ball.empty:
        return None
    ball = ball.iloc[0]  # need a smarter idea to handle cases with multiple detections
    return ball

# find instant velocity, i.e. use only the indicated frame and a close one in the specified scan direction.
# the ball is too unpredictable for any average.
def _get_velocity(balldf, frame, scan_dir, delta_frame):
    # setup
    indicated_ball = _get_ball(balldf, frame)
    other_ball = None
    assert scan_dir==1 or scan_dir==-1
    assert indicated_ball is not None

    # find next ball
    while other_ball is None and delta_frame > 0:
        other_ball = _get_ball(balldf, frame + delta_frame*scan_dir)
        delta_frame -= 1
    delta_frame += 1

    # calculate velocity
    if other_ball is None:
        return None
    delta_pos = scan_dir * (_pos(other_ball) - _pos(indicated_ball))
    return delta_pos / delta_frame


def _template(ball1):
    return {
        # frame to be specified
        "id": ball1.id,
        "type": "ball",
        "color": ball1.color,
        # field_x and field_y to be specified
        "bounding_box_in_image_space": (-1,-1,-1,-1),
    }

def _jump_interp(p1, p2, alpha):
    return p1 if alpha < 0.5 else p2

def _interpolate_moving(frame1, frame2, balldf, delta_frame, output):
    dt = frame2 - frame1
    ball1 = _get_ball(balldf, frame1)
    vel1 = _get_velocity(balldf, frame1, -1, delta_frame)
    ball2 = _get_ball(balldf, frame2)
    vel2 = _get_velocity(balldf, frame2, 1, delta_frame)
    if vel1 is not None and vel2 is not None:
        # Memo: there's also Hermite RBF interp if we wanna implement it
        # https://stackoverflow.com/questions/75560981/how-to-interpolate-position-using-velocity-and-position-samples#79699137
        interpolator = scipy.interpolate.CubicHermiteSpline(
            [frame1, frame2],
            [_pos(ball1), _pos(ball2)],
            [vel1, vel2],
        )
    else:
        # linear interpolation fallback
        interpolator = scipy.interpolate.make_interp_spline(
            [frame1, frame2],
            [_pos(ball1), _pos(ball2)],
            k=1,
        )
    for i in range(1, dt):
        p = interpolator(frame1+i)
        newdata = _template(ball1)
        newdata["frame"] = frame1 + i
        newdata["field_x"] = p[0]
        newdata["field_y"] = p[1]
        output.append(newdata)

def _interpolate_still(ball1, ball2, dt, frame1, output):
    for i in range(1, dt):
        p = _jump_interp(_pos(ball1), _pos(ball2), i/dt)
        newdata = _template(ball1)
        newdata["frame"] = frame1 + i
        newdata["field_x"] = p[0]
        newdata["field_y"] = p[1]
        output.append(newdata)

def go(df, config):
    balldf = df[df['type'] == 'ball']
    frames_with_ball = balldf['frame'].unique()
    interpolation_rows = []
    for i in range(len(frames_with_ball) - 1):
        frame1 = frames_with_ball[i]
        frame2 = frames_with_ball[i+1]
        dt = frame2 - frame1

        # wanna handle cases where the ball disappears
        if dt <= 1:
            continue

        ball1 = _get_ball(balldf, frame1)
        ball2 = _get_ball(balldf, frame2)
        assert ball1 is not None and ball2 is not None
        dist = np.linalg.norm(_pos(ball1) - _pos(ball2))

        # if the disappearance is short enough, fill in missing frames via interpolation
        if dt <= config.max_frames_for_moving_ball_interp:
            _interpolate_moving(frame1, frame2, balldf, config.mario_postprocessing.max_frame_lookup_for_velocity, interpolation_rows)
        
        # if the ball remained still, fill in the missing frames by having it remain still
        elif dist <= config.mario_postprocessing.max_field_dist_for_still_ball_interp and dt <= config.max_frames_for_still_ball_interp:
            _interpolate_still(ball1, ball2, dt, frame1, interpolation_rows)

    df = pd.concat([df, pd.DataFrame(interpolation_rows)])
    df.sort_values(by=["frame", "type"], axis=0, inplace=True, ignore_index=True)
    return df

