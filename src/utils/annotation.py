import cv2
from .drawing import TEAM_COLORS

BALL_COLOR = (255, 255, 255)  # white BGR

def annotate(box, color, track_id, frame):
    x1, y1, x2, y2 = box.as_int().to_x1y1x2y2()
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    cv2.rectangle(frame, p1, p2, color, thickness=2)
    cv2.putText(frame, f"ID: {track_id}", (p1[0], p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)
    return frame

def elaborate_track_results_and_annotate(frame, *, detections, max_tracks_display, get_robot_color_name):
    annotated_frame = frame.copy()

    detections_sorted = sorted(
        detections,
        key=lambda d: d.confidence,
        reverse=True
    )[:max_tracks_display]

    for det in detections_sorted:
        if det.cls_name == "ball":
            color = BALL_COLOR
        else:
            color = TEAM_COLORS[get_robot_color_name(det.id)]
        annotate(det.box, color, f"{det.id} {det.cls_name}", annotated_frame)

    return annotated_frame