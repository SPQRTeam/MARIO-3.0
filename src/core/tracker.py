"""Main tracking and processing module for MARIO."""

import contextlib
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2

from ..utils.opencv_qt_fonts import ensure_opencv_qt_fonts

ensure_opencv_qt_fonts()

import numpy as np
import pandas as pd
import tqdm
import yaml

from .detection import FrameIteratorFile, FrameIteratorStreaming, Detection
from ..utils.config.config_loader import MarioConfig, VISION_TYPE_TO_CNN
from ..utils.config.paths import ROOT_DIR
from ..utils.logger import setup_logger
from ..utils.drawing import Drawer, field_image_hw
from ..utils.team_info import NoTagLoader
from ..vision.color_cnn import NullColorCNN
from ..commentator import Commentator
from ..analysis.field_reprojection_view import FIELD_REPROJECTION_WINDOW_NAME
from ..commentator.commentary_display import COMMENTARY_DEBUG_WINDOW_NAME
from .realtime_display import RealtimeDisplayState
from .event_processor import EventProcessor
from data.team_names import TEAM_NAMES_GO

import src.core.files_handling as files_module
from .calibration_work import CalibrationStudioJob, do_calibration_studio
from .output_worker import OutputJob, do_output_work
from .graphics_thread import start_graphics_thread, shutdown_graphics_thread
from src.vision.calibration import Calibration
from src.vision.calibration_studio import CalibrationStudioAbortedError

import copy

from src.utils.team_info import make_team_mapping_lr

COLUMN_NAMES = [
    'frame', 'id', 'type', 'color', 'bounding_box_in_image_space',
    'field_x', 'field_y', 'score_left', 'score_right',
]

class Tracking:
    """
    Main tracking class for processing video frames with detection and tracking.
    """

    def __init__(self, config: MarioConfig, section_name: str, models: any, side_hint: any, calibration_timestamp_secs: float, log_level="INFO"):
        """
        Initialize tracking with configuration.

        Args:
            config: MARIO configuration object.
        """
        self.logger = setup_logger(__name__, level=log_level)
        self.config = config
        self.paths = config.get_paths(mario_section_name=section_name)
        self.models = models
        self.match_tts = models.match_tts
        self.section_name = section_name
        self.graphics_submission_queue = None
        self.graphics_return_queue = None
        self.graphics_thread = None
        if self.match_tts is not None:
            self.logger.info("Match commentary TTS enabled (LLM lines → speech).")
        self.track_history = defaultdict(lambda: [])
        self.track_values: Dict[int, List[Detection]] = {}
        self.track_color_names: Dict[int, str] = {}
        self.frame_id = 0
        self.working_files_tracker = files_module.TempWorkingFilesManager(self.paths.mario_section_dir)

        self.color_cnn = NullColorCNN()
        try:
            self.color_cnn = VISION_TYPE_TO_CNN[config.vision_type](config, log_level)
        except FileNotFoundError:
            self.logger.warning("Color CNN not found, defaulting to none.")
        except Exception as e:
            self.logger.warning(f"Could not load color CNN: {e}")

        use_v12_only = "HSL" in config.field_type
        robot_nested_suppression = config.robot_nested_suppression

        try:
            with open(self.paths.gameinfo, 'r') as f:
                self.gameinfo = yaml.load(f, Loader=NoTagLoader)
        except FileNotFoundError:
            with open(self.paths.default_gameinfo, 'r') as f:
                self.gameinfo = yaml.load(f, Loader=NoTagLoader)

        self.team_map = make_team_mapping_lr(self.gameinfo, side_hint, section_name, TEAM_NAMES_GO)
        self.team_map.dump(
            self.working_files_tracker.register_and_get_twf(
                self.paths.team_mapping_lr
            ),
            indent=4,
        )

        # Initialize frame iterator
        if config.is_streaming:
            self.video_iterator = FrameIteratorStreaming(
                streaming_url=config.streaming_url,
                time_range=config.time_range,
                models=models,
                calibration_frame_offset=calibration_timestamp_secs * config.processing.fps,
                bytetrack_config=ROOT_DIR / config.models.bytetrack_config,
                use_v12_only=use_v12_only,
                robot_nested_suppression=robot_nested_suppression,
                mario_config=config,
                team_map=self.team_map,
            )
        else:
            self.video_iterator = FrameIteratorFile(
                video_file=self.paths.source_video,
                models=models,
                calibration_frame_offset=calibration_timestamp_secs * config.processing.fps,
                bytetrack_config=ROOT_DIR / config.models.bytetrack_config,
                use_v12_only=use_v12_only,
                robot_nested_suppression=robot_nested_suppression,
                mario_config=config,
                team_map=self.team_map,
            )

        start_frame = config.processing.start_frame
        if start_frame:
            self.video_iterator.seek(start_frame)
            self.frame_id = start_frame

        # event processing
        self.eventproc = EventProcessor(
            config,
            self.working_files_tracker.register_and_get_twf(
                self.paths.symbolic_events_jsonl
            ),
            self.team_map
        )
        self.commentator = Commentator(config, self.team_map, self.eventproc)

        # OpenCV debug windows, ball trail, last event label — not part of tracking logic.
        self._display = RealtimeDisplayState(config)

        # Per track id: raw field positions and jersey color-name samples for temporal smoothing.
        self._field_pos_hist: Dict[int, Deque[Tuple[float, float]]] = {}
        self._robot_color_hist: Dict[int, Deque[str]] = {}

        self.logger.info("Tracking initialized successfully")


    def _assign_color_name_position(self, field_x: float, left_color: str, right_color: str) -> str:
        return left_color if field_x < 0 else right_color

    def _cleanup_runtime_outputs(self) -> None:
        """Best-effort cleanup for writers, async output worker, windows, and TTS."""
        async_out = self.config.features.async_output_pipeline
        show_ball_debug = self.config.features.show_ball_debug
        show_commentary_window = self.config.features.show_commentary_window

        if async_out and self.graphics_submission_queue is not None and self.graphics_return_queue is not None and self.graphics_thread is not None:
            try:
                shutdown_graphics_thread(
                    self.graphics_submission_queue,
                    self.graphics_return_queue,
                    self.graphics_thread,
                )
            except Exception as ex:
                self.logger.warning("Output worker shutdown warning: %s", ex)
            finally:
                self.graphics_submission_queue = None
                self.graphics_return_queue = None
                self.graphics_thread = None

        if self.match_tts is not None:
            try:
                self.match_tts.shutdown()
            except Exception as ex:
                self.logger.warning("Match TTS shutdown warning: %s", ex)
            finally:
                self.match_tts = None

        if show_ball_debug:
            try:
                cv2.destroyWindow(FIELD_REPROJECTION_WINDOW_NAME)
            except cv2.error:
                pass

        if bool(show_commentary_window) and not async_out:
            try:
                cv2.destroyWindow(COMMENTARY_DEBUG_WINDOW_NAME)
            except cv2.error:
                pass

    def _robot_color(self, track_id: int) -> str:
        return self.track_color_names.get(track_id, "unknown")

    @staticmethod
    def _median_field_xy(samples: Deque[Tuple[float, float]]) -> Tuple[float, float]:
        if not samples:
            return (0.0, 0.0)
        arr = np.array(list(samples), dtype=np.float64)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    @staticmethod
    def _majority_color_name(samples: Deque[str]) -> str:
        if not samples:
            return "unknown"
        return Counter(samples).most_common(1)[0][0]

    def _apply_temporal_smoothing(self, detections: List[Detection]) -> None:
        """Median field position + majority jersey color per track id."""
        feats = self.config.features
        if not feats.temporal_smooth_enabled:
            return
        pos_w = max(1, feats.temporal_smooth_position_window)
        col_w = max(1, feats.temporal_smooth_color_window)
        seen = set()
        for det in detections:
            if det.position is None:
                continue
            tid = int(det.id)
            seen.add(tid)
            fx, fy = float(det.position[0]), float(det.position[1])
            if tid not in self._field_pos_hist:
                self._field_pos_hist[tid] = deque(maxlen=pos_w)
            self._field_pos_hist[tid].append((fx, fy))
            det.position = self._median_field_xy(self._field_pos_hist[tid])

            if det.cls_name == "robot":
                color_name = self.track_color_names.get(tid, "unknown")
                if tid not in self._robot_color_hist:
                    self._robot_color_hist[tid] = deque(maxlen=col_w)
                self._robot_color_hist[tid].append(color_name)
                self.track_color_names[tid] = self._majority_color_name(self._robot_color_hist[tid])

        for tid in list(self._field_pos_hist.keys()):
            if tid not in seen:
                del self._field_pos_hist[tid]
        for tid in list(self._robot_color_hist.keys()):
            if tid not in seen:
                del self._robot_color_hist[tid]

    def track_image(self, out_data: Path, recalibrate: bool = False) -> None:
        """
        Main tracking loop - process video and save results.

        Args:
            out_data: Output CSV file path (unused when ``write_csv`` is false).
        """
        self.logger.info("Starting video processing")
        self.eventproc.begin_section()
        if self.match_tts is not None:
            self.match_tts.begin_section(self.config, self.section_name, start_frame=self.frame_id)

        transforms = self.config.make_transforms()
        # Drawer with H_inv=None: position_to_image_pipeline returns MARIO image space coords directly,
        # suitable for drawing on plan_view (the warped overhead image).
        plan_view_drawer = Drawer.NullDrawer(self.config)

        left_color = str(self.team_map.left_color_name)
        right_color = str(self.team_map.right_color_name)

        fps = self.config.processing.fps
        dst_hw = field_image_hw(self.config.field_config)

        OUT_FRAME_HEIGHT = self.config.processing.output_frame_height
        OUT_FRAME_WIDTH = int(
            self.video_iterator.frame_width *
            OUT_FRAME_HEIGHT /
            self.video_iterator.frame_height
        )
        fps = self.config.processing.fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        multiwriter = files_module.MultiWriter(
            csv_params=files_module.MultiWriterCSVParams(
                path=self.working_files_tracker.register_and_get_twf(
                    out_data
                ),
                header=COLUMN_NAMES,
            ) if self.config.features.write_csv else None,
            video_params=files_module.MultiWriterVideoParams(
                path=str(self.working_files_tracker.register_and_get_twf(
                    self.paths.annotated_video
                )),
                fourcc=fourcc,
                fps=fps,
                width=OUT_FRAME_WIDTH,
                height=OUT_FRAME_HEIGHT,
            ) if self.config.features.write_videos else None,
            plan_view_params=files_module.MultiWriterVideoParams(
                path=str(self.working_files_tracker.register_and_get_twf(
                    self.paths.planview_video
                )),
                fourcc=fourcc,
                fps=fps,
                width=dst_hw[1],
                height=dst_hw[0],
            ) if self.config.features.write_videos else None,
        )

        debug_timings = self.config.features.debug_frame_timings
        timing_every = max(1, self.config.features.debug_timing_every_n_frames)

        async_out = self.config.features.async_output_pipeline
        show_comm_win = self.config.features.show_commentary_window
        async_queue_sz = max(1, self.config.features.async_output_queue_size)
        if async_out:
            self.graphics_submission_queue, self.graphics_return_queue, self.graphics_thread = start_graphics_thread(
                queue_size=async_queue_sz,
                multiwriter=multiwriter,
                config=copy.deepcopy(self.config),
                logger=self.logger,
                kwargs_worker={
                    "tracker": self,
                    "dst_hw": dst_hw,
                    "plan_view_drawer": plan_view_drawer,
                },
            )
            self.logger.info(
                "async_output_pipeline: remap/annotate/write/OpenCV run in a background thread "
                "(queue_size=%s). Main thread: YOLO + field projection + CNN + rows + commentator. "
                "If the queue fills, the main thread blocks (backpressure) so exports stay complete.",
                async_queue_sz,
            )

        cs_job = CalibrationStudioJob(
            frame=self.video_iterator.calibration_frame,
            config=self.config,
            section_name=self.section_name,
            logger=self.logger,
            recalibrate=recalibrate,
        )
        if async_out:
            self.graphics_submission_queue.put(cs_job)
            calibration = self.graphics_return_queue.get()
            if isinstance(calibration, CalibrationStudioAbortedError):
                raise calibration
            assert isinstance(calibration, Calibration)
        else:
            calibration = do_calibration_studio(cs_job)

        # entering the multiwriter context will delete all output files. Keep the calibration before this line! So it can be safely cancelled
        writing_context = contextlib.nullcontext() if async_out else multiwriter
        with writing_context:  # manages opening and closing the multiwriter, but only if not async out. Even if I don't refer to this variable.
            # Process frames
            for frame, results, reading in tqdm.tqdm(self.video_iterator, total=len(self.video_iterator)):
                t_loop = time.perf_counter()

                self.track_values[self.frame_id] = results

                t1 = time.perf_counter()
                for i, det in enumerate(self.track_values[self.frame_id]):
                    box_t = (det.box.xc, det.box.y2)
                    box_t = calibration.antiproject_points(np.array([box_t], dtype=np.float32))
                    mario_pos = (box_t[0][0][0], box_t[0][0][1])
                    field_x = transforms.image_to_field_x(mario_pos)
                    field_y = transforms.image_to_field_y(mario_pos)
                    self.track_values[self.frame_id][i].position = (field_x, field_y)

                    if det.cls_name == "robot":
                        color_by_cnn = self.color_cnn.predict_color(det, frame)
                        if color_by_cnn is not None:
                            self.track_color_names[det.id] = color_by_cnn
                        else:
                            self.track_color_names[det.id] = self._assign_color_name_position(
                                field_x, left_color, right_color
                            )
                project_ms = (time.perf_counter() - t1) * 1000.0

                t_ball0 = time.perf_counter()
                raw_balls = [d for d in self.track_values[self.frame_id] if d.cls_name == "ball"]
                robots = [d for d in self.track_values[self.frame_id] if d.cls_name == "robot"]
                reject_oob = self.config.features.reject_ball_outside_field
                if reject_oob:
                    fc = self.config.field_config
                    margin = self.config.features.ball_inside_field_margin_mm

                    def _in_playfield(d: Detection) -> bool:
                        if d.position is None:
                            return False
                        fx, fy = d.position
                        return (
                            abs(float(fx)) <= float(fc.xlimit) + margin
                            and abs(float(fy)) <= float(fc.ylimit) + margin
                        )

                    balls = [d for d in raw_balls if _in_playfield(d)]
                    if not balls and raw_balls:
                        self._display.clear_ball_trail()
                else:
                    balls = raw_balls

                if balls:
                    self.track_values[self.frame_id] = robots + [
                        max(balls, key=lambda d: d.confidence)
                    ]
                else:
                    self.track_values[self.frame_id] = robots
                ball_filter_ms = (time.perf_counter() - t_ball0) * 1000.0

                self._apply_temporal_smoothing(self.track_values[self.frame_id])

                show_ball_dbg = self.config.features.show_ball_debug
                t_tr = time.perf_counter() if show_ball_dbg else None
                ball_debug = self._display.update_ball_debug(
                    self.frame_id,
                    float(fps),
                    self.track_values[self.frame_id],
                    plan_view_drawer,
                    show_ball_dbg,
                )
                ball_trail_ms = (time.perf_counter() - t_tr) * 1000.0 if t_tr is not None else 0.0

                possession_snapshot = self._display.possession_snapshot(
                    self.track_values[self.frame_id], show_comm_win, self._robot_color
                )

                t_rows = time.perf_counter()
                _rows = []
                for det in self.track_values[self.frame_id]:
                    field_x, field_y = det.position if det.position is not None else (None, None)
                    color = self._robot_color(det.id) if det.cls_name == "robot" else "ball"
                    _rows.append(
                        [
                            self.frame_id,
                            det.id,
                            det.cls_name,
                            color,
                            tuple(det.box.to_x1y1wh()),
                            field_x,
                            field_y,
                            reading.score_left,
                            reading.score_right,
                        ]
                    )
                build_rows_ms = (time.perf_counter() - t_rows) * 1000.0

                frame_df = pd.DataFrame(_rows, columns=COLUMN_NAMES)
                events = self.eventproc.process_frame(frame_df)

                commentary_event_label = None
                commentary_text = None
                ball_motion_debug = None
                t_comm = time.perf_counter()
                commentary_event_label, commentary_text, ball_motion_debug = self._display.commentary_labels(
                    events=events,
                    commentate=bool(self.config.features.commentate),
                    eventproc=self.eventproc,
                    commentator=self.commentator,
                    match_tts=self.match_tts,
                    llm=self.models.llm,
                    show_window=show_comm_win,
                )
                commentate_ms = (time.perf_counter() - t_comm) * 1000.0

                output_job = OutputJob(
                    frame_id=self.frame_id,
                    frame=frame.copy(),
                    detections=list(self.track_values[self.frame_id]),
                    rows=_rows,
                    reading=reading,
                    ball_debug=ball_debug,
                    commentary_event_label=commentary_event_label,
                    commentary_text=commentary_text,
                    possession_snapshot=possession_snapshot,
                    ball_motion_debug=ball_motion_debug,
                )
                if async_out:
                    assert self.graphics_submission_queue is not None and self.graphics_return_queue is not None
                    t_put = time.perf_counter()
                    self.graphics_submission_queue.put(output_job)
                    queue_put_ms = (time.perf_counter() - t_put) * 1000.0
                else:
                    output_times = do_output_work(output_job, calibration, multiwriter, self.config, self.logger, tracker=self, dst_hw=dst_hw, plan_view_drawer=plan_view_drawer)
                    queue_put_ms = 0.0

                tracker_ms = (time.perf_counter() - t_loop) * 1000.0
                det = self.video_iterator.last_iter_timings
                det_total = float(det.get("detector_total_ms", 0.0))
                pipeline_ms = det_total + tracker_ms

                if debug_timings and (self.frame_id % timing_every == 0):
                    if async_out:
                        self.logger.info(
                            "[timing:main] frame=%s pipeline_ms=%.1f | det: total=%.1f read=%.1f y12=%.1f "
                            "y8=%.1f nested=%.1f ocr_sched=%.1f | trk: project=%.1f ball_filt=%.1f ball_trail=%.1f "
                            "rows=%.1f comment=%.1f queue_put=%.1f tracker_sum=%.1f",
                            self.frame_id,
                            pipeline_ms,
                            det_total,
                            float(det.get("video_read_ms", 0.0)),
                            float(det.get("yolo_v12_ms", 0.0)),
                            float(det.get("yolo_v8_ms", 0.0)),
                            float(det.get("nested_suppress_ms", 0.0)),
                            float(det.get("ocr_schedule_ms", 0.0)),
                            project_ms,
                            ball_filter_ms,
                            ball_trail_ms,
                            build_rows_ms,
                            commentate_ms,
                            queue_put_ms,
                            tracker_ms,
                        )
                    else:
                        self.logger.info(
                            "[timing] frame=%s pipeline_ms=%.1f | det: total=%.1f read=%.1f y12=%.1f y8=%.1f "
                            "nested=%.1f ocr_sched=%.1f | trk: plan=%.1f project=%.1f ball_filt=%.1f annotate=%.1f "
                            "draw=%.1f field_repr=%.1f rows=%.1f write=%.1f preview=%.1f comment=%.1f waitkey=%.1f "
                            "tracker_sum=%.1f",
                            self.frame_id,
                            pipeline_ms,
                            det_total,
                            float(det.get("video_read_ms", 0.0)),
                            float(det.get("yolo_v12_ms", 0.0)),
                            float(det.get("yolo_v8_ms", 0.0)),
                            float(det.get("nested_suppress_ms", 0.0)),
                            float(det.get("ocr_schedule_ms", 0.0)),
                            output_times.plan_ms,
                            project_ms,
                            ball_filter_ms,
                            output_times.annotate_ms,
                            output_times.draw_overlays_ms,
                            output_times.field_reproj_ms,
                            build_rows_ms,
                            output_times.write_out_ms,
                            output_times.preview_ms,
                            commentate_ms,
                            output_times.waitkey_ms,
                            tracker_ms,
                        )

                self.frame_id += 1

            if async_out:
                shutdown_graphics_thread(
                    self.graphics_submission_queue,
                    self.graphics_return_queue,
                    self.graphics_thread,
                )
                self.graphics_submission_queue = None
                self.graphics_return_queue = None
                self.graphics_thread = None

        # Cleanup
        self._cleanup_runtime_outputs()
        self.working_files_tracker.finalize_and_cleanup()

        self.logger.info(f"Processing complete. Processed {self.frame_id} frames")
        self.logger.info(f"Output saved to: {out_data}")
