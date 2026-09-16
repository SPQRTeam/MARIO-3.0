import queue
import threading
from dataclasses import dataclass

import cv2

import numpy as np


from ..analysis.field_reprojection_view import FIELD_REPROJECTION_WINDOW_NAME
from ..commentator.commentary_display import COMMENTARY_DEBUG_WINDOW_NAME
from .output_worker import OutputJob, do_output_work
from .calibration_work import CalibrationStudioJob, do_calibration_studio
from src.vision.calibration_studio import CalibrationStudioAbortedError



def graphics_thread_loop(submission_queue, return_queue, multiwriter, config, logger, **kwargs_for_output):
    calibration = None

    while True:
        job = submission_queue.get()
        if isinstance(job, CalibrationStudioJob):
            try:
                calibration = do_calibration_studio(job)
                return_queue.put(calibration)
                break
            except CalibrationStudioAbortedError as e:
                return_queue.put(e)
                raise e
        else:
            raise ValueError(f"Job type {type(job)} is not accepted before calibration")

    assert calibration is not None

    # entering this context will delete all output files. Keep the calibration before this line! So it can be safely cancelled
    with multiwriter:
        while True:
            job = submission_queue.get()
            if isinstance(job, OutputJob):
                work_times = do_output_work(job, calibration, multiwriter, config, logger, **kwargs_for_output)
                if config.features.debug_frame_timings and (job.frame_id % max(1, config.features.debug_timing_every_n_frames) == 0):
                    logger.info(
                        "[timing:output_worker] frame=%s worker_sum_ms=%.1f | plan=%.1f annotate=%.1f "
                        "draw=%.1f field_repr=%.1f write=%.1f preview=%.1f waitkey=%.1f",
                        job.frame_id,
                        work_times.worker_ms,
                        work_times.plan_ms,
                        work_times.annotate_ms,
                        work_times.draw_overlays_ms,
                        work_times.field_reproj_ms,
                        work_times.write_out_ms,
                        work_times.preview_ms,
                        work_times.waitkey_ms,
                    )
            elif job is None:
                break  # done
            else:
                raise ValueError(f"Job type {type(job)} is not accepted after calibration")


def start_graphics_thread(
    *,
    queue_size: int,
    multiwriter: any,
    config: any,
    logger: any,
    kwargs_worker: dict,
):
    submission_queue = queue.Queue(maxsize=max(1, queue_size))
    return_queue = queue.Queue(maxsize=max(1, queue_size))
    t = threading.Thread(
        target=graphics_thread_loop,
        kwargs={
            "submission_queue": submission_queue,
            "return_queue": return_queue,
            "multiwriter": multiwriter,
            "config": config,
            "logger": logger,
            **kwargs_worker
        },
        daemon=True
    )
    t.start()
    return submission_queue, return_queue, t


def shutdown_graphics_thread(
    submission_queue,
    return_queue,  # doesn't matter rn but w/e, might want it
    thread: threading.Thread,
    *,
    join_timeout_sec: float = 120.0,
) -> None:
    submission_queue.put(None)
    thread.join(timeout=join_timeout_sec)
    try:
        cv2.destroyWindow(FIELD_REPROJECTION_WINDOW_NAME)
    except cv2.error:
        pass
    try:
        cv2.destroyWindow(COMMENTARY_DEBUG_WINDOW_NAME)
    except cv2.error:
        pass