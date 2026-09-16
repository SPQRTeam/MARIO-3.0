from dataclasses import dataclass

from ..vision.calibration_studio import CalibrationStudio


@dataclass
class CalibrationStudioJob:
    frame: any
    config: any
    section_name: str
    logger: any
    recalibrate: bool

def do_calibration_studio(job):
    calibrator = CalibrationStudio(job.frame, job.config)
    need_something = False

    camera_calibration_fname = job.config.game_dir / job.section_name / "camera_calbration.npz"
    loaded = calibrator.try_load(camera_calibration_fname)
    if loaded:
        job.logger.info("Camera calibration loaded from cache")
    else:
        job.logger.info("Camera calibration not found")
        need_something = True

    if need_something or job.recalibrate:
        job.logger.info("Starting Calibration Studio...")
        calibrator.go()
        assert calibrator.result_ok
        calibrator.save(camera_calibration_fname)
        job.logger.info("Calibration saved successfully.")

    return calibrator.calibration
