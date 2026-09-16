from dataclasses import dataclass

from ..vision.calibration_studio import CalibrationStudio


@dataclass
class CalibrationStudioJob:
    frame: any
    config: any
    section_name: str
    logger: any
    recalibrate: bool

    # calibration happens one per section, so I'm not gonna care about the inefficiency of generating more pathsconfig objects rather than passing one from above
    @property
    def paths(self):
        return self.config.get_paths(mario_section_name=self.section_name)

def do_calibration_studio(job):
    calibrator = CalibrationStudio(job.frame, job.config)
    need_something = False

    camera_calibration_fname = job.paths.camera_calibration_npz
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
