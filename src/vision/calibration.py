import cv2 as cv
import numpy as np

class NullCalibrationError(ValueError):
    pass

class Calibration:
    def __init__(self, mtx, dist, rvec, tvec):
        self.mtx = mtx
        self.dist = dist
        self.rvec = rvec
        self.tvec = tvec
        self._Hcto = None

        # ensures validity and helps simplify same_params
        if self.mtx is None or self.dist is None or self.rvec is None or self.tvec is None:
            raise NullCalibrationError()

        # precalc these to save time
        R_calib, _ = cv.Rodrigues(self.rvec)
        H_obj_to_cam = self.mtx @ np.column_stack((R_calib[:, 0], R_calib[:, 1], self.tvec.flatten()))
        self.H_cam_to_obj = np.linalg.inv(H_obj_to_cam)

    @classmethod
    def load(cls, camera_calibration_path):
        calibration_npz = np.load(camera_calibration_path)
        return cls(
            mtx=calibration_npz["mtx"],
            dist=calibration_npz["dist"],
            rvec=calibration_npz["rvec"],
            tvec=calibration_npz["tvec"],
        )

    def same_params(self, other_mtx, other_dist, other_rvec, other_tvec):
        return (
            np.array_equal(self.mtx, other_mtx) and
            np.array_equal(self.dist, other_dist) and
            np.array_equal(self.rvec, other_rvec) and
            np.array_equal(self.tvec, other_tvec)
        )

    # returns the same tuple that cv.projectPoints does
    def project_points(self, points):
        return cv.projectPoints(points, self.rvec, self.tvec, self.mtx, self.dist)

    def antiproject_points(self, points):
        undistorted_pts = cv.undistortPoints(points, self.mtx, self.dist, P=self.mtx)
        return cv.perspectiveTransform(undistorted_pts, self.H_cam_to_obj)

    def antiproject_image(self, frame, dst_hw):
        undistorted_img = cv.undistort(frame, self.mtx, self.dist)
        return cv.warpPerspective(undistorted_img, self.H_cam_to_obj, (dst_hw[1], dst_hw[0]))
