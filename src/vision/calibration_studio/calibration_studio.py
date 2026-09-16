import cv2 as cv
import numpy as np
import argparse
from pathlib import Path
import src.utils as utils
from src.utils.drawing import Drawer, render_field_image

from ..calibration import Calibration, NullCalibrationError
from .params_gui import DParams, CSParamGUI
from .image import CSImage


class CalibrationStudioAbortedError(RuntimeError):
    pass


class CalibrationStudio:
    MAX_K_ORDER = 3
    def __init__(self, src_im, config):
        dst_im, corners_px = render_field_image(config.field_config)
        # images
        self.src = CSImage("src", (80, 80), src_im)
        self.dst = CSImage("dst", (100+src_im.shape[1], 80), dst_im)
        self.ds = DParams()  # this is the new ks
        self.d_gui = CSParamGUI("D-params initial guess", self.ds, bottom_left_anchor=(100+src_im.shape[1], 70+src_im.shape[0]))
        # warping parameters
        self.mtx = None
        self.dist = None
        self.rvec = None
        self.tvec = None
        # calibration caching so I don't have to remake it again and again with the same values
        self._calibration = None
        self.dst.points = np.array(corners_px)
        self.src.set_drawer(Drawer(config, None))
        self.used = False
        self.result_ok = False

    @property
    def center(self):
        return self.src.center

    # returns success true/false
    def try_load(self, camera_calibration_path):
        if not camera_calibration_path.exists():
            return False
        calibration_npz = np.load(camera_calibration_path)
        self.ds.array = calibration_npz["ds"]
        self.src.center = calibration_npz["frame_center"]
        self.src.points = calibration_npz["src_points"]
        self.dst.points = calibration_npz["dst_points"]
        self.mtx = calibration_npz["mtx"]
        self.dist = calibration_npz["dist"]
        self.rvec = calibration_npz["rvec"]
        self.tvec = calibration_npz["tvec"]
        return True

    def save(self, camera_calibration_path):
        np.savez(
            camera_calibration_path,
            ds=self.ds.array,
            frame_center=self.center,
            src_points=self.src.points,
            dst_points=self.dst.points,
            mtx=self.mtx,
            dist=self.dist,
            rvec=self.rvec,
            tvec=self.tvec,
            allow_pickle=False,
        )

    @property
    def calibration(self):
        if self._calibration is None or not self._calibration.same_params(self.mtx, self.dist, self.rvec, self.tvec):
            try:
                self._calibration = Calibration(mtx=self.mtx, dist=self.dist, rvec=self.rvec, tvec=self.tvec)
            except NullCalibrationError:
                self._calibration = None
        return self._calibration

    def show_src(self):
        self.src.imshow(self.calibration)

    def show_dst(self):
        self.dst.imshow()

    def go(self):
        if self.used:
            raise RuntimeError("Already used.")
        self.used = True

        print("""INSTRUCTIONS

        Homography:
        - Click on the images to select at least 4 points on both of them.
        - Click on a previously placed point to select it, then click somewhere else to reposition it, or press X to delete it.
            
        Radial:
        - Once you're done with the homography, use Q for a big increase, A for a big decrease, W for a small increase, or S for a small decrease.
        - Use the number keys to select the order (1, 2 and 3 are supported).
        - Use J/L to move the distortion center left/right, I/K to move it up/down.
        - You can also use middle mouse button to change the transformation center.
        All this dumb stuff because OPENCV DOES NOT SUPPORT REMOVING THAT USELESS CONTEXT MENU ON MOUSE RIGHT AND ZOOM ON MOUSE WHEEL.
        This system would have been MUCH cooler if it weren't for that.

        Check:
        - Press R to review the final result of the transformation.
        - Press F instead to view the same result, but with an overlay of the field on top.
        These do NOT update automatically (too expensive), so press R or F to refresh manually.

        Finish: press Space.""")


        current_order = 1
        prev_src_pts = None
        prev_dst_pts = None
        prev_center = None
        prev_ds = None

        while True:
            self.src.imshow(self.calibration)
            self.dst.imshow()
            self.d_gui.imshow()
            key = cv.waitKey(1)

            if not self.src.window_is_visible() or not self.dst.window_is_visible() or not self.d_gui.window_is_visible():
                raise CalibrationStudioAbortedError

            if key == ord('x'):
                self.src.destroy_selected()
                self.dst.destroy_selected()

            n = min(self.src.points.shape[0], self.dst.points.shape[0])
            homography_ready = n >= 8

            if key == 32:
                break
            elif homography_ready:
                order_string = f"k{current_order}"
                if key == ord('q'):
                    self.ds[order_string] += self.ds.PARAM_SPECS_BY_NAME[order_string].shift_step
                    print(self.ds, "selected order:", current_order)
                elif key == ord('a'):
                    self.ds[order_string] -= self.ds.PARAM_SPECS_BY_NAME[order_string].shift_step
                    print(self.ds, "selected order:", current_order)
                elif key == ord('w'):
                    self.ds[order_string] += self.ds.PARAM_SPECS_BY_NAME[order_string].step
                    print(self.ds, "selected order:", current_order)
                elif key == ord('s'):
                    self.ds[order_string] -= self.ds.PARAM_SPECS_BY_NAME[order_string].step
                    print(self.ds, "selected order:", current_order)

                elif key in {ord(d) for d in '123456789'}:
                    int_key = int(chr(key))
                    if int_key <= self.MAX_K_ORDER:
                        current_order = int_key
                        print(self.ds, "selected order:", current_order)

            self.d_gui.selected_param = f"k{current_order}"

            # center movement with j/k/i/l (always active)
            CENTER_STEP = 1
            if key == ord('j'):
                self.src.center[0] -= CENTER_STEP
            elif key == ord('l'):
                self.src.center[0] += CENTER_STEP
            elif key == ord('i'):
                self.src.center[1] -= CENTER_STEP
            elif key == ord('k'):
                self.src.center[1] += CENTER_STEP

            if homography_ready:
                objp = np.hstack([self.dst.points[:n], np.zeros([n,1])], dtype=np.float32)
                imgp = self.src.points[:n].astype(np.float32, copy=True)
                center_changed = prev_center is None or tuple(prev_center) != tuple(self.src.center)
                ds_changed = prev_ds is None or not np.allclose(prev_ds, self.ds.array, atol=0.0, rtol=0.0)
                points_changed = (
                    prev_src_pts is None
                    or imgp.shape != prev_src_pts.shape
                    or objp.shape != prev_dst_pts.shape
                    or (imgp != prev_src_pts).any()
                    or (objp != prev_dst_pts).any()
                )
                if points_changed or center_changed or ds_changed:
                    self.src.clamp_center()
                    dist_h, dist_w = self.src.base_image.shape[:2]
                    K_guess = np.array([
                        [dist_w,      0, self.src.center[0]], 
                        [     0, dist_w, self.src.center[1]], 
                        [     0,      0,                 1]], dtype=np.float32)
                    # some flags are commented b/c I tried with and without them and it's better without. The final 0 is just to allow each line to end in |
                    flags = (
                        cv.CALIB_USE_INTRINSIC_GUESS |
                        cv.CALIB_FIX_PRINCIPAL_POINT |
                        # cv.CALIB_FIX_ASPECT_RATIO |
                        # cv.CALIB_FIX_K1 |
                        # cv.CALIB_FIX_K2 |
                        # cv.CALIB_FIX_K3 |
                        # cv.CALIB_ZERO_TANGENT_DIST |
                        0)
                    ret, self.mtx, self.dist, rvecs, tvecs = cv.calibrateCamera([objp], [imgp], (dist_w, dist_h), K_guess, self.ds.array, flags=flags)
                    assert ret
                    self.rvec = rvecs[0]
                    self.tvec = tvecs[0]
                    self.src.draw.set_warp_params(self.calibration)
                    prev_src_pts = imgp
                    prev_dst_pts = objp
                    prev_center = tuple(self.src.center)
                    prev_ds = self.ds.array.copy()

                if self.mtx is not None and (key == ord('r') or key == ord('f')):
                    plan_view = self.calibration.antiproject_image(self.src.base_image, self.dst.shape)
                    if key == ord('f'):
                        # merge_view
                        for i in range(0,plan_view.shape[0]):
                            for j in range(0, plan_view.shape[1]):
                                try:
                                    if(self.dst.base_image[i,j].min() > 15):
                                        plan_view[i,j] = np.array([0,0,255])
                                except IndexError:
                                    pass
                    cv.imshow("plan_view", plan_view)

        cv.destroyAllWindows()
        self.result_ok = True








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, default=None, help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument('--game-name', '--game', '-g', type=str, default=None, help="Name of the game to work with (a directory in data/). If unspecified, will use the last one used in this project.")
    parser.add_argument('--section', '-s', type=str, help="Section to work with. ONLY THE LETTER.")
    args = parser.parse_args()
    config = utils.MarioConfig.from_yaml(args.config, args.game_name)

    cap = cv.VideoCapture(str(config.video_path(f"mario_{args.section}")))
    _, src_im = cap.read()

    CalibrationStudio(src_im, config).go()
