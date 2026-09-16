import cv2 as cv
import numpy as np

POINT_RADIUS = 5
POINT_LIMIT = 27
POINT_COLORS = [(0,0,255), (0,128,255), (255,255,0), (255,0,0), (0,255,0), (255,0,255), (0,255,255)]

def sqrdist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

class CSImage:
    def __init__(self, name, win_move, frame):
        self.name = name
        self.window_initted = False
        self.win_move = win_move
        self.base_image = frame.copy()
        self.points = np.empty([0,2], dtype=int)
        self.selected_point = -1
        self.center = [frame.shape[1]//2, frame.shape[0]//2]
        self.moving_center = False
        self.draw = None

    def _init_window(self):
        if not self.window_initted:
            cv.namedWindow(self.name)
            cv.moveWindow(self.name, *self.win_move)
            cv.setMouseCallback(self.name, self._mouse_callback)
            self.window_initted = True

    def window_is_visible(self):
        return cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) >= 1

    def __getitem__(self, *args):
        return self.base_image[args]

    def set_drawer(self, draw):
        self.draw = draw

    @property
    def shape(self):
        return self.base_image.shape

    def point_is_clicked(self, point_xy, mouse_xy):
        return sqrdist(point_xy, mouse_xy) <= POINT_RADIUS*POINT_RADIUS

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            selection_changed = False
            for i, p in enumerate(self.points):
                if self.point_is_clicked(p, (x,y)) and self.selected_point != i:
                    self.selected_point = i  # This is NOT drag and drop, unless sb wants to implement it
                    selection_changed = True

            if not selection_changed:
                if self.selected_point == -1:
                    if self.points.shape[0] < POINT_LIMIT:
                        self.points = np.append(self.points, [[x,y]], axis=0)
                else:
                    self.points[self.selected_point] = [x,y]
                    self.selected_point = -1
                print(self.points)

        elif event == cv.EVENT_MBUTTONDOWN:
            if self.name == "src":
                self.center = [x,y]
                self.moving_center = True
        elif event == cv.EVENT_MBUTTONUP:
            self.moving_center = False
        
        elif event == cv.EVENT_MOUSEMOVE:
            if self.moving_center:
                self.center = [x,y]

    def destroy_selected(self):
        if self.selected_point != -1:
            self.points = np.delete(self.points, self.selected_point, axis=0)
            self.selected_point = -1

    def _render(self, calibration=None):
        img = self.base_image.copy()

        if self.name == "src":
            if calibration is not None:
                self.draw.set_warp_params(calibration)
                self.draw.warped_field(img)
            self.draw.cross(img, self.center, (20,20), (0,0,0), 2)

        for i, [x,y] in enumerate(self.points):
            border = 1 if self.selected_point == i else -1
            cv.circle(img, (x,y), POINT_RADIUS, POINT_COLORS[i % len(POINT_COLORS)], border)
            cv.putText(img, str(i+1), (x+7, y-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, POINT_COLORS[i % len(POINT_COLORS)], 2)
        return img
    
    def imshow(self, calibration=None):
        if not self.window_initted:
            self._init_window()
        cv.imshow(self.name, self._render(calibration))

    def clamp_center(self):
        self.center[0] = min(max(self.center[0], 0), self.shape[1]-1)
        self.center[1] = min(max(self.center[1], 0), self.shape[0]-1)
