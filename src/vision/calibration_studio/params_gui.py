from dataclasses import dataclass

import cv2 as cv
import numpy as np
from munch import Munch

class DParams:
    PARAM_SPECS = [
        Munch(name="k1", step=0.1, shift_step=0.3, visible=True),
        Munch(name="k2", step=0.1, shift_step=0.3, visible=True),
        Munch(name="p1", step=0.1, shift_step=0.3, visible=False),
        Munch(name="p2", step=0.1, shift_step=0.3, visible=False),
        Munch(name="k3", step=0.1, shift_step=0.3, visible=True),
    ]
    PARAM_SPECS_BY_NAME = {ps.name: ps for ps in PARAM_SPECS}
    PARAM_INDICES = {ps.name: i for i, ps in enumerate(PARAM_SPECS)}

    def __init__(self):
        self.array = np.zeros(len(self.PARAM_SPECS), dtype=np.float32)

    # access params via name
    def __getitem__(self, key):
        return self.array[self.PARAM_INDICES[key]]

    def __setitem__(self, key, value):
        self.array[self.PARAM_INDICES[key]] = value

    # for the sake of my debug prints
    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return repr(self.array)


# this is actually generic, could be used for more other sets of params
@dataclass
class GUIButton:
    action: str
    x1: int
    y1: int
    x2: int
    y2: int

    def in_bounding_box(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

@dataclass
class GUIRow:
    p_name: str
    y_baseline: int  # takes the text baseline as reference
    buttons: list[GUIButton]

class CSParamGUI:
    def __init__(self, window_name, params, bottom_left_anchor=None):
        self.params = params  # will read and write here!
        self.window_name = window_name
        self.bottom_left_anchor = bottom_left_anchor
        self.window_initted = False
        self.visible_specs = [ps for ps in self.params.PARAM_SPECS if ps.visible]
        self.selected_param = ""  # this doesn't matter for the gui, it just shows which param is selected for keyboard adjustment
        self.selected_marker_points = np.empty((3,2), dtype=np.int64)  # allocate it once and write on it every time before drawing

        # UI Styling
        row_height = 50
        self.width = 370
        self.height = max(100, len(self.visible_specs) * row_height)
        bg_color = (40, 40, 40)
        text_color = (255, 255, 255)
        btn_color = (120, 120, 120)
        self.value_color = (100, 255, 100)
        self.selected_color = (50, 50, 255)
        x_name = 60
        x_btn1_x1 = 120
        x_btn1_x2 = 160
        x_btn2_x1 = 170
        x_btn2_x2 = 210
        self.x_value = 230

        # save useful info on the rows
        self.rows: list[GUIRow] = []
        current_y = 0

        for ps in self.visible_specs:
            buttons = [
                GUIButton("-", x_btn1_x1, current_y + 10, x_btn1_x2, current_y + 40),
                GUIButton("+", x_btn2_x1, current_y + 10, x_btn2_x2, current_y + 40)
            ]
            self.rows.append(GUIRow(p_name=ps.name, y_baseline=current_y + 35, buttons=buttons))
            current_y += row_height

        # static base elements
        self.base_img = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)

        for row in self.rows:
            # param name
            cv.putText(self.base_img, row.p_name, (x_name, row.y_baseline), cv.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

            # buttons
            for btn in row.buttons:
                cv.rectangle(self.base_img, (btn.x1, btn.y1), (btn.x2, btn.y2), btn_color, -1)
                cv.putText(self.base_img, btn.action, (btn.x1 + 12, row.y_baseline - 2), cv.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            # and the value isn't static of course :p

    def _init_window(self):
        if not self.window_initted:
            cv.namedWindow(self.window_name)
            cv.setMouseCallback(self.window_name, self._mouse_event)

            if self.bottom_left_anchor is not None:
                target_x, target_y = self.bottom_left_anchor
                cv.moveWindow(self.window_name, target_x, target_y - self.height)

            self.window_initted = True

    def window_is_visible(self):
        return cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) >= 1

    def _mouse_event(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            shift_pressed = (flags & cv.EVENT_FLAG_SHIFTKEY) != 0

            for row in self.rows:
                for btn in row.buttons:
                    if btn.in_bounding_box(x, y):
                        p_spec = self.params.PARAM_SPECS_BY_NAME[row.p_name]
                        amt = p_spec.shift_step if shift_pressed else p_spec.step

                        if btn.action == "-":
                            self.params[row.p_name] -= amt
                        elif btn.action == "+":
                            self.params[row.p_name] += amt

    def _render(self):
        img = self.base_img.copy()

        for row in self.rows:
            # python doesn't have a format that omits trailing zeros but doesn't switch to scientific notation for small numbers :(
            val_str = f"{self.params[row.p_name]:.4f}"
            if "." in val_str:
                val_str = val_str.rstrip("0")
                if val_str.endswith("."):
                    val_str += "0"

            cv.putText(img, val_str, (self.x_value, row.y_baseline), cv.FONT_HERSHEY_SIMPLEX, 0.6, self.value_color, 2)

            if row.p_name == self.selected_param:
                self.selected_marker_points[:] = [[28, row.y_baseline - 25], [13, row.y_baseline - 15], [50, row.y_baseline - 7]]
                cv.fillPoly(img, [self.selected_marker_points], self.selected_color)

        return img

    def imshow(self):
        if not self.window_initted:
            self._init_window()
        cv.imshow(self.window_name, self._render())
