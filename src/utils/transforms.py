from dataclasses import dataclass
import numpy as np

@dataclass
class TransformsCalculator:
    MARIO_POS_MIN: np.array
    MARIO_POS_MAX: np.array
    FIELD_POS_MIN: np.array
    FIELD_POS_MAX: np.array

    def interpolate(self, pos, in_min, in_max, out_min, out_max):
        pos_in_01 = (pos - in_min) / (in_max - in_min)
        return pos_in_01 * (out_max - out_min) + out_min
    # faster special case where in_min==out_max and in_max==out_min
    def flip(self, pos, min, max):
        return min + max - pos

    def image_to_field_x(self, img_tuple):
        return self.interpolate(img_tuple[0], self.MARIO_POS_MIN[0], self.MARIO_POS_MAX[0], self.FIELD_POS_MIN[0], self.FIELD_POS_MAX[0])
    def image_to_field_y(self, img_tuple):
        # flip Y axis in the passage from image to field
        y = self.flip(img_tuple[1], self.MARIO_POS_MIN[1], self.MARIO_POS_MAX[1])
        return self.interpolate(y, self.MARIO_POS_MIN[1], self.MARIO_POS_MAX[1], self.FIELD_POS_MIN[1], self.FIELD_POS_MAX[1])

    def field_to_image_x(self, field_tuple):
        return self.interpolate(field_tuple[0], self.FIELD_POS_MIN[0], self.FIELD_POS_MAX[0], self.MARIO_POS_MIN[0], self.MARIO_POS_MAX[0])
    def field_to_image_y(self, field_tuple):
        y = self.interpolate(field_tuple[1], self.FIELD_POS_MIN[1], self.FIELD_POS_MAX[1], self.MARIO_POS_MIN[1], self.MARIO_POS_MAX[1])
        # flip Y axis in the passage from field to image
        return self.flip(y, self.MARIO_POS_MIN[1], self.MARIO_POS_MAX[1])

    # https://stackoverflow.com/questions/57399915/how-do-i-determine-the-locations-of-the-points-after-perspective-transform-in-t
    def perspective_transform(self, p, matrix):
        px = int((matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2])))
        py = int((matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2])))
        return px, py
