"""Bounding box utility class for object detection."""

from typing import Tuple, Union
import numpy as np
import numpy.typing as npt


class BoundingBox:
    """
    Bounding box representation supporting multiple formats.

    Supports formats:
    - x1y1wh: (top-left x, top-left y, width, height)
    - xcycwh: (center x, center y, width, height)
    - x1y1x2y2: (top-left x, top-left y, bottom-right x, bottom-right y)
    """

    def __init__(
        self,
        x1: Union[int, float],
        y1: Union[int, float],
        w: Union[int, float],
        h: Union[int, float],
        _int: bool = False
    ):
        """
        Initialize bounding box with x1, y1, width, height.

        Args:
            x1: Top-left x coordinate.
            y1: Top-left y coordinate.
            w: Width.
            h: Height.
            _int: If True, cast all values to integers.
        """
        self._int = _int
        if _int:
            self._x1 = int(x1)
            self._y1 = int(y1)
            self._w = int(w)
            self._h = int(h)
        else:
            self._x1 = float(x1)
            self._y1 = float(y1)
            self._w = float(w)
            self._h = float(h)

    @property
    def x1(self) -> Union[int, float]:
        """Top-left x coordinate."""
        return self._x1

    @property
    def y1(self) -> Union[int, float]:
        """Top-left y coordinate."""
        return self._y1

    @property
    def w(self) -> Union[int, float]:
        """Width."""
        return self._w

    @property
    def h(self) -> Union[int, float]:
        """Height."""
        return self._h

    @property
    def xc(self) -> Union[int, float]:
        """Center x coordinate."""
        if self._int:
            return self._x1 + self._w // 2
        else:
            return self._x1 + self._w / 2

    @property
    def yc(self) -> Union[int, float]:
        """Center y coordinate."""
        if self._int:
            return self._y1 + self._h // 2
        else:
            return self._y1 + self._h / 2

    @property
    def x2(self) -> Union[int, float]:
        """Bottom-right x coordinate."""
        return self._x1 + self._w

    @property
    def y2(self) -> Union[int, float]:
        """Bottom-right y coordinate."""
        return self._y1 + self._h

    @staticmethod
    def from_x1y1wh(
        x1: Union[int, float],
        y1: Union[int, float],
        w: Union[int, float],
        h: Union[int, float]
    ) -> "BoundingBox":
        """Create bounding box from (x1, y1, width, height) format."""
        return BoundingBox(x1, y1, w, h)

    @staticmethod
    def from_xcycwh(
        xc: Union[int, float],
        yc: Union[int, float],
        w: Union[int, float],
        h: Union[int, float]
    ) -> "BoundingBox":
        """Create bounding box from (center_x, center_y, width, height) format."""
        x1 = xc - w / 2
        y1 = yc - h / 2
        return BoundingBox(x1, y1, w, h)

    @staticmethod
    def from_x1y1x2y2(
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float]
    ) -> "BoundingBox":
        """Create bounding box from (x1, y1, x2, y2) format."""
        w = x2 - x1
        h = y2 - y1
        return BoundingBox(x1, y1, w, h)

    def as_int(self) -> "BoundingBox":
        """Return integer version of this bounding box."""
        return BoundingBox(self._x1, self._y1, self._w, self._h, _int=True)

    def to_x1y1wh(self) -> npt.NDArray[np.float32]:
        """Convert to (x1, y1, width, height) numpy array."""
        return np.array([self.x1, self.y1, self.w, self.h], dtype=np.float32)

    def to_xcycwh(self) -> npt.NDArray[np.float32]:
        """Convert to (center_x, center_y, width, height) numpy array."""
        return np.array([self.xc, self.yc, self.w, self.h], dtype=np.float32)

    def to_x1y1x2y2(self) -> npt.NDArray[np.float32]:
        """Convert to (x1, y1, x2, y2) numpy array."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def cut(self, img: npt.NDArray) -> npt.NDArray:
        """
        Extract the bounding box region from an image.

        Args:
            img: Input image (H, W, C) or (H, W) array.

        Returns:
            Cropped image region.
        """
        return img[int(self.y1):int(self.y2), int(self.x1):int(self.x2)]

    def __repr__(self) -> str:
        """String representation of the bounding box."""
        return f"BoundingBox(x1={self.x1}, y1={self.y1}, w={self.w}, h={self.h}, int={self._int})"
