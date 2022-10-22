import math
import random

import numpy as np

from augmentoo.core.decorators import angle_2pi_range
from augmentoo.core.transforms_interface import DualTransform

__all__ = ["RandomRotate90"]


def image_rot90(img, factor):
    img = np.rot90(img, factor)
    return img


def bbox_rot90(bbox, factor, rows, cols):  # skipcq: PYL-W0613
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)

    Args:
        bbox (tuple): A bounding box tuple (x_min, y_min, x_max, y_max).
        factor (int): Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        tuple: A bounding box tuple (x_min, y_min, x_max, y_max).

    """
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
    x_min, y_min, x_max, y_max = bbox[:4]
    if factor == 1:
        bbox = y_min, 1 - x_max, y_max, 1 - x_min
    elif factor == 2:
        bbox = 1 - x_max, 1 - y_max, 1 - x_min, 1 - y_min
    elif factor == 3:
        bbox = 1 - y_max, x_min, 1 - y_min, x_max
    return bbox


@angle_2pi_range
def keypoint_rot90(keypoint, factor, rows, cols, **params):
    """Rotates a keypoint by 90 degrees CCW (see np.rot90)

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        factor (int): Number of CCW rotations. Must be in range [0;3] See np.rot90.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if factor not in set {0, 1, 2, 3}

    """
    x, y, angle, scale = keypoint[:4]

    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter n must be in set {0, 1, 2, 3}")

    if factor == 1:
        x, y, angle = y, (cols - 1) - x, angle - math.pi / 2
    elif factor == 2:
        x, y, angle = (cols - 1) - x, (rows - 1) - y, angle - math.pi
    elif factor == 3:
        x, y, angle = (rows - 1) - y, x, angle + math.pi / 2

    return x, y, angle, scale


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        return image_rot90(img, factor)

    def get_params(self):
        # Random int in the range [0, 3]
        return {"factor": random.randint(0, 3)}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return bbox_rot90(bbox, factor, **params)

    def apply_to_keypoint(self, keypoint, factor=0, **params):
        return keypoint_rot90(keypoint, factor, **params)

    def get_transform_init_args_names(self):
        return ()
