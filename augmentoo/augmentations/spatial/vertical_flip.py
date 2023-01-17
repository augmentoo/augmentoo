import numpy as np

from augmentoo.core.decorators import angle_2pi_range
from augmentoo.core.transforms_interface import DualTransform

__all__ = ["VerticalFlip"]


def vflip(img):
    return np.ascontiguousarray(img[::-1, ...])


@angle_2pi_range
def keypoint_vflip(keypoint, rows, cols):
    """Flip a keypoint vertically around the x-axis.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        rows (int): Image height.
        cols( int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint
    angle = -angle
    return x, (rows - 1) - y, angle, scale


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        return vflip(img)

    def apply_to_bbox(self, bbox, **params):
        return bbox_vflip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint_vflip(keypoint, **params)
