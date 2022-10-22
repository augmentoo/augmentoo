import math

import cv2
import numpy as np

from augmentoo.core.decorators import angle_2pi_range
from augmentoo.core.targets.bbox import AxisAlignedBoxTarget
from augmentoo.core.transforms_interface import DualTransform

__all__ = ["HorizontalFlip"]


def hflip(img):
    return img[:, ::-1, ...]


def hflip_cv2(img):
    return cv2.flip(img, 1)


@angle_2pi_range
def keypoint_hflip(keypoint, rows, cols):
    """Flip a keypoint horizontally around the y-axis.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint
    angle = math.pi - angle
    return (cols - 1) - x, y, angle, scale



class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return hflip_cv2(img)

        return hflip(img)

    def apply_to_bbox(self, bbox, **params):
        return AxisAlignedBoxTarget.bbox_hflip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint_hflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()
