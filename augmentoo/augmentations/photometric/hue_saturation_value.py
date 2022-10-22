import random
import warnings

import cv2
import numpy as np


from augmentoo.core.decorators import preserve_shape
from augmentoo.core.targets import is_rgb_image, is_grayscale_image
from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
    to_tuple,
)

__all__ = ["HueSaturationValue"]


def _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)
        hue = cv2.LUT(hue, lut_hue)

    if sat_shift != 0:
        lut_sat = np.arange(0, 256, dtype=np.int16)
        lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype(dtype)
        sat = cv2.LUT(sat, lut_sat)

    if val_shift != 0:
        lut_val = np.arange(0, 256, dtype=np.int16)
        lut_val = np.clip(lut_val + val_shift, 0, 255).astype(dtype)
        val = cv2.LUT(val, lut_val)

    img = cv2.merge((hue, sat, val)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        hue = cv2.add(hue, hue_shift)
        hue = np.mod(hue, 360)  # OpenCV fails with negative values

    if sat_shift != 0:
        sat = clip(cv2.add(sat, sat_shift), dtype, 1.0)

    if val_shift != 0:
        val = clip(cv2.add(val, val_shift), dtype, 1.0)

    img = cv2.merge((hue, sat, val))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


@preserve_shape
def shift_hsv(img, hue_shift, sat_shift, val_shift):
    if hue_shift == 0 and sat_shift == 0 and val_shift == 0:
        return img

    is_gray = is_grayscale_image(img)
    if is_gray:
        if hue_shift != 0 or sat_shift != 0:
            hue_shift = 0
            sat_shift = 0
            warnings.warn(
                "HueSaturationValue: hue_shift and sat_shift are not applicable to grayscale image. "
                "Set them to 0 or use RGB image"
            )
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.dtype == np.uint8:
        img = _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift)
    else:
        img = _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift)

    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


class HueSaturationValue(ImageOnlyTransform):
    """Randomly change hue, saturation and value of the input image.

    Args:
        hue_shift_limit ((int, int) or int): range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
        sat_shift_limit ((int, int) or int): range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
        val_shift_limit ((int, int) or int): range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        always_apply=False,
        p=0.5,
    ):
        super(HueSaturationValue, self).__init__(always_apply, p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0, **params):
        if not is_rgb_image(image) and not is_grayscale_image(image):
            raise TypeError("HueSaturationValue transformation expects 1-channel or 3-channel images.")
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return {
            "hue_shift": random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
            "sat_shift": random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
            "val_shift": random.uniform(self.val_shift_limit[0], self.val_shift_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("hue_shift_limit", "sat_shift_limit", "val_shift_limit")
