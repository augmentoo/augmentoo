import random

import cv2
import numpy as np

from augmentoo.core.decorators import preserve_shape, clipped
from augmentoo.core.dtypes import MAX_VALUES_BY_DTYPE
from augmentoo.core.targets import is_rgb_image
from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
    to_tuple,
)

__all__ = ["RGBShift"]


@clipped
def _shift_rgb_non_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        return img + r_shift

    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = img[..., i] + shift

    return result_img


def _shift_image_uint8(img, value):
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]

    lut = np.arange(0, max_value + 1).astype("float32")
    lut += value

    lut = np.clip(lut, 0, max_value).astype(img.dtype)
    return cv2.LUT(img, lut)


@preserve_shape
def _shift_rgb_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        h, w, c = img.shape
        img = img.reshape([h, w * c])

        return _shift_image_uint8(img, r_shift)

    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = _shift_image_uint8(img[..., i], shift)

    return result_img


def shift_rgb(img, r_shift, g_shift, b_shift):
    if img.dtype == np.uint8:
        return _shift_rgb_uint8(img, r_shift, g_shift, b_shift)

    return _shift_rgb_non_uint8(img, r_shift, g_shift, b_shift)


class RGBShift(ImageOnlyTransform):
    """Randomly shift values for each channel of the input RGB image.

    Args:
        r_shift_limit ((int, int) or int): range for changing values for the red channel. If r_shift_limit is a single
            int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
        g_shift_limit ((int, int) or int): range for changing values for the green channel. If g_shift_limit is a
            single int, the range  will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
        b_shift_limit ((int, int) or int): range for changing values for the blue channel. If b_shift_limit is a single
            int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        r_shift_limit=20,
        g_shift_limit=20,
        b_shift_limit=20,
        always_apply=False,
        p=0.5,
    ):
        super(RGBShift, self).__init__(always_apply, p)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(self, image, r_shift=0, g_shift=0, b_shift=0, **params):
        if not is_rgb_image(image):
            raise TypeError("RGBShift transformation expects 3-channel images.")
        return shift_rgb(image, r_shift, g_shift, b_shift)

    def get_params(self):
        return {
            "r_shift": random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
            "g_shift": random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
            "b_shift": random.uniform(self.b_shift_limit[0], self.b_shift_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("r_shift_limit", "g_shift_limit", "b_shift_limit")
