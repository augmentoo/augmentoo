from __future__ import division, absolute_import

import cv2
import numpy as np

from augmentoo import random_utils
from augmentoo.augmentations import _maybe_process_in_chunks
from augmentoo.core.decorators import clipped, preserve_shape
from augmentoo.core.dtypes import MAX_VALUES_BY_DTYPE
from augmentoo.core.targets import is_grayscale_image

from augmentoo.core.transforms_interface import ImageOnlyTransform, to_tuple
from augmentoo.targets import clip


@clipped
def _multiply_uint8(img, multiplier):
    img = img.astype(np.float32)
    return np.multiply(img, multiplier)


@preserve_shape
def _multiply_uint8_optimized(img, multiplier):
    if is_grayscale_image(img) or len(multiplier) == 1:
        multiplier = multiplier[0]
        lut = np.arange(0, 256, dtype=np.float32)
        lut *= multiplier
        lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut)
        return func(img)

    channels = img.shape[-1]
    lut = [np.arange(0, 256, dtype=np.float32)] * channels
    lut = np.stack(lut, axis=-1)

    lut *= multiplier
    lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])

    images = []
    for i in range(channels):
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut[:, i])
        images.append(func(img[:, :, i]))
    return np.stack(images, axis=-1)


@clipped
def _multiply_non_uint8(img, multiplier):
    return img * multiplier


def multiply(img, multiplier):
    """
    Args:
        img (numpy.ndarray): Image.
        multiplier (numpy.ndarray): Multiplier coefficient.

    Returns:
        numpy.ndarray: Image multiplied by `multiplier` coefficient.

    """
    if img.dtype == np.uint8:
        if len(multiplier.shape) == 1:
            return _multiply_uint8_optimized(img, multiplier)

        return _multiply_uint8(img, multiplier)

    return _multiply_non_uint8(img, multiplier)


class MultiplicativeNoise(ImageOnlyTransform):
    """Multiply image to random number or array of numbers.

    Args:
        multiplier (float or tuple of floats): If single float image will be multiplied to this number.
            If tuple of float multiplier will be in range `[multiplier[0], multiplier[1])`. Default: (0.9, 1.1).
        per_channel (bool): If `False`, same values for all channels will be used.
            If `True` use sample values for each channels. Default False.
        elementwise (bool): If `False` multiply multiply all pixels in an image with a random value sampled once.
            If `True` Multiply image pixels with values that are pixelwise randomly sampled. Defaule: False.

    Targets:
        image

    Image types:
        Any
    """

    def __init__(
        self,
        multiplier=(0.9, 1.1),
        per_channel=False,
        elementwise=False,
        always_apply=False,
        p=0.5,
    ):
        super(MultiplicativeNoise, self).__init__(always_apply, p)
        self.multiplier = to_tuple(multiplier, multiplier)
        self.per_channel = per_channel
        self.elementwise = elementwise

    def apply(self, img: np.ndarray, multiplier=np.array([1]), **kwargs):
        return multiply(img, multiplier)

    def get_params_dependent_on_targets(self, params):
        if self.multiplier[0] == self.multiplier[1]:
            return {"multiplier": np.array([self.multiplier[0]])}

        img = params["image"]

        h, w = img.shape[:2]

        if self.per_channel:
            c = 1 if is_grayscale_image(img) else img.shape[-1]
        else:
            c = 1

        if self.elementwise:
            shape = [h, w, c]
        else:
            shape = [c]

        multiplier = random_utils.uniform(self.multiplier[0], self.multiplier[1], shape)
        if is_grayscale_image(img) and img.ndim == 2:
            multiplier = np.squeeze(multiplier)

        return {"multiplier": multiplier}

    @property
    def targets_as_params(self):
        return ["image"]
