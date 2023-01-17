import math
from functools import wraps

import numpy as np

from augmentoo.core.dtypes import MAX_VALUES_BY_DTYPE, MIN_VALUES_BY_DTYPE

__all__ = [
    "angle_2pi_range",
    "clipped",
    "preserve_shape",
    "preserve_channel_dim",
    "ensure_contiguous",
]


def angle_to_2pi_range(angle: float) -> float:
    two_pi = 2 * math.pi
    return angle % two_pi


def angle_2pi_range(func):
    @wraps(func)
    def wrapped_function(keypoint, *args, **kwargs):
        (x, y, a, s) = func(keypoint, *args, **kwargs)
        return (x, y, angle_to_2pi_range(a), s)

    return wrapped_function


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        minval = MIN_VALUES_BY_DTYPE[dtype]
        maxval = MAX_VALUES_BY_DTYPE[dtype]
        return np.clip(func(img, *args, **kwargs), a_min=minval, a_max=maxval).astype(dtype, copy=False)

    return wrapped_function


def preserve_shape(func):
    """
    Preserve shape of the image

    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


def preserve_channel_dim(func):
    """
    Preserve dummy channel dim.

    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


def ensure_contiguous(func):
    """
    Ensure that input img is contiguous.
    """

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args, **kwargs):
        img = np.require(img, requirements=["C_CONTIGUOUS"])
        result = func(img, *args, **kwargs)
        return result

    return wrapped_function
