from typing import Union

import cv2
import numpy as np

__all__ = [
    "MIN_VALUES_BY_DTYPE",
    "MAX_VALUES_BY_DTYPE",
    "NUMPY_DTYPE_TO_OPENCV_DTYPE",
    "BORDER_MODE_TO_OPENCV",
    "get_opencv_border_mode",
    "get_opencv_dtype_from_numpy",
]

BORDER_MODE_TO_OPENCV = {
    cv2.BORDER_CONSTANT: cv2.BORDER_CONSTANT,
    "constant": cv2.BORDER_CONSTANT,
    "cv2.BORDER_CONSTANT": cv2.BORDER_CONSTANT,
    cv2.BORDER_REFLECT: cv2.BORDER_REFLECT,
    "reflect": cv2.BORDER_REFLECT,
    "cv2.BORDER_REFLECT": cv2.BORDER_REFLECT,
}

MIN_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 0,
    np.dtype("uint16"): 0,
    np.dtype("uint32"): 0,
    np.dtype("float32"): 0.0,
}

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

NUMPY_DTYPE_TO_OPENCV_DTYPE = {
    np.uint8: cv2.CV_8U,
    np.uint16: cv2.CV_16U,
    np.int32: cv2.CV_32S,
    np.float32: cv2.CV_32F,
    np.float64: cv2.CV_64F,
    np.dtype("uint8"): cv2.CV_8U,
    np.dtype("uint16"): cv2.CV_16U,
    np.dtype("int32"): cv2.CV_32S,
    np.dtype("float32"): cv2.CV_32F,
    np.dtype("float64"): cv2.CV_64F,
}


def get_opencv_border_mode(border_mode: Union[str, int]) -> int:
    if border_mode not in BORDER_MODE_TO_OPENCV:
        raise KeyError(f"Border mode f{border_mode} is not supported")
    return BORDER_MODE_TO_OPENCV[border_mode]


def get_opencv_dtype_from_numpy(value: Union[np.ndarray, int, np.dtype, object]) -> int:
    """
    Return a corresponding OpenCV dtype for a numpy's dtype
    :param value: Input dtype of numpy array
    :return: Corresponding dtype for OpenCV
    """
    if isinstance(value, np.ndarray):
        value = value.dtype
    return NUMPY_DTYPE_TO_OPENCV_DTYPE[value]
