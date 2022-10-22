import numbers
from dataclasses import dataclass
from functools import wraps
from typing import Tuple, Optional

import cv2
import numpy as np

__all__ = [
    "ImageTarget",
    "read_bgr_image",
    "read_rgb_image",
]

from augmentoo.core.dtypes import MIN_VALUES_BY_DTYPE, MAX_VALUES_BY_DTYPE
from augmentoo.core.decorators import preserve_channel_dim, preserve_shape, clipped
from augmentoo.core.targets.abstract_target import AbstractTarget


def read_bgr_image(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def read_rgb_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclass
class ImageTarget(AbstractTarget):
    """
    Represents a 2D-raster image target.
    Support images of shape [rows, cols] or [rows, cols, channels]

    """

    # Specifies dtype of the input image. If the input dtype is set, will raise an exception if actual dtype.
    # does not match the expected dtype
    input_dtype: Optional[np.dtype] = None

    intermediate_dtype: Optional[np.ndtype] = None

    # Specifies dtype of the output image. If set, will saturate cast the result image to the desired dtype.
    output_dtype: Optional[np.ndtype] = None
    output_ensure_contiguous: bool = False
    output_dummy_channel_dim: bool = True

    data_range: Tuple[numbers.Number, numbers.Number] = None
    channel_order: str = "HWC"

    pad_value: int = 0
    pad_mode: int = cv2.BORDER_CONSTANT

    interpolation = cv2.INTER_LINEAR

    def validate_input(self, img: np.ndarray):
        if self.input_dtype is not None and self.input_dtype != img.dtype:
            raise RuntimeError(
                f"Input image has wrong dtype. Expected image of {self.input_dtype} type, got image of {img.dtype} type."
            )

    def preprocess_input(self, img: np.ndarray) -> np.ndarray:
        self.validate_input(img)

        if self.input_dtype is not None and self.input_dtype != img.dtype:
            raise RuntimeError(
                f"Input image has wrong dtype. Expected image of {self.input_dtype} type, got image of {img.dtype} type."
            )

        return img

    def postprocess_result(self, img: np.ndarray) -> np.ndarray:
        if self.output_dtype is not None and self.output_dtype != img.dtype:
            img = self.clip(
                img,
                self.output_dtype,
                minval=MIN_VALUES_BY_DTYPE[self.output_dtype],
                maxval=MAX_VALUES_BY_DTYPE[self.output_dtype],
            )
        if self.output_ensure_contiguous:
            img = np.ascontiguousarray(img)

        if self.output_dummy_channel_dim and len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        return img

    @classmethod
    def clip(cls, img: np.ndarray, dtype, minval, maxval) -> np.ndarray:
        return np.clip(img, minval, maxval).astype(dtype, copy=False)

    @classmethod
    @preserve_channel_dim
    def resize(cls, img: np.ndarray, height: int, width: int, interpolation=cv2.INTER_LINEAR):
        img_height, img_width = img.shape[:2]
        if height == img_height and width == img_width:
            return img
        resize_fn = cls.maybe_process_in_chunks(cv2.resize, dsize=(width, height), interpolation=interpolation)
        return resize_fn(img)

    @classmethod
    @preserve_shape
    def convolve(cls, img: np.ndarray, kernel: np.ndarray, border_type=cv2.BORDER_CONSTANT) -> np.ndarray:
        conv_fn = cls.maybe_process_in_chunks(cv2.filter2D, ddepth=-1, kernel=kernel, borderType=border_type)
        return conv_fn(img)

    @classmethod
    @clipped
    @preserve_shape
    def apply_linear(cls, img: np.ndarray, transformation_matrix: np.ndarray):
        """
        Apply linear transform to a given image individually for each channel

        Args:
            img: Input image of [H,W,C] shape
            transformation_matrix:  [C,C] linear transformation matrix

        Returns:

        """

        result_img = cv2.transform(img, transformation_matrix)
        return result_img

    @classmethod
    def is_rgb_image(cls, image: np.ndarray) -> bool:
        return len(image.shape) == 3 and image.shape[-1] == 3

    @classmethod
    def is_grayscale_image(cls, image: np.ndarray) -> bool:
        return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)

    @classmethod
    def is_multispectral_image(cls, image: np.ndarray) -> bool:
        return len(image.shape) == 3 and image.shape[-1] not in [1, 3]

    @classmethod
    def get_num_channels(cls, image: np.ndarray) -> int:
        return image.shape[2] if len(image.shape) == 3 else 1

    @classmethod
    def get_shape(cls, img: np.ndarray) -> Tuple[int, int]:
        rows, cols = img.shape[:2]
        return rows, cols

    @classmethod
    def maybe_process_in_chunks(cls, process_fn, **kwargs):
        """
        Wrap OpenCV function to enable processing images with more than 4 channels.

        Limitations:
            This wrapper requires image to be the first argument and rest must be sent via named arguments.

        Args:
            process_fn: Transform function (e.g cv2.resize).
            kwargs: Additional parameters.

        Returns:
            numpy.ndarray: Transformed image.

        """

        @wraps(process_fn)
        def __process_fn(img):
            num_channels = ImageTarget.get_num_channels(img)
            if num_channels > 4:
                chunks = []
                for index in range(0, num_channels, 4):
                    if num_channels - index == 2:
                        # Many OpenCV functions cannot work with 2-channel images
                        for i in range(2):
                            chunk = img[:, :, index + i : index + i + 1]
                            chunk = process_fn(chunk, **kwargs)
                            chunk = np.expand_dims(chunk, -1)
                            chunks.append(chunk)
                    else:
                        chunk = img[:, :, index : index + 4]
                        chunk = process_fn(chunk, **kwargs)
                        chunks.append(chunk)
                img = np.dstack(chunks)
            else:
                img = process_fn(img, **kwargs)
            return img

        return __process_fn
