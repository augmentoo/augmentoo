from __future__ import absolute_import, division

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from augmentoo.core.decorators import preserve_shape
from augmentoo.core.dtypes import MAX_VALUES_BY_DTYPE
from augmentoo.core.targets import is_grayscale_image
from augmentoo.core.transforms_interface import (
    DualTransform,
)

__all__ = ["PixelDropout"]


@preserve_shape
def pixel_dropout(image: np.ndarray, drop_mask: np.ndarray, drop_value: Union[float, Sequence[float]]) -> np.ndarray:
    if isinstance(drop_value, (int, float)) and drop_value == 0:
        drop_values = np.zeros_like(image)
    else:
        drop_values = np.full_like(image, drop_value)  # type: ignore
    return np.where(drop_mask, drop_values, image)


class PixelDropout(DualTransform):
    """Set pixels to 0 with some probability.

    Args:
        dropout_prob (float): pixel drop probability. Default: 0.01
        per_channel (bool): if set to `True` drop mask will be sampled fo each channel,
            otherwise the same mask will be sampled for all channels. Default: False
        drop_value (number or sequence of numbers or None): Value that will be set in dropped place.
            If set to None value will be sampled randomly, default ranges will be used:
                - uint8 - [0, 255]
                - uint16 - [0, 65535]
                - uint32 - [0, 4294967295]
                - float, double - [0, 1]
            Default: 0
        mask_drop_value (number or sequence of numbers or None): Value that will be set in dropped place in masks.
            If set to None masks will be unchanged. Default: 0
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    Image types:
        any
    """

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: Optional[Union[float, Sequence[float]]] = 0,
        mask_drop_value: Optional[Union[float, Sequence[float]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value
        self.mask_drop_value = mask_drop_value

        if self.mask_drop_value is not None and self.per_channel:
            raise ValueError("PixelDropout supports mask only with per_channel=False")

    def apply(
        self, img: np.ndarray, drop_mask: np.ndarray = None, drop_value: Union[float, Sequence[float]] = None, **params
    ) -> np.ndarray:
        return pixel_dropout(img, drop_mask, drop_value)

    def apply_to_mask(self, img: np.ndarray, drop_mask: np.ndarray = np.array([]), **params) -> np.ndarray:
        if self.mask_drop_value is None:
            return img

        if img.ndim == 2:
            drop_mask = np.squeeze(drop_mask)

        return pixel_dropout(img, drop_mask, self.mask_drop_value)

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        shape = img.shape if self.per_channel else img.shape[:2]

        rnd = np.random.RandomState(random.randint(0, 1 << 31))
        # Use choice to create boolean matrix, if we will use binomial after that we will need type conversion
        drop_mask = rnd.choice([True, False], shape, p=[self.dropout_prob, 1 - self.dropout_prob])

        drop_value: Union[float, Sequence[float], np.ndarray]
        if drop_mask.ndim != img.ndim:
            drop_mask = np.expand_dims(drop_mask, -1)
        if self.drop_value is None:
            drop_shape = 1 if is_grayscale_image(img) else int(img.shape[-1])

            if img.dtype in (np.uint8, np.uint16, np.uint32):
                drop_value = rnd.randint(0, int(MAX_VALUES_BY_DTYPE[img.dtype]), drop_shape, img.dtype)
            elif img.dtype in [np.float32, np.double]:
                drop_value = rnd.uniform(0, 1, drop_shape).astype(img.dtpye)
            else:
                raise ValueError(f"Unsupported dtype: {img.dtype}")
        else:
            drop_value = self.drop_value

        return {"drop_mask": drop_mask, "drop_value": drop_value}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return ("dropout_prob", "per_channel", "drop_value", "mask_drop_value")
