import random

import cv2
import numpy as np


from augmentoo.core.decorators import preserve_channel_dim
from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
    to_tuple,
)

__all__ = ["CLAHE"]


@preserve_channel_dim
def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img


class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5):
        super(CLAHE, self).__init__(always_apply, p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = tuple(tile_grid_size)

    def apply(self, img, clip_limit=2, **params):
        if not F.is_rgb_image(img) and not F.is_grayscale_image(img):
            raise TypeError("CLAHE transformation expects 1-channel or 3-channel images.")

        return F.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")
