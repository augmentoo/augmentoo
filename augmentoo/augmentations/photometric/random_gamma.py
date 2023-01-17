from __future__ import absolute_import, division

import random

import cv2
import numpy as np


from augmentoo.core.decorators import preserve_shape
from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
    to_tuple,
)

__all__ = ["RandomGamma"]


@preserve_shape
def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    else:
        img = np.power(img, gamma)

    return img


class RandomGamma(ImageOnlyTransform):
    """
    Args:
        gamma_limit (float or (float, float)): If gamma_limit is a single float value,
            the range will be (-gamma_limit, gamma_limit). Default: (80, 120).
        eps: Deprecated.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5):
        super(RandomGamma, self).__init__(always_apply, p)
        self.gamma_limit = to_tuple(gamma_limit)
        self.eps = eps

    def apply(self, img, gamma=1, **params):
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self):
        return {"gamma": random.randint(self.gamma_limit[0], self.gamma_limit[1]) / 100.0}
