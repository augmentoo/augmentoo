from __future__ import division, absolute_import

import random

import cv2
import numpy as np

from augmentoo.core.decorators import preserve_shape
from augmentoo.core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = ["Blur"]


@preserve_shape
def blur(img, ksize):
    blur_fn = maybe_process_in_chunks(cv2.blur, ksize=(ksize, ksize))
    return blur_fn(img)


class Blur(ImageOnlyTransform):
    """Blur the input image using a random-sized kernel.

    Args:
        blur_limit (int, (int, int)): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=7, always_apply=False, p=0.5):
        super(Blur, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, image, ksize=3, **params):
        return blur(image, ksize)

    def get_params(self):
        return {"ksize": int(random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2)))}
