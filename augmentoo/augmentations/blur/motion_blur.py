from __future__ import division, absolute_import

import random

import cv2
import numpy as np

from augmentoo.augmentations.blur.box_blur import Blur
from augmentoo.core.targets.image import ImageTarget


class MotionBlur(Blur):
    """Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, img, kernel=None, **params):
        return ImageTarget.convolve(img, kernel=kernel)

    def get_params(self):
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        if ksize <= 2:
            raise ValueError("ksize must be > 2. Got: {}".format(ksize))
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        xs, xe = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if xs == xe:
            ys, ye = random.sample(range(ksize), 2)
        else:
            ys, ye = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return {"kernel": kernel}
