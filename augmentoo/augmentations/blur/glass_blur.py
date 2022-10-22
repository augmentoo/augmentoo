from __future__ import division, absolute_import

import itertools

import cv2
import numpy as np

from augmentoo import random_utils
from augmentoo.augmentations.blur.box_blur import Blur

from augmentoo.core.decorators import preserve_shape

__all__ = ["GlassBlur"]


@preserve_shape
def glass_blur(img, sigma, max_delta, iterations, dxy, mode):
    x = cv2.GaussianBlur(np.array(img), sigmaX=sigma, ksize=(0, 0))

    if mode == "fast":
        hs = np.arange(img.shape[0] - max_delta, max_delta, -1)
        ws = np.arange(img.shape[1] - max_delta, max_delta, -1)
        h = np.tile(hs, ws.shape[0])
        w = np.repeat(ws, hs.shape[0])

        for i in range(iterations):
            dy = dxy[:, i, 0]
            dx = dxy[:, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]

    elif mode == "exact":
        for ind, (i, h, w) in enumerate(
            itertools.product(
                range(iterations),
                range(img.shape[0] - max_delta, max_delta, -1),
                range(img.shape[1] - max_delta, max_delta, -1),
            )
        ):
            ind = ind if ind < len(dxy) else ind % len(dxy)
            dy = dxy[ind, i, 0]
            dx = dxy[ind, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]

    return cv2.GaussianBlur(x, sigmaX=sigma, ksize=(0, 0))


class GlassBlur(Blur):
    """Apply glass noise to the input image.

    Args:
        sigma (float): standard deviation for Gaussian kernel.
        max_delta (int): max distance between pixels which are swapped.
        iterations (int): number of repeats.
            Should be in range [1, inf). Default: (2).
        mode (str): mode of computation: fast or exact. Default: "fast".
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1903.12261
    |  https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
    """

    def __init__(
        self,
        sigma=0.7,
        max_delta=4,
        iterations=2,
        always_apply=False,
        mode="fast",
        p=0.5,
    ):
        super(GlassBlur, self).__init__(always_apply=always_apply, p=p)
        if iterations < 1:
            raise ValueError("Iterations should be more or equal to 1, but we got {}".format(iterations))

        if mode not in ["fast", "exact"]:
            raise ValueError("Mode should be 'fast' or 'exact', but we got {}".format(iterations))

        self.sigma = sigma
        self.max_delta = max_delta
        self.iterations = iterations
        self.mode = mode

    def apply(self, img, dxy=0, **params):
        return glass_blur(img, self.sigma, self.max_delta, self.iterations, dxy, self.mode)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]

        # generate array containing all necessary values for transformations
        width_pixels = img.shape[0] - self.max_delta * 2
        height_pixels = img.shape[1] - self.max_delta * 2
        total_pixels = width_pixels * height_pixels
        dxy = random_utils.randint(-self.max_delta, self.max_delta, size=(total_pixels, self.iterations, 2))

        return {"dxy": dxy}

    def get_transform_init_args_names(self):
        return ("sigma", "max_delta", "iterations")

    @property
    def targets_as_params(self):
        return ["image"]
