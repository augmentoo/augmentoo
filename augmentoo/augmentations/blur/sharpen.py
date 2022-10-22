from __future__ import division, absolute_import

import random

import numpy as np

from augmentoo.core.targets.image import ImageTarget
from augmentoo.core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = ["Sharpen"]


class Sharpen(ImageOnlyTransform):
    """Sharpen the input image and overlays the result with the original image.

    Args:
        alpha ((float, float)): range to choose the visibility of the sharpened image. At 0, only the original image is
            visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
        lightness ((float, float)): range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5):
        super(Sharpen, self).__init__(always_apply, p)
        self.alpha = self.__check_values(to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0))
        self.lightness = self.__check_values(to_tuple(lightness, 0.0), name="lightness")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError("{} values should be between {}".format(name, bounds))
        return value

    @staticmethod
    def __generate_sharpening_matrix(alpha_sample, lightness_sample):
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [[-1, -1, -1], [-1, 8 + lightness_sample, -1], [-1, -1, -1]],
            dtype=np.float32,
        )

        matrix = (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return matrix

    def get_params(self):
        alpha = random.uniform(*self.alpha)
        lightness = random.uniform(*self.lightness)
        sharpening_matrix = self.__generate_sharpening_matrix(alpha_sample=alpha, lightness_sample=lightness)
        return {"sharpening_matrix": sharpening_matrix}

    def apply(self, img, sharpening_matrix=None, **params):
        return ImageTarget.convolve(img, sharpening_matrix)

    def get_transform_init_args_names(self):
        return ("alpha", "lightness")
