from __future__ import division, absolute_import

import random
from typing import Union, Sequence

import numpy as np
from scipy import special

from augmentoo.core.targets.image import ImageTarget
from augmentoo.core.transforms_interface import ImageOnlyTransform, to_tuple


class RingingOvershoot(ImageOnlyTransform):
    """Create ringing or overshoot artefacts by conlvolving image with 2D sinc filter.

    Args:
        blur_limit (int, (int, int)): maximum kernel size for sinc filter.
            Should be in range [3, inf). Default: (7, 15).
        cutoff (float, (float, float)): range to choose the cutoff frequency in radians.
            Should be in range (0, np.pi)
            Default: (np.pi / 4, np.pi / 2).
        p (float): probability of applying the transform. Default: 0.5.

    Reference:
        dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
        https://arxiv.org/abs/2107.10833

    Targets:
        image
    """

    def __init__(
        self,
        blur_limit: Union[int, Sequence[int]] = (7, 15),
        cutoff: Union[float, Sequence[float]] = (np.pi / 4, np.pi / 2),
        always_apply=False,
        p=0.5,
    ):
        super(RingingOvershoot, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.cutoff = self.__check_values(to_tuple(cutoff, np.pi / 2), name="cutoff", bounds=(0, np.pi))

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def get_params(self):
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        if ksize % 2 == 0:
            raise ValueError(f"Kernel size must be odd. Got: {ksize}")

        cutoff = random.uniform(*self.cutoff)

        # From dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
        kernel = np.fromfunction(
            lambda x, y: cutoff
            * special.j1(cutoff * np.sqrt((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2))
            / (2 * np.pi * np.sqrt((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2)),
            [ksize, ksize],
        )
        kernel[(ksize - 1) // 2, (ksize - 1) // 2] = cutoff**2 / (4 * np.pi)

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)

        return {"kernel": kernel}

    def apply(self, img, kernel=None, **params):
        return ImageTarget.convolve(img, kernel)
