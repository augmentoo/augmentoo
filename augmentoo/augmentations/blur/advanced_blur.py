from __future__ import division, absolute_import

import random
from typing import Union, Sequence

import numpy as np

from augmentoo.core.targets.image import ImageTarget
from augmentoo.core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = ["AdvancedBlur"]


class AdvancedBlur(ImageOnlyTransform):
    """Blur the input image using a Generalized Normal filter with a randomly selected parameters.
        This transform also adds multiplicative noise to generated kernel before convolution.

    Args:
        blur_limit: maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigmaX_limit: Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigmaX_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        sigmaY_limit: Same as `sigmaY_limit` for another dimension.
        rotate_limit: Range from which a random angle used to rotate Gaussian kernel is picked.
            If limit is a single int an angle is picked from (-rotate_limit, rotate_limit). Default: (-90, 90).
        beta_limit: Distribution shape parameter, 1 is the normal distribution. Values below 1.0 make distribution
            tails heavier than normal, values above 1.0 make it lighter than normal. Default: (0.5, 8.0).
        noise_limit: Multiplicative factor that control strength of kernel noise. Must be positive and preferably
            centered around 1.0. If set single value `noise_limit` will be in range (0, noise_limit).
            Default: (0.75, 1.25).
        p (float): probability of applying the transform. Default: 0.5.

    Reference:
        https://arxiv.org/abs/2107.10833

    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: Union[int, Sequence[int]] = (3, 7),
        sigmaX_limit: Union[float, Sequence[float]] = (0.2, 1.0),
        sigmaY_limit: Union[float, Sequence[float]] = (0.2, 1.0),
        rotate_limit: Union[int, Sequence[int]] = 90,
        beta_limit: Union[float, Sequence[float]] = (0.5, 8.0),
        noise_limit: Union[float, Sequence[float]] = (0.9, 1.1),
        always_apply=False,
        p=0.5,
    ):
        super(AdvancedBlur, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.sigmaX_limit = self.__check_values(to_tuple(sigmaX_limit, 0.0), name="sigmaX_limit")
        self.sigmaY_limit = self.__check_values(to_tuple(sigmaY_limit, 0.0), name="sigmaY_limit")
        self.rotate_limit = to_tuple(rotate_limit)
        self.beta_limit = to_tuple(beta_limit, low=0.0)
        self.noise_limit = self.__check_values(to_tuple(noise_limit, 0.0), name="noise_limit")

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("AdvancedBlur supports only odd blur limits.")

        if self.sigmaX_limit[0] == 0 and self.sigmaY_limit[0] == 0:
            raise ValueError("sigmaX_limit and sigmaY_limit minimum value can not be both equal to 0.")

        if not (self.beta_limit[0] < 1.0 < self.beta_limit[1]):
            raise ValueError("Beta limit is expected to include 1.0")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def apply(self, img, kernel=None, **params):
        return ImageTarget.convolve(img, kernel=kernel)

    def get_params(self):
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        sigmaX = random.uniform(*self.sigmaX_limit)
        sigmaY = random.uniform(*self.sigmaY_limit)
        angle = np.deg2rad(random.uniform(*self.rotate_limit))

        # Split into 2 cases to avoid selection of narrow kernels (beta > 1) too often.
        if random.random() < 0.5:
            beta = random.uniform(self.beta_limit[0], 1)
        else:
            beta = random.uniform(1, self.beta_limit[1])

        random_state = np.random.RandomState(random.randint(0, 65536))
        noise_matrix = random_state.uniform(*self.noise_limit, size=[ksize, ksize])

        # Generate mesh grid centered at zero.
        ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
        # Shape (ksize, ksize, 2)
        grid = np.stack(np.meshgrid(ax, ax), axis=-1)

        # Calculate rotated sigma matrix
        d_matrix = np.array([[sigmaX**2, 0], [0, sigmaY**2]])
        u_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        sigma_matrix = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

        inverse_sigma = np.linalg.inv(sigma_matrix)
        # Described in "Parameter Estimation For Multivariate Generalized Gaussian Distributions"
        kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
        # Add noise
        kernel = kernel * noise_matrix

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return {"kernel": kernel}

    def get_transform_init_args_names(self):
        return (
            "blur_limit",
            "sigmaX_limit",
            "sigmaY_limit",
            "rotate_limit",
            "beta_limit",
            "noise_limit",
        )
