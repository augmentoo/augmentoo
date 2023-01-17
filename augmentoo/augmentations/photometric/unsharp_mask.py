import random
from typing import Union, Sequence

import augmentoo.augmentations.blur.gaussian

from augmentoo.core.decorators import clipped, preserve_shape
from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
    to_tuple,
)

__all__ = ["unsharp_mask", "UnsharpMask"]


@clipped
@preserve_shape
def unsharp_mask(
    image: np.ndarray,
    ksize: int,
    sigma: float = 0.0,
    alpha: float = 0.2,
    threshold: int = 10,
):
    blur_fn = _maybe_process_in_chunks(
        augmentoo.augmentations.blur.gaussian.GaussianBlur,
        ksize=(ksize, ksize),
        sigmaX=sigma,
    )

    input_dtype = image.dtype
    if input_dtype == np.uint8:
        image = to_float(image)
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for UnsharpMask augmentation".format(input_dtype))

    blur = blur_fn(image)
    residual = image - blur

    # Do not sharpen noise
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype("float32")

    sharp = image + alpha * residual
    # Avoid color noise artefacts.
    sharp = np.clip(sharp, 0, 1)

    soft_mask = blur_fn(mask)
    output = soft_mask * sharp + (1 - soft_mask) * image
    return from_float(output, dtype=input_dtype)


class UnsharpMask(ImageOnlyTransform):
    """
    Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.

    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        alpha (float, (float, float)): range to choose the visibility of the sharpened image.
            At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
            Default: (0.2, 0.5).
        threshold (int): Value to limit sharpening only for areas with high pixel difference between original image
            and it's smoothed version. Higher threshold means less sharpening on flat areas.
            Must be in range [0, 255]. Default: 10.
        p (float): probability of applying the transform. Default: 0.5.

    Reference:
        arxiv.org/pdf/2107.10833.pdf

    Targets:
        image
    """

    def __init__(
        self,
        blur_limit: Union[int, Sequence[int]] = (3, 7),
        sigma_limit: Union[float, Sequence[float]] = 0.0,
        alpha: Union[float, Sequence[float]] = (0.2, 0.5),
        threshold: int = 10,
        always_apply=False,
        p=0.5,
    ):
        super(UnsharpMask, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.sigma_limit = self.__check_values(to_tuple(sigma_limit, 0.0), name="sigma_limit")
        self.alpha = self.__check_values(to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0))
        self.threshold = threshold

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            raise ValueError("blur_limit and sigma_limit minimum value can not be both equal to 0.")

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("UnsharpMask supports only odd blur limits.")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def get_params(self):
        return {
            "ksize": random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2),
            "sigma": random.uniform(*self.sigma_limit),
            "alpha": random.uniform(*self.alpha),
        }

    def apply(self, img, ksize=3, sigma=0, alpha=0.2, **params):
        return F.unsharp_mask(img, ksize, sigma=sigma, alpha=alpha, threshold=self.threshold)
