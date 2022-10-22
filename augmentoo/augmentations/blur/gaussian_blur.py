from __future__ import division, absolute_import

import random
import warnings

import cv2


from augmentoo.core.decorators import preserve_shape
from augmentoo.core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = ["GaussianBlur"]


@preserve_shape
def gaussian_blur(img, ksize, sigma=0):
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    blur_fn = maybe_process_in_chunks(cv2.GaussianBlur, ksize=(ksize, ksize), sigmaX=sigma)
    return blur_fn(img)


class GaussianBlur(ImageOnlyTransform):
    """Blur the input image using a Gaussian filter with a random kernel size.

    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5):
        super(GaussianBlur, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 0)
        self.sigma_limit = to_tuple(sigma_limit if sigma_limit is not None else 0, 0)

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            warnings.warn(
                "blur_limit and sigma_limit minimum value can not be both equal to 0. "
                "blur_limit minimum value changed to 3."
            )

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("GaussianBlur supports only odd blur limits.")

    def apply(self, image, ksize=3, sigma=0, **params):
        return F.gaussian_blur(image, ksize, sigma=sigma)

    def get_params(self):
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1)
        if ksize != 0 and ksize % 2 != 1:
            ksize = (ksize + 1) % (self.blur_limit[1] + 1)

        return {"ksize": ksize, "sigma": random.uniform(*self.sigma_limit)}

    def get_transform_init_args_names(self):
        return ("blur_limit", "sigma_limit")
