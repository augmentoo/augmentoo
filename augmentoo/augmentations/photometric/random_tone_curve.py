import cv2
import numpy as np

from augmentoo import random_utils, ImageOnlyTransform, preserve_shape
from augmentoo.augmentations import _maybe_process_in_chunks


@preserve_shape
def move_tone_curve(img, low_y, high_y):
    """Rescales the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        img (numpy.ndarray): RGB or grayscale image.
        low_y (float): y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y (float): y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]
    """
    input_dtype = img.dtype

    if low_y < 0 or low_y > 1:
        raise ValueError("low_shift must be in range [0, 1]")
    if high_y < 0 or high_y > 1:
        raise ValueError("high_shift must be in range [0, 1]")

    if input_dtype != np.uint8:
        raise ValueError("Unsupported image type {}".format(input_dtype))

    t = np.linspace(0.0, 1.0, 256)

    # Defines responze of a four-point bezier curve
    def evaluate_bez(t):
        return 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t**2 * high_y + t**3

    evaluate_bez = np.vectorize(evaluate_bez)
    remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)

    lut_fn = _maybe_process_in_chunks(cv2.LUT, lut=remapping)
    img = lut_fn(img)
    return img


class RandomToneCurve(ImageOnlyTransform):
    """Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        scale (float): standard deviation of the normal distribution.
            Used to sample random distances to move two control points that modify the image's curve.
            Values should be in range [0, 1]. Default: 0.1


    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        scale=0.1,
        always_apply=False,
        p=0.5,
    ):
        super(RandomToneCurve, self).__init__(always_apply, p)
        self.scale = scale

    def apply(self, image, low_y, high_y, **params):
        return F.move_tone_curve(image, low_y, high_y)

    def get_params(self):
        return {
            "low_y": np.clip(random_utils.normal(loc=0.25, scale=self.scale), 0, 1),
            "high_y": np.clip(random_utils.normal(loc=0.75, scale=self.scale), 0, 1),
        }

    def get_transform_init_args_names(self):
        return ("scale",)
