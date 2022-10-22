import cv2
import numpy as np


from augmentoo.core.decorators import preserve_channel_dim
from augmentoo.core.targets import is_grayscale_image, is_rgb_image
from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
)

__all__ = ["Equalize"]


def _equalize_pil(img, mask=None):
    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    h = [_f for _f in histogram if _f]

    if len(h) <= 1:
        return img.copy()

    step = np.sum(h[:-1]) // 255
    if not step:
        return img.copy()

    lut = np.empty(256, dtype=np.uint8)
    n = step // 2
    for i in range(256):
        lut[i] = min(n // step, 255)
        n += histogram[i]

    return cv2.LUT(img, np.array(lut))


def _equalize_cv(img, mask=None):
    if mask is None:
        return cv2.equalizeHist(img)

    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    i = 0
    for val in histogram:
        if val > 0:
            break
        i += 1
    i = min(i, 255)

    total = np.sum(histogram)
    if histogram[i] == total:
        return np.full_like(img, i)

    scale = 255.0 / (total - histogram[i])
    _sum = 0

    lut = np.zeros(256, dtype=np.uint8)
    i += 1
    for i in range(i, len(histogram)):
        _sum += histogram[i]
        lut[i] = clip(round(_sum * scale), np.dtype("uint8"), 255)

    return cv2.LUT(img, lut)


@preserve_channel_dim
def equalize(img: np.ndarray, mask=None, mode="cv", by_channels=True):
    """Equalize the image histogram.

    Args:
        img (numpy.ndarray): RGB or grayscale image.
        mask (numpy.ndarray): An optional mask.  If given, only the pixels selected by
            the mask are included in the analysis. Maybe 1 channel or 3 channel array.
        mode (str): {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.

    Returns:
        numpy.ndarray: Equalized image.

    """
    if img.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")

    modes = ["cv", "pil"]

    if mode not in modes:
        raise ValueError("Unsupported equalization mode. Supports: {}. " "Got: {}".format(modes, mode))
    if mask is not None:
        if is_rgb_image(mask) and is_grayscale_image(img):
            raise ValueError("Wrong mask shape. Image shape: {}. " "Mask shape: {}".format(img.shape, mask.shape))
        if not by_channels and not is_grayscale_image(mask):
            raise ValueError(
                "When by_channels=False only 1-channel mask supports. " "Mask shape: {}".format(mask.shape)
            )

    if mode == "pil":
        function = _equalize_pil
    else:
        function = _equalize_cv

    if mask is not None:
        mask = mask.astype(np.uint8)

    if is_grayscale_image(img):
        return function(img, mask)

    if not by_channels:
        result_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        result_img[..., 0] = function(result_img[..., 0], mask)
        return cv2.cvtColor(result_img, cv2.COLOR_YCrCb2RGB)

    result_img = np.empty_like(img)
    for i in range(3):
        if mask is None:
            _mask = None
        elif is_grayscale_image(mask):
            _mask = mask
        else:
            _mask = mask[..., i]

        result_img[..., i] = function(img[..., i], _mask)

    return result_img


class Equalize(ImageOnlyTransform):
    """Equalize the image histogram.

    Args:
        mode (str): {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.
        mask (np.ndarray, callable): If given, only the pixels selected by
            the mask are included in the analysis. Maybe 1 channel or 3 channel array or callable.
            Function signature must include `image` argument.
        mask_params (list of str): Params for mask function.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        mode="cv",
        by_channels=True,
        mask=None,
        mask_params=(),
        always_apply=False,
        p=0.5,
    ):
        modes = ["cv", "pil"]
        if mode not in modes:
            raise ValueError("Unsupported equalization mode. Supports: {}. " "Got: {}".format(modes, mode))

        super(Equalize, self).__init__(always_apply, p)
        self.mode = mode
        self.by_channels = by_channels
        self.mask = mask
        self.mask_params = mask_params

    def apply(self, image, mask=None, **params):
        return F.equalize(image, mode=self.mode, by_channels=self.by_channels, mask=mask)

    def get_params_dependent_on_targets(self, params):
        if not callable(self.mask):
            return {"mask": self.mask}

        return {"mask": self.mask(**params)}

    @property
    def targets_as_params(self):
        return ["image"] + list(self.mask_params)

    def get_transform_init_args_names(self):
        return ("mode", "by_channels")
