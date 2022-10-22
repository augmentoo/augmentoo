import numpy as np

from augmentoo.core.targets.image import is_rgb_image, is_grayscale_image, is_multispectral_image

__all__ = ["non_rgb_warning"]


def non_rgb_warning(image: np.ndarray) -> None:
    if not is_rgb_image(image):
        message = "This transformation expects 3-channel images"
        if is_grayscale_image(image):
            message += "\nYou can convert your grayscale image to RGB using cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))"
        if is_multispectral_image(image):  # Any image with a number of channels other than 1 and 3
            message += "\nThis transformation cannot be applied to multi-spectral images"

        raise ValueError(message)
