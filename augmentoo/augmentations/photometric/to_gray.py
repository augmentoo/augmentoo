from __future__ import division, absolute_import

import warnings


from augmentoo.core.transforms_interface import ImageOnlyTransform


def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class ToGray(ImageOnlyTransform):
    """Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater
    than 127, invert the resulting grayscale image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        if F.is_grayscale_image(img):
            warnings.warn("The image is already gray.")
            return img
        if not F.is_rgb_image(img):
            raise TypeError("ToGray transformation expects 3-channel images.")

        return F.to_gray(img)
