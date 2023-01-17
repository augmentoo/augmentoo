from __future__ import division, absolute_import


from augmentoo.core.transforms_interface import ImageOnlyTransform


def invert(img):
    return 255 - img


class InvertImg(ImageOnlyTransform):
    """Invert the input image by subtracting pixel values from 255.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def apply(self, img, **params):
        return invert(img)
