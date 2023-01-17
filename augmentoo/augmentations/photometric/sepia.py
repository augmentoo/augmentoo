import numpy as np


from augmentoo.core.targets import is_rgb_image
from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
)

__all__ = ["ToSepia"]


class ToSepia(ImageOnlyTransform):
    """Applies sepia filter to the input RGB image

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=0.5):
        super(ToSepia, self).__init__(always_apply, p)
        self.sepia_transformation_matrix = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        )

    def apply(self, image, **params):
        if not is_rgb_image(image):
            raise RuntimeError("ToSepia transformation expects 3-channel images.")
        return F.linear_transformation_rgb(image, self.sepia_transformation_matrix)
