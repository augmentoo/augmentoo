from dataclasses import dataclass

import cv2

from augmentoo.core.targets.image import ImageTarget

__all__ = ["MaskTarget"]


@dataclass
class MaskTarget(ImageTarget):
    interpolation = cv2.INTER_NEAREST
