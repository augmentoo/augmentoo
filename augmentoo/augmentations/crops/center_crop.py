from augmentoo.augmentations.crops import functional as F
from augmentoo.core.transforms_interface import DualTransform

__all__ = ["CenterCrop"]


class CenterCrop(DualTransform):
    """Crop the central part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        It is recommended to use uint8 images as input.
        Otherwise the operation will require internal conversion
        float32 -> uint8 -> float32 that causes worse performance.
    """

    def __init__(self, height, width, always_apply=False, p=1.0):
        super(CenterCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        return F.center_crop(img, self.height, self.width)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_center_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_center_crop(keypoint, self.height, self.width, **params)

    def get_transform_init_args_names(self):
        return ("height", "width")
