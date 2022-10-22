import cv2

from augmentoo.augmentations.crops import functional as F
from augmentoo.augmentations.spatial.resize import keypoint_scale
from augmentoo.core.transforms_interface import DualTransform
from augmentoo.targets import ImageTarget


class _BaseRandomSizedCrop(DualTransform):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(_BaseRandomSizedCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return ImageTarget.resize(crop, self.height, self.width, interpolation)

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    def apply_to_keypoint(self, keypoint, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        keypoint = F.keypoint_random_crop(keypoint, crop_height, crop_width, h_start, w_start, rows, cols)
        scale_x = self.width / crop_width
        scale_y = self.height / crop_height
        keypoint = keypoint_scale(keypoint, scale_x, scale_y)
        return keypoint
