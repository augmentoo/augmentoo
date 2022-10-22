import random

import cv2

from augmentoo.augmentations.geometric.resize import image_scale, keypoint_scale
from augmentoo.core.transforms_interface import DualTransform, to_tuple


class RandomScale(DualTransform):
    """Randomly resize the input. Output image size is different from the input image size.

    Args:
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (1 - scale_limit, 1 + scale_limit). Default: (0.9, 1.1).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, scale_limit=0.1, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        super(RandomScale, self).__init__(always_apply, p)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.interpolation = interpolation

    def get_params(self):
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale=0, interpolation=cv2.INTER_LINEAR, **params):
        return image_scale(img, scale, interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, scale=0, **params):
        return keypoint_scale(keypoint, scale, scale)

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "scale_limit": to_tuple(self.scale_limit, bias=-1.0)}
