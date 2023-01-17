import random

import cv2

from augmentoo.augmentations.crops.base_random_crop import _BaseRandomSizedCrop

__all__ = ["RandomSizedCrop"]


class RandomSizedCrop(_BaseRandomSizedCrop):
    """Crop a random part of the input and rescale it to some size.

    Args:
        min_max_height ((int, int)): crop size limits.
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        w2h_ratio (float): aspect ratio of crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        min_max_height,
        height,
        width,
        w2h_ratio=1.0,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):
        super(RandomSizedCrop, self).__init__(
            height=height,
            width=width,
            interpolation=interpolation,
            always_apply=always_apply,
            p=p,
        )
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def get_params(self):
        crop_height = random.randint(self.min_max_height[0], self.min_max_height[1])
        return {
            "h_start": random.random(),
            "w_start": random.random(),
            "crop_height": crop_height,
            "crop_width": int(crop_height * self.w2h_ratio),
        }
