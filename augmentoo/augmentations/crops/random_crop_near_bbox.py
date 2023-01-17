import random
from typing import Union, Tuple, Dict, Any, List

import numpy as np

from augmentoo.augmentations.crops import functional as F
from augmentoo.core.transforms_interface import DualTransform, to_tuple

__all__ = ["RandomCropNearBBox"]


class RandomCropNearBBox(DualTransform):
    """Crop bbox from image with random shift by x,y coordinates

    Args:
        max_part_shift (float, (float, float)): Max shift in `height` and `width` dimensions relative
            to `cropping_bbox` dimension.
            If max_part_shift is a single float, the range will be (max_part_shift, max_part_shift).
            Default (0.3, 0.3).
        cropping_box_key (str): Additional target key for cropping box. Default `cropping_bbox`
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Examples:
        >>> aug = Compose([RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_box_key='test_box')],
        >>>              bbox_params=BboxParams("pascal_voc"))
        >>> result = aug(image=image, bboxes=bboxes, test_box=[0, 5, 10, 20])

    """

    def __init__(
        self,
        max_part_shift: Union[float, Tuple[float, float]] = (0.3, 0.3),
        cropping_box_key: str = "cropping_bbox",
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(RandomCropNearBBox, self).__init__(always_apply, p)
        self.max_part_shift = to_tuple(max_part_shift, low=max_part_shift)
        self.cropping_bbox_key = cropping_box_key

        if min(self.max_part_shift) < 0 or max(self.max_part_shift) > 1:
            raise ValueError("Invalid max_part_shift. Got: {}".format(max_part_shift))

    def apply(
        self,
        img: np.ndarray,
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        **params,
    ) -> np.ndarray:
        return F.clamping_crop(img, x_min, y_min, x_max, y_max)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, int]:
        bbox = params[self.cropping_bbox_key]
        h_max_shift = round((bbox[3] - bbox[1]) * self.max_part_shift[0])
        w_max_shift = round((bbox[2] - bbox[0]) * self.max_part_shift[1])

        x_min = bbox[0] - random.randint(-w_max_shift, w_max_shift)
        x_max = bbox[2] + random.randint(-w_max_shift, w_max_shift)

        y_min = bbox[1] - random.randint(-h_max_shift, h_max_shift)
        y_max = bbox[3] + random.randint(-h_max_shift, h_max_shift)

        x_min = max(0, x_min)
        y_min = max(0, y_min)

        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def apply_to_bbox(self, bbox: Tuple[float, float, float, float], **params) -> Tuple[float, float, float, float]:
        return F.bbox_crop(bbox, **params)

    def apply_to_keypoint(
        self,
        keypoint: Tuple[float, float, float, float],
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        **params,
    ) -> Tuple[float, float, float, float]:
        return F.crop_keypoint_by_coords(keypoint, crop_coords=(x_min, y_min, x_max, y_max))

    @property
    def targets_as_params(self) -> List[str]:
        return [self.cropping_bbox_key]
