__all__ = ["Pipeline"]

from typing import Mapping, Iterable

from augmentoo.core.targets.abstract_target import AbstractTarget
from augmentoo.core.transforms_interface import AbstractTransform


class Pipeline:
    """
    >>> import cv2
    >>> import numpy as np
    >>> import augmentoo as A2
    >>>
    >>> # Simple use-case scenario, input is image, corresponding semantic mask, bounding box and facial keypoints.
    >>> transform = A2.Pipeline(
    >>>     targets=dict(
    >>>         image=A2.ImageTarget(interpolation="bilinear", output_dtype=np.float32),
    >>>         mask=A2.MaskTarget(interpolation="nearest", output_dummy_channel_dim=True),
    >>>         bboxes=A2.AxisAlignedBoxTarget(input_format="xywh", output_format="yolo"),
    >>>         keypoints=A2.KeypointTarget(semantic_labels_field="keypoints_labels",
    >>>           symmetry_group={"hflip": [("left_eye", "right_eye"), ("left_hand", "right_hand")],
    >>>                           "vflip": [("top_xxx"), ("bottom_xxx")]
    >>>                          }
    >>>         )
    >>>     ),
    >>>     transforms=[
    >>>       A2.HorizontalFlip(p=0.5),
    >>>       A2.ShiftScaleRotate(rotate_limit=A2.Uniform(-10, 10),
    >>>          scale_limit=A2.TruncatedNormal(mean=1, std=0.2, lower_bound=0.8, upper_bound=1.2),
    >>>          p=0.5
    >>>     ]
    >>> )
    >>>
    >>> data = transform(image=cv2.imread("lena.jpg"), mask=cv2.imread("lena.png"),
    >>>   bboxes=[],
    >>>   keypoints=[],
    >>>   keypoints_labels=["eye","arm"]
    >>> )
    >>>
    >>> transform = A2.Pipeline(
    >>>     targets=dict(images=A2.BatchTarget(A2.ImageTarget(interpolation="bilinear")), mask=A2.MaskTarget()),
    >>>     transforms=[]
    >>> )
    >>>
    >>> data = transform(image=cv2.imread("lena.jpg"), mask=cv2.imread("lena.png"))
    
    """
    targets: Mapping[str, AbstractTarget]

    def __init__(self, targets: Mapping[str, AbstractTarget], transforms: Iterable[AbstractTransform]):
        self.targets = targets
        self.transforms = transforms

    def __call__(self, **kwargs):
        data = [(k, self.targets[k].preprocess_input(v)) for k, v in kwargs.items() if k in self.targets]
        meta = [(k, v) for k, v in kwargs.items() if k not in self.targets]

        for transform in self.transforms:
            data = transform(data, self.targets)

        data = [(k, self.targets[k].postprocess_result(v)) for k, v in data.items()]
        return {**data, **meta}
