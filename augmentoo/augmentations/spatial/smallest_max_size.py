from augmentoo import image_resize
from augmentoo.augmentations.geometric.functional import py3round
from augmentoo.core.decorators import preserve_channel_dim
from augmentoo.core.transforms_interface import DualTransform

__all__ = ["SmallestMaxSize"]


@preserve_channel_dim
def smallest_max_size(img, max_size, interpolation):
    height, width = img.shape[:2]
    scale = max_size / float(min(width, height))

    if scale != 1.0:
        new_height, new_width = tuple(py3round(dim * scale) for dim in (height, width))
        img = image_resize(img, height=new_height, width=new_width, interpolation=interpolation)
    return img


class SmallestMaxSize(DualTransform):
    """Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, list of int): maximum size of smallest side of the image after the transformation. When using a
            list, max size will be randomly selected from the values in the list.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        max_size: Union[int, Sequence[int]] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(SmallestMaxSize, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(
        self, img: np.ndarray, max_size: int = 1024, interpolation: int = cv2.INTER_LINEAR, **params
    ) -> np.ndarray:
        return smallest_max_size(img, max_size=max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: Sequence[float], **params) -> Sequence[float]:
        return bbox

    def apply_to_keypoint(self, keypoint: Sequence[float], max_size: int = 1024, **params) -> Sequence[float]:
        height = params["rows"]
        width = params["cols"]

        scale = max_size / min([height, width])
        return F.keypoint_scale(keypoint, scale, scale)

    def get_params(self) -> Dict[str, int]:
        return {"max_size": self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size)}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_size", "interpolation")
