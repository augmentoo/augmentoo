import random
import warnings
from enum import IntEnum

import cv2
import numpy as np

from augmentoo.core.decorators import preserve_shape
from augmentoo.core.transforms_interface import ImageOnlyTransform

__all__ = ["ImageCompression"]


@preserve_shape
def image_compression(img, quality, image_type):
    if image_type in [".jpeg", ".jpg"]:
        quality_flag = cv2.IMWRITE_JPEG_QUALITY
    elif image_type == ".webp":
        quality_flag = cv2.IMWRITE_WEBP_QUALITY
    else:
        raise NotImplementedError("Only '.jpg' and '.webp' compression transforms are implemented. ")

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        warnings.warn(
            "Image compression augmentation "
            "is most effective with uint8 inputs, "
            "{} is used as input.".format(input_dtype),
            UserWarning,
        )
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(f"Unexpected dtype {input_dtype} for image augmentation")

    _, encoded_img = cv2.imencode(image_type, img, (int(quality_flag), quality))
    img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    if needs_float:
        img = to_float(img, max_value=255)
    return img


class ImageCompression(ImageOnlyTransform):
    """Decrease Jpeg, WebP compression of an image.

    Args:
        quality_lower (float): lower bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        quality_upper (float): upper bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        compression_type (ImageCompressionType): should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.
            Default: ImageCompressionType.JPEG

    Targets:
        image

    Image types:
        uint8, float32
    """

    class ImageCompressionType(IntEnum):
        JPEG = 0
        WEBP = 1

    quality_lower: int
    quality_upper: int

    def __init__(
        self,
        quality_lower: int = 99,
        quality_upper: int = 100,
        compression_type=ImageCompressionType.JPEG,
        always_apply=False,
        p=0.5,
    ):
        super(ImageCompression, self).__init__(always_apply, p)

        self.compression_type = ImageCompression.ImageCompressionType(compression_type)
        low_thresh_quality_assert = 0

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            low_thresh_quality_assert = 1

        if not low_thresh_quality_assert <= quality_lower <= 100:
            raise ValueError("Invalid quality_lower. Got: {}".format(quality_lower))
        if not low_thresh_quality_assert <= quality_upper <= 100:
            raise ValueError("Invalid quality_upper. Got: {}".format(quality_upper))

        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def apply(self, image, quality=100, image_type=".jpg", **params):
        if not image.ndim == 2 and image.shape[-1] not in (1, 3, 4):
            raise TypeError("ImageCompression transformation expects 1, 3 or 4 channel images.")
        return image_compression(image, quality, image_type)

    def get_params(self):
        image_type = ".jpg"

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            image_type = ".webp"

        return {
            "quality": random.randint(self.quality_lower, self.quality_upper),
            "image_type": image_type,
        }

    def get_transform_init_args(self):
        return {
            "quality_lower": self.quality_lower,
            "quality_upper": self.quality_upper,
            "compression_type": self.compression_type.value,
        }
