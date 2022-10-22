import random
from typing import Union, Sequence, Optional, Tuple

import cv2
import numpy as np
import skimage


from augmentoo.core.decorators import preserve_shape
from augmentoo.core.dtypes import MAX_VALUES_BY_DTYPE
from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
    to_tuple,
)


@preserve_shape
def superpixels(
    image: np.ndarray, n_segments: int, replace_samples: Sequence[bool], max_size: Optional[int], interpolation: int
) -> np.ndarray:
    if not np.any(replace_samples):
        return image

    orig_shape = image.shape
    if max_size is not None:
        size = max(image.shape[:2])
        if size > max_size:
            scale = max_size / size
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            resize_fn = maybe_process_in_chunks(cv2.resize, dsize=(new_width, new_height), interpolation=interpolation)
            image = resize_fn(image)

    segments = skimage.segmentation.slic(image, n_segments=n_segments, compactness=10)

    min_value = 0
    max_value = MAX_VALUES_BY_DTYPE[image.dtype]
    image = np.copy(image)
    if image.ndim == 2:
        image = image.reshape(*image.shape, 1)
    nb_channels = image.shape[2]
    for c in range(nb_channels):
        # segments+1 here because otherwise regionprops always misses the last label
        regions = skimage.measure.regionprops(segments + 1, intensity_image=image[..., c])
        for ridx, region in enumerate(regions):
            # with mod here, because slic can sometimes create more superpixel than requested.
            # replace_samples then does not have enough values, so we just start over with the first one again.
            if replace_samples[ridx % len(replace_samples)]:
                mean_intensity = region.mean_intensity
                image_sp_c = image[..., c]

                if image_sp_c.dtype.kind in ["i", "u", "b"]:
                    # After rounding the value can end up slightly outside of the value_range. Hence, we need to clip.
                    # We do clip via min(max(...)) instead of np.clip because
                    # the latter one does not seem to keep dtypes for dtypes with large itemsizes (e.g. uint64).
                    value: Union[int, float]
                    value = int(np.round(mean_intensity))
                    value = min(max(value, min_value), max_value)
                else:
                    value = mean_intensity

                image_sp_c[segments == ridx] = value

    if orig_shape != image.shape:
        resize_fn = maybe_process_in_chunks(
            cv2.resize, dsize=(orig_shape[1], orig_shape[0]), interpolation=interpolation
        )
        image = resize_fn(image)

    return image


class Superpixels(ImageOnlyTransform):
    """Transform images partially/completely to their superpixel representation.
    This implementation uses skimage's version of the SLIC algorithm.

    Args:
        p_replace (float or tuple of float): Defines for any segment the probability that the pixels within that
            segment are replaced by their average color (otherwise, the pixels are not changed).
            Examples:
                * A probability of ``0.0`` would mean, that the pixels in no
                  segment are replaced by their average color (image is not
                  changed at all).
                * A probability of ``0.5`` would mean, that around half of all
                  segments are replaced by their average color.
                * A probability of ``1.0`` would mean, that all segments are
                  replaced by their average color (resulting in a voronoi
                  image).
            Behaviour based on chosen data types for this parameter:
                * If a ``float``, then that ``flat`` will always be used.
                * If ``tuple`` ``(a, b)``, then a random probability will be
                  sampled from the interval ``[a, b]`` per image.
        n_segments (int, or tuple of int): Rough target number of how many superpixels to generate (the algorithm
            may deviate from this number). Lower value will lead to coarser superpixels.
            Higher values are computationally more intensive and will hence lead to a slowdown
            * If a single ``int``, then that value will always be used as the
              number of segments.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[A2..b]`` will be sampled per image.
        max_size (int or None): Maximum image size at which the augmentation is performed.
            If the width or height of an image exceeds this value, it will be
            downscaled before the augmentation so that the longest side matches `max_size`.
            This is done to speed up the process. The final output image has the same size as the input image.
            Note that in case `p_replace` is below ``1.0``,
            the down-/upscaling will affect the not-replaced pixels too.
            Use ``None`` to apply no down-/upscaling.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
        self,
        p_replace: Union[float, Sequence[float]] = 0.1,
        n_segments: Union[int, Sequence[int]] = 100,
        max_size: Optional[int] = 128,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.p_replace = to_tuple(p_replace, p_replace)
        self.n_segments = to_tuple(n_segments, n_segments)
        self.max_size = max_size
        self.interpolation = interpolation

        if min(self.n_segments) < 1:
            raise ValueError(f"n_segments must be >= 1. Got: {n_segments}")

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return ("p_replace", "n_segments", "max_size", "interpolation")

    def get_params(self) -> dict:
        n_segments = random.randint(*self.n_segments)
        p = random.uniform(*self.p_replace)
        return {"replace_samples": random_utils.random(n_segments) < p, "n_segments": n_segments}

    def apply(self, img: np.ndarray, replace_samples: Sequence[bool] = (False,), n_segments: int = 1, **kwargs):
        return F.superpixels(img, n_segments, replace_samples, self.max_size, self.interpolation)
