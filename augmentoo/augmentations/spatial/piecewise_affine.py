import random
from typing import Sequence, Union, Tuple

import numpy as np
import skimage.transform

from augmentoo import random_utils
from augmentoo.augmentations.geometric.functional import (
    to_distance_maps,
    from_distance_maps,
)
from augmentoo.core.decorators import clipped
from augmentoo.core.transforms_interface import DualTransform, to_tuple

__all__ = ["PiecewiseAffine"]


@clipped
def piecewise_affine(
    img: np.ndarray,
    matrix: skimage.transform.PiecewiseAffineTransform,
    interpolation: int,
    mode: str,
    cval: float,
) -> np.ndarray:
    return skimage.transform.warp(
        img,
        matrix,
        order=interpolation,
        mode=mode,
        cval=cval,
        preserve_range=True,
        output_shape=img.shape,
    )


def keypoint_piecewise_affine(
    keypoint: Sequence[float],
    matrix: skimage.transform.PiecewiseAffineTransform,
    h: int,
    w: int,
    keypoints_threshold: float,
) -> Tuple[float, float, float, float]:
    x, y, a, s = keypoint
    dist_maps = to_distance_maps([(x, y)], h, w, True)
    dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)
    x, y = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)[0]
    return x, y, a, s


def bbox_piecewise_affine(
    bbox: Sequence[float],
    matrix: skimage.transform.PiecewiseAffineTransform,
    h: int,
    w: int,
    keypoints_threshold: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = denormalize_bbox(tuple(bbox), h, w)
    keypoints = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]
    dist_maps = to_distance_maps(keypoints, h, w, True)
    dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)
    keypoints = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)
    keypoints = [i for i in keypoints if 0 <= i[0] < w and 0 <= i[1] < h]
    keypoints_arr = np.array(keypoints)
    x1 = keypoints_arr[:, 0].min()
    y1 = keypoints_arr[:, 1].min()
    x2 = keypoints_arr[:, 0].max()
    y2 = keypoints_arr[:, 1].max()
    return normalize_bbox((x1, y1, x2, y2), h, w)


class PiecewiseAffine(DualTransform):
    """Apply affine transformations that differ between local neighbourhoods.
    This augmentation places a regular grid of points on an image and randomly moves the neighbourhood of these point
    around via affine transformations. This leads to local distortions.

    This is mostly a wrapper around scikit-image's ``PiecewiseAffine``.
    See also ``Affine`` for a similar technique.

    Note:
        This augmenter is very slow. Try to use ``ElasticTransformation`` instead, which is at least 10x faster.

    Note:
        For coordinate-based inputs (keypoints, bounding boxes, polygons, ...),
        this augmenter still has to perform an image-based augmentation,
        which will make it significantly slower and not fully correct for such inputs than other transforms.

    Args:
        scale (float, tuple of float): Each point on the regular grid is moved around via a normal distribution.
            This scale factor is equivalent to the normal distribution's sigmA2.
            Note that the jitter (how far each point is moved in which direction) is multiplied by the height/width of
            the image if ``absolute_scale=False`` (default), so this scale can be the same for different sized images.
            Recommended values are in the range ``0.01`` to ``0.05`` (weak to strong augmentations).
                * If a single ``float``, then that value will always be used as the scale.
                * If a tuple ``(a, b)`` of ``float`` s, then a random value will
                  be uniformly sampled per image from the interval ``[a, b]``.
        nb_rows (int, tuple of int): Number of rows of points that the regular grid should have.
            Must be at least ``2``. For large images, you might want to pick a higher value than ``4``.
            You might have to then adjust scale to lower values.
                * If a single ``int``, then that value will always be used as the number of rows.
                * If a tuple ``(a, b)``, then a value from the discrete interval
                  ``[A2..b]`` will be uniformly sampled per image.
        nb_cols (int, tuple of int): Number of columns. Analogous to `nb_rows`.
        interpolation (int): The order of interpolation. The order has to be in the range 0-5:
             - 0: Nearest-neighbor
             - 1: Bi-linear (default)
             - 2: Bi-quadratic
             - 3: Bi-cubic
             - 4: Bi-quartic
             - 5: Bi-quintic
        mask_interpolation (int): same as interpolation but for mask.
        cval (number): The constant value to use when filling in newly created pixels.
        cval_mask (number): Same as cval but only for masks.
        mode (str): {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
            Points outside the boundaries of the input are filled according
            to the given mode.  Modes match the behaviour of `numpy.pad`.
        absolute_scale (bool): Take `scale` as an absolute value rather than a relative value.
        keypoints_threshold (float): Used as threshold in conversion from distance maps to keypoints.
            The search for keypoints works by searching for the
            argmin (non-inverted) or argmax (inverted) in each channel. This
            parameters contains the maximum (non-inverted) or minimum (inverted) value to accept in order to view a hit
            as a keypoint. Use ``None`` to use no min/max. Default: 0.01

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        scale: Union[float, Sequence[float]] = (0.03, 0.05),
        nb_rows: Union[int, Sequence[int]] = 4,
        nb_cols: Union[int, Sequence[int]] = 4,
        interpolation: int = 1,
        mask_interpolation: int = 0,
        cval: int = 0,
        cval_mask: int = 0,
        mode: str = "constant",
        absolute_scale: bool = False,
        always_apply: bool = False,
        keypoints_threshold: float = 0.01,
        p: float = 0.5,
    ):
        super(PiecewiseAffine, self).__init__(always_apply, p)

        self.scale = to_tuple(scale, scale)
        self.nb_rows = to_tuple(nb_rows, nb_rows)
        self.nb_cols = to_tuple(nb_cols, nb_cols)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.absolute_scale = absolute_scale
        self.keypoints_threshold = keypoints_threshold

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params) -> dict:
        h, w = params["image"].shape[:2]

        nb_rows = np.clip(random.randint(*self.nb_rows), 2, None)
        nb_cols = np.clip(random.randint(*self.nb_cols), 2, None)
        nb_cells = nb_cols * nb_rows
        scale = random.uniform(*self.scale)

        jitter: np.ndarray = random_utils.normal(0, scale, (nb_cells, 2))
        if not np.any(jitter > 0):
            return {"matrix": None}

        y = np.linspace(0, h, nb_rows)
        x = np.linspace(0, w, nb_cols)

        # (H, W) and (H, W) for H=rows, W=cols
        xx_src, yy_src = np.meshgrid(x, y)

        # (1, HW, 2) => (HW, 2) for H=rows, W=cols
        points_src = np.dstack([yy_src.flat, xx_src.flat])[0]

        if self.absolute_scale:
            jitter[:, 0] = jitter[:, 0] / h if h > 0 else 0.0
            jitter[:, 1] = jitter[:, 1] / w if w > 0 else 0.0

        jitter[:, 0] = jitter[:, 0] * h
        jitter[:, 1] = jitter[:, 1] * w

        points_dest = np.copy(points_src)
        points_dest[:, 0] = points_dest[:, 0] + jitter[:, 0]
        points_dest[:, 1] = points_dest[:, 1] + jitter[:, 1]

        # Restrict all destination points to be inside the image plane.
        # This is necessary, as otherwise keypoints could be augmented
        # outside of the image plane and these would be replaced by
        # (-1, -1), which would not conform with the behaviour of the other augmenters.
        points_dest[:, 0] = np.clip(points_dest[:, 0], 0, h - 1)
        points_dest[:, 1] = np.clip(points_dest[:, 1], 0, w - 1)

        matrix = skimage.transform.PiecewiseAffineTransform()
        matrix.estimate(points_src[:, ::-1], points_dest[:, ::-1])

        return {
            "matrix": matrix,
        }

    def apply(
        self,
        img: np.ndarray,
        matrix: skimage.transform.PiecewiseAffineTransform = None,
        **params,
    ) -> np.ndarray:
        return piecewise_affine(img, matrix, self.interpolation, self.mode, self.cval)

    def apply_to_mask(
        self,
        img: np.ndarray,
        matrix: skimage.transform.PiecewiseAffineTransform = None,
        **params,
    ) -> np.ndarray:
        return piecewise_affine(img, matrix, self.mask_interpolation, self.mode, self.cval_mask)

    def apply_to_bbox(
        self,
        bbox: Sequence[float],
        rows: int = 0,
        cols: int = 0,
        matrix: skimage.transform.PiecewiseAffineTransform = None,
        **params,
    ) -> Sequence[float]:
        return bbox_piecewise_affine(bbox, matrix, rows, cols, self.keypoints_threshold)

    def apply_to_keypoint(
        self,
        keypoint: Sequence[float],
        rows: int = 0,
        cols: int = 0,
        matrix: skimage.transform.PiecewiseAffineTransform = None,
        **params,
    ):
        return keypoint_piecewise_affine(keypoint, matrix, rows, cols, self.keypoints_threshold)
