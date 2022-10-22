import random
import typing
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import skimage.transform

from augmentoo.augmentations.geometric.functional import is_identity_matrix
from augmentoo.core.decorators import angle_2pi_range, preserve_channel_dim
from augmentoo.core.transforms_interface import DualTransform, to_tuple

__all__ = ["Affine"]


@preserve_channel_dim
def warp_affine(
    image: np.ndarray,
    matrix: skimage.transform.ProjectiveTransform,
    interpolation: int,
    cval: Union[int, float, Sequence[int], Sequence[float]],
    mode: int,
    output_shape: Sequence[int],
) -> np.ndarray:
    if is_identity_matrix(matrix):
        return image

    dsize = int(np.round(output_shape[1])), int(np.round(output_shape[0]))
    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix.params[:2], dsize=dsize, flags=interpolation, borderMode=mode, borderValue=cval
    )
    tmp = warp_fn(image)
    return tmp


@angle_2pi_range
def keypoint_affine(
    keypoint: Sequence[float],
    matrix: skimage.transform.ProjectiveTransform,
    scale: dict,
) -> Sequence[float]:
    if is_identity_matrix(matrix):
        return keypoint

    x, y, a, s = keypoint[:4]
    x, y = skimage.transform.matrix_transform(np.array([[x, y]]), matrix.params).ravel()
    a += rotation2DMatrixToEulerAngles(matrix.params[:2])
    s *= np.max([scale["x"], scale["y"]])
    return x, y, a, s


def bbox_affine(
    bbox: Sequence[float],
    matrix: skimage.transform.ProjectiveTransform,
    rows: int,
    cols: int,
    output_shape: Sequence[int],
) -> Sequence[float]:
    if is_identity_matrix(matrix):
        return bbox

    x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
    points = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
    )
    points = skimage.transform.matrix_transform(points, matrix.params)
    points[:, 0] = np.clip(points[:, 0], 0, output_shape[1])
    points[:, 1] = np.clip(points[:, 1], 0, output_shape[0])
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    return normalize_bbox((x_min, y_min, x_max, y_max), output_shape[0], output_shape[1])


class Affine(DualTransform):
    """Augmentation to apply affine transformations to images.
    This is mostly a wrapper around the corresponding classes and functions in OpenCV.

    Affine transformations involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a defined content, e.g.
    if the image is translated to the left, pixels are created on the right.
    A method has to be defined to deal with these pixel values.
    The parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameters `interpolation` and
    `mask_interpolation` deals with the method of interpolation used for this.

    Args:
        scale (number, tuple of number or dict): Scaling factor to use, where ``1.0`` denotes "no change" and
            ``0.5`` is zoomed out to ``50`` percent of the original size.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That value will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        translate_percent (None, number, tuple of number or dict): Translation as a fraction of the image height/width
            (x-translation, y-translation), where ``0`` denotes "no change"
            and ``0.5`` denotes "half of the axis size".
                * If ``None`` then equivalent to ``0.0`` unless `translate_px` has a value other than ``None``.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That sampled fraction value will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        translate_px (None, int, tuple of int or dict): Translation in pixels.
                * If ``None`` then equivalent to ``0`` unless `translate_percent` has a value other than ``None``.
                * If a single int, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from
                  the discrete interval ``[A2..b]``. That number will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        rotate (number or tuple of number): Rotation in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``. Rotation happens around the *center* of the image,
            not the top left corner as in some other frameworks.
                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``
                  and used as the rotation value.
        shear (number, tuple of number or dict): Shear in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``, with reasonable values being in the range of ``[-45, 45]``.
                * If a number, then that value will be used for all images as
                  the shear on the x-axis (no shear on the y-axis will be done).
                * If a tuple ``(a, b)``, then two value will be uniformly sampled per image
                  from the interval ``[a, b]`` and be used as the x- and y-shear value.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        interpolation (int): OpenCV interpolation flag.
        mask_interpolation (int): OpenCV interpolation flag.
        cval (number or sequence of number): The constant value to use when filling in newly created pixels.
            (E.g. translating by 1px to the right will create a new 1px-wide column of pixels
            on the left of the image).
            The value is only used when `mode=constant`. The expected value range is ``[0, 255]`` for ``uint8`` images.
        cval_mask (number or tuple of number): Same as cval but only for masks.
        mode (int): OpenCV border flag.
        fit_output (bool): Whether to modify the affine transformation so that the whole output image is always
            contained in the image plane (``True``) or accept parts of the image being outside
            the image plane (``False``). This can be thought of as first applying the affine transformation
            and then applying a second transformation to "zoom in" on the new image so that it fits the image plane,
            This is useful to avoid corners of the image being outside of the image plane after applying rotations.
            It will however negate translation and scaling.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        scale: Optional[Union[float, Sequence[float], dict]] = None,
        translate_percent: Optional[Union[float, Sequence[float], dict]] = None,
        translate_px: Optional[Union[int, Sequence[int], dict]] = None,
        rotate: Optional[Union[float, Sequence[float]]] = None,
        shear: Optional[Union[float, Sequence[float], dict]] = None,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        cval: Union[int, float, Sequence[int], Sequence[float]] = 0,
        cval_mask: Union[int, float, Sequence[int], Sequence[float]] = 0,
        mode: int = cv2.BORDER_CONSTANT,
        fit_output: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        params = [scale, translate_percent, translate_px, rotate, shear]
        if all([p is None for p in params]):
            scale = {"x": (0.9, 1.1), "y": (0.9, 1.1)}
            translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
            rotate = (-15, 15)
            shear = {"x": (-10, 10), "y": (-10, 10)}
        else:
            scale = scale if scale is not None else 1.0
            rotate = rotate if rotate is not None else 0.0
            shear = shear if shear is not None else 0.0

        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.scale = self._handle_dict_arg(scale, "scale")
        self.translate_percent, self.translate_px = self._handle_translate_arg(translate_px, translate_percent)
        self.rotate = to_tuple(rotate, rotate)
        self.fit_output = fit_output
        self.shear = self._handle_dict_arg(shear, "shear")

    def get_transform_init_args_names(self):
        return (
            "interpolation",
            "mask_interpolation",
            "cval",
            "mode",
            "scale",
            "translate_percent",
            "translate_px",
            "rotate",
            "fit_output",
            "shear",
            "cval_mask",
        )

    @staticmethod
    def _handle_dict_arg(val: Union[float, Sequence[float], typing.Mapping], name: str):
        if isinstance(val, typing.Mapping):
            if "x" not in val and "y" not in val:
                raise ValueError(
                    f'Expected {name} dictionary to contain at least key "x" or ' 'key "y". Found neither of them.'
                )
            x = val.get("x", 1.0)
            y = val.get("y", 1.0)
            return {"x": to_tuple(x, x), "y": to_tuple(y, y)}
        return {"x": to_tuple(val, val), "y": to_tuple(val, val)}

    @classmethod
    def _handle_translate_arg(
        cls,
        translate_px: Optional[Union[float, Sequence[float], dict]],
        translate_percent: Optional[Union[float, Sequence[float], dict]],
    ):
        if translate_percent is None and translate_px is None:
            translate_px = 0

        if translate_percent is not None and translate_px is not None:
            raise ValueError(
                "Expected either translate_percent or translate_px to be " "provided, " "but neither of them was."
            )

        if translate_percent is not None:
            # translate by percent
            return cls._handle_dict_arg(translate_percent, "translate_percent"), translate_px

        if translate_px is None:
            raise ValueError("translate_px is None.")
        # translate by pixels
        return translate_percent, cls._handle_dict_arg(translate_px, "translate_px")

    def apply(
        self,
        img: np.ndarray,
        matrix: skimage.transform.ProjectiveTransform = None,
        output_shape: Sequence[int] = None,
        **params,
    ) -> np.ndarray:
        return warp_affine(
            img,
            matrix,
            interpolation=self.interpolation,
            cval=self.cval,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_mask(
        self,
        img: np.ndarray,
        matrix: skimage.transform.ProjectiveTransform = None,
        output_shape: Sequence[int] = None,
        **params,
    ) -> np.ndarray:
        return warp_affine(
            img,
            matrix,
            interpolation=self.mask_interpolation,
            cval=self.cval_mask,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_bbox(
        self,
        bbox: Sequence[float],
        matrix: skimage.transform.ProjectiveTransform = None,
        rows: int = 0,
        cols: int = 0,
        output_shape: Sequence[int] = (),
        **params,
    ) -> Sequence[float]:
        return bbox_affine(bbox, matrix, rows, cols, output_shape)

    def apply_to_keypoint(
        self,
        keypoint: Sequence[float],
        matrix: skimage.transform.ProjectiveTransform = None,
        scale: dict = None,
        **params,
    ) -> Sequence[float]:
        return keypoint_affine(keypoint, matrix=matrix, scale=scale)

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict) -> dict:
        h, w = params["image"].shape[:2]

        translate: Dict[str, Union[int, float]]
        if self.translate_px is not None:
            translate = {key: random.randint(*value) for key, value in self.translate_px.items()}
        elif self.translate_percent is not None:
            translate = {key: random.uniform(*value) for key, value in self.translate_percent.items()}
            translate["x"] = translate["x"] * w
            translate["y"] = translate["y"] * h
        else:
            translate = {"x": 0, "y": 0}

        shear = {key: random.uniform(*value) for key, value in self.shear.items()}
        scale = {key: random.uniform(*value) for key, value in self.scale.items()}
        rotate = random.uniform(*self.rotate)

        # for images we use additional shifts of (0.5, 0.5) as otherwise
        # we get an ugly black border for 90deg rotations
        shift_x = w / 2 - 0.5
        shift_y = h / 2 - 0.5

        matrix_to_topleft = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_shear_y_rot = skimage.transform.AffineTransform(rotation=-np.pi / 2)
        matrix_shear_y = skimage.transform.AffineTransform(shear=np.deg2rad(shear["y"]))
        matrix_shear_y_rot_inv = skimage.transform.AffineTransform(rotation=np.pi / 2)
        matrix_transforms = skimage.transform.AffineTransform(
            scale=(scale["x"], scale["y"]),
            translation=(translate["x"], translate["y"]),
            rotation=np.deg2rad(rotate),
            shear=np.deg2rad(shear["x"]),
        )
        matrix_to_center = skimage.transform.SimilarityTransform(translation=[shift_x, shift_y])
        matrix = (
            matrix_to_topleft
            + matrix_shear_y_rot
            + matrix_shear_y
            + matrix_shear_y_rot_inv
            + matrix_transforms
            + matrix_to_center
        )
        if self.fit_output:
            matrix, output_shape = self._compute_affine_warp_output_shape(matrix, params["image"].shape)
        else:
            output_shape = params["image"].shape

        return {
            "rotate": rotate,
            "scale": scale,
            "matrix": matrix,
            "output_shape": output_shape,
        }

    @staticmethod
    def _compute_affine_warp_output_shape(
        matrix: skimage.transform.ProjectiveTransform, input_shape: Sequence[int]
    ) -> Tuple[skimage.transform.ProjectiveTransform, Sequence[int]]:
        height, width = input_shape[:2]

        if height == 0 or width == 0:
            return matrix, input_shape

        # determine shape of output image
        corners = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
        corners = matrix(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_height = maxr - minr + 1
        out_width = maxc - minc + 1
        if len(input_shape) == 3:
            output_shape = np.ceil((out_height, out_width, input_shape[2]))
        else:
            output_shape = np.ceil((out_height, out_width))
        output_shape_tuple = tuple([int(v) for v in output_shape.tolist()])
        # fit output image in new shape
        translation = (-minc, -minr)
        matrix_to_fit = skimage.transform.SimilarityTransform(translation=translation)
        matrix = matrix + matrix_to_fit
        return matrix, output_shape_tuple
