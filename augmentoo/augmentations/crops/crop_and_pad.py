import random
from typing import Optional, Union, Sequence, Tuple, List

import cv2
import numpy as np

from augmentoo.augmentations.crops import functional as F
from augmentoo.core.transforms_interface import DualTransform

__all__ = ["CropAndPad"]


class CropAndPad(DualTransform):
    """Crop and pad images by pixel amounts or fractions of image sizes.
    Cropping removes pixels at the sides (i.e. extracts a subimage from a given full image).
    Padding adds pixels to the sides (e.g. black pixels).
    This transformation will never crop images below a height or width of ``1``.

    Note:
        This transformation automatically resizes images back to their original size. To deactivate this, add the
        parameter ``keep_size=False``.

    Args:
        px (int or tuple):
            The number of pixels to crop (negative values) or pad (positive values)
            on each side of the image. Either this or the parameter `percent` may
            be set, not both at the same time.
                * If ``None``, then pixel-based cropping/padding will not be used.
                * If ``int``, then that exact number of pixels will always be cropped/padded.
                * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
                  then each side will be cropped/padded by a random amount sampled
                  uniformly per image and side from the interval ``[a, b]``. If
                  however `sample_independently` is set to ``False``, only one
                  value will be sampled per image and used for all sides.
                * If a ``tuple`` of four entries, then the entries represent top,
                  right, bottom, left. Each entry may be a single ``int`` (always
                  crop/pad by exactly that value), a ``tuple`` of two ``int`` s
                  ``a`` and ``b`` (crop/pad by an amount within ``[a, b]``), a
                  ``list`` of ``int`` s (crop/pad by a random value that is
                  contained in the ``list``).
        percent (float or tuple):
            The number of pixels to crop (negative values) or pad (positive values)
            on each side of the image given as a *fraction* of the image
            height/width. E.g. if this is set to ``-0.1``, the transformation will
            always crop away ``10%`` of the image's height at both the top and the
            bottom (both ``10%`` each), as well as ``10%`` of the width at the
            right and left.
            Expected value range is ``(-1.0, inf)``.
            Either this or the parameter `px` may be set, not both
            at the same time.
                * If ``None``, then fraction-based cropping/padding will not be
                  used.
                * If ``float``, then that fraction will always be cropped/padded.
                * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
                  then each side will be cropped/padded by a random fraction
                  sampled uniformly per image and side from the interval
                  ``[a, b]``. If however `sample_independently` is set to
                  ``False``, only one value will be sampled per image and used for
                  all sides.
                * If a ``tuple`` of four entries, then the entries represent top,
                  right, bottom, left. Each entry may be a single ``float``
                  (always crop/pad by exactly that percent value), a ``tuple`` of
                  two ``float`` s ``a`` and ``b`` (crop/pad by a fraction from
                  ``[a, b]``), a ``list`` of ``float`` s (crop/pad by a random
                  value that is contained in the list).
        pad_mode (int): OpenCV border mode.
        pad_cval (number, Sequence[number]):
            The constant value to use if the pad mode is ``BORDER_CONSTANT``.
                * If ``number``, then that value will be used.
                * If a ``tuple`` of two ``number`` s and at least one of them is
                  a ``float``, then a random number will be uniformly sampled per
                  image from the continuous interval ``[a, b]`` and used as the
                  value. If both ``number`` s are ``int`` s, the interval is
                  discrete.
                * If a ``list`` of ``number``, then a random value will be chosen
                  from the elements of the ``list`` and used as the value.
        pad_cval_mask (number, Sequence[number]): Same as pad_cval but only for masks.
        keep_size (bool):
            After cropping and padding, the result image will usually have a
            different height/width compared to the original input image. If this
            parameter is set to ``True``, then the cropped/padded image will be
            resized to the input image's size, i.e. the output shape is always identical to the input shape.
        sample_independently (bool):
            If ``False`` *and* the values for `px`/`percent` result in exactly
            *one* probability distribution for all image sides, only one single
            value will be sampled from that probability distribution and used for
            all sides. I.e. the crop/pad amount then is the same for all sides.
            If ``True``, four values will be sampled independently, one per side.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        any
    """

    def __init__(
        self,
        px: Optional[Union[int, Sequence[float], Sequence[Tuple]]] = None,
        percent: Optional[Union[float, Sequence[float], Sequence[Tuple]]] = None,
        pad_mode: int = cv2.BORDER_CONSTANT,
        pad_cval: Union[float, Sequence[float]] = 0,
        pad_cval_mask: Union[float, Sequence[float]] = 0,
        keep_size: bool = True,
        sample_independently: bool = True,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)

        if px is None and percent is None:
            raise ValueError("px and percent are empty!")
        if px is not None and percent is not None:
            raise ValueError("Only px or percent may be set!")

        self.px = px
        self.percent = percent

        self.pad_mode = pad_mode
        self.pad_cval = pad_cval
        self.pad_cval_mask = pad_cval_mask

        self.keep_size = keep_size
        self.sample_independently = sample_independently

        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_params: Sequence[int] = None,
        pad_params: Sequence[int] = None,
        pad_value: Union[int, float] = None,
        rows: int = None,
        cols: int = None,
        interpolation: int = cv2.INTER_LINEAR,
        **params,
    ) -> np.ndarray:
        return F.crop_and_pad(
            img,
            crop_params,
            pad_params,
            pad_value,
            rows,
            cols,
            interpolation,
            self.pad_mode,
            self.keep_size,
        )

    def apply_to_mask(
        self,
        img: np.ndarray,
        crop_params: Optional[Sequence[int]] = None,
        pad_params: Optional[Sequence[int]] = None,
        pad_value_mask: Union[int, float] = None,
        rows: int = None,
        cols: int = None,
        interpolation: int = cv2.INTER_NEAREST,
        **params,
    ) -> np.ndarray:
        return F.crop_and_pad(
            img,
            crop_params,
            pad_params,
            pad_value_mask,
            rows,
            cols,
            interpolation,
            self.pad_mode,
            self.keep_size,
        )

    def apply_to_bbox(
        self,
        bbox: Sequence[float],
        crop_params: Optional[Sequence[int]] = None,
        pad_params: Optional[Sequence[int]] = None,
        rows: int = 0,
        cols: int = 0,
        result_rows: int = 0,
        result_cols: int = 0,
        **params,
    ) -> Sequence[float]:
        return F.crop_and_pad_bbox(
            bbox,
            crop_params,
            pad_params,
            rows,
            cols,
            result_rows,
            result_cols,
            self.keep_size,
        )

    def apply_to_keypoint(
        self,
        keypoint: Sequence[float],
        crop_params: Optional[Sequence[int]] = None,
        pad_params: Optional[Sequence[int]] = None,
        rows: int = 0,
        cols: int = 0,
        result_rows: int = 0,
        result_cols: int = 0,
        **params,
    ) -> Sequence[float]:
        return F.crop_and_pad_keypoint(
            keypoint,
            crop_params,
            pad_params,
            rows,
            cols,
            result_rows,
            result_cols,
            self.keep_size,
        )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    @staticmethod
    def __prevent_zero(val1: int, val2: int, max_val: int) -> Tuple[int, int]:
        regain = abs(max_val) + 1
        regain1 = regain // 2
        regain2 = regain // 2
        if regain1 + regain2 < regain:
            regain1 += 1

        if regain1 > val1:
            diff = regain1 - val1
            regain1 = val1
            regain2 += diff
        elif regain2 > val2:
            diff = regain2 - val2
            regain2 = val2
            regain1 += diff

        val1 = val1 - regain1
        val2 = val2 - regain2

        return val1, val2

    @staticmethod
    def _prevent_zero(crop_params: List[int], height: int, width: int) -> Sequence[int]:
        top, right, bottom, left = crop_params

        remaining_height = height - (top + bottom)
        remaining_width = width - (left + right)

        if remaining_height < 1:
            top, bottom = CropAndPad.__prevent_zero(top, bottom, height)
        if remaining_width < 1:
            left, right = CropAndPad.__prevent_zero(left, right, width)

        return [max(top, 0), max(right, 0), max(bottom, 0), max(left, 0)]

    def get_params_dependent_on_targets(self, params) -> dict:
        height, width = params["image"].shape[:2]

        if self.px is not None:
            params = self._get_px_params()
        else:
            params = self._get_percent_params()
            params[0] = int(params[0] * height)
            params[1] = int(params[1] * width)
            params[2] = int(params[2] * height)
            params[3] = int(params[3] * width)

        pad_params = [max(i, 0) for i in params]

        crop_params = self._prevent_zero([-min(i, 0) for i in params], height, width)

        top, right, bottom, left = crop_params
        crop_params = [left, top, width - right, height - bottom]
        result_rows = crop_params[3] - crop_params[1]
        result_cols = crop_params[2] - crop_params[0]
        if result_cols == width and result_rows == height:
            crop_params = []

        top, right, bottom, left = pad_params
        pad_params = [top, bottom, left, right]
        if any(pad_params):
            result_rows += top + bottom
            result_cols += left + right
        else:
            pad_params = []

        return {
            "crop_params": crop_params or None,
            "pad_params": pad_params or None,
            "pad_value": None if pad_params is None else self._get_pad_value(self.pad_cval),
            "pad_value_mask": None if pad_params is None else self._get_pad_value(self.pad_cval_mask),
            "result_rows": result_rows,
            "result_cols": result_cols,
        }

    def _get_px_params(self) -> List[int]:
        if self.px is None:
            raise ValueError("px is not set")

        if isinstance(self.px, int):
            params = [self.px] * 4
        elif len(self.px) == 2:
            if self.sample_independently:
                params = [random.randrange(*self.px) for _ in range(4)]
            else:
                px = random.randrange(*self.px)
                params = [px] * 4
        else:
            params = [i if isinstance(i, int) else random.randrange(*i) for i in self.px]  # type: ignore

        return params  # [top, right, bottom, left]

    def _get_percent_params(self) -> List[float]:
        if self.percent is None:
            raise ValueError("percent is not set")

        if isinstance(self.percent, float):
            params = [self.percent] * 4
        elif len(self.percent) == 2:
            if self.sample_independently:
                params = [random.uniform(*self.percent) for _ in range(4)]
            else:
                px = random.uniform(*self.percent)
                params = [px] * 4
        else:
            params = [i if isinstance(i, (int, float)) else random.uniform(*i) for i in self.percent]

        return params  # params = [top, right, bottom, left]

    @staticmethod
    def _get_pad_value(pad_value: Union[float, Sequence[float]]) -> Union[int, float]:
        if isinstance(pad_value, (int, float)):
            return pad_value

        if len(pad_value) == 2:
            a, b = pad_value
            if isinstance(a, int) and isinstance(b, int):
                return random.randint(a, b)

            return random.uniform(a, b)

        return random.choice(pad_value)
