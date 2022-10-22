import warnings
from types import LambdaType

import numpy as np

from augmentoo.core.transforms_interface import NoOp, ImageOnlyTransform

__all__ = ["ToFloat", "FromFloat", "Lambda"]


class ToFloat(ImageOnlyTransform):
    """Divide pixel values by `max_value` to get a float32 output array where all values lie in the range [0, 1.0].
    If `max_value` is None the transform will try to infer the maximum value by inspecting the data type of the input
    image.

    See Also:
        :class:`~augmentoo.augmentations.transforms.FromFloat`

    Args:
        max_value (float): maximum possible input value. Default: None.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        any type

    """

    def __init__(self, max_value=None, always_apply=False, p=1.0):
        super(ToFloat, self).__init__(always_apply, p)
        self.max_value = max_value

    def apply(self, img, **params):
        return F.to_float(img, self.max_value)

    def get_transform_init_args_names(self):
        return ("max_value",)


class FromFloat(ImageOnlyTransform):
    """Take an input array where all values should lie in the range [0, 1.0], multiply them by `max_value` and then
    cast the resulted value to a type specified by `dtype`. If `max_value` is None the transform will try to infer
    the maximum value for the data type from the `dtype` argument.

    This is the inverse transform for :class:`~augmentoo.augmentations.transforms.ToFloat`.

    Args:
        max_value (float): maximum possible input value. Default: None.
        dtype (string or numpy data type): data type of the output. See the `'Data types' page from the NumPy docs`_.
            Default: 'uint16'.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        float32

    .. _'Data types' page from the NumPy docs:
       https://docs.scipy.org/doc/numpy/user/basics.types.html
    """

    def __init__(self, dtype="uint16", max_value=None, always_apply=False, p=1.0):
        super(FromFloat, self).__init__(always_apply, p)
        self.dtype = np.dtype(dtype)
        self.max_value = max_value

    def apply(self, img, **params):
        return F.from_float(img, self.dtype, self.max_value)

    def get_transform_init_args(self):
        return {"dtype": self.dtype.name, "max_value": self.max_value}


def _noop(input_obj, **params):  # skipcq: PYL-W0613
    return input_obj


class Lambda(NoOp):
    """A flexible transformation class for using user-defined transformation functions per targets.
    Function signature must include **kwargs to accept optinal arguments like interpolation method, image size, etc:

    Args:
        image (callable): Image transformation function.
        mask (callable): Mask transformation function.
        keypoint (callable): Keypoint transformation function.
        bbox (callable): BBox transformation function.
        always_apply (bool): Indicates whether this transformation should be always applied.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        Any
    """

    def __init__(
        self,
        image=None,
        mask=None,
        keypoint=None,
        bbox=None,
        name=None,
        always_apply=False,
        p=1.0,
    ):
        super(Lambda, self).__init__(always_apply, p)

        self.name = name
        self.custom_apply_fns = {target_name: _noop for target_name in ("image", "mask", "keypoint", "bbox")}
        for target_name, custom_apply_fn in {
            "image": image,
            "mask": mask,
            "keypoint": keypoint,
            "bbox": bbox,
        }.items():
            if custom_apply_fn is not None:
                if isinstance(custom_apply_fn, LambdaType) and custom_apply_fn.__name__ == "<lambda>":
                    warnings.warn(
                        "Using lambda is incompatible with multiprocessing. "
                        "Consider using regular functions or partial()."
                    )

                self.custom_apply_fns[target_name] = custom_apply_fn

    def apply(self, img, **params):
        fn = self.custom_apply_fns["image"]
        return fn(img, **params)

    def apply_to_mask(self, mask, **params):
        fn = self.custom_apply_fns["mask"]
        return fn(mask, **params)

    def apply_to_bbox(self, bbox, **params):
        fn = self.custom_apply_fns["bbox"]
        return fn(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        fn = self.custom_apply_fns["keypoint"]
        return fn(keypoint, **params)

    @classmethod
    def is_serializable(cls):
        return False

    def _to_dict(self):
        if self.name is None:
            raise ValueError(
                "To make a Lambda transform serializable you should provide the `name` argument, "
                "e.g. `Lambda(name='my_transform', image=<some func>, ...)`."
            )
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}

    def __repr__(self):
        state = {"name": self.name}
        state.update(self.custom_apply_fns.items())
        state.update(self.get_base_init_args())
        return "{name}({args})".format(name=self.__class__.__name__, args=format_args(state))
