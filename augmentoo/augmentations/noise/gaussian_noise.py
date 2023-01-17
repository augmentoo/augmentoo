import random
import typing

import numpy as np

from augmentoo import random_utils
from augmentoo.core.decorators import clipped
from augmentoo.core.transforms_interface import ImageOnlyTransform


@clipped
def gauss_noise(image, gauss):
    image = image.astype("float32")
    return image + gauss


class GaussNoise(ImageOnlyTransform):
    """Apply gaussian noise to the input image.

    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        per_channel (bool): if set to True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels. Default: True
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        var_limit=(10.0, 50.0),
        mean=0,
        per_channel=True,
        always_apply=False,
        p=0.5,
    ):
        super(GaussNoise, self).__init__(always_apply, p)
        if isinstance(var_limit, typing.Iterable) and not isinstance(var_limit, typing.Mapping):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )

        self.mean = mean
        self.per_channel = per_channel

    def apply(self, img, gauss=None, **params):
        return gauss_noise(img, gauss=gauss)

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var**0.5

        if self.per_channel:
            gauss = random_utils.normal(self.mean, sigma, image.shape)
        else:
            gauss = random_utils.normal(self.mean, sigma, image.shape[:2])
            if len(image.shape) == 3:
                gauss = np.expand_dims(gauss, -1)

        return {"gauss": gauss}

    @property
    def targets_as_params(self):
        return ["image"]
