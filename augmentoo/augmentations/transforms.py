from __future__ import absolute_import, division

import random
import typing
import warnings
from enum import Enum
from types import LambdaType
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class Downscale(ImageOnlyTransform):
    """Decreases image quality by downscaling and upscaling back.

    Args:
        scale_min (float): lower bound on the image scale. Should be < 1.
        scale_max (float):  lower bound on the image scale. Should be .
        interpolation: cv2 interpolation method. cv2.INTER_NEAREST by default

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale_min=0.25,
        scale_max=0.25,
        interpolation=cv2.INTER_NEAREST,
        always_apply=False,
        p=0.5,
    ):
        super(Downscale, self).__init__(always_apply, p)
        if scale_min > scale_max:
            raise ValueError("Expected scale_min be less or equal scale_max, got {} {}".format(scale_min, scale_max))
        if scale_max >= 1:
            raise ValueError("Expected scale_max to be less than 1, got {}".format(scale_max))
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.interpolation = interpolation

    def apply(self, image, scale, interpolation, **params):
        return F.downscale(image, scale=scale, interpolation=interpolation)

    def get_params(self):
        return {
            "scale": random.uniform(self.scale_min, self.scale_max),
            "interpolation": self.interpolation,
        }


class TemplateTransform(ImageOnlyTransform):
    """
    Apply blending of input image with specified templates
    Args:
        templates (list of numpy arrays): Images as template for transform.
        img_weight ((float, float) or float): If single float will be used as weight for input image.
            If tuple of float img_weight will be in range `[img_weight[0], img_weight[1])`. Default: 0.5.
        template_weight ((float, float) or float): If single float will be used as weight for template.
            If tuple of float template_weight will be in range `[template_weight[0], template_weight[1])`.
            Default: 0.5.
        template_transform: transformation object which could be applied to template,
            must produce template the same size as input image.
        name (string): (Optional) Name of transform, used only for deserialization.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        templates: List[np.ndarray],
        img_weight=0.5,
        template_weight=0.5,
        template_transform=None,
        name=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)

        if isinstance(templates, np.ndarray) or not isinstance(templates, typing.Iterable):
            raise ValueError("templates must be list of numpy arrays")
        self.templates = templates
        self.img_weight = to_tuple(img_weight, img_weight)
        self.template_weight = to_tuple(template_weight, template_weight)
        self.template_transform = template_transform
        self.name = name

    def apply(self, img, template=None, img_weight=0.5, template_weight=0.5, **params):
        return F.add_weighted(img, img_weight, template, template_weight)

    def get_params(self):
        return {
            "img_weight": random.uniform(self.img_weight[0], self.img_weight[1]),
            "template_weight": random.uniform(self.template_weight[0], self.template_weight[1]),
        }

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        template = random.choice(self.templates)

        if self.template_transform is not None:
            template = self.template_transform(image=template)["image"]

        if F.get_num_channels(template) not in [1, F.get_num_channels(img)]:
            raise ValueError(
                "Template must be a single channel or "
                "has the same number of channels as input image ({}), got {}".format(
                    F.get_num_channels(img), F.get_num_channels(template)
                )
            )

        if template.dtype != img.dtype:
            raise ValueError("Image and template must be the same image type")

        if img.shape[:2] != template.shape[:2]:
            raise ValueError(
                "Image and template must be the same size, got {} and {}".format(img.shape[:2], template.shape[:2])
            )

        if F.get_num_channels(template) == 1 and F.get_num_channels(img) > 1:
            template = np.stack((template,) * F.get_num_channels(img), axis=-1)

        # in order to support grayscale image with dummy dim
        template = template.reshape(img.shape)

        return {"template": template}

    @classmethod
    def is_serializable(cls):
        return False

    @property
    def targets_as_params(self):
        return ["image"]

    def _to_dict(self):
        if self.name is None:
            raise ValueError(
                "To make a TemplateTransform serializable you should provide the `name` argument, "
                "e.g. `TemplateTransform(name='my_transform', ...)`."
            )
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}
