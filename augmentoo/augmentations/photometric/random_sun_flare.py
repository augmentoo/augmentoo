from __future__ import division, absolute_import

import math
import random

import cv2
import numpy as np

from augmentoo.core.decorators import preserve_shape
from augmentoo.core.transforms_interface import ImageOnlyTransform
from augmentoo.core.warnings import non_rgb_warning


@preserve_shape
def add_sun_flare(img, flare_center_x, flare_center_y, src_radius, src_color, circles):
    """Add sun flare.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray):
        flare_center_x (float):
        flare_center_y (float):
        src_radius:
        src_color (int, int, int):
        circles (list):

    Returns:
        numpy.ndarray:

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomSunFlareaugmentation".format(input_dtype))

    overlay = img.copy()
    output = img.copy()

    for (alpha, (x, y), rad3, (r_color, g_color, b_color)) in circles:
        cv2.circle(overlay, (x, y), rad3, (r_color, g_color, b_color), -1)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    point = (int(flare_center_x), int(flare_center_y))

    overlay = output.copy()
    num_times = src_radius // 10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, src_radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        cv2.addWeighted(overlay, alp, output, 1 - alp, 0, output)

    image_rgb = output

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


class RandomSunFlare(ImageOnlyTransform):
    """Simulates Sun Flare for the image

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        flare_roi (float, float, float, float): region of the image where flare will
            appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
        angle_lower (float): should be in range [0, `angle_upper`].
        angle_upper (float): should be in range [`angle_lower`, 1].
        num_flare_circles_lower (int): lower limit for the number of flare circles.
            Should be in range [0, `num_flare_circles_upper`].
        num_flare_circles_upper (int): upper limit for the number of flare circles.
            Should be in range [`num_flare_circles_lower`, inf].
        src_radius (int):
        src_color ((int, int, int)): color of the flare

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        flare_roi=(0, 0, 1, 0.5),
        angle_lower=0,
        angle_upper=1,
        num_flare_circles_lower=6,
        num_flare_circles_upper=10,
        src_radius=400,
        src_color=(255, 255, 255),
        always_apply=False,
        p=0.5,
    ):
        super(RandomSunFlare, self).__init__(always_apply, p)

        (
            flare_center_lower_x,
            flare_center_lower_y,
            flare_center_upper_x,
            flare_center_upper_y,
        ) = flare_roi

        if (
            not 0 <= flare_center_lower_x < flare_center_upper_x <= 1
            or not 0 <= flare_center_lower_y < flare_center_upper_y <= 1
        ):
            raise ValueError("Invalid flare_roi. Got: {}".format(flare_roi))
        if not 0 <= angle_lower < angle_upper <= 1:
            raise ValueError(
                "Invalid combination of angle_lower nad angle_upper. Got: {}".format((angle_lower, angle_upper))
            )
        if not 0 <= num_flare_circles_lower < num_flare_circles_upper:
            raise ValueError(
                "Invalid combination of num_flare_circles_lower nad num_flare_circles_upper. Got: {}".format(
                    (num_flare_circles_lower, num_flare_circles_upper)
                )
            )

        self.flare_center_lower_x = flare_center_lower_x
        self.flare_center_upper_x = flare_center_upper_x

        self.flare_center_lower_y = flare_center_lower_y
        self.flare_center_upper_y = flare_center_upper_y

        self.angle_lower = angle_lower
        self.angle_upper = angle_upper
        self.num_flare_circles_lower = num_flare_circles_lower
        self.num_flare_circles_upper = num_flare_circles_upper

        self.src_radius = src_radius
        self.src_color = src_color

    def apply(self, image, flare_center_x=0.5, flare_center_y=0.5, circles=(), **params):
        return F.add_sun_flare(
            image,
            flare_center_x,
            flare_center_y,
            self.src_radius,
            self.src_color,
            circles,
        )

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        angle = 2 * math.pi * random.uniform(self.angle_lower, self.angle_upper)

        flare_center_x = random.uniform(self.flare_center_lower_x, self.flare_center_upper_x)
        flare_center_y = random.uniform(self.flare_center_lower_y, self.flare_center_upper_y)

        flare_center_x = int(width * flare_center_x)
        flare_center_y = int(height * flare_center_y)

        num_circles = random.randint(self.num_flare_circles_lower, self.num_flare_circles_upper)

        circles = []

        x = []
        y = []

        for rand_x in range(0, width, 10):
            rand_y = math.tan(angle) * (rand_x - flare_center_x) + flare_center_y
            x.append(rand_x)
            y.append(2 * flare_center_y - rand_y)

        for _i in range(num_circles):
            alpha = random.uniform(0.05, 0.2)
            r = random.randint(0, len(x) - 1)
            rad = random.randint(1, max(height // 100 - 2, 2))

            r_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])
            g_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])
            b_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])

            circles += [
                (
                    alpha,
                    (int(x[r]), int(y[r])),
                    pow(rad, 3),
                    (r_color, g_color, b_color),
                )
            ]

        return {
            "circles": circles,
            "flare_center_x": flare_center_x,
            "flare_center_y": flare_center_y,
        }

    def get_transform_init_args(self):
        return {
            "flare_roi": (
                self.flare_center_lower_x,
                self.flare_center_lower_y,
                self.flare_center_upper_x,
                self.flare_center_upper_y,
            ),
            "angle_lower": self.angle_lower,
            "angle_upper": self.angle_upper,
            "num_flare_circles_lower": self.num_flare_circles_lower,
            "num_flare_circles_upper": self.num_flare_circles_upper,
            "src_radius": self.src_radius,
            "src_color": self.src_color,
        }
