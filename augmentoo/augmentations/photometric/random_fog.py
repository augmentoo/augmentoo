from __future__ import division, absolute_import

import random


from augmentoo.core.transforms_interface import ImageOnlyTransform


@preserve_shape
def add_fog(img, fog_coef, alpha_coef, haze_list):
    """Add fog to the image.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray): Image.
        fog_coef (float): Fog coefficient.
        alpha_coef (float): Alpha coefficient.
        haze_list (list):

    Returns:
        numpy.ndarray: Image.

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomFog augmentation".format(input_dtype))

    width = img.shape[1]

    hw = max(int(width // 3 * fog_coef), 10)

    for haze_points in haze_list:
        x, y = haze_points
        overlay = img.copy()
        output = img.copy()
        alpha = alpha_coef * fog_coef
        rad = hw // 2
        point = (x + hw // 2, y + hw // 2)
        cv2.circle(overlay, point, int(rad), (255, 255, 255), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        img = output.copy()

    image_rgb = cv2.blur(img, (hw // 10, hw // 10))

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


class RandomFog(ImageOnlyTransform):
    """Simulates fog for the image

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        fog_coef_lower (float): lower limit for fog intensity coefficient. Should be in [0, 1] range.
        fog_coef_upper (float): upper limit for fog intensity coefficient. Should be in [0, 1] range.
        alpha_coef (float): transparency of the fog circles. Should be in [0, 1] range.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        fog_coef_lower=0.3,
        fog_coef_upper=1,
        alpha_coef=0.08,
        always_apply=False,
        p=0.5,
    ):
        super(RandomFog, self).__init__(always_apply, p)

        if not 0 <= fog_coef_lower <= fog_coef_upper <= 1:
            raise ValueError(
                "Invalid combination if fog_coef_lower and fog_coef_upper. Got: {}".format(
                    (fog_coef_lower, fog_coef_upper)
                )
            )
        if not 0 <= alpha_coef <= 1:
            raise ValueError("alpha_coef must be in range [0, 1]. Got: {}".format(alpha_coef))

        self.fog_coef_lower = fog_coef_lower
        self.fog_coef_upper = fog_coef_upper
        self.alpha_coef = alpha_coef

    def apply(self, image, fog_coef=0.1, haze_list=(), **params):
        return F.add_fog(image, fog_coef, self.alpha_coef, haze_list)

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        fog_coef = random.uniform(self.fog_coef_lower, self.fog_coef_upper)

        height, width = imshape = img.shape[:2]

        hw = max(1, int(width // 3 * fog_coef))

        haze_list = []
        midx = width // 2 - 2 * hw
        midy = height // 2 - hw
        index = 1

        while midx > -hw or midy > -hw:
            for _i in range(hw // 10 * index):
                x = random.randint(midx, width - midx - hw)
                y = random.randint(midy, height - midy - hw)
                haze_list.append((x, y))

            midx -= 3 * hw * width // sum(imshape)
            midy -= 3 * hw * height // sum(imshape)
            index += 1

        return {"haze_list": haze_list, "fog_coef": fog_coef}

    def get_transform_init_args_names(self):
        return ("fog_coef_lower", "fog_coef_upper", "alpha_coef")
