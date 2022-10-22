@preserve_shape
def grid_distortion(
    img,
    num_steps=10,
    xsteps=(),
    ysteps=(),
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
):
    """Perform a grid distortion of an input image.

    Reference:
        https://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    """
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        y = idx * y_step
        start = int(y)
        end = int(y) + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    remap_fn = _maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)


class GridDistortion(DualTransform):
    """
    Args:
        num_steps (int): count of grid cells on each side.
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.03, 0.03).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        num_steps=5,
        distort_limit=0.3,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(GridDistortion, self).__init__(always_apply, p)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, stepsx=(), stepsy=(), interpolation=cv2.INTER_LINEAR, **params):
        return F.grid_distortion(
            img,
            self.num_steps,
            stepsx,
            stepsy,
            interpolation,
            self.border_mode,
            self.value,
        )

    def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
        return F.grid_distortion(
            img,
            self.num_steps,
            stepsx,
            stepsy,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
        )

    def get_params(self):
        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        return {"stepsx": stepsx, "stepsy": stepsy}

    def get_transform_init_args_names(self):
        return (
            "num_steps",
            "distort_limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
        )
