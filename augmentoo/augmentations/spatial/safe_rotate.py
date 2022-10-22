import random

import cv2
import numpy as np

from augmentoo.augmentations import _maybe_process_in_chunks
from augmentoo.augmentations.geometric.resize import image_resize, keypoint_scale
from augmentoo.augmentations.geometric.rotate import keypoint_rotate, bbox_rotate
from augmentoo.core.decorators import preserve_channel_dim
from augmentoo.core.transforms_interface import DualTransform, to_tuple


@preserve_channel_dim
def image_safe_rotate(
    img: np.ndarray,
    angle: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
    value: int = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
):

    old_rows, old_cols = img.shape[:2]

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (old_cols / 2, old_rows / 2)

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    # Rotation Matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Shift the image to create padding
    rotation_mat[0, 2] += new_cols / 2 - image_center[0]
    rotation_mat[1, 2] += new_rows / 2 - image_center[1]

    # CV2 Transformation function
    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine,
        M=rotation_mat,
        dsize=(new_cols, new_rows),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )

    # rotate image with the new bounds
    rotated_img = warp_affine_fn(img)

    # Resize image back to the original size
    resized_img = image_resize(img=rotated_img, height=old_rows, width=old_cols, interpolation=interpolation)
    return resized_img


def bbox_safe_rotate(bbox, angle, rows, cols):
    old_rows = rows
    old_cols = cols

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    col_diff = int(np.ceil(abs(new_cols - old_cols) / 2))
    row_diff = int(np.ceil(abs(new_rows - old_rows) / 2))

    # Normalize shifts
    norm_col_shift = col_diff / new_cols
    norm_row_shift = row_diff / new_rows

    # shift bbox
    shifted_bbox = (
        bbox[0] + norm_col_shift,
        bbox[1] + norm_row_shift,
        bbox[2] + norm_col_shift,
        bbox[3] + norm_row_shift,
    )

    rotated_bbox = bbox_rotate(bbox=shifted_bbox, angle=angle, rows=new_rows, cols=new_cols)

    # Bounding boxes are scale invariant, so this does not need to be rescaled to the old size
    return rotated_bbox


def keypoint_safe_rotate(keypoint, angle, rows, cols):
    old_rows = rows
    old_cols = cols

    # Rows and columns of the rotated image (not cropped)
    new_rows, new_cols = safe_rotate_enlarged_img_size(angle=angle, rows=old_rows, cols=old_cols)

    col_diff = int(np.ceil(abs(new_cols - old_cols) / 2))
    row_diff = int(np.ceil(abs(new_rows - old_rows) / 2))

    # Shift keypoint
    shifted_keypoint = (keypoint[0] + col_diff, keypoint[1] + row_diff, keypoint[2], keypoint[3])

    # Rotate keypoint
    rotated_keypoint = keypoint_rotate(shifted_keypoint, angle, rows=new_rows, cols=new_cols)

    # Scale the keypoint
    return keypoint_scale(rotated_keypoint, old_cols / new_cols, old_rows / new_rows)


def safe_rotate_enlarged_img_size(angle: float, rows: int, cols: int):

    deg_angle = abs(angle)

    # The rotation angle
    angle = np.deg2rad(deg_angle % 90)

    # The width of the frame to contain the rotated image
    r_cols = cols * np.cos(angle) + rows * np.sin(angle)

    # The height of the frame to contain the rotated image
    r_rows = cols * np.sin(angle) + rows * np.cos(angle)

    # The above calculations work as is for 0<90 degrees, and for 90<180 the cols and rows are flipped
    if deg_angle > 90:
        return int(r_cols), int(r_rows)
    else:
        return int(r_rows), int(r_cols)


class SafeRotate(DualTransform):
    """Rotate the input inside the input's frame by an angle selected randomly from the uniform distribution.

    The resulting image may have artifacts in it. After rotation, the image may have a different aspect ratio, and
    after resizing, it returns to its original shape with the original aspect ratio of the image. For these reason we
    may see some artifacts.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
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
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(SafeRotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return image_safe_rotate(
            img=img, value=self.value, angle=angle, interpolation=interpolation, border_mode=self.border_mode
        )

    def apply_to_mask(self, img, angle=0, **params):
        return image_safe_rotate(
            img=img, value=self.mask_value, angle=angle, interpolation=cv2.INTER_NEAREST, border_mode=self.border_mode
        )

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=0, **params):
        return bbox_safe_rotate(bbox=bbox, angle=angle, rows=params["rows"], cols=params["cols"])

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return keypoint_safe_rotate(keypoint, angle=angle, rows=params["rows"], cols=params["cols"])

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value", "mask_value")
