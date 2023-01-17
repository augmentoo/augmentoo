from __future__ import division, absolute_import


from augmentoo.core.transforms_interface import DualTransform


def bbox_transpose(bbox, axis, rows, cols):  # skipcq: PYL-W0613
    """Transposes a bounding box along given axis.

    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        axis (int): 0 - main axis, 1 - secondary axis.
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        tuple: A bounding box tuple `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If axis not equal to 0 or 1.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    if axis not in {0, 1}:
        raise ValueError("Axis must be either 0 or 1.")
    if axis == 0:
        bbox = (y_min, x_min, y_max, x_max)
    if axis == 1:
        bbox = (1 - y_max, 1 - x_max, 1 - y_min, 1 - x_min)
    return bbox


def keypoint_transpose(keypoint):
    """Rotate a keypoint by angle.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]

    if angle <= np.pi:
        angle = np.pi - angle
    else:
        angle = 3 * np.pi - angle

    return y, x, angle, scale


class Transpose(DualTransform):
    """Transpose the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        return F.transpose(img)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_transpose(bbox, 0, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_transpose(keypoint)
