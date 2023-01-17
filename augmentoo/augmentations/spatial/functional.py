import typing
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import skimage.transform

__all__ = [
    "py3round",
    "is_identity_matrix",
    "to_distance_maps",
    "from_distance_maps",
]


def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))


def is_identity_matrix(matrix: skimage.transform.ProjectiveTransform) -> bool:
    return np.allclose(matrix.params, np.eye(3, dtype=np.float32))


def to_distance_maps(
    keypoints: Sequence[Sequence[float]],
    height: int,
    width: int,
    inverted: bool = False,
) -> np.ndarray:
    """Generate a ``(H,W,N)`` array of distance maps for ``N`` keypoints.

    The ``n``-th distance map contains at every location ``(y, x)`` the
    euclidean distance to the ``n``-th keypoint.

    This function can be used as a helper when augmenting keypoints with a
    method that only supports the augmentation of images.

    Args:
        keypoint (sequence of float): keypoint coordinates
        height (int): image height
        width (int): image width
        inverted (bool): If ``True``, inverted distance maps are returned where each
            distance value d is replaced by ``d/(d+1)``, i.e. the distance
            maps have values in the range ``(0.0, 1.0]`` with ``1.0`` denoting
            exactly the position of the respective keypoint.

    Returns:
        (H,W,N) ndarray
            A ``float32`` array containing ``N`` distance maps for ``N``
            keypoints. Each location ``(y, x, n)`` in the array denotes the
            euclidean distance at ``(y, x)`` to the ``n``-th keypoint.
            If `inverted` is ``True``, the distance ``d`` is replaced
            by ``d/(d+1)``. The height and width of the array match the
            height and width in ``KeypointsOnImage.shape``.
    """
    distance_maps = np.zeros((height, width, len(keypoints)), dtype=np.float32)

    yy = np.arange(0, height)
    xx = np.arange(0, width)
    grid_xx, grid_yy = np.meshgrid(xx, yy)

    for i, (x, y) in enumerate(keypoints):
        distance_maps[:, :, i] = (grid_xx - x) ** 2 + (grid_yy - y) ** 2

    distance_maps = np.sqrt(distance_maps)
    if inverted:
        return 1 / (distance_maps + 1)
    return distance_maps


def from_distance_maps(
    distance_maps: np.ndarray,
    inverted: bool,
    if_not_found_coords: Optional[Union[Sequence[int], dict]],
    threshold: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Convert outputs of ``to_distance_maps()`` to ``KeypointsOnImage``.
    This is the inverse of `to_distance_maps`.

    Args:
        distance_maps (np.ndarray): The distance maps. ``N`` is the number of keypoints.
        inverted (bool): Whether the given distance maps were generated in inverted mode
            (i.e. :func:`KeypointsOnImage.to_distance_maps` was called with ``inverted=True``) or in non-inverted mode.
        if_not_found_coords (tuple, list, dict or None, optional):
            Coordinates to use for keypoints that cannot be found in `distance_maps`.

            * If this is a ``list``/``tuple``, it must contain two ``int`` values.
            * If it is a ``dict``, it must contain the keys ``x`` and ``y`` with each containing one ``int`` value.
            * If this is ``None``, then the keypoint will not be added.
        threshold (float): The search for keypoints works by searching for the
            argmin (non-inverted) or argmax (inverted) in each channel. This
            parameters contains the maximum (non-inverted) or minimum (inverted) value to accept in order to view a hit
            as a keypoint. Use ``None`` to use no min/max.
        nb_channels (None, int): Number of channels of the image on which the keypoints are placed.
            Some keypoint augmenters require that information. If set to ``None``, the keypoint's shape will be set
            to ``(height, width)``, otherwise ``(height, width, nb_channels)``.
    """
    if distance_maps.ndim != 3:
        raise ValueError(
            f"Expected three-dimensional input, "
            f"got {distance_maps.ndim} dimensions and shape {distance_maps.shape}."
        )
    height, width, nb_keypoints = distance_maps.shape

    drop_if_not_found = False
    if if_not_found_coords is None:
        drop_if_not_found = True
        if_not_found_x = -1
        if_not_found_y = -1
    elif isinstance(if_not_found_coords, typing.Mapping):
        if_not_found_x = if_not_found_coords["x"]
        if_not_found_y = if_not_found_coords["y"]
    elif isinstance(if_not_found_coords, typing.Iterable):
        if len(if_not_found_coords) != 2:
            raise ValueError(
                f"Expected tuple/list 'if_not_found_coords' to contain exactly two entries, "
                f"got {len(if_not_found_coords)}."
            )
        if_not_found_x = if_not_found_coords[0]
        if_not_found_y = if_not_found_coords[1]
    else:
        raise ValueError(
            f"Expected if_not_found_coords to be None or tuple or list or dict, got {type(if_not_found_coords)}."
        )

    keypoints = []
    for i in range(nb_keypoints):
        if inverted:
            hitidx_flat = np.argmax(distance_maps[..., i])
        else:
            hitidx_flat = np.argmin(distance_maps[..., i])
        hitidx_ndim = np.unravel_index(hitidx_flat, (height, width))
        if not inverted and threshold is not None:
            found = distance_maps[hitidx_ndim[0], hitidx_ndim[1], i] < threshold
        elif inverted and threshold is not None:
            found = distance_maps[hitidx_ndim[0], hitidx_ndim[1], i] >= threshold
        else:
            found = True
        if found:
            keypoints.append((float(hitidx_ndim[1]), float(hitidx_ndim[0])))
        else:
            if not drop_if_not_found:
                keypoints.append((if_not_found_x, if_not_found_y))

    return keypoints
