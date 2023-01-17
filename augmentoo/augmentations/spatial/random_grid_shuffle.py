from typing import Tuple

import numpy as np

from augmentoo import random_utils
from augmentoo.core.transforms_interface import DualTransform


def swap_tiles_on_image(image, tiles):
    """
    Swap tiles on image.

    Args:
        image (np.ndarray): Input image.
        tiles (np.ndarray): array of tuples(
            current_left_up_corner_row, current_left_up_corner_col,
            old_left_up_corner_row, old_left_up_corner_col,
            height_tile, width_tile)

    Returns:
        np.ndarray: Output image.

    """
    new_image = image.copy()

    for tile in tiles:
        new_image[tile[0] : tile[0] + tile[4], tile[1] : tile[1] + tile[5]] = image[
            tile[2] : tile[2] + tile[4], tile[3] : tile[3] + tile[5]
        ]

    return new_image


class RandomGridShuffle(DualTransform):
    """
    Random shuffle grid's cells on image.

    Args:
        grid ((int, int)): size of grid for splitting image.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, grid: Tuple[int, int] = (3, 3), always_apply: bool = False, p: float = 0.5):
        super(RandomGridShuffle, self).__init__(always_apply, p)
        self.grid = grid

    def apply(self, img: np.ndarray, tiles: np.ndarray = None, **params):
        if tiles is not None:
            img = swap_tiles_on_image(img, tiles)
        return img

    def apply_to_mask(self, img: np.ndarray, tiles: np.ndarray = None, **params):
        if tiles is not None:
            img = swap_tiles_on_image(img, tiles)
        return img

    def apply_to_keypoint(
        self, keypoint: Tuple[float, ...], tiles: np.ndarray = None, rows: int = 0, cols: int = 0, **params
    ):
        if tiles is None:
            return keypoint

        for (
            current_left_up_corner_row,
            current_left_up_corner_col,
            old_left_up_corner_row,
            old_left_up_corner_col,
            height_tile,
            width_tile,
        ) in tiles:
            x, y = keypoint[:2]

            if (old_left_up_corner_row <= y < (old_left_up_corner_row + height_tile)) and (
                old_left_up_corner_col <= x < (old_left_up_corner_col + width_tile)
            ):
                x = x - old_left_up_corner_col + current_left_up_corner_col
                y = y - old_left_up_corner_row + current_left_up_corner_row
                keypoint = (x, y) + tuple(keypoint[2:])
                break

        return keypoint

    def get_params_dependent_on_targets(self, params):
        height, width = params["image"].shape[:2]
        n, m = self.grid

        if n <= 0 or m <= 0:
            raise ValueError("Grid's values must be positive. Current grid [%s, %s]" % (n, m))

        if n > height // 2 or m > width // 2:
            raise ValueError("Incorrect size cell of grid. Just shuffle pixels of image")

        height_split = np.linspace(0, height, n + 1, dtype=int)
        width_split = np.linspace(0, width, m + 1, dtype=int)

        height_matrix, width_matrix = np.meshgrid(height_split, width_split, indexing="ij")

        index_height_matrix = height_matrix[:-1, :-1]
        index_width_matrix = width_matrix[:-1, :-1]

        shifted_index_height_matrix = height_matrix[1:, 1:]
        shifted_index_width_matrix = width_matrix[1:, 1:]

        height_tile_sizes = shifted_index_height_matrix - index_height_matrix
        width_tile_sizes = shifted_index_width_matrix - index_width_matrix

        tiles_sizes = np.stack((height_tile_sizes, width_tile_sizes), axis=2)

        index_matrix = np.indices((n, m))
        new_index_matrix = np.stack(index_matrix, axis=2)

        for bbox_size in np.unique(tiles_sizes.reshape(-1, 2), axis=0):
            eq_mat = np.all(tiles_sizes == bbox_size, axis=2)
            new_index_matrix[eq_mat] = random_utils.permutation(new_index_matrix[eq_mat])

        new_index_matrix = np.split(new_index_matrix, 2, axis=2)

        old_x = index_height_matrix[new_index_matrix[0], new_index_matrix[1]].reshape(-1)
        old_y = index_width_matrix[new_index_matrix[0], new_index_matrix[1]].reshape(-1)

        shift_x = height_tile_sizes.reshape(-1)
        shift_y = width_tile_sizes.reshape(-1)

        curr_x = index_height_matrix.reshape(-1)
        curr_y = index_width_matrix.reshape(-1)

        tiles = np.stack([curr_x, curr_y, old_x, old_y, shift_x, shift_y], axis=1)

        return {"tiles": tiles}

    @property
    def targets_as_params(self):
        return ["image"]
