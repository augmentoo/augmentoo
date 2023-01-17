import random

import numpy as np

from augmentoo.core.targets import is_rgb_image
from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
)

__all__ = ["FancyPCA"]


def fancy_pca(img: np.ndarray, alpha=0.1):
    """Perform 'Fancy PCA' augmentation from:
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    Args:
        img:  numpy array with (h, w, rgb) shape, as ints between 0-255)
        alpha:  how much to perturb/scale the eigen vecs and vals
                the paper used std=0.1

    Returns:
        numpy image-like array as float range(0, 1)

    """
    if not is_rgb_image(img) or img.dtype != np.uint8:
        raise TypeError("Image must be RGB image in uint8 format.")

    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    # alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx] * 255

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    return orig_img


class FancyPCA(ImageOnlyTransform):
    """Augment RGB image using FancyPCA from Krizhevsky's paper
    "ImageNet Classification with Deep Convolutional Neural Networks"

    Args:
        alpha (float):  how much to perturb/scale the eigen vecs and vals.
            scale is samples from gaussian distribution (mu=0, sigma=alpha)

    Targets:
        image

    Image types:
        3-channel uint8 images only

    Credit:
        https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
        https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
    """

    def __init__(self, alpha=0.1, always_apply=False, p=0.5):
        super(FancyPCA, self).__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, img, alpha=0.1, **params):
        img = fancy_pca(img, alpha)
        return img

    def get_params(self):
        return {"alpha": random.gauss(0, self.alpha)}
