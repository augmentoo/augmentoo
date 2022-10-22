from augmentoo.augmentations.blur.box_blur import Blur

__all__ = ["MedianBlur"]


@preserve_shape
def median_blur(img, ksize):
    if img.dtype == np.float32 and ksize not in {3, 5}:
        raise ValueError(
            "Invalid ksize value {}. For a float32 image the only valid ksize values are 3 and 5".format(ksize)
        )

    blur_fn = _maybe_process_in_chunks(cv2.medianBlur, ksize=ksize)
    return blur_fn(img)


class MedianBlur(Blur):
    """Blur the input image using a median filter with a random aperture linear size.

    Args:
        blur_limit (int): maximum aperture linear size for blurring the input image.
            Must be odd and in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=7, always_apply=False, p=0.5):
        super(MedianBlur, self).__init__(blur_limit, always_apply, p)

        if self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError("MedianBlur supports only odd blur limits.")

    def apply(self, image, ksize=3, **params):
        return median_blur(image, ksize)
