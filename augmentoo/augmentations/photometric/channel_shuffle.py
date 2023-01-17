import random


from augmentoo.core.transforms_interface import (
    ImageOnlyTransform,
)

__all__ = ["ChannelShuffle"]


def channel_shuffle(img, channels_shuffled):
    img = img[..., channels_shuffled]
    return img


class ChannelShuffle(ImageOnlyTransform):
    """Randomly rearrange channels of the input RGB image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    @property
    def targets_as_params(self):
        return ["image"]

    def apply(self, img, channels_shuffled=(0, 1, 2), **params):
        return F.channel_shuffle(img, channels_shuffled)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        ch_arr = list(range(img.shape[2]))
        random.shuffle(ch_arr)
        return {"channels_shuffled": ch_arr}
