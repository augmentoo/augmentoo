__version__ = "1.2.0"

from .core.pipeline import Pipeline
from .core.targets.image import ImageTarget
from .core.targets.mask import MaskTarget
from .core.targets.bbox import AxisAlignedBoxTarget, BoundingBoxFormat
from .core.targets.keypoint import KeypointTarget

from .augmentations import *
