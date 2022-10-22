import io
import os
import re

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

INSTALL_REQUIRES = ["numpy>=1.11.1", "scipy", "scikit-image>=0.16.1,<0.19", "PyYAML", "qudida>=0.0.4"]


def is_docker() -> bool:
    """
    Check whether setup is running in Docker environment.
    """
    # Note: You have to set the environment variable AM_I_IN_A_DOCKER_CONTAINER manually
    # in your Dockerfile .
    if os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False):
        return True

    path = "/proc/self/cgroup"
    if not os.path.isfile(path):
        return False

    with open(path) as f:
        for line in f:
            if re.match("\\d+:[\\w=]+:/docker(-[ce]e)?/\\w+", line):
                return True

    return False


def is_kaggle() -> bool:
    """
    Check whether setup is running in Kaggle environment.
    This is not 100% bulletproff solution to detect whether we are in Kaggle Notebooks,
    but it should be enough unless Kaggle change their environment variables.
    """
    return (
        ("KAGGLE_CONTAINER_NAME" in os.environ)
        or ("KAGGLE_URL_BASE" in os.environ)
        or ("KAGGLE_DOCKER_IMAGE" in os.environ)
    )


def is_colab() -> bool:
    """
    Check whether setup is running in Google Colab.
    This is not 100% bulletproff solution to detect whether we are in Colab,
    but it should be enough unless Google change their environment variables.
    """
    return (
        ("COLAB_GPU" in os.environ)
        or ("GCE_METADATA_TIMEOUT" in os.environ)
        or ("GCS_READ_CACHE_BLOCK_SIZE_MB" in os.environ)
    )


def get_opencv_requirement():
    """
    Return the OpenCV requirement string.
    Since opencv library is distributed in several independent packages,
    we first check whether any form of opencv is already installed. If not,
    we choose between opencv-python vs opencv-python-headless version based
    on the environment.
    For headless environment (Docker, Colab & Kaggle notebooks), we install
    opencv-python-headless; otherwise - default to opencv-python.
    """
    try:
        import cv2

        return []
    except ImportError:
        default_requirement = "opencv-python>=4.1"
        headless_requirement = "opencv-python-headless>=4.1"

        if is_docker() or is_kaggle() or is_colab():
            return [headless_requirement]
        else:
            return [default_requirement]

def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "augmentoo", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def choose_requirement(mains, secondary):
    """If some version of main requirement installed, return main,
    else return secondary.

    """
    chosen = secondary
    for main in mains:
        try:
            name = re.split(r"[!<>=]", main)[0]
            get_distribution(name)
            chosen = main
            break
        except DistributionNotFound:
            pass

    return str(chosen)


def get_install_requirements(install_requires, choose_install_requires):
    for mains, secondary in choose_install_requires:
        install_requires.append(choose_requirement(mains, secondary))

    return install_requires


setup(
    name="augmentoo",
    version=get_version(),
    description="Data augmentation library for image, masks, boxes, keypoints",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Eugene Khvedchenya",
    license="MIT",
    url="https://github.com/BloodAxe/augmentoo",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES),
    extras_require={"tests": ["pytest"], "imgaug": ["imgaug>=0.4.0"], "develop": ["pytest", "imgaug>=0.4.0"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
