# Augmentoo

Augmentoo is a Python library for image augmentation. Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing datA2.

** This is unofficial fork of the library. Maintained by [Eugene Khvedchenya](https://github.com/BloodAxe) (Ex. Albumentations core team). Use at own risk **


# Important Update

![ukraine-flag](docs/480px-Flag_of_Ukraine.jpg)

On February 24th, 2022, Russia declared war and invaded peaceful Ukraine. 
After the annexation of Crimea and the occupation of the Donbas region, Putin's regime decided to destroy Ukrainian nationality.
Ukrainians show fierce resistance and demonstrate to the entire world what it's like to fight for the nation's independence.

Ukraine's government launched a website to help russian mothers, wives & sisters find their beloved ones killed or captured in Ukraine - https://200rf.com & https://t.me/rf200_now (Telegram channel).
Our goal is to inform those still in Russia & Belarus, so they refuse to assault Ukraine. 

Help us get maximum exposure to what is happening in Ukraine, violence, and inhuman acts of terror that the "Russian World" has brought to Ukraine. 
This is a comprehensive Wiki on how you can help end this war: https://how-to-help-ukraine-now.super.site/ 

Official channels
* [Official account of the Parliament of Ukraine](https://t.me/verkhovnaradaofukraine)
* [Ministry of Defence](https://www.facebook.com/MinistryofDefence.UA)
* [Office of the president](https://www.facebook.com/president.gov.ua)
* [Cabinet of Ministers of Ukraine](https://www.facebook.com/KabminUA)
* [Center of strategic communications](https://www.facebook.com/StratcomCentreUA)
* [Minister of Foreign Affairs of Ukraine](https://twitter.com/DmytroKuleba)

Glory to Ukraine!



Here is an example of how you can apply some augmentations from augmentoo to create new images from the original one:
![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

## Why Albumentations
- Albumentations **[supports all common computer vision tasks](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation)** such as classification, semantic segmentation, instance segmentation, object detection, and pose estimation.
- The library provides **[a simple unified API](#a-simple-example)** to work with all data types: images (RBG-images, grayscale images, multispectral images), segmentation masks, bounding boxes, and keypoints.
- The library contains **[more than 70 different augmentations](#list-of-augmentations)** to generate new training samples from the existing datA2.
- Albumentations is [**fast**](#benchmarking-results). We benchmark each new release to ensure that augmentations provide maximum speed.
- It **[works with popular deep learning frameworks](#i-want-to-know-how-to-use-albumentations-with-deep-learning-frameworks)** such as PyTorch and TensorFlow. By the way, Albumentations is a part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/).
- [**Written by experts**](#authors). The authors have experience both working on production computer vision systems and participating in competitive machine learning. Many core team members are Kaggle Masters and Grandmasters.
- The library is [**widely used**](#who-is-using-albumentations) in industry, deep learning research, machine learning competitions, and open source projects.

## Table of contents
- [Authors](#authors)
- [Installation](#installation)
- [Documentation](#documentation)
- [A simple example](#a-simple-example)
- [Getting started](#getting-started)
  - [I am new to image augmentation](#i-am-new-to-image-augmentation)
  - [I want to use Albumentations for the specific task such as classification or segmentation](#i-want-to-use-albumentations-for-the-specific-task-such-as-classification-or-segmentation)
  - [I want to know how to use Albumentations with deep learning frameworks](#i-want-to-know-how-to-use-albumentations-with-deep-learning-frameworks)
  - [I want to explore augmentations and see Albumentations in action](#i-want-to-explore-augmentations-and-see-albumentations-in-action)
- [Who is using Albumentations](#who-is-using-albumentations)
- [List of augmentations](#list-of-augmentations)
  - [Pixel-level transforms](#pixel-level-transforms)
  - [Spatial-level transforms](#spatial-level-transforms)
- [A few more examples of augmentations](#a-few-more-examples-of-augmentations)
- [Benchmarking results](#benchmarking-results)
- [Contributing](#contributing)
- [Comments](#comments)
- [Citing](#citing)

## Authors
[**Alexander Buslaev** — Computer Vision Engineer at Mapbox](https://www.linkedin.com/in/al-buslaev/) | [Kaggle Master](https://www.kaggle.com/albuslaev)

[**Alex Parinov** — Tech Lead at SberDevices](https://www.linkedin.com/in/alex-parinov/) | [Kaggle Master](https://www.kaggle.com/creafz)

[**Vladimir I. Iglovikov** — Staff Engineer at Lyft Level5](https://www.linkedin.com/in/iglovikov/) | [Kaggle Grandmaster](https://www.kaggle.com/iglovikov)

[**Evegene Khvedchenya** — Computer Vision Research Engineer at Piñata Farms](https://www.linkedin.com/in/cvtalks/) | [Kaggle Grandmaster](https://www.kaggle.com/bloodaxe)

[**Mikhail Druzhinin** — Computer Vision Engineer at ID R&D](https://www.linkedin.com/in/mikhail-druzhinin-548229100/) | [Kaggle Expert](https://www.kaggle.com/dipetm)


## Installation
Albumentations requires Python 3.6 or higher. To install the latest version from PyPI:

```
pip install -U albumentations
```

Other installation options are described in the [documentation](https://augmentoo.ai/docs/getting_started/installation/).

## Documentation
The full documentation is available at **[https://augmentoo.ai/docs/](https://augmentoo.ai/docs/)**.

## A simple example

```python
import augmentoo.augmentations.crops.random_crop
import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A2.Compose([
  augmentoo.augmentations.crops.random_crop.RandomCrop(width=256, height=256),
  A2.HorizontalFlip(p=0.5),
  A2.RandomBrightnessContrast(p=0.2),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]
```

## Getting started

### I am new to image augmentation
Please start with the [introduction articles](https://augmentoo.ai/docs/#introduction-to-image-augmentation) about why image augmentation is important and how it helps to build better models.

### I want to use Albumentations for the specific task such as classification or segmentation
If you want to use Albumentations for a specific task such as classification, segmentation, or object detection, refer to the [set of articles](https://augmentoo.ai/docs/#getting-started-with-albumentations) that has an in-depth description of this task. We also have a [list of examples](https://augmentoo.ai/docs/examples/) on applying Albumentations for different use cases.

### I want to know how to use Albumentations with deep learning frameworks
We have [examples of using Albumentations](https://augmentoo.ai/docs/#examples-of-how-to-use-albumentations-with-different-deep-learning-frameworks) along with PyTorch and TensorFlow.

### I want to explore augmentations and see Albumentations in action
Check the [online demo of the library](https://albumentations-demo.herokuapp.com/). With it, you can apply augmentations to different images and see the result. Also, we have a [list of all available augmentations and their targets](#list-of-augmentations).

## Who is using Albumentations
<a href="https://www.lyft.com/" target="_blank"><img src="https://habrastorage.org/webt/ce/bs/sa/cebssajf_5asst5yshmyykqjhcg.png" width="100"/></a>
<a href="https://sberdevices.ru/" target="_blank"><img src="https://habrastorage.org/webt/3b/jz/pv/3bjzpvcmvdvdmmeer51y89noank.png" width="100"/></a>
<a href="https://www.x5.ru/en" target="_blank"><img src="https://habrastorage.org/webt/9y/dv/f1/9ydvf1fbxotkl6nyhydrn9v8cqw.png" width="100"/></a>
<a href="https://imedhub.org/" target="_blank"><img src="https://habrastorage.org/webt/eq/8x/m-/eq8xm-fjfx_uqkka4_ekxsdwtiq.png" width="100"/></a>
<a href="https://recursionpharmA2.com" target="_blank"><img src="https://pbs.twimg.com/profile_images/925897897165639683/jI8YvBfC_400x400.jpg" width="100"/></a>
<a href="https://www.everypixel.com/" target="_blank"><img src="https://www.everypixel.com/i/logo_sq.png" width="100"/></a>
<a href="https://neuromation.io/" target="_blank"><img src="https://habrastorage.org/webt/yd/_4/xa/yd_4xauvggn1tuz5xgrtkif6lyA2.png" width="100"/></a>
<a href="https://ultralytics.com/" target="_blank"><img src="https://augmentoo.ai/assets/img/industry/ultralytics.png" width="100"/></a>
<a href="https://www.cft.ru/" target="_blank"><img src="https://habrastorage.org/webt/dv/fa/uq/dvfauqyl5cor5yzrfrpthjzm0mi.jpeg" width="100"/></a>
<a href="https://www.pinatafarm.com/" target="_blank"><img src="https://www.pinatafarm.com/pfLogo.png" width="100"/></a>
<a href="https://incode.com/" target="_blank"><img src="https://habrastorage.org/webt/sh/eg/bs/shegbsyzy-0lebwqxkgl_rkkx3m.png" width="100"/></a>
<a href="https://sharpershape.com/" target="_blank"><img src="https://lh3.googleusercontent.com/pw/AM-JKLWe2-aRXcZMqZOnL67Gw8v46LTwJw5a6RyufgAiLCKncxSI4U7wzHopt5Lacbc4wpDF7uJYMrWcVXPK-3Z3cxopV9jmtriuWSdzNpAO6gDC963nPd3BrWlE6eFwstLCb4il6lYXT49BbamdUipZrLk=w1870-h1574-no?authuser=0" width="100"/></a>
<a href="https://vitechlab.com/" target="_blank"><img src="https://res2.weblium.site/res/5f842a47d2077f0022e59f1d/5f842ba81ff15b00214a447f_optimized_389.webp" width="100"/></a>
<a href="https://borzodelivery.com/" target="_blank"><img src="https://borzodelivery.com/img/global/big-logo.svg" width="100"/></a>
<a href="https://anadeA2.info/" target="_blank"><img src="https://habrastorage.org/webt/oc/lt/8u/oclt8uwyyc-vgmwwcgcsk5cw7wy.png" width="100"/></a>
<a href="https://www.idrnd.ai/" target="_blank"><img src="https://www.idrnd.ai/wp-content/uploads/2019/09/Logo-IDRND.png.webp" width="100"/></a>
<a href="https://openface.me/en/" target="_blank"><img src="https://drive.google.com/uc?export=view&id=1mC8B55CPFlpUC69Wnli2vitp6pImIfz7" width="100"/></a>

#### See also:
- [A list of papers that cite Albumentations](https://augmentoo.ai/whos_using#research).
- [A list of teams that were using Albumentations and took high places in machine learning competitions](https://augmentoo.ai/whos_using#competitions).
- [Open source projects that use Albumentations](https://augmentoo.ai/whos_using#open-source).

## List of augmentations

### Pixel-level transforms
Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. The list of pixel-level transforms:

- [AdvancedBlur](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.AdvancedBlur)
- [Blur](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Blur)
- [CLAHE](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.CLAHE)
- [ChannelDropout](https://augmentoo.ai/docs/api_reference/augmentations/dropout/channel_dropout/#augmentoo.augmentations.dropout.channel_dropout.ChannelDropout)
- [ChannelShuffle](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.ChannelShuffle)
- [ColorJitter](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.ColorJitter)
- [Downscale](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Downscale)
- [Emboss](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Emboss)
- [Equalize](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Equalize)
- [FDA](https://augmentoo.ai/docs/api_reference/augmentations/domain_adaptation/#augmentoo.augmentations.domain_adaptation.FDA)
- [FancyPCA](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.FancyPCA)
- [FromFloat](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.FromFloat)
- [GaussNoise](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.GaussNoise)
- [GaussianBlur](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.GaussianBlur)
- [GlassBlur](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.GlassBlur)
- [HistogramMatching](https://augmentoo.ai/docs/api_reference/augmentations/domain_adaptation/#augmentoo.augmentations.domain_adaptation.HistogramMatching)
- [HueSaturationValue](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.HueSaturationValue)
- [ISONoise](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.ISONoise)
- [ImageCompression](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.ImageCompression)
- [InvertImg](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.InvertImg)
- [MedianBlur](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.MedianBlur)
- [MotionBlur](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.MotionBlur)
- [MultiplicativeNoise](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.MultiplicativeNoise)
- [Normalize](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Normalize)
- [PixelDistributionAdaptation](https://augmentoo.ai/docs/api_reference/augmentations/domain_adaptation/#augmentoo.augmentations.domain_adaptation.PixelDistributionAdaptation)
- [Posterize](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Posterize)
- [RGBShift](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RGBShift)
- [RandomBrightnessContrast](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RandomBrightnessContrast)
- [RandomFog](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RandomFog)
- [RandomGamma](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RandomGamma)
- [RandomRain](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RandomRain)
- [RandomShadow](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RandomShadow)
- [RandomSnow](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RandomSnow)
- [RandomSunFlare](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RandomSunFlare)
- [RandomToneCurve](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RandomToneCurve)
- [RingingOvershoot](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RingingOvershoot)
- [Sharpen](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Sharpen)
- [Solarize](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Solarize)
- [Superpixels](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Superpixels)
- [TemplateTransform](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.TemplateTransform)
- [ToFloat](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.ToFloat)
- [ToGray](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.ToGray)
- [ToSepia](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.ToSepia)
- [UnsharpMask](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.UnsharpMask)

### Spatial-level transforms
Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. The following table shows which additional targets are supported by each transform.

| Transform                                                                                                                                                                       | Image | Masks | BBoxes | Keypoints |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :---: | :----: | :-------: |
| [Affine](https://augmentoo.ai/docs/api_reference/augmentations/geometric/transforms/#augmentoo.augmentations.geometric.transforms.Affine)                             | ✓     | ✓     | ✓      | ✓         |
| [CenterCrop](https://augmentoo.ai/docs/api_reference/augmentations/crops/transforms/#augmentoo.augmentations.crops.transforms.CenterCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [CoarseDropout](https://augmentoo.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#augmentoo.augmentations.dropout.coarse_dropout.CoarseDropout)           | ✓     | ✓     |        | ✓         |
| [Crop](https://augmentoo.ai/docs/api_reference/augmentations/crops/transforms/#augmentoo.augmentations.crops.transforms.Crop)                                         | ✓     | ✓     | ✓      | ✓         |
| [CropAndPad](https://augmentoo.ai/docs/api_reference/augmentations/crops/transforms/#augmentoo.augmentations.crops.transforms.CropAndPad)                             | ✓     | ✓     | ✓      | ✓         |
| [CropNonEmptyMaskIfExists](https://augmentoo.ai/docs/api_reference/augmentations/crops/transforms/#augmentoo.augmentations.crops.transforms.CropNonEmptyMaskIfExists) | ✓     | ✓     | ✓      | ✓         |
| [ElasticTransform](https://augmentoo.ai/docs/api_reference/augmentations/geometric/transforms/#augmentoo.augmentations.geometric.transforms.ElasticTransform)         | ✓     | ✓     |        |           |
| [Flip](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Flip)                                                     | ✓     | ✓     | ✓      | ✓         |
| [GridDistortion](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.GridDistortion)                                 | ✓     | ✓     |        |           |
| [GridDropout](https://augmentoo.ai/docs/api_reference/augmentations/dropout/grid_dropout/#augmentoo.augmentations.dropout.grid_dropout.GridDropout)                   | ✓     | ✓     |        |           |
| [HorizontalFlip](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.HorizontalFlip)                                 | ✓     | ✓     | ✓      | ✓         |
| [Lambda](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Lambda)                                                 | ✓     | ✓     | ✓      | ✓         |
| [LongestMaxSize](https://augmentoo.ai/docs/api_reference/augmentations/geometric/resize/#augmentoo.augmentations.geometric.resize.LongestMaxSize)                     | ✓     | ✓     | ✓      | ✓         |
| [MaskDropout](https://augmentoo.ai/docs/api_reference/augmentations/dropout/mask_dropout/#augmentoo.augmentations.dropout.mask_dropout.MaskDropout)                   | ✓     | ✓     |        |           |
| [NoOp](https://augmentoo.ai/docs/api_reference/core/transforms_interface/#augmentoo.core.transforms_interface.NoOp)                                                   | ✓     | ✓     | ✓      | ✓         |
| [OpticalDistortion](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.OpticalDistortion)                           | ✓     | ✓     |        |           |
| [PadIfNeeded](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.PadIfNeeded)                                       | ✓     | ✓     | ✓      | ✓         |
| [Perspective](https://augmentoo.ai/docs/api_reference/augmentations/geometric/transforms/#augmentoo.augmentations.geometric.transforms.Perspective)                   | ✓     | ✓     | ✓      | ✓         |
| [PiecewiseAffine](https://augmentoo.ai/docs/api_reference/augmentations/geometric/transforms/#augmentoo.augmentations.geometric.transforms.PiecewiseAffine)           | ✓     | ✓     | ✓      | ✓         |
| [PixelDropout](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.PixelDropout)                                     | ✓     | ✓     | ✓      | ✓         |
| [RandomCrop](https://augmentoo.ai/docs/api_reference/augmentations/crops/transforms/#augmentoo.augmentations.crops.transforms.RandomCrop)                             | ✓     | ✓     | ✓      | ✓         |
| [RandomCropNearBBox](https://augmentoo.ai/docs/api_reference/augmentations/crops/transforms/#augmentoo.augmentations.crops.transforms.RandomCropNearBBox)             | ✓     | ✓     | ✓      | ✓         |
| [RandomGridShuffle](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.RandomGridShuffle)                           | ✓     | ✓     |        | ✓         |
| [RandomResizedCrop](https://augmentoo.ai/docs/api_reference/augmentations/crops/transforms/#augmentoo.augmentations.crops.transforms.RandomResizedCrop)               | ✓     | ✓     | ✓      | ✓         |
| [RandomRotate90](https://augmentoo.ai/docs/api_reference/augmentations/geometric/rotate/#augmentoo.augmentations.geometric.rotate.RandomRotate90)                     | ✓     | ✓     | ✓      | ✓         |
| [RandomScale](https://augmentoo.ai/docs/api_reference/augmentations/geometric/resize/#augmentoo.augmentations.geometric.resize.RandomScale)                           | ✓     | ✓     | ✓      | ✓         |
| [RandomSizedBBoxSafeCrop](https://augmentoo.ai/docs/api_reference/augmentations/crops/transforms/#augmentoo.augmentations.crops.transforms.RandomSizedBBoxSafeCrop)   | ✓     | ✓     | ✓      |           |
| [RandomSizedCrop](https://augmentoo.ai/docs/api_reference/augmentations/crops/transforms/#augmentoo.augmentations.crops.transforms.RandomSizedCrop)                   | ✓     | ✓     | ✓      | ✓         |
| [Resize](https://augmentoo.ai/docs/api_reference/augmentations/geometric/resize/#augmentoo.augmentations.geometric.resize.Resize)                                     | ✓     | ✓     | ✓      | ✓         |
| [Rotate](https://augmentoo.ai/docs/api_reference/augmentations/geometric/rotate/#augmentoo.augmentations.geometric.rotate.Rotate)                                     | ✓     | ✓     | ✓      | ✓         |
| [SafeRotate](https://augmentoo.ai/docs/api_reference/augmentations/geometric/rotate/#augmentoo.augmentations.geometric.rotate.SafeRotate)                             | ✓     | ✓     | ✓      | ✓         |
| [ShiftScaleRotate](https://augmentoo.ai/docs/api_reference/augmentations/geometric/transforms/#augmentoo.augmentations.geometric.transforms.ShiftScaleRotate)         | ✓     | ✓     | ✓      | ✓         |
| [SmallestMaxSize](https://augmentoo.ai/docs/api_reference/augmentations/geometric/resize/#augmentoo.augmentations.geometric.resize.SmallestMaxSize)                   | ✓     | ✓     | ✓      | ✓         |
| [Transpose](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.Transpose)                                           | ✓     | ✓     | ✓      | ✓         |
| [VerticalFlip](https://augmentoo.ai/docs/api_reference/augmentations/transforms/#augmentoo.augmentations.transforms.VerticalFlip)                                     | ✓     | ✓     | ✓      | ✓         |


## A few more examples of augmentations
### Semantic segmentation on the Inria dataset

![inria](https://habrastorage.org/webt/su/wa/np/suwanpeo6ww7wpwtobtrzd_cg20.jpeg)

### Medical imaging
![medical](https://habrastorage.org/webt/1i/fi/wz/1ifiwzy0lxetc4nwjvss-71nkw0.jpeg)

### Object detection and semantic segmentation on the Mapillary Vistas dataset
![vistas](https://habrastorage.org/webt/rz/-h/3j/rz-h3jalbxic8o_fhucxysts4tc.jpeg)

### Keypoints augmentation
<img src="https://habrastorage.org/webt/e-/6k/z-/e-6kz-fugp2heak3jzns3bc-r8o.jpeg" width=100%>


## Benchmarking results
To run the benchmark yourself, follow the instructions in [benchmark/README.md](https://github.com/albumentations-team/albumentations/blob/master/benchmark/README.md)

Results for running the benchmark on the first 2000 images from the ImageNet validation set using an Intel(R) Xeon(R) Gold 6140 CPU.
All outputs are converted to a contiguous NumPy array with the np.uint8 data type.
The table shows how many images per second can be processed on a single core; higher is better.


|                      |albumentations<br><small>1.1.0</small>|imgaug<br><small>0.4.0</small>|torchvision (Pillow-SIMD backend)<br><small>0.10.1</small>|keras<br><small>2.6.0</small>|augmentor<br><small>0.2.8</small>|solt<br><small>0.1.9</small>|
|----------------------|:------------------------------------:|:----------------------------:|:--------------------------------------------------------:|:---------------------------:|:-------------------------------:|:--------------------------:|
|HorizontalFlip        |              **10220**               |             2702             |                           2517                           |             876             |              2528               |            6798            |
|VerticalFlip          |               **4438**               |             2141             |                           2151                           |            4381             |              2155               |            3659            |
|Rotate                |               **389**                |             283              |                           165                            |             28              |               60                |            367             |
|ShiftScaleRotate      |               **669**                |             425              |                           146                            |             29              |                -                |             -              |
|Brightness            |               **2765**               |             1124             |                           411                            |             229             |               408               |            2335            |
|Contrast              |               **2767**               |             1137             |                           349                            |              -              |               346               |            2341            |
|BrightnessContrast    |               **2746**               |             629              |                           190                            |              -              |               189               |            1196            |
|ShiftRGB              |               **2758**               |             1093             |                            -                             |             360             |                -                |             -              |
|ShiftHSV              |               **598**                |             259              |                            59                            |              -              |                -                |            144             |
|Gamma                 |               **2849**               |              -               |                           388                            |              -              |                -                |            933             |
|Grayscale             |               **5219**               |             393              |                           723                            |              -              |              1082               |            1309            |
|RandomCrop64          |              **163550**              |             2562             |                          50159                           |              -              |              42842              |           22260            |
|PadToSize512          |               **3609**               |              -               |                           602                            |              -              |                -                |            3097            |
|Resize512             |                 1049                 |             611              |                         **1066**                         |              -              |              1041               |            1017            |
|RandomSizedCrop_64_512|               **3224**               |             858              |                           1660                           |              -              |              1598               |            2675            |
|Posterize             |               **2789**               |              -               |                            -                             |              -              |                -                |             -              |
|Solarize              |               **2761**               |              -               |                            -                             |              -              |                -                |             -              |
|Equalize              |                 647                  |             385              |                            -                             |              -              |             **765**             |             -              |
|Multiply              |               **2659**               |             1129             |                            -                             |              -              |                -                |             -              |
|MultiplyElementwise   |                 111                  |           **200**            |                            -                             |              -              |                -                |             -              |
|ColorJitter           |               **351**                |              78              |                            57                            |              -              |                -                |             -              |

Python and library versions: Python 3.9.5 (default, Jun 23 2021, 15:01:51) [GCC 8.3.0], numpy 1.19.5, pillow-simd 7.0.0.post3, opencv-python 4.5.3.56, scikit-image 0.18.3, scipy 1.7.1.

## Contributing

To create a pull request to the repository, follow the documentation at [https://augmentoo.ai/docs/contributing/](https://augmentoo.ai/docs/contributing/)


## Citing

If you find this library useful for your research, please consider citing [Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125):

```bibtex
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A2.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```
