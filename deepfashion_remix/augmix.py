import math
import numpy as np
from PIL import ImageOps, Image, ImageEnhance
import random
import torch
import torchvision.transforms as transforms

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.

def sample_level(n):
    return np.random.uniform(low=0.1, high=n)

def rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)

def autocontrast(pil_img, level=None):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, level=None):
    return ImageOps.equalize(pil_img)

def rotate(pil_img, level):
    degrees = int_parameter(rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)

def solarize(pil_img, level):
    level = int_parameter(rand_lvl(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((256, 256), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def shear_y(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((256, 256), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_x(pil_img, level):
    level = int_parameter(rand_lvl(level), 256 / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((256, 256), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_y(pil_img, level):
    level = int_parameter(rand_lvl(level), 256 / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((256, 256), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR, fillcolor=128)

def posterize(pil_img, level):
    level = int_parameter(rand_lvl(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)

# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


class AugMix(torch.utils.data.Dataset):
    def __init__(self, dataset, severity=None):
        self.dataset = dataset
        self.aug_severity = severity
        self.aug_width = 5
        self.alpha = 1
        self.augmentations = [
            autocontrast,
            equalize,
            lambda x: rotate(x,      self.aug_severity),
            lambda x: solarize(x,    self.aug_severity),
            lambda x: shear_x(x,     self.aug_severity),
            lambda x: shear_y(x,     self.aug_severity),
            lambda x: translate_x(x, self.aug_severity),
            lambda x: translate_y(x, self.aug_severity),
            lambda x: posterize(x,   self.aug_severity),
            lambda x: color(x,       self.aug_severity),
            lambda x: contrast(x,    self.aug_severity),
            lambda x: brightness(x,  self.aug_severity),
            lambda x: sharpness(x,   self.aug_severity),
        ]
        
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


    def __getitem__(self, i):
        x_orig, y = self.dataset[i]

        if type(x_orig) == list:
            x_orig = x_orig[0]
        x_processed = self.preprocess(x_orig)
        mix1 = self.get_mixture(x_orig, x_processed)
        mix2 = self.get_mixture(x_orig, x_processed)

        # done so that on 2 GPUs, there is an equal split of clean images for each GPU's batch norm
        return [mix1, x_processed], y

    def __len__(self):
        return len(self.dataset)


    def get_mixture(self, x_orig, x_processed):
        if self.aug_width > 1:
            w = np.float32(np.random.dirichlet([self.alpha] * self.aug_width))
        else:
            w = [1.]
        m = np.float32(np.random.beta(self.alpha, self.alpha))
    
        mix = torch.zeros_like(x_processed)
        for i in range(self.aug_width):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.augmentations)(x_aug)
            mix += w[i] * self.preprocess(x_aug)
        mix = m * x_processed + (1 - m) * mix
        return mix


