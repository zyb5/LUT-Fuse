import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image= t(image)
        return image


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = F.resize(image, self.size)
        return image

class Resize_16(object):
    def __init__(self):
        pass

    def __call__(self, image):
        width, height = image.size
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        image = F.resize(image, (new_height, new_width))

        return image


class Resize_20(object):
    def __init__(self):
        pass

    def __call__(self, image):
        width, height = image.size
        new_width = (width // 20) * 20
        new_height = (height // 20) * 20

        image = F.resize(image, (new_height, new_width))

        return image


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
        return image


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
        return image


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = pad_if_smaller(image, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        return image

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = F.center_crop(image, self.size)

        return image

class ToTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image