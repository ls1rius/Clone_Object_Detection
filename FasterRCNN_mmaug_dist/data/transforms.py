# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
from CFG import CFG

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        h, w = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        ori_h, ori_w = image.shape[-2:]
        size = self.get_size((ori_h, ori_w))
        image = F.resize(image, size)

        if target is None or 'boxes' not in target.keys():
            return image
        boxes = target['boxes']
        cur_h, cur_w = size
        boxes[:,0] = boxes[:,0] * (cur_w / ori_w)
        boxes[:,2] = boxes[:,2] * (cur_w / ori_w)

        boxes[:,1] = boxes[:,1] * (cur_h / ori_h)
        boxes[:,3] = boxes[:,3] * (cur_h / ori_h)

        target['boxes'] = boxes
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            h, w = image.shape[-2:]
            image = F.hflip(image)
            if target is None or 'boxes' not in target.keys():
                return image
            boxes = target['boxes']
            boxes[:,0], boxes[:,2] = w - boxes[:,2], w - boxes[:,0]
            target['boxes'] = boxes
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            h, w = image.shape[-2:]
            image = F.vflip(image)
            if target is None or 'boxes' not in target.keys():
                return image
            boxes = target['boxes']
            boxes[:,1], boxes[:,3] = h - boxes[:,3], h - boxes[:,1]
            target['boxes'] = boxes
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

    
def build_transforms(is_train=True):
    transform = []
    transform.append(ToTensor())
    transform.append(Resize((CFG.RESIZE), CFG.RESIZE))
    if is_train:
        transform += [
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
        ]
    transform.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return Compose(transform)