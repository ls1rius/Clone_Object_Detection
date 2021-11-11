#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .CloneDataSet import CloneDataSet
from .clone_classes import CLONE_CLASSES
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection

