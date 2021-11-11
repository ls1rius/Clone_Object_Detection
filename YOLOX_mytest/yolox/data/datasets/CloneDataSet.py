#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
import pandas as pd
import string

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
from .clone_classes import CLONE_CLASSES


class CloneDataSet(Dataset):
    """
    clone dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train.csv",
        name="img_clip_all",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.csv_data = pd.read_csv(os.path.join(self.data_dir, self.json_file))
        self._classes = CLONE_CLASSES
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.data = self.load_data()
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.data)

    def __del__(self):
        del self.imgs

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
#         import pdb;pdb.set_trace()
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 2 minutes for CLONE DATA"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.data), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.data)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.data))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.data), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_data(self):
        objs_info = []
        for filename, df_cur in self.csv_data.groupby('filename_gfp'):
            objs = []
            for idx in range(len(df_cur)):
                width, height = int(df_cur.iloc[idx]['size']), int(df_cur.iloc[idx]['size'])
                r = min(self.img_size[0] / height, self.img_size[1] / width)
                x1 = int(df_cur.iloc[idx]['x1'] * r)
                y1 = int(df_cur.iloc[idx]['y1'] * r)
                x2 = int(df_cur.iloc[idx]['x2'] * r)
                y2 = int(df_cur.iloc[idx]['y2'] * r)
                label = self._classes.index(df_cur.iloc[idx]['label'].strip(string.digits))
                if x2 >= x1 and y2 >= y1:
                    objs.append([x1, y1, x2, y2, label])
            objs_info.append([np.array(objs), (width, height, r), filename, int(df_cur.iloc[idx]['image_id'])])
        
        return objs_info
    
    def load_anno(self, index):
        return self.data[index][0]
    
    def load_resized_img(self, index):
#         import pdb;pdb.set_trace()
        # h, w, c
        img = self.load_image(self.data[index][-2])
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, file_name):
#         import pdb;pdb.set_trace()
        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def pull_item(self, index):
#         import pdb;pdb.set_trace()
        res = self.data[index][0]
        width, height, r = self.data[index][1]
        if self.imgs is not None:
            img = self.imgs[index]
#             pad_img = self.imgs[index]
#             img = pad_img[: int(height * r), : int(width * r), :].copy()
        else:
            img = self.load_resized_img(index)
        return img, res.copy(), (height, width), np.array([self.data[index][-1]])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)
#         print('CloneDataSet')
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
