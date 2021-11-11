import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import os
import cv2
import string

from torchvision.transforms import functional as F
from torch.utils.data.sampler import BatchSampler, Sampler

CLONE_CLASSES = ['__background__', 'holoclone', 'meroclone', 'paraclone']

def collate_fn(batch):
    return list(tuple(zip(*batch)))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
        
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
        
def get_transforms(train=True):
    transforms = []
    transforms.append(ToTensor())
#     if train:
#         transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
    
class CloneDataSet(object):
    def __init__(self, img_path, anno_path, transforms):
        self.img_path = img_path
        self.csv_data = pd.read_csv(anno_path)
        self.img_size = 648
        self.data = self.load_data()
        self.transforms = transforms
        
    def load_data(self):
        objs_info = []
        for _, df_cur in self.csv_data.groupby('image_id'):
            objs = []
            filename = df_cur.iloc[0]['filename_gfp']
            image_id = int(df_cur.iloc[0]['image_id'])
            for idx in range(len(df_cur)):
#                 ratio = self.img_size / df_cur.iloc[idx]['size']
                ratio = 1
                x1 = int(df_cur.iloc[idx]['x1'] * ratio)
                y1 = int(df_cur.iloc[idx]['y1'] * ratio)
                x2 = int(df_cur.iloc[idx]['x2'] * ratio)
                y2 = int(df_cur.iloc[idx]['y2'] * ratio)
                label = CLONE_CLASSES.index(df_cur.iloc[idx]['label'].strip(string.digits))
                if x2 >= x1 and y2 >= y1:
                    objs.append([x1, y1, x2, y2, label])
            objs_info.append([np.array(objs), filename, image_id])
        return objs_info
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # h, w, c
        img = cv2.imread(os.path.join(self.img_path, self.data[idx][1]))
#         resized_img = cv2.resize(
#             img,
#             (self.img_size, self.img_size), # (h, w)
#             interpolation=cv2.INTER_LINEAR,
#         ).astype(np.float32)
        resized_img = img.astype(np.float32)
        img = resized_img / 255.

        target = {}
        target['boxes'] = torch.Tensor(self.data[idx][0][:, :4])
        target['labels'] = torch.LongTensor(self.data[idx][0][:, 4].astype(np.int64))
#         target['masks'] = None
        target['image_id'] = torch.LongTensor([self.data[idx][2]])
        
        img, target = self.transforms(img, target)
        
        return img, target
        

        
class CloneDataSet_Test(object):
    def __init__(self, img_path, anno_path, transforms):
        self.img_path = img_path
        self.csv_data = pd.read_csv(anno_path)
        self.img_size = 648
        self.data = self.load_data()
        self.transforms = transforms
        
    def load_data(self):
        objs_info = []
        for _, df_cur in self.csv_data.groupby('image_id'):
            filename = df_cur.iloc[0]['filename_gfp']
            image_id = int(df_cur.iloc[0]['image_id'])
            objs_info.append([filename, image_id])
        return objs_info
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # h,w,c
        img = cv2.imread(os.path.join(self.img_path, self.data[idx][0]))
#         resized_img = cv2.resize(
#             img,
#             (self.img_size, self.img_size), # (h, w)
#             interpolation=cv2.INTER_LINEAR,
#         ).astype(np.float32)
        resized_img = img.astype(np.float32)
        img = resized_img / 255.

        target = {}
        target['image_id'] = torch.LongTensor([self.data[idx][1]])
        
        img, target = self.transforms(img, target)
        
        return img, target