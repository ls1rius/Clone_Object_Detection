import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import os
import cv2
import string
from torch.utils.data import DataLoader
from data.datasets import (CLONE_CLASSES,
                            CloneDataSet_Test,
                            get_transforms,
                            collate_fn)
from data.utils import calc_mAP
from CFG import CFG
import torch.distributed as dist
from utils import get_rank, synchronize, gather, is_main_process
import itertools

class Tester():
    def __init__(self, model, device,
                 img_path, test_anno_path, 
                 batch_size = 16):
        self.device = device
        self.test_loader = None
        self.model = model
        self.batch_size = batch_size
        self.test_loader, self.test_sampler = self.load_test_loader(img_path, test_anno_path,
                                                  get_transforms(train=False), collate_fn,
                                                  batch_size)
    
    def load_test_loader(self, img_path, anno_path,
                          transforms, collate_fn,
                          batch_size):
        dataset = CloneDataSet_Test(img_path=img_path, anno_path=anno_path, transforms=transforms)
        
        if CFG.DISTRIBUTED:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            batch_size = batch_size // dist.get_world_size()
        else:
            sampler = None
        
        loader = DataLoader(dataset=dataset, num_workers=CFG.WORKERS, sampler=sampler,
                            batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        
        return loader, sampler
    
    def test(self):
        self.model.eval()
        df_res = None
        for idx, batch_data in enumerate(self.test_loader):
            df_dt_deal = self.test_one_batch(batch_data)
            if df_res is None:
                df_res = df_dt_deal
            else:
                df_res = pd.concat((df_res, df_dt_deal))
                    
        if CFG.DISTRIBUTED:
            df_res = gather(df_res, dst=0)
            if len(df_res) > 0:
                df_res = pd.concat(df_res)
            
        synchronize()
        
        if is_main_process():
            df_res.to_csv('./test_res.csv', index=False, sep=',')
            
        
        
    def test_one_batch(self, batch_data):
#         import pdb;pdb.set_trace()
        images, targets = batch_data
        images = list(image.to(self.device) for image in images)
        outputs = self.model(images)
        outputs = [{k: v.data.to('cpu').numpy() for k, v in t.items()} for t in outputs]
        df_dt = pd.DataFrame.from_dict(outputs)
        df_dt['image_id'] = pd.DataFrame.from_dict(targets)['image_id']
#         resize = self.test_loader.dataset.img_size
        resize = CFG.RESIZE
        data_list = []
        for i in range(len(df_dt)):
            for j in range(len(df_dt.iloc[i]['labels'])):
                x1, y1, x2, y2 = df_dt.iloc[i]['boxes'][j]
                data_list.append([x1, y1, x2, y2, 
                                  df_dt.iloc[i]['labels'][j], 
                                  int(df_dt.iloc[i]['image_id'][0]),
                                  resize,
                                  df_dt.iloc[i]['scores'][j]])
        df_dt_deal = pd.DataFrame(data=data_list, 
                                  columns=['x1_dt', 'y1_dt', 'x2_dt', 'y2_dt', 
                                           'label_dt', 
                                           'image_id',
                                           'resize',
                                           'scores'])
        return df_dt_deal
        
  