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
                           get_transforms)
from CFG import CFG



class Tester():
    def __init__(self, model, device,
                 img_path, test_anno_path, 
                 batch_size = 16):
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.test_loader, self.test_sampler = self.load_test_loader(img_path, test_anno_path,
                                                                    get_transforms(train=False), None,
                                                                    batch_size)
    
    def load_test_loader(self, img_path, anno_path,
                          transforms, collate_fn,
                          batch_size):
        dataset = CloneDataSet_Test(img_path=img_path, anno_path=anno_path, transforms=transforms)
        
        if CFG.DISTRIBUTED:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = None
        
        loader = DataLoader(dataset=dataset, num_workers=CFG.WORKERS, sampler=sampler,
                            batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        
        return loader, sampler
    
    def test(self):
        self.model.eval()
        res = []
        for idx, batch_data in enumerate(self.test_loader):
            outputs = self.test_one_batch(batch_data)
            res += outputs
            
        pd.DataFrame(data=res, columns=['image_id'] + CLONE_CLASSES).to_csv('./test_res.csv', index=False, sep=',')
        
        
    def test_one_batch(self, batch_data):
        images, image_ids = batch_data
        images = images.to(self.device)
        outputs = self.model(images)
        outputs = torch.softmax(outputs.data.to('cpu'), dim=-1)
        outputs = torch.cat((image_ids.view(-1, 1), outputs), dim=1)
        return outputs.tolist()
        
  