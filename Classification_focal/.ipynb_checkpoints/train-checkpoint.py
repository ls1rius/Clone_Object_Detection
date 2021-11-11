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
                           CloneDataSet,
                           get_transforms)
from CFG import CFG
import time

class Trainer():
    def __init__(self, model, optimizer, loss_func, device,
                 img_path, train_anno_path, eval_anno_path, 
                 batch_size = 16):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.train_loader, self.train_sampler = self.load_loader(img_path, train_anno_path, 
                                                                 get_transforms(train=True), None,
                                                                 batch_size)
        self.eval_loader, self.eval_sampler = self.load_loader(img_path, eval_anno_path,
                                                               get_transforms(train=False), None,
                                                               batch_size)
        self.cur_epoch = 0
        self.epoch = 300
        self.best_acc = 0
        self.pth_path = time.strftime("%Y%m%d_%H%M_%S", time.localtime())
        
        if not os.path.exists(self.pth_path):
            os.mkdir(self.pth_path)
    
    def load_loader(self, img_path, anno_path, 
                         transforms, collate_fn,
                         batch_size):
        dataset = CloneDataSet(img_path=img_path, anno_path=anno_path, transforms=transforms)
        
        if CFG.DISTRIBUTED:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = None
        
        loader = DataLoader(dataset=dataset, num_workers=CFG.WORKERS, sampler=sampler,
                            batch_size=batch_size, collate_fn=collate_fn, 
                            shuffle=(CFG.DISTRIBUTED is None) or (sampler is None)) 
        
        return loader, sampler

    def train(self):
        for self.cur_epoch in range(self.epoch):
            print('epoch: {}/{}:'.format(self.cur_epoch+1, self.epoch))
            if CFG.DISTRIBUTED:
                self.train_sampler.set_epoch(self.cur_epoch)
            
            train_info = self.train_one_epoch()
            eval_info, acc = self.eval_one_epoch()
            
            print(train_info)
            if (self.cur_epoch + 1) % 1==0:
                print(eval_info)
                
            if acc > self.best_acc:
                self.best_acc = acc
                
                if CFG.DISTRIBUTED:
                    torch.cuda.synchornize()
                
                if not CFG.DISTRIBUTED or torch.distributed.get_rank() == 0:
                    torch.save({'epoch': self.cur_epoch + 1,
                                 'state_dict': self.model.state_dict(),
                                 'best_acc': round(self.best_acc*100, 2),
                                 'optimizer': self.optimizer.state_dict()
                               },
                               './' + self.pth_path + '/model_best_acc.pth')

    def train_one_epoch(self):
        self.model.train()
        losses = 0
        for idx, batch_data in enumerate(self.train_loader):
            losses += self.train_one_batch(batch_data)
        info = 'train:\n'
        info = info + 'total_loss: {}\n'.format(losses/len(self.train_loader))
        return info
        
    def train_one_batch(self, batch_data):
#         import pdb;pdb.set_trace()
        images, targets, image_ids = batch_data
        images, targets = images.to(self.device), targets.to(self.device)
        outputs = self.model(images)
        loss = self.loss_func(outputs, targets)
        loss_info = loss.data.to('cpu').numpy()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss_info
        
    def eval_one_epoch(self):
        self.model.eval()
        res_list = []
        for idx, batch_data in enumerate(self.eval_loader):
            res = self.eval_one_batch(batch_data)
            res_list.append(res)
        
#         import pdb;pdb.set_trace()
        res_list = np.array(res_list)
        correct, total = res_list.sum(0)
        acc = correct / (total + 1e-8)
        
        info = 'eval:\n'
        info = info + 'correct:{}, total:{}, accuracy:{} \n'.format(correct, total, acc)
        
        return info, acc
        
    def eval_one_batch(self, batch_data):
        images, targets, image_ids = batch_data
        outputs = self.model(images.to(self.device))
        correct = int((torch.argmax(outputs.data.to('cpu'), dim=-1) == targets).sum())
        return [correct, len(targets)]

