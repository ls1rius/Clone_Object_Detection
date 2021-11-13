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
                            get_transforms,
                            collate_fn)
from data.utils import calc_mAP
from CFG import CFG
import time
import torch.distributed as dist
from utils import get_rank, synchronize, gather, is_main_process
import itertools

class Trainer():
    def __init__(self, model, optimizer, device,
                 img_path, train_anno_path, eval_anno_path, 
                 batch_size = 16):
        self.rank = get_rank()
        self.device = device
        self.train_loader = None
        self.eval_loader = None
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.train_loader, self.train_sampler = self.load_loader(img_path, train_anno_path, 
                                                   get_transforms(train=True), collate_fn,
                                                   batch_size, train=True)
        self.eval_loader, self.eval_sampler = self.load_loader(img_path, eval_anno_path,
                                                  get_transforms(train=False), collate_fn,
                                                  batch_size, train=False)
        self.cur_epoch = 0
        self.epoch = 300
        self.beta = 1
        self.best_metric_dict = {}
        self.pth_path = time.strftime("%Y%m%d_%H%M_%S", time.localtime())
        if is_main_process():
            if not os.path.exists(self.pth_path):
                os.mkdir(self.pth_path)
    
    def load_loader(self, img_path, anno_path, 
                         transforms, collate_fn,
                         batch_size, train=False):
        dataset = CloneDataSet(img_path=img_path, anno_path=anno_path, transforms=transforms)
        
        if CFG.DISTRIBUTED:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            batch_size = batch_size // dist.get_world_size()
        else:
            sampler = None
        
        loader = DataLoader(dataset=dataset, num_workers=CFG.WORKERS, sampler=sampler,
                            batch_size=batch_size, collate_fn=collate_fn, shuffle=((train is True) and (sampler is None))) 
        
        return loader, sampler

    def train(self):
        for self.cur_epoch in range(self.epoch):
            
            if is_main_process():
                print('epoch: {}/{}:'.format(self.cur_epoch+1, self.epoch))
                
            if CFG.DISTRIBUTED:
                self.train_sampler.set_epoch(self.cur_epoch)
            
            train_info = self.train_one_epoch()
            eval_info, metric_info_dict = self.eval_one_epoch()
            
            synchronize()
            
            if is_main_process():
                print(train_info)
                if (self.cur_epoch + 1) % 1 == 0:
                    print(eval_info)
                    
                # save pth
                for k, v in metric_info_dict.items():
                    best_metric_info_key = 'best_' + k
                    if best_metric_info_key not in self.best_metric_dict.keys() or v > self.best_metric_dict[best_metric_info_key]:
                        self.best_metric_dict.update({best_metric_info_key: v})
                        torch.save({'epoch': self.cur_epoch + 1,
                                     'state_dict': self.model.state_dict(),
                                     best_metric_info_key: round(v*100, 2),
                                     'optimizer': self.optimizer.state_dict()
                                   },
                                   './' + self.pth_path + '/model_' + best_metric_info_key + '.pth')
                    
        
    def train_one_epoch(self):
        self.model.train()
        loss_info_all = None
        for idx, batch_data in enumerate(self.train_loader):
            loss_info = self.train_one_batch(batch_data)
            
            if loss_info_all is None:
                loss_info_all = loss_info
            else:
                for k,v in loss_info.items(): 
                    loss_info_all[k] += v
            
        losses = 0
        info = 'train:\n'
        for k, v in loss_info_all.items():
            info = info + 'loss_{}: {} \n'.format(k, v/len(self.train_loader))
            losses += v/len(self.train_loader)
        info = info + 'total_loss: {}\n'.format(losses)
        
        return info
        
    def train_one_batch(self, batch_data):
#         import pdb;pdb.set_trace()
        images, targets = batch_data
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_info = {k:v.data.to('cpu').numpy() for k,v in loss_dict.items()}
        
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        
        return loss_info
        
    def eval_one_epoch(self):
        self.model.eval()
        metric_info_list = ['95', '75', '50', '25', '00']
        metric_res_list = []
        
        for idx, batch_data in enumerate(self.eval_loader):
            metric_res_list_cur = self.eval_one_batch(batch_data, metric_info_list)
            # 数据记录
            metric_res_list.append(metric_res_list_cur)
            
        if CFG.DISTRIBUTED:
            metric_res_list = gather(metric_res_list, dst=0)
            metric_res_list = list(itertools.chain(*metric_res_list))
#         import pdb;pdb.set_trace()
        metric_res_list = np.array(metric_res_list).sum(0)
            
        info = 'eval:\n'
        metric_info_dict = {}
        
        if is_main_process():
            for idx, (metric_res, metric_info) in enumerate(zip(metric_res_list, metric_info_list)):
                mAP = metric_res[0] / (metric_res[0] + metric_res[1] + 1e-8)
                mAR = metric_res[0] / (metric_res[0] + metric_res[2] + 1e-8)
                mFS = (1 + self.beta**2)*(mAP*mAR)/((self.beta**2)*mAP + mAR + 1e-8)
                info = info + ('correct50_' + metric_info + \
                        ':{}, error50_' + metric_info + \
                        ':{}, miss50_' + metric_info + \
                        ':{}, mAP50_' + metric_info + \
                        ':{}, mAR50_' + metric_info + \
                        ':{}, mFS50_' + metric_info + \
                        ':{} \n').format(metric_res[0], metric_res[1], metric_res[2], mAP, mAR, mFS)
                metric_info_dict.update({'mAP50_' + metric_info: mAP})
                metric_info_dict.update({'mAR50_' + metric_info: mAR})
                metric_info_dict.update({'mFS50_' + metric_info: mFS})
            
        synchronize()
        return info, metric_info_dict
        
    def eval_one_batch(self, batch_data, metric_info_list):
#         import pdb;pdb.set_trace()
        images, targets = batch_data
        images = list(image.to(self.device) for image in images)
        outputs = self.model(images)
        outputs = [{k: v.data.to('cpu').numpy() for k, v in t.items()} for t in outputs]
        targets = [{k: v.data.to('cpu').numpy() for k, v in t.items()} for t in targets]
        
        df_dt = pd.DataFrame.from_dict(outputs)
        df_gt = pd.DataFrame.from_dict(targets)
        df_dt['image_id'] = df_gt['image_id']
        
        data_list = []
        box_cnt = 0
        for i in range(len(df_dt)):
            for j in range(len(df_dt.iloc[i]['labels'])):
                x1, y1, x2, y2 = df_dt.iloc[i]['boxes'][j]
                data_list.append([x1, y1, x2, y2, 
                                  df_dt.iloc[i]['labels'][j], 
                                  box_cnt, 
                                  int(df_dt.iloc[i]['image_id'][0]),
                                  df_dt.iloc[i]['scores'][j]])
                box_cnt += 1
        df_dt_deal = pd.DataFrame(data=data_list, 
                                  columns=['x1_dt', 'y1_dt', 'x2_dt', 'y2_dt', 
                                           'label_dt', 
                                           'bbox_idx_dt', 
                                           'image_id',
                                           'scores'])
        
#         import pdb;pdb.set_trace()
        data_list = []
        box_cnt = 0
        for i in range(len(df_gt)):
            for j in range(len(df_gt.iloc[i]['labels'])):
                x1, y1, x2, y2 = df_gt.iloc[i]['boxes'][j]
                data_list.append([x1, y1, x2, y2, 
                                  df_gt.iloc[i]['labels'][j], 
                                  box_cnt, 
                                  int(df_gt.iloc[i]['image_id'][0])])
                box_cnt += 1
        df_gt_deal = pd.DataFrame(data=data_list, 
                                  columns=['x1_gt', 'y1_gt', 'x2_gt', 'y2_gt', 
                                           'label_gt', 
                                           'bbox_idx_gt', 
                                           'image_id'])
        
#         import pdb;pdb.set_trace()
        metric_res_list = []
        for metric_info in metric_info_list:
            metric_res = calc_mAP(df_dt_deal[df_dt_deal['scores']>=(float(metric_info)/100.0)], df_gt_deal)
            metric_res_list.append(list(metric_res))
            
        return metric_res_list

