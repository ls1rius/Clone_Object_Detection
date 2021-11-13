from train import Trainer
from test import Tester
import torch
import torchvision
from data.datasets import CLONE_CLASSES
import os
from CFG import CFG
import argparse
from models.detection import fasterrcnn_resnet_fpn, fasterrcnn_dcnresnet_fpn

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help='node rank for distributed training')
#     parser.add_argument("--world_size", type=int, default=-1)
    args = parser.parse_args()

    if CFG.DISTRIBUTED:
        torch.distributed.init_process_group(backend='nccl')
#         torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
        CFG.DEVICE = torch.device("cuda", args.local_rank)
        print("cur local_rank: {} ".format(args.local_rank))
        
    train(args.local_rank)

def train(local_rank):
    device = CFG.DEVICE
    model = fasterrcnn_dcnresnet_fpn(num_classes=len(CLONE_CLASSES), pretrained_backbone=True, trainable_backbone_layers=5, backbone_name='dcnresnet18')
#     model = fasterrcnn_resnet_fpn(num_classes=len(CLONE_CLASSES), pretrained_backbone=True, trainable_backbone_layers=5, backbone_name='resnet18')
    model = model.to(device)
    if CFG.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    trainer = Trainer(model = model,
                      optimizer = optimizer,
                      device = device,
                      img_path = '../data/img_clip_all',
                      train_anno_path = '../data/train_day6.csv',
                      eval_anno_path = '../data/test_day6.csv',
                      batch_size = CFG.BATCH_SIZE)
    trainer.train()
    
def test(local_rank):
    device = CFG.DEVICE
    model = fasterrcnn_dcnresnet_fpn(num_classes=len(CLONE_CLASSES), pretrained_backbone=True, trainable_backbone_layers=5, backbone_name='dcnresnet18')
#     model = fasterrcnn_resnet_fpn(num_classes=len(CLONE_CLASSES), pretrained_backbone=True, trainable_backbone_layers=5, backbone_name='resnet18')
    checkpoint_dict = torch.load('./model_best_ap50_95.pth', map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.to(device)
    if CFG.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)
    tester = Tester(model = model,
                    device = device,
                    img_path = '../data/img_clip_all', 
                    test_anno_path = '../data/test_day6.csv',
                    batch_size = CFG.BATCH_SIZE)
        
    tester.test()
    
if __name__ == '__main__':
    
    main()