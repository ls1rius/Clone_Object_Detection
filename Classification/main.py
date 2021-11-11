from train import Trainer
from test import Tester
import torch
import torchvision
from data.datasets import CLONE_CLASSES
import os
from CFG import CFG
import argparse
from models import resnet18

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--world_size", type=int, default=8)
    args = parser.parse_args()

    if CFG.DISTRIBUTED:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
        CFG.DEVICE = torch.device("cuda", args.local_rank)
        print("cur local_rank: {} ".format(args.local_rank))
        
    train(args.local_rank)

def train(local_rank):
    model = resnet18(num_classes=len(CLONE_CLASSES))
    device = CFG.DEVICE
    model = model.to(device)
    if CFG.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model = model,
                      optimizer = optimizer,
                      loss_func = loss_func,
                      device = device,
                      img_path = '../data/img_clip_all',
                      train_anno_path = '../data/train_day6.csv',
                      eval_anno_path = '../data/test_day6.csv',
                      batch_size = CFG.BATCH_SIZE)
    trainer.train()
    
def test(local_rank):
    model = resnet18(num_classes=len(CLONE_CLASSES))
    device = CFG.DEVICE
    checkpoint_dict = torch.load('./exp/20211109_1125_53/model_best_acc.pth')
    model.load_state_dict(checkpoint_dict['state_dict'])
    model = model.to(device)
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