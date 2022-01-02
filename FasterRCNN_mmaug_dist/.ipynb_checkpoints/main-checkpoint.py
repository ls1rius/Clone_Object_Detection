from train import Trainer
from test import Tester
import torch
import torchvision
from data.datasets import CLONE_CLASSES
import os
from CFG import CFG
import argparse
from models.detection import fasterrcnn_resnet_fpn
import torch.multiprocessing as mp
from utils import configure_nccl, configure_omp, get_num_devices 
import utils.dist as comm
from datetime import timedelta
from loguru import logger
import torch.distributed as dist
CUDA_VISIBLE_DEVICES=0
DEFAULT_TIMEOUT = timedelta(minutes=30)

def main(args):
    main_func = train if args.train else test
    if CFG.DISTRIBUTED:
        dist_url = "auto" if args.dist_url is None else args.dist_url
        backend = args.dist_backend
        
        machine_rank = 0
        num_machines = 1
        num_gpus_per_machine = get_num_devices()
        world_size = num_machines * num_gpus_per_machine
            
        if world_size > 1:
            if dist_url == "auto":
                assert (
                    num_machines == 1
                ), "dist_url=auto cannot work with distributed training."
                port = _find_free_port()
                dist_url = f"tcp://127.0.0.1:{port}"
            mp.start_processes(
                _distributed_worker,
                nprocs=num_gpus_per_machine,
                args=(
                    main_func,
                    world_size,
                    num_gpus_per_machine,
                    machine_rank,
                    backend,
                    dist_url,
                    args,
                ),
                daemon=False,
                start_method='spawn',
            )
        else:
            CFG.DISTRIBUTED = False
            main_func(args)
            
    else:
        CFG.DISTRIBUTED = False
        main_func(args)
        
            
def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port  


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    backend,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
     # set environment variables for distributed training
    args.local_rank = local_rank
    configure_nccl()
    configure_omp()
    main_func(args)


def train(args):
    device = CFG.DEVICE
    if CFG.DISTRIBUTED:
        device = "cuda:{}".format(args.local_rank)
    model = fasterrcnn_resnet_fpn(num_classes=len(CLONE_CLASSES), pretrained_backbone=True, trainable_backbone_layers=5, backbone_name='resnet18')
    model = model.to(device)
    if CFG.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank])
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model = model,
                      optimizer = optimizer,
                      device = device,
                      img_path = '../data/img_clip_all',
                      train_anno_path = '../data/train.csv',
                      eval_anno_path = '../data/test.csv',
                      batch_size = CFG.BATCH_SIZE)
    trainer.train()
    
def test(args):
    device = CFG.DEVICE
    if CFG.DISTRIBUTED:
        device = "cuda:{}".format(args.local_rank)
    model = fasterrcnn_resnet_fpn(num_classes=len(CLONE_CLASSES), pretrained_backbone=True, trainable_backbone_layers=5, backbone_name='resnet18')
    
    # MODEL RESUME
    checkpoint_dict = torch.load('./20211220_0226_23/model_best_mFS50_50.pth', map_location='cpu')
    if "state_dict" in checkpoint_dict:
        checkpoint_dict = checkpoint_dict["state_dict"]
    model_dict = model.state_dict()
    if "module" in list(checkpoint_dict.keys())[0] and "module" not in list(model_dict.keys())[0]:
        checkpoint_dict = {k.partition('module.')[2]: v for k, v in checkpoint_dict.items()}
    if "model" in list(checkpoint_dict.keys())[0] and "model" not in list(model_dict.keys())[0]:
        checkpoint_dict = {k.partition('model.')[2]: v for k, v in checkpoint_dict.items()}
        checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if (k in model_dict)}
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    
    model.to(device)
    if CFG.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank])
    tester = Tester(model = model,
                    device = device,
                    img_path = '../data/img_clip_all', 
                    test_anno_path = '../data/test_day4.csv',
                    batch_size = CFG.BATCH_SIZE)
        
    tester.test()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # distributed
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist_url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--train",
        default=False,
        action="store_true",
        help="whether to train",
    )
    args = parser.parse_args()
    main(args)