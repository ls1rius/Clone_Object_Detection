python main.py &> train_20211006_0000.txt &



CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 main.py



CUDA_VISIBLE_DEVICES=3 python main.py &> train_20211027_1116.txt &


