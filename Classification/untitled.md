python main.py &> train_20211006_0000.txt &



CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 main.py



CUDA_VISIBLE_DEVICES=1 python main.py &> train_20211102_2026.txt &


