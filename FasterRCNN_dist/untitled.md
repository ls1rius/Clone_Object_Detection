python main.py &> train_20211006_0000.txt &



CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 main.py



CUDA_VISIBLE_DEVICES=0 python main.py &> train_20211026_2002.txt &



CUDA_VISIBLE_DEVICES=0,1 python main.py