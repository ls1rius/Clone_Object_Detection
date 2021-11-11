from easydict import EasyDict as edict
import torch
CFG = edict()
CFG.BATCH_SIZE = 128
CFG.WORKERS = 8
CFG.DISTRIBUTED = False
CFG.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 



