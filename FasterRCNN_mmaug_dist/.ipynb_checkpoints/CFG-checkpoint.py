from easydict import EasyDict as edict
import torch
CFG = edict()
CFG.BATCH_SIZE = 8
CFG.WORKERS = 4
CFG.DISTRIBUTED = True
CFG.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
CFG.RESIZE = 896


