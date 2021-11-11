from easydict import EasyDict as edict
import torch
CFG = edict()
CFG.BATCH_SIZE = 24
CFG.WORKERS = 8
CFG.DISTRIBUTED = False
CFG.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
CFG.RESIZE = 896


