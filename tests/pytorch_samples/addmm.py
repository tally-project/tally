import torch
import random
import numpy as np

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

seed = 42 # any number 
set_deterministic(seed=seed)

bias = torch.randn(4800).cuda()
x = torch.randn(512, 1600).cuda()
weight = torch.randn(1600, 4800).cuda()

x = torch.addmm(bias, x, weight)

print(x)