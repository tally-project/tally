import torch
import math
import random
import numpy as np
import os

# make ourselves as deterministic as possible.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# (deterministic) messy number generator
v = torch.Tensor([(x*math.pi)%1 for x in range(1000000)]).to(torch.device("cuda"))

# init a and b to the same cumulative sum.
a = b = v.cumsum(0)

print((a-b).abs()[a-b!=0])

torch.cuda.synchronize()