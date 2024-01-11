import torch
import torch.nn.functional as F
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

# Check if CUDA is available and use it, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a sample tensor
x = torch.randn(12, 60, 60).to(device)

# Apply dropout using a dropout probability of 0.2
# The 'training=True' flag is important to apply dropout. In evaluation mode, dropout is not applied.
y = F.dropout(x, p=0.1, training=True, inplace=False)

print(y.max())