import random
import numpy as np

import torch
import torch.nn as nn

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

# Check if CUDA is available and set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a random input tensor on the GPU
input_tensor = torch.randn(256, 3, 128, 128).to(device)

# Define the max pooling layer
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Move the max pooling layer to the GPU
max_pool.to(device)

# Apply max pooling to the input tensor on the GPU
output = max_pool(input_tensor)

# Move the output tensor back to the CPU for printing
output = output.to("cpu")

# Print the shapes of input and output tensors
print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)