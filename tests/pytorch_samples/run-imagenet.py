import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

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

# model_name = "resnet50"
model_name = "mobilenet_v3_large"

batch_size = 64
mixed_precision = False
iterations = 10

cudnn.benchmark = True

model = getattr(torchvision.models, model_name)()
model = model.cuda()

data = torch.randn(batch_size, 3, 224, 224)
target = torch.LongTensor(batch_size).random_() % 1000
data, target = data.cuda(), target.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

if mixed_precision:
    scaler = torch.cuda.amp.GradScaler(enabled=True)
else:
    scaler = None

for i in range(iterations):
    optimizer.zero_grad()

    if mixed_precision:
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = F.cross_entropy(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    
    print(loss)