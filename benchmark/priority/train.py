import time

import hidet
import torch

# disable CUDA Graph
hidet.torch.dynamo_config.use_cuda_graph(False)

hidet.torch.dynamo_config.use_tensor_core(True)
hidet.torch.dynamo_config.use_fp16(flag=True)
# hidet.torch.dynamo_config.search_space(level=2)

x = torch.randn(64, 3, 224, 224).cuda()
model = torch.hub.load(
    'pytorch/vision:v0.9.0', 'resnet50', pretrained=True, verbose=False
)
model = model.cuda().eval()

# optimize the model with 'hidet' backend
model_opt = torch.compile(model, backend='hidet')

# run the optimized model
y1 = model_opt(x)

print("Start training")

# benchmark the performance

start_time = time.time()
iters = 0

for _ in range(10000):

    y = model_opt(x)
    torch.cuda.synchronize()

    iters += 1
    end_time = time.time()

    time_elapsed = end_time - start_time
    if end_time - start_time >= 1:
        print(f"Throughput: {iters / time_elapsed}iters/s")
        iters = 0
        start_time = time.time()

print("End training")