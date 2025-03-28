import random
import time

import hidet
import torch

# disable CUDA Graph
hidet.torch.dynamo_config.use_cuda_graph(False)

hidet.torch.dynamo_config.use_tensor_core(True)
hidet.torch.dynamo_config.use_fp16(flag=True)
# hidet.torch.dynamo_config.search_space(level=2)

x = torch.randn(1, 3, 224, 224).cuda()
model = torch.hub.load(
    'pytorch/vision:v0.9.0', 'resnet50', pretrained=True, verbose=False
)
model = model.cuda().eval()

# # optimize the model with 'hidet' backend
# model_opt = torch.compile(model, backend='hidet')

model_opt = model

# run the optimized model
y1 = model_opt(x)

# benchmark the performance

enter_time = time.time()
iters = 0

for _ in range(100000):

    # milli_seconds = random.randint(1, 1000)
    # time.sleep(milli_seconds / 1000.)

    torch.cuda.synchronize()
    start_time = time.time()

    y = model_opt(x)

    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000

    # print('{:>10}: {:.3f} ms'.format('hidet', elapsed_time_ms))

    iters += 1
    out_time = time.time()

    time_elapsed = out_time - enter_time
    if time_elapsed >= 1:
        print(f"Throughput: {iters / time_elapsed}iters/s")
        iters = 0
        enter_time = time.time()