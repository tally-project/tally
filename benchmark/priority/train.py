import hidet
import torch
import time

a = hidet.randn([5120, 5120], device='cuda')
b = hidet.randn([5120, 5120], device='cuda')

start_time = time.time()
iters = 0

for i in range(10000):

    d = hidet.ops.matmul(a, b)
    torch.cuda.synchronize()

    iters += 1
    end_time = time.time()

    time_elapsed = end_time - start_time
    if end_time - start_time >= 1:
        print(f"Throughput: {iters / time_elapsed}iters/s")
        iters = 0
        start_time = time.time()