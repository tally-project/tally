import hidet
import torch

# disable CUDA Graph
hidet.torch.dynamo_config.use_cuda_graph(False)

hidet.torch.dynamo_config.use_tensor_core(True)
hidet.torch.dynamo_config.use_fp16(flag=True)
# hidet.torch.dynamo_config.search_space(level=2)

# take resnet18 as an example
x = torch.randn(64, 3, 224, 224).cuda()
model = torch.hub.load(
    'pytorch/vision:v0.9.0', 'resnet50', pretrained=True, verbose=False
)
model = model.cuda().eval()

# optimize the model with 'hidet' backend
model_opt = torch.compile(model, backend='hidet')

# run the optimized model
y1 = model_opt(x)

# benchmark the performance
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
start_event.record()
for _ in range(100):
    y = model_opt(x)
end_event.record()
torch.cuda.synchronize()
print('{:>10}: {:.3f} ms'.format('hidet', start_event.elapsed_time(end_event) / 100.0))