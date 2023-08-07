import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

_tensor = torch.randn(10, 100, device="cuda")

mod = MyModule().cuda()
opt_mod = torch.compile(mod, backend='inductor', mode="max-autotune")

res = opt_mod(_tensor)
res_ref = mod(_tensor)

torch.cuda.synchronize()

if torch.allclose(res, res_ref):
    print("Results match")
else:
    print("Results mismatch")