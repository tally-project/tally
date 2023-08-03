import torch

# def foo(x, y):
#     a = torch.sin(x)
#     b = torch.cos(y)
#     return a + b

# opt_foo1 = torch.compile(foo)
# print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyModule().cuda()
opt_mod = torch.compile(mod, backend='inductor', mode="max-autotune")
print(opt_mod(torch.randn(10, 100, device="cuda")))

torch.cuda.synchronize()

b = torch.randn(10, 100, device="cuda")

for i in range(10):
    c = opt_mod(b)

torch.cuda.synchronize()

print(c)