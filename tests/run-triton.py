
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_zhaowe58/ld/cld7gxhkgh4ndznpyyoaouso2auageq6br2wsnl4aanfwp27hncq.py
# Source Nodes: [l__self___lin, relu], Original ATen: [aten.addmm, aten.relu, aten.threshold_backward]
# l__self___lin => addmm
# relu => relu
triton_unk_fused_addmm_relu_threshold_backward_0 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=4, num_warps=1, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, arg_A, arg_B, out_ptr1, out_ptr2):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 10
    N = 10
    K = 100
    stride_am = 100
    stride_ak = 1
    stride_bk = 1
    stride_bn = 100

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (10*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, mask.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tmp2 = triton_helpers.maximum(0, tmp1)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, mask.shape)), tmp2, mask)
    tl.store(out_ptr2 + (tl.broadcast_to(xindex, mask.shape)), tmp4, mask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import torch._inductor.kernel.mm_common
meta0 = {'GROUP_M': 8, 'EVEN_K': False, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (10, 100), (100, 1))
    assert_size_stride(primals_2, (10, ), (1, ))
    assert_size_stride(primals_3, (10, 100), (100, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        # Source Nodes: [l__self___lin, relu], Original ATen: [aten.addmm, aten.relu, aten.threshold_backward]
        stream0 = get_cuda_stream(0)
        triton_unk_fused_addmm_relu_threshold_backward_0.run(primals_2, primals_3, primals_1, buf1, buf2, grid=torch._inductor.kernel.mm_common.mm_grid(10, 10, meta0), stream=stream0)
        del primals_1
        del primals_2
        return (buf1, primals_3, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10, 100), (100, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((10, 100), (100, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
