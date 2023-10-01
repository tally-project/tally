from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_zhaowe58/lx/clx2d27z72ugk6fzs7vmg42g2zpxck7wkmlksjfcuigyn2nsdalx.py
# Source Nodes: [], Original ATen: [aten.mm]

triton_tem_fused_mm_0 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=5, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_mm_0'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 64
    N = 2048
    K = 1000
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1000
    stride_ak = 1
    stride_bk = 2048
    stride_bn = 1

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
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
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
    xindex = idx_n + (2048*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
import torch._inductor.kernel.mm_common
meta0 = {'GROUP_M': 8, 'EVEN_K': False, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}


# kernel path: /tmp/torchinductor_zhaowe58/ld/cldwiidn3pcwjsjbtu2hgmysrw62ugqs4m2aku3p5ycgj5ndayg4.py
# Source Nodes: [], Original ATen: [aten.mm]

triton_tem_fused_mm_1 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_mm_1'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16

    A = arg_A
    B = arg_B

    M = 1000
    N = 2048
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1
    stride_ak = 1000
    stride_bk = 2048
    stride_bn = 1

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
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
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
    xindex = idx_n + (2048*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')
meta1 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_zhaowe58/g7/cg7bu6vrbqu7yecstalgsxbdcu2tcr7qocs6j25r77kwv2bm7phl.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_2', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/y5/cy5qf4sruyodjgejbse7ldddh35ljshywdpjn5xr62536schzfyh.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_div_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_threshold_backward_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2)
        tmp4 = tl.load(in_ptr1 + (x0 + (2048*(((r2 + (126*x1)) // 49) % 64))), rmask & tmp2, other=0)
        tmp5 = 49.0
        tmp6 = tmp4 / tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp3, tmp7, tmp6)
        tmp9 = tl.where(tmp2, tmp8, 0)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, other=0)
        tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp8 * tmp15
        tmp17 = tl.where(tmp2, tmp16, 0)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/37/c37cqwonbdeha4euzwouclumzns2qqnwpg7s3jfq2hso7gqae7au.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_4', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/kg/ckgxwd4m3w6okehaxfrohpcsjujqlwpuiuuntzn2azazc76dfqq2.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_5', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0)
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, None)
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/jw/cjw24dhzabpdulhc2gewmlrfhdmhovchyc55l4wbp54gsao7xh7y.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 2048
    x2 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = 49.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.00031887755102040814
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/z6/cz6leyjqshlwavdy6rvonvpzwjcjaeipa46lyowgg7e4jwxcidbq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(r2 % 7)) + (3584*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2 & xmask, other=0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((7*(((r2 + (126*x1)) // 7) % 7)) + (49*x0) + (25088*(((r2 + (126*x1)) // 49) % 64)) + (r2 % 7)), rmask & tmp2 & xmask, other=0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp12 = tl.load(in_ptr2 + (x0 + (512*(r2 % 7)) + (3584*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2 & xmask, other=0)
        tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.where(tmp2, tmp15, 0)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/px/cpxgvzmf624v2fjyptzhf52lb4anhgp2cwkpah3eskxvhbkeptee.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_8', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/mt/cmtmdgffkgrsgkurgsvsgf252rwproinimdx5yiillunagas6vpn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_9', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/mr/cmrznqosxmllbx67ty736rptxeypnm7f323w6n5huetkmjxeprrx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (49*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00031887755102040814
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/bs/cbsvzse4aiu4puiwhsqcsg76jyu37yt7odokd2lerym3lprlbewg.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_11', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, other=0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2)
        tmp7 = tl.load(in_ptr2 + (x0 + (2048*(((r2 + (126*x1)) // 49) % 64))), rmask & tmp2, other=0)
        tmp8 = 49.0
        tmp9 = tmp7 / tmp8
        tmp10 = tl.where(tmp6, tmp4, tmp9)
        tmp11 = tl.load(in_ptr3 + ((7*(((r2 + (126*x1)) // 7) % 7)) + (49*x0) + (100352*(((r2 + (126*x1)) // 49) % 64)) + (r2 % 7)), rmask & tmp2, other=0)
        tmp12 = tmp10 + tmp11
        tmp13 = tl.where(tmp5, tmp4, tmp12)
        tmp14 = tl.where(tmp2, tmp13, 0)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp18 = tl.load(in_ptr4 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, other=0)
        tmp19 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp13 * tmp20
        tmp22 = tl.where(tmp2, tmp21, 0)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, None)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp24, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/cc/cccdafp2kgm6bvd6jvvkmbcbyigbwx2scevyef7ndftk6zm3mbld.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (2048*y1)), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = 0.00031887755102040814
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp10 - tmp20
    tmp23 = tmp22 * tmp15
    tmp24 = tmp21 - tmp23
    tmp26 = tmp17 * tmp25
    tmp27 = tmp24 * tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (2048*y3)), tmp27, ymask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/aj/cajkk2adu34x4ger33dfyykpm7sgprg46fjd5knhp6hy3psyb4zw.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]

triton_poi_fused_add_div_threshold_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_threshold_backward_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (2048*y3)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (2048*y1)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (y0 + (49*x2) + (100352*y1)), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tl.store(out_ptr0 + (x2 + (2048*y3)), tmp15, ymask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/sp/cspc5la4eofamyr7pwmay4mwkz4pjpjfhtd6kieqmuldzb3bxdlg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_14', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, other=0)
        tmp4 = tl.where(tmp2, tmp3, 0)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp8 = tl.load(in_ptr1 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, other=0)
        tmp9 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp3 * tmp10
        tmp12 = tl.where(tmp2, tmp11, 0)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp16 = tl.load(in_ptr3 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, other=0)
        tmp17 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp3 * tmp18
        tmp20 = tl.where(tmp2, tmp19, 0)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, None)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/m6/cm6k55wmcrhojywdroejzhiptqog5zh2zq3bvksdjmbqv3puhhgx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_15', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), None)
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00031887755102040814
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr1 + (x2), tmp31, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/kp/ckpvq4zakhsevodvrufdtnhwfhkcviddl4jgntl62a3xzpxyqvoj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_16', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (512*r2) + (65536*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((196*x1) + (100352*((r2 + (128*x0)) // 196)) + ((r2 + (128*x0)) % 196)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/sy/csyt3kmtutddvhjnsx2g7peyvxgegcaxple2rzwjhz5z5y4pzor6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_17', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (98*x0)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/b3/cb3d63rwxwmtbd2ugv3raky2cosktpnatkt75cjnp5lplsufbdb6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_18', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((196*x0) + (100352*((r2 + (128*x1)) // 196)) + ((r2 + (128*x1)) % 196)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp5 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/rf/crfcteqta6isb6xtirf676bsgddls3qgoi6i63uwd33gccauobfo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/5y/c5y5axilpywatyaupu64exmjdordb7aa2yxtsjmygvmlxqzoaakc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 7.971938775510203e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/vk/cvk77yddwze67ads6x7uqowd34fiuagbdzm2xo37ajvcnvezwjpc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_21', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r3)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/3g/c3g4gbenqyvclhkraep6ji4sp6hjm54ma6fhqc3qu4xlirsksi2k.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_22', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (1024*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((196*x1) + (200704*((r2 + (128*x0)) // 196)) + ((r2 + (128*x0)) % 196)), rmask, eviction_policy='evict_last', other=0)
        tmp4 = tl.load(in_ptr2 + ((196*x1) + (200704*((r2 + (128*x0)) // 196)) + ((r2 + (128*x0)) % 196)), rmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.load(in_ptr3 + (x1 + (1024*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/6u/c6uwiiwja4smzpfu4wi5edu65le5m6g7sj5h6dmzr4lnf7jl56nm.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_23', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (98*x0)), rmask & xmask, other=0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/od/codxxfiy2giij5xvf5y6estgo2nxubwbf4awruj6idex2t5o3mpg.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 7.971938775510203e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/cw/ccwpkkpdzbzxp26mytz4zdoxkgvkxgthju6dcfwsxqyr4fg6xjuf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_25', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((196*x1) + (50176*((r2 + (128*x0)) // 196)) + ((r2 + (128*x0)) % 196)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/gm/cgmabotvl6y7lum7ipoxvbdes7nwljyoimfmjgte7pp5xbqjzrf5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_26', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (98*x0)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/xi/cxire2hrymz4i6cf4zq2d56qwtptbvz4zrcnm3ervcip5gbjfef5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_27', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((196*x0) + (50176*((r2 + (128*x1)) // 196)) + ((r2 + (128*x1)) % 196)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp5 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/mu/cmubihxmf2ap45ir5eqfrfchrytzvi6gvn6in3vg7nu2sj37mlbc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_28', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/54/c54sfoj2ea7cxgdhc7vt5jhyvnguvtfpiiubnt4zoafse7amvggi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 7.971938775510203e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/3y/c3yriqvycapkkgayinw6k35etrxglq5fikr65i5v2uoseof6y3c7.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_30', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/p3/cp3k5c7qxifm2atuo4e3cdk2dk77naxfnyogswm3bsgfmf3tjbs4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_31', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ii/cii55h7xqevumuwiemfykgzmwoofugplpcww6kaln22oxkaim3wr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_32', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x1) + (200704*((r2 + (128*x0)) // 196)) + ((r2 + (128*x0)) % 196)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.load(in_ptr1 + (x1 + (1024*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/3g/c3gcm7x5vpk3lx6cvwv2rzsq756hukpb5kpo3fa4gzhatbkrq4ur.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_33', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 7.971938775510203e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/sv/csvrx72c7gfykduhkqcpqmtqquu3iwnnidjrsh2vicvogjjoen4l.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_34', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ns/cnsc7zhmpeu2h4lg7p2pi45qfrs6neghzbz6mahk45d44ziygwte.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_35', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((196*x1) + (200704*((r2 + (128*x0)) // 196)) + ((r2 + (128*x0)) % 196)), rmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x1 + (1024*r2) + (131072*x0)), rmask, other=0)
        tmp8 = tl.load(in_ptr3 + (x1 + (1024*r2) + (131072*x0)), rmask, other=0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/4p/c4puvjny2elywetcthjaa4sw3hrlnxorijmpojqtuyu3yjfrz5pk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_36', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 7.971938775510203e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1024*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/w6/cw6vulj7zrkx7wrizjobe63vynzkfppy73gszjibjq74rdqwcbjt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_37', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 392
    x1 = (xindex // 392)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (200704*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/7z/c7zy7zkvdobdz27doah7hvd2j2jgfvx7d26t52esknffjzq3zwsl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_38', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (392*x0)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/vc/cvcmqmtjjj3zhzbyshvnrnvbsocebvkxxqzfxaj3hrai2ikbz337.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_39', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (200704*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask, eviction_policy='evict_last', other=0)
        tmp5 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/vk/cvkppqhqrzdr2l7glcmya6rpie26an5menpsrt7zut3uqzqzjiph.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_40', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ws/cwsdsa6k344ctwmnbxcizk7q5o5nadv5vsokvwtufcfyixzaohws.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 1.992984693877551e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/5w/c5w2onmt7j7775ak2mms6im7zblonrw7v6a4gkvgxiljlitvokht.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_42', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/oa/coab54ohkrzz7dp66nbi67ddq2padi5p2lzn72cbcacj5qdvb3j6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_43', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (512*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (401408*((r2 + (256*x0)) // 784)) + ((r2 + (256*x0)) % 784)), rmask, eviction_policy='evict_last', other=0)
        tmp4 = tl.load(in_ptr2 + ((784*x1) + (401408*((r2 + (256*x0)) // 784)) + ((r2 + (256*x0)) % 784)), rmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.load(in_ptr3 + (x1 + (512*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/gh/cghajlegwveoxcaoai3hr5sxc4rpnqwjiswvm56sqpoydn6qjtsr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_44', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/te/cteaz4fnee55xn5zenpenmuwv4mlhxdxbr5b7rzq3p67plsjhjls.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_45', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (512*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (784*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (512*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 1.992984693877551e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/p7/cp7mdgsb6sia2lb7pbrgobou57e3ehkbfdnqltivq3sgm3gssphu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_46', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 392
    x1 = (xindex // 392)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (100352*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/6n/c6nzewx2ojj6mcd3sqgz34mgerclykfgkpennhq73c2ikvfbovo6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_47', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (392*x0)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/h3/ch3at2waqtjyno2p6i36vlgu3jy3auu2ict2wrlbzqsri2xejewt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_48', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (100352*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/eh/cehfhj3ttgejx7tvcav63lnazuyyytsiwb3pwrf6i7af4cnax2h5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_49', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/jm/cjmw6nbtxuxnusr7sptdvfnxmxl7rjibiyhx2gwoqaix243okmzx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 1.992984693877551e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/s2/cs25aztgxji64jsube6gn37npyedewxb3qcdbzju5ubxpgbj3ejl.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_51', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (512*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/3e/c3eknbsa5nrg3nlgj46vnb5xnbpt2f4o336syrjpoyzktds5oold.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_52', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/th/cthwu4q7irricypfmwgwmil4mhvmv6e3ssnng2azomdvi6dsogxb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_53', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (401408*((r2 + (256*x0)) // 784)) + ((r2 + (256*x0)) % 784)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.load(in_ptr1 + (x1 + (512*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ez/cezpqt5b5pqi5rmuuxzl7j3bjua5zddtalbsc7phgg6nhlaw3thu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_54', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1.992984693877551e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/hz/chzkmf7rsiwqflietzjabcavwvhjfm47mxofeqi2fpbq462rzwcn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_55', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (401408*((r2 + (256*x0)) // 784)) + ((r2 + (256*x0)) % 784)), rmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x1 + (512*r2) + (131072*x0)), rmask, other=0)
        tmp8 = tl.load(in_ptr3 + (x1 + (512*r2) + (131072*x0)), rmask, other=0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/cn/ccnda73lw76nhutcaekne27ai22kunhulagcdg65y7uomofa3qbj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_56', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (512*y3)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1.992984693877551e-05
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 * tmp5
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp0 - tmp26
    tmp28 = tmp27 - tmp13
    tmp30 = tmp23 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask)
    tl.store(out_ptr1 + (x2 + (512*y3)), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/hv/chvc7vtvdfc4rxjbvt5lnqkiqlfp3nxvuyxto7l5d5zfcwqrzpur.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_57', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (401408*((r2 + (256*x0)) // 3136)) + ((r2 + (256*x0)) % 3136)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/5v/c5vnkwiv4bjorhvi3c7ihq4d2sjfsiozcet4e7nw5f2z6u4ozjg6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_58', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ik/cikv7dk5x4543sgnojcjkewgljoqbmk3g4pmkxzm3qvze24ikbfx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_59', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (401408*((r2 + (256*x1)) // 3136)) + ((r2 + (256*x1)) % 3136)), rmask, eviction_policy='evict_last', other=0)
        tmp5 = tl.load(in_ptr2 + (x0 + (128*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/7b/c7btfmrsiya7kjsurqyeullzy6gudrckubxlisn5vfpqvzs5w3u2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_60', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/p5/cp5ygjmx3m7w53yxmygbmzemaawxrvzhm4kuy2lkzzthot3xhfo7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 4.982461734693877e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/te/cte3epqzoxn3r6njheckwkomc3q4gmhswor5sttbf2rof3ljhjto.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 262144],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_62', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/2q/c2qge2yxkc6pjeqjxa34g5nasmv2htpkqbajeqldo7dpdsjisz6c.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 1024],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_63', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 83968
    rnumel = 612
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 328
    x1 = (xindex // 328)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (612*x0)
        tmp1 = tl.full([1, 1], 200704, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (256*((r2 + (612*x0)) % 200704))), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((3136*x1) + (802816*(((r2 + (612*x0)) // 3136) % 64)) + ((r2 + (612*x0)) % 3136)), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp7 = tl.load(in_ptr2 + ((3136*x1) + (802816*(((r2 + (612*x0)) // 3136) % 64)) + ((r2 + (612*x0)) % 3136)), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.load(in_ptr3 + (x1 + (256*((r2 + (612*x0)) % 200704))), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp9 * tmp12
        tmp14 = tl.where(tmp2, tmp13, 0)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/7n/c7nj276wilxvxdd6viqxsqe7g2ohtfyam6unfpetgktimktlf6i3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_64', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 328
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (328*x0)), rmask & xmask, other=0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/nv/cnv5lfgw5xs2eaxafpbuiidrnsts6ygjiuece4hkhyuobspcdca7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_65', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (3136*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 4.982461734693877e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/uc/cucujnxkfw3fy3nocgohguzc6r4eu2umfs3givedfm7jmco5jit5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_66', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (200704*((r2 + (256*x0)) // 3136)) + ((r2 + (256*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/tm/ctm7gjk4iz6yiiayspm7urw6uwx4yd5ofc763ghdifcw5hhsjomc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_67', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/bz/cbzaeivlluutequmapcdjwtw6wxgf2lobffqo5fgfpoddatf2lrx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_68', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (200704*((r2 + (256*x1)) // 3136)) + ((r2 + (256*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/cw/ccwgeamr56fzd3msw737kjkykhlrscdu3ljwakck7olu5cf7f7uq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_69', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/nn/cnnqnvnqio7tjdwor4xyskdve6wubdtsoq3a3cykit7zebmkx3cp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 4.982461734693877e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/r7/cr7asorhtkvnfetwgxy2t3ljvhhuc2swaxd47f2xrfmhsbot4q6t.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_71', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/s6/cs6fjnou4f6grq3enfn3p3amuxifidzthx4jcwhjkeggtd3saflz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 262144],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_72', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ht/chtuspop3f2y5bandxwus6iq77j6blng5odxfhi322pfvkueieu7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 1024],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_73', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 83968
    rnumel = 612
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 328
    x1 = (xindex // 328)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (612*x0)
        tmp1 = tl.full([1, 1], 200704, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x1) + (802816*(((r2 + (612*x0)) // 3136) % 64)) + ((r2 + (612*x0)) % 3136)), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp4 = tl.load(in_ptr1 + (x1 + (256*((r2 + (612*x0)) % 200704))), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/v4/cv4jam6lxxmyozzwn4x474kmzohfyb2hd7yh2vtgz56ivriu3fpd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_74', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 4.982461734693877e-06
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/pk/cpkw7sw7rmb2l3agne62jrad3jn7dnv2opl63dgd77z57mfcohrh.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 1024],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_75', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 83968
    rnumel = 612
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 328
    x1 = (xindex // 328)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (612*x0)
        tmp1 = tl.full([1, 1], 200704, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (256*((r2 + (612*x0)) % 200704))), rmask & tmp2, other=0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((3136*x1) + (802816*(((r2 + (612*x0)) // 3136) % 64)) + ((r2 + (612*x0)) % 3136)), rmask & tmp2, other=0)
        tmp7 = tl.load(in_ptr2 + ((3136*x1) + (802816*(((r2 + (612*x0)) // 3136) % 64)) + ((r2 + (612*x0)) % 3136)), rmask & tmp2, other=0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.load(in_ptr3 + (x1 + (256*((r2 + (612*x0)) % 200704))), rmask & tmp2, other=0)
        tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp9 * tmp12
        tmp14 = tl.where(tmp2, tmp13, 0)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp18 = tl.load(in_ptr5 + (x1 + (256*((r2 + (612*x0)) % 200704))), rmask & tmp2, other=0)
        tmp19 = tl.load(in_ptr6 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2, eviction_policy='evict_last', other=0)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp9 * tmp20
        tmp22 = tl.where(tmp2, tmp21, 0)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, None)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp24, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/z3/cz35yh3w2ifwziozfrdyibahkxmu5eyrz5cvwyp4iimzcq6ebgyk.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_76', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (3136*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 4.982461734693877e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp23, xmask)
    tl.store(out_ptr1 + (x2 + (256*y3)), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/op/copnzeydmihztzvb7nlb2wnmuqlje7jd6xbkqikhp3ig2ifs5pw4.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_77', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/l7/cl7cmu5pc6oolswmk7g2pgcaou3dogcaljlr3ojufpkjuhc2avbu.py
# Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]

triton_poi_fused_add_max_pool2d_with_indices_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_backward_78', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 112
    x3 = (xindex // 112)
    y0 = yindex % 64
    y1 = (yindex // 64)
    y4 = yindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*(tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2)))))) + (64*(tl.where((tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) >= 0, 0, 56))) + (3584*(tl.math.min(tl.math.max(0, (x3 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2)))))) + (3584*(tl.where((tl.math.min(tl.math.max(0, (x3 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2))))) >= 0, 0, 56))) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x3 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x3 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2))))) >= 0, 0, 56))) + (3136*y4) + (tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) >= 0, 0, 56))), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (y0 + (64*(tl.math.min(1 + (tl.math.max(0, (x2 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2)))))) + (64*(tl.where((tl.math.min(1 + (tl.math.max(0, (x2 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) >= 0, 0, 56))) + (3584*(tl.math.min(tl.math.max(0, (x3 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2)))))) + (3584*(tl.where((tl.math.min(tl.math.max(0, (x3 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2))))) >= 0, 0, 56))) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x3 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x3 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2))))) >= 0, 0, 56))) + (3136*y4) + (tl.math.min(1 + (tl.math.max(0, (x2 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x2 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) >= 0, 0, 56))), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (y0 + (64*(tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2)))))) + (64*(tl.where((tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) >= 0, 0, 56))) + (3584*(tl.math.min(1 + (tl.math.max(0, (x3 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2)))))) + (3584*(tl.where((tl.math.min(1 + (tl.math.max(0, (x3 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2))))) >= 0, 0, 56))) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x3 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x3 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2))))) >= 0, 0, 56))) + (3136*y4) + (tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x2 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) >= 0, 0, 56))), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (y0 + (64*(tl.math.min(1 + (tl.math.max(0, (x2 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2)))))) + (64*(tl.where((tl.math.min(1 + (tl.math.max(0, (x2 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) >= 0, 0, 56))) + (3584*(tl.math.min(1 + (tl.math.max(0, (x3 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2)))))) + (3584*(tl.where((tl.math.min(1 + (tl.math.max(0, (x3 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2))))) >= 0, 0, 56))) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x3 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x3 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x3) // 2))))) >= 0, 0, 56))) + (3136*y4) + (tl.math.min(1 + (tl.math.max(0, (x2 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x2 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x2) // 2))))) >= 0, 0, 56))), xmask, eviction_policy='evict_last')
    tmp2 = x5
    tmp3 = tmp0 == tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp8 = tmp6 == tmp2
    tmp9 = tl.math.max(0, (x3 // 2))
    tmp10 = tl.math.min(56, 1 + ((1 + x3) // 2))
    tmp11 = tmp9 < tmp10
    tmp12 = 1 + (tl.math.max(0, (x2 // 2)))
    tmp13 = tl.math.min(56, 1 + ((1 + x2) // 2))
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp15 & tmp8
    tmp17 = tmp5 + tmp7
    tmp18 = tl.where(tmp16, tmp17, tmp5)
    tmp21 = tmp19 == tmp2
    tmp22 = 1 + (tl.math.max(0, (x3 // 2)))
    tmp23 = tmp22 < tmp10
    tmp24 = tl.math.max(0, (x2 // 2))
    tmp25 = tmp24 < tmp13
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp21
    tmp28 = tmp18 + tmp20
    tmp29 = tl.where(tmp27, tmp28, tmp18)
    tmp32 = tmp30 == tmp2
    tmp33 = tmp23 & tmp14
    tmp34 = tmp33 & tmp32
    tmp35 = tmp29 + tmp31
    tmp36 = tl.where(tmp34, tmp35, tmp29)
    tl.store(out_ptr0 + (x5 + (12544*y4)), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/oy/coykbtm5tcfzfwlxspvibqvg7vn7cfy5xa42shvevbi7vuar7rsk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_79', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 41984
    rnumel = 1224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 656
    x1 = (xindex // 656)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (1224*x0)
        tmp1 = tl.full([1, 1], 802816, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*((r2 + (1224*x0)) % 802816))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x1) + (802816*(((r2 + (1224*x0)) // 12544) % 64)) + ((r2 + (1224*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/fa/cfaurpblbz4xof3hu25juzeshz6xfmxktojiz5qdk46iskr525re.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_80', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 656
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (656*x0)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/37/c37vxstpet2cubpjfldjeb372amg53swzm33g3h7oe2rrith3ayx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_81', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 41984
    rnumel = 1224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (1224*x1)
        tmp1 = tl.full([1, 1], 802816, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (1224*x1)) % 802816))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x0) + (802816*(((r2 + (1224*x1)) // 12544) % 64)) + ((r2 + (1224*x1)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (64*((r2 + (1224*x1)) % 802816))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.where(tmp2, tmp11, 0)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/zk/czkyrzetfug6zvptpxirgshrutnv5tlrw4ef5tyox7trw2j6x744.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_82', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/zr/czr5tpivqkv3mrea54bgijvigdvnrcxxpsooil32ksq7x5clkfuc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 802816
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 1.2456154336734693e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp21, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_10, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_16, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_28, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_31, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_34, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_37, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_46, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_52, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_55, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_58, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_61, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_64, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_67, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_70, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_73, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_79, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_85, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_88, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_91, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_94, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_100, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_103, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_106, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_109, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_112, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_115, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_118, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_121, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_124, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_127, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_130, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_133, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_136, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_142, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_145, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_148, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_151, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_154, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_157, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_321, (64, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (64, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (64, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem_2, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(getitem_3, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_1, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu_1, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_2, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_3, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_10, (256, ), (1, ))
    assert_size_stride(convolution_4, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_13, (256, ), (1, ))
    assert_size_stride(relu_3, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_5, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(relu_4, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_6, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(relu_5, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_7, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_22, (256, ), (1, ))
    assert_size_stride(relu_6, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_8, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_7, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_9, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_28, (64, ), (1, ))
    assert_size_stride(relu_8, (64, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_10, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(relu_9, (64, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convolution_11, (64, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(relu_10, (64, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_12, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(relu_11, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_13, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_40, (512, ), (1, ))
    assert_size_stride(convolution_14, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_43, (512, ), (1, ))
    assert_size_stride(relu_12, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_15, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(relu_13, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_16, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_49, (128, ), (1, ))
    assert_size_stride(relu_14, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_17, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_52, (512, ), (1, ))
    assert_size_stride(relu_15, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_18, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_55, (128, ), (1, ))
    assert_size_stride(relu_16, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_19, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(relu_17, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_20, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_61, (512, ), (1, ))
    assert_size_stride(relu_18, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_21, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(relu_19, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_22, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(relu_20, (64, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convolution_23, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_70, (512, ), (1, ))
    assert_size_stride(relu_21, (64, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convolution_24, (64, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(relu_22, (64, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_25, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_76, (256, ), (1, ))
    assert_size_stride(relu_23, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_26, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_79, (1024, ), (1, ))
    assert_size_stride(convolution_27, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_82, (1024, ), (1, ))
    assert_size_stride(relu_24, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_28, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(relu_25, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_29, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_88, (256, ), (1, ))
    assert_size_stride(relu_26, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_30, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_91, (1024, ), (1, ))
    assert_size_stride(relu_27, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_31, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_94, (256, ), (1, ))
    assert_size_stride(relu_28, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_32, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_97, (256, ), (1, ))
    assert_size_stride(relu_29, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_33, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_100, (1024, ), (1, ))
    assert_size_stride(relu_30, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_34, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_103, (256, ), (1, ))
    assert_size_stride(relu_31, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_35, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_106, (256, ), (1, ))
    assert_size_stride(relu_32, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_36, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_109, (1024, ), (1, ))
    assert_size_stride(relu_33, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_37, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_112, (256, ), (1, ))
    assert_size_stride(relu_34, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_38, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(relu_35, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_39, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_118, (1024, ), (1, ))
    assert_size_stride(relu_36, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_40, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_121, (256, ), (1, ))
    assert_size_stride(relu_37, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_41, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_124, (256, ), (1, ))
    assert_size_stride(relu_38, (64, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convolution_42, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_127, (1024, ), (1, ))
    assert_size_stride(relu_39, (64, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convolution_43, (64, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_130, (512, ), (1, ))
    assert_size_stride(relu_40, (64, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_44, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_133, (512, ), (1, ))
    assert_size_stride(relu_41, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_45, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_136, (2048, ), (1, ))
    assert_size_stride(convolution_46, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_139, (2048, ), (1, ))
    assert_size_stride(relu_42, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_47, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_142, (512, ), (1, ))
    assert_size_stride(relu_43, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_48, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_145, (512, ), (1, ))
    assert_size_stride(relu_44, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_49, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_148, (2048, ), (1, ))
    assert_size_stride(relu_45, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convolution_50, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_151, (512, ), (1, ))
    assert_size_stride(relu_46, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_51, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_154, (512, ), (1, ))
    assert_size_stride(relu_47, (64, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convolution_52, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_157, (2048, ), (1, ))
    assert_size_stride(view, (64, 2048), (2048, 1))
    assert_size_stride(permute_1, (1000, 2048), (2048, 1))
    assert_size_stride(le, (64, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(unsqueeze_214, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_238, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_514, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_586, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_682, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_706, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_718, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_730, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_754, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_766, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_778, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_790, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_802, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_826, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_838, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (64, ), (1, ))
    assert_size_stride(tangents_2, (64, ), (1, ))
    assert_size_stride(tangents_3, (), ())
    assert_size_stride(tangents_4, (64, ), (1, ))
    assert_size_stride(tangents_5, (64, ), (1, ))
    assert_size_stride(tangents_6, (), ())
    assert_size_stride(tangents_7, (64, ), (1, ))
    assert_size_stride(tangents_8, (64, ), (1, ))
    assert_size_stride(tangents_9, (), ())
    assert_size_stride(tangents_10, (256, ), (1, ))
    assert_size_stride(tangents_11, (256, ), (1, ))
    assert_size_stride(tangents_12, (), ())
    assert_size_stride(tangents_13, (256, ), (1, ))
    assert_size_stride(tangents_14, (256, ), (1, ))
    assert_size_stride(tangents_15, (), ())
    assert_size_stride(tangents_16, (64, ), (1, ))
    assert_size_stride(tangents_17, (64, ), (1, ))
    assert_size_stride(tangents_18, (), ())
    assert_size_stride(tangents_19, (64, ), (1, ))
    assert_size_stride(tangents_20, (64, ), (1, ))
    assert_size_stride(tangents_21, (), ())
    assert_size_stride(tangents_22, (256, ), (1, ))
    assert_size_stride(tangents_23, (256, ), (1, ))
    assert_size_stride(tangents_24, (), ())
    assert_size_stride(tangents_25, (64, ), (1, ))
    assert_size_stride(tangents_26, (64, ), (1, ))
    assert_size_stride(tangents_27, (), ())
    assert_size_stride(tangents_28, (64, ), (1, ))
    assert_size_stride(tangents_29, (64, ), (1, ))
    assert_size_stride(tangents_30, (), ())
    assert_size_stride(tangents_31, (256, ), (1, ))
    assert_size_stride(tangents_32, (256, ), (1, ))
    assert_size_stride(tangents_33, (), ())
    assert_size_stride(tangents_34, (128, ), (1, ))
    assert_size_stride(tangents_35, (128, ), (1, ))
    assert_size_stride(tangents_36, (), ())
    assert_size_stride(tangents_37, (128, ), (1, ))
    assert_size_stride(tangents_38, (128, ), (1, ))
    assert_size_stride(tangents_39, (), ())
    assert_size_stride(tangents_40, (512, ), (1, ))
    assert_size_stride(tangents_41, (512, ), (1, ))
    assert_size_stride(tangents_42, (), ())
    assert_size_stride(tangents_43, (512, ), (1, ))
    assert_size_stride(tangents_44, (512, ), (1, ))
    assert_size_stride(tangents_45, (), ())
    assert_size_stride(tangents_46, (128, ), (1, ))
    assert_size_stride(tangents_47, (128, ), (1, ))
    assert_size_stride(tangents_48, (), ())
    assert_size_stride(tangents_49, (128, ), (1, ))
    assert_size_stride(tangents_50, (128, ), (1, ))
    assert_size_stride(tangents_51, (), ())
    assert_size_stride(tangents_52, (512, ), (1, ))
    assert_size_stride(tangents_53, (512, ), (1, ))
    assert_size_stride(tangents_54, (), ())
    assert_size_stride(tangents_55, (128, ), (1, ))
    assert_size_stride(tangents_56, (128, ), (1, ))
    assert_size_stride(tangents_57, (), ())
    assert_size_stride(tangents_58, (128, ), (1, ))
    assert_size_stride(tangents_59, (128, ), (1, ))
    assert_size_stride(tangents_60, (), ())
    assert_size_stride(tangents_61, (512, ), (1, ))
    assert_size_stride(tangents_62, (512, ), (1, ))
    assert_size_stride(tangents_63, (), ())
    assert_size_stride(tangents_64, (128, ), (1, ))
    assert_size_stride(tangents_65, (128, ), (1, ))
    assert_size_stride(tangents_66, (), ())
    assert_size_stride(tangents_67, (128, ), (1, ))
    assert_size_stride(tangents_68, (128, ), (1, ))
    assert_size_stride(tangents_69, (), ())
    assert_size_stride(tangents_70, (512, ), (1, ))
    assert_size_stride(tangents_71, (512, ), (1, ))
    assert_size_stride(tangents_72, (), ())
    assert_size_stride(tangents_73, (256, ), (1, ))
    assert_size_stride(tangents_74, (256, ), (1, ))
    assert_size_stride(tangents_75, (), ())
    assert_size_stride(tangents_76, (256, ), (1, ))
    assert_size_stride(tangents_77, (256, ), (1, ))
    assert_size_stride(tangents_78, (), ())
    assert_size_stride(tangents_79, (1024, ), (1, ))
    assert_size_stride(tangents_80, (1024, ), (1, ))
    assert_size_stride(tangents_81, (), ())
    assert_size_stride(tangents_82, (1024, ), (1, ))
    assert_size_stride(tangents_83, (1024, ), (1, ))
    assert_size_stride(tangents_84, (), ())
    assert_size_stride(tangents_85, (256, ), (1, ))
    assert_size_stride(tangents_86, (256, ), (1, ))
    assert_size_stride(tangents_87, (), ())
    assert_size_stride(tangents_88, (256, ), (1, ))
    assert_size_stride(tangents_89, (256, ), (1, ))
    assert_size_stride(tangents_90, (), ())
    assert_size_stride(tangents_91, (1024, ), (1, ))
    assert_size_stride(tangents_92, (1024, ), (1, ))
    assert_size_stride(tangents_93, (), ())
    assert_size_stride(tangents_94, (256, ), (1, ))
    assert_size_stride(tangents_95, (256, ), (1, ))
    assert_size_stride(tangents_96, (), ())
    assert_size_stride(tangents_97, (256, ), (1, ))
    assert_size_stride(tangents_98, (256, ), (1, ))
    assert_size_stride(tangents_99, (), ())
    assert_size_stride(tangents_100, (1024, ), (1, ))
    assert_size_stride(tangents_101, (1024, ), (1, ))
    assert_size_stride(tangents_102, (), ())
    assert_size_stride(tangents_103, (256, ), (1, ))
    assert_size_stride(tangents_104, (256, ), (1, ))
    assert_size_stride(tangents_105, (), ())
    assert_size_stride(tangents_106, (256, ), (1, ))
    assert_size_stride(tangents_107, (256, ), (1, ))
    assert_size_stride(tangents_108, (), ())
    assert_size_stride(tangents_109, (1024, ), (1, ))
    assert_size_stride(tangents_110, (1024, ), (1, ))
    assert_size_stride(tangents_111, (), ())
    assert_size_stride(tangents_112, (256, ), (1, ))
    assert_size_stride(tangents_113, (256, ), (1, ))
    assert_size_stride(tangents_114, (), ())
    assert_size_stride(tangents_115, (256, ), (1, ))
    assert_size_stride(tangents_116, (256, ), (1, ))
    assert_size_stride(tangents_117, (), ())
    assert_size_stride(tangents_118, (1024, ), (1, ))
    assert_size_stride(tangents_119, (1024, ), (1, ))
    assert_size_stride(tangents_120, (), ())
    assert_size_stride(tangents_121, (256, ), (1, ))
    assert_size_stride(tangents_122, (256, ), (1, ))
    assert_size_stride(tangents_123, (), ())
    assert_size_stride(tangents_124, (256, ), (1, ))
    assert_size_stride(tangents_125, (256, ), (1, ))
    assert_size_stride(tangents_126, (), ())
    assert_size_stride(tangents_127, (1024, ), (1, ))
    assert_size_stride(tangents_128, (1024, ), (1, ))
    assert_size_stride(tangents_129, (), ())
    assert_size_stride(tangents_130, (512, ), (1, ))
    assert_size_stride(tangents_131, (512, ), (1, ))
    assert_size_stride(tangents_132, (), ())
    assert_size_stride(tangents_133, (512, ), (1, ))
    assert_size_stride(tangents_134, (512, ), (1, ))
    assert_size_stride(tangents_135, (), ())
    assert_size_stride(tangents_136, (2048, ), (1, ))
    assert_size_stride(tangents_137, (2048, ), (1, ))
    assert_size_stride(tangents_138, (), ())
    assert_size_stride(tangents_139, (2048, ), (1, ))
    assert_size_stride(tangents_140, (2048, ), (1, ))
    assert_size_stride(tangents_141, (), ())
    assert_size_stride(tangents_142, (512, ), (1, ))
    assert_size_stride(tangents_143, (512, ), (1, ))
    assert_size_stride(tangents_144, (), ())
    assert_size_stride(tangents_145, (512, ), (1, ))
    assert_size_stride(tangents_146, (512, ), (1, ))
    assert_size_stride(tangents_147, (), ())
    assert_size_stride(tangents_148, (2048, ), (1, ))
    assert_size_stride(tangents_149, (2048, ), (1, ))
    assert_size_stride(tangents_150, (), ())
    assert_size_stride(tangents_151, (512, ), (1, ))
    assert_size_stride(tangents_152, (512, ), (1, ))
    assert_size_stride(tangents_153, (), ())
    assert_size_stride(tangents_154, (512, ), (1, ))
    assert_size_stride(tangents_155, (512, ), (1, ))
    assert_size_stride(tangents_156, (), ())
    assert_size_stride(tangents_157, (2048, ), (1, ))
    assert_size_stride(tangents_158, (2048, ), (1, ))
    assert_size_stride(tangents_159, (), ())
    assert_size_stride(tangents_160, (64, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 2048), (2048, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        stream0 = get_cuda_stream(0)
        triton_tem_fused_mm_0.run(tangents_160, permute_1, buf0, grid=torch._inductor.kernel.mm_common.mm_grid(64, 2048, meta0), stream=stream0)
        del permute_1
        buf1 = empty_strided((1000, 2048), (2048, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        triton_tem_fused_mm_1.run(tangents_160, view, buf1, grid=torch._inductor.kernel.mm_common.mm_grid(1000, 2048, meta1), stream=stream0)
        del view
        buf2 = empty_strided((1, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_2.run(tangents_160, buf2, 1000, 64, grid=grid(1000), stream=stream0)
        del tangents_160
        buf3 = empty_strided((2048, 25), (1, 2048), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((2048, 25), (1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_div_native_batch_norm_backward_threshold_backward_3.run(le, buf0, convolution_52, unsqueeze_214, buf3, buf5, 51200, 126, grid=grid(51200), stream=stream0)
        buf4 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_4.run(buf3, buf4, 2048, 25, grid=grid(2048), stream=stream0)
        buf6 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_5.run(buf5, squeeze_157, buf6, buf7, 2048, 25, grid=grid(2048), stream=stream0)
        buf8 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_6.run(le, buf0, convolution_52, unsqueeze_214, buf6, squeeze_157, buf4, primals_158, buf8, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_52
        del primals_158
        del squeeze_157
        del unsqueeze_214
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf9 = aten.convolution_backward(buf8, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_157
        buf10 = buf9[0]
        assert_size_stride(buf10, (64, 512, 7, 7), (25088, 49, 7, 1))
        buf11 = buf9[1]
        assert_size_stride(buf11, (2048, 512, 1, 1), (512, 1, 1, 1))
        del buf9
        buf12 = empty_strided((512, 25), (1, 512), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((512, 25), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_7.run(relu_47, buf10, convolution_51, unsqueeze_226, buf12, buf14, 12800, 126, grid=grid(12800), stream=stream0)
        buf13 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf12, buf13, 512, 25, grid=grid(512), stream=stream0)
        buf15 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf14, squeeze_154, buf15, buf16, 512, 25, grid=grid(512), stream=stream0)
        buf17 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(relu_47, buf10, convolution_51, unsqueeze_226, buf15, squeeze_154, buf13, primals_155, buf17, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del buf10
        del convolution_51
        del primals_155
        del relu_47
        del squeeze_154
        del unsqueeze_226
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf18 = aten.convolution_backward(buf17, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_154
        buf19 = buf18[0]
        assert_size_stride(buf19, (64, 512, 7, 7), (25088, 49, 7, 1))
        buf20 = buf18[1]
        assert_size_stride(buf20, (512, 512, 3, 3), (4608, 9, 3, 1))
        del buf18
        buf21 = buf14; del buf14  # reuse
        buf23 = buf12; del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_7.run(relu_46, buf19, convolution_50, unsqueeze_238, buf21, buf23, 12800, 126, grid=grid(12800), stream=stream0)
        buf22 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf21, buf22, 512, 25, grid=grid(512), stream=stream0)
        buf24 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf23, squeeze_151, buf24, buf25, 512, 25, grid=grid(512), stream=stream0)
        buf26 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(relu_46, buf19, convolution_50, unsqueeze_238, buf24, squeeze_151, buf22, primals_152, buf26, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del buf19
        del convolution_50
        del primals_152
        del relu_46
        del squeeze_151
        del unsqueeze_238
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf27 = aten.convolution_backward(buf26, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_151
        buf28 = buf27[0]
        assert_size_stride(buf28, (64, 2048, 7, 7), (100352, 49, 7, 1))
        buf29 = buf27[1]
        assert_size_stride(buf29, (512, 2048, 1, 1), (2048, 1, 1, 1))
        del buf27
        buf30 = buf5; del buf5  # reuse
        buf32 = buf3; del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_11.run(relu_45, le, buf0, buf28, convolution_49, unsqueeze_250, buf30, buf32, 51200, 126, grid=grid(51200), stream=stream0)
        buf31 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_4.run(buf30, buf31, 2048, 25, grid=grid(2048), stream=stream0)
        buf33 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf35 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_5.run(buf32, squeeze_148, buf33, buf35, 2048, 25, grid=grid(2048), stream=stream0)
        buf34 = buf8; del buf8  # reuse
        buf36 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_12.run(buf36, relu_45, le, buf0, buf28, convolution_49, unsqueeze_250, buf33, squeeze_148, buf31, primals_149, 3136, 2048, grid=grid(3136, 2048), stream=stream0)
        del convolution_49
        del primals_149
        del squeeze_148
        del unsqueeze_250
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf37 = aten.convolution_backward(buf36, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_148
        buf38 = buf37[0]
        assert_size_stride(buf38, (64, 512, 7, 7), (25088, 49, 7, 1))
        buf39 = buf37[1]
        assert_size_stride(buf39, (2048, 512, 1, 1), (512, 1, 1, 1))
        del buf37
        buf40 = buf23; del buf23  # reuse
        buf42 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_7.run(relu_44, buf38, convolution_48, unsqueeze_262, buf40, buf42, 12800, 126, grid=grid(12800), stream=stream0)
        buf41 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf40, buf41, 512, 25, grid=grid(512), stream=stream0)
        buf43 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf42, squeeze_145, buf43, buf44, 512, 25, grid=grid(512), stream=stream0)
        buf45 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(relu_44, buf38, convolution_48, unsqueeze_262, buf43, squeeze_145, buf41, primals_146, buf45, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del buf38
        del convolution_48
        del primals_146
        del relu_44
        del squeeze_145
        del unsqueeze_262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf46 = aten.convolution_backward(buf45, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_145
        buf47 = buf46[0]
        assert_size_stride(buf47, (64, 512, 7, 7), (25088, 49, 7, 1))
        buf48 = buf46[1]
        assert_size_stride(buf48, (512, 512, 3, 3), (4608, 9, 3, 1))
        del buf46
        buf49 = buf42; del buf42  # reuse
        buf51 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_7.run(relu_43, buf47, convolution_47, unsqueeze_274, buf49, buf51, 12800, 126, grid=grid(12800), stream=stream0)
        buf50 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf49, buf50, 512, 25, grid=grid(512), stream=stream0)
        buf52 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf53 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf51, squeeze_142, buf52, buf53, 512, 25, grid=grid(512), stream=stream0)
        buf54 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(relu_43, buf47, convolution_47, unsqueeze_274, buf52, squeeze_142, buf50, primals_143, buf54, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del buf47
        del convolution_47
        del primals_143
        del relu_43
        del squeeze_142
        del unsqueeze_274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf55 = aten.convolution_backward(buf54, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_142
        buf56 = buf55[0]
        assert_size_stride(buf56, (64, 2048, 7, 7), (100352, 49, 7, 1))
        buf57 = buf55[1]
        assert_size_stride(buf57, (512, 2048, 1, 1), (2048, 1, 1, 1))
        del buf55
        buf58 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]
        triton_poi_fused_add_div_threshold_backward_13.run(relu_42, relu_45, le, buf0, buf28, buf56, buf58, 3136, 2048, grid=grid(3136, 2048), stream=stream0)
        del buf0
        del le
        del relu_42
        del relu_45
        buf59 = buf32; del buf32  # reuse
        buf61 = buf30; del buf30  # reuse
        buf68 = empty_strided((2048, 25), (1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_14.run(buf58, convolution_46, unsqueeze_286, convolution_45, unsqueeze_298, buf59, buf61, buf68, 51200, 126, grid=grid(51200), stream=stream0)
        buf60 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_4.run(buf59, buf60, 2048, 25, grid=grid(2048), stream=stream0)
        del buf59
        buf62 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf63 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_5.run(buf61, squeeze_139, buf62, buf63, 2048, 25, grid=grid(2048), stream=stream0)
        del buf61
        buf69 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_5.run(buf68, squeeze_136, buf69, buf70, 2048, 25, grid=grid(2048), stream=stream0)
        del buf68
        buf64 = reinterpret_tensor(buf56, (64, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf56  # reuse
        buf71 = reinterpret_tensor(buf28, (64, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_15.run(buf58, convolution_46, unsqueeze_286, buf62, squeeze_139, buf60, primals_140, convolution_45, unsqueeze_298, buf69, squeeze_136, primals_137, buf64, buf71, 6422528, grid=grid(6422528), stream=stream0)
        del buf58
        del buf62
        del buf69
        del convolution_45
        del convolution_46
        del primals_137
        del primals_140
        del squeeze_136
        del squeeze_139
        del unsqueeze_286
        del unsqueeze_298
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf65 = aten.convolution_backward(buf64, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf64
        del primals_139
        buf66 = buf65[0]
        assert_size_stride(buf66, (64, 1024, 14, 14), (200704, 196, 14, 1))
        buf67 = buf65[1]
        assert_size_stride(buf67, (2048, 1024, 1, 1), (1024, 1, 1, 1))
        del buf65
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf72 = aten.convolution_backward(buf71, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_136
        buf73 = buf72[0]
        assert_size_stride(buf73, (64, 512, 7, 7), (25088, 49, 7, 1))
        buf74 = buf72[1]
        assert_size_stride(buf74, (2048, 512, 1, 1), (512, 1, 1, 1))
        del buf72
        buf75 = buf51; del buf51  # reuse
        buf77 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_7.run(relu_41, buf73, convolution_44, unsqueeze_310, buf75, buf77, 12800, 126, grid=grid(12800), stream=stream0)
        buf76 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_8.run(buf75, buf76, 512, 25, grid=grid(512), stream=stream0)
        del buf75
        buf78 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf79 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_9.run(buf77, squeeze_133, buf78, buf79, 512, 25, grid=grid(512), stream=stream0)
        del buf77
        buf80 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10.run(relu_41, buf73, convolution_44, unsqueeze_310, buf78, squeeze_133, buf76, primals_134, buf80, 3136, 512, grid=grid(3136, 512), stream=stream0)
        del buf73
        del convolution_44
        del primals_134
        del relu_41
        del squeeze_133
        del unsqueeze_310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf81 = aten.convolution_backward(buf80, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf80
        del primals_133
        buf82 = buf81[0]
        assert_size_stride(buf82, (64, 512, 14, 14), (100352, 196, 14, 1))
        buf83 = buf81[1]
        assert_size_stride(buf83, (512, 512, 3, 3), (4608, 9, 3, 1))
        del buf81
        buf84 = empty_strided((512, 98), (98, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_16.run(relu_40, buf82, buf84, 50176, 128, grid=grid(50176), stream=stream0)
        buf85 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_17.run(buf84, buf85, 512, 98, grid=grid(512), stream=stream0)
        buf86 = reinterpret_tensor(buf84, (512, 98), (1, 512), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_18.run(relu_40, buf82, convolution_43, unsqueeze_322, buf86, 50176, 128, grid=grid(50176), stream=stream0)
        buf87 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf88 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(buf86, squeeze_130, buf87, buf88, 512, 98, grid=grid(512), stream=stream0)
        buf89 = reinterpret_tensor(buf71, (64, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20.run(relu_40, buf82, convolution_43, unsqueeze_322, buf87, squeeze_130, buf85, primals_131, buf89, 12544, 512, grid=grid(12544, 512), stream=stream0)
        del buf82
        del convolution_43
        del primals_131
        del relu_40
        del squeeze_130
        del unsqueeze_322
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf90 = aten.convolution_backward(buf89, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_130
        buf91 = buf90[0]
        assert_size_stride(buf91, (64, 1024, 14, 14), (200704, 196, 14, 1))
        buf92 = buf90[1]
        assert_size_stride(buf92, (512, 1024, 1, 1), (1024, 1, 1, 1))
        del buf90
        buf93 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_39, buf66, buf91, buf93, 1024, 12544, grid=grid(1024), stream=stream0)
        buf94 = empty_strided((1024, 98), (98, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_22.run(relu_39, buf66, buf91, convolution_42, unsqueeze_334, buf94, 100352, 128, grid=grid(100352), stream=stream0)
        buf95 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf97 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf94, squeeze_127, buf95, buf97, 1024, 98, grid=grid(1024), stream=stream0)
        buf96 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24.run(relu_39, buf66, buf91, convolution_42, unsqueeze_334, buf95, squeeze_127, buf93, primals_128, buf96, 12544, 1024, grid=grid(12544, 1024), stream=stream0)
        del convolution_42
        del primals_128
        del squeeze_127
        del unsqueeze_334
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf98 = aten.convolution_backward(buf96, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf96
        del primals_127
        buf99 = buf98[0]
        assert_size_stride(buf99, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf100 = buf98[1]
        assert_size_stride(buf100, (1024, 256, 1, 1), (256, 1, 1, 1))
        del buf98
        buf101 = empty_strided((256, 98), (98, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_38, buf99, buf101, 25088, 128, grid=grid(25088), stream=stream0)
        buf102 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf101, buf102, 256, 98, grid=grid(256), stream=stream0)
        buf103 = reinterpret_tensor(buf101, (256, 98), (1, 256), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_38, buf99, convolution_41, unsqueeze_346, buf103, 25088, 128, grid=grid(25088), stream=stream0)
        buf104 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf105 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf103, squeeze_124, buf104, buf105, 256, 98, grid=grid(256), stream=stream0)
        buf106 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_38, buf99, convolution_41, unsqueeze_346, buf104, squeeze_124, buf102, primals_125, buf106, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf99
        del convolution_41
        del primals_125
        del relu_38
        del squeeze_124
        del unsqueeze_346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf107 = aten.convolution_backward(buf106, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_124
        buf108 = buf107[0]
        assert_size_stride(buf108, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf109 = buf107[1]
        assert_size_stride(buf109, (256, 256, 3, 3), (2304, 9, 3, 1))
        del buf107
        buf110 = reinterpret_tensor(buf103, (256, 98), (98, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_37, buf108, buf110, 25088, 128, grid=grid(25088), stream=stream0)
        buf111 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf110, buf111, 256, 98, grid=grid(256), stream=stream0)
        buf112 = reinterpret_tensor(buf110, (256, 98), (1, 256), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_37, buf108, convolution_40, unsqueeze_358, buf112, 25088, 128, grid=grid(25088), stream=stream0)
        buf113 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf114 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf112, squeeze_121, buf113, buf114, 256, 98, grid=grid(256), stream=stream0)
        buf115 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_37, buf108, convolution_40, unsqueeze_358, buf113, squeeze_121, buf111, primals_122, buf115, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf108
        del convolution_40
        del primals_122
        del relu_37
        del squeeze_121
        del unsqueeze_358
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf116 = aten.convolution_backward(buf115, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_121
        buf117 = buf116[0]
        assert_size_stride(buf117, (64, 1024, 14, 14), (200704, 196, 14, 1))
        buf118 = buf116[1]
        assert_size_stride(buf118, (256, 1024, 1, 1), (1024, 1, 1, 1))
        del buf116
        buf119 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_30.run(buf119, relu_36, relu_39, buf66, buf91, 65536, 196, grid=grid(65536, 196), stream=stream0)
        del buf66
        del relu_36
        del relu_39
        buf120 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf119, buf120, 1024, 12544, grid=grid(1024), stream=stream0)
        buf121 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf119, convolution_39, unsqueeze_370, buf121, 100352, 128, grid=grid(100352), stream=stream0)
        buf122 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf123 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf121, squeeze_118, buf122, buf123, 1024, 98, grid=grid(1024), stream=stream0)
        buf124 = reinterpret_tensor(buf91, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf119, convolution_39, unsqueeze_370, buf122, squeeze_118, buf120, primals_119, buf124, 12544, 1024, grid=grid(12544, 1024), stream=stream0)
        del convolution_39
        del primals_119
        del squeeze_118
        del unsqueeze_370
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf125 = aten.convolution_backward(buf124, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_118
        buf126 = buf125[0]
        assert_size_stride(buf126, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf127 = buf125[1]
        assert_size_stride(buf127, (1024, 256, 1, 1), (256, 1, 1, 1))
        del buf125
        buf128 = reinterpret_tensor(buf112, (256, 98), (98, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_35, buf126, buf128, 25088, 128, grid=grid(25088), stream=stream0)
        buf129 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf128, buf129, 256, 98, grid=grid(256), stream=stream0)
        buf130 = reinterpret_tensor(buf128, (256, 98), (1, 256), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_35, buf126, convolution_38, unsqueeze_382, buf130, 25088, 128, grid=grid(25088), stream=stream0)
        buf131 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf132 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf130, squeeze_115, buf131, buf132, 256, 98, grid=grid(256), stream=stream0)
        buf133 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_35, buf126, convolution_38, unsqueeze_382, buf131, squeeze_115, buf129, primals_116, buf133, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf126
        del convolution_38
        del primals_116
        del relu_35
        del squeeze_115
        del unsqueeze_382
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf134 = aten.convolution_backward(buf133, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_115
        buf135 = buf134[0]
        assert_size_stride(buf135, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf136 = buf134[1]
        assert_size_stride(buf136, (256, 256, 3, 3), (2304, 9, 3, 1))
        del buf134
        buf137 = reinterpret_tensor(buf130, (256, 98), (98, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_34, buf135, buf137, 25088, 128, grid=grid(25088), stream=stream0)
        buf138 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf137, buf138, 256, 98, grid=grid(256), stream=stream0)
        buf139 = reinterpret_tensor(buf137, (256, 98), (1, 256), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_34, buf135, convolution_37, unsqueeze_394, buf139, 25088, 128, grid=grid(25088), stream=stream0)
        buf140 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf141 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf139, squeeze_112, buf140, buf141, 256, 98, grid=grid(256), stream=stream0)
        buf142 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_34, buf135, convolution_37, unsqueeze_394, buf140, squeeze_112, buf138, primals_113, buf142, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf135
        del convolution_37
        del primals_113
        del relu_34
        del squeeze_112
        del unsqueeze_394
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf143 = aten.convolution_backward(buf142, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_112
        buf144 = buf143[0]
        assert_size_stride(buf144, (64, 1024, 14, 14), (200704, 196, 14, 1))
        buf145 = buf143[1]
        assert_size_stride(buf145, (256, 1024, 1, 1), (1024, 1, 1, 1))
        del buf143
        buf146 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_33, buf119, buf144, buf146, 1024, 12544, grid=grid(1024), stream=stream0)
        buf147 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_22.run(relu_33, buf119, buf144, convolution_36, unsqueeze_406, buf147, 100352, 128, grid=grid(100352), stream=stream0)
        buf148 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf150 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf147, squeeze_109, buf148, buf150, 1024, 98, grid=grid(1024), stream=stream0)
        buf149 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24.run(relu_33, buf119, buf144, convolution_36, unsqueeze_406, buf148, squeeze_109, buf146, primals_110, buf149, 12544, 1024, grid=grid(12544, 1024), stream=stream0)
        del convolution_36
        del primals_110
        del squeeze_109
        del unsqueeze_406
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf151 = aten.convolution_backward(buf149, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf149
        del primals_109
        buf152 = buf151[0]
        assert_size_stride(buf152, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf153 = buf151[1]
        assert_size_stride(buf153, (1024, 256, 1, 1), (256, 1, 1, 1))
        del buf151
        buf154 = reinterpret_tensor(buf139, (256, 98), (98, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_32, buf152, buf154, 25088, 128, grid=grid(25088), stream=stream0)
        buf155 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf154, buf155, 256, 98, grid=grid(256), stream=stream0)
        buf156 = reinterpret_tensor(buf154, (256, 98), (1, 256), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_32, buf152, convolution_35, unsqueeze_418, buf156, 25088, 128, grid=grid(25088), stream=stream0)
        buf157 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf158 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf156, squeeze_106, buf157, buf158, 256, 98, grid=grid(256), stream=stream0)
        buf159 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_32, buf152, convolution_35, unsqueeze_418, buf157, squeeze_106, buf155, primals_107, buf159, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf152
        del convolution_35
        del primals_107
        del relu_32
        del squeeze_106
        del unsqueeze_418
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf160 = aten.convolution_backward(buf159, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_106
        buf161 = buf160[0]
        assert_size_stride(buf161, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf162 = buf160[1]
        assert_size_stride(buf162, (256, 256, 3, 3), (2304, 9, 3, 1))
        del buf160
        buf163 = reinterpret_tensor(buf156, (256, 98), (98, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_31, buf161, buf163, 25088, 128, grid=grid(25088), stream=stream0)
        buf164 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf163, buf164, 256, 98, grid=grid(256), stream=stream0)
        buf165 = reinterpret_tensor(buf163, (256, 98), (1, 256), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_31, buf161, convolution_34, unsqueeze_430, buf165, 25088, 128, grid=grid(25088), stream=stream0)
        buf166 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf167 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf165, squeeze_103, buf166, buf167, 256, 98, grid=grid(256), stream=stream0)
        buf168 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_31, buf161, convolution_34, unsqueeze_430, buf166, squeeze_103, buf164, primals_104, buf168, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf161
        del convolution_34
        del primals_104
        del relu_31
        del squeeze_103
        del unsqueeze_430
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf169 = aten.convolution_backward(buf168, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_103
        buf170 = buf169[0]
        assert_size_stride(buf170, (64, 1024, 14, 14), (200704, 196, 14, 1))
        buf171 = buf169[1]
        assert_size_stride(buf171, (256, 1024, 1, 1), (1024, 1, 1, 1))
        del buf169
        buf172 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_34.run(buf172, relu_30, relu_33, buf144, buf170, 65536, 196, grid=grid(65536, 196), stream=stream0)
        del buf144
        del relu_30
        del relu_33
        buf173 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf172, buf173, 1024, 12544, grid=grid(1024), stream=stream0)
        buf174 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_32.run(buf172, convolution_33, unsqueeze_442, buf174, 100352, 128, grid=grid(100352), stream=stream0)
        buf175 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf176 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf174, squeeze_100, buf175, buf176, 1024, 98, grid=grid(1024), stream=stream0)
        buf177 = reinterpret_tensor(buf170, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_33.run(buf172, convolution_33, unsqueeze_442, buf175, squeeze_100, buf173, primals_101, buf177, 12544, 1024, grid=grid(12544, 1024), stream=stream0)
        del convolution_33
        del primals_101
        del squeeze_100
        del unsqueeze_442
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf178 = aten.convolution_backward(buf177, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_100
        buf179 = buf178[0]
        assert_size_stride(buf179, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf180 = buf178[1]
        assert_size_stride(buf180, (1024, 256, 1, 1), (256, 1, 1, 1))
        del buf178
        buf181 = reinterpret_tensor(buf165, (256, 98), (98, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_29, buf179, buf181, 25088, 128, grid=grid(25088), stream=stream0)
        buf182 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf181, buf182, 256, 98, grid=grid(256), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (256, 98), (1, 256), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_29, buf179, convolution_32, unsqueeze_454, buf183, 25088, 128, grid=grid(25088), stream=stream0)
        buf184 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf185 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf183, squeeze_97, buf184, buf185, 256, 98, grid=grid(256), stream=stream0)
        buf186 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_29, buf179, convolution_32, unsqueeze_454, buf184, squeeze_97, buf182, primals_98, buf186, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf179
        del convolution_32
        del primals_98
        del relu_29
        del squeeze_97
        del unsqueeze_454
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf187 = aten.convolution_backward(buf186, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_97
        buf188 = buf187[0]
        assert_size_stride(buf188, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf189 = buf187[1]
        assert_size_stride(buf189, (256, 256, 3, 3), (2304, 9, 3, 1))
        del buf187
        buf190 = reinterpret_tensor(buf183, (256, 98), (98, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_28, buf188, buf190, 25088, 128, grid=grid(25088), stream=stream0)
        buf191 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf190, buf191, 256, 98, grid=grid(256), stream=stream0)
        buf192 = reinterpret_tensor(buf190, (256, 98), (1, 256), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_28, buf188, convolution_31, unsqueeze_466, buf192, 25088, 128, grid=grid(25088), stream=stream0)
        buf193 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf194 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf192, squeeze_94, buf193, buf194, 256, 98, grid=grid(256), stream=stream0)
        buf195 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_28, buf188, convolution_31, unsqueeze_466, buf193, squeeze_94, buf191, primals_95, buf195, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf188
        del convolution_31
        del primals_95
        del relu_28
        del squeeze_94
        del unsqueeze_466
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf196 = aten.convolution_backward(buf195, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_94
        buf197 = buf196[0]
        assert_size_stride(buf197, (64, 1024, 14, 14), (200704, 196, 14, 1))
        buf198 = buf196[1]
        assert_size_stride(buf198, (256, 1024, 1, 1), (1024, 1, 1, 1))
        del buf196
        buf199 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_21.run(relu_27, buf172, buf197, buf199, 1024, 12544, grid=grid(1024), stream=stream0)
        buf200 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_22.run(relu_27, buf172, buf197, convolution_30, unsqueeze_478, buf200, 100352, 128, grid=grid(100352), stream=stream0)
        buf201 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf203 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf200, squeeze_91, buf201, buf203, 1024, 98, grid=grid(1024), stream=stream0)
        buf202 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_24.run(relu_27, buf172, buf197, convolution_30, unsqueeze_478, buf201, squeeze_91, buf199, primals_92, buf202, 12544, 1024, grid=grid(12544, 1024), stream=stream0)
        del convolution_30
        del primals_92
        del squeeze_91
        del unsqueeze_478
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf204 = aten.convolution_backward(buf202, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf202
        del primals_91
        buf205 = buf204[0]
        assert_size_stride(buf205, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf206 = buf204[1]
        assert_size_stride(buf206, (1024, 256, 1, 1), (256, 1, 1, 1))
        del buf204
        buf207 = reinterpret_tensor(buf192, (256, 98), (98, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_26, buf205, buf207, 25088, 128, grid=grid(25088), stream=stream0)
        buf208 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf207, buf208, 256, 98, grid=grid(256), stream=stream0)
        buf209 = reinterpret_tensor(buf207, (256, 98), (1, 256), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_26, buf205, convolution_29, unsqueeze_490, buf209, 25088, 128, grid=grid(25088), stream=stream0)
        buf210 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf211 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf209, squeeze_88, buf210, buf211, 256, 98, grid=grid(256), stream=stream0)
        buf212 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_26, buf205, convolution_29, unsqueeze_490, buf210, squeeze_88, buf208, primals_89, buf212, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf205
        del convolution_29
        del primals_89
        del relu_26
        del squeeze_88
        del unsqueeze_490
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf213 = aten.convolution_backward(buf212, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_88
        buf214 = buf213[0]
        assert_size_stride(buf214, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf215 = buf213[1]
        assert_size_stride(buf215, (256, 256, 3, 3), (2304, 9, 3, 1))
        del buf213
        buf216 = reinterpret_tensor(buf209, (256, 98), (98, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_25, buf214, buf216, 25088, 128, grid=grid(25088), stream=stream0)
        buf217 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf216, buf217, 256, 98, grid=grid(256), stream=stream0)
        buf218 = reinterpret_tensor(buf216, (256, 98), (1, 256), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_25, buf214, convolution_28, unsqueeze_502, buf218, 25088, 128, grid=grid(25088), stream=stream0)
        buf219 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf220 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf218, squeeze_85, buf219, buf220, 256, 98, grid=grid(256), stream=stream0)
        buf221 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_25, buf214, convolution_28, unsqueeze_502, buf219, squeeze_85, buf217, primals_86, buf221, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf214
        del convolution_28
        del primals_86
        del relu_25
        del squeeze_85
        del unsqueeze_502
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf222 = aten.convolution_backward(buf221, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_85
        buf223 = buf222[0]
        assert_size_stride(buf223, (64, 1024, 14, 14), (200704, 196, 14, 1))
        buf224 = buf222[1]
        assert_size_stride(buf224, (256, 1024, 1, 1), (1024, 1, 1, 1))
        del buf222
        buf225 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_34.run(buf225, relu_24, relu_27, buf197, buf223, 65536, 196, grid=grid(65536, 196), stream=stream0)
        del relu_24
        del relu_27
        buf226 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_31.run(buf225, buf226, 1024, 12544, grid=grid(1024), stream=stream0)
        buf227 = buf200; del buf200  # reuse
        buf234 = empty_strided((1024, 98), (98, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_35.run(buf225, convolution_27, unsqueeze_514, convolution_26, unsqueeze_526, buf227, buf234, 100352, 128, grid=grid(100352), stream=stream0)
        buf228 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf229 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf227, squeeze_82, buf228, buf229, 1024, 98, grid=grid(1024), stream=stream0)
        buf235 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf236 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf234, squeeze_79, buf235, buf236, 1024, 98, grid=grid(1024), stream=stream0)
        buf230 = reinterpret_tensor(buf223, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf223  # reuse
        buf237 = reinterpret_tensor(buf197, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_36.run(buf225, convolution_27, unsqueeze_514, buf228, squeeze_82, buf226, primals_83, convolution_26, unsqueeze_526, buf235, squeeze_79, primals_80, buf230, buf237, 12544, 1024, grid=grid(12544, 1024), stream=stream0)
        del buf225
        del buf228
        del buf235
        del convolution_26
        del convolution_27
        del primals_80
        del primals_83
        del squeeze_79
        del squeeze_82
        del unsqueeze_514
        del unsqueeze_526
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf231 = aten.convolution_backward(buf230, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf230
        del primals_82
        buf232 = buf231[0]
        assert_size_stride(buf232, (64, 512, 28, 28), (401408, 784, 28, 1))
        buf233 = buf231[1]
        assert_size_stride(buf233, (1024, 512, 1, 1), (512, 1, 1, 1))
        del buf231
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf238 = aten.convolution_backward(buf237, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_79
        buf239 = buf238[0]
        assert_size_stride(buf239, (64, 256, 14, 14), (50176, 196, 14, 1))
        buf240 = buf238[1]
        assert_size_stride(buf240, (1024, 256, 1, 1), (256, 1, 1, 1))
        del buf238
        buf241 = reinterpret_tensor(buf218, (256, 98), (98, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_25.run(relu_23, buf239, buf241, 25088, 128, grid=grid(25088), stream=stream0)
        buf242 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_26.run(buf241, buf242, 256, 98, grid=grid(256), stream=stream0)
        buf243 = reinterpret_tensor(buf241, (256, 98), (1, 256), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_27.run(relu_23, buf239, convolution_25, unsqueeze_538, buf243, 25088, 128, grid=grid(25088), stream=stream0)
        buf244 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf245 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_28.run(buf243, squeeze_76, buf244, buf245, 256, 98, grid=grid(256), stream=stream0)
        del buf243
        buf246 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(relu_23, buf239, convolution_25, unsqueeze_538, buf244, squeeze_76, buf242, primals_77, buf246, 12544, 256, grid=grid(12544, 256), stream=stream0)
        del buf239
        del convolution_25
        del primals_77
        del relu_23
        del squeeze_76
        del unsqueeze_538
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf247 = aten.convolution_backward(buf246, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf246
        del primals_76
        buf248 = buf247[0]
        assert_size_stride(buf248, (64, 256, 28, 28), (200704, 784, 28, 1))
        buf249 = buf247[1]
        assert_size_stride(buf249, (256, 256, 3, 3), (2304, 9, 3, 1))
        del buf247
        buf250 = reinterpret_tensor(buf234, (256, 392), (392, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_22, buf248, buf250, 100352, 128, grid=grid(100352), stream=stream0)
        buf251 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf250, buf251, 256, 392, grid=grid(256), stream=stream0)
        buf252 = reinterpret_tensor(buf250, (256, 392), (1, 256), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(relu_22, buf248, convolution_24, unsqueeze_550, buf252, 100352, 128, grid=grid(100352), stream=stream0)
        buf253 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf254 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_40.run(buf252, squeeze_73, buf253, buf254, 256, 392, grid=grid(256), stream=stream0)
        buf255 = reinterpret_tensor(buf237, (64, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41.run(relu_22, buf248, convolution_24, unsqueeze_550, buf253, squeeze_73, buf251, primals_74, buf255, 50176, 256, grid=grid(50176, 256), stream=stream0)
        del buf248
        del convolution_24
        del primals_74
        del relu_22
        del squeeze_73
        del unsqueeze_550
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf256 = aten.convolution_backward(buf255, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_73
        buf257 = buf256[0]
        assert_size_stride(buf257, (64, 512, 28, 28), (401408, 784, 28, 1))
        buf258 = buf256[1]
        assert_size_stride(buf258, (256, 512, 1, 1), (512, 1, 1, 1))
        del buf256
        buf259 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_42.run(relu_21, buf232, buf257, buf259, 512, 50176, grid=grid(512), stream=stream0)
        buf260 = reinterpret_tensor(buf252, (512, 196), (196, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_43.run(relu_21, buf232, buf257, convolution_23, unsqueeze_562, buf260, 100352, 256, grid=grid(100352), stream=stream0)
        buf261 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf263 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_44.run(buf260, squeeze_70, buf261, buf263, 512, 196, grid=grid(512), stream=stream0)
        buf262 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_45.run(relu_21, buf232, buf257, convolution_23, unsqueeze_562, buf261, squeeze_70, buf259, primals_71, buf262, 50176, 512, grid=grid(50176, 512), stream=stream0)
        del convolution_23
        del primals_71
        del squeeze_70
        del unsqueeze_562
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf264 = aten.convolution_backward(buf262, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf262
        del primals_70
        buf265 = buf264[0]
        assert_size_stride(buf265, (64, 128, 28, 28), (100352, 784, 28, 1))
        buf266 = buf264[1]
        assert_size_stride(buf266, (512, 128, 1, 1), (128, 1, 1, 1))
        del buf264
        buf267 = reinterpret_tensor(buf86, (128, 392), (392, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_20, buf265, buf267, 50176, 128, grid=grid(50176), stream=stream0)
        buf268 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf267, buf268, 128, 392, grid=grid(128), stream=stream0)
        buf269 = reinterpret_tensor(buf267, (128, 392), (1, 128), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_20, buf265, convolution_22, unsqueeze_574, buf269, 50176, 128, grid=grid(50176), stream=stream0)
        buf270 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf271 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_49.run(buf269, squeeze_67, buf270, buf271, 128, 392, grid=grid(128), stream=stream0)
        buf272 = reinterpret_tensor(buf89, (64, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(relu_20, buf265, convolution_22, unsqueeze_574, buf270, squeeze_67, buf268, primals_68, buf272, 50176, 128, grid=grid(50176, 128), stream=stream0)
        del buf265
        del convolution_22
        del primals_68
        del relu_20
        del squeeze_67
        del unsqueeze_574
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf273 = aten.convolution_backward(buf272, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_67
        buf274 = buf273[0]
        assert_size_stride(buf274, (64, 128, 28, 28), (100352, 784, 28, 1))
        buf275 = buf273[1]
        assert_size_stride(buf275, (128, 128, 3, 3), (1152, 9, 3, 1))
        del buf273
        buf276 = reinterpret_tensor(buf269, (128, 392), (392, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_19, buf274, buf276, 50176, 128, grid=grid(50176), stream=stream0)
        buf277 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf276, buf277, 128, 392, grid=grid(128), stream=stream0)
        buf278 = reinterpret_tensor(buf276, (128, 392), (1, 128), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_19, buf274, convolution_21, unsqueeze_586, buf278, 50176, 128, grid=grid(50176), stream=stream0)
        buf279 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf280 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_49.run(buf278, squeeze_64, buf279, buf280, 128, 392, grid=grid(128), stream=stream0)
        buf281 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(relu_19, buf274, convolution_21, unsqueeze_586, buf279, squeeze_64, buf277, primals_65, buf281, 50176, 128, grid=grid(50176, 128), stream=stream0)
        del buf274
        del convolution_21
        del primals_65
        del relu_19
        del squeeze_64
        del unsqueeze_586
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf282 = aten.convolution_backward(buf281, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_64
        buf283 = buf282[0]
        assert_size_stride(buf283, (64, 512, 28, 28), (401408, 784, 28, 1))
        buf284 = buf282[1]
        assert_size_stride(buf284, (128, 512, 1, 1), (512, 1, 1, 1))
        del buf282
        buf285 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_51.run(buf285, relu_18, relu_21, buf257, buf283, 32768, 784, grid=grid(32768, 784), stream=stream0)
        del buf257
        del relu_18
        del relu_21
        buf286 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf285, buf286, 512, 50176, grid=grid(512), stream=stream0)
        buf287 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_53.run(buf285, convolution_20, unsqueeze_598, buf287, 100352, 256, grid=grid(100352), stream=stream0)
        buf288 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf289 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_44.run(buf287, squeeze_61, buf288, buf289, 512, 196, grid=grid(512), stream=stream0)
        buf290 = reinterpret_tensor(buf283, (64, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_54.run(buf285, convolution_20, unsqueeze_598, buf288, squeeze_61, buf286, primals_62, buf290, 50176, 512, grid=grid(50176, 512), stream=stream0)
        del convolution_20
        del primals_62
        del squeeze_61
        del unsqueeze_598
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf291 = aten.convolution_backward(buf290, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_61
        buf292 = buf291[0]
        assert_size_stride(buf292, (64, 128, 28, 28), (100352, 784, 28, 1))
        buf293 = buf291[1]
        assert_size_stride(buf293, (512, 128, 1, 1), (128, 1, 1, 1))
        del buf291
        buf294 = reinterpret_tensor(buf278, (128, 392), (392, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_17, buf292, buf294, 50176, 128, grid=grid(50176), stream=stream0)
        buf295 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf294, buf295, 128, 392, grid=grid(128), stream=stream0)
        buf296 = reinterpret_tensor(buf294, (128, 392), (1, 128), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_17, buf292, convolution_19, unsqueeze_610, buf296, 50176, 128, grid=grid(50176), stream=stream0)
        buf297 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf298 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_49.run(buf296, squeeze_58, buf297, buf298, 128, 392, grid=grid(128), stream=stream0)
        buf299 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(relu_17, buf292, convolution_19, unsqueeze_610, buf297, squeeze_58, buf295, primals_59, buf299, 50176, 128, grid=grid(50176, 128), stream=stream0)
        del buf292
        del convolution_19
        del primals_59
        del relu_17
        del squeeze_58
        del unsqueeze_610
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf300 = aten.convolution_backward(buf299, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_58
        buf301 = buf300[0]
        assert_size_stride(buf301, (64, 128, 28, 28), (100352, 784, 28, 1))
        buf302 = buf300[1]
        assert_size_stride(buf302, (128, 128, 3, 3), (1152, 9, 3, 1))
        del buf300
        buf303 = reinterpret_tensor(buf296, (128, 392), (392, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_16, buf301, buf303, 50176, 128, grid=grid(50176), stream=stream0)
        buf304 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf303, buf304, 128, 392, grid=grid(128), stream=stream0)
        buf305 = reinterpret_tensor(buf303, (128, 392), (1, 128), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_16, buf301, convolution_18, unsqueeze_622, buf305, 50176, 128, grid=grid(50176), stream=stream0)
        buf306 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf307 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_49.run(buf305, squeeze_55, buf306, buf307, 128, 392, grid=grid(128), stream=stream0)
        buf308 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(relu_16, buf301, convolution_18, unsqueeze_622, buf306, squeeze_55, buf304, primals_56, buf308, 50176, 128, grid=grid(50176, 128), stream=stream0)
        del buf301
        del convolution_18
        del primals_56
        del relu_16
        del squeeze_55
        del unsqueeze_622
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf309 = aten.convolution_backward(buf308, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_55
        buf310 = buf309[0]
        assert_size_stride(buf310, (64, 512, 28, 28), (401408, 784, 28, 1))
        buf311 = buf309[1]
        assert_size_stride(buf311, (128, 512, 1, 1), (512, 1, 1, 1))
        del buf309
        buf312 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_42.run(relu_15, buf285, buf310, buf312, 512, 50176, grid=grid(512), stream=stream0)
        buf313 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_43.run(relu_15, buf285, buf310, convolution_17, unsqueeze_634, buf313, 100352, 256, grid=grid(100352), stream=stream0)
        buf314 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf316 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_44.run(buf313, squeeze_52, buf314, buf316, 512, 196, grid=grid(512), stream=stream0)
        buf315 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_45.run(relu_15, buf285, buf310, convolution_17, unsqueeze_634, buf314, squeeze_52, buf312, primals_53, buf315, 50176, 512, grid=grid(50176, 512), stream=stream0)
        del convolution_17
        del primals_53
        del squeeze_52
        del unsqueeze_634
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf317 = aten.convolution_backward(buf315, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf315
        del primals_52
        buf318 = buf317[0]
        assert_size_stride(buf318, (64, 128, 28, 28), (100352, 784, 28, 1))
        buf319 = buf317[1]
        assert_size_stride(buf319, (512, 128, 1, 1), (128, 1, 1, 1))
        del buf317
        buf320 = reinterpret_tensor(buf305, (128, 392), (392, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_14, buf318, buf320, 50176, 128, grid=grid(50176), stream=stream0)
        buf321 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf320, buf321, 128, 392, grid=grid(128), stream=stream0)
        buf322 = reinterpret_tensor(buf320, (128, 392), (1, 128), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_14, buf318, convolution_16, unsqueeze_646, buf322, 50176, 128, grid=grid(50176), stream=stream0)
        buf323 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf324 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_49.run(buf322, squeeze_49, buf323, buf324, 128, 392, grid=grid(128), stream=stream0)
        buf325 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(relu_14, buf318, convolution_16, unsqueeze_646, buf323, squeeze_49, buf321, primals_50, buf325, 50176, 128, grid=grid(50176, 128), stream=stream0)
        del buf318
        del convolution_16
        del primals_50
        del relu_14
        del squeeze_49
        del unsqueeze_646
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf326 = aten.convolution_backward(buf325, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_49
        buf327 = buf326[0]
        assert_size_stride(buf327, (64, 128, 28, 28), (100352, 784, 28, 1))
        buf328 = buf326[1]
        assert_size_stride(buf328, (128, 128, 3, 3), (1152, 9, 3, 1))
        del buf326
        buf329 = reinterpret_tensor(buf322, (128, 392), (392, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_13, buf327, buf329, 50176, 128, grid=grid(50176), stream=stream0)
        buf330 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf329, buf330, 128, 392, grid=grid(128), stream=stream0)
        buf331 = reinterpret_tensor(buf329, (128, 392), (1, 128), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_13, buf327, convolution_15, unsqueeze_658, buf331, 50176, 128, grid=grid(50176), stream=stream0)
        buf332 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf333 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_49.run(buf331, squeeze_46, buf332, buf333, 128, 392, grid=grid(128), stream=stream0)
        buf334 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(relu_13, buf327, convolution_15, unsqueeze_658, buf332, squeeze_46, buf330, primals_47, buf334, 50176, 128, grid=grid(50176, 128), stream=stream0)
        del buf327
        del convolution_15
        del primals_47
        del relu_13
        del squeeze_46
        del unsqueeze_658
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf335 = aten.convolution_backward(buf334, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_46
        buf336 = buf335[0]
        assert_size_stride(buf336, (64, 512, 28, 28), (401408, 784, 28, 1))
        buf337 = buf335[1]
        assert_size_stride(buf337, (128, 512, 1, 1), (512, 1, 1, 1))
        del buf335
        buf338 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_51.run(buf338, relu_12, relu_15, buf310, buf336, 32768, 784, grid=grid(32768, 784), stream=stream0)
        del relu_12
        del relu_15
        buf339 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf338, buf339, 512, 50176, grid=grid(512), stream=stream0)
        buf340 = buf313; del buf313  # reuse
        buf347 = reinterpret_tensor(buf227, (512, 196), (196, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_55.run(buf338, convolution_14, unsqueeze_670, convolution_13, unsqueeze_682, buf340, buf347, 100352, 256, grid=grid(100352), stream=stream0)
        buf341 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf342 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_44.run(buf340, squeeze_43, buf341, buf342, 512, 196, grid=grid(512), stream=stream0)
        del buf340
        buf348 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf349 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_44.run(buf347, squeeze_40, buf348, buf349, 512, 196, grid=grid(512), stream=stream0)
        buf343 = reinterpret_tensor(buf336, (64, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf336  # reuse
        buf350 = reinterpret_tensor(buf310, (64, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_56.run(buf338, convolution_14, unsqueeze_670, buf341, squeeze_43, buf339, primals_44, convolution_13, unsqueeze_682, buf348, squeeze_40, primals_41, buf343, buf350, 50176, 512, grid=grid(50176, 512), stream=stream0)
        del buf338
        del buf341
        del buf348
        del convolution_13
        del convolution_14
        del primals_41
        del primals_44
        del squeeze_40
        del squeeze_43
        del unsqueeze_670
        del unsqueeze_682
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf344 = aten.convolution_backward(buf343, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf343
        del primals_43
        buf345 = buf344[0]
        assert_size_stride(buf345, (64, 256, 56, 56), (802816, 3136, 56, 1))
        buf346 = buf344[1]
        assert_size_stride(buf346, (512, 256, 1, 1), (256, 1, 1, 1))
        del buf344
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf351 = aten.convolution_backward(buf350, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_40
        buf352 = buf351[0]
        assert_size_stride(buf352, (64, 128, 28, 28), (100352, 784, 28, 1))
        buf353 = buf351[1]
        assert_size_stride(buf353, (512, 128, 1, 1), (128, 1, 1, 1))
        del buf351
        buf354 = reinterpret_tensor(buf331, (128, 392), (392, 1), 0); del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(relu_11, buf352, buf354, 50176, 128, grid=grid(50176), stream=stream0)
        buf355 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_47.run(buf354, buf355, 128, 392, grid=grid(128), stream=stream0)
        buf356 = reinterpret_tensor(buf354, (128, 392), (1, 128), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(relu_11, buf352, convolution_12, unsqueeze_694, buf356, 50176, 128, grid=grid(50176), stream=stream0)
        buf357 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf358 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_49.run(buf356, squeeze_37, buf357, buf358, 128, 392, grid=grid(128), stream=stream0)
        buf359 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(relu_11, buf352, convolution_12, unsqueeze_694, buf357, squeeze_37, buf355, primals_38, buf359, 50176, 128, grid=grid(50176, 128), stream=stream0)
        del buf352
        del convolution_12
        del primals_38
        del relu_11
        del squeeze_37
        del unsqueeze_694
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf360 = aten.convolution_backward(buf359, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf359
        del primals_37
        buf361 = buf360[0]
        assert_size_stride(buf361, (64, 128, 56, 56), (401408, 3136, 56, 1))
        buf362 = buf360[1]
        assert_size_stride(buf362, (128, 128, 3, 3), (1152, 9, 3, 1))
        del buf360
        buf363 = reinterpret_tensor(buf347, (128, 784), (784, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_10, buf361, buf363, 100352, 256, grid=grid(100352), stream=stream0)
        buf364 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_58.run(buf363, buf364, 128, 784, grid=grid(128), stream=stream0)
        buf365 = reinterpret_tensor(buf363, (128, 784), (1, 128), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_59.run(relu_10, buf361, convolution_11, unsqueeze_706, buf365, 100352, 256, grid=grid(100352), stream=stream0)
        buf366 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf367 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_60.run(buf365, squeeze_34, buf366, buf367, 128, 784, grid=grid(128), stream=stream0)
        del buf365
        buf368 = reinterpret_tensor(buf350, (64, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_61.run(relu_10, buf361, convolution_11, unsqueeze_706, buf366, squeeze_34, buf364, primals_35, buf368, 200704, 128, grid=grid(200704, 128), stream=stream0)
        del buf361
        del buf366
        del convolution_11
        del primals_35
        del relu_10
        del squeeze_34
        del unsqueeze_706
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf369 = aten.convolution_backward(buf368, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf368
        del primals_34
        buf370 = buf369[0]
        assert_size_stride(buf370, (64, 256, 56, 56), (802816, 3136, 56, 1))
        buf371 = buf369[1]
        assert_size_stride(buf371, (128, 256, 1, 1), (256, 1, 1, 1))
        del buf369
        buf372 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_62.run(relu_9, buf345, buf370, buf372, 256, 200704, grid=grid(256), stream=stream0)
        buf373 = empty_strided((256, 328), (328, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_63.run(relu_9, buf345, buf370, convolution_10, unsqueeze_718, buf373, 83968, 612, grid=grid(83968), stream=stream0)
        buf374 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf376 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_64.run(buf373, squeeze_31, buf374, buf376, 256, 328, grid=grid(256), stream=stream0)
        buf375 = empty_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_65.run(relu_9, buf345, buf370, convolution_10, unsqueeze_718, buf374, squeeze_31, buf372, primals_32, buf375, 200704, 256, grid=grid(200704, 256), stream=stream0)
        del convolution_10
        del primals_32
        del squeeze_31
        del unsqueeze_718
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf377 = aten.convolution_backward(buf375, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf375
        del primals_31
        buf378 = buf377[0]
        assert_size_stride(buf378, (64, 64, 56, 56), (200704, 3136, 56, 1))
        buf379 = buf377[1]
        assert_size_stride(buf379, (256, 64, 1, 1), (64, 1, 1, 1))
        del buf377
        buf380 = reinterpret_tensor(buf356, (64, 784), (784, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_8, buf378, buf380, 50176, 256, grid=grid(50176), stream=stream0)
        buf381 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf380, buf381, 64, 784, grid=grid(64), stream=stream0)
        buf382 = reinterpret_tensor(buf380, (64, 784), (1, 64), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_8, buf378, convolution_9, unsqueeze_730, buf382, 50176, 256, grid=grid(50176), stream=stream0)
        buf383 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf384 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(buf382, squeeze_28, buf383, buf384, 64, 784, grid=grid(64), stream=stream0)
        buf385 = reinterpret_tensor(buf255, (64, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(relu_8, buf378, convolution_9, unsqueeze_730, buf383, squeeze_28, buf381, primals_29, buf385, 200704, 64, grid=grid(200704, 64), stream=stream0)
        del buf378
        del convolution_9
        del primals_29
        del relu_8
        del squeeze_28
        del unsqueeze_730
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf386 = aten.convolution_backward(buf385, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_28
        buf387 = buf386[0]
        assert_size_stride(buf387, (64, 64, 56, 56), (200704, 3136, 56, 1))
        buf388 = buf386[1]
        assert_size_stride(buf388, (64, 64, 3, 3), (576, 9, 3, 1))
        del buf386
        buf389 = reinterpret_tensor(buf382, (64, 784), (784, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_7, buf387, buf389, 50176, 256, grid=grid(50176), stream=stream0)
        buf390 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf389, buf390, 64, 784, grid=grid(64), stream=stream0)
        buf391 = reinterpret_tensor(buf389, (64, 784), (1, 64), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_7, buf387, convolution_8, unsqueeze_742, buf391, 50176, 256, grid=grid(50176), stream=stream0)
        buf392 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf393 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(buf391, squeeze_25, buf392, buf393, 64, 784, grid=grid(64), stream=stream0)
        buf394 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(relu_7, buf387, convolution_8, unsqueeze_742, buf392, squeeze_25, buf390, primals_26, buf394, 200704, 64, grid=grid(200704, 64), stream=stream0)
        del buf387
        del convolution_8
        del primals_26
        del relu_7
        del squeeze_25
        del unsqueeze_742
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf395 = aten.convolution_backward(buf394, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_25
        buf396 = buf395[0]
        assert_size_stride(buf396, (64, 256, 56, 56), (802816, 3136, 56, 1))
        buf397 = buf395[1]
        assert_size_stride(buf397, (64, 256, 1, 1), (256, 1, 1, 1))
        del buf395
        buf398 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_71.run(buf398, relu_6, relu_9, buf370, buf396, 16384, 3136, grid=grid(16384, 3136), stream=stream0)
        del relu_6
        del relu_9
        buf399 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_72.run(buf398, buf399, 256, 200704, grid=grid(256), stream=stream0)
        buf400 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_73.run(buf398, convolution_7, unsqueeze_754, buf400, 83968, 612, grid=grid(83968), stream=stream0)
        buf401 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf402 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_64.run(buf400, squeeze_22, buf401, buf402, 256, 328, grid=grid(256), stream=stream0)
        buf403 = reinterpret_tensor(buf396, (64, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_74.run(buf398, convolution_7, unsqueeze_754, buf401, squeeze_22, buf399, primals_23, buf403, 200704, 256, grid=grid(200704, 256), stream=stream0)
        del convolution_7
        del primals_23
        del squeeze_22
        del unsqueeze_754
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf404 = aten.convolution_backward(buf403, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_22
        buf405 = buf404[0]
        assert_size_stride(buf405, (64, 64, 56, 56), (200704, 3136, 56, 1))
        buf406 = buf404[1]
        assert_size_stride(buf406, (256, 64, 1, 1), (64, 1, 1, 1))
        del buf404
        buf407 = reinterpret_tensor(buf391, (64, 784), (784, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_5, buf405, buf407, 50176, 256, grid=grid(50176), stream=stream0)
        buf408 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf407, buf408, 64, 784, grid=grid(64), stream=stream0)
        buf409 = reinterpret_tensor(buf407, (64, 784), (1, 64), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_5, buf405, convolution_6, unsqueeze_766, buf409, 50176, 256, grid=grid(50176), stream=stream0)
        buf410 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf411 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(buf409, squeeze_19, buf410, buf411, 64, 784, grid=grid(64), stream=stream0)
        buf412 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(relu_5, buf405, convolution_6, unsqueeze_766, buf410, squeeze_19, buf408, primals_20, buf412, 200704, 64, grid=grid(200704, 64), stream=stream0)
        del buf405
        del convolution_6
        del primals_20
        del relu_5
        del squeeze_19
        del unsqueeze_766
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf413 = aten.convolution_backward(buf412, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_19
        buf414 = buf413[0]
        assert_size_stride(buf414, (64, 64, 56, 56), (200704, 3136, 56, 1))
        buf415 = buf413[1]
        assert_size_stride(buf415, (64, 64, 3, 3), (576, 9, 3, 1))
        del buf413
        buf416 = reinterpret_tensor(buf409, (64, 784), (784, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_4, buf414, buf416, 50176, 256, grid=grid(50176), stream=stream0)
        buf417 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf416, buf417, 64, 784, grid=grid(64), stream=stream0)
        buf418 = reinterpret_tensor(buf416, (64, 784), (1, 64), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_4, buf414, convolution_5, unsqueeze_778, buf418, 50176, 256, grid=grid(50176), stream=stream0)
        buf419 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf420 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(buf418, squeeze_16, buf419, buf420, 64, 784, grid=grid(64), stream=stream0)
        buf421 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(relu_4, buf414, convolution_5, unsqueeze_778, buf419, squeeze_16, buf417, primals_17, buf421, 200704, 64, grid=grid(200704, 64), stream=stream0)
        del buf414
        del convolution_5
        del primals_17
        del relu_4
        del squeeze_16
        del unsqueeze_778
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf422 = aten.convolution_backward(buf421, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_16
        buf423 = buf422[0]
        assert_size_stride(buf423, (64, 256, 56, 56), (802816, 3136, 56, 1))
        buf424 = buf422[1]
        assert_size_stride(buf424, (64, 256, 1, 1), (256, 1, 1, 1))
        del buf422
        buf425 = buf401; del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_62.run(relu_3, buf398, buf423, buf425, 256, 200704, grid=grid(256), stream=stream0)
        buf426 = buf400; del buf400  # reuse
        buf433 = empty_strided((256, 328), (328, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_75.run(relu_3, buf398, buf423, convolution_4, unsqueeze_790, convolution_3, unsqueeze_802, buf426, buf433, 83968, 612, grid=grid(83968), stream=stream0)
        buf427 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf429 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_64.run(buf426, squeeze_13, buf427, buf429, 256, 328, grid=grid(256), stream=stream0)
        del buf426
        buf434 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf436 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_64.run(buf433, squeeze_10, buf434, buf436, 256, 328, grid=grid(256), stream=stream0)
        del buf433
        buf428 = buf403; del buf403  # reuse
        buf435 = reinterpret_tensor(buf370, (64, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_76.run(relu_3, buf398, buf423, convolution_4, unsqueeze_790, buf427, squeeze_13, buf425, primals_14, convolution_3, unsqueeze_802, buf434, squeeze_10, primals_11, buf428, buf435, 200704, 256, grid=grid(200704, 256), stream=stream0)
        del buf398
        del buf423
        del buf427
        del buf434
        del convolution_3
        del convolution_4
        del primals_11
        del primals_14
        del relu_3
        del squeeze_10
        del squeeze_13
        del unsqueeze_790
        del unsqueeze_802
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf430 = aten.convolution_backward(buf428, getitem_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_13
        buf431 = buf430[0]
        assert_size_stride(buf431, (64, 64, 56, 56), (200704, 3136, 56, 1))
        buf432 = buf430[1]
        assert_size_stride(buf432, (256, 64, 1, 1), (64, 1, 1, 1))
        del buf430
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf437 = aten.convolution_backward(buf435, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_10
        buf438 = buf437[0]
        assert_size_stride(buf438, (64, 64, 56, 56), (200704, 3136, 56, 1))
        buf439 = buf437[1]
        assert_size_stride(buf439, (256, 64, 1, 1), (64, 1, 1, 1))
        del buf437
        buf440 = reinterpret_tensor(buf418, (64, 784), (784, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_2, buf438, buf440, 50176, 256, grid=grid(50176), stream=stream0)
        buf441 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf440, buf441, 64, 784, grid=grid(64), stream=stream0)
        buf442 = reinterpret_tensor(buf440, (64, 784), (1, 64), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_2, buf438, convolution_2, unsqueeze_814, buf442, 50176, 256, grid=grid(50176), stream=stream0)
        buf443 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf444 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(buf442, squeeze_7, buf443, buf444, 64, 784, grid=grid(64), stream=stream0)
        buf445 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(relu_2, buf438, convolution_2, unsqueeze_814, buf443, squeeze_7, buf441, primals_8, buf445, 200704, 64, grid=grid(200704, 64), stream=stream0)
        del buf438
        del convolution_2
        del primals_8
        del relu_2
        del squeeze_7
        del unsqueeze_814
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf446 = aten.convolution_backward(buf445, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_7
        buf447 = buf446[0]
        assert_size_stride(buf447, (64, 64, 56, 56), (200704, 3136, 56, 1))
        buf448 = buf446[1]
        assert_size_stride(buf448, (64, 64, 3, 3), (576, 9, 3, 1))
        del buf446
        buf449 = reinterpret_tensor(buf442, (64, 784), (784, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_66.run(relu_1, buf447, buf449, 50176, 256, grid=grid(50176), stream=stream0)
        buf450 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_67.run(buf449, buf450, 64, 784, grid=grid(64), stream=stream0)
        buf451 = reinterpret_tensor(buf449, (64, 784), (1, 64), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_68.run(relu_1, buf447, convolution_1, unsqueeze_826, buf451, 50176, 256, grid=grid(50176), stream=stream0)
        buf452 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf453 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_69.run(buf451, squeeze_4, buf452, buf453, 64, 784, grid=grid(64), stream=stream0)
        del buf451
        buf454 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_70.run(relu_1, buf447, convolution_1, unsqueeze_826, buf452, squeeze_4, buf450, primals_5, buf454, 200704, 64, grid=grid(200704, 64), stream=stream0)
        del buf447
        del convolution_1
        del primals_5
        del relu_1
        del squeeze_4
        del unsqueeze_826
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf455 = aten.convolution_backward(buf454, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf454
        del getitem_2
        del primals_4
        buf456 = buf455[0]
        assert_size_stride(buf456, (64, 64, 56, 56), (200704, 3136, 56, 1))
        buf457 = buf455[1]
        assert_size_stride(buf457, (64, 64, 1, 1), (64, 1, 1, 1))
        del buf455
        buf458 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_77.run(buf458, buf456, 12845056, grid=grid(12845056), stream=stream0)
        del buf456
        buf459 = reinterpret_tensor(buf435, (64, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
        triton_poi_fused_add_max_pool2d_with_indices_backward_78.run(getitem_3, buf458, buf459, 4096, 12544, grid=grid(4096, 12544), stream=stream0)
        del buf458
        del getitem_3
        buf460 = empty_strided((64, 656), (656, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_79.run(relu, buf459, buf460, 41984, 1224, grid=grid(41984), stream=stream0)
        buf461 = buf452; del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_80.run(buf460, buf461, 64, 656, grid=grid(64), stream=stream0)
        buf462 = reinterpret_tensor(buf460, (64, 656), (1, 64), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_81.run(relu, buf459, convolution, unsqueeze_838, buf462, 41984, 1224, grid=grid(41984), stream=stream0)
        buf463 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf464 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_82.run(buf462, squeeze_1, buf463, buf464, 64, 656, grid=grid(64), stream=stream0)
        del buf462
        buf465 = reinterpret_tensor(buf428, (64, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_83.run(relu, buf459, convolution, unsqueeze_838, buf463, squeeze_1, buf461, primals_2, buf465, 802816, 64, grid=grid(802816, 64), stream=stream0)
        del buf459
        del buf463
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_838
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf466 = aten.convolution_backward(buf465, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf465
        del primals_1
        del primals_321
        buf467 = buf466[1]
        assert_size_stride(buf467, (64, 3, 7, 7), (147, 49, 7, 1))
        del buf466
        return (buf467, buf464, buf461, buf457, buf453, buf450, buf448, buf444, buf441, buf439, buf436, buf425, buf432, buf429, buf425, buf424, buf420, buf417, buf415, buf411, buf408, buf406, buf402, buf399, buf397, buf393, buf390, buf388, buf384, buf381, buf379, buf376, buf372, buf371, buf367, buf364, buf362, buf358, buf355, buf353, buf349, buf339, buf346, buf342, buf339, buf337, buf333, buf330, buf328, buf324, buf321, buf319, buf316, buf312, buf311, buf307, buf304, buf302, buf298, buf295, buf293, buf289, buf286, buf284, buf280, buf277, buf275, buf271, buf268, buf266, buf263, buf259, buf258, buf254, buf251, buf249, buf245, buf242, buf240, buf236, buf226, buf233, buf229, buf226, buf224, buf220, buf217, buf215, buf211, buf208, buf206, buf203, buf199, buf198, buf194, buf191, buf189, buf185, buf182, buf180, buf176, buf173, buf171, buf167, buf164, buf162, buf158, buf155, buf153, buf150, buf146, buf145, buf141, buf138, buf136, buf132, buf129, buf127, buf123, buf120, buf118, buf114, buf111, buf109, buf105, buf102, buf100, buf97, buf93, buf92, buf88, buf85, buf83, buf79, buf76, buf74, buf70, buf60, buf67, buf63, buf60, buf57, buf53, buf50, buf48, buf44, buf41, buf39, buf35, buf31, buf29, buf25, buf22, buf20, buf16, buf13, buf11, buf7, buf4, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((64, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((64, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.int64)
    convolution_1 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((64, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((64, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((64, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((64, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((64, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((64, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((64, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bool)
    unsqueeze_214 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_706 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_754 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_802 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_27 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_33 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_36 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_39 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_42 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_45 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_48 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_54 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_57 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_60 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_66 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_69 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_72 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_75 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_78 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_79 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_81 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_82 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_84 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_87 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_90 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_91 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_93 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_96 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_99 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_102 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_105 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_108 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_111 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_114 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_117 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_118 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_120 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_123 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_126 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_132 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_136 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_138 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_139 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_141 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_144 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_147 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_150 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_152 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_153 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_156 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_157 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_159 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    tangents_160 = rand_strided((64, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, relu_27, convolution_31, squeeze_94, relu_28, convolution_32, squeeze_97, relu_29, convolution_33, squeeze_100, relu_30, convolution_34, squeeze_103, relu_31, convolution_35, squeeze_106, relu_32, convolution_36, squeeze_109, relu_33, convolution_37, squeeze_112, relu_34, convolution_38, squeeze_115, relu_35, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, tangents_51, tangents_52, tangents_53, tangents_54, tangents_55, tangents_56, tangents_57, tangents_58, tangents_59, tangents_60, tangents_61, tangents_62, tangents_63, tangents_64, tangents_65, tangents_66, tangents_67, tangents_68, tangents_69, tangents_70, tangents_71, tangents_72, tangents_73, tangents_74, tangents_75, tangents_76, tangents_77, tangents_78, tangents_79, tangents_80, tangents_81, tangents_82, tangents_83, tangents_84, tangents_85, tangents_86, tangents_87, tangents_88, tangents_89, tangents_90, tangents_91, tangents_92, tangents_93, tangents_94, tangents_95, tangents_96, tangents_97, tangents_98, tangents_99, tangents_100, tangents_101, tangents_102, tangents_103, tangents_104, tangents_105, tangents_106, tangents_107, tangents_108, tangents_109, tangents_110, tangents_111, tangents_112, tangents_113, tangents_114, tangents_115, tangents_116, tangents_117, tangents_118, tangents_119, tangents_120, tangents_121, tangents_122, tangents_123, tangents_124, tangents_125, tangents_126, tangents_127, tangents_128, tangents_129, tangents_130, tangents_131, tangents_132, tangents_133, tangents_134, tangents_135, tangents_136, tangents_137, tangents_138, tangents_139, tangents_140, tangents_141, tangents_142, tangents_143, tangents_144, tangents_145, tangents_146, tangents_147, tangents_148, tangents_149, tangents_150, tangents_151, tangents_152, tangents_153, tangents_154, tangents_155, tangents_156, tangents_157, tangents_158, tangents_159, tangents_160]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)