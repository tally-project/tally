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


# kernel path: /tmp/torchinductor_zhaowe58/u5/cu5754sbajfywlslo5v6av5dhdqfktz477rh625ahxgzcjglygnz.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (147*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_zhaowe58/hr/chruvrdnmjk5eaqj4hbljq4fwmh3by274qh2k5fnnswcqphfw47b.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/4o/c4orjx55yfnxx6s2lzod5pzfz7p3xoobhvctwkxvvzsekul55bwp.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/jk/cjkojff55ev3egumdfixrouaycezhb6pshdw7qprsiu2aniz4fxn.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/2g/c2glw23r4g5vvq6i7wb6w5aidy2pgay5tp5ksssgkqkkezyon7xj.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (4608*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/oz/coz7gig6z5l7e2nm5fho5taymuyonscmmtm6rttajkp2n7o3zcxw.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/j7/cj7ndcdrgt3uu25372vnsiqo3j43gx5zbo75cdz4ekoukpd3zmrb.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_tem_fused_convolution_6 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_6'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 7
    KERNEL_W : tl.constexpr = 7
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 3
    PADDING_W : tl.constexpr = 3
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 3
    IN_H = 224
    IN_W = 224
    OUT_C = 64
    OUT_H = 112
    OUT_W = 112

    # Strides:
    stride_xn = 150528
    stride_xc = 1
    stride_xh = 672
    stride_xw = 3
    stride_wc_out = 147
    stride_wc_in = 1
    stride_wh = 21
    stride_ww = 3

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (112*idx_h) + (12544*idx_c) + (802816*idx_n)
    x5 = xindex % 12544
    tl.store(out_ptr1 + (idx_c + (64*x5) + (802816*idx_n)), acc, mask)
''')
import torch._inductor.kernel.conv
meta0 = {'KERNEL_H': 7, 'KERNEL_W': 7, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 3, 'PADDING_W': 3, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': True, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_zhaowe58/cs/ccs72yiuoqkzy4fo2fh2gene746qz6frtvgldxdrsga3cwkqxrah.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 41984
    rnumel = 1224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (1224*x1)
        tmp1 = tl.full([1, 1], 802816, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (1224*x1)) % 802816))), rmask & tmp2 & xmask, other=0)
        tmp4 = tl.where(tmp2, tmp3, 0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp2, tmp5, 0)
        tmp7 = 1.0
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp10 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_combine(
            tmp12_mean, tmp12_m2, tmp12_weight,
            tmp9, tmp10, tmp11
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
    tl.store(out_ptr2 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/xg/cxg27dfk24qxqbw6vjkyknw2dlr3djvduxohhdqytteainvrwhby.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_8', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 110
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6)
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (110*x0)
        tmp1 = tl.full([1, 1], 656, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*r2) + (7040*x0)), rmask & tmp2 & xmask, other=0)
        tmp4 = tl.where(tmp2, tmp3, 0)
        tmp5 = tl.load(in_ptr1 + (x1 + (64*r2) + (7040*x0)), rmask & tmp2 & xmask, other=0)
        tmp6 = tl.where(tmp2, tmp5, 0)
        tmp7 = tl.load(in_ptr2 + (x1 + (64*r2) + (7040*x0)), rmask & tmp2 & xmask, other=0)
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp10 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_combine(
            tmp12_mean, tmp12_m2, tmp12_weight,
            tmp9, tmp10, tmp11
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp12, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp13, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/qh/cqhs3s2yrezl6tgru7u3rpkr7shqnigl5x2bod23fl6dczoobp2j.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_9', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 802816.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 1.0000012456169853
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ru/crupbm4hyoe4x7h4tvlbnaayv3tqpaujpuzkojmqedw3a5rrlxdf.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 802816.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/al/cal2c3n64qlbgslyuvia4zdyaiylhznwk4cqwwooiakaidvhfdak.py
# Source Nodes: [identity], Original ATen: [aten.max_pool2d_with_indices]
# identity => getitem_2, getitem_3
triton_poi_fused_max_pool2d_with_indices_11 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_11', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3584) % 56
    x1 = (xindex // 64) % 56
    x0 = xindex % 64
    x5 = (xindex // 3584)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + (128*x1) + (14336*x5)), tmp10, other=0)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = 2*x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-7168) + x0 + (128*x1) + (14336*x5)), tmp17, other=0)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = triton_helpers.maximum(tmp19, tmp12)
    tmp21 = 1 + (2*x1)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-7104) + x0 + (128*x1) + (14336*x5)), tmp25, other=0)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = triton_helpers.maximum(tmp27, tmp20)
    tmp29 = 2*x2
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (14336*x5)), tmp33, other=0)
    tmp35 = tl.where(tmp33, tmp34, float("-inf"))
    tmp36 = triton_helpers.maximum(tmp35, tmp28)
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (x0 + (128*x1) + (14336*x5)), tmp37, other=0)
    tmp39 = tl.where(tmp37, tmp38, float("-inf"))
    tmp40 = triton_helpers.maximum(tmp39, tmp36)
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (14336*x5)), tmp41, other=0)
    tmp43 = tl.where(tmp41, tmp42, float("-inf"))
    tmp44 = triton_helpers.maximum(tmp43, tmp40)
    tmp45 = 1 + (2*x2)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (7104 + x0 + (128*x1) + (14336*x5)), tmp49, other=0)
    tmp51 = tl.where(tmp49, tmp50, float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp44)
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (7168 + x0 + (128*x1) + (14336*x5)), tmp53, other=0)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = triton_helpers.maximum(tmp55, tmp52)
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (7232 + x0 + (128*x1) + (14336*x5)), tmp57, other=0)
    tmp59 = tl.where(tmp57, tmp58, float("-inf"))
    tmp60 = triton_helpers.maximum(tmp59, tmp56)
    tmp61 = tmp19 > tmp12
    tmp62 = (-112) + (2*x1) + (224*x2)
    tmp63 = (-113) + (2*x1) + (224*x2)
    tmp64 = tl.where(tmp61, tmp62, tmp63)
    tmp65 = tmp27 > tmp20
    tmp66 = (-111) + (2*x1) + (224*x2)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp35 > tmp28
    tmp69 = (-1) + (2*x1) + (224*x2)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp39 > tmp36
    tmp72 = (2*x1) + (224*x2)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp43 > tmp40
    tmp75 = 1 + (2*x1) + (224*x2)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp51 > tmp44
    tmp78 = 111 + (2*x1) + (224*x2)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp55 > tmp52
    tmp81 = 112 + (2*x1) + (224*x2)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp59 > tmp56
    tmp84 = 113 + (2*x1) + (224*x2)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tl.store(out_ptr0 + (x6), tmp60, None)
    tl.store(out_ptr1 + (x6), tmp85, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/si/csianaj4yaicnb4rz2v7ljv6seoltzlwzbmnomct37f4bl6lflhf.py
# Source Nodes: [out], Original ATen: [aten.convolution]
# out => convolution_1
triton_tem_fused_convolution_12 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=3, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_12'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 200704
    N = 64
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 64
    stride_ak = 1
    stride_bk = 1
    stride_bn = 64

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
    xindex = idx_n + (64*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')
import torch._inductor.kernel.mm_common
meta1 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}


# kernel path: /tmp/torchinductor_zhaowe58/zx/czxdfdagucpdyk3aacwv72asn6efoih7iqnabykbulipel723bni.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (16384*x1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/7e/c7eynni2idmszlisr2kj4bx6mfqm775j4465j5bdnfsfywon7avb.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 448
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, other=0)
        tmp2 = tl.load(in_ptr2 + (x1 + (64*r2) + (7168*x0)), rmask & xmask, other=0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (64*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (64*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/iz/cizd5dhfk3j25zeo26q2ie3hsfftclxoj2ryq2k7lahqdlhwtz67.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_15', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 200704.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 1.0000049824865598
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/2z/c2zici6pibw2ebiyfqbnbp2xjliavrhmiqihbflgzb5ri263vgrv.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_1 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# out_2 => relu_1
triton_poi_fused__native_batch_norm_legit_functional_relu_16 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_16', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 200704.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/rn/crnbnibily67e7eabs3lbdizg5cd6hgd6or3lrdvxq44hjwbjpm4.py
# Source Nodes: [out_3], Original ATen: [aten.convolution]
# out_3 => convolution_2
triton_tem_fused_convolution_17 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_17'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 64
    IN_H = 56
    IN_W = 56
    OUT_C = 64
    OUT_H = 56
    OUT_W = 56

    # Strides:
    stride_xn = 200704
    stride_xc = 1
    stride_xh = 3584
    stride_xw = 64
    stride_wc_out = 576
    stride_wc_in = 1
    stride_wh = 192
    stride_ww = 64

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (56*idx_h) + (3136*idx_c) + (200704*idx_n)
    x5 = xindex % 3136
    tl.store(out_ptr1 + (idx_c + (64*x5) + (200704*idx_n)), acc, mask)
''')
meta2 = {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': True, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_zhaowe58/24/c24wsdvcvl5vh7bdgzokodxbpkxk5n4rdhgibpmcvjj5ek7nrtim.py
# Source Nodes: [out_6], Original ATen: [aten.convolution]
# out_6 => convolution_3
triton_tem_fused_convolution_18 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_18'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 200704
    N = 256
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 64
    stride_ak = 1
    stride_bk = 1
    stride_bn = 64

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
    xindex = idx_n + (256*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')
meta3 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}


# kernel path: /tmp/torchinductor_zhaowe58/zu/czu3jznciw2acqw4uwxqm2snrasrcaeqqzk7e37apdplk72upajw.py
# Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
# out_7 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 83968
    rnumel = 612
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (612*x1)
        tmp1 = tl.full([1, 1], 200704, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (612*x1)) % 200704))), rmask & tmp2, other=0)
        tmp4 = tl.where(tmp2, tmp3, 0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp2, tmp5, 0)
        tmp7 = 1.0
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp10 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_combine(
            tmp12_mean, tmp12_m2, tmp12_weight,
            tmp9, tmp10, tmp11
        )
        tmp12_mean = tl.where(rmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask, tmp12_weight_next, tmp12_weight)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp13, None)
    tl.store(out_ptr2 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/a4/ca46d3dksh7uwbwpvieodh7dsfgdbpcv2xgmguvw22tzhk5l7sxf.py
# Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
# out_7 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 110
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3)
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (110*x0)
        tmp1 = tl.full([1, 1], 328, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (256*r2) + (28160*x0)), rmask & tmp2 & xmask, other=0)
        tmp4 = tl.where(tmp2, tmp3, 0)
        tmp5 = tl.load(in_ptr1 + (x1 + (256*r2) + (28160*x0)), rmask & tmp2 & xmask, other=0)
        tmp6 = tl.where(tmp2, tmp5, 0)
        tmp7 = tl.load(in_ptr2 + (x1 + (256*r2) + (28160*x0)), rmask & tmp2 & xmask, other=0)
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp10 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_combine(
            tmp12_mean, tmp12_m2, tmp12_weight,
            tmp9, tmp10, tmp11
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (256*x0)), tmp12, xmask)
    tl.store(out_ptr1 + (x1 + (256*x0)), tmp13, xmask)
    tl.store(out_ptr2 + (x1 + (256*x0)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/n4/cn4mvl3mpavaniekihft5q6pzpqmhigrcrvnvfyyckl6b4b47spp.py
# Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
# out_7 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_21', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 200704.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 1.0000049824865598
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/pe/cpesuzufut2mrei64lx5kh3pw5rowje6jfgls6ec7ivvpad7uuwe.py
# Source Nodes: [identity_1, identity_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_1 => add_21, add_24, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# identity_2 => relu_3
# out_7 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# out_8 => add_25
triton_poi_fused__native_batch_norm_legit_functional_add_relu_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_22', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 200704.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ux/cuxizxpbbiuc2dn6glwqvvlkowgjqukskv7r3vyt5akxp32pwwec.py
# Source Nodes: [out_10], Original ATen: [aten.convolution]
# out_10 => convolution_5
triton_tem_fused_convolution_23 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=3, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_23'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 200704
    N = 64
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

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
    xindex = idx_n + (64*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/pd/cpd4zj3skdyksa6u55gaopgcox7zlkmdwv6o775arualoghj75ng.py
# Source Nodes: [identity_3, out_17, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_3 => relu_6
# out_17 => add_37, add_40, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
# out_18 => add_41
triton_poi_fused__native_batch_norm_legit_functional_add_relu_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_24', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 200704.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ok/cokzdnjei7jnjeeu3iwjhmqu6bdfm2hqv2jw7bz4pv5ermw6c5h2.py
# Source Nodes: [out_30], Original ATen: [aten.convolution]
# out_30 => convolution_11
triton_tem_fused_convolution_25 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_25'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 200704
    N = 128
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

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
    xindex = idx_n + (128*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/7y/c7yoik25d46i6oeaysx6jbxx2kwhqijga3tpcg7l5ivtydyytevr.py
# Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
# out_31 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_26', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (32768*x1)), rmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/gu/cguzns5hr46oy7elih4fd22fetdqb7xsw7j3ycrlmuvo6fpng3b5.py
# Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
# out_31 => var_mean_11
triton_red_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_27', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (14336*x0)), rmask & xmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (14336*x0)), rmask & xmask, other=0)
        tmp2 = tl.load(in_ptr2 + (x1 + (128*r2) + (14336*x0)), rmask & xmask, other=0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (128*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (128*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (128*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/o6/co6uqtgqxf7zlyju2sr6gg2et2yhn5duzhecnm3iofx4fr75qb5g.py
# Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
# out_31 => add_59, add_60, add_61, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_per_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_28', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 200704.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 1.0000049824865598
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/jr/cjrbehdjs6y7qjq7rp3vuyfprto4nenvizzdrcknwrha6546ulf7.py
# Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_31 => add_59, add_62, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# out_32 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_relu_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_29', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 200704.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/3i/c3isew4i3zktxfsejg77o7lsovuo4tbsrgsrkgv7pha4cez3sgg2.py
# Source Nodes: [out_33], Original ATen: [aten.convolution]
# out_33 => convolution_12
triton_tem_fused_convolution_30 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_30'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 16

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 128
    IN_H = 56
    IN_W = 56
    OUT_C = 128
    OUT_H = 28
    OUT_W = 28

    # Strides:
    stride_xn = 401408
    stride_xc = 1
    stride_xh = 7168
    stride_xw = 128
    stride_wc_out = 1152
    stride_wc_in = 1
    stride_wh = 384
    stride_ww = 128

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (28*idx_h) + (784*idx_c) + (100352*idx_n)
    x5 = xindex % 784
    tl.store(out_ptr1 + (idx_c + (128*x5) + (100352*idx_n)), acc, mask)
''')
meta4 = {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': True, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_zhaowe58/zj/czjafhbxduvs76ekruhsst5ma5jdke5camvdmgesnfnrbp6mzpas.py
# Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
# out_34 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_31', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/kt/cktkg3xkeyrvw3c75wen2xyqdhyfhw4sw6ezoaezd3zpzzc6jvda.py
# Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
# out_34 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_32 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_32', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, other=0)
        tmp2 = tl.load(in_ptr2 + (x1 + (128*r2) + (12544*x0)), rmask & xmask, other=0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (128*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (128*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (128*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/sd/csdlysorszg7nqtqer7fkhjpiznzu5nyyg2klaow75kmi7vapz4k.py
# Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
# out_34 => add_64, add_65, add_66, mul_85, mul_86, mul_87, mul_88, mul_89, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_33', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 50176.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 1.0000199302441455
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/xd/cxd2n27v5srpyqbbyo55eeieoshc6dp5bsixgn2sghlwfhrcjyc6.py
# Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_34 => add_64, add_67, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
# out_35 => relu_11
triton_poi_fused__native_batch_norm_legit_functional_relu_34 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_34', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 50176.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/d7/cd7x3azfjdai2id3o3q26tet2r6flthneyoe2zvaacrzedw3hynz.py
# Source Nodes: [out_36], Original ATen: [aten.convolution]
# out_36 => convolution_13
triton_tem_fused_convolution_35 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_35'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 50176
    N = 512
    K = 128
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 128
    stride_ak = 1
    stride_bk = 1
    stride_bn = 128

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
    xindex = idx_n + (512*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/2r/c2rd5iyuqtkwpxdgdc5i37x5rrnuhwmnileanezmmw4ylwfbu3cw.py
# Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
# out_37 => var_mean_13
triton_red_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_36', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (131072*x1)), rmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/fy/cfynweh7xdrz7lvysch3psifohbpvztsicn7nd5nj4pdo77gbmfe.py
# Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
# out_37 => var_mean_13
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (512*r2) + (50176*x0)), rmask & xmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x1 + (512*r2) + (50176*x0)), rmask & xmask, other=0)
        tmp2 = tl.load(in_ptr2 + (x1 + (512*r2) + (50176*x0)), rmask & xmask, other=0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (512*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (512*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (512*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/3s/c3srsywntxc6kxjaszy6xgemsscxhffoglez2qicp6vsirh3luk5.py
# Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
# out_37 => add_69, add_70, add_71, mul_92, mul_93, mul_94, mul_95, mul_96, rsqrt_13, squeeze_40, var_mean_13
triton_per_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_38', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0)
    tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 50176.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 1.0000199302441455
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/27/c27rf4b2exchukjlwemwxb4mfyh6txlrrzb4atxsl67fmn2llnst.py
# Source Nodes: [getattr_l__self___layer2___0___downsample_0], Original ATen: [aten.convolution]
# getattr_l__self___layer2___0___downsample_0 => convolution_14
triton_tem_fused_convolution_39 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_39'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 16

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 256
    IN_H = 56
    IN_W = 56
    OUT_C = 512
    OUT_H = 28
    OUT_W = 28

    # Strides:
    stride_xn = 802816
    stride_xc = 1
    stride_xh = 14336
    stride_xw = 256
    stride_wc_out = 256
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)





    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (28*idx_h) + (784*idx_c) + (401408*idx_n)
    x5 = xindex % 784
    tl.store(out_ptr1 + (idx_c + (512*x5) + (401408*idx_n)), acc, mask)
''')
meta5 = {'KERNEL_H': 1, 'KERNEL_W': 1, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 0, 'PADDING_W': 0, 'GROUPS': 1, 'UNROLL': True, 'ALLOW_TF32': True, 'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_zhaowe58/v7/cv7f4rgvvsdb4jqcniuqac6pbhrqeghqrzglzkxx4os6t7iokzjq.py
# Source Nodes: [identity_5, identity_6, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_5 => add_74, add_77, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
# identity_6 => relu_12
# out_37 => add_69, add_72, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
# out_38 => add_78
triton_poi_fused__native_batch_norm_legit_functional_add_relu_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_40', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 50176.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/eq/cequfubfgwx3wnch7qdwsmiyhzuu3vrrz2fao2hawcowtuwl2oih.py
# Source Nodes: [out_40], Original ATen: [aten.convolution]
# out_40 => convolution_15
triton_tem_fused_convolution_41 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_41'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 50176
    N = 128
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
    xindex = idx_n + (128*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ss/cssmmeyd2e3t2vrkfhamkdr2ldvmcrzm6v5a3xj6omi3hnvprnhq.py
# Source Nodes: [out_43], Original ATen: [aten.convolution]
# out_43 => convolution_16
triton_tem_fused_convolution_42 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_42'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 16

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 128
    IN_H = 28
    IN_W = 28
    OUT_C = 128
    OUT_H = 28
    OUT_W = 28

    # Strides:
    stride_xn = 100352
    stride_xc = 1
    stride_xh = 3584
    stride_xw = 128
    stride_wc_out = 1152
    stride_wc_in = 1
    stride_wh = 384
    stride_ww = 128

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (28*idx_h) + (784*idx_c) + (100352*idx_n)
    x5 = xindex % 784
    tl.store(out_ptr1 + (idx_c + (128*x5) + (100352*idx_n)), acc, mask)
''')
meta6 = {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': True, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_zhaowe58/6v/c6v5tfv5k46fc3qu2eccvtre6yd7u7zd227zb7os6defevhkjanz.py
# Source Nodes: [identity_7, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_7 => relu_15
# out_47 => add_90, add_93, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
# out_48 => add_94
triton_poi_fused__native_batch_norm_legit_functional_add_relu_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_43', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 50176.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/e3/ce3i2tnse7upmk2hkffderw472ma4v2lhdrms6pbcix4weravpad.py
# Source Nodes: [out_70], Original ATen: [aten.convolution]
# out_70 => convolution_24
triton_tem_fused_convolution_44 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_44'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 50176
    N = 256
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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
    xindex = idx_n + (256*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ut/cutgkyt7lma3xk544xqh4adfk3oihq4t7xt4rjrdnogr2lsiwib3.py
# Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
# out_71 => var_mean_24
triton_red_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_45', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/p6/cp6v5i74pznuxnudjypoth72ta5aeat67bzjvqrsfzxavsbapy54.py
# Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
# out_71 => var_mean_24
triton_red_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_46', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, other=0)
        tmp2 = tl.load(in_ptr2 + (x1 + (256*r2) + (25088*x0)), rmask & xmask, other=0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x1 + (256*x0)), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + (256*x0)), tmp7, xmask)
    tl.store(out_ptr2 + (x1 + (256*x0)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/hc/chcz5jpgcaxwy74okrkzfr72nzeocgqxzgwaltyqk4z33eyc7v7p.py
# Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
# out_71 => add_128, add_129, add_130, mul_169, mul_170, mul_171, mul_172, mul_173, rsqrt_24, squeeze_73, var_mean_24
triton_per_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_47', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 50176.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 1.0000199302441455
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/jq/cjqs2tcbb2a3nxqqlqphwfjgasid6pknf3bbmqj7yhuukrfjjqaa.py
# Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_71 => add_128, add_131, mul_168, mul_174, rsqrt_24, sub_24, var_mean_24
# out_72 => relu_22
triton_poi_fused__native_batch_norm_legit_functional_relu_48 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_48', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 50176.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/3s/c3sbuv26jup7prbpyelutfkefnnofxxr4tyfanbzkdc7clgmlbrk.py
# Source Nodes: [out_73], Original ATen: [aten.convolution]
# out_73 => convolution_25
triton_tem_fused_convolution_49 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_49'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 256
    IN_H = 28
    IN_W = 28
    OUT_C = 256
    OUT_H = 14
    OUT_W = 14

    # Strides:
    stride_xn = 200704
    stride_xc = 1
    stride_xh = 7168
    stride_xw = 256
    stride_wc_out = 2304
    stride_wc_in = 1
    stride_wh = 768
    stride_ww = 256

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (14*idx_h) + (196*idx_c) + (50176*idx_n)
    x5 = xindex % 196
    tl.store(out_ptr1 + (idx_c + (256*x5) + (50176*idx_n)), acc, mask)
''')
meta7 = {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': True, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}


# kernel path: /tmp/torchinductor_zhaowe58/dd/cddodlnzlszkm64nid22457mrwy33ihdsqfyckszbsiu5iwseiou.py
# Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
# out_74 => var_mean_25
triton_red_fused__native_batch_norm_legit_functional_50 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_50', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/lx/clxo2g5bs4pig5pms73vs2hv7y2xxydcfx5padkqzhvxfmwqasoc.py
# Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
# out_74 => add_133, add_134, add_135, mul_176, mul_177, mul_178, mul_179, mul_180, rsqrt_25, squeeze_76, var_mean_25
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0)
        tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp18 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 12544.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 1.0000797257434426
    tmp15 = tmp10 * tmp14
    tmp16 = 0.1
    tmp17 = tmp15 * tmp16
    tmp19 = 0.9
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = tmp6 * tmp16
    tmp24 = tmp23 * tmp19
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
    tl.store(out_ptr4 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/x3/cx3r3ixdosekirb3p5juhbd5fschfvyk2dsmtmyk4kpw4sysz4s4.py
# Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_74 => add_133, add_136, mul_175, mul_181, rsqrt_25, sub_25, var_mean_25
# out_75 => relu_23
triton_poi_fused__native_batch_norm_legit_functional_relu_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_52', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/tt/cttyw74wgxasdtswk7c66ucehq32r4qmddgv7anjij7n7usl3voy.py
# Source Nodes: [out_76], Original ATen: [aten.convolution]
# out_76 => convolution_26
triton_tem_fused_convolution_53 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_53'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 12544
    N = 1024
    K = 256
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 256
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

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
    xindex = idx_n + (1024*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/bf/cbfrhg54dibiy3evq3cbifl4wsnmytifsoqa6ohaibfj5fqpxusw.py
# Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
# out_77 => var_mean_26
triton_red_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_54', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/mt/cmtswq3bfgmk3vlicohpofb5elzdpnoulebfmy6pzxa2s3hgfzqc.py
# Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
# out_77 => add_138, add_139, add_140, mul_183, mul_184, mul_185, mul_186, mul_187, rsqrt_26, squeeze_79, var_mean_26
triton_red_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_55', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0)
        tmp2 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp18 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 12544.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 1.0000797257434426
    tmp15 = tmp10 * tmp14
    tmp16 = 0.1
    tmp17 = tmp15 * tmp16
    tmp19 = 0.9
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = tmp6 * tmp16
    tmp24 = tmp23 * tmp19
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
    tl.store(out_ptr4 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/pr/cprxfjdbhkgvrkdp2ja6jbg4wvt7gyhgosqfgjv3dfynz5bwshyi.py
# Source Nodes: [getattr_l__self___layer3___0___downsample_0], Original ATen: [aten.convolution]
# getattr_l__self___layer3___0___downsample_0 => convolution_27
triton_tem_fused_convolution_56 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_56'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 256
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 512
    IN_H = 28
    IN_W = 28
    OUT_C = 1024
    OUT_H = 14
    OUT_W = 14

    # Strides:
    stride_xn = 401408
    stride_xc = 1
    stride_xh = 14336
    stride_xw = 512
    stride_wc_out = 512
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)





    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (14*idx_h) + (196*idx_c) + (200704*idx_n)
    x5 = xindex % 196
    tl.store(out_ptr1 + (idx_c + (1024*x5) + (200704*idx_n)), acc, mask)
''')
meta8 = {'KERNEL_H': 1, 'KERNEL_W': 1, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 0, 'PADDING_W': 0, 'GROUPS': 1, 'UNROLL': True, 'ALLOW_TF32': True, 'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_zhaowe58/px/cpxnsj3v6uxbympju663fuitvb5hyinlydmpk6a3r3rbm7va6qmj.py
# Source Nodes: [identity_10, identity_11, out_77, out_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_10 => add_143, add_146, mul_189, mul_195, rsqrt_27, sub_27, var_mean_27
# identity_11 => relu_24
# out_77 => add_138, add_141, mul_182, mul_188, rsqrt_26, sub_26, var_mean_26
# out_78 => add_147
triton_poi_fused__native_batch_norm_legit_functional_add_relu_57 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_57', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/hj/chjpscvdb2vy2mclhxhudwifiyzur237x2tfcfvaolajm46bcbcd.py
# Source Nodes: [out_80], Original ATen: [aten.convolution]
# out_80 => convolution_28
triton_tem_fused_convolution_58 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=3, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_58'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 12544
    N = 256
    K = 1024
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1024
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1024

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
    xindex = idx_n + (256*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')
meta9 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}


# kernel path: /tmp/torchinductor_zhaowe58/ey/ceyoflljyl4obwbneqv44mf7cwyy45cq4zerljmirogczv73m3i6.py
# Source Nodes: [out_83], Original ATen: [aten.convolution]
# out_83 => convolution_29
triton_tem_fused_convolution_59 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_59'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 256
    IN_H = 14
    IN_W = 14
    OUT_C = 256
    OUT_H = 14
    OUT_W = 14

    # Strides:
    stride_xn = 50176
    stride_xc = 1
    stride_xh = 3584
    stride_xw = 256
    stride_wc_out = 2304
    stride_wc_in = 1
    stride_wh = 768
    stride_ww = 256

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (14*idx_h) + (196*idx_c) + (50176*idx_n)
    x5 = xindex % 196
    tl.store(out_ptr1 + (idx_c + (256*x5) + (50176*idx_n)), acc, mask)
''')
meta10 = {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': True, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}


# kernel path: /tmp/torchinductor_zhaowe58/gt/cgtudw2ygp67evvlfnjgie4jty4btnvn74hdvlo6fagwecaix6wu.py
# Source Nodes: [identity_12, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_12 => relu_27
# out_87 => add_159, add_162, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
# out_88 => add_163
triton_poi_fused__native_batch_norm_legit_functional_add_relu_60 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_60', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/in/cinhnfz7ffrnjq2cbuj75q4cgqatj73dy35clzjt6y2g6dc6piro.py
# Source Nodes: [out_130], Original ATen: [aten.convolution]
# out_130 => convolution_43
triton_tem_fused_convolution_61 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_61'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 12544
    N = 512
    K = 1024
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1024
    stride_ak = 1
    stride_bk = 1
    stride_bn = 1024

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
    xindex = idx_n + (512*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/24/c243lj4bxiwoismszwiwczjdqgtwp3jw4l42xz66clo5qc2rmrm6.py
# Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
# out_131 => var_mean_43
triton_red_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_62', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, other=0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ge/cgezci2jlbrgxpwqcsvphiajapqwmtb6rcmvgxbnpkm2uro7xfso.py
# Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
# out_131 => add_229, add_230, add_231, mul_302, mul_303, mul_304, mul_305, mul_306, rsqrt_43, squeeze_130, var_mean_43
triton_red_fused__native_batch_norm_legit_functional_63 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_63', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0)
        tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp18 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 12544.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 1.0000797257434426
    tmp15 = tmp10 * tmp14
    tmp16 = 0.1
    tmp17 = tmp15 * tmp16
    tmp19 = 0.9
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = tmp6 * tmp16
    tmp24 = tmp23 * tmp19
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
    tl.store(out_ptr4 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/u6/cu64ma6vlobylb5v74pkq5nnt67vvqsjjbcicsj5a2cgrp2wgpqe.py
# Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_131 => add_229, add_232, mul_301, mul_307, rsqrt_43, sub_43, var_mean_43
# out_132 => relu_40
triton_poi_fused__native_batch_norm_legit_functional_relu_64 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_64', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 12544.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ni/cnidhekkcggmqr7bhosorxjtlju6wlu42af26vffq344mysr5xka.py
# Source Nodes: [out_133], Original ATen: [aten.convolution]
# out_133 => convolution_44
triton_tem_fused_convolution_65 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_65'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 512
    IN_H = 14
    IN_W = 14
    OUT_C = 512
    OUT_H = 7
    OUT_W = 7

    # Strides:
    stride_xn = 100352
    stride_xc = 1
    stride_xh = 7168
    stride_xw = 512
    stride_wc_out = 4608
    stride_wc_in = 1
    stride_wh = 1536
    stride_ww = 512

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (7*idx_h) + (49*idx_c) + (25088*idx_n)
    x5 = xindex % 49
    tl.store(out_ptr1 + (idx_c + (512*x5) + (25088*idx_n)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/37/c37trtpsanglxpszzcf5t5lnvdenfhtawg3hchc3tzy7ygajlwis.py
# Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
# out_134 => var_mean_44
triton_red_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_66', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(r2 % 7)) + (3584*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2 & xmask, other=0)
        tmp4 = tl.where(tmp2, tmp3, 0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp2, tmp5, 0)
        tmp7 = 1.0
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp10 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_combine(
            tmp12_mean, tmp12_m2, tmp12_weight,
            tmp9, tmp10, tmp11
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
    tl.store(out_ptr2 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/fv/cfvknx63abjqukekfrxxaqva6jxohev44uuruhfsvkkmmoffa55b.py
# Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
# out_134 => add_234, add_235, add_236, mul_309, mul_310, mul_311, mul_312, mul_313, rsqrt_44, squeeze_133, var_mean_44
triton_per_fused__native_batch_norm_legit_functional_67 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_67', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0)
    tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 3136.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 1.0003189792663476
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/wu/cwuz2oqovassaobuea6n7frx2ipdovwrprjk5ikyb22anyozvq3h.py
# Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_134 => add_234, add_237, mul_308, mul_314, rsqrt_44, sub_44, var_mean_44
# out_135 => relu_41
triton_poi_fused__native_batch_norm_legit_functional_relu_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_68', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3136.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/wi/cwixzwzrl73z6aekvteulugh57apeksgf7e4uqq4cod2ot2pi52b.py
# Source Nodes: [out_136], Original ATen: [aten.convolution]
# out_136 => convolution_45
triton_tem_fused_convolution_69 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_69'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 3136
    N = 2048
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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


# kernel path: /tmp/torchinductor_zhaowe58/uo/cuouute7rwd2yckf6oaworqz45vy2sawyzyydvbgfvx44hqplxo4.py
# Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
# out_137 => var_mean_45
triton_red_fused__native_batch_norm_legit_functional_70 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_70', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*(r2 % 7)) + (14336*(((r2 + (126*x1)) // 7) % 448))), rmask & tmp2, other=0)
        tmp4 = tl.where(tmp2, tmp3, 0)
        tmp5 = 0.0
        tmp6 = tl.where(tmp2, tmp5, 0)
        tmp7 = 1.0
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp10 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_combine(
            tmp12_mean, tmp12_m2, tmp12_weight,
            tmp9, tmp10, tmp11
        )
        tmp12_mean = tl.where(rmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask, tmp12_weight_next, tmp12_weight)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp13, None)
    tl.store(out_ptr2 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/7y/c7yidiy65fcy6krzq6kcfsvtfjfqajivl5o3uow4kdyqddectoes.py
# Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
# out_137 => add_239, add_240, add_241, mul_316, mul_317, mul_318, mul_319, mul_320, rsqrt_45, squeeze_136, var_mean_45
triton_per_fused__native_batch_norm_legit_functional_71 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_71', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (2048*r1)), rmask, other=0)
    tmp2 = tl.load(in_ptr2 + (x0 + (2048*r1)), rmask, other=0)
    tmp25 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp3, 0)
    tmp8 = tl.where(rmask, tmp4, 0)
    tmp9 = tl.where(rmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 3136.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 1.0003189792663476
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, None)
    tl.store(out_ptr3 + (x0), tmp28, None)
    tl.store(out_ptr4 + (x0), tmp32, None)
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(out_ptr1 + (x0), tmp14, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/ta/ctazlncu7upubbvo4bd3vaykhhuvig7q2u56pvngf4lt7cbrh4vl.py
# Source Nodes: [getattr_l__self___layer4___0___downsample_0], Original ATen: [aten.convolution]
# getattr_l__self___layer4___0___downsample_0 => convolution_46
triton_tem_fused_convolution_72 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=8, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_72'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 1024
    IN_H = 14
    IN_W = 14
    OUT_C = 2048
    OUT_H = 7
    OUT_W = 7

    # Strides:
    stride_xn = 200704
    stride_xc = 1
    stride_xh = 14336
    stride_xw = 1024
    stride_wc_out = 1024
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)





    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (7*idx_h) + (49*idx_c) + (100352*idx_n)
    x5 = xindex % 49
    tl.store(out_ptr1 + (idx_c + (2048*x5) + (100352*idx_n)), acc, mask)
''')
meta11 = {'KERNEL_H': 1, 'KERNEL_W': 1, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 0, 'PADDING_W': 0, 'GROUPS': 1, 'UNROLL': True, 'ALLOW_TF32': True, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}


# kernel path: /tmp/torchinductor_zhaowe58/7t/c7tyc2creny27ndq7pd7j3ju5iv7n2ejamvz5yznbwgpklsffm2f.py
# Source Nodes: [identity_17, identity_18, out_137, out_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_17 => add_244, add_247, mul_322, mul_328, rsqrt_46, sub_46, var_mean_46
# identity_18 => relu_42
# out_137 => add_239, add_242, mul_315, mul_321, rsqrt_45, sub_45, var_mean_45
# out_138 => add_248
triton_poi_fused__native_batch_norm_legit_functional_add_relu_73 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_73', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 3136.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/bc/cbczwbbot77h7cjpaauq576uv5sgfypo7dgq5r6uyt6jwjnrxzg2.py
# Source Nodes: [out_140], Original ATen: [aten.convolution]
# out_140 => convolution_47
triton_tem_fused_convolution_74 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=3, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_74'})
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 3136
    N = 512
    K = 2048
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2048
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2048

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
    xindex = idx_n + (512*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/43/c43muc5hme2rpradfhjxw2xveiq5l2qqzvrkphu3fki7yfgsfn4h.py
# Source Nodes: [out_143], Original ATen: [aten.convolution]
# out_143 => convolution_48
triton_tem_fused_convolution_75 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_convolution_75'})
@triton.jit
def triton_(arg_X, arg_W, out_ptr1):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32

    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 64
    IN_C = 512
    IN_H = 7
    IN_W = 7
    OUT_C = 512
    OUT_H = 7
    OUT_W = 7

    # Strides:
    stride_xn = 25088
    stride_xc = 1
    stride_xh = 3584
    stride_xw = 512
    stride_wc_out = 4608
    stride_wc_in = 1
    stride_wh = 1536
    stride_ww = 512

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + (7*idx_h) + (49*idx_c) + (25088*idx_n)
    x5 = xindex % 49
    tl.store(out_ptr1 + (idx_c + (512*x5) + (25088*idx_n)), acc, mask)
''')


# kernel path: /tmp/torchinductor_zhaowe58/uy/cuyusallauv6yl5m3iovnuqty6x63nvm5az3xgn46wyctvjj3sa7.py
# Source Nodes: [identity_19, out_147, out_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# identity_19 => relu_45
# out_147 => add_260, add_263, mul_343, mul_349, rsqrt_49, sub_49, var_mean_49
# out_148 => add_264
triton_poi_fused__native_batch_norm_legit_functional_add_relu_76 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_76', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 3136.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/a7/ca7qvrup5uhuwxe2s4kjtb5h4yakicopnrcvultqkem7ah2swaxq.py
# Source Nodes: [out_157, out_158, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# out_157 => add_276, add_279, mul_364, mul_370, rsqrt_52, sub_52, var_mean_52
# out_158 => add_280
# x_7 => relu_48
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_77 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_77', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 3136.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + (x2), tmp16, None)
    tl.store(out_ptr1 + (x2), tmp18, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/tb/ctbpu64dr3qokkc5fs72n2kyzeldiidvejkmcmhu5znebhkzqor2.py
# Source Nodes: [x_8, x_9], Original ATen: [aten.mean, aten.view]
# x_8 => mean
# x_9 => view
triton_per_fused_mean_view_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_78', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (100352*x1)), rmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 49.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_zhaowe58/bi/cbilk4ftioqerzue2x3tmzu2uk6v7entfojn6bnp2q5cuw4azdpp.py
# Source Nodes: [x_10], Original ATen: [aten.addmm]
# x_10 => addmm
triton_tem_fused_addmm_79 = async_compile.triton('triton_', '''
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@template(num_stages=2, num_warps=4, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())], 'kernel_name': 'triton_tem_fused_addmm_79'})
@triton.jit
def triton_(in_ptr0, arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128

    A = arg_A
    B = arg_B

    M = 64
    N = 1000
    K = 2048
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2048
    stride_ak = 1
    stride_bk = 1
    stride_bn = 2048

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
    xindex = idx_n + (1000*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, mask.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), tmp1, mask)
''')
meta12 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}


# kernel path: /tmp/torchinductor_zhaowe58/kf/ckf3zjjkslarlxu4xvk3tu2l5tguzln2bdcbzzniw53s3v4qt4rg.py
# Source Nodes: [x_1], Original ATen: [aten.add]
# x_1 => add
triton_poi_fused_add_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_80', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_81, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_85, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1024, ), (1, ))
    assert_size_stride(primals_94, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_102, (1024, ), (1, ))
    assert_size_stride(primals_103, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (256, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_120, (1024, ), (1, ))
    assert_size_stride(primals_121, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (256, ), (1, ))
    assert_size_stride(primals_127, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (1024, ), (1, ))
    assert_size_stride(primals_130, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (512, ), (1, ))
    assert_size_stride(primals_136, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_138, (2048, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (2048, ), (1, ))
    assert_size_stride(primals_142, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (512, ), (1, ))
    assert_size_stride(primals_148, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_150, (2048, ), (1, ))
    assert_size_stride(primals_151, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_153, (512, ), (1, ))
    assert_size_stride(primals_154, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (512, ), (1, ))
    assert_size_stride(primals_157, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_159, (2048, ), (1, ))
    assert_size_stride(primals_160, (1000, 2048), (2048, 1))
    assert_size_stride(primals_161, (1000, ), (1, ))
    assert_size_stride(primals_162, (64, ), (1, ))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (), ())
    assert_size_stride(primals_165, (64, ), (1, ))
    assert_size_stride(primals_166, (64, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (64, ), (1, ))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (64, ), (1, ))
    assert_size_stride(primals_178, (64, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (64, ), (1, ))
    assert_size_stride(primals_181, (64, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (64, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (128, ), (1, ))
    assert_size_stride(primals_199, (128, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (512, ), (1, ))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (128, ), (1, ))
    assert_size_stride(primals_208, (128, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (128, ), (1, ))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (128, ), (1, ))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (128, ), (1, ))
    assert_size_stride(primals_220, (128, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (128, ), (1, ))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (256, ), (1, ))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (256, ), (1, ))
    assert_size_stride(primals_247, (256, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (256, ), (1, ))
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (1024, ), (1, ))
    assert_size_stride(primals_253, (1024, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (256, ), (1, ))
    assert_size_stride(primals_256, (256, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (256, ), (1, ))
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (1024, ), (1, ))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_263, (), ())
    assert_size_stride(primals_264, (256, ), (1, ))
    assert_size_stride(primals_265, (256, ), (1, ))
    assert_size_stride(primals_266, (), ())
    assert_size_stride(primals_267, (256, ), (1, ))
    assert_size_stride(primals_268, (256, ), (1, ))
    assert_size_stride(primals_269, (), ())
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_272, (), ())
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_274, (256, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (256, ), (1, ))
    assert_size_stride(primals_277, (256, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (256, ), (1, ))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (256, ), (1, ))
    assert_size_stride(primals_286, (256, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (512, ), (1, ))
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (2048, ), (1, ))
    assert_size_stride(primals_298, (2048, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (2048, ), (1, ))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (512, ), (1, ))
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (512, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (2048, ), (1, ))
    assert_size_stride(primals_310, (2048, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (512, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (512, ), (1, ))
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (2048, ), (1, ))
    assert_size_stride(primals_319, (2048, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (64, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 49, grid=grid(192, 49), stream=stream0)
        del primals_1
        buf1 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_7, buf1, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_7
        buf2 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_19, buf2, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_19
        buf3 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_28, buf3, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_28
        buf4 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_37, buf4, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_37
        buf5 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_49, buf5, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_49
        buf6 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_58, buf6, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_58
        buf7 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_67, buf7, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_67
        buf8 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_76, buf8, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_76
        buf9 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_88, buf9, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_88
        buf10 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_97, buf10, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_97
        buf11 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_106, buf11, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_106
        buf12 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_115, buf12, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_115
        buf13 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_124, buf13, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_124
        buf14 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_133, buf14, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_133
        buf15 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_145, buf15, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_145
        buf16 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_154, buf16, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_154
        buf17 = empty_strided((64, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_321, buf17, 192, 50176, grid=grid(192, 50176), stream=stream0)
        del primals_321
        buf19 = empty_strided((64, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_6.run(buf17, buf0, buf19, grid=torch._inductor.kernel.conv.conv_grid(64, 64, 112, 112, meta0), stream=stream0)
        buf20 = empty_strided((1, 64, 1, 1, 656), (41984, 1, 41984, 41984, 64), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((1, 64, 1, 1, 656), (41984, 1, 41984, 41984, 64), device='cuda', dtype=torch.float32)
        buf22 = empty_strided((1, 64, 1, 1, 656), (41984, 1, 41984, 41984, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf19, buf20, buf21, buf22, 41984, 1224, grid=grid(41984), stream=stream0)
        buf23 = empty_strided((1, 64, 1, 1, 6), (384, 1, 384, 384, 64), device='cuda', dtype=torch.float32)
        buf24 = empty_strided((1, 64, 1, 1, 6), (384, 1, 384, 384, 64), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((1, 64, 1, 1, 6), (384, 1, 384, 384, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf20, buf21, buf22, buf23, buf24, buf25, 384, 110, grid=grid(384), stream=stream0)
        del buf20
        del buf21
        del buf22
        buf26 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf27 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf29 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf31 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf30 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_9.run(buf23, buf24, buf25, primals_163, primals_162, buf26, buf27, buf29, buf31, buf30, 64, 6, grid=grid(64), stream=stream0)
        del buf23
        del buf24
        del buf25
        del primals_162
        del primals_163
        buf32 = empty_strided((64, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_10.run(buf19, buf26, buf27, primals_2, primals_3, buf32, 51380224, grid=grid(51380224), stream=stream0)
        del primals_3
        buf33 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        buf34 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [identity], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_11.run(buf32, buf33, buf34, 12845056, grid=grid(12845056), stream=stream0)
        buf35 = empty_strided((200704, 64), (64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_12.run(buf33, primals_4, buf35, grid=torch._inductor.kernel.mm_common.mm_grid(200704, 64, meta1), stream=stream0)
        buf36 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        buf38 = empty_strided((1, 64, 1, 1, 784), (50176, 1, 50176, 50176, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf35, buf36, buf37, buf38, 50176, 256, grid=grid(50176), stream=stream0)
        buf39 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf40 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        buf41 = empty_strided((1, 64, 1, 1, 7), (448, 1, 448, 448, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf36, buf37, buf38, buf39, buf40, buf41, 448, 112, grid=grid(448), stream=stream0)
        buf42 = buf27; del buf27  # reuse
        buf43 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf45 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf46 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf39, buf40, buf41, primals_166, primals_165, buf42, buf43, buf45, buf47, buf46, 64, 7, grid=grid(64), stream=stream0)
        del primals_165
        del primals_166
        buf48 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf35, buf42, buf43, primals_5, primals_6, buf48, 12845056, grid=grid(12845056), stream=stream0)
        del primals_6
        buf50 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_3], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_17.run(buf48, buf1, buf50, grid=torch._inductor.kernel.conv.conv_grid(64, 64, 56, 56, meta2), stream=stream0)
        buf51 = buf38; del buf38  # reuse
        buf52 = buf37; del buf37  # reuse
        buf53 = buf36; del buf36  # reuse
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf50, buf51, buf52, buf53, 50176, 256, grid=grid(50176), stream=stream0)
        buf54 = buf41; del buf41  # reuse
        buf55 = buf40; del buf40  # reuse
        buf56 = buf39; del buf39  # reuse
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf51, buf52, buf53, buf54, buf55, buf56, 448, 112, grid=grid(448), stream=stream0)
        buf57 = buf43; del buf43  # reuse
        buf58 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf60 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf62 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf54, buf55, buf56, primals_169, primals_168, buf57, buf58, buf60, buf62, buf61, 64, 7, grid=grid(64), stream=stream0)
        del primals_168
        del primals_169
        buf63 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf50, buf57, buf58, primals_8, primals_9, buf63, 12845056, grid=grid(12845056), stream=stream0)
        del primals_9
        buf64 = empty_strided((200704, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_6], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_18.run(buf63, primals_10, buf64, grid=torch._inductor.kernel.mm_common.mm_grid(200704, 256, meta3), stream=stream0)
        buf65 = empty_strided((1, 256, 1, 1, 328), (83968, 1, 83968, 83968, 256), device='cuda', dtype=torch.float32)
        buf66 = empty_strided((1, 256, 1, 1, 328), (83968, 1, 83968, 83968, 256), device='cuda', dtype=torch.float32)
        buf67 = empty_strided((1, 256, 1, 1, 328), (83968, 1, 83968, 83968, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf64, buf65, buf66, buf67, 83968, 612, grid=grid(83968), stream=stream0)
        buf68 = empty_strided((1, 256, 1, 1, 3), (768, 1, 768, 768, 256), device='cuda', dtype=torch.float32)
        buf69 = empty_strided((1, 256, 1, 1, 3), (768, 1, 768, 768, 256), device='cuda', dtype=torch.float32)
        buf70 = empty_strided((1, 256, 1, 1, 3), (768, 1, 768, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf65, buf66, buf67, buf68, buf69, buf70, 768, 110, grid=grid(768), stream=stream0)
        buf71 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf74 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf76 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf68, buf69, buf70, primals_172, primals_171, buf71, buf72, buf74, buf76, buf75, 256, 3, grid=grid(256), stream=stream0)
        del primals_171
        del primals_172
        buf77 = empty_strided((200704, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__self___layer1___0___downsample_0], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_18.run(buf33, primals_13, buf77, grid=torch._inductor.kernel.mm_common.mm_grid(200704, 256, meta3), stream=stream0)
        buf78 = buf67; del buf67  # reuse
        buf79 = buf66; del buf66  # reuse
        buf80 = buf65; del buf65  # reuse
        # Source Nodes: [identity_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf77, buf78, buf79, buf80, 83968, 612, grid=grid(83968), stream=stream0)
        buf81 = buf70; del buf70  # reuse
        buf82 = buf69; del buf69  # reuse
        buf83 = buf68; del buf68  # reuse
        # Source Nodes: [identity_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf78, buf79, buf80, buf81, buf82, buf83, 768, 110, grid=grid(768), stream=stream0)
        buf84 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf85 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf87 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf89 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf88 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf81, buf82, buf83, primals_175, primals_174, buf84, buf85, buf87, buf89, buf88, 256, 3, grid=grid(256), stream=stream0)
        del primals_174
        del primals_175
        buf90 = empty_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        buf91 = buf90; del buf90  # reuse
        # Source Nodes: [identity_1, identity_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf91, buf64, buf71, buf72, primals_11, primals_12, buf77, buf84, buf85, primals_14, primals_15, 51380224, grid=grid(51380224), stream=stream0)
        del primals_12
        del primals_15
        buf92 = empty_strided((200704, 64), (64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_10], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_23.run(buf91, primals_16, buf92, grid=torch._inductor.kernel.mm_common.mm_grid(200704, 64, meta1), stream=stream0)
        buf93 = buf53; del buf53  # reuse
        buf94 = buf52; del buf52  # reuse
        buf95 = buf51; del buf51  # reuse
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf92, buf93, buf94, buf95, 50176, 256, grid=grid(50176), stream=stream0)
        buf96 = buf56; del buf56  # reuse
        buf97 = buf55; del buf55  # reuse
        buf98 = buf54; del buf54  # reuse
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf93, buf94, buf95, buf96, buf97, buf98, 448, 112, grid=grid(448), stream=stream0)
        buf99 = buf58; del buf58  # reuse
        buf100 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf102 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf103 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf96, buf97, buf98, primals_178, primals_177, buf99, buf100, buf102, buf104, buf103, 64, 7, grid=grid(64), stream=stream0)
        del primals_177
        del primals_178
        buf105 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf92, buf99, buf100, primals_17, primals_18, buf105, 12845056, grid=grid(12845056), stream=stream0)
        del primals_18
        buf107 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_13], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_17.run(buf105, buf2, buf107, grid=torch._inductor.kernel.conv.conv_grid(64, 64, 56, 56, meta2), stream=stream0)
        buf108 = buf95; del buf95  # reuse
        buf109 = buf94; del buf94  # reuse
        buf110 = buf93; del buf93  # reuse
        # Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf107, buf108, buf109, buf110, 50176, 256, grid=grid(50176), stream=stream0)
        buf111 = buf98; del buf98  # reuse
        buf112 = buf97; del buf97  # reuse
        buf113 = buf96; del buf96  # reuse
        # Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf108, buf109, buf110, buf111, buf112, buf113, 448, 112, grid=grid(448), stream=stream0)
        buf114 = buf100; del buf100  # reuse
        buf115 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf117 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf119 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf111, buf112, buf113, primals_181, primals_180, buf114, buf115, buf117, buf119, buf118, 64, 7, grid=grid(64), stream=stream0)
        del primals_180
        del primals_181
        buf120 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf107, buf114, buf115, primals_20, primals_21, buf120, 12845056, grid=grid(12845056), stream=stream0)
        del primals_21
        buf121 = empty_strided((200704, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_18.run(buf120, primals_22, buf121, grid=torch._inductor.kernel.mm_common.mm_grid(200704, 256, meta3), stream=stream0)
        buf122 = buf80; del buf80  # reuse
        buf123 = buf79; del buf79  # reuse
        buf124 = buf78; del buf78  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf121, buf122, buf123, buf124, 83968, 612, grid=grid(83968), stream=stream0)
        buf125 = buf83; del buf83  # reuse
        buf126 = buf82; del buf82  # reuse
        buf127 = buf81; del buf81  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf122, buf123, buf124, buf125, buf126, buf127, 768, 110, grid=grid(768), stream=stream0)
        buf128 = buf85; del buf85  # reuse
        buf129 = buf72; del buf72  # reuse
        buf131 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf133 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf132 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf125, buf126, buf127, primals_184, primals_183, buf128, buf129, buf131, buf133, buf132, 256, 3, grid=grid(256), stream=stream0)
        del primals_183
        del primals_184
        buf134 = empty_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_3, out_17, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_24.run(buf121, buf128, buf129, primals_23, primals_24, buf91, buf134, 51380224, grid=grid(51380224), stream=stream0)
        del primals_24
        buf135 = empty_strided((200704, 64), (64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_23.run(buf134, primals_25, buf135, grid=torch._inductor.kernel.mm_common.mm_grid(200704, 64, meta1), stream=stream0)
        buf136 = buf110; del buf110  # reuse
        buf137 = buf109; del buf109  # reuse
        buf138 = buf108; del buf108  # reuse
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf135, buf136, buf137, buf138, 50176, 256, grid=grid(50176), stream=stream0)
        buf139 = buf113; del buf113  # reuse
        buf140 = buf112; del buf112  # reuse
        buf141 = buf111; del buf111  # reuse
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf136, buf137, buf138, buf139, buf140, buf141, 448, 112, grid=grid(448), stream=stream0)
        buf142 = buf115; del buf115  # reuse
        buf143 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf145 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf147 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf146 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf139, buf140, buf141, primals_187, primals_186, buf142, buf143, buf145, buf147, buf146, 64, 7, grid=grid(64), stream=stream0)
        del primals_186
        del primals_187
        buf148 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf135, buf142, buf143, primals_26, primals_27, buf148, 12845056, grid=grid(12845056), stream=stream0)
        del primals_27
        buf150 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_23], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_17.run(buf148, buf3, buf150, grid=torch._inductor.kernel.conv.conv_grid(64, 64, 56, 56, meta2), stream=stream0)
        buf151 = buf138; del buf138  # reuse
        buf152 = buf137; del buf137  # reuse
        buf153 = buf136; del buf136  # reuse
        # Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf150, buf151, buf152, buf153, 50176, 256, grid=grid(50176), stream=stream0)
        buf154 = buf141; del buf141  # reuse
        buf155 = buf140; del buf140  # reuse
        buf156 = buf139; del buf139  # reuse
        # Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf151, buf152, buf153, buf154, buf155, buf156, 448, 112, grid=grid(448), stream=stream0)
        buf157 = buf143; del buf143  # reuse
        buf158 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf160 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf162 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        buf161 = empty_strided((64, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf154, buf155, buf156, primals_190, primals_189, buf157, buf158, buf160, buf162, buf161, 64, 7, grid=grid(64), stream=stream0)
        del buf154
        del buf155
        del buf156
        del primals_189
        del primals_190
        buf163 = empty_strided((64, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf150, buf157, buf158, primals_29, primals_30, buf163, 12845056, grid=grid(12845056), stream=stream0)
        del buf158
        del primals_30
        buf164 = empty_strided((200704, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_26], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_18.run(buf163, primals_31, buf164, grid=torch._inductor.kernel.mm_common.mm_grid(200704, 256, meta3), stream=stream0)
        buf165 = buf124; del buf124  # reuse
        buf166 = buf123; del buf123  # reuse
        buf167 = buf122; del buf122  # reuse
        # Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_19.run(buf164, buf165, buf166, buf167, 83968, 612, grid=grid(83968), stream=stream0)
        buf168 = buf127; del buf127  # reuse
        buf169 = buf126; del buf126  # reuse
        buf170 = buf125; del buf125  # reuse
        # Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf165, buf166, buf167, buf168, buf169, buf170, 768, 110, grid=grid(768), stream=stream0)
        del buf165
        del buf166
        del buf167
        buf171 = buf129; del buf129  # reuse
        buf172 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf174 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf176 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf175 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_21.run(buf168, buf169, buf170, primals_193, primals_192, buf171, buf172, buf174, buf176, buf175, 256, 3, grid=grid(256), stream=stream0)
        del buf168
        del buf169
        del buf170
        del primals_192
        del primals_193
        buf177 = empty_strided((64, 256, 56, 56), (802816, 1, 14336, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_4, out_27, out_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_24.run(buf164, buf171, buf172, primals_32, primals_33, buf134, buf177, 51380224, grid=grid(51380224), stream=stream0)
        del primals_33
        buf178 = empty_strided((200704, 128), (128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_30], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_25.run(buf177, primals_34, buf178, grid=torch._inductor.kernel.mm_common.mm_grid(200704, 128, meta3), stream=stream0)
        buf179 = empty_strided((1, 128, 1, 1, 784), (100352, 1, 100352, 100352, 128), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((1, 128, 1, 1, 784), (100352, 1, 100352, 100352, 128), device='cuda', dtype=torch.float32)
        buf181 = empty_strided((1, 128, 1, 1, 784), (100352, 1, 100352, 100352, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf178, buf179, buf180, buf181, 100352, 256, grid=grid(100352), stream=stream0)
        buf182 = empty_strided((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), device='cuda', dtype=torch.float32)
        buf183 = empty_strided((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), device='cuda', dtype=torch.float32)
        buf184 = empty_strided((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_27.run(buf179, buf180, buf181, buf182, buf183, buf184, 896, 112, grid=grid(896), stream=stream0)
        buf185 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf186 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf188 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf190 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf189 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_28.run(buf182, buf183, buf184, primals_196, primals_195, buf185, buf186, buf188, buf190, buf189, 128, 7, grid=grid(128), stream=stream0)
        del buf182
        del buf183
        del buf184
        del primals_195
        del primals_196
        buf191 = empty_strided((64, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_29.run(buf178, buf185, buf186, primals_35, primals_36, buf191, 25690112, grid=grid(25690112), stream=stream0)
        del primals_36
        buf193 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_33], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_30.run(buf191, buf4, buf193, grid=torch._inductor.kernel.conv.conv_grid(64, 128, 28, 28, meta4), stream=stream0)
        buf194 = reinterpret_tensor(buf153, (1, 128, 1, 1, 392), (50176, 1, 50176, 50176, 128), 0); del buf153  # reuse
        buf195 = reinterpret_tensor(buf152, (1, 128, 1, 1, 392), (50176, 1, 50176, 50176, 128), 0); del buf152  # reuse
        buf196 = reinterpret_tensor(buf151, (1, 128, 1, 1, 392), (50176, 1, 50176, 50176, 128), 0); del buf151  # reuse
        # Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf193, buf194, buf195, buf196, 50176, 128, grid=grid(50176), stream=stream0)
        buf197 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf199 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf194, buf195, buf196, buf197, buf198, buf199, 512, 98, grid=grid(512), stream=stream0)
        buf200 = buf186; del buf186  # reuse
        buf201 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf203 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf205 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf204 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_33.run(buf197, buf198, buf199, primals_199, primals_198, buf200, buf201, buf203, buf205, buf204, 128, 4, grid=grid(128), stream=stream0)
        del primals_198
        del primals_199
        buf206 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_34.run(buf193, buf200, buf201, primals_38, primals_39, buf206, 6422528, grid=grid(6422528), stream=stream0)
        del primals_39
        buf207 = empty_strided((50176, 512), (512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_35.run(buf206, primals_40, buf207, grid=torch._inductor.kernel.mm_common.mm_grid(50176, 512, meta3), stream=stream0)
        buf208 = reinterpret_tensor(buf181, (1, 512, 1, 1, 196), (100352, 1, 100352, 100352, 512), 0); del buf181  # reuse
        buf209 = reinterpret_tensor(buf180, (1, 512, 1, 1, 196), (100352, 1, 100352, 100352, 512), 0); del buf180  # reuse
        buf210 = reinterpret_tensor(buf179, (1, 512, 1, 1, 196), (100352, 1, 100352, 100352, 512), 0); del buf179  # reuse
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf207, buf208, buf209, buf210, 100352, 256, grid=grid(100352), stream=stream0)
        buf211 = empty_strided((1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), device='cuda', dtype=torch.float32)
        buf212 = empty_strided((1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), device='cuda', dtype=torch.float32)
        buf213 = empty_strided((1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf208, buf209, buf210, buf211, buf212, buf213, 1024, 98, grid=grid(1024), stream=stream0)
        buf214 = reinterpret_tensor(buf199, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf199  # reuse
        buf215 = reinterpret_tensor(buf198, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf198  # reuse
        buf217 = reinterpret_tensor(buf197, (512, ), (1, ), 0); del buf197  # reuse
        buf219 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf218 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf211, buf212, buf213, primals_202, primals_201, buf214, buf215, buf217, buf219, buf218, 512, 2, grid=grid(512), stream=stream0)
        del primals_201
        del primals_202
        buf221 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__self___layer2___0___downsample_0], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_39.run(buf177, primals_43, buf221, grid=torch._inductor.kernel.conv.conv_grid(64, 512, 28, 28, meta5), stream=stream0)
        buf222 = buf210; del buf210  # reuse
        buf223 = buf209; del buf209  # reuse
        buf224 = buf208; del buf208  # reuse
        # Source Nodes: [identity_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf221, buf222, buf223, buf224, 100352, 256, grid=grid(100352), stream=stream0)
        buf225 = buf213; del buf213  # reuse
        buf226 = buf212; del buf212  # reuse
        buf227 = buf211; del buf211  # reuse
        # Source Nodes: [identity_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf222, buf223, buf224, buf225, buf226, buf227, 1024, 98, grid=grid(1024), stream=stream0)
        buf228 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf229 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf231 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf233 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf232 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf225, buf226, buf227, primals_205, primals_204, buf228, buf229, buf231, buf233, buf232, 512, 2, grid=grid(512), stream=stream0)
        del primals_204
        del primals_205
        buf234 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        buf235 = buf234; del buf234  # reuse
        # Source Nodes: [identity_5, identity_6, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_40.run(buf235, buf207, buf214, buf215, primals_41, primals_42, buf221, buf228, buf229, primals_44, primals_45, 25690112, grid=grid(25690112), stream=stream0)
        del primals_42
        del primals_45
        buf236 = empty_strided((50176, 128), (128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_41.run(buf235, primals_46, buf236, grid=torch._inductor.kernel.mm_common.mm_grid(50176, 128, meta3), stream=stream0)
        buf237 = buf196; del buf196  # reuse
        buf238 = buf195; del buf195  # reuse
        buf239 = buf194; del buf194  # reuse
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf236, buf237, buf238, buf239, 50176, 128, grid=grid(50176), stream=stream0)
        buf240 = reinterpret_tensor(buf229, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf229  # reuse
        buf241 = reinterpret_tensor(buf215, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf215  # reuse
        buf242 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf237, buf238, buf239, buf240, buf241, buf242, 512, 98, grid=grid(512), stream=stream0)
        buf243 = buf201; del buf201  # reuse
        buf244 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf246 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf248 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf247 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_33.run(buf240, buf241, buf242, primals_208, primals_207, buf243, buf244, buf246, buf248, buf247, 128, 4, grid=grid(128), stream=stream0)
        del primals_207
        del primals_208
        buf249 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_34.run(buf236, buf243, buf244, primals_47, primals_48, buf249, 6422528, grid=grid(6422528), stream=stream0)
        del primals_48
        buf251 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_43], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_42.run(buf249, buf5, buf251, grid=torch._inductor.kernel.conv.conv_grid(64, 128, 28, 28, meta6), stream=stream0)
        buf252 = buf239; del buf239  # reuse
        buf253 = buf238; del buf238  # reuse
        buf254 = buf237; del buf237  # reuse
        # Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf251, buf252, buf253, buf254, 50176, 128, grid=grid(50176), stream=stream0)
        buf255 = buf242; del buf242  # reuse
        buf256 = buf241; del buf241  # reuse
        buf257 = buf240; del buf240  # reuse
        # Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf252, buf253, buf254, buf255, buf256, buf257, 512, 98, grid=grid(512), stream=stream0)
        buf258 = buf244; del buf244  # reuse
        buf259 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf261 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf263 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf262 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_33.run(buf255, buf256, buf257, primals_211, primals_210, buf258, buf259, buf261, buf263, buf262, 128, 4, grid=grid(128), stream=stream0)
        del primals_210
        del primals_211
        buf264 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_44, out_45], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_34.run(buf251, buf258, buf259, primals_50, primals_51, buf264, 6422528, grid=grid(6422528), stream=stream0)
        del primals_51
        buf265 = empty_strided((50176, 512), (512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_46], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_35.run(buf264, primals_52, buf265, grid=torch._inductor.kernel.mm_common.mm_grid(50176, 512, meta3), stream=stream0)
        buf266 = buf224; del buf224  # reuse
        buf267 = buf223; del buf223  # reuse
        buf268 = buf222; del buf222  # reuse
        # Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf265, buf266, buf267, buf268, 100352, 256, grid=grid(100352), stream=stream0)
        buf269 = buf227; del buf227  # reuse
        buf270 = buf226; del buf226  # reuse
        buf271 = buf225; del buf225  # reuse
        # Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf266, buf267, buf268, buf269, buf270, buf271, 1024, 98, grid=grid(1024), stream=stream0)
        buf272 = reinterpret_tensor(buf257, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf257  # reuse
        buf273 = reinterpret_tensor(buf256, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf256  # reuse
        buf275 = reinterpret_tensor(buf255, (512, ), (1, ), 0); del buf255  # reuse
        buf277 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf276 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf269, buf270, buf271, primals_214, primals_213, buf272, buf273, buf275, buf277, buf276, 512, 2, grid=grid(512), stream=stream0)
        del primals_213
        del primals_214
        buf278 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_7, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_43.run(buf265, buf272, buf273, primals_53, primals_54, buf235, buf278, 25690112, grid=grid(25690112), stream=stream0)
        del primals_54
        buf279 = empty_strided((50176, 128), (128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_50], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_41.run(buf278, primals_55, buf279, grid=torch._inductor.kernel.mm_common.mm_grid(50176, 128, meta3), stream=stream0)
        buf280 = buf254; del buf254  # reuse
        buf281 = buf253; del buf253  # reuse
        buf282 = buf252; del buf252  # reuse
        # Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf279, buf280, buf281, buf282, 50176, 128, grid=grid(50176), stream=stream0)
        buf283 = reinterpret_tensor(buf273, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf273  # reuse
        buf284 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf285 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf280, buf281, buf282, buf283, buf284, buf285, 512, 98, grid=grid(512), stream=stream0)
        buf286 = buf259; del buf259  # reuse
        buf287 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf289 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf291 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf290 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_33.run(buf283, buf284, buf285, primals_217, primals_216, buf286, buf287, buf289, buf291, buf290, 128, 4, grid=grid(128), stream=stream0)
        del primals_216
        del primals_217
        buf292 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_51, out_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_34.run(buf279, buf286, buf287, primals_56, primals_57, buf292, 6422528, grid=grid(6422528), stream=stream0)
        del primals_57
        buf294 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_53], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_42.run(buf292, buf6, buf294, grid=torch._inductor.kernel.conv.conv_grid(64, 128, 28, 28, meta6), stream=stream0)
        buf295 = buf282; del buf282  # reuse
        buf296 = buf281; del buf281  # reuse
        buf297 = buf280; del buf280  # reuse
        # Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf294, buf295, buf296, buf297, 50176, 128, grid=grid(50176), stream=stream0)
        buf298 = buf285; del buf285  # reuse
        buf299 = buf284; del buf284  # reuse
        buf300 = buf283; del buf283  # reuse
        # Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf295, buf296, buf297, buf298, buf299, buf300, 512, 98, grid=grid(512), stream=stream0)
        buf301 = buf287; del buf287  # reuse
        buf302 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf304 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf306 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf305 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_33.run(buf298, buf299, buf300, primals_220, primals_219, buf301, buf302, buf304, buf306, buf305, 128, 4, grid=grid(128), stream=stream0)
        del primals_219
        del primals_220
        buf307 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_54, out_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_34.run(buf294, buf301, buf302, primals_59, primals_60, buf307, 6422528, grid=grid(6422528), stream=stream0)
        del primals_60
        buf308 = empty_strided((50176, 512), (512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_35.run(buf307, primals_61, buf308, grid=torch._inductor.kernel.mm_common.mm_grid(50176, 512, meta3), stream=stream0)
        buf309 = buf268; del buf268  # reuse
        buf310 = buf267; del buf267  # reuse
        buf311 = buf266; del buf266  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf308, buf309, buf310, buf311, 100352, 256, grid=grid(100352), stream=stream0)
        buf312 = buf271; del buf271  # reuse
        buf313 = buf270; del buf270  # reuse
        buf314 = buf269; del buf269  # reuse
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf309, buf310, buf311, buf312, buf313, buf314, 1024, 98, grid=grid(1024), stream=stream0)
        buf315 = reinterpret_tensor(buf300, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf300  # reuse
        buf316 = reinterpret_tensor(buf299, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf299  # reuse
        buf318 = reinterpret_tensor(buf298, (512, ), (1, ), 0); del buf298  # reuse
        buf320 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf319 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf312, buf313, buf314, primals_223, primals_222, buf315, buf316, buf318, buf320, buf319, 512, 2, grid=grid(512), stream=stream0)
        del primals_222
        del primals_223
        buf321 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_8, out_57, out_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_43.run(buf308, buf315, buf316, primals_62, primals_63, buf278, buf321, 25690112, grid=grid(25690112), stream=stream0)
        del primals_63
        buf322 = empty_strided((50176, 128), (128, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_41.run(buf321, primals_64, buf322, grid=torch._inductor.kernel.mm_common.mm_grid(50176, 128, meta3), stream=stream0)
        buf323 = buf297; del buf297  # reuse
        buf324 = buf296; del buf296  # reuse
        buf325 = buf295; del buf295  # reuse
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf322, buf323, buf324, buf325, 50176, 128, grid=grid(50176), stream=stream0)
        buf326 = reinterpret_tensor(buf316, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf316  # reuse
        buf327 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf328 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf323, buf324, buf325, buf326, buf327, buf328, 512, 98, grid=grid(512), stream=stream0)
        buf329 = buf302; del buf302  # reuse
        buf330 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf332 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf334 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf333 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_33.run(buf326, buf327, buf328, primals_226, primals_225, buf329, buf330, buf332, buf334, buf333, 128, 4, grid=grid(128), stream=stream0)
        del primals_225
        del primals_226
        buf335 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61, out_62], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_34.run(buf322, buf329, buf330, primals_65, primals_66, buf335, 6422528, grid=grid(6422528), stream=stream0)
        del primals_66
        buf337 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_63], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_42.run(buf335, buf7, buf337, grid=torch._inductor.kernel.conv.conv_grid(64, 128, 28, 28, meta6), stream=stream0)
        buf338 = buf325; del buf325  # reuse
        buf339 = buf324; del buf324  # reuse
        buf340 = buf323; del buf323  # reuse
        # Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf337, buf338, buf339, buf340, 50176, 128, grid=grid(50176), stream=stream0)
        buf341 = buf328; del buf328  # reuse
        buf342 = buf327; del buf327  # reuse
        buf343 = buf326; del buf326  # reuse
        # Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf338, buf339, buf340, buf341, buf342, buf343, 512, 98, grid=grid(512), stream=stream0)
        buf344 = buf330; del buf330  # reuse
        buf345 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf347 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf349 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        buf348 = empty_strided((128, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_33.run(buf341, buf342, buf343, primals_229, primals_228, buf344, buf345, buf347, buf349, buf348, 128, 4, grid=grid(128), stream=stream0)
        del primals_228
        del primals_229
        buf350 = empty_strided((64, 128, 28, 28), (100352, 1, 3584, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_64, out_65], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_34.run(buf337, buf344, buf345, primals_68, primals_69, buf350, 6422528, grid=grid(6422528), stream=stream0)
        del buf345
        del primals_69
        buf351 = empty_strided((50176, 512), (512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_66], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_35.run(buf350, primals_70, buf351, grid=torch._inductor.kernel.mm_common.mm_grid(50176, 512, meta3), stream=stream0)
        buf352 = buf311; del buf311  # reuse
        buf353 = buf310; del buf310  # reuse
        buf354 = buf309; del buf309  # reuse
        # Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf351, buf352, buf353, buf354, 100352, 256, grid=grid(100352), stream=stream0)
        buf355 = buf314; del buf314  # reuse
        buf356 = buf313; del buf313  # reuse
        buf357 = buf312; del buf312  # reuse
        # Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf352, buf353, buf354, buf355, buf356, buf357, 1024, 98, grid=grid(1024), stream=stream0)
        buf358 = reinterpret_tensor(buf343, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf343  # reuse
        buf359 = reinterpret_tensor(buf342, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf342  # reuse
        buf361 = reinterpret_tensor(buf341, (512, ), (1, ), 0); del buf341  # reuse
        buf363 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf362 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_38.run(buf355, buf356, buf357, primals_232, primals_231, buf358, buf359, buf361, buf363, buf362, 512, 2, grid=grid(512), stream=stream0)
        del primals_231
        del primals_232
        buf364 = empty_strided((64, 512, 28, 28), (401408, 1, 14336, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_9, out_67, out_68], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_43.run(buf351, buf358, buf359, primals_71, primals_72, buf321, buf364, 25690112, grid=grid(25690112), stream=stream0)
        del primals_72
        buf365 = empty_strided((50176, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_70], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_44.run(buf364, primals_73, buf365, grid=torch._inductor.kernel.mm_common.mm_grid(50176, 256, meta3), stream=stream0)
        buf366 = reinterpret_tensor(buf354, (1, 256, 1, 1, 392), (100352, 1, 100352, 100352, 256), 0); del buf354  # reuse
        buf367 = reinterpret_tensor(buf353, (1, 256, 1, 1, 392), (100352, 1, 100352, 100352, 256), 0); del buf353  # reuse
        buf368 = reinterpret_tensor(buf352, (1, 256, 1, 1, 392), (100352, 1, 100352, 100352, 256), 0); del buf352  # reuse
        # Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf365, buf366, buf367, buf368, 100352, 128, grid=grid(100352), stream=stream0)
        buf369 = reinterpret_tensor(buf357, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf357  # reuse
        buf370 = reinterpret_tensor(buf356, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf356  # reuse
        buf371 = reinterpret_tensor(buf355, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf355  # reuse
        # Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_46.run(buf366, buf367, buf368, buf369, buf370, buf371, 1024, 98, grid=grid(1024), stream=stream0)
        buf372 = buf172; del buf172  # reuse
        buf373 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf375 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf377 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf376 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf369, buf370, buf371, primals_235, primals_234, buf372, buf373, buf375, buf377, buf376, 256, 4, grid=grid(256), stream=stream0)
        del primals_234
        del primals_235
        buf378 = empty_strided((64, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf365, buf372, buf373, primals_74, primals_75, buf378, 12845056, grid=grid(12845056), stream=stream0)
        del primals_75
        buf380 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_49.run(buf378, buf8, buf380, grid=torch._inductor.kernel.conv.conv_grid(64, 256, 14, 14, meta7), stream=stream0)
        buf381 = empty_strided((1, 256, 1, 1, 98), (25088, 1, 25088, 25088, 256), device='cuda', dtype=torch.float32)
        buf382 = empty_strided((1, 256, 1, 1, 98), (25088, 1, 25088, 25088, 256), device='cuda', dtype=torch.float32)
        buf383 = empty_strided((1, 256, 1, 1, 98), (25088, 1, 25088, 25088, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf380, buf381, buf382, buf383, 25088, 128, grid=grid(25088), stream=stream0)
        buf384 = buf373; del buf373  # reuse
        buf385 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf387 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf389 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf388 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf381, buf382, buf383, primals_238, primals_237, buf384, buf385, buf387, buf389, buf388, 256, 98, grid=grid(256), stream=stream0)
        del primals_237
        del primals_238
        buf390 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf380, buf384, buf385, primals_77, primals_78, buf390, 3211264, grid=grid(3211264), stream=stream0)
        del primals_78
        buf391 = empty_strided((12544, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_53.run(buf390, primals_79, buf391, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 1024, meta3), stream=stream0)
        buf392 = reinterpret_tensor(buf368, (1, 1024, 1, 1, 98), (100352, 1, 100352, 100352, 1024), 0); del buf368  # reuse
        buf393 = reinterpret_tensor(buf367, (1, 1024, 1, 1, 98), (100352, 1, 100352, 100352, 1024), 0); del buf367  # reuse
        buf394 = reinterpret_tensor(buf366, (1, 1024, 1, 1, 98), (100352, 1, 100352, 100352, 1024), 0); del buf366  # reuse
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf391, buf392, buf393, buf394, 100352, 128, grid=grid(100352), stream=stream0)
        buf395 = reinterpret_tensor(buf371, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf371  # reuse
        buf396 = reinterpret_tensor(buf370, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf370  # reuse
        buf398 = reinterpret_tensor(buf369, (1024, ), (1, ), 0); del buf369  # reuse
        buf400 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf399 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf392, buf393, buf394, primals_241, primals_240, buf395, buf396, buf398, buf400, buf399, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_240
        del primals_241
        buf402 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__self___layer3___0___downsample_0], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_56.run(buf364, primals_82, buf402, grid=torch._inductor.kernel.conv.conv_grid(64, 1024, 14, 14, meta8), stream=stream0)
        buf403 = buf394; del buf394  # reuse
        buf404 = buf393; del buf393  # reuse
        buf405 = buf392; del buf392  # reuse
        # Source Nodes: [identity_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf402, buf403, buf404, buf405, 100352, 128, grid=grid(100352), stream=stream0)
        buf406 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf407 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf409 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf411 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf410 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf403, buf404, buf405, primals_244, primals_243, buf406, buf407, buf409, buf411, buf410, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_243
        del primals_244
        buf412 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        buf413 = buf412; del buf412  # reuse
        # Source Nodes: [identity_10, identity_11, out_77, out_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_57.run(buf413, buf391, buf395, buf396, primals_80, primals_81, buf402, buf406, buf407, primals_83, primals_84, 12845056, grid=grid(12845056), stream=stream0)
        del primals_81
        del primals_84
        buf414 = empty_strided((12544, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_58.run(buf413, primals_85, buf414, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 256, meta9), stream=stream0)
        buf415 = buf383; del buf383  # reuse
        buf416 = buf382; del buf382  # reuse
        buf417 = buf381; del buf381  # reuse
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf414, buf415, buf416, buf417, 25088, 128, grid=grid(25088), stream=stream0)
        buf418 = buf385; del buf385  # reuse
        buf419 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf421 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf423 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf422 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf415, buf416, buf417, primals_247, primals_246, buf418, buf419, buf421, buf423, buf422, 256, 98, grid=grid(256), stream=stream0)
        del primals_246
        del primals_247
        buf424 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf414, buf418, buf419, primals_86, primals_87, buf424, 3211264, grid=grid(3211264), stream=stream0)
        del primals_87
        buf426 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_83], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_59.run(buf424, buf9, buf426, grid=torch._inductor.kernel.conv.conv_grid(64, 256, 14, 14, meta10), stream=stream0)
        buf427 = buf417; del buf417  # reuse
        buf428 = buf416; del buf416  # reuse
        buf429 = buf415; del buf415  # reuse
        # Source Nodes: [out_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf426, buf427, buf428, buf429, 25088, 128, grid=grid(25088), stream=stream0)
        buf430 = buf419; del buf419  # reuse
        buf431 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf433 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf435 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf434 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf427, buf428, buf429, primals_250, primals_249, buf430, buf431, buf433, buf435, buf434, 256, 98, grid=grid(256), stream=stream0)
        del primals_249
        del primals_250
        buf436 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_84, out_85], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf426, buf430, buf431, primals_89, primals_90, buf436, 3211264, grid=grid(3211264), stream=stream0)
        del primals_90
        buf437 = empty_strided((12544, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_86], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_53.run(buf436, primals_91, buf437, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 1024, meta3), stream=stream0)
        buf438 = buf405; del buf405  # reuse
        buf439 = buf404; del buf404  # reuse
        buf440 = buf403; del buf403  # reuse
        # Source Nodes: [out_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf437, buf438, buf439, buf440, 100352, 128, grid=grid(100352), stream=stream0)
        buf441 = buf407; del buf407  # reuse
        buf442 = buf396; del buf396  # reuse
        buf444 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf446 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf445 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf438, buf439, buf440, primals_253, primals_252, buf441, buf442, buf444, buf446, buf445, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_252
        del primals_253
        buf447 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_12, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf437, buf441, buf442, primals_92, primals_93, buf413, buf447, 12845056, grid=grid(12845056), stream=stream0)
        del primals_93
        buf448 = empty_strided((12544, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_90], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_58.run(buf447, primals_94, buf448, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 256, meta9), stream=stream0)
        buf449 = buf429; del buf429  # reuse
        buf450 = buf428; del buf428  # reuse
        buf451 = buf427; del buf427  # reuse
        # Source Nodes: [out_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf448, buf449, buf450, buf451, 25088, 128, grid=grid(25088), stream=stream0)
        buf452 = buf431; del buf431  # reuse
        buf453 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf455 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf457 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf456 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf449, buf450, buf451, primals_256, primals_255, buf452, buf453, buf455, buf457, buf456, 256, 98, grid=grid(256), stream=stream0)
        del primals_255
        del primals_256
        buf458 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_91, out_92], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf448, buf452, buf453, primals_95, primals_96, buf458, 3211264, grid=grid(3211264), stream=stream0)
        del primals_96
        buf460 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_93], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_59.run(buf458, buf10, buf460, grid=torch._inductor.kernel.conv.conv_grid(64, 256, 14, 14, meta10), stream=stream0)
        buf461 = buf451; del buf451  # reuse
        buf462 = buf450; del buf450  # reuse
        buf463 = buf449; del buf449  # reuse
        # Source Nodes: [out_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf460, buf461, buf462, buf463, 25088, 128, grid=grid(25088), stream=stream0)
        buf464 = buf453; del buf453  # reuse
        buf465 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf467 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf469 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf468 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf461, buf462, buf463, primals_259, primals_258, buf464, buf465, buf467, buf469, buf468, 256, 98, grid=grid(256), stream=stream0)
        del primals_258
        del primals_259
        buf470 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_94, out_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf460, buf464, buf465, primals_98, primals_99, buf470, 3211264, grid=grid(3211264), stream=stream0)
        del primals_99
        buf471 = empty_strided((12544, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_53.run(buf470, primals_100, buf471, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 1024, meta3), stream=stream0)
        buf472 = buf440; del buf440  # reuse
        buf473 = buf439; del buf439  # reuse
        buf474 = buf438; del buf438  # reuse
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf471, buf472, buf473, buf474, 100352, 128, grid=grid(100352), stream=stream0)
        buf475 = buf442; del buf442  # reuse
        buf476 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf478 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf480 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf479 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf472, buf473, buf474, primals_262, primals_261, buf475, buf476, buf478, buf480, buf479, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_261
        del primals_262
        buf481 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_13, out_97, out_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf471, buf475, buf476, primals_101, primals_102, buf447, buf481, 12845056, grid=grid(12845056), stream=stream0)
        del primals_102
        buf482 = empty_strided((12544, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_58.run(buf481, primals_103, buf482, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 256, meta9), stream=stream0)
        buf483 = buf463; del buf463  # reuse
        buf484 = buf462; del buf462  # reuse
        buf485 = buf461; del buf461  # reuse
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf482, buf483, buf484, buf485, 25088, 128, grid=grid(25088), stream=stream0)
        buf486 = buf465; del buf465  # reuse
        buf487 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf489 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf491 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf490 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf483, buf484, buf485, primals_265, primals_264, buf486, buf487, buf489, buf491, buf490, 256, 98, grid=grid(256), stream=stream0)
        del primals_264
        del primals_265
        buf492 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf482, buf486, buf487, primals_104, primals_105, buf492, 3211264, grid=grid(3211264), stream=stream0)
        del primals_105
        buf494 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_103], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_59.run(buf492, buf11, buf494, grid=torch._inductor.kernel.conv.conv_grid(64, 256, 14, 14, meta10), stream=stream0)
        buf495 = buf485; del buf485  # reuse
        buf496 = buf484; del buf484  # reuse
        buf497 = buf483; del buf483  # reuse
        # Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf494, buf495, buf496, buf497, 25088, 128, grid=grid(25088), stream=stream0)
        buf498 = buf487; del buf487  # reuse
        buf499 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf501 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf503 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf502 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf495, buf496, buf497, primals_268, primals_267, buf498, buf499, buf501, buf503, buf502, 256, 98, grid=grid(256), stream=stream0)
        del primals_267
        del primals_268
        buf504 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf494, buf498, buf499, primals_107, primals_108, buf504, 3211264, grid=grid(3211264), stream=stream0)
        del primals_108
        buf505 = empty_strided((12544, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_106], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_53.run(buf504, primals_109, buf505, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 1024, meta3), stream=stream0)
        buf506 = buf474; del buf474  # reuse
        buf507 = buf473; del buf473  # reuse
        buf508 = buf472; del buf472  # reuse
        # Source Nodes: [out_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf505, buf506, buf507, buf508, 100352, 128, grid=grid(100352), stream=stream0)
        buf509 = buf476; del buf476  # reuse
        buf510 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf512 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf514 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf513 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf506, buf507, buf508, primals_271, primals_270, buf509, buf510, buf512, buf514, buf513, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_270
        del primals_271
        buf515 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_14, out_107, out_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf505, buf509, buf510, primals_110, primals_111, buf481, buf515, 12845056, grid=grid(12845056), stream=stream0)
        del primals_111
        buf516 = empty_strided((12544, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_110], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_58.run(buf515, primals_112, buf516, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 256, meta9), stream=stream0)
        buf517 = buf497; del buf497  # reuse
        buf518 = buf496; del buf496  # reuse
        buf519 = buf495; del buf495  # reuse
        # Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf516, buf517, buf518, buf519, 25088, 128, grid=grid(25088), stream=stream0)
        buf520 = buf499; del buf499  # reuse
        buf521 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf523 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf525 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf524 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf517, buf518, buf519, primals_274, primals_273, buf520, buf521, buf523, buf525, buf524, 256, 98, grid=grid(256), stream=stream0)
        del primals_273
        del primals_274
        buf526 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf516, buf520, buf521, primals_113, primals_114, buf526, 3211264, grid=grid(3211264), stream=stream0)
        del primals_114
        buf528 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_113], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_59.run(buf526, buf12, buf528, grid=torch._inductor.kernel.conv.conv_grid(64, 256, 14, 14, meta10), stream=stream0)
        buf529 = buf519; del buf519  # reuse
        buf530 = buf518; del buf518  # reuse
        buf531 = buf517; del buf517  # reuse
        # Source Nodes: [out_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf528, buf529, buf530, buf531, 25088, 128, grid=grid(25088), stream=stream0)
        buf532 = buf521; del buf521  # reuse
        buf533 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf535 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf537 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf536 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf529, buf530, buf531, primals_277, primals_276, buf532, buf533, buf535, buf537, buf536, 256, 98, grid=grid(256), stream=stream0)
        del primals_276
        del primals_277
        buf538 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_114, out_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf528, buf532, buf533, primals_116, primals_117, buf538, 3211264, grid=grid(3211264), stream=stream0)
        del primals_117
        buf539 = empty_strided((12544, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_53.run(buf538, primals_118, buf539, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 1024, meta3), stream=stream0)
        buf540 = buf508; del buf508  # reuse
        buf541 = buf507; del buf507  # reuse
        buf542 = buf506; del buf506  # reuse
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf539, buf540, buf541, buf542, 100352, 128, grid=grid(100352), stream=stream0)
        buf543 = buf510; del buf510  # reuse
        buf544 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf546 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf548 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf547 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf540, buf541, buf542, primals_280, primals_279, buf543, buf544, buf546, buf548, buf547, 1024, 98, grid=grid(1024), stream=stream0)
        del primals_279
        del primals_280
        buf549 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_15, out_117, out_118], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf539, buf543, buf544, primals_119, primals_120, buf515, buf549, 12845056, grid=grid(12845056), stream=stream0)
        del primals_120
        buf550 = empty_strided((12544, 256), (256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_58.run(buf549, primals_121, buf550, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 256, meta9), stream=stream0)
        buf551 = buf531; del buf531  # reuse
        buf552 = buf530; del buf530  # reuse
        buf553 = buf529; del buf529  # reuse
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf550, buf551, buf552, buf553, 25088, 128, grid=grid(25088), stream=stream0)
        buf554 = buf533; del buf533  # reuse
        buf555 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf557 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf559 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf558 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf551, buf552, buf553, primals_283, primals_282, buf554, buf555, buf557, buf559, buf558, 256, 98, grid=grid(256), stream=stream0)
        del primals_282
        del primals_283
        buf560 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf550, buf554, buf555, primals_122, primals_123, buf560, 3211264, grid=grid(3211264), stream=stream0)
        del primals_123
        buf562 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_123], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_59.run(buf560, buf13, buf562, grid=torch._inductor.kernel.conv.conv_grid(64, 256, 14, 14, meta10), stream=stream0)
        buf563 = buf553; del buf553  # reuse
        buf564 = buf552; del buf552  # reuse
        buf565 = buf551; del buf551  # reuse
        # Source Nodes: [out_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf562, buf563, buf564, buf565, 25088, 128, grid=grid(25088), stream=stream0)
        buf566 = buf555; del buf555  # reuse
        buf567 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf569 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf571 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        buf570 = empty_strided((256, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_124], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf563, buf564, buf565, primals_286, primals_285, buf566, buf567, buf569, buf571, buf570, 256, 98, grid=grid(256), stream=stream0)
        del buf563
        del buf564
        del buf565
        del primals_285
        del primals_286
        buf572 = empty_strided((64, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_124, out_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf562, buf566, buf567, primals_125, primals_126, buf572, 3211264, grid=grid(3211264), stream=stream0)
        del buf567
        del primals_126
        buf573 = empty_strided((12544, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_126], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_53.run(buf572, primals_127, buf573, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 1024, meta3), stream=stream0)
        buf574 = buf542; del buf542  # reuse
        buf575 = buf541; del buf541  # reuse
        buf576 = buf540; del buf540  # reuse
        # Source Nodes: [out_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf573, buf574, buf575, buf576, 100352, 128, grid=grid(100352), stream=stream0)
        buf577 = buf544; del buf544  # reuse
        buf578 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf580 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf582 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        buf581 = empty_strided((1024, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf574, buf575, buf576, primals_289, primals_288, buf577, buf578, buf580, buf582, buf581, 1024, 98, grid=grid(1024), stream=stream0)
        del buf574
        del buf575
        del buf576
        del primals_288
        del primals_289
        buf583 = empty_strided((64, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_16, out_127, out_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf573, buf577, buf578, primals_128, primals_129, buf549, buf583, 12845056, grid=grid(12845056), stream=stream0)
        del buf578
        del primals_129
        buf584 = empty_strided((12544, 512), (512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_130], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_61.run(buf583, primals_130, buf584, grid=torch._inductor.kernel.mm_common.mm_grid(12544, 512, meta3), stream=stream0)
        buf585 = reinterpret_tensor(buf340, (1, 512, 1, 1, 98), (50176, 1, 50176, 50176, 512), 0); del buf340  # reuse
        buf586 = reinterpret_tensor(buf339, (1, 512, 1, 1, 98), (50176, 1, 50176, 50176, 512), 0); del buf339  # reuse
        buf587 = reinterpret_tensor(buf338, (1, 512, 1, 1, 98), (50176, 1, 50176, 50176, 512), 0); del buf338  # reuse
        # Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_62.run(buf584, buf585, buf586, buf587, 50176, 128, grid=grid(50176), stream=stream0)
        buf588 = buf359; del buf359  # reuse
        buf589 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf591 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf593 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf592 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_63.run(buf585, buf586, buf587, primals_292, primals_291, buf588, buf589, buf591, buf593, buf592, 512, 98, grid=grid(512), stream=stream0)
        del buf585
        del buf586
        del buf587
        del primals_291
        del primals_292
        buf594 = empty_strided((64, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_64.run(buf584, buf588, buf589, primals_131, primals_132, buf594, 6422528, grid=grid(6422528), stream=stream0)
        del primals_132
        buf596 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_133], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_65.run(buf594, buf14, buf596, grid=torch._inductor.kernel.conv.conv_grid(64, 512, 7, 7, meta7), stream=stream0)
        buf597 = empty_strided((1, 512, 1, 1, 25), (12800, 1, 12800, 12800, 512), device='cuda', dtype=torch.float32)
        buf598 = empty_strided((1, 512, 1, 1, 25), (12800, 1, 12800, 12800, 512), device='cuda', dtype=torch.float32)
        buf599 = empty_strided((1, 512, 1, 1, 25), (12800, 1, 12800, 12800, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf596, buf597, buf598, buf599, 12800, 126, grid=grid(12800), stream=stream0)
        buf600 = buf589; del buf589  # reuse
        buf601 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf603 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf605 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf604 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf597, buf598, buf599, primals_295, primals_294, buf600, buf601, buf603, buf605, buf604, 512, 25, grid=grid(512), stream=stream0)
        del primals_294
        del primals_295
        buf606 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_68.run(buf596, buf600, buf601, primals_134, primals_135, buf606, 1605632, grid=grid(1605632), stream=stream0)
        del primals_135
        buf607 = empty_strided((3136, 2048), (2048, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_136], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_69.run(buf606, primals_136, buf607, grid=torch._inductor.kernel.mm_common.mm_grid(3136, 2048, meta3), stream=stream0)
        buf608 = empty_strided((1, 2048, 1, 1, 25), (51200, 1, 51200, 51200, 2048), device='cuda', dtype=torch.float32)
        buf609 = empty_strided((1, 2048, 1, 1, 25), (51200, 1, 51200, 51200, 2048), device='cuda', dtype=torch.float32)
        buf610 = empty_strided((1, 2048, 1, 1, 25), (51200, 1, 51200, 51200, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf607, buf608, buf609, buf610, 51200, 126, grid=grid(51200), stream=stream0)
        buf611 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf612 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf614 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf616 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf615 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_71.run(buf608, buf609, buf610, primals_298, primals_297, buf611, buf612, buf614, buf616, buf615, 2048, 25, grid=grid(2048), stream=stream0)
        del primals_297
        del primals_298
        buf618 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__self___layer4___0___downsample_0], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_72.run(buf583, primals_139, buf618, grid=torch._inductor.kernel.conv.conv_grid(64, 2048, 7, 7, meta11), stream=stream0)
        buf619 = buf610; del buf610  # reuse
        buf620 = buf609; del buf609  # reuse
        buf621 = buf608; del buf608  # reuse
        # Source Nodes: [identity_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf618, buf619, buf620, buf621, 51200, 126, grid=grid(51200), stream=stream0)
        buf622 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf623 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf625 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf627 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf626 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_71.run(buf619, buf620, buf621, primals_301, primals_300, buf622, buf623, buf625, buf627, buf626, 2048, 25, grid=grid(2048), stream=stream0)
        del primals_300
        del primals_301
        buf628 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        buf629 = buf628; del buf628  # reuse
        # Source Nodes: [identity_17, identity_18, out_137, out_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_73.run(buf629, buf607, buf611, buf612, primals_137, primals_138, buf618, buf622, buf623, primals_140, primals_141, 6422528, grid=grid(6422528), stream=stream0)
        del primals_138
        del primals_141
        buf630 = empty_strided((3136, 512), (512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_74.run(buf629, primals_142, buf630, grid=torch._inductor.kernel.mm_common.mm_grid(3136, 512, meta9), stream=stream0)
        buf631 = buf599; del buf599  # reuse
        buf632 = buf598; del buf598  # reuse
        buf633 = buf597; del buf597  # reuse
        # Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf630, buf631, buf632, buf633, 12800, 126, grid=grid(12800), stream=stream0)
        buf634 = buf601; del buf601  # reuse
        buf635 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf637 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf639 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf638 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf631, buf632, buf633, primals_304, primals_303, buf634, buf635, buf637, buf639, buf638, 512, 25, grid=grid(512), stream=stream0)
        del primals_303
        del primals_304
        buf640 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_141, out_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_68.run(buf630, buf634, buf635, primals_143, primals_144, buf640, 1605632, grid=grid(1605632), stream=stream0)
        del primals_144
        buf642 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_143], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_75.run(buf640, buf15, buf642, grid=torch._inductor.kernel.conv.conv_grid(64, 512, 7, 7, meta10), stream=stream0)
        buf643 = buf633; del buf633  # reuse
        buf644 = buf632; del buf632  # reuse
        buf645 = buf631; del buf631  # reuse
        # Source Nodes: [out_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf642, buf643, buf644, buf645, 12800, 126, grid=grid(12800), stream=stream0)
        buf646 = buf635; del buf635  # reuse
        buf647 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf649 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf651 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf650 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_144], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf643, buf644, buf645, primals_307, primals_306, buf646, buf647, buf649, buf651, buf650, 512, 25, grid=grid(512), stream=stream0)
        del primals_306
        del primals_307
        buf652 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_144, out_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_68.run(buf642, buf646, buf647, primals_146, primals_147, buf652, 1605632, grid=grid(1605632), stream=stream0)
        del primals_147
        buf653 = empty_strided((3136, 2048), (2048, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_146], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_69.run(buf652, primals_148, buf653, grid=torch._inductor.kernel.mm_common.mm_grid(3136, 2048, meta3), stream=stream0)
        buf654 = buf621; del buf621  # reuse
        buf655 = buf620; del buf620  # reuse
        buf656 = buf619; del buf619  # reuse
        # Source Nodes: [out_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf653, buf654, buf655, buf656, 51200, 126, grid=grid(51200), stream=stream0)
        buf657 = buf623; del buf623  # reuse
        buf658 = buf612; del buf612  # reuse
        buf660 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf662 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf661 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_71.run(buf654, buf655, buf656, primals_310, primals_309, buf657, buf658, buf660, buf662, buf661, 2048, 25, grid=grid(2048), stream=stream0)
        del primals_309
        del primals_310
        buf663 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_19, out_147, out_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_76.run(buf653, buf657, buf658, primals_149, primals_150, buf629, buf663, 6422528, grid=grid(6422528), stream=stream0)
        del primals_150
        buf664 = empty_strided((3136, 512), (512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_150], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_74.run(buf663, primals_151, buf664, grid=torch._inductor.kernel.mm_common.mm_grid(3136, 512, meta9), stream=stream0)
        buf665 = buf645; del buf645  # reuse
        buf666 = buf644; del buf644  # reuse
        buf667 = buf643; del buf643  # reuse
        # Source Nodes: [out_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf664, buf665, buf666, buf667, 12800, 126, grid=grid(12800), stream=stream0)
        buf668 = buf647; del buf647  # reuse
        buf669 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf671 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf673 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf672 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf665, buf666, buf667, primals_313, primals_312, buf668, buf669, buf671, buf673, buf672, 512, 25, grid=grid(512), stream=stream0)
        del primals_312
        del primals_313
        buf674 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_151, out_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_68.run(buf664, buf668, buf669, primals_152, primals_153, buf674, 1605632, grid=grid(1605632), stream=stream0)
        del primals_153
        buf676 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_153], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_75.run(buf674, buf16, buf676, grid=torch._inductor.kernel.conv.conv_grid(64, 512, 7, 7, meta10), stream=stream0)
        buf677 = buf667; del buf667  # reuse
        buf678 = buf666; del buf666  # reuse
        buf679 = buf665; del buf665  # reuse
        # Source Nodes: [out_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_66.run(buf676, buf677, buf678, buf679, 12800, 126, grid=grid(12800), stream=stream0)
        buf680 = buf669; del buf669  # reuse
        buf681 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf683 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf685 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        buf684 = empty_strided((512, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_67.run(buf677, buf678, buf679, primals_316, primals_315, buf680, buf681, buf683, buf685, buf684, 512, 25, grid=grid(512), stream=stream0)
        del buf677
        del buf678
        del buf679
        del primals_315
        del primals_316
        buf686 = empty_strided((64, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_154, out_155], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_68.run(buf676, buf680, buf681, primals_155, primals_156, buf686, 1605632, grid=grid(1605632), stream=stream0)
        del buf681
        del primals_156
        buf687 = empty_strided((3136, 2048), (2048, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_156], Original ATen: [aten.convolution]
        triton_tem_fused_convolution_69.run(buf686, primals_157, buf687, grid=torch._inductor.kernel.mm_common.mm_grid(3136, 2048, meta3), stream=stream0)
        buf688 = buf656; del buf656  # reuse
        buf689 = buf655; del buf655  # reuse
        buf690 = buf654; del buf654  # reuse
        # Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_70.run(buf687, buf688, buf689, buf690, 51200, 126, grid=grid(51200), stream=stream0)
        buf691 = buf658; del buf658  # reuse
        buf692 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf694 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf696 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        buf695 = empty_strided((2048, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_71.run(buf688, buf689, buf690, primals_319, primals_318, buf691, buf692, buf694, buf696, buf695, 2048, 25, grid=grid(2048), stream=stream0)
        del buf688
        del buf689
        del buf690
        del primals_318
        del primals_319
        buf697 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.float32)
        buf701 = empty_strided((64, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_157, out_158, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_77.run(buf687, buf691, buf692, primals_158, primals_159, buf663, buf697, buf701, 6422528, grid=grid(6422528), stream=stream0)
        del buf692
        del primals_159
        buf698 = empty_strided((64, 2048, 1, 1), (2048, 1, 131072, 131072), device='cuda', dtype=torch.float32)
        buf699 = reinterpret_tensor(buf698, (64, 2048), (2048, 1), 0); del buf698  # reuse
        # Source Nodes: [x_8, x_9], Original ATen: [aten.mean, aten.view]
        triton_per_fused_mean_view_78.run(buf699, buf697, 131072, 49, grid=grid(131072), stream=stream0)
        del buf697
        buf700 = empty_strided((64, 1000), (1000, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_79.run(primals_161, buf699, primals_160, buf700, grid=torch._inductor.kernel.mm_common.mm_grid(64, 1000, meta12), stream=stream0)
        del primals_161
        buf702 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_1], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_164, buf702, 1, grid=grid(1), stream=stream0)
        del primals_164
        buf703 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_1], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_167, buf703, 1, grid=grid(1), stream=stream0)
        del primals_167
        buf704 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_4], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_170, buf704, 1, grid=grid(1), stream=stream0)
        del primals_170
        buf705 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_7], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_173, buf705, 1, grid=grid(1), stream=stream0)
        del primals_173
        buf706 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [identity_1], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_176, buf706, 1, grid=grid(1), stream=stream0)
        del primals_176
        buf707 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_11], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_179, buf707, 1, grid=grid(1), stream=stream0)
        del primals_179
        buf708 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_14], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_182, buf708, 1, grid=grid(1), stream=stream0)
        del primals_182
        buf709 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_17], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_185, buf709, 1, grid=grid(1), stream=stream0)
        del primals_185
        buf710 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_21], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_188, buf710, 1, grid=grid(1), stream=stream0)
        del primals_188
        buf711 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_24], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_191, buf711, 1, grid=grid(1), stream=stream0)
        del primals_191
        buf712 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_27], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_194, buf712, 1, grid=grid(1), stream=stream0)
        del primals_194
        buf713 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_31], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_197, buf713, 1, grid=grid(1), stream=stream0)
        del primals_197
        buf714 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_34], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_200, buf714, 1, grid=grid(1), stream=stream0)
        del primals_200
        buf715 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_37], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_203, buf715, 1, grid=grid(1), stream=stream0)
        del primals_203
        buf716 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [identity_5], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_206, buf716, 1, grid=grid(1), stream=stream0)
        del primals_206
        buf717 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_41], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_209, buf717, 1, grid=grid(1), stream=stream0)
        del primals_209
        buf718 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_44], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_212, buf718, 1, grid=grid(1), stream=stream0)
        del primals_212
        buf719 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_47], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_215, buf719, 1, grid=grid(1), stream=stream0)
        del primals_215
        buf720 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_51], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_218, buf720, 1, grid=grid(1), stream=stream0)
        del primals_218
        buf721 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_54], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_221, buf721, 1, grid=grid(1), stream=stream0)
        del primals_221
        buf722 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_57], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_224, buf722, 1, grid=grid(1), stream=stream0)
        del primals_224
        buf723 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_61], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_227, buf723, 1, grid=grid(1), stream=stream0)
        del primals_227
        buf724 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_64], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_230, buf724, 1, grid=grid(1), stream=stream0)
        del primals_230
        buf725 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_67], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_233, buf725, 1, grid=grid(1), stream=stream0)
        del primals_233
        buf726 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_71], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_236, buf726, 1, grid=grid(1), stream=stream0)
        del primals_236
        buf727 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_74], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_239, buf727, 1, grid=grid(1), stream=stream0)
        del primals_239
        buf728 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_77], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_242, buf728, 1, grid=grid(1), stream=stream0)
        del primals_242
        buf729 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [identity_10], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_245, buf729, 1, grid=grid(1), stream=stream0)
        del primals_245
        buf730 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_81], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_248, buf730, 1, grid=grid(1), stream=stream0)
        del primals_248
        buf731 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_84], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_251, buf731, 1, grid=grid(1), stream=stream0)
        del primals_251
        buf732 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_87], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_254, buf732, 1, grid=grid(1), stream=stream0)
        del primals_254
        buf733 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_91], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_257, buf733, 1, grid=grid(1), stream=stream0)
        del primals_257
        buf734 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_94], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_260, buf734, 1, grid=grid(1), stream=stream0)
        del primals_260
        buf735 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_97], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_263, buf735, 1, grid=grid(1), stream=stream0)
        del primals_263
        buf736 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_101], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_266, buf736, 1, grid=grid(1), stream=stream0)
        del primals_266
        buf737 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_104], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_269, buf737, 1, grid=grid(1), stream=stream0)
        del primals_269
        buf738 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_107], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_272, buf738, 1, grid=grid(1), stream=stream0)
        del primals_272
        buf739 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_111], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_275, buf739, 1, grid=grid(1), stream=stream0)
        del primals_275
        buf740 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_114], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_278, buf740, 1, grid=grid(1), stream=stream0)
        del primals_278
        buf741 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_117], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_281, buf741, 1, grid=grid(1), stream=stream0)
        del primals_281
        buf742 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_121], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_284, buf742, 1, grid=grid(1), stream=stream0)
        del primals_284
        buf743 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_124], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_287, buf743, 1, grid=grid(1), stream=stream0)
        del primals_287
        buf744 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_127], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_290, buf744, 1, grid=grid(1), stream=stream0)
        del primals_290
        buf745 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_131], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_293, buf745, 1, grid=grid(1), stream=stream0)
        del primals_293
        buf746 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_134], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_296, buf746, 1, grid=grid(1), stream=stream0)
        del primals_296
        buf747 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_137], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_299, buf747, 1, grid=grid(1), stream=stream0)
        del primals_299
        buf748 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [identity_17], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_302, buf748, 1, grid=grid(1), stream=stream0)
        del primals_302
        buf749 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_141], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_305, buf749, 1, grid=grid(1), stream=stream0)
        del primals_305
        buf750 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_144], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_308, buf750, 1, grid=grid(1), stream=stream0)
        del primals_308
        buf751 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_147], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_311, buf751, 1, grid=grid(1), stream=stream0)
        del primals_311
        buf752 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_151], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_314, buf752, 1, grid=grid(1), stream=stream0)
        del primals_314
        buf753 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_154], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_317, buf753, 1, grid=grid(1), stream=stream0)
        del primals_317
        buf754 = empty_strided((), (), device='cuda', dtype=torch.int64)
        # Source Nodes: [out_157], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(primals_320, buf754, 1, grid=grid(1), stream=stream0)
        del primals_320
        return (buf30, buf31, buf702, buf46, buf47, buf703, buf61, buf62, buf704, buf75, buf76, buf705, buf88, buf89, buf706, buf103, buf104, buf707, buf118, buf119, buf708, buf132, buf133, buf709, buf146, buf147, buf710, buf161, buf162, buf711, buf175, buf176, buf712, buf189, buf190, buf713, buf204, buf205, buf714, buf218, buf219, buf715, buf232, buf233, buf716, buf247, buf248, buf717, buf262, buf263, buf718, buf276, buf277, buf719, buf290, buf291, buf720, buf305, buf306, buf721, buf319, buf320, buf722, buf333, buf334, buf723, buf348, buf349, buf724, buf362, buf363, buf725, buf376, buf377, buf726, buf388, buf389, buf727, buf399, buf400, buf728, buf410, buf411, buf729, buf422, buf423, buf730, buf434, buf435, buf731, buf445, buf446, buf732, buf456, buf457, buf733, buf468, buf469, buf734, buf479, buf480, buf735, buf490, buf491, buf736, buf502, buf503, buf737, buf513, buf514, buf738, buf524, buf525, buf739, buf536, buf537, buf740, buf547, buf548, buf741, buf558, buf559, buf742, buf570, buf571, buf743, buf581, buf582, buf744, buf592, buf593, buf745, buf604, buf605, buf746, buf615, buf616, buf747, buf626, buf627, buf748, buf638, buf639, buf749, buf650, buf651, buf750, buf661, buf662, buf751, buf672, buf673, buf752, buf684, buf685, buf753, buf695, buf696, buf754, buf700, buf0, primals_2, primals_4, primals_5, buf1, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, buf2, primals_20, primals_22, primals_23, primals_25, primals_26, buf3, primals_29, primals_31, primals_32, primals_34, primals_35, buf4, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, buf5, primals_50, primals_52, primals_53, primals_55, primals_56, buf6, primals_59, primals_61, primals_62, primals_64, primals_65, buf7, primals_68, primals_70, primals_71, primals_73, primals_74, buf8, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, buf9, primals_89, primals_91, primals_92, primals_94, primals_95, buf10, primals_98, primals_100, primals_101, primals_103, primals_104, buf11, primals_107, primals_109, primals_110, primals_112, primals_113, buf12, primals_116, primals_118, primals_119, primals_121, primals_122, buf13, primals_125, primals_127, primals_128, primals_130, primals_131, buf14, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, buf15, primals_146, primals_148, primals_149, primals_151, primals_152, buf16, primals_155, primals_157, primals_158, buf17, buf19, buf29, buf32, buf33, buf34, reinterpret_tensor(buf35, (64, 64, 56, 56), (200704, 1, 3584, 64), 0), buf45, buf48, buf50, buf60, buf63, reinterpret_tensor(buf64, (64, 256, 56, 56), (802816, 1, 14336, 256), 0), buf74, reinterpret_tensor(buf77, (64, 256, 56, 56), (802816, 1, 14336, 256), 0), buf87, buf91, reinterpret_tensor(buf92, (64, 64, 56, 56), (200704, 1, 3584, 64), 0), buf102, buf105, buf107, buf117, buf120, reinterpret_tensor(buf121, (64, 256, 56, 56), (802816, 1, 14336, 256), 0), buf131, buf134, reinterpret_tensor(buf135, (64, 64, 56, 56), (200704, 1, 3584, 64), 0), buf145, buf148, buf150, buf160, buf163, reinterpret_tensor(buf164, (64, 256, 56, 56), (802816, 1, 14336, 256), 0), buf174, buf177, reinterpret_tensor(buf178, (64, 128, 56, 56), (401408, 1, 7168, 128), 0), buf188, buf191, buf193, buf203, buf206, reinterpret_tensor(buf207, (64, 512, 28, 28), (401408, 1, 14336, 512), 0), buf217, buf221, buf231, buf235, reinterpret_tensor(buf236, (64, 128, 28, 28), (100352, 1, 3584, 128), 0), buf246, buf249, buf251, buf261, buf264, reinterpret_tensor(buf265, (64, 512, 28, 28), (401408, 1, 14336, 512), 0), buf275, buf278, reinterpret_tensor(buf279, (64, 128, 28, 28), (100352, 1, 3584, 128), 0), buf289, buf292, buf294, buf304, buf307, reinterpret_tensor(buf308, (64, 512, 28, 28), (401408, 1, 14336, 512), 0), buf318, buf321, reinterpret_tensor(buf322, (64, 128, 28, 28), (100352, 1, 3584, 128), 0), buf332, buf335, buf337, buf347, buf350, reinterpret_tensor(buf351, (64, 512, 28, 28), (401408, 1, 14336, 512), 0), buf361, buf364, reinterpret_tensor(buf365, (64, 256, 28, 28), (200704, 1, 7168, 256), 0), buf375, buf378, buf380, buf387, buf390, reinterpret_tensor(buf391, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0), buf398, buf402, buf409, buf413, reinterpret_tensor(buf414, (64, 256, 14, 14), (50176, 1, 3584, 256), 0), buf421, buf424, buf426, buf433, buf436, reinterpret_tensor(buf437, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0), buf444, buf447, reinterpret_tensor(buf448, (64, 256, 14, 14), (50176, 1, 3584, 256), 0), buf455, buf458, buf460, buf467, buf470, reinterpret_tensor(buf471, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0), buf478, buf481, reinterpret_tensor(buf482, (64, 256, 14, 14), (50176, 1, 3584, 256), 0), buf489, buf492, buf494, buf501, buf504, reinterpret_tensor(buf505, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0), buf512, buf515, reinterpret_tensor(buf516, (64, 256, 14, 14), (50176, 1, 3584, 256), 0), buf523, buf526, buf528, buf535, buf538, reinterpret_tensor(buf539, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0), buf546, buf549, reinterpret_tensor(buf550, (64, 256, 14, 14), (50176, 1, 3584, 256), 0), buf557, buf560, buf562, buf569, buf572, reinterpret_tensor(buf573, (64, 1024, 14, 14), (200704, 1, 14336, 1024), 0), buf580, buf583, reinterpret_tensor(buf584, (64, 512, 14, 14), (100352, 1, 7168, 512), 0), buf591, buf594, buf596, buf603, buf606, reinterpret_tensor(buf607, (64, 2048, 7, 7), (100352, 1, 14336, 2048), 0), buf614, buf618, buf625, buf629, reinterpret_tensor(buf630, (64, 512, 7, 7), (25088, 1, 3584, 512), 0), buf637, buf640, buf642, buf649, buf652, reinterpret_tensor(buf653, (64, 2048, 7, 7), (100352, 1, 14336, 2048), 0), buf660, buf663, reinterpret_tensor(buf664, (64, 512, 7, 7), (25088, 1, 3584, 512), 0), buf671, buf674, buf676, buf683, buf686, reinterpret_tensor(buf687, (64, 2048, 7, 7), (100352, 1, 14336, 2048), 0), buf694, buf699, reinterpret_tensor(primals_160, (1000, 2048), (2048, 1), 0), buf701, reinterpret_tensor(buf691, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf680, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf668, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf657, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf646, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf634, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf622, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf611, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf600, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf588, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf577, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf566, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf554, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf543, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf532, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf520, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf509, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf498, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf486, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf475, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf464, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf452, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf441, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf430, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf418, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf406, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf395, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf384, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf372, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf358, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf344, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf329, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf315, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf301, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf286, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf272, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf258, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf243, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf228, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf214, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf200, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf185, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf171, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf157, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf142, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf128, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf114, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf99, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf84, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf71, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf57, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf42, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf26, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_165 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_168 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_177 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_180 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_183 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_186 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_189 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_198 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_207 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_213 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_219 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_228 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_231 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_237 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_246 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_249 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_252 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_255 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_258 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_264 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_267 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_273 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_276 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_279 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_285 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_288 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_300 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_303 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_309 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_312 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_315 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_318 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((64, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)