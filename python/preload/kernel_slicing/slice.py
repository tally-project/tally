import re

ptx_code = """
.visible .entry _Z19elementwiseAdditionPfS_S_i(
.param .u64 _Z19elementwiseAdditionPfS_S_i_param_0,
.param .u64 _Z19elementwiseAdditionPfS_S_i_param_1,
.param .u64 _Z19elementwiseAdditionPfS_S_i_param_2,
.param .u32 _Z19elementwiseAdditionPfS_S_i_param_3
)
{
.reg .pred %p<2>;
.reg .f32 %f<4>;
.reg .b32 %r<6>;
.reg .b64 %rd<11>;


ld.param.u64 %rd1, [_Z19elementwiseAdditionPfS_S_i_param_0];
ld.param.u64 %rd2, [_Z19elementwiseAdditionPfS_S_i_param_1];
ld.param.u64 %rd3, [_Z19elementwiseAdditionPfS_S_i_param_2];
ld.param.u32 %r2, [_Z19elementwiseAdditionPfS_S_i_param_3];
mov.u32 %r3, %tid.x;
mov.u32 %r4, %ntid.x;
mov.u32 %r5, %ctaid.x;
mad.lo.s32 %r1, %r5, %r4, %r3;
setp.ge.s32 %p1, %r1, %r2;
@%p1 bra $L__BB0_2;

cvta.to.global.u64 %rd4, %rd1;
mul.wide.s32 %rd5, %r1, 4;
add.s64 %rd6, %rd4, %rd5;
cvta.to.global.u64 %rd7, %rd2;
add.s64 %rd8, %rd7, %rd5;
ld.global.f32 %f1, [%rd8];
ld.global.f32 %f2, [%rd6];
add.f32 %f3, %f2, %f1;
cvta.to.global.u64 %rd9, %rd3;
add.s64 %rd10, %rd9, %rd5;
st.global.f32 [%rd10], %f3;

$L__BB0_2:
ret;

}


"""
kernel_name_pattern = r'\.visible \.entry (\w+)'
kernel_param_pattern = None
b32_reg_decl_pattern = r'\.reg \.b32 %r<(\d+)>;'
block_idx_pattern = r'mov\.u32 %r(\d+), %ctaid\.([xyz])'

kernel_name = None
num_params = 0
num_b32_regs = 0
use_block_idx_x = False
use_block_idx_y = False
use_block_idx_z = False

for line in ptx_code.split("\n"):

    kernel_name_match = re.search(kernel_name_pattern, line)
    if kernel_name_match:
        kernel_name = kernel_name_match.group(1)
        kernel_param_pattern = f"\.param (.+) {kernel_name}_param_(\d+)"
    
    if kernel_param_pattern:
        kernel_param_match = re.search(kernel_param_pattern, line)
        if kernel_param_match:
            num_params += 1

    b32_reg_decl_match = re.search(b32_reg_decl_pattern, line)
    if b32_reg_decl_match:
        num_b32_regs = int(b32_reg_decl_match.group(1))
    
    block_idx_match = re.search(block_idx_pattern, line)
    if block_idx_match:
        block_idx_match_dim = block_idx_match.group(2)
        if block_idx_match_dim == "x":
            use_block_idx_x = True
        if block_idx_match_dim == "y":
            use_block_idx_y = True
        if block_idx_match_dim == "z":
            use_block_idx_z = True

last_param_pattern = f"\.param \.(\w+) {kernel_name}_param_{num_params - 1}"
last_ld_param_pattern = f"ld\.param(.+)\[{kernel_name}_param_{num_params - 1}\];"
b32_inst_pattern = f"%r(\d+)"
num_additional_b32 = [use_block_idx_x, use_block_idx_y, use_block_idx_z].count(True) * 2

block_offset_x_reg = num_b32_regs
block_offset_y_reg = num_b32_regs + 2
block_offset_z_reg = num_b32_regs + 4

new_block_idx_x_reg = num_b32_regs + 1
new_block_idx_y_reg = num_b32_regs + 3
new_block_idx_z_reg = num_b32_regs + 5

block_idx_mapping = {}
reg_replacement_rules = {}

sliced_ptx_code = ""

for line in ptx_code.split("\n"):
    last_param_match = re.search(last_param_pattern, line)
    if last_param_match:
        sliced_ptx_code += f"{line},\n"
        sliced_ptx_code += f".param .align 4 .b8 {kernel_name}_param_{num_params}[12]\n"
        continue

    b32_reg_decl_match = re.search(b32_reg_decl_pattern, line)
    if b32_reg_decl_match:
        sliced_ptx_code += f".reg .b32 %r<{num_b32_regs + num_additional_b32}>;\n"
        continue

    last_ld_param_match = re.search(last_ld_param_pattern, line)
    if last_ld_param_match:
        sliced_ptx_code += f"{line}\n"
        
        if use_block_idx_x:
            sliced_ptx_code += f"ld.param.u32 %r{block_offset_x_reg}, [{kernel_name}_param_4];\n"
        if use_block_idx_y:
            sliced_ptx_code += f"ld.param.u32 %r{block_offset_y_reg}, [{kernel_name}_param_4+4];\n"
        if use_block_idx_z:
            sliced_ptx_code += f"ld.param.u32 %r{block_offset_z_reg}, [{kernel_name}_param_4+8];\n"

        continue

    block_idx_match = re.search(block_idx_pattern, line)
    if block_idx_match:
        block_idx_match_dim = block_idx_match.group(2)
        block_idx_match_reg = int(block_idx_match.group(1))
        block_idx_mapping[block_idx_match_dim] = block_idx_match_reg

        sliced_ptx_code += f"{line}\n"
        if block_idx_match_dim == "x":
            sliced_ptx_code += f"add.u32 %r{new_block_idx_x_reg}, %r{block_idx_match_reg}, %r{block_offset_x_reg};\n"
            reg_replacement_rules[f"%r{block_idx_match_reg}"] = f"%r{new_block_idx_x_reg}"
        if block_idx_match_dim == "y":
            sliced_ptx_code += f"add.u32 %r{new_block_idx_y_reg}, %r{block_idx_match_reg}, %r{block_offset_y_reg};\n"
            reg_replacement_rules[f"%r{block_idx_match_reg}"] = f"%r{new_block_idx_y_reg}"
        if block_idx_match_dim == "z":
            sliced_ptx_code += f"add.u32 %r{new_block_idx_z_reg}, %r{block_idx_match_reg}, %r{block_offset_z_reg};\n"
            reg_replacement_rules[f"%r{block_idx_match_reg}"] = f"%r{new_block_idx_z_reg}"
        continue

    for pattern, replacement in reg_replacement_rules.items():
        line = re.sub(pattern, replacement, line)

    sliced_ptx_code += f"{line}\n"

print(sliced_ptx_code)
