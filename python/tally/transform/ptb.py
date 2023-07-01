import re

kernel_name_pattern = r'(\.visible\s+)?\.entry (\w+)'
kernel_param_pattern = None
b32_reg_decl_pattern = r'\.reg \.b32 %r<(\d+)>;'
pred_reg_decl_pattern = r'\.reg \.pred %p<(\d+)>;'
block_idx_pattern = r'mov\.u32 %r(\d+), %ctaid\.([xyz])'

PTB_RETURN_BLOCK_NAME = "L__PTB_RETURN"
PTB_LOOP_BLOCK_NAME = "L__PTB_LOOP"

def ptb_kernel(ptx_code_path, output_code_path):
    ptx_code = None
    with open(ptx_code_path, 'r') as file:
        ptx_code = file.read()

    ptb_ptx_code = ""

    kernel_param_pattern = None
    record_kernel = False
    kernel_lines = []

    for line in ptx_code.split("\n"):

        kernel_name_match = re.search(kernel_name_pattern, line)
        if kernel_name_match:

            record_kernel = True
            kernel_name = kernel_name_match.group(2)
            kernel_param_pattern = f"\.param (.+) {kernel_name}_param_(\d+)"
            num_params = 0
            num_b32_regs = 0
            num_pred_regs = 0
            kernel_lines.clear()
        
        if record_kernel:
            kernel_lines.append(line)
        else:
            ptb_ptx_code += f"{line}\n"
        
        if kernel_param_pattern:
            kernel_param_match = re.search(kernel_param_pattern, line)
            if kernel_param_match:
                num_params += 1

        b32_reg_decl_match = re.search(b32_reg_decl_pattern, line)
        if b32_reg_decl_match:
            num_b32_regs = int(b32_reg_decl_match.group(1))
        
        pred_reg_decl_match = re.search(pred_reg_decl_pattern, line)
        if pred_reg_decl_match:
            num_pred_regs = int(pred_reg_decl_match.group(1))
        
        if record_kernel and line == "ret;":
            record_kernel = False

            last_param_pattern = f"\.param (.+) {kernel_name}_param_{num_params - 1}"
            num_additional_b32 = 15     
            num_additional_pred_regs = 2

            origGridDim_x_reg = num_b32_regs
            origGridDim_y_reg = num_b32_regs + 1
            origGridDim_z_reg = num_b32_regs + 2
            threadIdx_x_reg = num_b32_regs + 3
            blockDim_x_reg = num_b32_regs + 4
            gridDim_x_reg = num_b32_regs + 5
            xy_tbs_reg = num_b32_regs + 6
            num_thread_blocks_reg = num_b32_regs + 7
            tb_idx_reg = num_b32_regs + 8
            newBlockIdx_z_reg = num_b32_regs + 9
            newBlockIdx_z_mul_xy_tbs_reg = num_b32_regs + 10
            tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg = num_b32_regs + 11
            newBlockIdx_y_reg = num_b32_regs + 12
            newBlockIdx_y_mul_origGridDim_x_reg = num_b32_regs + 13
            newBlockIdx_x_reg = num_b32_regs + 14

            reg_replacement_rules = {}

            for line in kernel_lines:
                last_param_match = re.search(last_param_pattern, line)
                if last_param_match:
                    ptb_ptx_code += f"{line},\n"
                    ptb_ptx_code += f".param .align 4 .b8 {kernel_name}_param_{num_params}[12]\n"
                    continue

                # Perform actions at the top
                if line.strip() == "{":
                    ptb_ptx_code += "{\n"
                    ptb_ptx_code += f".reg .b32 %r<{num_b32_regs + num_additional_b32}>;\n"
                    ptb_ptx_code += f".reg .pred %p<{num_pred_regs + num_additional_pred_regs}>;\n"

                    # Load origGridDim.x
                    ptb_ptx_code += f"ld.param.u32 %r{origGridDim_x_reg}, [{kernel_name}_param_{num_params}];\n"
                    # Load origGridDim.y
                    ptb_ptx_code += f"ld.param.u32 %r{origGridDim_y_reg}, [{kernel_name}_param_{num_params}+4];\n"
                    # Load origGridDim.z
                    ptb_ptx_code += f"ld.param.u32 %r{origGridDim_z_reg}, [{kernel_name}_param_{num_params}+8];\n"

                    # threadIdx.x
                    ptb_ptx_code += f"mov.u32 %r{threadIdx_x_reg}, %tid.x;\n"
                    # blockDim.x
                    ptb_ptx_code += f"mov.u32 %r{blockDim_x_reg}, %ntid.x;\n"
                    # gridDim.x
                    ptb_ptx_code += f"mov.u32 %r{gridDim_x_reg}, %nctaid.x;\n"

                    # xy_tbs = origGridDim.x * origGridDim.y
                    ptb_ptx_code += f"mul.lo.s32 %r{xy_tbs_reg}, %r{origGridDim_x_reg}, %r{origGridDim_y_reg};\n"
                    # num_thread_blocks = origGridDim.x * origGridDim.y * origGridDim.z
                    ptb_ptx_code += f"mul.lo.s32 %r{num_thread_blocks_reg}, %r{origGridDim_z_reg}, %r{xy_tbs_reg};\n"

                    # tb_idx = blockIdx.x
                    ptb_ptx_code += f"mov.u32 %r{tb_idx_reg}, %ctaid.x;\n"
                    # tb_idx >= num_thread_blocks
                    ptb_ptx_code += f"setp.ge.u32 %p{num_pred_regs}, %r{tb_idx_reg}, %r{num_thread_blocks_reg};\n"
                    # branch to return if tb_idx >= num_thread_blocks
                    ptb_ptx_code += f"@%p{num_pred_regs} bra ${PTB_RETURN_BLOCK_NAME};\n\n"

                    ptb_ptx_code += f"${PTB_LOOP_BLOCK_NAME}:\n"
                    
                    # newBlockIdx.z
                    ptb_ptx_code += f"div.u32 %r{newBlockIdx_z_reg}, %r{tb_idx_reg}, %r{xy_tbs_reg};\n"
                    # newBlockIdx.z * xy_tbs
                    ptb_ptx_code += f"mul.lo.s32 %r{newBlockIdx_z_mul_xy_tbs_reg}, %r{newBlockIdx_z_reg}, %r{xy_tbs_reg};\n"
                    # tb_idx - newBlockIdx.z * xy_tbs
                    ptb_ptx_code += f"sub.s32 %r{tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg}, %r{tb_idx_reg}, %r{newBlockIdx_z_mul_xy_tbs_reg};\n"
                    # newBlockIdx.y
                    ptb_ptx_code += f"div.u32 %r{newBlockIdx_y_reg}, %r{tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg}, %r{origGridDim_x_reg};\n"
                    # newBlockIdx.y * origGridDim.x
                    ptb_ptx_code += f"mul.lo.s32 %r{newBlockIdx_y_mul_origGridDim_x_reg}, %r{newBlockIdx_y_reg}, %r{origGridDim_x_reg};\n"
                    # newBlockIdx.x
                    ptb_ptx_code += f"sub.s32 %r{newBlockIdx_x_reg}, %r{tb_idx_sub_newBlockIdx_z_mul_xy_tbs_reg}, %r{newBlockIdx_y_mul_origGridDim_x_reg};\n"

                    continue

                b32_reg_decl_match = re.search(b32_reg_decl_pattern, line)
                if b32_reg_decl_match:
                    continue
            
                pred_reg_decl_match = re.search(pred_reg_decl_pattern, line)
                if pred_reg_decl_match:
                    continue
            
                block_idx_match = re.search(block_idx_pattern, line)
                if block_idx_match:
                    block_idx_match_dim = block_idx_match.group(2)
                    block_idx_match_reg = int(block_idx_match.group(1))

                    # ptb_ptx_code += f"{line}\n"
                    
                    if block_idx_match_dim == "x":
                        reg_replacement_rules[f"%r{block_idx_match_reg}(?!\d)"] = f"%r{newBlockIdx_x_reg}"
                    if block_idx_match_dim == "y":
                        reg_replacement_rules[f"%r{block_idx_match_reg}(?!\d)"] = f"%r{newBlockIdx_y_reg}"
                    if block_idx_match_dim == "z":
                        reg_replacement_rules[f"%r{block_idx_match_reg}(?!\d)"] = f"%r{newBlockIdx_z_reg}"
                    continue
            
                # instead of return, now in a loop
                if line.strip() == "ret;":

                    # tb_idx += gridDim.x
                    ptb_ptx_code += f"add.s32 %r{tb_idx_reg}, %r{tb_idx_reg}, %r{gridDim_x_reg};\n"
                    # tb_idx < num_thread_blocks
                    ptb_ptx_code += f"setp.lt.u32 %p{num_pred_regs + 1}, %r{tb_idx_reg}, %r{num_thread_blocks_reg};\n"
                    # branch to L__PTB_LOOP if tb_idx < num_thread_blocks
                    ptb_ptx_code += f"@%p{num_pred_regs + 1} bra ${PTB_LOOP_BLOCK_NAME};\n\n"

                    # Add return block
                    ptb_ptx_code += f"${PTB_RETURN_BLOCK_NAME}:\n"
                    ptb_ptx_code += f"ret;\n"

                    continue

                for pattern, replacement in reg_replacement_rules.items():
                    line = re.sub(pattern, replacement, line)
                
                ptb_ptx_code += f"{line}\n"

    with open(output_code_path, 'w') as out:
        out.write(ptb_ptx_code)