import re

kernel_name_pattern = r'(\.visible\s+)?\.entry (\w+)'
kernel_param_pattern = None
b32_reg_decl_pattern = r'\.reg \.b32 %r<(\d+)>;'
block_idx_pattern = r'mov\.u32 %r(\d+), %ctaid\.([xyz])'

def slice_kernel(ptx_code_path, output_code_path):

    ptx_code = None
    with open(ptx_code_path, 'r') as file:
        ptx_code = file.read()
    
    sliced_ptx_code = ""

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
            use_block_idx_x = False
            use_block_idx_y = False
            use_block_idx_z = False
            kernel_lines.clear()
        
        if record_kernel:
            kernel_lines.append(line)
        else:
            sliced_ptx_code += f"{line}\n"
        
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
        
        if record_kernel and line == "ret;":
            record_kernel = False

            last_param_pattern = f"\.param (.+) {kernel_name}_param_{num_params - 1}"
            last_ld_param_pattern = f"ld\.param(.+)\[{kernel_name}_param_{num_params - 1}\];"
            num_additional_b32 = [use_block_idx_x, use_block_idx_y, use_block_idx_z].count(True) * 2

            curr_reg = num_b32_regs

            if use_block_idx_x:
                block_offset_x_reg = curr_reg
                new_block_idx_x_reg = curr_reg + 1
                curr_reg += 2
            
            if use_block_idx_y:
                block_offset_y_reg = curr_reg
                new_block_idx_y_reg = curr_reg + 1
                curr_reg += 2

            if use_block_idx_z:
                block_offset_z_reg = curr_reg
                new_block_idx_z_reg = curr_reg + 1

            reg_replacement_rules = {}

            for line in kernel_lines:
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
                        sliced_ptx_code += f"ld.param.u32 %r{block_offset_x_reg}, [{kernel_name}_param_{num_params}];\n"
                    if use_block_idx_y:
                        sliced_ptx_code += f"ld.param.u32 %r{block_offset_y_reg}, [{kernel_name}_param_{num_params}+4];\n"
                    if use_block_idx_z:
                        sliced_ptx_code += f"ld.param.u32 %r{block_offset_z_reg}, [{kernel_name}_param_{num_params}+8];\n"

                    continue

                block_idx_match = re.search(block_idx_pattern, line)
                if block_idx_match:
                    block_idx_match_dim = block_idx_match.group(2)
                    block_idx_match_reg = int(block_idx_match.group(1))

                    sliced_ptx_code += f"{line}\n"
                    if block_idx_match_dim == "x":
                        sliced_ptx_code += f"add.u32 %r{new_block_idx_x_reg}, %r{block_idx_match_reg}, %r{block_offset_x_reg};\n"
                        reg_replacement_rules[f"%r{block_idx_match_reg}(?!\d)"] = f"%r{new_block_idx_x_reg}"
                    if block_idx_match_dim == "y":
                        sliced_ptx_code += f"add.u32 %r{new_block_idx_y_reg}, %r{block_idx_match_reg}, %r{block_offset_y_reg};\n"
                        reg_replacement_rules[f"%r{block_idx_match_reg}(?!\d)"] = f"%r{new_block_idx_y_reg}"
                    if block_idx_match_dim == "z":
                        sliced_ptx_code += f"add.u32 %r{new_block_idx_z_reg}, %r{block_idx_match_reg}, %r{block_offset_z_reg};\n"
                        reg_replacement_rules[f"%r{block_idx_match_reg}(?!\d)"] = f"%r{new_block_idx_z_reg}"
                    continue

                for pattern, replacement in reg_replacement_rules.items():
                    line = re.sub(pattern, replacement, line)

                sliced_ptx_code += f"{line}\n"

    with open(output_code_path, 'w') as out:
        out.write(sliced_ptx_code)

def get_kernel_names(ptx_code_path):

    kernel_names = []
    with open(ptx_code_path, 'r') as file:
        ptx_code = file.read()
    
    for line in ptx_code.split("\n"):

        kernel_name_match = re.search(kernel_name_pattern, line)
        if kernel_name_match:
            kernel_name = kernel_name_match.group(1)
            kernel_names.append(kernel_name)
    
    for k in kernel_names:
        print(k)