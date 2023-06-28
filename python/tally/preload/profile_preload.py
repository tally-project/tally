from tally.preload.consts import CUDA_API_HEADER_FILES, EXTERN_C_BEGIN, EXTERN_C_END
from tally.preload.profile_consts import (
    EXCLUDE_TRACE_FUNCTIONS,
    PROFILE_KERNEL_START,
    PROFILE_KERNEL_END,
    PROFILE_CPU_START,
    PROFILE_CPU_END,
    PRELOAD_TEMPLATE,
    get_trace_initialize_code,
    special_preload_funcs
)
from tally.preload.preload_util import (
    gen_func_sig_args_str,
    parse_func_sig,
    generate_func_sig_from_file,
    preprocess_header_file
)

def gen_func_profile_preload(func_sig, profile_kernel):

    parse_res = parse_func_sig(func_sig)
    if parse_res:
        ret_type, func_name, arg_types, arg_names, arg_vals = parse_res
    else:
        return None

    arg_types_str = ", ".join(arg_types)
    arg_names_str = ", ".join(arg_names)
    args_str_no_val = gen_func_sig_args_str(arg_types, arg_names)

    preload_func_name = f"l{func_name}"
    handle = "RTLD_NEXT"
    if "cudnn" in func_name.lower() or "cudnn" in ret_type.lower():
        handle = "tracer.cudnn_handle";

    # Generate function preload
    func_preload_builder = ""

    # Signature
    func_preload_builder += f"{ret_type} {func_name}({args_str_no_val})\n"
    func_preload_builder += "{\n"

    # dlsys
    func_preload_builder += f"\tstatic {ret_type} (*{preload_func_name}) ({arg_types_str});\n"
    func_preload_builder += f"\tif (!{preload_func_name}) {{\n"
    func_preload_builder += f"\t\t{preload_func_name} = ({ret_type} (*) ({arg_types_str})) dlsym({handle}, \"{func_name}\");\n"
    func_preload_builder += f"\t\ttracer._kernel_map[(void *) {preload_func_name}] = std::string(\"{func_name}\");\n"
    func_preload_builder += f"\t}}\n"
    func_preload_builder += f"\tassert({preload_func_name});\n"

    # print
    # func_preload_builder += f"\tprintf(\"{func_name} hooked\\n\");\n"

    # Trace the function
    if func_name not in EXCLUDE_TRACE_FUNCTIONS:
        if profile_kernel:
            func_preload_builder += PROFILE_KERNEL_START
        else:
            func_preload_builder += PROFILE_CPU_START

    # call original
    if ret_type != "void":
        func_preload_builder += f"\t{ret_type} res = \n"
    func_preload_builder += f"\t\t{preload_func_name}({arg_names_str});\n"
    
    if func_name not in EXCLUDE_TRACE_FUNCTIONS:
        if profile_kernel:
            func_preload_builder += PROFILE_KERNEL_END
        else:
            func_preload_builder += PROFILE_CPU_END
        
        func_preload_builder += "\tif (tracer.profile_start) {\n"
        func_preload_builder += f"\t\ttracer._kernel_seq.push_back((void *){preload_func_name});\n"
        func_preload_builder += "\t}\n"

    if ret_type != "void":
        func_preload_builder += f"\treturn res;\n"

    # close bracket
    func_preload_builder += "}"

    return func_name, func_preload_builder


def gen_profile_preload_from_file(file, profile_kernel=False):
    func_sig_list = generate_func_sig_from_file(file)
    generated_preload = {}

    for func_sig in func_sig_list:
        try:
            res = gen_func_profile_preload(func_sig, profile_kernel)
            if res:
                func_name, func_preload = res
                if func_name and func_preload:
                    generated_preload[func_name] = func_preload
        except Exception as e:
            print(f"func_sig: {func_sig}")
            raise e

    return generated_preload

def gen_profile_preload(header_files=CUDA_API_HEADER_FILES, profile_kernel=False,
                     output_file="preload.cpp", print_trace=True):
    
    generated_preload = {}

    for header_file in header_files:

        preprocess_header_file(header_file, output_file="/tmp/gcc_output.txt")
        generated_preload.update(gen_profile_preload_from_file("/tmp/gcc_output.txt", profile_kernel))

    # Some special preload functions
    generated_preload.update(special_preload_funcs(profile_kernel))

    # Write to preload.cpp
    with open(output_file, "w") as f:

        f.write(PRELOAD_TEMPLATE)
        f.write(get_trace_initialize_code(print_trace))

        f.write(EXTERN_C_BEGIN)

        for func_name in generated_preload:
            f.write(generated_preload[func_name])
            f.write("\n\n")
        
        f.write(EXTERN_C_END)