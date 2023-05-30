import re
import subprocess

from consts import *

def split_and_strip(_str, splitter=" ", max_count=-1, rsplit=False):
    if not rsplit:
        parts = _str.split(splitter, max_count)
    else:
        parts = _str.rsplit(splitter, max_count)
    new_parts = []
    for part in parts:
        new_part = part.strip()
        if new_part:
            new_parts.append(new_part)
    
    return new_parts


def remove_keywords(_str):

    for keyword in ignore_keywords:
        if keyword in _str:

            idx = _str.index(keyword)

            # need to ensure keyword is only by itself, not within a name
            if idx > 0 and _str[idx - 1] != " ":
                continue
            if idx + len(keyword) < len(_str) and _str[idx + len(keyword)] != " ":
                continue

            _str = _str.replace(keyword, " ")
    
    return _str.strip()


def parse_arg(arg):

    if arg == "void":
        return None, None, None

    arg_type_name = arg
    arg_val = None

    if "=" in arg:
        arg_type_name, arg_val = split_and_strip(arg, "=")

    last_idx = len(arg_type_name) - 1

    if "[]" in arg_type_name:
        assert(arg_type_name[-2:] == "[]")
        last_idx = arg_type_name.index("[]") - 1

    name_idx = None
    for i in range(last_idx, -1, -1):
        char_i = arg_type_name[i]
        if char_i.isalnum() or (char_i == "_"):
            name_idx = i
        else:
            break
    
    arg_type = arg_type_name[:name_idx]
    arg_name = arg_type_name[name_idx:last_idx + 1]

    if "[]" in arg_type_name:
        arg_type += "[]"
        arg_name

    return arg_type, arg_name, arg_val


def gen_func_sig_args_str(arg_types, arg_names):
    sig_args_str = ""

    for i in range(len(arg_types)):
        
        arg_type = arg_types[i]
        arg_name = arg_names[i]

        if arg_type.endswith("[]"):
            arg_type = arg_type[:-2]
            arg_name += "[]"

        sig_args_str += f"{arg_type} {arg_name}"

        if i != len(arg_types) - 1:
            sig_args_str += ", "

    return sig_args_str


def gen_func_preload(func_sig):

    func_sig = remove_keywords(func_sig)
    before_args, args = func_sig.split("(", 1)
    if "=" in before_args:
        return None, None

    # Extract return type and function name
    before_args_parts = split_and_strip(before_args, max_count=1, rsplit=True)
    if len(before_args_parts) == 2:
        ret_type, func_name = before_args_parts
    else:
        return None, None

    # Extract argument types and names
    args_str, _ = args.rsplit(")", 1)
    arg_list = split_and_strip(args_str, ", ")

    arg_types = []
    arg_names = []
    arg_vals = []

    for arg in arg_list:
        arg_type, arg_name, arg_val = parse_arg(arg)
        if arg_type and arg_name:
            arg_types.append(arg_type)
            arg_names.append(arg_name)
            arg_vals.append(arg_val)

    arg_types_str = ", ".join(arg_types)
    arg_names_str = ", ".join(arg_names)
    args_str_no_val = gen_func_sig_args_str(arg_types, arg_names)

    preload_func_name = f"l{func_name}"
    handle = "RTLD_NEXT"
    if ret_type == "cudnnStatus_t":
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
    if func_name not in exclude_trace_functions:
        if profile_kernel:
            func_preload_builder += profile_kernel_start
        else:
            func_preload_builder += profile_cpu_start

    # call original
    if ret_type != "void":
        func_preload_builder += f"\t{ret_type} res = \n"
    func_preload_builder += f"\t{preload_func_name}({arg_names_str});\n"
    
    if func_name not in exclude_trace_functions:
        if profile_kernel:
            func_preload_builder += profile_kernel_end
        else:
            func_preload_builder += profile_cpu_end
        
        func_preload_builder += f"\ttracer._kernel_seq.push_back((void *){preload_func_name});\n"

    if ret_type != "void":
        func_preload_builder += f"\treturn res;\n"

    # close bracket
    func_preload_builder += "}"

    return func_name, func_preload_builder


def gen_preload_from_file(file):
    generated_preload = {}

    start_acc = False
    acc = ""

    with open(file, "r") as f:
        for line in f:

            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if not start_acc:
                # potential cuda api declarations
                if ("typedef" not in line) and (
                        "(" in line or
                        "CUresult" in line or
                        "cudaError_t" in line or
                        len(line.split(" ")) == 1 or
                        "const" in line
                    ):

                    # span across multiple lines
                    if ";" not in line:
                        start_acc = True
                        acc = line
                        continue
                    
                    func_sig = line
                else:
                    continue
            else:
                acc = f"{acc} {line}"
                if ";" in line:
                    start_acc = False
                    func_sig = acc
                else:
                    continue
            
            if (all([word in func_sig for word in func_sig_must_contain]) and
                not any([word in func_sig for word in func_sig_must_not_contain])
            ):
                try:
                    func_name, func_preload = gen_func_preload(func_sig)
                    if func_name and func_preload:
                        generated_preload[func_name] = func_preload
                except Exception as e:
                    print(f"func_sig: {func_sig}")
                    raise e

    return generated_preload


def main():

    cuda_api_headers = [
        "/usr/local/cuda/include/cuda.h",
        "/usr/local/cuda/include/cuda_runtime.h",
        "/usr/local/cuda/include/cudnn.h",
        "/usr/local/cuda/include/cublas_v2.h",
    ]
    
    generated_preload = {}

    for header_file in cuda_api_headers:

        command = f"g++ -I/usr/local/cuda/include -E {header_file} > gcc_output.txt"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check the output
        if result.returncode == 0:
            gcc_output = result.stdout
        else:
            print(f"Command {command} failed.")
            exit(result.returncode)

        generated_preload.update(gen_preload_from_file("gcc_output.txt"))

    # Some special preload functions
    generated_preload.update(special_preload_funcs(profile_kernel))

    # Write to preload.cpp
    with open("preload.cpp", "w") as f:

        f.write(preload_template)
        f.write(trace_initialize_code)

        f.write("extern \"C\" { \n\n")

        for func_name in generated_preload:
            f.write(generated_preload[func_name])
            f.write("\n\n")
        
        f.write("}")

        
if __name__ == "__main__":
    main()