import subprocess
import re

from tally.preload.consts import IGNORE_KEYWORDS, FUNC_SIG_MUST_CONTAIN, FUNC_SIG_MUST_NOT_CONTAIN, FUNC_SIG_MUST_NOT_CONTAIN_KEYWORDS
from tally.util.util import split_and_strip, remove_keywords, is_alnum_underscore

def get_func_name_from_sig(func_sig):
    parse_res = parse_func_sig(func_sig)
    if parse_res:
        ret_type, func_name, arg_types, arg_names, arg_vals = parse_res
    else:
        return None
    
    return func_name

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


def parse_func_sig(func_sig):
    func_sig = remove_keywords(func_sig, IGNORE_KEYWORDS)
    before_args, args = func_sig.split("(", 1)
    if "=" in before_args:
        return None

    # Extract return type and function name
    before_args_parts = split_and_strip(before_args, max_count=1, rsplit=True)
    if len(before_args_parts) == 2:
        ret_type, func_name = before_args_parts
    else:
        return None

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
    
    return ret_type, func_name, arg_types, arg_names, arg_vals


def generate_func_sig_from_file(file):
    start_acc = False
    acc = ""
    func_sig_list = []

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
                        (len(line.split(" ")) == 1 and is_alnum_underscore(line)) or
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
            
            if (all([word in func_sig for word in FUNC_SIG_MUST_CONTAIN]) and
                    not any([word in func_sig for word in FUNC_SIG_MUST_NOT_CONTAIN]) and
                    not any([word in re.findall(r'\w+', func_sig) for word in FUNC_SIG_MUST_NOT_CONTAIN_KEYWORDS])):

                func_sig_list.append(func_sig)
        
    return func_sig_list


def compile_preload(preload_cpp_file="preload.cpp", output_file=None):
    file_prefix = preload_cpp_file.rsplit(".", maxsplit=1)[0]
    if output_file is None:
        output_file = f"{file_prefix}.so"
    
    compile_cmd = f"g++ -I/usr/local/cuda/include -fPIC -shared -o {output_file} {preload_cpp_file}"
    process = subprocess.Popen(compile_cmd, shell=True, universal_newlines=True)
    process.wait()


def preprocess_header_file(header_file, output_file="/tmp/gcc_output.txt"):
    command = f"g++ -I/usr/local/cuda/include -E {header_file} > {output_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check the output
    if result.returncode != 0:
        raise Exception(f"Command {command} failed.")