from tally.preload.consts import CUDA_API_HEADER_FILES, EXTERN_C_BEGIN, EXTERN_C_END
from tally.preload.client_consts import (
    API_ENUM_TEMPLATE_TOP,
    API_ENUM_TEMPLATE_BUTTOM,
    API_SPECIAL_ENUM,
    API_DECL_TEMPLATE_TOP,
    API_DECL_TEMPLATE_BUTTOM,
    API_DEF_TEMPLATE_TOP,
    API_DEF_TEMPLATE_BUTTOM,
    SPECIAL_CLIENT_PRELOAD_FUNCS,
    CLIENT_PRELOAD_TEMPLATE
)
from tally.preload.preload_util import gen_func_sig_args_str, parse_func_sig, preprocess_header_file, generate_func_sig_from_file

def gen_client_func_decl_def(func_sig):

    parse_res = parse_func_sig(func_sig)
    if parse_res:
        ret_type, func_name, arg_types, arg_names, arg_vals = parse_res
    else:
        return None

    args_str_no_val = gen_func_sig_args_str(arg_types, arg_names)

    preload_func_name = f"l{func_name}"
    handle = "RTLD_NEXT"
    if "cudnn" in func_name.lower() or "cudnn" in ret_type.lower():
        handle = "cudnn_handle";
    elif "CUresult" in ret_type:
        handle = "cuda_handle"
    elif "cudaError_t" in ret_type:
        handle = "cudart_handle"

    func_declaration = f"extern {ret_type} (*{preload_func_name}) ({args_str_no_val});\n"

    func_definition = ""
    func_definition += f"{ret_type} (*{preload_func_name}) ({args_str_no_val}) =\n"
    func_definition += f"\t({ret_type} (*) ({args_str_no_val})) dlsym({handle}, \"{func_name}\");\n\n"

    return func_name, func_declaration, func_definition


def gen_client_api_from_file(file):
    func_sig_list = generate_func_sig_from_file(file)
    client_api = {}

    for func_sig in func_sig_list:
        try:
            res = gen_client_func_decl_def(func_sig)
            if res:
                func_name, func_declaration, func_definition = res
                client_api[func_name] = (func_declaration, func_definition)
        except Exception as e:
            print(f"func_sig: {func_sig}")
            raise e

    return client_api


def gen_func_client_preload(func_sig):

    parse_res = parse_func_sig(func_sig)
    if parse_res:
        ret_type, func_name, arg_types, arg_names, arg_vals = parse_res
    else:
        return None

    arg_types_str = ", ".join(arg_types)
    arg_names_str = ", ".join(arg_names)
    args_str_no_val = gen_func_sig_args_str(arg_types, arg_names)

    preload_func_name = f"l{func_name}"
 
    # Generate function preload
    func_preload_builder = ""

    # Signature
    func_preload_builder += f"{ret_type} {func_name}({args_str_no_val})\n"
    func_preload_builder += "{\n"

    # print
    func_preload_builder += f"\tprintf(\"{func_name} hooked\\n\");\n"

    # call original
    if ret_type != "void":
        func_preload_builder += f"\t{ret_type} res = \n"
        
    func_preload_builder += f"\t\t{preload_func_name}({arg_names_str});\n"

    if ret_type != "void":
        func_preload_builder += f"\treturn res;\n"

    # close bracket
    func_preload_builder += "}"

    return func_name, func_preload_builder


def gen_client_preload_from_file(file):
    func_sig_list = generate_func_sig_from_file(file)
    generated_preload = {}

    for func_sig in func_sig_list:
        try:
            res = gen_func_client_preload(func_sig)
            if res:
                func_name, func_preload = res
                if func_name and func_preload:
                    generated_preload[func_name] = func_preload
        except Exception as e:
            print(f"func_sig: {func_sig}")
            raise e
        
    return generated_preload


def gen_client_api(header_files=CUDA_API_HEADER_FILES, decl_output_file="cuda_api.h",
                   def_output_file="cuda_api.cpp", enum_output_file="cuda_api_enum.h"):

    declarations = {}
    definitions = {}

    for header_file in header_files:

        preprocess_header_file(header_file, output_file="/tmp/gcc_output.txt")

        client_api = gen_client_api_from_file("/tmp/gcc_output.txt")
        for func_name in client_api:
            func_declaration, func_definition = client_api[func_name]

            if func_name not in declarations:
                declarations[func_name] = func_declaration
                definitions[func_name] = func_definition
    
    with open(decl_output_file, "w") as f:

        f.write(API_DECL_TEMPLATE_TOP)
        for func_name in declarations:
            f.write(declarations[func_name])
        f.write(API_DECL_TEMPLATE_BUTTOM)
    
    with open(def_output_file, "w") as f:

        f.write(API_DEF_TEMPLATE_TOP)
        for func_name in definitions:
            f.write(definitions[func_name])
        f.write(API_DEF_TEMPLATE_BUTTOM)

    with open(enum_output_file, 'w') as f:
        f.write(API_ENUM_TEMPLATE_TOP)
        f.write("\n\nenum CUDA_API_ENUM {\n")

        for func_name in definitions:
            f.write(f"\t{func_name.upper()},\n")
        
        for idx, func_name in enumerate(API_SPECIAL_ENUM):
            if idx != len(API_SPECIAL_ENUM) - 1:
                f.write(f"\t{func_name.upper()},\n")
            else:
                f.write(f"\t{func_name.upper()}\n")

        f.write("};\n")

        f.write(API_ENUM_TEMPLATE_BUTTOM)


def gen_client_preload(header_files=CUDA_API_HEADER_FILES, output_file="tally_client.cpp"):
    func_preload = {}

    for header_file in header_files:

        preprocess_header_file(header_file, output_file="/tmp/gcc_output.txt")
        func_preload.update(gen_client_preload_from_file("/tmp/gcc_output.txt"))

    # Some special preload functions
    func_preload.update(SPECIAL_CLIENT_PRELOAD_FUNCS)

    # Write to preload.cpp
    with open(output_file, "w") as f:

        f.write(CLIENT_PRELOAD_TEMPLATE)

        f.write(EXTERN_C_BEGIN)

        for func_name in func_preload:
            f.write(func_preload[func_name])
            f.write("\n\n")
        
        f.write(EXTERN_C_END)

