from tally.preload.consts import CUDA_API_HEADER_FILES, EXTERN_C_BEGIN, EXTERN_C_END
from tally.preload.client_consts import (
    API_ENUM_TEMPLATE_TOP,
    API_ENUM_TEMPLATE_BUTTOM,
    API_SPECIAL_ENUM,
    API_DECL_TEMPLATE_TOP,
    API_DECL_TEMPLATE_BUTTOM,
    API_DEF_TEMPLATE_TOP,
    API_DEF_TEMPLATE_BUTTOM,
    MSG_STRUCT_TEMPLATE_TOP,
    MSG_STRUCT_TEMPLATE_BUTTOM,
    TALLY_SERVER_HEADER_TEMPLATE_TOP,
    TALLY_SERVER_HEADER_TEMPLATE_BUTTOM,
    REGISTER_FUNCS,
    SPECIAL_CLIENT_PRELOAD_FUNCS,
    PARAM_INDICES,
    CLIENT_PRELOAD_TEMPLATE,
    FORWARD_API_CALLS,
    IGNORE_CALLS,
    get_preload_func_template_iox,
    is_get_param_func,
    should_check_cuda_err,
    get_param_group
)
from tally.preload.preload_util import (
    gen_func_sig_args_str,
    parse_func_sig,
    preprocess_header_file,
    generate_func_sig_from_file,
    get_func_name_from_sig
)
from tally.util.util import rreplace

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
    elif "cudaError_t" in ret_type or "cudaGetError" in func_name:
        handle = "cudart_handle"
    elif "cublaslt" in func_name.lower() or "cublaslt" in ret_type.lower():
        handle = "cublasLt_handle"
    elif "cublas" in func_name.lower() or "cublas" in ret_type.lower():
        handle = "cublas_handle"
    elif "cusparse" in func_name.lower() or "cusparse" in ret_type.lower():
        handle = "cusparse_handle"
    elif "nvrtc" in func_name.lower():
        handle = "nvrtc_handle"
    elif "nccl" in func_name.lower():
        handle = "nccl_handle"
    elif "curand" in func_name.lower():
        handle = "curand_handle"
    elif "nvml" in func_name.lower():
        handle = "nvml_handle"

    func_declaration = f"extern {ret_type} (*{preload_func_name}) ({args_str_no_val});\n"

    func_definition = ""
    func_definition += f"{ret_type} (*{preload_func_name}) ({args_str_no_val}) =\n"
    func_definition += f"\t({ret_type} (*) ({args_str_no_val})) dlsym({handle}, \"{func_name}\");\n\n"

    return func_declaration, func_definition


def gen_client_msg_struct(func_sig):
    parse_res = parse_func_sig(func_sig)
    if parse_res:
        ret_type, func_name, arg_types, arg_names, arg_vals = parse_res
    else:
        return None

    msg_struct = ""

    # Currently only generate functions that are forward calls
    if (func_name in FORWARD_API_CALLS or
        is_get_param_func(func_name)):

        msg_struct += f"struct {func_name}Arg {{\n"

        for i in range(len(arg_types)):
            _arg_type = arg_types[i].replace("const ", "").replace("[]", "*")
            msg_struct += f"""\t{_arg_type} {arg_names[i]};\n"""
        
        msg_struct += "};\n"

    if is_get_param_func(func_name):
        group = get_param_group(func_name)

        msg_struct += "\n"
        msg_struct += f"struct {func_name}Response {{\n"

        for idx in PARAM_INDICES[group]:
            new_arg_type = rreplace(arg_types[idx].strip().replace("const ", ""), "*", "")
            msg_struct += f"""\t{new_arg_type} {arg_names[idx]};\n"""

        msg_struct += f"\t{ret_type} err;\n"
        msg_struct += "};\n"
    
    if msg_struct:
        return msg_struct

    return None


def gen_server_handler(func_sig):
    parse_res = parse_func_sig(func_sig)
    if parse_res:
        ret_type, func_name, arg_types, arg_names, arg_vals = parse_res
    else:
        return None

    if func_name in SPECIAL_CLIENT_PRELOAD_FUNCS:
        return None

    handler = ""
    handler += f"""void TallyServer::handle_{func_name}(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{{
"""
    handler += f"\tTALLY_SPD_LOG(\"Received request: {func_name}\");\n"

    if is_get_param_func(func_name):
        group = get_param_group(func_name)
        indices = PARAM_INDICES[group]

        handler += f"\tauto args = (struct {func_name}Arg *) __args;\n"
        handler += f"\tauto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);\n"

        for idx in range(len(arg_names)):
            arg_idx_type = arg_types[idx].strip()
            if arg_idx_type == "cudaStream_t" or arg_idx_type == "CUstream":
                handler += f"""
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    {arg_idx_type} __stream = args->{arg_names[idx]};

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {{
        __stream = client_data_all[client_uid].default_stream;
    }}
"""
                break

        handler += f"""
    iox_server->loan(requestHeader, sizeof({func_name}Response), alignof({func_name}Response))
        .and_then([&](auto& responsePayload) {{
            auto response = static_cast<{func_name}Response*>(responsePayload);
            response->err = {func_name}(
"""

        for idx in range(len(arg_names)):
            if idx in indices:
                handler += f"\t\t\t\t(args->{arg_names[idx]} ? &(response->{arg_names[idx]}) : NULL)"
            else:
                if arg_types[idx].strip() == "cudaStream_t" or arg_types[idx].strip() == "CUstream":
                    handler += f"\t\t\t\t__stream"
                else:
                    handler += f"\t\t\t\targs->{arg_names[idx]}"
            if idx != len(arg_names) - 1:
                handler += ",\n"
            else:
                handler += "\n\t\t\t);\n"

        handler += f"""
            {"CHECK_CUDA_ERROR(response->err);" if should_check_cuda_err(ret_type, func_name) else ""}
            iox_server->send(response).or_else(
                [&](auto& error) {{ LOG_ERR_AND_EXIT("Could not send Response: ", error); }});
        }})
        .or_else(
            [&](auto& error) {{ LOG_ERR_AND_EXIT("Could not allocate Response: ", error); }});
"""
    elif func_name in FORWARD_API_CALLS:
        handler += f"\tauto args = (struct {func_name}Arg *) __args;\n"
        handler += f"\tauto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);\n"

        for idx in range(len(arg_names)):
            arg_idx_type = arg_types[idx].strip()
            if arg_idx_type == "cudaStream_t" or arg_idx_type == "CUstream":
                handler += f"""
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    {arg_idx_type} __stream = args->{arg_names[idx]};

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {{
        __stream = client_data_all[client_uid].default_stream;
    }}
"""
                break

        handler += f"""
    iox_server->loan(requestHeader, sizeof({ret_type}), alignof({ret_type}))
        .and_then([&](auto& responsePayload) {{
            auto response = static_cast<{ret_type}*>(responsePayload);
            *response = {func_name}(
"""
        for idx, arg_name in enumerate(arg_names):
            arg_idx_type = arg_types[idx].strip()
            if arg_idx_type == "cudaStream_t" or arg_idx_type == "CUstream":
                handler += f"\t\t\t\t__stream"
            else:
                handler += f"\t\t\t\targs->{arg_name}"
            if idx != len(arg_names) - 1:
                handler += ",\n"

        handler += f"""
            );
            {"CHECK_CUDA_ERROR(*response);" if should_check_cuda_err(ret_type, func_name) else ""}
            iox_server->send(response).or_else(
                [&](auto& error) {{ LOG_ERR_AND_EXIT("Could not send Response: ", error); }});
        }})
        .or_else(
            [&](auto& error) {{ LOG_ERR_AND_EXIT("Could not allocate Response: ", error); }});
"""
    else:
        handler += f"\tthrow std::runtime_error(std::string(__FILE__) + \":\" + std::to_string(__LINE__) + \": Unimplemented.\");\n"

    handler += "}\n"
    return handler


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

    func_preload_builder += f"\tTALLY_SPD_LOG(\"{func_name} hooked\");\n"
    func_preload_builder += "\tIOX_CLIENT_ACQUIRE_LOCK;\n"

    if is_get_param_func(func_name):
        group = get_param_group(func_name)
        indices = PARAM_INDICES[group]
        res_struct = f"{func_name}Response"

        func_preload_builder += "\tTALLY_CLIENT_PROFILE_START;\n"
        func_preload_builder += "#if defined(RUN_LOCALLY)\n"
        func_preload_builder += f"\tauto err = l{func_name}({arg_names_str});\n"

        func_preload_builder += "#else\n"

        if ret_type == "cublasStatus_t":
            func_preload_builder += "\tif (REPLACE_CUBLAS) {\n"
            func_preload_builder += "\t\tthrow std::runtime_error(std::string(__FILE__) + \":\" + std::to_string(__LINE__) + \": cublas function is not handled when REPLACE_CUBLAS is set.\");\n"
            func_preload_builder += "\t}\n"

        func_preload_builder += get_preload_func_template_iox(func_name, arg_names, arg_types, ret_type)
        func_preload_builder += f"""
    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {{
            auto response = static_cast<const {func_name}Response*>(responsePayload);
"""

        for idx in indices:
            func_preload_builder += f"\t\t\tif ({arg_names[idx]}) {{ *{arg_names[idx]} = response->{arg_names[idx]}; }}\n"

        func_preload_builder += """
            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
"""
        func_preload_builder += "#endif\n"

        if ret_type == "cudaError_t":
            func_preload_builder += f"\tLAST_CUDA_ERR = err;\n"

        func_preload_builder += f"\tTALLY_CLIENT_PROFILE_END;\n"
        func_preload_builder += "\treturn err;\n"
    
    elif func_name in FORWARD_API_CALLS:
        func_preload_builder += "\tTALLY_CLIENT_PROFILE_START;\n"
        func_preload_builder += "#if defined(RUN_LOCALLY)\n"
        func_preload_builder += f"\tauto err = l{func_name}({arg_names_str});\n"

        func_preload_builder += "#else\n"

        if ret_type == "cublasStatus_t":
            func_preload_builder += "\tif (REPLACE_CUBLAS) {\n"
            func_preload_builder += "\t\tthrow std::runtime_error(std::string(__FILE__) + \":\" + std::to_string(__LINE__) + \": cublas function is not handled when REPLACE_CUBLAS is set.\");\n"
            func_preload_builder += "\t};"

        func_preload_builder += get_preload_func_template_iox(func_name, arg_names, arg_types, ret_type)

        func_preload_builder += f"""
    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {{
            
            auto response = static_cast<const {ret_type}*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }}))
    {{}};
"""
        func_preload_builder += "#endif\n"

        if ret_type == "cudaError_t":
            func_preload_builder += f"\tLAST_CUDA_ERR = err;\n"
            
        func_preload_builder += f"\tTALLY_CLIENT_PROFILE_END;\n"

        func_preload_builder += "\treturn err;\n"

    elif func_name in IGNORE_CALLS:
        func_preload_builder += "#if defined(RUN_LOCALLY)\n"
        func_preload_builder += f"\treturn {preload_func_name}({arg_names_str});\n"
        func_preload_builder += "#else\n"
        func_preload_builder += f"\treturn ({ret_type}) 0;\n"
        func_preload_builder += "#endif\n"
    else:
        func_preload_builder += "#if defined(RUN_LOCALLY)\n"
        func_preload_builder += f"\treturn {preload_func_name}({arg_names_str});\n"
        func_preload_builder += "#else\n"
        func_preload_builder += f"\tthrow std::runtime_error(std::string(__FILE__) + \":\" + std::to_string(__LINE__) + \": Unimplemented.\");\n"
        func_preload_builder += "#endif\n"

    # close bracket
    func_preload_builder += "}"

    return func_preload_builder


def gen_client_code_from_file(file):
    func_sig_list = generate_func_sig_from_file(file)
    client_code_dict = {}

    for func_sig in func_sig_list:
        try:
            func_name = get_func_name_from_sig(func_sig)
            if func_name:
                func_preload = gen_func_client_preload(func_sig)
                func_declaration, func_definition = gen_client_func_decl_def(func_sig)
                msg_struct = gen_client_msg_struct(func_sig)
                server_handler = gen_server_handler(func_sig)

                client_code_dict[func_name] = {
                    "decl": func_declaration,
                    "def": func_definition,
                    "preload": func_preload,
                    "struct": msg_struct,
                    "handler": server_handler
                }

        except Exception as e:
            print(f"func_sig: {func_sig}")
            raise e
        
    return client_code_dict

def gen_client_code(header_files=CUDA_API_HEADER_FILES, client_preload_output_file="tally_client.cpp",
                    decl_output_file="cuda_api.h", def_output_file="cuda_api.cpp",
                    enum_output_file="cuda_api_enum.h", msg_struct_output_file="msg_struct.h",
                    server_header_output_file="server.h", server_cpp_output_file="server.cpp"):
    client_code_dict = {}

    for header_file in header_files:

        preprocess_header_file(header_file, output_file="/tmp/gcc_output.txt")
        client_code_dict.update(gen_client_code_from_file("/tmp/gcc_output.txt"))

    func_names = sorted(list(set(list(client_code_dict.keys()))))

    with open(client_preload_output_file, "w") as f:

        f.write(CLIENT_PRELOAD_TEMPLATE)
        f.write(EXTERN_C_BEGIN)
        for func_name in func_names:
            if func_name not in SPECIAL_CLIENT_PRELOAD_FUNCS and client_code_dict[func_name]["preload"]:
                f.write(client_code_dict[func_name]["preload"])
                f.write("\n\n")
        f.write(EXTERN_C_END)
    
    with open(decl_output_file, "w") as f:

        f.write(API_DECL_TEMPLATE_TOP)
        for func_name in func_names:
            if client_code_dict[func_name]["decl"]:
                f.write(client_code_dict[func_name]["decl"])
        f.write(API_DECL_TEMPLATE_BUTTOM)
    
    with open(def_output_file, "w") as f:

        f.write(API_DEF_TEMPLATE_TOP)
        for func_name in func_names:
            if client_code_dict[func_name]["def"]:
                f.write(client_code_dict[func_name]["def"])
        f.write(API_DEF_TEMPLATE_BUTTOM)

    with open(enum_output_file, 'w') as f:
        f.write(API_ENUM_TEMPLATE_TOP)
        f.write("\n\nenum CUDA_API_ENUM {\n")

        for func_name in client_code_dict:
            f.write(f"\t{func_name.upper()},\n")
        
        for idx, func_name in enumerate(API_SPECIAL_ENUM):
            if idx != len(API_SPECIAL_ENUM) - 1:
                f.write(f"\t{func_name.upper()},\n")
            else:
                f.write(f"\t{func_name.upper()}\n")

        f.write("};\n")
        f.write(API_ENUM_TEMPLATE_BUTTOM)
    
    with open(msg_struct_output_file, 'w') as f:
        f.write(MSG_STRUCT_TEMPLATE_TOP)
    
        for func_name in func_names:
            if client_code_dict[func_name]['struct']:
                f.write(f"{client_code_dict[func_name]['struct']}\n")

        f.write(MSG_STRUCT_TEMPLATE_BUTTOM)
    
    with open(server_header_output_file, 'w') as f:
        f.write(TALLY_SERVER_HEADER_TEMPLATE_TOP)

        for func_name in func_names + REGISTER_FUNCS:
            f.write(f"\tvoid handle_{func_name}(void *args, iox::popo::UntypedServer *iox_server, const void* const requestPayload);\n")

        f.write(TALLY_SERVER_HEADER_TEMPLATE_BUTTOM)

    with open(server_cpp_output_file, 'w') as f:
        f.write("""
#include <cstring>
                
#include "spdlog/spdlog.h"

#include <tally/util.h>
#include <tally/ipc_util.h>
#include <tally/msg_struct.h>
#include <tally/generated/cuda_api.h>
#include <tally/generated/msg_struct.h>
#include <tally/generated/server.h>
        
""")

        f.write("void TallyServer::register_api_handler() {\n")

        for func_name in func_names + SPECIAL_CLIENT_PRELOAD_FUNCS:
            f.write(f"\tcuda_api_handler_map[CUDA_API_ENUM::{func_name.upper()}] = std::bind(&TallyServer::handle_{func_name}, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);\n")

        f.write("}\n")

        for func_name in func_names:
            if func_name not in SPECIAL_CLIENT_PRELOAD_FUNCS:
                f.write(client_code_dict[func_name]["handler"])
        