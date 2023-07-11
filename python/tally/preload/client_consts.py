
API_ENUM_TEMPLATE_TOP = """
#ifndef TALLY_CUDA_API_ENUM_H
#define TALLY_CUDA_API_ENUM_H
"""

API_ENUM_TEMPLATE_BUTTOM = """
#endif // TALLY_CUDA_API_ENUM_H
"""

API_SPECIAL_ENUM = [
    "__CUDAREGISTERFUNCTION",
    "__CUDAREGISTERFATBINARY",
    "__CUDAREGISTERFATBINARYEND"
]

API_DECL_TEMPLATE_TOP = """

#ifndef TALLY_CUDA_API_H
#define TALLY_CUDA_API_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <nvrtc.h>
#include <cublasLt.h>

"""

API_DECL_TEMPLATE_BUTTOM = """

extern void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
extern void** (*l__cudaRegisterFatBinary) (void *);
extern void (*l__cudaRegisterFatBinaryEnd) (void **);

#endif // TALLY_CUDA_API_H

"""

API_DEF_TEMPLATE_TOP = """

#include <dlfcn.h>

#include <tally/generated/cuda_api.h>
#include <tally/env.h>

void *cuda_handle = dlopen(LIBCUDA_PATH, RTLD_LAZY);
void *cudart_handle = dlopen(LIBCUDART_PATH, RTLD_LAZY);
void *cudnn_handle = dlopen(LIBCUDNN_PATH, RTLD_LAZY);

"""

API_DEF_TEMPLATE_BUTTOM = """

void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)
    = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(cudart_handle, "__cudaRegisterFunction");

void** (*l__cudaRegisterFatBinary) (void *) = 
    (void** (*) (void *)) dlsym(cudart_handle, "__cudaRegisterFatBinary");

void (*l__cudaRegisterFatBinaryEnd) (void **) =
	(void (*) (void **)) dlsym(cudart_handle, "__cudaRegisterFatBinaryEnd");

"""

MSG_STRUCT_TEMPLATE_TOP = """
#ifndef TALLY_GENERATED_MSG_STRUCT_H
#define TALLY_GENERATED_MSG_STRUCT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <nvrtc.h>
#include <cublasLt.h>


"""

MSG_STRUCT_TEMPLATE_BUTTOM = """

#endif // TALLY_GENERATED_MSG_STRUCT_H
"""

CLIENT_PRELOAD_TEMPLATE = """
#include <dlfcn.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cxxabi.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <unistd.h>
#include <cstring>

#include "tally/cuda_util.h"
#include "tally/msg_struct.h"
#include "tally/client.h"
#include "tally/ipc_util.h"
#include "tally/generated/cuda_api.h"
#include "tally/generated/cuda_api_enum.h"
#include "tally/generated/msg_struct.h"

"""

TALLY_SERVER_HEADER_TEMPLATE_TOP = """
#ifndef TALLY_SERVER_H
#define TALLY_SERVER_H

#include <signal.h>
#include <string>
#include <atomic>
#include <map>
#include <iostream>
#include <functional>
#include <memory>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <nvrtc.h>
#include <cublasLt.h>

#include "libipc/ipc.h"

#include <tally/msg_struct.h>

static std::function<void(int)> __exit;

static void __exit_wrapper(int signal) {
    __exit(signal);
}

class TallyServer {

public:

    static std::unique_ptr<TallyServer> server;

    int magic;
    int version;
    unsigned long long* fatbin_data = nullptr;
    uint32_t fatBinSize;
    bool cubin_registered = false;

    std::atomic<bool> is_quit__ {false};
    ipc::channel *send_ipc = nullptr;
    ipc::channel *recv_ipc = nullptr;
    std::map<void *, std::vector<uint32_t>> _kernel_addr_to_args;
    std::map<std::string, void *> _kernel_name_to_addr;
    std::map<void *, void *> _kernel_client_addr_mapping;
    std::vector<std::pair<void *, std::string>> register_queue;
    std::unordered_map<CUDA_API_ENUM, std::function<void(void *)>> cuda_api_handler_map;

    TallyServer();

    void start(uint32_t interval);
    void register_api_handler();
    void load_cache();

"""

TALLY_SERVER_HEADER_TEMPLATE_BUTTOM = """
    void handle___cudaRegisterFatBinary(void *args);
    void handle___cudaRegisterFunction(void *args);
    void handle___cudaRegisterFatBinaryEnd(void *args);
};

#endif // TALLY_SERVER_H
"""

# These api calls can be directly forwarded to the server without addtional logic
# this means no value needs to be assigned
FORWARD_API_CALLS = [
    "cudaFree",
    "cudaProfilerStart",
    "cudaProfilerStop",
    "cuInit",
    "cudaDeviceReset",
    "cudaDeviceSynchronize",
    "cudaDeviceSetLimit",
    "cudaDeviceSetCacheConfig",
    "cudaDeviceSetSharedMemConfig",
    "cudaThreadExit",
    "cudaThreadSynchronize",
    "cudaThreadSetLimit",
    "cudaThreadSetCacheConfig",
    "cudaGetLastError",
    "cudaPeekAtLastError",
    "cudaSetDevice",
    "cudaSetDeviceFlags",
    "cudaCtxResetPersistingL2Cache",
    "cudaEventRecord",
    "cudaEventRecordWithFlags",
    "cudaEventQuery",
    "cudaEventSynchronize",
    "cudaEventDestroy",
    "cudaDeviceSetMemPool",
    "cudaSetDeviceFlags",
    "cudaStreamCopyAttributes",
    "cudaStreamDestroy",
    "cudaStreamWaitEvent",
    "cudaStreamSynchronize",
    "cudaStreamQuery",
    "cudaStreamBeginCapture",
    "cudaStreamEndCapture",
    "cublasDestroy_v2",
    "cublasGetCudartVersion",
    "cuDeviceSetMemPool",
    "cuFlushGPUDirectRDMAWrites",
    "cuDevicePrimaryCtxRelease_v2",
    "cuDevicePrimaryCtxSetFlags_v2",
    "cuDevicePrimaryCtxReset_v2",
    "cublasSetStream_v2",
    "cublasSetWorkspace_v2",
    "cublasSetMathMode",
    "cublasLtMatmulDescDestroy",
    "cublasLtMatrixLayoutDestroy",
    "cublasLtMatmulPreferenceDestroy",
    "cublasLtDestroy"
]

# implement manually
SPECIAL_CLIENT_PRELOAD_FUNCS = [
    "cudaMalloc",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaLaunchKernel",
    "cublasSgemm_v2",
    "cublasLtMatmulDescSetAttribute",
    "cublasLtMatrixLayoutSetAttribute",
    "cublasLtMatmulPreferenceSetAttribute",
    "cublasLtMatmulAlgoGetHeuristic",
    "cublasLtMatmul",
    "cudaGetErrorString",
    "cuGetProcAddress",
    "__cudaRegisterFunction",
    "__cudaRegisterFatBinary",
    "__cudaRegisterFatBinaryEnd"
]

# API calls that has the first argument set
# by CUDA API call, such as cudaStreamCreate
CUDA_GET_1_PARAM_FUNCS = [
    "cudaStreamCreate",
    "cudaStreamCreateWithFlags",
    "cudaStreamCreateWithPriority",
    "cudaEventCreate",
    "cudaEventCreateWithFlags",
    "cudaEventElapsedTime",
    "cudaDeviceGetLimit",
    "cudaDeviceGetCacheConfig",
    "cudaDeviceGetSharedMemConfig"
    "cudaIpcGetEventHandle",
    "cudaIpcOpenEventHandle",
    "cudaThreadGetLimit",
    "cudaThreadGetCacheConfig",
    "cudaGetDeviceCount",
    "cudaGetDeviceProperties",
    "cudaDeviceGetAttribute",
    "cudaDeviceGetDefaultMemPool",
    "cudaDeviceGetMemPool",
    "cudaDeviceGetP2PAttribute",
    "cudaGetDevice",
    "cudaGetDeviceFlags",
    "cudaGraphCreate",
    "cublasCreate_v2",
    "cuDriverGetVersion",
    "cuDeviceGet",
    "cuDeviceGetCount",
    "cuDeviceGetUuid",
    "cuDeviceGetUuid_v2",
    "cuDeviceTotalMem_v2",
    "cuDeviceGetAttribute",
    "cuDeviceGetMemPool",
    "cuDeviceGetDefaultMemPool",
    "cuDeviceGetProperties",
    "cuDevicePrimaryCtxRetain",
    "cublasLtMatmulDescCreate",
    "cublasLtMatrixLayoutCreate",
    "cublasLtMatmulPreferenceCreate",
    "cublasLtCreate"
]

CUDA_GET_2_PARAM_FUNCS = [
    "cudaStreamIsCapturing",
    "cublasGetVersion_v2"
]

CUDA_GET_2_3_PARAM_FUNCS = [
    "cuDevicePrimaryCtxGetState"
]

CUDA_GET_1_2_PARAM_FUNCS = [
    "cudaDeviceGetStreamPriorityRange"
]

CUDA_GET_1_PARAM_FUNC_KEY = 1
CUDA_GET_2_3_PARAM_FUNC_KEY = 2
CUDA_GET_1_2_PARAM_FUNC_KEY = 3
CUDA_GET_2_PARAM_FUNC_KEY = 4

PARAM_INDICES = {
    CUDA_GET_1_PARAM_FUNC_KEY: [0],
    CUDA_GET_2_PARAM_FUNC_KEY: [1],
    CUDA_GET_2_3_PARAM_FUNC_KEY: [1, 2],
    CUDA_GET_1_2_PARAM_FUNC_KEY: [0, 1]
}

def is_get_param_func(func_name):
    if (func_name in CUDA_GET_1_PARAM_FUNCS or
        func_name in CUDA_GET_2_3_PARAM_FUNCS or
        func_name in CUDA_GET_1_2_PARAM_FUNCS or
        func_name in CUDA_GET_2_PARAM_FUNCS):
        return True
    return False

def get_param_group(func_name):
    if func_name in CUDA_GET_1_PARAM_FUNCS:
        return CUDA_GET_1_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_2_3_PARAM_FUNCS:
        return CUDA_GET_2_3_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_1_2_PARAM_FUNCS:
        return CUDA_GET_1_2_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_2_PARAM_FUNCS:
        return CUDA_GET_2_PARAM_FUNC_KEY
    else:
        assert(False)

def get_preload_func_template(func_name, arg_names):
    arg_struct = f"{func_name}Arg"

    preload_body = ""
    preload_body += f"""
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct {arg_struct});

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::{func_name.upper()};
    
    struct {arg_struct} *arg_ptr = (struct {arg_struct} *)(msg + sizeof(CUDA_API_ENUM));
"""

    for arg_name in arg_names:
        preload_body += f"\targ_ptr->{arg_name} = {arg_name};\n"
    
    preload_body += "\tCLIENT_SEND_MSG_AND_FREE;\n"
    preload_body += "\tCLIENT_RECV_MSG;\n"

    return preload_body