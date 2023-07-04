
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
#include <tally/const.h>

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

"""

MSG_STRUCT_TEMPLATE_BUTTOM = """

#endif // TALLY_GENERATED_MSG_STRUCT_H
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
    
]

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

#include "tally/msg_struct.h"
#include "tally/client.h"
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

#include <cuda_runtime.h>
#include <cuda.h>

#include "libipc/ipc.h"

#include <tally/msg_struct.h>

static std::function<void(int)> __exit;

static void __exit_wrapper(int signal) {
    __exit(signal);
}

class TallyServer {

public:

    static TallyServer *server;

    int magic;
    int version;
    unsigned long long* fatbin_data;
    uint32_t fatBinSize;

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

"""

TALLY_SERVER_HEADER_TEMPLATE_BUTTOM = """
    void handle___cudaRegisterFatBinary(void *args);
    void handle___cudaRegisterFunction(void *args);
    void handle___cudaRegisterFatBinaryEnd(void *args);
};

#endif // TALLY_SERVER_H
"""

SPECIAL_CLIENT_PRELOAD_FUNCS = [
    "cudaMalloc",
    "cudaMemcpy",
    "cudaLaunchKernel",
    "__cudaRegisterFunction",
    "__cudaRegisterFatBinary",
    "__cudaRegisterFatBinaryEnd"
]
