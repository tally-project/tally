
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

#include <tally/cuda_api.h>
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

# These api calls can be directly forwarded to the server without addtional logic
# this means no value needs to be assigned
FORWARD_API_CALLS = [

]

CLIENT_PRELOAD_TEMPLATE = """
#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
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
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>
#include <thread>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda.h>
#include <fatbinary_section.h>

#include "libipc/ipc.h"

#include "tally/util.h"
#include "tally/msg_struct.h"
#include "tally/cuda_api.h"
#include "tally/kernel_slice.h"
#include "tally/cuda_api_enum.h"

class Preload {

public:

    std::map<std::string, std::vector<uint32_t>> _kernel_name_to_args;
    std::map<const void *, std::vector<uint32_t>> _kernel_addr_to_args;
    std::map<const void *, std::string> _kernel_map;
    ipc::channel *send_ipc;
    ipc::channel *recv_ipc;

    Preload()
    {
        send_ipc = new ipc::channel("client-to-server", ipc::sender);
        recv_ipc = new ipc::channel("server-to-client", ipc::receiver);
    }

    ~Preload()
    {
        if (send_ipc != nullptr) send_ipc->disconnect();
        if (recv_ipc != nullptr) recv_ipc->disconnect();
    }
};

Preload tracer;
"""

SPECIAL_CLIENT_PRELOAD_FUNCS = {
    "cudaMalloc": """
cudaError_t cudaMalloc(void ** devPtr, size_t  size)
{
    static const uint32_t msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMallocArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAMALLOC;
    
    struct cudaMallocArg *arg_ptr = (struct cudaMallocArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->devPtr = devPtr;
    arg_ptr->size = size;

    while (!tracer.send_ipc->send(msg, msg_len)) {
        tracer.send_ipc->wait_for_recv(1);
    }
    std::free(msg);

    ipc::buff_t buf;
    while (buf.empty()) {
        buf = tracer.recv_ipc->recv(1000);
    }

    const char *dat = buf.get<const char *>();
    struct cudaMallocResponse *res = (struct cudaMallocResponse *) dat;

    *devPtr = res->ptr;
	return res->err;
}
""", 
    "cudaMemcpy": """
cudaError_t cudaMemcpy(void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind)
{
    uint32_t msg_len;
    uint8_t *msg;
    
    if (kind == cudaMemcpyHostToDevice) {
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyArg) + count;
    } else if (kind == cudaMemcpyDeviceToHost){
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyArg);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAMEMCPY;
    
    struct cudaMemcpyArg *arg_ptr = (struct cudaMemcpyArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->dst = dst;
    arg_ptr->src = (void *)src;
    arg_ptr->count = count;
    arg_ptr->kind = kind;

    // Copy data to the message
    if (kind == cudaMemcpyHostToDevice) {
        memcpy(arg_ptr->data, src, count);
    }

    while (!tracer.send_ipc->send(msg, msg_len)) {
        tracer.send_ipc->wait_for_recv(1);
    }
    std::free(msg);

    ipc::buff_t buf;
    while (buf.empty()) {
        buf = tracer.recv_ipc->recv(1000);
    }

    const char *dat = buf.get<const char *>();

    struct cudaMemcpyResponse *res = (struct cudaMemcpyResponse *) dat;

    // Copy data to the host ptr
    if (kind == cudaMemcpyDeviceToHost) {
        memcpy(dst, res->data, count);
    }

	return res->err;
}
""", 
    "cudaLaunchKernel": """
cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    auto &params_info = tracer._kernel_addr_to_args[func];
    uint32_t params_size =  std::accumulate(params_info.begin(), params_info.end(), 0);

    size_t offset = 0;
    char params_data[params_size];

    for (size_t i = 0; i < params_info.size(); i++) {
        memcpy(params_data + offset, args[i], params_info[i]);
        offset += params_info[i];
    }

    uint32_t msg_len = sizeof(CUDA_API_ENUM) + sizeof(struct cudaLaunchKernelArg) + params_size;
    uint8_t *msg;

    msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDALAUNCHKERNEL;

    struct cudaLaunchKernelArg *arg_ptr = (struct cudaLaunchKernelArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->host_func = func;
    arg_ptr->gridDim = gridDim;
    arg_ptr->blockDim = blockDim;
    arg_ptr->sharedMem = sharedMem;
    memcpy(arg_ptr->params, params_data, params_size);

    while (!tracer.send_ipc->send(msg, msg_len)) {
        tracer.send_ipc->wait_for_recv(1);
    }
    std::free(msg);

    ipc::buff_t buf;
    while (buf.empty()) {
        buf = tracer.recv_ipc->recv(1000);
    }

    const char *dat = buf.get<const char *>();
    cudaError_t *err = (cudaError_t *) dat;

    return *err;
}
""", "__cudaRegisterFunction": """
void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    std::string deviceFunName (deviceFun);
    uint32_t kernel_func_len = deviceFunName.size();

    uint32_t msg_len = sizeof(CUDA_API_ENUM) + sizeof(struct registerKernelArg) + kernel_func_len * sizeof(char);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::__CUDAREGISTERFUNCTION;

    struct registerKernelArg *arg_ptr = (struct registerKernelArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->host_func = (void*) hostFun;
    arg_ptr->kernel_func_len = kernel_func_len;
    memcpy(arg_ptr->data, deviceFun, kernel_func_len * sizeof(char));

    while (!tracer.send_ipc->send(msg, msg_len)) {
        tracer.send_ipc->wait_for_recv(1);
    }
    std::free(msg);

    tracer._kernel_addr_to_args[hostFun] = tracer._kernel_name_to_args[deviceFunName];
}
""", 
    "__cudaRegisterFatBinary": """
void** __cudaRegisterFatBinary( void *fatCubin ) {
    __fatBinC_Wrapper_t *wp = (__fatBinC_Wrapper_t *) fatCubin;

    int magic = wp->magic;
    int version = wp->version;

    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) wp->data;
    size_t fatCubin_data_size_bytes = fbh->headerSize + fbh->fatSize;
    uint32_t msg_len = sizeof(CUDA_API_ENUM) + sizeof(struct __cudaRegisterFatBinaryArg) + fatCubin_data_size_bytes;

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::__CUDAREGISTERFATBINARY;

    struct __cudaRegisterFatBinaryArg *arg_ptr = (struct __cudaRegisterFatBinaryArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->magic = magic;
    arg_ptr->version = version;
    memcpy(arg_ptr->data, wp->data, fatCubin_data_size_bytes);

    while (!tracer.send_ipc->send(msg, msg_len)) {
        tracer.send_ipc->wait_for_recv(1);
    }
    std::free(msg);

    write_binary_to_file("/tmp/tmp.cubin", reinterpret_cast<const char*>(wp->data), fatCubin_data_size_bytes);
    exec("cuobjdump /tmp/tmp.cubin -elf > /tmp/tmp_cubin.elf");
    
    std::string elf_filename = "/tmp/tmp_cubin.elf";
    auto kernel_names_and_param_sizes = get_kernel_names_and_param_sizes_from_elf(elf_filename);

    for (auto &pair : kernel_names_and_param_sizes) {
        auto &kernel_name = pair.first;
        auto &param_sizes = pair.second;
        tracer._kernel_name_to_args[kernel_name] = param_sizes;
    }

    return nullptr;
}
""",
    "__cudaRegisterFatBinaryEnd": """
void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
    uint32_t msg_len = sizeof(CUDA_API_ENUM);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::__CUDAREGISTERFATBINARYEND;

    while (!tracer.send_ipc->send(msg, msg_len)) {
        tracer.send_ipc->wait_for_recv(1);
    }
    std::free(msg);
}
""", "cudaFree": """
cudaError_t cudaFree(void * devPtr)
{
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaFreeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAFREE;
    
    struct cudaFreeArg *arg_ptr = (struct cudaFreeArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->devPtr = devPtr;

    while (!tracer.send_ipc->send(msg, msg_len)) {
        tracer.send_ipc->wait_for_recv(1);
    }
    std::free(msg);

    ipc::buff_t buf;
    while (buf.empty()) {
        buf = tracer.recv_ipc->recv(1000);
    }

    const char *dat = buf.get<const char *>();
    cudaError_t *res = (cudaError_t *) dat;
	return *res;
}
"""
}