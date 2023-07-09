
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
#include <string>
#include <cstring>
#include <numeric>
#include <thread>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda.h>
#include <fatbinary_section.h>

#include "libipc/ipc.h"

#include "tally/util.h"
#include "tally/ipc_util.h"
#include "tally/cache.h"
#include "tally/msg_struct.h"
#include "tally/transform.h"
#include "tally/client.h"
#include "tally/generated/cuda_api.h"
#include "tally/generated/cuda_api_enum.h"

extern "C" {

void** __cudaRegisterFatBinary( void *fatCubin ) {
    auto wp = (__fatBinC_Wrapper_t *) fatCubin;
    int magic = wp->magic;
    int version = wp->version;

    auto fbh = (struct fatBinaryHeader *) wp->data;
    const char *cubin_data = (const char *) wp->data;
    size_t cubin_size = fbh->headerSize + fbh->fatSize;

    bool cached = TallyCache::cache->cubin_cache.contains(cubin_data, cubin_size);
    uint32_t msg_len;

    if (!cached) {
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(struct __cudaRegisterFatBinaryArg) + cubin_size;
    } else {
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(struct __cudaRegisterFatBinaryArg);
    }

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::__CUDAREGISTERFATBINARY;

    auto arg_ptr = (struct __cudaRegisterFatBinaryArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->cached = cached;
    arg_ptr->magic = magic;
    arg_ptr->version = version;
    if (!cached) {
        memcpy(arg_ptr->data, wp->data, cubin_size);
    }

    CLIENT_SEND_MSG_AND_FREE;

    std::map<std::string, std::vector<uint32_t>> kernel_args;

    if (cached) {
        kernel_args = TallyCache::cache->cubin_cache.get_kernel_args(cubin_data, cubin_size);
    } else {
        auto tmp_cubin_file = get_tmp_file_path(".cubin");
        write_binary_to_file(tmp_cubin_file, cubin_data, cubin_size);
        auto tmp_elf_file = get_tmp_file_path(".elf");
        std::string command("cuobjdump " + tmp_cubin_file + " -elf > " + tmp_elf_file);
        exec(command);
        
        kernel_args = get_kernel_names_and_param_sizes_from_elf(tmp_elf_file);
    }

    for (auto &pair : kernel_args) {
        auto &kernel_name = pair.first;
        auto &param_sizes = pair.second;
        TallyClient::client->_kernel_name_to_args[kernel_name] = param_sizes;
    }

    return nullptr;
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    std::string deviceFunName (deviceFun);
    uint32_t kernel_func_len = deviceFunName.size();

    uint32_t msg_len = sizeof(CUDA_API_ENUM) + sizeof(struct registerKernelArg) + kernel_func_len * sizeof(char);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::__CUDAREGISTERFUNCTION;

    auto arg_ptr = (struct registerKernelArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->host_func = (void*) hostFun;
    arg_ptr->kernel_func_len = kernel_func_len;
    memcpy(arg_ptr->data, deviceFun, kernel_func_len * sizeof(char));

    CLIENT_SEND_MSG_AND_FREE;

    TallyClient::client->_kernel_addr_to_args[hostFun] = TallyClient::client->_kernel_name_to_args[deviceFunName];
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
    uint32_t msg_len = sizeof(CUDA_API_ENUM);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::__CUDAREGISTERFATBINARYEND;

    CLIENT_SEND_MSG_AND_FREE;
}

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
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAMEMCPY;
    
    auto arg_ptr = (struct cudaMemcpyArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->dst = dst;
    arg_ptr->src = (void *)src;
    arg_ptr->count = count;
    arg_ptr->kind = kind;

    // Copy data to the message
    if (kind == cudaMemcpyHostToDevice) {
        memcpy(arg_ptr->data, src, count);
    }

    CLIENT_SEND_MSG_AND_FREE;
    CLIENT_RECV_MSG;

    auto res = (struct cudaMemcpyResponse *) dat;

    // Copy data to the host ptr
    if (kind == cudaMemcpyDeviceToHost) {
        memcpy(dst, res->data, count);
    }

	return res->err;
}

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    auto &params_info = TallyClient::client->_kernel_addr_to_args[func];
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
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDALAUNCHKERNEL;

    auto arg_ptr = (struct cudaLaunchKernelArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->host_func = func;
    arg_ptr->gridDim = gridDim;
    arg_ptr->blockDim = blockDim;
    arg_ptr->sharedMem = sharedMem;
    arg_ptr->stream = stream;
    memcpy(arg_ptr->params, params_data, params_size);

    CLIENT_SEND_MSG_AND_FREE;
    CLIENT_RECV_MSG;

    auto err = (cudaError_t *) dat;
    return *err;
}

cudaError_t cudaMalloc(void ** devPtr, size_t  size)
{
    static const uint32_t msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMallocArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAMALLOC;
    
    auto arg_ptr = (struct cudaMallocArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->devPtr = devPtr;
    arg_ptr->size = size;

    CLIENT_SEND_MSG_AND_FREE;
    CLIENT_RECV_MSG;

    auto res = (struct cudaMallocResponse *) dat;

    *devPtr = res->ptr;
	return res->err;
}

}