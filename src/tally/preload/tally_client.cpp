
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
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <fatbinary_section.h>

#include "libipc/ipc.h"

#include "tally/util.h"
#include "tally/ipc_util.h"
#include "tally/cache.h"
#include "tally/msg_struct.h"
#include "tally/generated/msg_struct.h"
#include "tally/transform.h"
#include "tally/client.h"
#include "tally/generated/cuda_api.h"
#include "tally/generated/cuda_api_enum.h"

cublasLtMatmulPreference_t preference = NULL;

extern "C" {

void *dlopen(const char *filename, int flag)
{
    static void* (*ldlopen) (const char *, int );
    if (!ldlopen) {
        ldlopen = (void* (*) (const char *, int  )) dlsym(RTLD_NEXT, "dlopen");
    }
    assert(ldlopen);

    if (filename) {
        std::string f_name(filename);
        if (f_name == "libcuda.so.1") {
            return ldlopen("/home/zhaowe58/tally/build/libtally_client.so", flag);
        }
    }

    return ldlopen(filename, flag);
}

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
        launch_shell(command);

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
    // printf("cudaMemcpy hooked\n");
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

cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
    // printf("cudaMemcpyAsync hooked\n");

	uint32_t msg_len;
    uint8_t *msg;
    
    if (kind == cudaMemcpyHostToDevice) {
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyAsyncArg) + count;
    } else if (kind == cudaMemcpyDeviceToHost){
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyAsyncArg);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    msg = (uint8_t *) std::malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAMEMCPYASYNC;

    auto arg_ptr = (struct cudaMemcpyAsyncArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->dst = dst;
    arg_ptr->src = (void *)src;
    arg_ptr->count = count;
    arg_ptr->kind = kind;
    arg_ptr->stream = stream;

    // Copy data to the message
    if (kind == cudaMemcpyHostToDevice) {
        memcpy(arg_ptr->data, src, count);
    }

    CLIENT_SEND_MSG_AND_FREE;
    CLIENT_RECV_MSG;

    auto res = (struct cudaMemcpyAsyncResponse *) dat;

    // Copy data to the host ptr
    if (kind == cudaMemcpyDeviceToHost) {
        memcpy(dst, res->data, count);
    }

	return res->err;
}

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    // printf("cudaLaunchKernel hooked\n");
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
    // printf("cudaMalloc hooked\n");
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

cublasStatus_t cublasSgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	// printf("cublasSgemm_v2 hooked\n");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSgemm_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASSGEMM_V2;
    
    struct cublasSgemm_v2Arg *arg_ptr = (struct cublasSgemm_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->transa = transa;
	arg_ptr->transb = transb;
	arg_ptr->m = m;
	arg_ptr->n = n;
	arg_ptr->k = k;
	arg_ptr->alpha = *alpha;
	arg_ptr->A = A;
	arg_ptr->lda = lda;
	arg_ptr->B = B;
	arg_ptr->ldb = ldb;
	arg_ptr->beta = *beta;
	arg_ptr->C = C;
	arg_ptr->ldc = ldc;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

// Warning: cublasLtMatmulAlgo_t may be a fake pointer
// when created by cublasLtMatmulAlgoInit
// At some point need to keep track which pointers are fake and which are real
cublasStatus_t cublasLtMatmul(cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  computeDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  B, cublasLtMatrixLayout_t  Bdesc, const void*  beta, const void*  C, cublasLtMatrixLayout_t  Cdesc, void*  D, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, void*  workspace, size_t  workspaceSizeInBytes, cudaStream_t  stream)
{
	// printf("cublasLtMatmul hooked\n");
	
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMUL;
    
    struct cublasLtMatmulArg *arg_ptr = (struct cublasLtMatmulArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->lightHandle = lightHandle;
    arg_ptr->computeDesc = computeDesc;
    arg_ptr->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
    arg_ptr->A = A;
    arg_ptr->Adesc = Adesc;
    arg_ptr->B = B;
    arg_ptr->Bdesc = Bdesc;
    arg_ptr->beta = *((uint64_t *) beta);
    arg_ptr->C = (void *)C;
    arg_ptr->Cdesc = Cdesc;
    arg_ptr->D = D;
    arg_ptr->Ddesc = Ddesc;
    memcpy(&(arg_ptr->algo), algo, sizeof(cublasLtMatmulAlgo_t));
    arg_ptr->workspace = workspace;
    arg_ptr->workspaceSizeInBytes = workspaceSizeInBytes;
    arg_ptr->stream = stream;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}


cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	// printf("cublasLtMatmulDescSetAttribute hooked\n");
	
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulDescSetAttributeArg) + sizeInBytes;

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULDESCSETATTRIBUTE;

    struct cublasLtMatmulDescSetAttributeArg *arg_ptr = (struct cublasLtMatmulDescSetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->matmulDesc = matmulDesc;
    arg_ptr->attr = attr;
    arg_ptr->sizeInBytes = sizeInBytes;
    memcpy(arg_ptr->buf, buf, sizeInBytes);

    CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t  matLayout, cublasLtMatrixLayoutAttribute_t  attr, const void*  buf, size_t  sizeInBytes)
{
	// printf("cublasLtMatrixLayoutSetAttribute hooked\n");
	
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatrixLayoutSetAttributeArg) + sizeInBytes;

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTSETATTRIBUTE;

    struct cublasLtMatrixLayoutSetAttributeArg *arg_ptr = (struct cublasLtMatrixLayoutSetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->matLayout = matLayout;
    arg_ptr->attr = attr;
    arg_ptr->sizeInBytes = sizeInBytes;
    memcpy(arg_ptr->buf, buf, sizeInBytes);

    CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t  pref, cublasLtMatmulPreferenceAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	// printf("cublasLtMatmulPreferenceSetAttribute hooked\n");
	
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulPreferenceSetAttributeArg) + sizeInBytes;

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCESETATTRIBUTE;

    struct cublasLtMatmulPreferenceSetAttributeArg *arg_ptr = (struct cublasLtMatmulPreferenceSetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pref = pref;
    arg_ptr->attr = attr;
    arg_ptr->sizeInBytes = sizeInBytes;
    memcpy(arg_ptr->buf, buf, sizeInBytes);

    CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, cublasLtMatmulPreference_t  preference, int  requestedAlgoCount, cublasLtMatmulHeuristicResult_t  heuristicResultsArray[], int*  returnAlgoCount)
{
	// printf("cublasLtMatmulAlgoGetHeuristic hooked\n");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulAlgoGetHeuristicArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULALGOGETHEURISTIC;

    struct cublasLtMatmulAlgoGetHeuristicArg *arg_ptr = (struct cublasLtMatmulAlgoGetHeuristicArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->lightHandle = lightHandle;
    arg_ptr->operationDesc = operationDesc;
    arg_ptr->Adesc = Adesc;
    arg_ptr->Bdesc = Bdesc;
    arg_ptr->Cdesc = Cdesc;
    arg_ptr->Ddesc = Ddesc;
    arg_ptr->preference = preference;
    arg_ptr->requestedAlgoCount = requestedAlgoCount;
    arg_ptr->heuristicResultsArray = heuristicResultsArray;

    CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	
    auto res = (cublasLtMatmulAlgoGetHeuristicResponse *) dat;
    *returnAlgoCount = res->returnAlgoCount;
    memcpy(heuristicResultsArray, res->heuristicResultsArray, sizeof(cublasLtMatmulHeuristicResult_t) * res->returnAlgoCount);

    return res->err;
}

const char* cudaGetErrorString(cudaError_t  error)
{
    // printf("cudaGetErrorString hooked\n");
    return lcudaGetErrorString(error);
}

CUresult cuGetProcAddress(const char * symbol, void ** pfn, int  cudaVersion, cuuint64_t  flags)
{
	// printf("cuGetProcAddress hooked\n");
	return lcuGetProcAddress(symbol, pfn, cudaVersion, flags);
}


}