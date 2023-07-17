#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
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
#include <unordered_set>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <fatbinary_section.h>

#include "libipc/ipc.h"

#include "tally/log.h"
#include "tally/util.h"
#include "tally/cuda_util.h"
#include "tally/ipc_util.h"
#include "tally/cache.h"
#include "tally/msg_struct.h"
#include "tally/generated/msg_struct.h"
#include "tally/transform.h"
#include "tally/client.h"
#include "tally/generated/cuda_api.h"
#include "tally/generated/cuda_api_enum.h"

// Used to keep track of seq length of a seq description
std::unordered_map<cudnnSeqDataDescriptor_t, int> seq_desc_to_seq_len_map;

// Used to check whether an address points to device memory
std::vector<DeviceMemoryKey> dev_addr_map;

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

#ifdef RUN_LOCALLY
    return l__cudaRegisterFatBinary(fatCubin);
#else
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

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
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
#endif
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    std::string deviceFunName (deviceFun);
    TallyClient::client->host_func_to_demangled_kernel_name_map[hostFun] = demangleFunc(deviceFunName);

#ifdef RUN_LOCALLY
    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
#else
    uint32_t kernel_func_len = deviceFunName.size();

    uint32_t msg_len = sizeof(CUDA_API_ENUM) + sizeof(struct registerKernelArg) + kernel_func_len * sizeof(char);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::__CUDAREGISTERFUNCTION;

    auto arg_ptr = (struct registerKernelArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->host_func = (void*) hostFun;
    arg_ptr->kernel_func_len = kernel_func_len;
    memcpy(arg_ptr->data, deviceFun, kernel_func_len * sizeof(char));

    CLIENT_SEND_MSG_AND_FREE;

    TallyClient::client->_kernel_addr_to_args[hostFun] = TallyClient::client->_kernel_name_to_args[deviceFunName];
#endif
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
#ifdef RUN_LOCALLY
    l__cudaRegisterFatBinaryEnd(fatCubinHandle);

    // Just to warm up
    int *arr;
    lcudaMalloc((void**)&arr, sizeof(int));
    lcudaFree(arr);
#else
    uint32_t msg_len = sizeof(CUDA_API_ENUM);
    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::__CUDAREGISTERFATBINARYEND;

    CLIENT_SEND_MSG_AND_FREE;
#endif
}

cudaError_t cudaMalloc(void ** devPtr, size_t  size)
{
	TALLY_LOG("cudaMalloc hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudaMalloc(devPtr, size);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaMallocArg);
    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);

    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAMALLOC;
    
    struct cudaMallocArg *arg_ptr = (struct cudaMallocArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->devPtr = devPtr;
	arg_ptr->size = size;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

	auto res = (cudaMallocResponse *) dat;
	if (devPtr) { *devPtr = res->devPtr; }

    if (res->err == cudaSuccess) {
        // Keep track that this addr is device memory
        dev_addr_map.push_back( DeviceMemoryKey(res->devPtr, size) );
    }

    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaMalloc);

    return err;
}

cudaError_t cudaFree(void * devPtr)
{
	TALLY_LOG("cudaFree hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudaFree(devPtr);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaFreeArg);
    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);

    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAFREE;
    
    struct cudaFreeArg *arg_ptr = (struct cudaFreeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->devPtr = devPtr;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    if (*res == cudaSuccess) {
        free_dev_addr(dev_addr_map, devPtr);
    }

    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaFree);

    return err;
}

cudaError_t cudaMemcpy(void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind)
{
    TALLY_LOG("cudaMemcpy hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudaMemcpy(dst, src, count, kind);
#else

    uint32_t msg_len;
    uint8_t *msg;
    
    if (kind == cudaMemcpyHostToDevice) {
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyArg) + count;
    } else if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice){
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
    
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaMemcpy);

    return err;
}

cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
    TALLY_LOG("cudaMemcpyAsync hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudaMemcpyAsync(dst, src, count, kind, stream);
#else
	uint32_t msg_len;
    uint8_t *msg;
    
    if (kind == cudaMemcpyHostToDevice) {
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyAsyncArg) + count;
    } else if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice){
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

    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaMemcpyAsync);

    return err;
}

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    TALLY_LOG("cudaLaunchKernel hooked");
    TALLY_CLIENT_PROFILE_START;
    
#ifdef RUN_LOCALLY
    auto err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
#else
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

    auto res = (cudaError_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_KERNEL_CALL(func);

    return err;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSgemm_v2 hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSgemm_v2Arg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
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
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasSgemm_v2);
    return err;
}

// Warning: cublasLtMatmulAlgo_t may be a fake pointer
// when created by cublasLtMatmulAlgoInit
// At some point need to keep track which pointers are fake and which are real
cublasStatus_t cublasLtMatmul(cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  computeDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  B, cublasLtMatrixLayout_t  Bdesc, const void*  beta, const void*  C, cublasLtMatrixLayout_t  Cdesc, void*  D, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, void*  workspace, size_t  workspaceSizeInBytes, cudaStream_t  stream)
{
	TALLY_LOG("cublasLtMatmul hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
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
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmul);

    return err;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_LOG("cublasLtMatmulDescSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcublasLtMatmulDescSetAttribute(matmulDesc, attr, buf, sizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulDescSetAttributeArg) + sizeInBytes;

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULDESCSETATTRIBUTE;

    struct cublasLtMatmulDescSetAttributeArg *arg_ptr = (struct cublasLtMatmulDescSetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->matmulDesc = matmulDesc;
    arg_ptr->attr = attr;
    arg_ptr->sizeInBytes = sizeInBytes;
    memcpy(arg_ptr->buf, buf, sizeInBytes);

    CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulDescSetAttribute);

    return err;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t  matLayout, cublasLtMatrixLayoutAttribute_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_LOG("cublasLtMatrixLayoutSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;
	
#ifdef RUN_LOCALLY
    auto err = lcublasLtMatrixLayoutSetAttribute(matLayout, attr, buf, sizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatrixLayoutSetAttributeArg) + sizeInBytes;

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTSETATTRIBUTE;

    struct cublasLtMatrixLayoutSetAttributeArg *arg_ptr = (struct cublasLtMatrixLayoutSetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->matLayout = matLayout;
    arg_ptr->attr = attr;
    arg_ptr->sizeInBytes = sizeInBytes;
    memcpy(arg_ptr->buf, buf, sizeInBytes);

    CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatrixLayoutSetAttribute);
    return err;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t  pref, cublasLtMatmulPreferenceAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_LOG("cublasLtMatmulPreferenceSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcublasLtMatmulPreferenceSetAttribute(pref, attr, buf, sizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulPreferenceSetAttributeArg) + sizeInBytes;

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCESETATTRIBUTE;

    struct cublasLtMatmulPreferenceSetAttributeArg *arg_ptr = (struct cublasLtMatmulPreferenceSetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pref = pref;
    arg_ptr->attr = attr;
    arg_ptr->sizeInBytes = sizeInBytes;
    memcpy(arg_ptr->buf, buf, sizeInBytes);

    CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulPreferenceSetAttribute);
    return err;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, cublasLtMatmulPreference_t  preference, int  requestedAlgoCount, cublasLtMatmulHeuristicResult_t  heuristicResultsArray[], int*  returnAlgoCount)
{
	TALLY_LOG("cublasLtMatmulAlgoGetHeuristic hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcublasLtMatmulAlgoGetHeuristic(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulAlgoGetHeuristicArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
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
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulAlgoGetHeuristic);
    return err;
}

cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  elementCount, const void * arrayOfElements)
{
	TALLY_LOG("cudnnBackendSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnBackendSetAttribute(descriptor, attributeName, attributeType, elementCount, arrayOfElements);
#else
    int32_t type_size = get_cudnn_attribute_size(attributeType);
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBackendSetAttributeArg) + elementCount * type_size;

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBACKENDSETATTRIBUTE;

    auto arg_ptr = (struct cudnnBackendSetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
    arg_ptr->descriptor = descriptor;
    arg_ptr->attributeName = attributeName;
    arg_ptr->attributeType = attributeType;
    arg_ptr->elementCount = elementCount;

    assert(arrayOfElements);
    memcpy(arg_ptr->arrayOfElements, arrayOfElements, type_size * elementCount);

    // print_arrayOfElements(attributeType, elementCount, arrayOfElements);

    if (attributeType == CUDNN_TYPE_VOID_PTR) {
        auto pointer_arr = (void **) (arg_ptr->arrayOfElements);

        for (int i = 0; i < elementCount; i++) {
            auto pointer = pointer_arr[i];

            if (pointer == nullptr) {
                continue;
            }

            auto found = is_dev_addr(dev_addr_map, pointer);

            // pointer points to CPU memory
            if (!found) {

                // Get the value from the CPU pointers
                uint64_t val = *((uint64_t *) pointer);

                // Store the value instead of addr
                pointer_arr[i] = (void *) val;
            }
        }
    }

    CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	
    auto res = (struct cudnnBackendSetAttributeResponse *) dat;
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnBackendSetAttribute);
    return err;
}

cudnnStatus_t cudnnBackendGetAttribute(cudnnBackendDescriptor_t const  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  requestedElementCount, int64_t * elementCount, void * arrayOfElements)
{
	TALLY_LOG("cudnnBackendGetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnBackendGetAttribute(descriptor, attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);
#else
    int32_t type_size = get_cudnn_attribute_size(attributeType);
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBackendGetAttributeArg) + requestedElementCount * type_size;

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBACKENDGETATTRIBUTE;

    struct cudnnBackendGetAttributeArg *arg_ptr = (struct cudnnBackendGetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->descriptor = descriptor;
    arg_ptr->attributeName = attributeName;
    arg_ptr->attributeType = attributeType;
    arg_ptr->requestedElementCount = requestedElementCount;
    arg_ptr->elementCount = elementCount;
    arg_ptr->arrayOfElements = arrayOfElements;
    if (arrayOfElements) {
        memcpy(arg_ptr->arrayOfElementsData, arrayOfElements, requestedElementCount * type_size);
    }

    assert(arg_ptr->requestedElementCount >= 0);

    CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	
    auto res = (cudnnBackendGetAttributeResponse *) dat;
    if (elementCount) {
        *elementCount = res->elementCount;
    }

    if (arrayOfElements) {
        memcpy(arrayOfElements, res->arrayOfElements, type_size * res->arrayOfElementsSize);
    }
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnBackendGetAttribute);
    return err;
}

cudnnStatus_t cudnnActivationForward(cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnActivationForward hooked");
    TALLY_CLIENT_PROFILE_START;
	
#ifdef RUN_LOCALLY
    auto err = lcudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
#else 
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnActivationForwardArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNACTIVATIONFORWARD;
    
    auto arg_ptr = (struct cudnnActivationForwardArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
	arg_ptr->activationDesc = activationDesc;
	arg_ptr->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
	arg_ptr->xDesc = xDesc;
	arg_ptr->x = const_cast<void*>(x);
	arg_ptr->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
	arg_ptr->yDesc = yDesc;
	arg_ptr->y = y;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err= *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnActivationForward);
    return err;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const int  strideA[])
{
    TALLY_LOG("cudnnSetTensorNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetTensorNdDescriptorArg) + 2 * nbDims * sizeof(int);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETTENSORNDDESCRIPTOR;
    
    auto arg_ptr = (struct cudnnSetTensorNdDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->tensorDesc = tensorDesc;
    arg_ptr->dataType = dataType;
    arg_ptr->nbDims = nbDims;
    memcpy(arg_ptr->dimA_and_strideA, dimA, sizeof(int) * nbDims);
    memcpy(arg_ptr->dimA_and_strideA + nbDims, strideA, sizeof(int) * nbDims);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetTensorNdDescriptor);
    return err;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t  convDesc, int  arrayLength, const int  padA[], const int  filterStrideA[], const int  dilationA[], cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType)
{
	TALLY_LOG("cudnnSetConvolutionNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetConvolutionNdDescriptorArg) + 3 * arrayLength * sizeof(int);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETCONVOLUTIONNDDESCRIPTOR;
    
    auto arg_ptr = (struct cudnnSetConvolutionNdDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->convDesc = convDesc;
    arg_ptr->arrayLength = arrayLength;
    arg_ptr->mode = mode;
    arg_ptr->computeType = computeType;
    memcpy(arg_ptr->padA_and_filterStrideA_and_dilationA, padA, sizeof(int) * arrayLength);
    memcpy(arg_ptr->padA_and_filterStrideA_and_dilationA + arrayLength, filterStrideA, sizeof(int) * arrayLength);
    memcpy(arg_ptr->padA_and_filterStrideA_and_dilationA + 2 * arrayLength, dilationA, sizeof(int) * arrayLength);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetConvolutionNdDescriptor);
    return err;
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  nbDims, const int  filterDimA[])
{
	TALLY_LOG("cudnnSetFilterNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;
	
#ifdef RUN_LOCALLY
    auto err = lcudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);;
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetFilterNdDescriptorArg) + nbDims * sizeof(int);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETFILTERNDDESCRIPTOR;
    
    auto arg_ptr = (struct cudnnSetFilterNdDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->filterDesc = filterDesc;
    arg_ptr->dataType = dataType;
    arg_ptr->format = format;
    arg_ptr->nbDims = nbDims;
    memcpy(arg_ptr->filterDimA, filterDimA, sizeof(int) * nbDims);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetFilterNdDescriptor);
    return err;
}

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnConvolutionForward hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnConvolutionForwardArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCONVOLUTIONFORWARD;
    
    auto arg_ptr = (struct cudnnConvolutionForwardArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
	arg_ptr->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
	arg_ptr->xDesc = xDesc;
	arg_ptr->x = const_cast<void*>(x);
    arg_ptr->wDesc = wDesc;
    arg_ptr->w = const_cast<void*>(w);
    arg_ptr->convDesc = convDesc;
    arg_ptr->algo = algo;
    arg_ptr->workSpace = workSpace;
    arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
	arg_ptr->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
	arg_ptr->yDesc = yDesc;
	arg_ptr->y = y;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnConvolutionForward);
    return err;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int  nbDims, int  tensorOuputDimA[])
{
	TALLY_LOG("cudnnGetConvolutionNdForwardOutputDim hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetConvolutionNdForwardOutputDimArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONNDFORWARDOUTPUTDIM;
    
    auto arg_ptr = (struct cudnnGetConvolutionNdForwardOutputDimArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->convDesc = convDesc;
    arg_ptr->inputTensorDesc = inputTensorDesc;
    arg_ptr->filterDesc = filterDesc;
    arg_ptr->nbDims = nbDims;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnGetConvolutionNdForwardOutputDimResponse *) dat;
    memcpy(tensorOuputDimA, res->tensorOuputDimA, sizeof(int) * nbDims);
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetConvolutionNdForwardOutputDim);
    return err;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnFilterDescriptor_t  filterDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  destDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults)
{
	TALLY_LOG("cudnnGetConvolutionForwardAlgorithm_v7 hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetConvolutionForwardAlgorithm_v7Arg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDALGORITHM_V7;
    
    auto arg_ptr = (struct cudnnGetConvolutionForwardAlgorithm_v7Arg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->srcDesc = srcDesc;
    arg_ptr->filterDesc = filterDesc;
    arg_ptr->convDesc = convDesc;
    arg_ptr->destDesc = destDesc;
    arg_ptr->requestedAlgoCount = requestedAlgoCount;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnGetConvolutionForwardAlgorithm_v7Response *) dat;
    *returnedAlgoCount = res->returnedAlgoCount;
    memcpy(perfResults, res->perfResults, sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount);
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetConvolutionForwardAlgorithm_v7);
    return err;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults)
{
	TALLY_LOG("cudnnFindConvolutionForwardAlgorithm hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnFindConvolutionForwardAlgorithmArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNFINDCONVOLUTIONFORWARDALGORITHM;
    
    auto arg_ptr = (struct cudnnFindConvolutionForwardAlgorithmArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->xDesc = xDesc;
    arg_ptr->wDesc = wDesc;
    arg_ptr->convDesc = convDesc;
    arg_ptr->yDesc = yDesc;
    arg_ptr->requestedAlgoCount = requestedAlgoCount;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnFindConvolutionForwardAlgorithmResponse *) dat;
    *returnedAlgoCount = res->returnedAlgoCount;
    memcpy(perfResults, res->perfResults, sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount);
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnFindConvolutionForwardAlgorithm);
    return err;
}

cudnnStatus_t cudnnAddTensor(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)
{
    TALLY_LOG("cudnnAddTensor hooked");
    TALLY_CLIENT_PROFILE_START;
    
#ifdef RUN_LOCALLY
    auto err = lcudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnAddTensorArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNADDTENSOR;
    
    auto arg_ptr = (struct cudnnAddTensorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
    arg_ptr->aDesc = aDesc;
    arg_ptr->A = const_cast<void *>(A);
    arg_ptr->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
    arg_ptr->cDesc = cDesc;
    arg_ptr->C = C;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnAddTensor);
    return err;	
}

cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t  poolingDesc, const cudnnPoolingMode_t  mode, const cudnnNanPropagation_t  maxpoolingNanOpt, int  nbDims, const int  windowDimA[], const int  paddingA[], const int  strideA[])
{
	TALLY_LOG("cudnnSetPoolingNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetPoolingNdDescriptorArg) + 3 * nbDims * sizeof(int);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETPOOLINGNDDESCRIPTOR;
    
    auto arg_ptr = (struct cudnnSetPoolingNdDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->poolingDesc = poolingDesc;
    arg_ptr->mode = mode;
    arg_ptr->maxpoolingNanOpt = maxpoolingNanOpt;
    arg_ptr->nbDims = nbDims;

    memcpy(arg_ptr->windowDimA_paddingA_strideA, windowDimA, sizeof(int) * nbDims);
    memcpy(arg_ptr->windowDimA_paddingA_strideA + nbDims, paddingA, sizeof(int) * nbDims);
    memcpy(arg_ptr->windowDimA_paddingA_strideA + 2 * nbDims, strideA, sizeof(int) * nbDims);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetPoolingNdDescriptor);
    return err;	
}

cudnnStatus_t cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t  poolingDesc, int  nbDimsRequested, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * nbDims, int  windowDimA[], int  paddingA[], int  strideA[])
{
	TALLY_LOG("cudnnGetPoolingNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetPoolingNdDescriptorArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETPOOLINGNDDESCRIPTOR;
    
    auto arg_ptr = (struct cudnnGetPoolingNdDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->poolingDesc = poolingDesc;
    arg_ptr->nbDimsRequested = nbDimsRequested;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnGetPoolingNdDescriptorResponse *) dat;
    *mode = res->mode;
    *maxpoolingNanOpt = res->maxpoolingNanOpt;
    *nbDims = res->nbDims;
    memcpy(windowDimA, res->windowDimA_paddingA_strideA, sizeof(int) * res->nbDims);
    memcpy(paddingA, res->windowDimA_paddingA_strideA + res->nbDims, sizeof(int) * res->nbDims);
    memcpy(strideA, res->windowDimA_paddingA_strideA + res->nbDims * 2, sizeof(int) * res->nbDims);

    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetPoolingNdDescriptor);
    return err;	
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int  nbDims, int  outputTensorDimA[])
{
	TALLY_LOG("cudnnGetPoolingNdForwardOutputDim hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetPoolingNdForwardOutputDimArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETPOOLINGNDFORWARDOUTPUTDIM;
    
    auto arg_ptr = (struct cudnnGetPoolingNdForwardOutputDimArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->poolingDesc = poolingDesc;
    arg_ptr->inputTensorDesc = inputTensorDesc;
    arg_ptr->nbDims = nbDims;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnGetPoolingNdForwardOutputDimResponse *) dat;
    memcpy(outputTensorDimA, res->outputTensorDimA, sizeof(int) * nbDims);
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetPoolingNdForwardOutputDim);
    return err;	
}

cudnnStatus_t cudnnPoolingForward(cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
    TALLY_LOG("cudnnPoolingForward hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnPoolingForwardArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNPOOLINGFORWARD;
    
    auto arg_ptr = (struct cudnnPoolingForwardArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->poolingDesc = poolingDesc;
    arg_ptr->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
    arg_ptr->xDesc = xDesc;
    arg_ptr->x = const_cast<void *>(x);
    arg_ptr->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
    arg_ptr->yDesc = yDesc;
    arg_ptr->y = y;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnPoolingForward);
    return err;
}

cublasStatus_t cublasSgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSgemv_v2 hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSgemv_v2Arg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASSGEMV_V2;
    
    auto arg_ptr = (struct cublasSgemv_v2Arg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->trans = trans;
    arg_ptr->m = m;
    arg_ptr->n = n;
    arg_ptr->alpha = *alpha;
    arg_ptr->A = const_cast<float *>(A);
    arg_ptr->lda = lda;
    arg_ptr->x = const_cast<float *>(x);
    arg_ptr->incx = incx;
    arg_ptr->beta = *beta;
    arg_ptr->y = y;
    arg_ptr->incy = incy;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasSgemv_v2);
    return err;
}

cudnnStatus_t cudnnLRNCrossChannelForward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnLRNCrossChannelForward hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnLRNCrossChannelForwardArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNLRNCROSSCHANNELFORWARD;
    
    auto arg_ptr = (struct cudnnLRNCrossChannelForwardArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->normDesc = normDesc;
    arg_ptr->lrnMode = lrnMode;
    arg_ptr->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
    arg_ptr->xDesc = xDesc;
    arg_ptr->x = const_cast<void*>(x);
    arg_ptr->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
    arg_ptr->yDesc = yDesc;
    arg_ptr->y = y;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnLRNCrossChannelForward);
    return err;
}

cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnSoftmaxForward hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSoftmaxForwardArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSOFTMAXFORWARD;
    
    auto arg_ptr = (struct cudnnSoftmaxForwardArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->algo = algo;
    arg_ptr->mode = mode;
    arg_ptr->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
    arg_ptr->xDesc = xDesc;
    arg_ptr->x = const_cast<void*>(x);
    arg_ptr->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
    arg_ptr->yDesc = yDesc;
    arg_ptr->y = y;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSoftmaxForward);
    return err;
}

cudnnStatus_t cudnnTransformTensor(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnTransformTensor hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnTransformTensorArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNTRANSFORMTENSOR;
    
    auto arg_ptr = (struct cudnnTransformTensorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
    arg_ptr->xDesc = xDesc;
    arg_ptr->x = const_cast<void*>(x);
    arg_ptr->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
    arg_ptr->yDesc = yDesc;
    arg_ptr->y = y;
  
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnTransformTensor);
    return err;
}

cublasStatus_t cublasSgemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasSgemmEx hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSgemmExArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASSGEMMEX;
    
    auto arg_ptr = (struct cublasSgemmExArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->transa = transa;
    arg_ptr->transb = transb;
    arg_ptr->m = m;
    arg_ptr->n = n;
    arg_ptr->k = k;
    arg_ptr->alpha = *alpha;
    arg_ptr->A = const_cast<void*>(A);
    arg_ptr->Atype = Atype;
    arg_ptr->lda = lda;
    arg_ptr->B = const_cast<void*>(B);
    arg_ptr->Btype = Btype;
    arg_ptr->ldb = ldb;
    arg_ptr->beta = *beta;
    arg_ptr->C = C;
    arg_ptr->Ctype = Ctype;
    arg_ptr->ldc = ldc;
  
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasSgemmEx);
    return err;
}

cudnnStatus_t cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const cudnnSeqDataAxis_t  axes[], size_t  seqLengthArraySize, const int  seqLengthArray[], void * paddingFill)
{
	TALLY_LOG("cudnnSetSeqDataDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetSeqDataDescriptorArg) + seqLengthArraySize * sizeof(int);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETSEQDATADESCRIPTOR;
    
    auto arg_ptr = (struct cudnnSetSeqDataDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->seqDataDesc = seqDataDesc;
    arg_ptr->dataType = dataType;
    arg_ptr->nbDims = 4;
    memcpy(arg_ptr->dimA, dimA, sizeof(int) * 4);
    memcpy(arg_ptr->axes, axes, sizeof(cudnnSeqDataAxis_t) * 4);
    arg_ptr->seqLengthArraySize = seqLengthArraySize;
    arg_ptr->paddingFill = NULL;
    memcpy(arg_ptr->seqLengthArray, seqLengthArray, sizeof(int) * seqLengthArraySize);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    int max_seq_len = -1;
    for (int i = 0; i < seqLengthArraySize; i++) {
        max_seq_len = std::max(seqLengthArray[i], max_seq_len);
    }

    seq_desc_to_seq_len_map[seqDataDesc] = max_seq_len;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetSeqDataDescriptor);
    return err;
}

cudnnStatus_t cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t * dataType, int * nbDims, int  nbDimsRequested, int  dimA[], cudnnSeqDataAxis_t  axes[], size_t * seqLengthArraySize, size_t  seqLengthSizeRequested, int  seqLengthArray[], void * paddingFill)
{
	TALLY_LOG("cudnnGetSeqDataDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;
	
#ifdef RUN_LOCALLY
    auto err = lcudnnGetSeqDataDescriptor(seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetSeqDataDescriptorArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETSEQDATADESCRIPTOR;
    
    auto arg_ptr = (struct cudnnGetSeqDataDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->seqDataDesc = seqDataDesc;
    arg_ptr->nbDimsRequested = nbDimsRequested;
    arg_ptr->seqLengthSizeRequested = seqLengthSizeRequested;
    arg_ptr->paddingFill = NULL;
  
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnGetSeqDataDescriptorResponse *) dat;
    *dataType = res->dataType;
    *nbDims = res->nbDims;
    *seqLengthArraySize = res->seqLengthArraySize;
    memcpy(dimA, res->dimA_axes_seqLengthArray, sizeof(int) * res->nbDims);
    memcpy(axes, res->dimA_axes_seqLengthArray + sizeof(int) * res->nbDims, sizeof(cudnnSeqDataAxis_t) * res->nbDims);
    memcpy(seqLengthArray, res->dimA_axes_seqLengthArray + sizeof(int) * res->nbDims + sizeof(cudnnSeqDataAxis_t) * res->nbDims, sizeof(int) * res->seqLengthArraySize);
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetSeqDataDescriptor);
    return err;
}

cudnnStatus_t cudnnMultiHeadAttnForward(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, int  currIdx, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsQO[], const int  devSeqLengthsKV[], const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const void * residuals, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  oDesc, void * out, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)
{
	TALLY_LOG("cudnnMultiHeadAttnForward hooked");
    TALLY_CLIENT_PROFILE_START;
	
#ifdef RUN_LOCALLY
    auto err = lcudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);
#else
    assert(seq_desc_to_seq_len_map.find(qDesc) != seq_desc_to_seq_len_map.end());
    int winIdxLen;

    if (currIdx < 0) {
        winIdxLen = seq_desc_to_seq_len_map[qDesc];
    } else {
        winIdxLen = currIdx + 1;
    }

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnMultiHeadAttnForwardArg) + sizeof(int) * winIdxLen * 2;

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNMULTIHEADATTNFORWARD;
    
    auto arg_ptr = (struct cudnnMultiHeadAttnForwardArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->attnDesc = attnDesc;
    arg_ptr->currIdx = currIdx;
    arg_ptr->devSeqLengthsQO = const_cast<int *>(devSeqLengthsQO);
    arg_ptr->devSeqLengthsKV = const_cast<int *>(devSeqLengthsKV);
    arg_ptr->qDesc = qDesc;
    arg_ptr->queries = const_cast<void *>(queries);
    arg_ptr->residuals = const_cast<void *>(residuals);
    arg_ptr->kDesc = kDesc;
    arg_ptr->keys = const_cast<void *>(keys);
    arg_ptr->vDesc = vDesc;
    arg_ptr->values = const_cast<void *>(values);
    arg_ptr->oDesc = oDesc;
    arg_ptr->out = out;
    arg_ptr->weightSizeInBytes = weightSizeInBytes;
    arg_ptr->weights = const_cast<void *>(weights);
    arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
    arg_ptr->workSpace = workSpace;
    arg_ptr->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
    arg_ptr->reserveSpace = reserveSpace;
    arg_ptr->winIdxLen = winIdxLen;

    memcpy(arg_ptr->loWinIdx_hiWinIdx, loWinIdx, sizeof(int) * winIdxLen);
    memcpy(arg_ptr->loWinIdx_hiWinIdx + winIdxLen, hiWinIdx, sizeof(int) * winIdxLen);
  
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnMultiHeadAttnForward);
    return err;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsDQDO[], const int  devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t  doDesc, const void * dout, const cudnnSeqDataDescriptor_t  dqDesc, void * dqueries, const void * queries, const cudnnSeqDataDescriptor_t  dkDesc, void * dkeys, const void * keys, const cudnnSeqDataDescriptor_t  dvDesc, void * dvalues, const void * values, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)
{
	TALLY_LOG("cudnnMultiHeadAttnBackwardData hooked");
    TALLY_CLIENT_PROFILE_START;
	
#ifdef RUN_LOCALLY
    auto err = lcudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);
#else
    assert(seq_desc_to_seq_len_map.find(dqDesc) != seq_desc_to_seq_len_map.end());
    int winIdxLen = seq_desc_to_seq_len_map[dqDesc];

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnMultiHeadAttnBackwardDataArg) + sizeof(int) * winIdxLen * 2;

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDDATA;
    
    auto arg_ptr = (struct cudnnMultiHeadAttnBackwardDataArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->handle = handle;
    arg_ptr->attnDesc = attnDesc;
    arg_ptr->devSeqLengthsDQDO = const_cast<int *>(devSeqLengthsDQDO);
    arg_ptr->devSeqLengthsDKDV = const_cast<int *>(devSeqLengthsDKDV);
    arg_ptr->doDesc = doDesc;
    arg_ptr->dout = const_cast<void *>(dout);
    arg_ptr->dqDesc = dqDesc;
    arg_ptr->dqueries = dqueries;
    arg_ptr->queries = const_cast<void *>(queries);
    arg_ptr->dkDesc = dkDesc;
    arg_ptr->dkeys = dkeys;
    arg_ptr->keys = const_cast<void *>(keys);
    arg_ptr->dvDesc = dvDesc;
    arg_ptr->dvalues = dvalues;
    arg_ptr->values = const_cast<void *>(values);
    arg_ptr->weightSizeInBytes = weightSizeInBytes;
    arg_ptr->weights = const_cast<void *>(weights);
    arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
    arg_ptr->workSpace = workSpace;
    arg_ptr->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
    arg_ptr->reserveSpace = reserveSpace;

    arg_ptr->winIdxLen = winIdxLen;
    memcpy(arg_ptr->loWinIdx_hiWinIdx, loWinIdx, sizeof(int) * winIdxLen);
    memcpy(arg_ptr->loWinIdx_hiWinIdx + winIdxLen, hiWinIdx, sizeof(int) * winIdxLen);
  
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnMultiHeadAttnBackwardData);
    return err;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnWgradMode_t  addGrad, const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  doDesc, const void * dout, size_t  weightSizeInBytes, const void * weights, void * dweights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)
{
	TALLY_LOG("cudnnMultiHeadAttnBackwardWeights hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnMultiHeadAttnBackwardWeightsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDWEIGHTS;
    
    struct cudnnMultiHeadAttnBackwardWeightsArg *arg_ptr = (struct cudnnMultiHeadAttnBackwardWeightsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->attnDesc = attnDesc;
	arg_ptr->addGrad = addGrad;
	arg_ptr->qDesc = qDesc;
	arg_ptr->queries = const_cast<void *>(queries);
	arg_ptr->kDesc = kDesc;
	arg_ptr->keys = const_cast<void *>(keys);
	arg_ptr->vDesc = vDesc;
	arg_ptr->values = const_cast<void *>(values);
	arg_ptr->doDesc = doDesc;
	arg_ptr->dout = const_cast<void *>(dout);
	arg_ptr->weightSizeInBytes = weightSizeInBytes;
	arg_ptr->weights = const_cast<void *>(weights);
	arg_ptr->dweights = dweights;
	arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
	arg_ptr->workSpace = workSpace;
	arg_ptr->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
	arg_ptr->reserveSpace = reserveSpace;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnMultiHeadAttnBackwardWeights);
    return err;
}

cudnnStatus_t cudnnReorderFilterAndBias(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, cudnnReorderType_t  reorderType, const void * filterData, void * reorderedFilterData, int  reorderBias, const void * biasData, void * reorderedBiasData)
{
	TALLY_LOG("cudnnReorderFilterAndBias hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnReorderFilterAndBiasArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNREORDERFILTERANDBIAS;
    
    struct cudnnReorderFilterAndBiasArg *arg_ptr = (struct cudnnReorderFilterAndBiasArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->filterDesc = filterDesc;
	arg_ptr->reorderType = reorderType;
	arg_ptr->filterData = const_cast<void *>(filterData);
	arg_ptr->reorderedFilterData = reorderedFilterData;
	arg_ptr->reorderBias = reorderBias;
	arg_ptr->biasData = const_cast<void *>(biasData);
	arg_ptr->reorderedBiasData = reorderedBiasData;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnReorderFilterAndBias);
    return err;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetRNNWorkspaceSize hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNWorkspaceSizeArg) + sizeof(cudnnTensorDescriptor_t) * seqLength;

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETRNNWORKSPACESIZE;
    
    auto arg_ptr = (struct cudnnGetRNNWorkspaceSizeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
    arg_ptr->rnnDesc = rnnDesc;
    arg_ptr->seqLength = seqLength;
    memcpy(arg_ptr->xDesc, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnGetRNNWorkspaceSizeResponse *) dat;
    *sizeInBytes = res->sizeInBytes;
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNWorkspaceSize);
    return err;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetRNNTrainingReserveSize hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNTrainingReserveSizeArg) + sizeof(cudnnTensorDescriptor_t) * seqLength;

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETRNNTRAININGRESERVESIZE;
    
    auto arg_ptr = (struct cudnnGetRNNTrainingReserveSizeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
    arg_ptr->rnnDesc = rnnDesc;
    arg_ptr->seqLength = seqLength;
    memcpy(arg_ptr->xDesc, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnGetRNNTrainingReserveSizeResponse *) dat;
    *sizeInBytes = res->sizeInBytes;
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNTrainingReserveSize);
    return err;
}

cudnnStatus_t cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t  filterDesc, int  nbDimsRequested, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * nbDims, int  filterDimA[])
{
	TALLY_LOG("cudnnGetFilterNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnGetFilterNdDescriptor(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetFilterNdDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETFILTERNDDESCRIPTOR;
    
    auto arg_ptr = (struct cudnnGetFilterNdDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->filterDesc = filterDesc;
    arg_ptr->nbDimsRequested = nbDimsRequested;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnGetFilterNdDescriptorResponse *) dat;
    *dataType = res->dataType;
    *format = res->format;
    *nbDims = res->nbDims;
    memcpy(filterDimA, res->filterDimA, sizeof(int) * res->nbDims);
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetFilterNdDescriptor);
    return err;
}

cudnnStatus_t cudnnRNNForwardTraining(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNForwardTraining hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnRNNForwardTraining(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnRNNForwardTrainingArg) + sizeof(cudnnTensorDescriptor_t) * seqLength * 2;

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNRNNFORWARDTRAINING;
    
    auto arg_ptr = (struct cudnnRNNForwardTrainingArg *)(msg + sizeof(CUDA_API_ENUM));

	arg_ptr->handle = handle;
    arg_ptr->rnnDesc = rnnDesc;
    arg_ptr->seqLength = seqLength;
    arg_ptr->x = const_cast<void *>(x);
    arg_ptr->hxDesc = hxDesc;
    arg_ptr->hx = const_cast<void *>(hx);
    arg_ptr->cxDesc = cxDesc;
    arg_ptr->cx = const_cast<void *>(cx);
    arg_ptr->wDesc = wDesc;
    arg_ptr->w = const_cast<void *>(w);
    arg_ptr->y = y;
    arg_ptr->hyDesc = hyDesc;
    arg_ptr->hy = hy;
    arg_ptr->cyDesc = cyDesc;
    arg_ptr->cy = cy;
    arg_ptr->workSpace = workSpace;
    arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
    arg_ptr->reserveSpace = reserveSpace;
    arg_ptr->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;

    memcpy(arg_ptr->xDesc_yDesc, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
    memcpy(arg_ptr->xDesc_yDesc + seqLength, yDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnRNNForwardTraining);
    return err;
}

cudnnStatus_t cudnnRNNBackwardData(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNBackwardData hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnRNNBackwardDataArg) + sizeof(cudnnTensorDescriptor_t) * seqLength * 3;

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDDATA;
    
    auto arg_ptr = (struct cudnnRNNBackwardDataArg *)(msg + sizeof(CUDA_API_ENUM));

	arg_ptr->handle = handle;
    arg_ptr->rnnDesc = rnnDesc;
    arg_ptr->seqLength = seqLength;
    arg_ptr->y = const_cast<void *>(y);
    arg_ptr->dy = const_cast<void *>(dy);
    arg_ptr->dhyDesc = dhyDesc;
    arg_ptr->dhy = const_cast<void *>(dhy);
    arg_ptr->dcyDesc = dcyDesc;
    arg_ptr->dcy = const_cast<void *>(dcy);
    arg_ptr->wDesc = wDesc;
    arg_ptr->w = const_cast<void *>(w);
    arg_ptr->hxDesc = hxDesc;
    arg_ptr->hx = const_cast<void *>(hx);
    arg_ptr->cxDesc = cxDesc;
    arg_ptr->cx = const_cast<void *>(cx);
    arg_ptr->dx = dx;
    arg_ptr->dhxDesc = dhxDesc;
    arg_ptr->dhx = dhx;
    arg_ptr->dcxDesc = dcxDesc;
    arg_ptr->dcx = dcx;
    arg_ptr->workSpace = workSpace;
    arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
    arg_ptr->reserveSpace = reserveSpace;
    arg_ptr->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;

    memcpy(arg_ptr->yDesc_dyDesc_dxDesc, yDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
    memcpy(arg_ptr->yDesc_dyDesc_dxDesc + seqLength, dyDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
    memcpy(arg_ptr->yDesc_dyDesc_dxDesc + seqLength * 2, dxDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnRNNBackwardData);
    return err;
}

cudnnStatus_t cudnnRNNBackwardWeights(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNBackwardWeights hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnRNNBackwardWeightsArg) + sizeof(cudnnTensorDescriptor_t) * seqLength * 2;

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTS;
    
    auto arg_ptr = (struct cudnnRNNBackwardWeightsArg *)(msg + sizeof(CUDA_API_ENUM));

	arg_ptr->handle = handle;
    arg_ptr->rnnDesc = rnnDesc;
    arg_ptr->seqLength = seqLength;
    arg_ptr->x = const_cast<void *>(x);
    arg_ptr->hxDesc = hxDesc;
    arg_ptr->hx = const_cast<void *>(hx);
    arg_ptr->y = const_cast<void *>(y);
    arg_ptr->workSpace = const_cast<void *>(workSpace);
    arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
    arg_ptr->dwDesc = dwDesc;
    arg_ptr->dw = dw;
    arg_ptr->reserveSpace = const_cast<void *>(reserveSpace);
    arg_ptr->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;

    memcpy(arg_ptr->xDesc_yDesc, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
    memcpy(arg_ptr->xDesc_yDesc + seqLength, yDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnRNNBackwardWeights);
    return err;
}

cudnnStatus_t cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t  dataType, cudnnRNNDataLayout_t  layout, int  maxSeqLength, int  batchSize, int  vectorSize, const int  seqLengthArray[], void * paddingFill)
{
	TALLY_LOG("cudnnSetRNNDataDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetRNNDataDescriptorArg) + batchSize * sizeof(int);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETRNNDATADESCRIPTOR;
    
    auto arg_ptr = (struct cudnnSetRNNDataDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->rnnDataDesc = rnnDataDesc;
    arg_ptr->dataType = dataType;
    arg_ptr->layout = layout;
    arg_ptr->maxSeqLength = maxSeqLength;
    arg_ptr->batchSize = batchSize;
    arg_ptr->vectorSize = vectorSize;
    arg_ptr->paddingFill = paddingFill;
    arg_ptr->paddingFillVal = paddingFill ? *((uint64_t *) paddingFill) : 0; // copy 64 bits if not NULL
    memcpy(arg_ptr->seqLengthArray, seqLengthArray, sizeof(int) * batchSize);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetRNNDataDescriptor);
    return err;
}

cudnnStatus_t cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t  tensorDesc, int  nbDimsRequested, cudnnDataType_t * dataType, int * nbDims, int  dimA[], int  strideA[])
{
	TALLY_LOG("cudnnGetTensorNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
#else
	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetTensorNdDescriptorArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETTENSORNDDESCRIPTOR;
    
    auto arg_ptr = (struct cudnnGetTensorNdDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));

    arg_ptr->tensorDesc = tensorDesc;
    arg_ptr->nbDimsRequested = nbDimsRequested;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnGetTensorNdDescriptorResponse *) dat;
    *dataType = res->dataType;
    *nbDims = res->nbDims;
    memcpy(dimA, res->dimA_strideA, sizeof(int) * res->nbDims);
    memcpy(strideA, res->dimA_strideA + res->nbDims, sizeof(int) * res->nbDims);
    auto err = res->err;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetTensorNdDescriptor);
    return err;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnBatchNormalizationForwardTrainingEx hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBatchNormalizationForwardTrainingExArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONFORWARDTRAININGEX;
    
    struct cudnnBatchNormalizationForwardTrainingExArg *arg_ptr = (struct cudnnBatchNormalizationForwardTrainingExArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->mode = mode;
	arg_ptr->bnOps = bnOps;
	arg_ptr->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
	arg_ptr->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
	arg_ptr->xDesc = xDesc;
	arg_ptr->xData = const_cast<void *>(xData);
	arg_ptr->zDesc = zDesc;
	arg_ptr->zData = const_cast<void *>(zData);
	arg_ptr->yDesc = yDesc;
	arg_ptr->yData = yData;
	arg_ptr->bnScaleBiasMeanVarDesc = bnScaleBiasMeanVarDesc;
	arg_ptr->bnScale = const_cast<void *>(bnScale);
	arg_ptr->bnBias = const_cast<void *>(bnBias);
	arg_ptr->exponentialAverageFactor = exponentialAverageFactor;
	arg_ptr->resultRunningMean = resultRunningMean;
	arg_ptr->resultRunningVariance = resultRunningVariance;
	arg_ptr->epsilon = epsilon;
	arg_ptr->resultSaveMean = resultSaveMean;
	arg_ptr->resultSaveInvVariance = resultSaveInvVariance;
	arg_ptr->activationDesc = activationDesc;
	arg_ptr->workspace = workspace;
	arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
	arg_ptr->reserveSpace = reserveSpace;
	arg_ptr->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnBatchNormalizationForwardTrainingEx);
    return err;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScaleData, const void * bnBiasData, void * dBnScaleData, void * dBnBiasData, double  epsilon, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnBatchNormalizationBackwardEx hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
    auto err = lcudnnBatchNormalizationBackwardEx(handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBatchNormalizationBackwardExArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONBACKWARDEX;
    
    struct cudnnBatchNormalizationBackwardExArg *arg_ptr = (struct cudnnBatchNormalizationBackwardExArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->mode = mode;
	arg_ptr->bnOps = bnOps;
	arg_ptr->alphaDataDiff = *((uint64_t *) alphaDataDiff); // copy the 64 bits from the pointer
	arg_ptr->betaDataDiff = *((uint64_t *) betaDataDiff); // copy the 64 bits from the pointer
	arg_ptr->alphaParamDiff = *((uint64_t *) alphaParamDiff); // copy the 64 bits from the pointer
	arg_ptr->betaParamDiff = *((uint64_t *) betaParamDiff); // copy the 64 bits from the pointer
	arg_ptr->xDesc = xDesc;
	arg_ptr->xData = const_cast<void *>(xData);
	arg_ptr->yDesc = yDesc;
	arg_ptr->yData = const_cast<void *>(yData);
	arg_ptr->dyDesc = dyDesc;
	arg_ptr->dyData = const_cast<void *>(dyData);
	arg_ptr->dzDesc = dzDesc;
	arg_ptr->dzData = dzData;
	arg_ptr->dxDesc = dxDesc;
	arg_ptr->dxData = dxData;
	arg_ptr->dBnScaleBiasDesc = dBnScaleBiasDesc;
	arg_ptr->bnScaleData = const_cast<void *>(bnScaleData);
	arg_ptr->bnBiasData = const_cast<void *>(bnBiasData);
	arg_ptr->dBnScaleData = dBnScaleData;
	arg_ptr->dBnBiasData = dBnBiasData;
	arg_ptr->epsilon = epsilon;
	arg_ptr->savedMean = const_cast<void *>(savedMean);
	arg_ptr->savedInvVariance = const_cast<void *>(savedInvVariance);
	arg_ptr->activationDesc = activationDesc;
	arg_ptr->workSpace = workSpace;
	arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
	arg_ptr->reserveSpace = reserveSpace;
	arg_ptr->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    auto err = *res;
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnBatchNormalizationBackwardEx);
    return err;
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  B, int  ldb, long long int  strideB, const float*  beta, float*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasSgemmStridedBatched hooked");
	TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
	auto err = lcublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSgemmStridedBatchedArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASSGEMMSTRIDEDBATCHED;
    
    struct cublasSgemmStridedBatchedArg *arg_ptr = (struct cublasSgemmStridedBatchedArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->transa = transa;
	arg_ptr->transb = transb;
	arg_ptr->m = m;
	arg_ptr->n = n;
	arg_ptr->k = k;
	arg_ptr->alpha = *alpha;
	arg_ptr->A = const_cast<float *>(A);
	arg_ptr->lda = lda;
	arg_ptr->strideA = strideA;
	arg_ptr->B = const_cast<float *>(B);
	arg_ptr->ldb = ldb;
	arg_ptr->strideB = strideB;
	arg_ptr->beta = *beta;
	arg_ptr->C = C;
	arg_ptr->ldc = ldc;
	arg_ptr->strideC = strideC;
	arg_ptr->batchCount = batchCount;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

	auto res = (cublasStatus_t *) dat;
	auto err = *res;
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasSgemmStridedBatched);
	return err;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr, const void * func)
{
	TALLY_LOG("cudaFuncGetAttributes hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
	auto err = lcudaFuncGetAttributes(attr, func);
#else

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaFuncGetAttributesArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    auto msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAFUNCGETATTRIBUTES;
    
    auto arg_ptr = (struct cudaFuncGetAttributesArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->attr = attr;
	arg_ptr->func = const_cast<void *>(func);

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

	auto res = (cudaFuncGetAttributesResponse *) dat;
	if (attr) { *attr = res->attr; }
	auto err = res->err;
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaFuncGetAttributes);
	return err;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)
{
	TALLY_LOG("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags hooked");
    TALLY_CLIENT_PROFILE_START;

#ifdef RUN_LOCALLY
	auto err = lcudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
#else

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsArg);

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSORWITHFLAGS;
    
    auto arg_ptr = (struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->numBlocks = numBlocks;
	arg_ptr->func = const_cast<void *>(func);
	arg_ptr->blockSize = blockSize;
	arg_ptr->dynamicSMemSize = dynamicSMemSize;
	arg_ptr->flags = flags;

	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

	auto res = (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse *) dat;
	*numBlocks = res->numBlocks;
	auto err = res->err;
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);

	return err;
}

}