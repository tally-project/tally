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

#if defined(RUN_LOCALLY)
    return l__cudaRegisterFatBinary(fatCubin);

#else
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::__CUDAREGISTERFATBINARY;
            
            auto request = (__cudaRegisterFatBinaryArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
            request->cached = cached;
            request->magic = magic;
            request->version = version;
            if (!cached) {
                memcpy(request->data, wp->data, cubin_size);
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });
#endif

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

    return l__cudaRegisterFatBinary(fatCubin);
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    std::string deviceFunName (deviceFun);
    TallyClient::client->host_func_to_demangled_kernel_name_map[hostFun] = demangleFunc(deviceFunName);
    uint32_t kernel_func_len = deviceFunName.size();
    uint32_t msg_len = sizeof(CUDA_API_ENUM) + sizeof(struct registerKernelArg) + kernel_func_len * sizeof(char);

#if defined(RUN_LOCALLY)
    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

#else
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::__CUDAREGISTERFUNCTION;
            
            auto request = (registerKernelArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
            request->host_func = (void*) hostFun;
            request->kernel_func_len = kernel_func_len;
            memcpy(request->data, deviceFun, kernel_func_len * sizeof(char));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });
#endif

    TallyClient::client->_kernel_addr_to_args[hostFun] = TallyClient::client->_kernel_name_to_args[deviceFunName];
    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
#if defined(RUN_LOCALLY)
    return l__cudaRegisterFatBinaryEnd(fatCubinHandle);

#else
    TallyClient::client->iox_client->loan(sizeof(CUDA_API_ENUM), alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::__CUDAREGISTERFATBINARYEND;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });
#endif

    return l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

cudaError_t cudaMalloc(void ** devPtr, size_t  size)
{
	TALLY_LOG("cudaMalloc hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaMallocArg);

#if defined(RUN_LOCALLY)
    auto err = lcudaMalloc(devPtr, size);

#else
    cudaError_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMALLOC;
            
            auto request = (cudaMallocArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
            request->devPtr = devPtr;
            request->size = size;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    //! [take response]
    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaMallocResponse*>(responsePayload);

            *devPtr = response->devPtr;
            err = response->err;

            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif

    if (err == cudaSuccess) {
        dev_addr_map.push_back( DeviceMemoryKey(*devPtr, size) );
    }

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaMalloc);

    
    return err;
}

cudaError_t cudaFree(void * devPtr)
{
	TALLY_LOG("cudaFree hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaFreeArg);

#if defined(RUN_LOCALLY)
    auto err = lcudaFree(devPtr);

#else
    cudaError_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAFREE;
            
            auto request = (cudaFreeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
            request->devPtr = devPtr;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudaError_t);
#endif

    if (err == cudaSuccess) {
        free_dev_addr(dev_addr_map, devPtr);
    }

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaFree);

    
    return err;
}

cudaError_t cudaMemcpy(void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind)
{
    TALLY_LOG("cudaMemcpy hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len;

    if (kind == cudaMemcpyHostToDevice) {
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyArg) + count;
    } else if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice){
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyArg);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

#if defined(RUN_LOCALLY)
    auto err = lcudaMemcpy(dst, src, count, kind);

#else
    cudaError_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {
            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMEMCPY;
            
            auto request = (cudaMemcpyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
            request->dst = dst;
            request->src = (void *)src;
            request->count = count;
            request->kind = kind;

            if (kind == cudaMemcpyHostToDevice) {
                memcpy(request->data, src, count);
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaMemcpyResponse*>(responsePayload);
            err = response->err;
            if (kind == cudaMemcpyDeviceToHost) {
                memcpy(dst, response->data, count);
            }
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaMemcpy);

    
    return err;
}

cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
    TALLY_LOG("cudaMemcpyAsync hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len;
    
    if (kind == cudaMemcpyHostToDevice) {
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyAsyncArg) + count;
    } else if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice){
        msg_len = sizeof(CUDA_API_ENUM) + sizeof(cudaMemcpyAsyncArg);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

#if defined(RUN_LOCALLY)
    auto err = lcudaMemcpyAsync(dst, src, count, kind, stream);

#else
    cudaError_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {
            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMEMCPYASYNC;
            
            auto request = (cudaMemcpyAsyncArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
            request->dst = dst;
            request->src = (void *)src;
            request->count = count;
            request->kind = kind;
            request->stream = stream;

            // Copy data to the message
            if (kind == cudaMemcpyHostToDevice) {
                memcpy(request->data, src, count);
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaMemcpyAsyncResponse*>(responsePayload);
            err = response->err;
            if (kind == cudaMemcpyDeviceToHost) {
                memcpy(dst, response->data, count);
            }
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaMemcpyAsync);

    
    return err;
}

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    TALLY_LOG("cudaLaunchKernel hooked");
    TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
    auto err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

#else
    assert(TallyClient::client->_kernel_addr_to_args.find(func) != TallyClient::client->_kernel_addr_to_args.end());

    auto &params_info = TallyClient::client->_kernel_addr_to_args[func];
    uint32_t params_size =  std::accumulate(params_info.begin(), params_info.end(), 0);
    uint32_t msg_len = sizeof(CUDA_API_ENUM) + sizeof(struct cudaLaunchKernelArg) + params_size;
    
    cudaError_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {
            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDALAUNCHKERNEL;
            
            auto request = (cudaLaunchKernelArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
            request->host_func = func;
            request->gridDim = gridDim;
            request->blockDim = blockDim;
            request->sharedMem = sharedMem;
            request->stream = stream;

            size_t offset = 0;
            for (size_t i = 0; i < params_info.size(); i++) {
                memcpy(request->params + offset, args[i], params_info[i]);
                offset += params_info[i];
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudaError_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_KERNEL_CALL(func);

    
    return err;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSgemm_v2 hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSgemm_v2Arg);

#if defined(RUN_LOCALLY)
    auto err = lcublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

#else
    cublasStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {
            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSGEMM_V2;
            
            auto request = (cublasSgemm_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
            request->handle = handle;
            request->transa = transa;
            request->transb = transb;
            request->m = m;
            request->n = n;
            request->k = k;
            request->alpha = *alpha;
            request->A = A;
            request->lda = lda;
            request->B = B;
            request->ldb = ldb;
            request->beta = *beta;
            request->C = C;
            request->ldc = ldc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);
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

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulArg);

#if defined(RUN_LOCALLY)
    auto err = lcublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);

#else
    cublasStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASLTMATMUL;
        
        auto request = (cublasLtMatmulArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->lightHandle = lightHandle;
        request->computeDesc = computeDesc;
        request->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
        request->A = A;
        request->Adesc = Adesc;
        request->B = B;
        request->Bdesc = Bdesc;
        request->beta = *((uint64_t *) beta);
        request->C = (void *)C;
        request->Cdesc = Cdesc;
        request->D = D;
        request->Ddesc = Ddesc;
        request->algo = *algo;
        request->workspace = workspace;
        request->workspaceSizeInBytes = workspaceSizeInBytes;
        request->stream = stream;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmul);

    
    return err;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_LOG("cublasLtMatmulDescSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulDescSetAttributeArg) + sizeInBytes;

#if defined(RUN_LOCALLY)
    auto err = lcublasLtMatmulDescSetAttribute(matmulDesc, attr, buf, sizeInBytes);

#else
    cublasStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASLTMATMULDESCSETATTRIBUTE;
        
        auto request = (cublasLtMatmulDescSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->matmulDesc = matmulDesc;
        request->attr = attr;
        request->sizeInBytes = sizeInBytes;
        memcpy(request->buf, buf, sizeInBytes);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulDescSetAttribute);

    
    return err;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t  matLayout, cublasLtMatrixLayoutAttribute_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_LOG("cublasLtMatrixLayoutSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatrixLayoutSetAttributeArg) + sizeInBytes;
	
#if defined(RUN_LOCALLY)
    auto err = lcublasLtMatrixLayoutSetAttribute(matLayout, attr, buf, sizeInBytes);

#else
    cublasStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTSETATTRIBUTE;
        
        auto request = (cublasLtMatrixLayoutSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->matLayout = matLayout;
        request->attr = attr;
        request->sizeInBytes = sizeInBytes;
        memcpy(request->buf, buf, sizeInBytes);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatrixLayoutSetAttribute);
    
    return err;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t  pref, cublasLtMatmulPreferenceAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_LOG("cublasLtMatmulPreferenceSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulPreferenceSetAttributeArg) + sizeInBytes;

#if defined(RUN_LOCALLY)
    auto err = lcublasLtMatmulPreferenceSetAttribute(pref, attr, buf, sizeInBytes);

#else
    cublasStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCESETATTRIBUTE;
        
        auto request = (cublasLtMatmulPreferenceSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->pref = pref;
        request->attr = attr;
        request->sizeInBytes = sizeInBytes;
        memcpy(request->buf, buf, sizeInBytes);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulPreferenceSetAttribute);
    
    return err;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, cublasLtMatmulPreference_t  preference, int  requestedAlgoCount, cublasLtMatmulHeuristicResult_t  heuristicResultsArray[], int*  returnAlgoCount)
{
	TALLY_LOG("cublasLtMatmulAlgoGetHeuristic hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulAlgoGetHeuristicArg);

#if defined(RUN_LOCALLY)
    auto err = lcublasLtMatmulAlgoGetHeuristic(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount);

#else
    cublasStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASLTMATMULALGOGETHEURISTIC;
        
        auto request = (cublasLtMatmulAlgoGetHeuristicArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->lightHandle = lightHandle;
        request->operationDesc = operationDesc;
        request->Adesc = Adesc;
        request->Bdesc = Bdesc;
        request->Cdesc = Cdesc;
        request->Ddesc = Ddesc;
        request->preference = preference;
        request->requestedAlgoCount = requestedAlgoCount;
        request->heuristicResultsArray = heuristicResultsArray;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtMatmulAlgoGetHeuristicResponse*>(responsePayload);
            err = response->err;
            *returnAlgoCount = response->returnAlgoCount;
            memcpy(heuristicResultsArray, response->heuristicResultsArray, sizeof(cublasLtMatmulHeuristicResult_t) * response->returnAlgoCount);
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulAlgoGetHeuristic);
    
    return err;
}

cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  elementCount, const void * arrayOfElements)
{
	TALLY_LOG("cudnnBackendSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

    int32_t type_size = get_cudnn_attribute_size(attributeType);
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBackendSetAttributeArg) + elementCount * type_size;

#if defined(RUN_LOCALLY)
    auto err = lcudnnBackendSetAttribute(descriptor, attributeName, attributeType, elementCount, arrayOfElements);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNBACKENDSETATTRIBUTE;
        
        auto request = (cudnnBackendSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->descriptor = descriptor;
        request->attributeName = attributeName;
        request->attributeType = attributeType;
        request->elementCount = elementCount;

        assert(arrayOfElements);
        memcpy(request->arrayOfElements, arrayOfElements, type_size * elementCount);

        if (attributeType == CUDNN_TYPE_VOID_PTR) {
            convert_stack_void_ptr_to_value(request->arrayOfElements, elementCount, dev_addr_map);
        }

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnBackendSetAttribute);
    
    return err;
}

cudnnStatus_t cudnnBackendGetAttribute(cudnnBackendDescriptor_t const  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  requestedElementCount, int64_t * elementCount, void * arrayOfElements)
{
	TALLY_LOG("cudnnBackendGetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

    int32_t type_size = get_cudnn_attribute_size(attributeType);
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBackendGetAttributeArg) + requestedElementCount * type_size;

#if defined(RUN_LOCALLY)
    auto err = lcudnnBackendGetAttribute(descriptor, attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNBACKENDGETATTRIBUTE;
        
        auto request = (cudnnBackendGetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->descriptor = descriptor;
        request->attributeName = attributeName;
        request->attributeType = attributeType;
        request->requestedElementCount = requestedElementCount;
        request->elementCount = elementCount;
        request->arrayOfElements = arrayOfElements;
        if (arrayOfElements) {
            memcpy(request->arrayOfElementsData, arrayOfElements, requestedElementCount * type_size);
        }

        assert(request->requestedElementCount >= 0);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnBackendGetAttributeResponse*>(responsePayload);
            err = response->err;
            if (elementCount) {
                *elementCount = response->elementCount;
            }

            if (arrayOfElements) {
                memcpy(arrayOfElements, response->arrayOfElements, type_size * response->arrayOfElementsSize);
            }
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnBackendGetAttribute);
    // 
    return err;
}

cudnnStatus_t cudnnActivationForward(cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnActivationForward hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnActivationForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNACTIVATIONFORWARD;
        
        auto request = (cudnnActivationForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->activationDesc = activationDesc;
        request->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
        request->xDesc = xDesc;
        request->x = const_cast<void*>(x);
        request->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
        request->yDesc = yDesc;
        request->y = y;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnActivationForward);
    
    return err;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const int  strideA[])
{
    TALLY_LOG("cudnnSetTensorNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetTensorNdDescriptorArg) + 2 * nbDims * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETTENSORNDDESCRIPTOR;
        
        auto request = (cudnnSetTensorNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->tensorDesc = tensorDesc;
        request->dataType = dataType;
        request->nbDims = nbDims;
        memcpy(request->dimA_and_strideA, dimA, sizeof(int) * nbDims);
        memcpy(request->dimA_and_strideA + nbDims, strideA, sizeof(int) * nbDims);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetTensorNdDescriptor);
    
    return err;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t  convDesc, int  arrayLength, const int  padA[], const int  filterStrideA[], const int  dilationA[], cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType)
{
	TALLY_LOG("cudnnSetConvolutionNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetConvolutionNdDescriptorArg) + 3 * arrayLength * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETCONVOLUTIONNDDESCRIPTOR;
        
        auto request = (cudnnSetConvolutionNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->convDesc = convDesc;
        request->arrayLength = arrayLength;
        request->mode = mode;
        request->computeType = computeType;
        memcpy(request->padA_and_filterStrideA_and_dilationA, padA, sizeof(int) * arrayLength);
        memcpy(request->padA_and_filterStrideA_and_dilationA + arrayLength, filterStrideA, sizeof(int) * arrayLength);
        memcpy(request->padA_and_filterStrideA_and_dilationA + 2 * arrayLength, dilationA, sizeof(int) * arrayLength);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetConvolutionNdDescriptor);
    
    return err;
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  nbDims, const int  filterDimA[])
{
	TALLY_LOG("cudnnSetFilterNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetFilterNdDescriptorArg) + nbDims * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETFILTERNDDESCRIPTOR;
        
        auto request = (cudnnSetFilterNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->filterDesc = filterDesc;
        request->dataType = dataType;
        request->format = format;
        request->nbDims = nbDims;
        memcpy(request->filterDimA, filterDimA, sizeof(int) * nbDims);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetFilterNdDescriptor);
    
    return err;
}

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnConvolutionForward hooked");
    TALLY_CLIENT_PROFILE_START;
    
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnConvolutionForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNCONVOLUTIONFORWARD;
        
        auto request = (cudnnConvolutionForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
        request->xDesc = xDesc;
        request->x = const_cast<void*>(x);
        request->wDesc = wDesc;
        request->w = const_cast<void*>(w);
        request->convDesc = convDesc;
        request->algo = algo;
        request->workSpace = workSpace;
        request->workSpaceSizeInBytes = workSpaceSizeInBytes;
        request->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
        request->yDesc = yDesc;
        request->y = y;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnConvolutionForward);
    
    return err;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int  nbDims, int  tensorOuputDimA[])
{
	TALLY_LOG("cudnnGetConvolutionNdForwardOutputDim hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetConvolutionNdForwardOutputDimArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONNDFORWARDOUTPUTDIM;
        
        auto request = (cudnnGetConvolutionNdForwardOutputDimArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->convDesc = convDesc;
        request->inputTensorDesc = inputTensorDesc;
        request->filterDesc = filterDesc;
        request->nbDims = nbDims;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetConvolutionNdForwardOutputDimResponse*>(responsePayload);
            err = response->err;
            memcpy(tensorOuputDimA, response->tensorOuputDimA, sizeof(int) * nbDims);
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetConvolutionNdForwardOutputDim);
    
    return err;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnFilterDescriptor_t  filterDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  destDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults)
{
	TALLY_LOG("cudnnGetConvolutionForwardAlgorithm_v7 hooked");
    TALLY_CLIENT_PROFILE_START;

	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetConvolutionForwardAlgorithm_v7Arg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDALGORITHM_V7;
        
        auto request = (cudnnGetConvolutionForwardAlgorithm_v7Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->srcDesc = srcDesc;
        request->filterDesc = filterDesc;
        request->convDesc = convDesc;
        request->destDesc = destDesc;
        request->requestedAlgoCount = requestedAlgoCount;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetConvolutionForwardAlgorithm_v7Response*>(responsePayload);
            err = response->err;
            *returnedAlgoCount = response->returnedAlgoCount;
            memcpy(perfResults, response->perfResults, sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount);
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetConvolutionForwardAlgorithm_v7);
    
    return err;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults)
{
	TALLY_LOG("cudnnFindConvolutionForwardAlgorithm hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnFindConvolutionForwardAlgorithmArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNFINDCONVOLUTIONFORWARDALGORITHM;
        
        auto request = (cudnnFindConvolutionForwardAlgorithmArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->xDesc = xDesc;
        request->wDesc = wDesc;
        request->convDesc = convDesc;
        request->yDesc = yDesc;
        request->requestedAlgoCount = requestedAlgoCount;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnFindConvolutionForwardAlgorithmResponse*>(responsePayload);
            err = response->err;
            *returnedAlgoCount = response->returnedAlgoCount;
            memcpy(perfResults, response->perfResults, sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount);
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnFindConvolutionForwardAlgorithm);
    
    return err;
}

cudnnStatus_t cudnnAddTensor(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)
{
    TALLY_LOG("cudnnAddTensor hooked");
    TALLY_CLIENT_PROFILE_START;

	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnAddTensorArg);
    
#if defined(RUN_LOCALLY)
    auto err = lcudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNADDTENSOR;
        
        auto request = (cudnnAddTensorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
        request->aDesc = aDesc;
        request->A = const_cast<void *>(A);
        request->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
        request->cDesc = cDesc;
        request->C = C;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnAddTensor);
    
    return err;	
}

cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t  poolingDesc, const cudnnPoolingMode_t  mode, const cudnnNanPropagation_t  maxpoolingNanOpt, int  nbDims, const int  windowDimA[], const int  paddingA[], const int  strideA[])
{
	TALLY_LOG("cudnnSetPoolingNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetPoolingNdDescriptorArg) + 3 * nbDims * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETPOOLINGNDDESCRIPTOR;
        
        auto request = (cudnnSetPoolingNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->poolingDesc = poolingDesc;
        request->mode = mode;
        request->maxpoolingNanOpt = maxpoolingNanOpt;
        request->nbDims = nbDims;

        memcpy(request->windowDimA_paddingA_strideA, windowDimA, sizeof(int) * nbDims);
        memcpy(request->windowDimA_paddingA_strideA + nbDims, paddingA, sizeof(int) * nbDims);
        memcpy(request->windowDimA_paddingA_strideA + 2 * nbDims, strideA, sizeof(int) * nbDims);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetPoolingNdDescriptor);
    
    return err;	
}

cudnnStatus_t cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t  poolingDesc, int  nbDimsRequested, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * nbDims, int  windowDimA[], int  paddingA[], int  strideA[])
{
	TALLY_LOG("cudnnGetPoolingNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetPoolingNdDescriptorArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETPOOLINGNDDESCRIPTOR;
        
        auto request = (cudnnGetPoolingNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->poolingDesc = poolingDesc;
        request->nbDimsRequested = nbDimsRequested;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetPoolingNdDescriptorResponse*>(responsePayload);
            err = response->err;
            *mode = response->mode;
            *maxpoolingNanOpt = response->maxpoolingNanOpt;
            *nbDims = response->nbDims;
            memcpy(windowDimA, response->windowDimA_paddingA_strideA, sizeof(int) * response->nbDims);
            memcpy(paddingA, response->windowDimA_paddingA_strideA + response->nbDims, sizeof(int) * response->nbDims);
            memcpy(strideA, response->windowDimA_paddingA_strideA + response->nbDims * 2, sizeof(int) * response->nbDims);
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetPoolingNdDescriptor);
    
    return err;	
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int  nbDims, int  outputTensorDimA[])
{
	TALLY_LOG("cudnnGetPoolingNdForwardOutputDim hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetPoolingNdForwardOutputDimArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETPOOLINGNDFORWARDOUTPUTDIM;
        
        auto request = (cudnnGetPoolingNdForwardOutputDimArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->poolingDesc = poolingDesc;
        request->inputTensorDesc = inputTensorDesc;
        request->nbDims = nbDims;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetPoolingNdForwardOutputDimResponse*>(responsePayload);
            err = response->err;
            memcpy(outputTensorDimA, response->outputTensorDimA, sizeof(int) * nbDims);
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetPoolingNdForwardOutputDim);
    
    return err;	
}

cudnnStatus_t cudnnPoolingForward(cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
    TALLY_LOG("cudnnPoolingForward hooked");
    TALLY_CLIENT_PROFILE_START;

	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnPoolingForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNPOOLINGFORWARD;
        
        auto request = (cudnnPoolingForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->poolingDesc = poolingDesc;
        request->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
        request->xDesc = xDesc;
        request->x = const_cast<void *>(x);
        request->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
        request->yDesc = yDesc;
        request->y = y;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnPoolingForward);
    
    return err;
}

cublasStatus_t cublasSgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSgemv_v2 hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSgemv_v2Arg);

#if defined(RUN_LOCALLY)
    auto err = lcublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);

#else
    cublasStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASSGEMV_V2;
        
        auto request = (cublasSgemv_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->trans = trans;
        request->m = m;
        request->n = n;
        request->alpha = *alpha;
        request->A = const_cast<float *>(A);
        request->lda = lda;
        request->x = const_cast<float *>(x);
        request->incx = incx;
        request->beta = *beta;
        request->y = y;
        request->incy = incy;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasSgemv_v2);
    
    return err;
}

cudnnStatus_t cudnnLRNCrossChannelForward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnLRNCrossChannelForward hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnLRNCrossChannelForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNLRNCROSSCHANNELFORWARD;
        
        auto request = (cudnnLRNCrossChannelForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->normDesc = normDesc;
        request->lrnMode = lrnMode;
        request->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
        request->xDesc = xDesc;
        request->x = const_cast<void*>(x);
        request->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
        request->yDesc = yDesc;
        request->y = y;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnLRNCrossChannelForward);
    
    return err;
}

cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnSoftmaxForward hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSoftmaxForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSOFTMAXFORWARD;
        
        auto request = (cudnnSoftmaxForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->algo = algo;
        request->mode = mode;
        request->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
        request->xDesc = xDesc;
        request->x = const_cast<void*>(x);
        request->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
        request->yDesc = yDesc;
        request->y = y;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSoftmaxForward);
    
    return err;
}

cudnnStatus_t cudnnTransformTensor(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnTransformTensor hooked");
    TALLY_CLIENT_PROFILE_START;

	uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnTransformTensorArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNTRANSFORMTENSOR;
        
        auto request = (cudnnTransformTensorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
        request->xDesc = xDesc;
        request->x = const_cast<void*>(x);
        request->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
        request->yDesc = yDesc;
        request->y = y;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnTransformTensor);
    
    return err;
}

cublasStatus_t cublasSgemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasSgemmEx hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSgemmExArg);

#if defined(RUN_LOCALLY)
    auto err = lcublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);

#else
    cublasStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASSGEMMEX;
        
        auto request = (cublasSgemmExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->transa = transa;
        request->transb = transb;
        request->m = m;
        request->n = n;
        request->k = k;
        request->alpha = *alpha;
        request->A = const_cast<void*>(A);
        request->Atype = Atype;
        request->lda = lda;
        request->B = const_cast<void*>(B);
        request->Btype = Btype;
        request->ldb = ldb;
        request->beta = *beta;
        request->C = C;
        request->Ctype = Ctype;
        request->ldc = ldc;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasSgemmEx);
    
    return err;
}

cudnnStatus_t cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const cudnnSeqDataAxis_t  axes[], size_t  seqLengthArraySize, const int  seqLengthArray[], void * paddingFill)
{
	TALLY_LOG("cudnnSetSeqDataDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    int max_seq_len = -1;
    for (int i = 0; i < seqLengthArraySize; i++) {
        max_seq_len = std::max(seqLengthArray[i], max_seq_len);
    }

    seq_desc_to_seq_len_map[seqDataDesc] = max_seq_len;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetSeqDataDescriptorArg) + seqLengthArraySize * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETSEQDATADESCRIPTOR;
        
        auto request = (cudnnSetSeqDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->seqDataDesc = seqDataDesc;
        request->dataType = dataType;
        request->nbDims = 4;
        memcpy(request->dimA, dimA, sizeof(int) * 4);
        memcpy(request->axes, axes, sizeof(cudnnSeqDataAxis_t) * 4);
        request->seqLengthArraySize = seqLengthArraySize;
        request->paddingFill = NULL;
        memcpy(request->seqLengthArray, seqLengthArray, sizeof(int) * seqLengthArraySize);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetSeqDataDescriptor);
    
    return err;
}

cudnnStatus_t cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t * dataType, int * nbDims, int  nbDimsRequested, int  dimA[], cudnnSeqDataAxis_t  axes[], size_t * seqLengthArraySize, size_t  seqLengthSizeRequested, int  seqLengthArray[], void * paddingFill)
{
	TALLY_LOG("cudnnGetSeqDataDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetSeqDataDescriptorArg);
	
#if defined(RUN_LOCALLY)
    auto err = lcudnnGetSeqDataDescriptor(seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETSEQDATADESCRIPTOR;
        
        auto request = (cudnnGetSeqDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->seqDataDesc = seqDataDesc;
        request->nbDimsRequested = nbDimsRequested;
        request->seqLengthSizeRequested = seqLengthSizeRequested;
        request->paddingFill = NULL;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetSeqDataDescriptorResponse*>(responsePayload);
            err = response->err;
            *dataType = response->dataType;
            *nbDims = response->nbDims;
            *seqLengthArraySize = response->seqLengthArraySize;
            memcpy(dimA, response->dimA_axes_seqLengthArray, sizeof(int) * response->nbDims);
            memcpy(axes, response->dimA_axes_seqLengthArray + sizeof(int) * nbDimsRequested, sizeof(cudnnSeqDataAxis_t) * response->nbDims);
            memcpy(seqLengthArray, response->dimA_axes_seqLengthArray + sizeof(int) * nbDimsRequested + sizeof(cudnnSeqDataAxis_t) * nbDimsRequested, sizeof(int) * response->seqLengthArraySize);
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetSeqDataDescriptor);
    
    return err;
}

cudnnStatus_t cudnnMultiHeadAttnForward(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, int  currIdx, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsQO[], const int  devSeqLengthsKV[], const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const void * residuals, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  oDesc, void * out, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)
{
	TALLY_LOG("cudnnMultiHeadAttnForward hooked");
    TALLY_CLIENT_PROFILE_START;

    assert(seq_desc_to_seq_len_map.find(qDesc) != seq_desc_to_seq_len_map.end());
    int winIdxLen;

    if (currIdx < 0) {
        winIdxLen = seq_desc_to_seq_len_map[qDesc];
    } else {
        winIdxLen = currIdx + 1;
    }

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnMultiHeadAttnForwardArg) + sizeof(int) * winIdxLen * 2;
	
#if defined(RUN_LOCALLY)
    auto err = lcudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNMULTIHEADATTNFORWARD;
        
        auto request = (cudnnMultiHeadAttnForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->attnDesc = attnDesc;
        request->currIdx = currIdx;
        request->devSeqLengthsQO = const_cast<int *>(devSeqLengthsQO);
        request->devSeqLengthsKV = const_cast<int *>(devSeqLengthsKV);
        request->qDesc = qDesc;
        request->queries = const_cast<void *>(queries);
        request->residuals = const_cast<void *>(residuals);
        request->kDesc = kDesc;
        request->keys = const_cast<void *>(keys);
        request->vDesc = vDesc;
        request->values = const_cast<void *>(values);
        request->oDesc = oDesc;
        request->out = out;
        request->weightSizeInBytes = weightSizeInBytes;
        request->weights = const_cast<void *>(weights);
        request->workSpaceSizeInBytes = workSpaceSizeInBytes;
        request->workSpace = workSpace;
        request->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
        request->reserveSpace = reserveSpace;
        request->winIdxLen = winIdxLen;

        memcpy(request->loWinIdx_hiWinIdx, loWinIdx, sizeof(int) * winIdxLen);
        memcpy(request->loWinIdx_hiWinIdx + winIdxLen, hiWinIdx, sizeof(int) * winIdxLen);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnMultiHeadAttnForward);
    
    return err;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsDQDO[], const int  devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t  doDesc, const void * dout, const cudnnSeqDataDescriptor_t  dqDesc, void * dqueries, const void * queries, const cudnnSeqDataDescriptor_t  dkDesc, void * dkeys, const void * keys, const cudnnSeqDataDescriptor_t  dvDesc, void * dvalues, const void * values, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)
{
	TALLY_LOG("cudnnMultiHeadAttnBackwardData hooked");
    TALLY_CLIENT_PROFILE_START;

    assert(seq_desc_to_seq_len_map.find(dqDesc) != seq_desc_to_seq_len_map.end());
    int winIdxLen = seq_desc_to_seq_len_map[dqDesc];

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnMultiHeadAttnBackwardDataArg) + sizeof(int) * winIdxLen * 2;

#if defined(RUN_LOCALLY)
    auto err = lcudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDDATA;
        
        auto request = (cudnnMultiHeadAttnBackwardDataArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->attnDesc = attnDesc;
        request->devSeqLengthsDQDO = const_cast<int *>(devSeqLengthsDQDO);
        request->devSeqLengthsDKDV = const_cast<int *>(devSeqLengthsDKDV);
        request->doDesc = doDesc;
        request->dout = const_cast<void *>(dout);
        request->dqDesc = dqDesc;
        request->dqueries = dqueries;
        request->queries = const_cast<void *>(queries);
        request->dkDesc = dkDesc;
        request->dkeys = dkeys;
        request->keys = const_cast<void *>(keys);
        request->dvDesc = dvDesc;
        request->dvalues = dvalues;
        request->values = const_cast<void *>(values);
        request->weightSizeInBytes = weightSizeInBytes;
        request->weights = const_cast<void *>(weights);
        request->workSpaceSizeInBytes = workSpaceSizeInBytes;
        request->workSpace = workSpace;
        request->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
        request->reserveSpace = reserveSpace;

        request->winIdxLen = winIdxLen;
        memcpy(request->loWinIdx_hiWinIdx, loWinIdx, sizeof(int) * winIdxLen);
        memcpy(request->loWinIdx_hiWinIdx + winIdxLen, hiWinIdx, sizeof(int) * winIdxLen);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnMultiHeadAttnBackwardData);
    
    return err;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnWgradMode_t  addGrad, const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  doDesc, const void * dout, size_t  weightSizeInBytes, const void * weights, void * dweights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)
{
	TALLY_LOG("cudnnMultiHeadAttnBackwardWeights hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnMultiHeadAttnBackwardWeightsArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDWEIGHTS;
        
        auto request = (cudnnMultiHeadAttnBackwardWeightsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->attnDesc = attnDesc;
        request->addGrad = addGrad;
        request->qDesc = qDesc;
        request->queries = const_cast<void *>(queries);
        request->kDesc = kDesc;
        request->keys = const_cast<void *>(keys);
        request->vDesc = vDesc;
        request->values = const_cast<void *>(values);
        request->doDesc = doDesc;
        request->dout = const_cast<void *>(dout);
        request->weightSizeInBytes = weightSizeInBytes;
        request->weights = const_cast<void *>(weights);
        request->dweights = dweights;
        request->workSpaceSizeInBytes = workSpaceSizeInBytes;
        request->workSpace = workSpace;
        request->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
        request->reserveSpace = reserveSpace;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnMultiHeadAttnBackwardWeights);
    
    return err;
}

cudnnStatus_t cudnnReorderFilterAndBias(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, cudnnReorderType_t  reorderType, const void * filterData, void * reorderedFilterData, int  reorderBias, const void * biasData, void * reorderedBiasData)
{
	TALLY_LOG("cudnnReorderFilterAndBias hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnReorderFilterAndBiasArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNREORDERFILTERANDBIAS;
        
        auto request = (cudnnReorderFilterAndBiasArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->filterDesc = filterDesc;
        request->reorderType = reorderType;
        request->filterData = const_cast<void *>(filterData);
        request->reorderedFilterData = reorderedFilterData;
        request->reorderBias = reorderBias;
        request->biasData = const_cast<void *>(biasData);
        request->reorderedBiasData = reorderedBiasData;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnReorderFilterAndBias);
    
    return err;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetRNNWorkspaceSize hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNWorkspaceSizeArg) + sizeof(cudnnTensorDescriptor_t) * seqLength;

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETRNNWORKSPACESIZE;
        
        auto request = (cudnnGetRNNWorkspaceSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->rnnDesc = rnnDesc;
        request->seqLength = seqLength;
        memcpy(request->xDesc, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNWorkspaceSizeResponse*>(responsePayload);
            err = response->err;
            *sizeInBytes = response->sizeInBytes;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNWorkspaceSize);
    
    return err;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetRNNTrainingReserveSize hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNTrainingReserveSizeArg) + sizeof(cudnnTensorDescriptor_t) * seqLength;

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETRNNTRAININGRESERVESIZE;
        
        auto request = (cudnnGetRNNTrainingReserveSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->rnnDesc = rnnDesc;
        request->seqLength = seqLength;
        memcpy(request->xDesc, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNTrainingReserveSizeResponse*>(responsePayload);
            err = response->err;
            *sizeInBytes = response->sizeInBytes;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNTrainingReserveSize);
    
    return err;
}

cudnnStatus_t cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t  filterDesc, int  nbDimsRequested, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * nbDims, int  filterDimA[])
{
	TALLY_LOG("cudnnGetFilterNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetFilterNdDescriptorArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetFilterNdDescriptor(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETFILTERNDDESCRIPTOR;
        
        auto request = (cudnnGetFilterNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->filterDesc = filterDesc;
        request->nbDimsRequested = nbDimsRequested;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetFilterNdDescriptorResponse*>(responsePayload);
            err = response->err;
            *dataType = response->dataType;
            *format = response->format;
            *nbDims = response->nbDims;
            memcpy(filterDimA, response->filterDimA, sizeof(int) * response->nbDims);
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetFilterNdDescriptor);
    
    return err;
}

cudnnStatus_t cudnnRNNForwardTraining(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNForwardTraining hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnRNNForwardTrainingArg) + sizeof(cudnnTensorDescriptor_t) * seqLength * 2;

#if defined(RUN_LOCALLY)
    auto err = lcudnnRNNForwardTraining(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNRNNFORWARDTRAINING;
        
        auto request = (cudnnRNNForwardTrainingArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->rnnDesc = rnnDesc;
        request->seqLength = seqLength;
        request->x = const_cast<void *>(x);
        request->hxDesc = hxDesc;
        request->hx = const_cast<void *>(hx);
        request->cxDesc = cxDesc;
        request->cx = const_cast<void *>(cx);
        request->wDesc = wDesc;
        request->w = const_cast<void *>(w);
        request->y = y;
        request->hyDesc = hyDesc;
        request->hy = hy;
        request->cyDesc = cyDesc;
        request->cy = cy;
        request->workSpace = workSpace;
        request->workSpaceSizeInBytes = workSpaceSizeInBytes;
        request->reserveSpace = reserveSpace;
        request->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;

        memcpy(request->xDesc_yDesc, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
        memcpy(request->xDesc_yDesc + seqLength, yDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnRNNForwardTraining);
    // 
    return err;
}

cudnnStatus_t cudnnRNNBackwardData(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNBackwardData hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnRNNBackwardDataArg) + sizeof(cudnnTensorDescriptor_t) * seqLength * 3;

#if defined(RUN_LOCALLY)
    auto err = lcudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDDATA;
        
        auto request = (cudnnRNNBackwardDataArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->rnnDesc = rnnDesc;
        request->seqLength = seqLength;
        request->y = const_cast<void *>(y);
        request->dy = const_cast<void *>(dy);
        request->dhyDesc = dhyDesc;
        request->dhy = const_cast<void *>(dhy);
        request->dcyDesc = dcyDesc;
        request->dcy = const_cast<void *>(dcy);
        request->wDesc = wDesc;
        request->w = const_cast<void *>(w);
        request->hxDesc = hxDesc;
        request->hx = const_cast<void *>(hx);
        request->cxDesc = cxDesc;
        request->cx = const_cast<void *>(cx);
        request->dx = dx;
        request->dhxDesc = dhxDesc;
        request->dhx = dhx;
        request->dcxDesc = dcxDesc;
        request->dcx = dcx;
        request->workSpace = workSpace;
        request->workSpaceSizeInBytes = workSpaceSizeInBytes;
        request->reserveSpace = reserveSpace;
        request->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;

        memcpy(request->yDesc_dyDesc_dxDesc, yDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
        memcpy(request->yDesc_dyDesc_dxDesc + seqLength, dyDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
        memcpy(request->yDesc_dyDesc_dxDesc + seqLength * 2, dxDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnRNNBackwardData);
    
    return err;
}

cudnnStatus_t cudnnRNNBackwardWeights(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNBackwardWeights hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnRNNBackwardWeightsArg) + sizeof(cudnnTensorDescriptor_t) * seqLength * 2;

#if defined(RUN_LOCALLY)
    auto err = lcudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTS;
        
        auto request = (cudnnRNNBackwardWeightsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->rnnDesc = rnnDesc;
        request->seqLength = seqLength;
        request->x = const_cast<void *>(x);
        request->hxDesc = hxDesc;
        request->hx = const_cast<void *>(hx);
        request->y = const_cast<void *>(y);
        request->workSpace = const_cast<void *>(workSpace);
        request->workSpaceSizeInBytes = workSpaceSizeInBytes;
        request->dwDesc = dwDesc;
        request->dw = dw;
        request->reserveSpace = const_cast<void *>(reserveSpace);
        request->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;

        memcpy(request->xDesc_yDesc, xDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);
        memcpy(request->xDesc_yDesc + seqLength, yDesc, sizeof(cudnnTensorDescriptor_t) * seqLength);

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

        IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnRNNBackwardWeights);
    
    return err;
}

cudnnStatus_t cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t  dataType, cudnnRNNDataLayout_t  layout, int  maxSeqLength, int  batchSize, int  vectorSize, const int  seqLengthArray[], void * paddingFill)
{
	TALLY_LOG("cudnnSetRNNDataDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetRNNDataDescriptorArg) + batchSize * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETRNNDATADESCRIPTOR;
        
        auto request = (cudnnSetRNNDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->rnnDataDesc = rnnDataDesc;
        request->dataType = dataType;
        request->layout = layout;
        request->maxSeqLength = maxSeqLength;
        request->batchSize = batchSize;
        request->vectorSize = vectorSize;
        request->paddingFill = paddingFill;
        request->paddingFillVal = paddingFill ? *((uint64_t *) paddingFill) : 0; // copy 64 bits if not NULL
        memcpy(request->seqLengthArray, seqLengthArray, sizeof(int) * batchSize);

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

        IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnSetRNNDataDescriptor);
    
    return err;
}

cudnnStatus_t cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t  tensorDesc, int  nbDimsRequested, cudnnDataType_t * dataType, int * nbDims, int  dimA[], int  strideA[])
{
	TALLY_LOG("cudnnGetTensorNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetTensorNdDescriptorArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETTENSORNDDESCRIPTOR;
        
        auto request = (cudnnGetTensorNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->tensorDesc = tensorDesc;
        request->nbDimsRequested = nbDimsRequested;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetTensorNdDescriptorResponse*>(responsePayload);
            err = response->err;
            *dataType = response->dataType;
            *nbDims = response->nbDims;
            memcpy(dimA, response->dimA_strideA, sizeof(int) * response->nbDims);
            memcpy(strideA, response->dimA_strideA + nbDimsRequested, sizeof(int) * response->nbDims);
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnGetTensorNdDescriptor);
    
    return err;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnBatchNormalizationForwardTrainingEx hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBatchNormalizationForwardTrainingExArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONFORWARDTRAININGEX;
        
        auto request = (cudnnBatchNormalizationForwardTrainingExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->mode = mode;
        request->bnOps = bnOps;
        request->alpha = *((uint64_t *) alpha); // copy the 64 bits from the pointer
        request->beta = *((uint64_t *) beta); // copy the 64 bits from the pointer
        request->xDesc = xDesc;
        request->xData = const_cast<void *>(xData);
        request->zDesc = zDesc;
        request->zData = const_cast<void *>(zData);
        request->yDesc = yDesc;
        request->yData = yData;
        request->bnScaleBiasMeanVarDesc = bnScaleBiasMeanVarDesc;
        request->bnScale = const_cast<void *>(bnScale);
        request->bnBias = const_cast<void *>(bnBias);
        request->exponentialAverageFactor = exponentialAverageFactor;
        request->resultRunningMean = resultRunningMean;
        request->resultRunningVariance = resultRunningVariance;
        request->epsilon = epsilon;
        request->resultSaveMean = resultSaveMean;
        request->resultSaveInvVariance = resultSaveInvVariance;
        request->activationDesc = activationDesc;
        request->workspace = workspace;
        request->workSpaceSizeInBytes = workSpaceSizeInBytes;
        request->reserveSpace = reserveSpace;
        request->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

        IOX_RECV_RETURN_STATUS(cudnnStatus_t);

#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnBatchNormalizationForwardTrainingEx);
    
    return err;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScaleData, const void * bnBiasData, void * dBnScaleData, void * dBnBiasData, double  epsilon, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnBatchNormalizationBackwardEx hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBatchNormalizationBackwardExArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnBatchNormalizationBackwardEx(handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONBACKWARDEX;
        
        auto request = (cudnnBatchNormalizationBackwardExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->mode = mode;
        request->bnOps = bnOps;
        request->alphaDataDiff = *((uint64_t *) alphaDataDiff); // copy the 64 bits from the pointer
        request->betaDataDiff = *((uint64_t *) betaDataDiff); // copy the 64 bits from the pointer
        request->alphaParamDiff = *((uint64_t *) alphaParamDiff); // copy the 64 bits from the pointer
        request->betaParamDiff = *((uint64_t *) betaParamDiff); // copy the 64 bits from the pointer
        request->xDesc = xDesc;
        request->xData = const_cast<void *>(xData);
        request->yDesc = yDesc;
        request->yData = const_cast<void *>(yData);
        request->dyDesc = dyDesc;
        request->dyData = const_cast<void *>(dyData);
        request->dzDesc = dzDesc;
        request->dzData = dzData;
        request->dxDesc = dxDesc;
        request->dxData = dxData;
        request->dBnScaleBiasDesc = dBnScaleBiasDesc;
        request->bnScaleData = const_cast<void *>(bnScaleData);
        request->bnBiasData = const_cast<void *>(bnBiasData);
        request->dBnScaleData = dBnScaleData;
        request->dBnBiasData = dBnBiasData;
        request->epsilon = epsilon;
        request->savedMean = const_cast<void *>(savedMean);
        request->savedInvVariance = const_cast<void *>(savedInvVariance);
        request->activationDesc = activationDesc;
        request->workSpace = workSpace;
        request->workSpaceSizeInBytes = workSpaceSizeInBytes;
        request->reserveSpace = reserveSpace;
        request->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

        IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudnnBatchNormalizationBackwardEx);
    
    return err;
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  B, int  ldb, long long int  strideB, const float*  beta, float*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasSgemmStridedBatched hooked");
	TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSgemmStridedBatchedArg);

#if defined(RUN_LOCALLY)
	auto err = lcublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);

#else
    cublasStatus_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASSGEMMSTRIDEDBATCHED;
        
        auto request = (cublasSgemmStridedBatchedArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->handle = handle;
        request->transa = transa;
        request->transb = transb;
        request->m = m;
        request->n = n;
        request->k = k;
        request->alpha = *alpha;
        request->A = const_cast<float *>(A);
        request->lda = lda;
        request->strideA = strideA;
        request->B = const_cast<float *>(B);
        request->ldb = ldb;
        request->strideB = strideB;
        request->beta = *beta;
        request->C = C;
        request->ldc = ldc;
        request->strideC = strideC;
        request->batchCount = batchCount;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

        IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasSgemmStridedBatched);
	
    return err;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr, const void * func)
{
	TALLY_LOG("cudaFuncGetAttributes hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaFuncGetAttributesArg);

#if defined(RUN_LOCALLY)
	auto err = lcudaFuncGetAttributes(attr, func);

#else
    cudaError_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDAFUNCGETATTRIBUTES;
        
        auto request = (cudaFuncGetAttributesArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->attr = attr;
	    request->func = const_cast<void *>(func);

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaFuncGetAttributesResponse*>(responsePayload);
            err = response->err;
            if (attr) { *attr = response->attr; }
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaFuncGetAttributes);
	
    return err;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)
{
	TALLY_LOG("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsArg);

#if defined(RUN_LOCALLY)
	auto err = lcudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);

#else
    cudaError_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSORWITHFLAGS;
        
        auto request = (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->numBlocks = numBlocks;
        request->func = const_cast<void *>(func);
        request->blockSize = blockSize;
        request->dynamicSMemSize = dynamicSMemSize;
        request->flags = flags;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse*>(responsePayload);
            err = response->err;
            *numBlocks = response->numBlocks;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);

    return err;
}

cudaError_t cudaChooseDevice(int * device, const struct cudaDeviceProp * prop)
{
	TALLY_LOG("cudaChooseDevice hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaChooseDeviceArg);

#if defined(RUN_LOCALLY)
	auto err = lcudaChooseDevice(device, prop);

#else
    cudaError_t err;

    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDACHOOSEDEVICE;
        
        auto request = (cudaChooseDeviceArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
        request->prop = *prop;

        TallyClient::client->iox_client->send(header).or_else(
        [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaChooseDeviceResponse*>(responsePayload);
            err = response->err;
            *device = response->device;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {}
#endif

    TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    return err;
}

cudaError_t cudaSetDevice(int  device)
{
	TALLY_LOG("cudaSetDevice hooked");
	TALLY_CLIENT_PROFILE_START;

    // Run this locally so local process know which device is being used
    // Thus, cudaGetDevice can be run completely locally
	auto err = lcudaSetDevice(device);

#ifndef RUN_LOCALLY

    TallyClient::client->iox_client->loan(sizeof(CUDA_API_ENUM) + sizeof(cudaSetDeviceArg), alignof(cudaSetDeviceArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASETDEVICE;
            
            auto request = (cudaSetDeviceArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
			request->device = device;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaSetDevice);
	return err;
}

}