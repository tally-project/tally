#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
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
#include <elf.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>
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
#include <tally/cublas_tracer.h>

cublasTracer cublas_tracer;
cublasLtTracer cublasLt_tracer;
cublasLtMatmulDescTracer cublasLtMatmulDesc_tracer;
cublasLtMatrixLayoutTracer cublasLtMatrixLayout_tracer;

std::map<size_t, std::vector<std::pair<std::string, std::string>>> ptx_to_fatbin_map;

// Used to keep track of seq length of a seq description
std::unordered_map<cudnnSeqDataDescriptor_t, int> seq_desc_to_seq_len_map;

// Used to check whether an address points to device memory
std::vector<DeviceMemoryKey> dev_addr_map;

std::map<std::string, void *> lib_name_to_lib_handle;
void *tally_handle = nullptr;

std::pair<const char *, size_t> get_fatbin_from_ptx(std::string &ptx_str)
{
    size_t str_len = ptx_str.size();

    if (ptx_to_fatbin_map.find(str_len) != ptx_to_fatbin_map.end()) {
        for (auto &pair : ptx_to_fatbin_map[str_len]) {

            auto &cached_ptx_str = pair.first;
            auto &cached_fatbin_str = pair.second;

            if (memcmp(ptx_str.c_str(), cached_ptx_str.c_str(), str_len) == 0) {
                return std::make_pair(cached_fatbin_str.c_str(), cached_fatbin_str.size());
            }
        }
    }

    auto fatbin_str = get_fatbin_str_from_ptx_str(ptx_str);
    ptx_to_fatbin_map[str_len].push_back(std::make_pair(ptx_str, fatbin_str));

    if (ptx_to_fatbin_map.find(str_len) != ptx_to_fatbin_map.end()) {
        for (auto &pair : ptx_to_fatbin_map[str_len]) {

            auto &cached_ptx_str = pair.first;
            auto &cached_fatbin_str = pair.second;

            if (memcmp(ptx_str.c_str(), cached_ptx_str.c_str(), str_len) == 0) {
                return std::make_pair(cached_fatbin_str.c_str(), cached_fatbin_str.size());
            }
        }
    }

    throw std::runtime_error("no way");
}

void *tally_cutlass_handle = nullptr;
void (*tally_register_cutlass)() = nullptr;
cudaError_t (*CutlassSgemmNN) (cutlassOperation_t transA, cutlassOperation_t transB, int M, int N,
                               int K, float alpha, float const *A, int lda,float const *B, int ldb,
                               float beta, float *C, int ldc, float *D, int ldd, void *workSpace, cudaStream_t stream) = nullptr;

void load_tally_cutlass_lib()
{
    if (!tally_cutlass_handle) {

        spdlog::info("Enabling replacing cublas with cutlass");

        auto client_preload_dir = get_client_preload_dir();
        auto tally_cutlass_lib_path = client_preload_dir / "libtally_cutlass.so";
        auto preload_str = tally_cutlass_lib_path.string();

        tally_cutlass_handle = dlopen(preload_str.c_str(), RTLD_LAZY);
        tally_register_cutlass = (void (*)()) dlsym(tally_cutlass_handle, "tally_register_cutlass");
        CutlassSgemmNN = (cudaError_t (*) (cutlassOperation_t, cutlassOperation_t, int, int, int, float, float const *, int,
                                           float const *, int, float, float *, int, float *, int, void *, cudaStream_t))
                                           dlsym(tally_cutlass_handle, "CutlassSgemmNN");
    }
}

extern "C" {

void *dlopen(const char *filename, int flag)
{
    static void* (*ldlopen) (const char *, int);
    if (!ldlopen) {
        ldlopen = (void* (*) (const char *, int)) dlsym(RTLD_NEXT, "dlopen");
    }
    assert(ldlopen);

    if (filename) {
        std::string f_name(filename);
        TALLY_SPD_LOG("dlopen: " + f_name);

        if (std::find(preload_libs.begin(), preload_libs.end(), f_name) != preload_libs.end()) {

            if (!tally_handle) {
                auto client_preload_dir = get_client_preload_dir();
                auto tally_lib_path = client_preload_dir / "libtally_client.so";
                auto preload_str = tally_lib_path.string();
                tally_handle = ldlopen(preload_str.c_str(), flag); 
            } 

            if (lib_name_to_lib_handle.find(f_name) == lib_name_to_lib_handle.end()) {
                void *lib_handle = ldlopen(filename, flag);
                lib_name_to_lib_handle[f_name] = lib_handle;
            }

            return tally_handle;
        }
    }

    return ldlopen(filename, flag);
}

void *dlsym(void * handle, const char * symbol)
{
    static void* (*ldlsym) (void * , const char *);
    if (!ldlsym) {
        ldlsym = (void* (*) (void * , const char *)) dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    assert(ldlsym);

    std::string symbol_str(symbol);
    TALLY_SPD_LOG("dlsym: " + symbol_str);

    if (std::find(cuGetProcAddress_v2funcs.begin(), cuGetProcAddress_v2funcs.end(), symbol_str) != cuGetProcAddress_v2funcs.end()) {
        symbol_str = symbol_str + "_v2";
    } else if (std::find(cuGetProcAddress_v3funcs.begin(), cuGetProcAddress_v3funcs.end(), symbol_str) != cuGetProcAddress_v3funcs.end()) {
        symbol_str = symbol_str + "_v3";
    }

    auto symbol_ptr = symbol_str.c_str();
    auto res = ldlsym(handle, symbol_ptr);

    if (!res) {
        
        if (handle == tally_handle) {
            TALLY_SPD_WARN("dlsym failed to find symbol in tally lib: " + symbol_str);

            // Try looking up symbol in the original cuda lib
            for (auto &lib_name : preload_libs) {
                auto lib_handle = lib_name_to_lib_handle[lib_name];

                res = ldlsym(lib_handle, symbol_ptr); 
            
                if (res) {
                    TALLY_SPD_WARN("dlsym falls back to retrieve symbol from " + lib_name);
                    return res;
                }
            }
        } else {
            // Try looking up symbol in tally lib
            res = ldlsym(tally_handle, symbol_ptr);
            if (res) {
                return res;
            }
        }
    }

    if (!res) {
        TALLY_SPD_WARN("dlsym cannot retrieve symbol from anywhere");
    }

    return res;
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream)
{
    TALLY_SPD_LOG("__cudaPopCallConfiguration hooked");

#if defined(RUN_LOCALLY)
    return l__cudaPopCallConfiguration(gridDim, blockDim, sharedMem, stream);
#else
    // throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
    return l__cudaPopCallConfiguration(gridDim, blockDim, sharedMem, stream);
#endif
    
}

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream)
{
    TALLY_SPD_LOG("__cudaPushCallConfiguration hooked");

#if defined(RUN_LOCALLY)
    return l__cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);
#else
    // throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
    return l__cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);
#endif
}

void** __cudaRegisterFatBinary( void *fatCubin ) {

    TALLY_SPD_LOG("__cudaRegisterFatBinary hooked");

#if defined(RUN_LOCALLY)
    return l__cudaRegisterFatBinary(fatCubin);
#else

    auto wp = (__fatBinC_Wrapper_t *) fatCubin;
    auto fbh = (struct fatBinaryHeader *) wp->data;
    const char *cubin_data = (const char *) wp->data;
    size_t cubin_size = fbh->headerSize + fbh->fatSize;

    bool cached = TallyCache::cache->cubin_cache.contains(cubin_data, cubin_size);
    uint32_t cubin_uid = 0;

    size_t msg_len;
    if (!cached) {
        msg_len = sizeof(MessageHeader_t) + sizeof(__cudaRegisterFatBinaryArg) + cubin_size;
    } else {
        msg_len = sizeof(MessageHeader_t) + sizeof(__cudaRegisterFatBinaryArg);
        cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);
    }

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::__CUDAREGISTERFATBINARY;
            header->client_id = TallyClient::client->client_id;

            auto request = (__cudaRegisterFatBinaryArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
            request->cached = cached;
            request->cubin_uid = cubin_uid;

            if (!cached) {
                memcpy(request->data, wp->data, cubin_size);
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    std::map<std::string, std::vector<uint32_t>> kernel_args;
    std::string tmp_elf_file;

    if (cached) {
        kernel_args = TallyCache::cache->cubin_cache.get_kernel_args(cubin_data, cubin_size);
    } else {

        while(!TallyClient::client->iox_client->take()
            .and_then([&](const auto& responsePayload) {
                auto response = static_cast<const char*>(responsePayload);
                
                tmp_elf_file = std::string(response);
        
                TallyClient::client->iox_client->releaseResponse(responsePayload);
            }))
        {}

        kernel_args = get_kernel_names_and_param_sizes_from_elf(tmp_elf_file);

        std::remove(tmp_elf_file.c_str());
    }

    for (auto &pair : kernel_args) {
        auto &kernel_name = pair.first;
        auto &param_sizes = pair.second;
        TallyClient::client->_kernel_name_to_args[kernel_name] = param_sizes;
    }

    return l__cudaRegisterFatBinary(fatCubin);

#endif
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    TALLY_SPD_LOG("__cudaRegisterFunction hooked");

    std::string deviceFunName (deviceFun);
    auto demangled_kernel_name = demangleFunc(deviceFunName);
    TallyClient::client->host_func_to_demangled_kernel_name_map[hostFun] = demangled_kernel_name;
    
    TALLY_SPD_LOG(demangled_kernel_name);

#if defined(RUN_LOCALLY)
    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

#else

    uint32_t kernel_func_len = deviceFunName.size();
    uint32_t msg_len = sizeof(MessageHeader_t) + sizeof(struct __cudaRegisterFunctionArg) + kernel_func_len * sizeof(char);
    
    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::__CUDAREGISTERFUNCTION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (__cudaRegisterFunctionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
    TALLY_SPD_LOG("__cudaRegisterFatBinaryEnd hooked");

#if defined(RUN_LOCALLY)
    return l__cudaRegisterFatBinaryEnd(fatCubinHandle);

#else
    
    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t), alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::__CUDAREGISTERFATBINARYEND;
            header->client_id = TallyClient::client->client_id;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });
#endif

    return l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

cudaError_t cudaMalloc(void ** devPtr, size_t  size)
{
	TALLY_SPD_LOG("cudaMalloc hooked");
    TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
    auto err = lcudaMalloc(devPtr, size);

#else
    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudaMallocArg);
    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMALLOC;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaMallocArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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

    if (err == cudaSuccess) {
        dev_addr_map.push_back( DeviceMemoryKey(*devPtr, size) );
    }
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaMalloc);

    return err;
}

cudaError_t cudaFree(void * devPtr)
{
	TALLY_SPD_LOG("cudaFree hooked");
    TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
    auto err = lcudaFree(devPtr);

#else

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudaFreeArg);

    cudaError_t err;
  
    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAFREE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaFreeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
            request->devPtr = devPtr;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudaError_t);

    if (err == cudaSuccess) {
        free_dev_addr(dev_addr_map, devPtr);
    }
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cudaFree);
    
    return err;
}

cudaError_t cudaMemcpy(void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind)
{
    TALLY_SPD_LOG("cudaMemcpy hooked");
    TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
    auto err = lcudaMemcpy(dst, src, count, kind);

#else

    uint32_t msg_len;

    if (kind == cudaMemcpyHostToDevice) {
        msg_len = sizeof(MessageHeader_t) + sizeof(cudaMemcpyArg) + count;
    } else if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice){
        msg_len = sizeof(MessageHeader_t) + sizeof(cudaMemcpyArg);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    cudaError_t err;
    
    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {
            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMEMCPY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaMemcpyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
    TALLY_SPD_LOG("cudaMemcpyAsync hooked");
    TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
    auto err = lcudaMemcpyAsync(dst, src, count, kind, stream);

#else

    uint32_t msg_len;
    
    if (kind == cudaMemcpyHostToDevice) {
        msg_len = sizeof(MessageHeader_t) + sizeof(cudaMemcpyAsyncArg) + count;
    } else if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice){
        msg_len = sizeof(MessageHeader_t) + sizeof(cudaMemcpyAsyncArg);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {
            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMEMCPYASYNC;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaMemcpyAsyncArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
    TALLY_SPD_LOG("cudaLaunchKernel hooked");
    TALLY_CLIENT_PROFILE_START;

    auto kernel_name = TallyClient::client->host_func_to_demangled_kernel_name_map[func];
    TALLY_SPD_LOG(kernel_name);

#if defined(RUN_LOCALLY)
    auto err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
#else
    assert(TallyClient::client->_kernel_addr_to_args.find(func) != TallyClient::client->_kernel_addr_to_args.end());

    auto &params_info = TallyClient::client->_kernel_addr_to_args[func];
    uint32_t params_size =  std::accumulate(params_info.begin(), params_info.end(), 0);
    uint32_t msg_len = sizeof(MessageHeader_t) + sizeof(struct cudaLaunchKernelArg) + params_size;
    
    cudaError_t err;
    
    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {
            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDALAUNCHKERNEL;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaLaunchKernelArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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

CUresult cuLaunchKernel(CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams, void ** extra)
{
	TALLY_SPD_LOG("cuLaunchKernel hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
#else
    assert(TallyClient::client->_jit_kernel_addr_to_args.find(f) != TallyClient::client->_jit_kernel_addr_to_args.end());
    assert(extra == NULL);
    if (extra != NULL) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": extra is not yet supported for cuLaunchKernel.");
    }

    auto &params_info = TallyClient::client->_jit_kernel_addr_to_args[f];
    uint32_t params_size =  std::accumulate(params_info.begin(), params_info.end(), 0);
    uint32_t msg_len = sizeof(MessageHeader_t) + sizeof(struct cuLaunchKernelArg) + params_size;
    
    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {
            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CULAUNCHKERNEL;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuLaunchKernelArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
            request->f = f;
            request->gridDimX = gridDimX;
            request->gridDimY = gridDimY;
            request->gridDimZ = gridDimZ;
            request->blockDimX = blockDimX;
            request->blockDimY = blockDimY;
            request->blockDimZ = blockDimZ;
            request->sharedMemBytes = sharedMemBytes;
            request->hStream = hStream;
            
            size_t offset = 0;
            for (size_t i = 0; i < params_info.size(); i++) {
                memcpy(request->kernelParams + offset, kernelParams[i], params_info[i]);
                offset += params_info[i];
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(CUresult);

#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuLaunchKernel);
	return err;
}

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

cublasStatus_t cublasSgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasSgemm_v2 hooked");
    TALLY_CLIENT_PROFILE_START;

    bool launched = false;
    cublasStatus_t err;

#if defined(REPLACE_CUBLAS)

    auto ctx = cublas_tracer.get_cublasCtx(handle);

    if (ctx.mode == CUBLAS_DEFAULT_MATH) { 

        TALLY_SPD_LOG("cublasSgemm_v2 replaced with CutlassSgemmNN");
        load_tally_cutlass_lib();

        auto cutlass_transa = cublas_op_to_cutlass_op(transa);
        auto cutlass_transb = cublas_op_to_cutlass_op(transb);

#if defined(VERIFY_CORRECTNESS)
        // Copy array C
        float *C_copy;
        cudaMalloc(&C_copy, sizeof(float) * m * n);
        cudaMemcpy(C_copy, C, sizeof(float) * m * n, cudaMemcpyDeviceToDevice);
#endif

        auto cuda_err = CutlassSgemmNN(cutlass_transa, cutlass_transb, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc, C, ldc, ctx.workspace, ctx.stream);

        if (!cuda_err) {
            err = CUBLAS_STATUS_SUCCESS;
        } else {
            err = CUBLAS_STATUS_INVALID_VALUE;
        }

        launched = true;

#if defined(VERIFY_CORRECTNESS)
        err = lcublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C_copy, ldc);

        cudaDeviceSynchronize();

        float *h_c_cublas = (float *) malloc(sizeof(float) * m * n);
        float *h_c_cutlass = (float *) malloc(sizeof(float) * m * n);

        cudaMemcpy(h_c_cublas, C_copy, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_cutlass, C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        bool results_match = true;

        for (int i = 0; i < m * n; i++) {
            if (abs(h_c_cublas[i] - h_c_cutlass[i]) > 0.001) {
                results_match = false;
                break;
            }
        }

        if (!results_match) {
            spdlog::warn("cublas and cutlass results do not match.");
        }

        free(h_c_cublas);
        free(h_c_cutlass);
        cudaFree(C_copy);
#endif
    }

    if (!launched) {
        spdlog::warn("Fail to replace cublasSgemm_v2 with cutlass implementation");
    }
#endif

    if (!launched) {

#if defined(RUN_LOCALLY)
        err = lcublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
   
        uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasSgemm_v2Arg);

        IOX_CLIENT_ACQUIRE_LOCK;
        TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
            .and_then([&](auto& requestPayload) {
                auto header = static_cast<MessageHeader_t*>(requestPayload);
                header->api_id = CUDA_API_ENUM::CUBLASSGEMM_V2;
                header->client_id = TallyClient::client->client_id;
                
                auto request = (cublasSgemm_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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

#endif // RUN_LOCALLY

    }

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasSgemm_v2);
    
    return err;
}

// Warning: cublasLtMatmulAlgo_t may be a fake pointer
// when created by cublasLtMatmulAlgoInit
// At some point need to keep track which pointers are fake and which are real
cublasStatus_t cublasLtMatmul(cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  computeDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  B, cublasLtMatrixLayout_t  Bdesc, const void*  beta, const void*  C, cublasLtMatrixLayout_t  Cdesc, void*  D, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, void*  workspace, size_t  workspaceSizeInBytes, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasLtMatmul hooked");
    TALLY_CLIENT_PROFILE_START;

    bool launched = false;
    cublasStatus_t err;

#if defined(REPLACE_CUBLAS)

    auto cublasLt_ctx = cublasLt_tracer.get_cublasLtCtx(lightHandle);
    auto matmul_desc = cublasLtMatmulDesc_tracer.get_cublasLtMatmulDescCtx(computeDesc);
    auto matrix_a_layout = cublasLtMatrixLayout_tracer.get_cublasLtMatrixLayoutCtx(Adesc);
    auto matrix_b_layout = cublasLtMatrixLayout_tracer.get_cublasLtMatrixLayoutCtx(Bdesc);
    auto matrix_d_layout = cublasLtMatrixLayout_tracer.get_cublasLtMatrixLayoutCtx(Ddesc);

    cublasLtMatrixLayoutCtx matrix_c_layout;
    if (Cdesc) {
        matrix_c_layout = cublasLtMatrixLayout_tracer.get_cublasLtMatrixLayoutCtx(Cdesc);
    } else {
        matrix_c_layout = matrix_d_layout;
    }

    // Check matmul descriptor is supported
    if (matmul_desc.scaleType == CUDA_R_32F && matmul_desc.computeType == CUBLAS_COMPUTE_32F) {

        // Check matrix layout is supported
        if (matrix_a_layout.type == CUDA_R_32F &&
            matrix_b_layout.type == CUDA_R_32F &&
            matrix_c_layout.type == CUDA_R_32F &&
            matrix_d_layout.type == CUDA_R_32F ) {

            TALLY_SPD_LOG("cublasLtMatmul replaced with CutlassSgemmNN");
            load_tally_cutlass_lib();
            
            uint64_t m = matrix_d_layout.rows;
            uint64_t n = matrix_d_layout.cols;
            uint64_t k;
            if (matmul_desc.cublaslt_matmul_desc_transa == CUBLAS_OP_N) {
                k = matrix_a_layout.cols;
            } else {
                k = matrix_a_layout.rows;
            }

#if defined(VERIFY_CORRECTNESS)
            // Copy array D
            float *D_copy;
            cudaMalloc(&D_copy, sizeof(float) * m * n);
            cudaMemcpy(D_copy, D, sizeof(float) * m * n, cudaMemcpyDeviceToDevice);
#endif

            auto cutlass_transa = cublas_op_to_cutlass_op(matmul_desc.cublaslt_matmul_desc_transa);
            auto cutlass_transb = cublas_op_to_cutlass_op(matmul_desc.cublaslt_matmul_desc_transb);

            auto cuda_err = CutlassSgemmNN(cutlass_transa, cutlass_transb, m, n, k, *((float *)alpha), (float *) A, matrix_a_layout.ld,
                                           (float *) B, matrix_a_layout.ld, *((float *)beta), (float *) C, matrix_d_layout.ld, (float *) D,
                                           matrix_d_layout.ld, NULL, stream);

            if (matmul_desc.cublaslt_matmul_desc_epilogue == CUBLASLT_EPILOGUE_DEFAULT) {
                // do nothing
            } else if (matmul_desc.cublaslt_matmul_desc_epilogue == CUBLASLT_EPILOGUE_BIAS) {
                assert(matmul_desc.cublaslt_matmul_desc_bias_pointer);
            } else {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Epilogue is not yet handled.");
            }

            if (!cuda_err) {
                err = CUBLAS_STATUS_SUCCESS;
            } else {
                err = CUBLAS_STATUS_INVALID_VALUE;
            }

            launched = true;

#if defined(VERIFY_CORRECTNESS)
            err = lcublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D_copy, Ddesc, algo, workspace, workspaceSizeInBytes, stream);

            cudaDeviceSynchronize();

            float *h_d_cublas = (float *) malloc(sizeof(float) * m * n);
            float *h_d_cutlass = (float *) malloc(sizeof(float) * m * n);

            cudaMemcpy(h_d_cublas, D_copy, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_d_cutlass, D, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

            bool results_match = true;

            for (int i = 0; i < m * n; i++) {
                if (abs(h_d_cublas[i] - h_d_cutlass[i]) > 0.001) {
                    results_match = false;
                    std::cout << "h_d_cublas[i]: " << h_d_cublas[i] << std::endl;
                    std::cout << "h_d_cutlass[i]: " << h_d_cutlass[i] << std::endl;
                    break;
                }
            }

            if (!results_match) {
                spdlog::warn("cublas and cutlass results do not match.");
            }

            free(h_d_cublas);
            free(h_d_cutlass);
            cudaFree(D_copy);
#endif

        }
    }

    if (!launched) {
        spdlog::warn("Fail to replace cublasLtMatmul with cutlass implementation");
    }

#endif

    if (!launched) {

#if defined(RUN_LOCALLY)
        err = lcublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
#else
        uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasLtMatmulArg);

        IOX_CLIENT_ACQUIRE_LOCK;
        TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {
            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATMUL;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatmulArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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

    }

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmul);

    return err;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_SPD_LOG("cublasLtMatmulDescSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

    cublasLtMatmulDesc_tracer.handle_cublasLtMatmulDescSetAttribute(matmulDesc, attr, buf, sizeInBytes);

#if defined(RUN_LOCALLY)
    auto err = lcublasLtMatmulDescSetAttribute(matmulDesc, attr, buf, sizeInBytes);
#else

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasLtMatmulDescSetAttributeArg) + sizeInBytes;

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASLTMATMULDESCSETATTRIBUTE;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cublasLtMatmulDescSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cublasLtMatrixLayoutSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
    auto err = lcublasLtMatrixLayoutSetAttribute(matLayout, attr, buf, sizeInBytes);
#else
    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasLtMatrixLayoutSetAttributeArg) + sizeInBytes;
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTSETATTRIBUTE;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cublasLtMatrixLayoutSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cublasLtMatmulPreferenceSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
    auto err = lcublasLtMatmulPreferenceSetAttribute(pref, attr, buf, sizeInBytes);
#else
    cublasStatus_t err;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasLtMatmulPreferenceSetAttributeArg) + sizeInBytes;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCESETATTRIBUTE;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cublasLtMatmulPreferenceSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cublasLtMatmulAlgoGetHeuristic hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasLtMatmulAlgoGetHeuristicArg);

#if defined(RUN_LOCALLY)
    auto err = lcublasLtMatmulAlgoGetHeuristic(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResultsArray, returnAlgoCount);

#else
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASLTMATMULALGOGETHEURISTIC;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cublasLtMatmulAlgoGetHeuristicArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnBackendSetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

    int32_t type_size = get_cudnn_attribute_size(attributeType);
    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnBackendSetAttributeArg) + elementCount * type_size;

#if defined(RUN_LOCALLY)
    auto err = lcudnnBackendSetAttribute(descriptor, attributeName, attributeType, elementCount, arrayOfElements);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNBACKENDSETATTRIBUTE;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnBackendSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnBackendGetAttribute hooked");
    TALLY_CLIENT_PROFILE_START;

    int32_t type_size = get_cudnn_attribute_size(attributeType);
    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnBackendGetAttributeArg) + requestedElementCount * type_size;

#if defined(RUN_LOCALLY)
    auto err = lcudnnBackendGetAttribute(descriptor, attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNBACKENDGETATTRIBUTE;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnBackendGetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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

    return err;
}

cudnnStatus_t cudnnActivationForward(cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_SPD_LOG("cudnnActivationForward hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(cudnnActivationForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNACTIVATIONFORWARD;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnActivationForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
    TALLY_SPD_LOG("cudnnSetTensorNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnSetTensorNdDescriptorArg) + 2 * nbDims * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETTENSORNDDESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnSetTensorNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnSetConvolutionNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnSetConvolutionNdDescriptorArg) + 3 * arrayLength * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETCONVOLUTIONNDDESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnSetConvolutionNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnSetFilterNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnSetFilterNdDescriptorArg) + nbDims * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETFILTERNDDESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnSetFilterNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnConvolutionForward hooked");
    TALLY_CLIENT_PROFILE_START;
    
    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(cudnnConvolutionForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNCONVOLUTIONFORWARD;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnConvolutionForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnGetConvolutionNdForwardOutputDim hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnGetConvolutionNdForwardOutputDimArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONNDFORWARDOUTPUTDIM;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnGetConvolutionNdForwardOutputDimArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnGetConvolutionForwardAlgorithm_v7 hooked");
    TALLY_CLIENT_PROFILE_START;

	uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnGetConvolutionForwardAlgorithm_v7Arg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDALGORITHM_V7;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnGetConvolutionForwardAlgorithm_v7Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnFindConvolutionForwardAlgorithm hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnFindConvolutionForwardAlgorithmArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNFINDCONVOLUTIONFORWARDALGORITHM;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnFindConvolutionForwardAlgorithmArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
    TALLY_SPD_LOG("cudnnAddTensor hooked");
    TALLY_CLIENT_PROFILE_START;

	uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(cudnnAddTensorArg);
    
#if defined(RUN_LOCALLY)
    auto err = lcudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNADDTENSOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnAddTensorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnSetPoolingNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnSetPoolingNdDescriptorArg) + 3 * nbDims * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETPOOLINGNDDESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnSetPoolingNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnGetPoolingNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnGetPoolingNdDescriptorArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETPOOLINGNDDESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnGetPoolingNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnGetPoolingNdForwardOutputDim hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnGetPoolingNdForwardOutputDimArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETPOOLINGNDFORWARDOUTPUTDIM;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnGetPoolingNdForwardOutputDimArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
    TALLY_SPD_LOG("cudnnPoolingForward hooked");
    TALLY_CLIENT_PROFILE_START;

	uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnPoolingForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNPOOLINGFORWARD;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnPoolingForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cublasSgemv_v2 hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasSgemv_v2Arg);

#if defined(RUN_LOCALLY)
    auto err = lcublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);

#else
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASSGEMV_V2;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cublasSgemv_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnLRNCrossChannelForward hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnLRNCrossChannelForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNLRNCROSSCHANNELFORWARD;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnLRNCrossChannelForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnSoftmaxForward hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(cudnnSoftmaxForwardArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSOFTMAXFORWARD;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnSoftmaxForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnTransformTensor hooked");
    TALLY_CLIENT_PROFILE_START;

	uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnTransformTensorArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNTRANSFORMTENSOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnTransformTensorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cublasSgemmEx hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasSgemmExArg);

#if defined(RUN_LOCALLY)
    auto err = lcublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);

#else
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASSGEMMEX;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cublasSgemmExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnSetSeqDataDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    int max_seq_len = -1;
    for (int i = 0; i < seqLengthArraySize; i++) {
        max_seq_len = std::max(seqLengthArray[i], max_seq_len);
    }

    seq_desc_to_seq_len_map[seqDataDesc] = max_seq_len;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnSetSeqDataDescriptorArg) + seqLengthArraySize * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETSEQDATADESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnSetSeqDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnGetSeqDataDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnGetSeqDataDescriptorArg);
	
#if defined(RUN_LOCALLY)
    auto err = lcudnnGetSeqDataDescriptor(seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETSEQDATADESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnGetSeqDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnMultiHeadAttnForward hooked");
    TALLY_CLIENT_PROFILE_START;

    assert(seq_desc_to_seq_len_map.find(qDesc) != seq_desc_to_seq_len_map.end());
    int winIdxLen;

    if (currIdx < 0) {
        winIdxLen = seq_desc_to_seq_len_map[qDesc];
    } else {
        winIdxLen = currIdx + 1;
    }

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnMultiHeadAttnForwardArg) + sizeof(int) * winIdxLen * 2;
	
#if defined(RUN_LOCALLY)
    auto err = lcudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNMULTIHEADATTNFORWARD;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnMultiHeadAttnForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnMultiHeadAttnBackwardData hooked");
    TALLY_CLIENT_PROFILE_START;

    assert(seq_desc_to_seq_len_map.find(dqDesc) != seq_desc_to_seq_len_map.end());
    int winIdxLen = seq_desc_to_seq_len_map[dqDesc];

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnMultiHeadAttnBackwardDataArg) + sizeof(int) * winIdxLen * 2;

#if defined(RUN_LOCALLY)
    auto err = lcudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDDATA;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnMultiHeadAttnBackwardDataArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnMultiHeadAttnBackwardWeights hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnMultiHeadAttnBackwardWeightsArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDWEIGHTS;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnMultiHeadAttnBackwardWeightsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnReorderFilterAndBias hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnReorderFilterAndBiasArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNREORDERFILTERANDBIAS;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnReorderFilterAndBiasArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnGetRNNWorkspaceSize hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnGetRNNWorkspaceSizeArg) + sizeof(cudnnTensorDescriptor_t) * seqLength;

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETRNNWORKSPACESIZE;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnGetRNNWorkspaceSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnGetRNNTrainingReserveSize hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnGetRNNTrainingReserveSizeArg) + sizeof(cudnnTensorDescriptor_t) * seqLength;

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETRNNTRAININGRESERVESIZE;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnGetRNNTrainingReserveSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnGetFilterNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnGetFilterNdDescriptorArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetFilterNdDescriptor(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETFILTERNDDESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnGetFilterNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnRNNForwardTraining hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnRNNForwardTrainingArg) + sizeof(cudnnTensorDescriptor_t) * seqLength * 2;

#if defined(RUN_LOCALLY)
    auto err = lcudnnRNNForwardTraining(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNRNNFORWARDTRAINING;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnRNNForwardTrainingArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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

    return err;
}

cudnnStatus_t cudnnRNNBackwardData(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnRNNBackwardData hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(cudnnRNNBackwardDataArg) + sizeof(cudnnTensorDescriptor_t) * seqLength * 3;

#if defined(RUN_LOCALLY)
    auto err = lcudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDDATA;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnRNNBackwardDataArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnRNNBackwardWeights hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnRNNBackwardWeightsArg) + sizeof(cudnnTensorDescriptor_t) * seqLength * 2;

#if defined(RUN_LOCALLY)
    auto err = lcudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTS;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnRNNBackwardWeightsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnSetRNNDataDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnSetRNNDataDescriptorArg) + batchSize * sizeof(int);

#if defined(RUN_LOCALLY)
    auto err = lcudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNSETRNNDATADESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnSetRNNDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnGetTensorNdDescriptor hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnGetTensorNdDescriptorArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNGETTENSORNDDESCRIPTOR;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnGetTensorNdDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnBatchNormalizationForwardTrainingEx hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnBatchNormalizationForwardTrainingExArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONFORWARDTRAININGEX;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnBatchNormalizationForwardTrainingExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudnnBatchNormalizationBackwardEx hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudnnBatchNormalizationBackwardExArg);

#if defined(RUN_LOCALLY)
    auto err = lcudnnBatchNormalizationBackwardEx(handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONBACKWARDEX;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudnnBatchNormalizationBackwardExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cublasSgemmStridedBatched hooked");
	TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasSgemmStridedBatchedArg);

#if defined(RUN_LOCALLY)
	auto err = lcublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);

#else
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASSGEMMSTRIDEDBATCHED;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cublasSgemmStridedBatchedArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsArg);

#if defined(RUN_LOCALLY)
	auto err = lcudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);

#else
    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSORWITHFLAGS;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudaChooseDevice hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cudaChooseDeviceArg);

#if defined(RUN_LOCALLY)
	auto err = lcudaChooseDevice(device, prop);

#else
    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUDACHOOSEDEVICE;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cudaChooseDeviceArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_SPD_LOG("cudaSetDevice hooked");
	TALLY_CLIENT_PROFILE_START;

    // Run this locally so local process know which device is being used
    // Thus, cudaGetDevice can be run completely locally
	auto err = lcudaSetDevice(device);

#ifndef RUN_LOCALLY

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaSetDeviceArg), alignof(cudaSetDeviceArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASETDEVICE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaSetDeviceArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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

cudnnStatus_t cudnnRNNBackwardWeights_v8(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnWgradMode_t  addGrad, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnTensorDescriptor_t  hDesc, const void * hx, cudnnRNNDataDescriptor_t  yDesc, const void * y, size_t  weightSpaceSize, void * dweightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)
{
	TALLY_SPD_LOG("cudnnRNNBackwardWeights_v8 hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcudnnRNNBackwardWeights_v8(handle, rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, hx, yDesc, y, weightSpaceSize, dweightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnRNNBackwardWeights_v8Arg), alignof(cudnnRNNBackwardWeights_v8Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTS_V8;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnRNNBackwardWeights_v8Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->addGrad = addGrad;
			request->devSeqLengths = const_cast<int32_t *>(devSeqLengths);
			request->xDesc = xDesc;
			request->x = const_cast<void *>(x);
			request->hDesc = hDesc;
			request->hx = const_cast<void *>(hx);
			request->yDesc = yDesc;
			request->y = const_cast<void *>(y);
			request->weightSpaceSize = weightSpaceSize;
			request->dweightSpace = dweightSpace;
			request->workSpaceSize = workSpaceSize;
			request->workSpace = workSpace;
			request->reserveSpaceSize = reserveSpaceSize;
			request->reserveSpace = reserveSpace;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

   IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnRNNBackwardWeights_v8);
	return err;
}

cudnnStatus_t cudnnRNNBackwardData_v8(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  yDesc, const void * y, const void * dy, cudnnRNNDataDescriptor_t  xDesc, void * dx, cudnnTensorDescriptor_t  hDesc, const void * hx, const void * dhy, void * dhx, cudnnTensorDescriptor_t  cDesc, const void * cx, const void * dcy, void * dcx, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)
{
	TALLY_SPD_LOG("cudnnRNNBackwardData_v8 hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcudnnRNNBackwardData_v8(handle, rnnDesc, devSeqLengths, yDesc, y, dy, xDesc, dx, hDesc, hx, dhy, dhx, cDesc, cx, dcy, dcx, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnRNNBackwardData_v8Arg), alignof(cudnnRNNBackwardData_v8Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDDATA_V8;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnRNNBackwardData_v8Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->devSeqLengths = const_cast<int32_t *>(devSeqLengths);
			request->yDesc = yDesc;
			request->y = const_cast<void *>(y);
			request->dy = const_cast<void *>(dy);
			request->xDesc = xDesc;
			request->dx = dx;
			request->hDesc = hDesc;
			request->hx = const_cast<void *>(hx);
			request->dhy = const_cast<void *>(dhy);
			request->dhx = dhx;
			request->cDesc = cDesc;
			request->cx = const_cast<void *>(cx);
			request->dcy = const_cast<void *>(dcy);
			request->dcx = dcx;
			request->weightSpaceSize = weightSpaceSize;
			request->weightSpace = const_cast<void *>(weightSpace);
			request->workSpaceSize = workSpaceSize;
			request->workSpace = workSpace;
			request->reserveSpaceSize = reserveSpaceSize;
			request->reserveSpace = reserveSpace;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnRNNBackwardData_v8);
	return err;
}

cudnnStatus_t cudnnRNNForward(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fwdMode, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnRNNDataDescriptor_t  yDesc, void * y, cudnnTensorDescriptor_t  hDesc, const void * hx, void * hy, cudnnTensorDescriptor_t  cDesc, const void * cx, void * cy, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)
{
	TALLY_SPD_LOG("cudnnRNNForward hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcudnnRNNForward(handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);

#else
    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnRNNForwardArg), alignof(cudnnRNNForwardArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNRNNFORWARD;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnRNNForwardArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->fwdMode = fwdMode;
			request->devSeqLengths = const_cast<int32_t *>(devSeqLengths);
			request->xDesc = xDesc;
			request->x = const_cast<void *>(x);
			request->yDesc = yDesc;
			request->y = y;
			request->hDesc = hDesc;
			request->hx = const_cast<void *>(hx);
			request->hy = hy;
			request->cDesc = cDesc;
			request->cx = const_cast<void *>(cx);
			request->cy = cy;
			request->weightSpaceSize = weightSpaceSize;
			request->weightSpace = const_cast<void *>(weightSpace);
			request->workSpaceSize = workSpaceSize;
			request->workSpace = workSpace;
			request->reserveSpaceSize = reserveSpaceSize;
			request->reserveSpace = reserveSpace;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnRNNForward);
	return err;
}

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t  handle, cudnnBackendDescriptor_t  executionPlan, cudnnBackendDescriptor_t  variantPack)
{
	TALLY_SPD_LOG("cudnnBackendExecute hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBackendExecute(handle, executionPlan, variantPack);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBackendExecuteArg), alignof(cudnnBackendExecuteArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNBACKENDEXECUTE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnBackendExecuteArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->executionPlan = executionPlan;
			request->variantPack = variantPack;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudnnStatus_t);

#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnBackendExecute);
	return err;
}

cudaError_t cudaThreadSynchronize()
{
	TALLY_SPD_LOG("cudaThreadSynchronize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadSynchronize();
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t), alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDATHREADSYNCHRONIZE;
            header->client_id = TallyClient::client->client_id;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudaError_t);
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadSynchronize);
	return err;
}

cudaError_t cudaEventRecord(cudaEvent_t  event, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaEventRecord hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventRecord(event, stream);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventRecordArg), alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAEVENTRECORD;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaEventRecordArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->event = event;
			request->stream = stream;

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
	TALLY_CLIENT_TRACE_API_CALL(cudaEventRecord);
	return err;
}

cudaError_t cudaDeviceSynchronize()
{
	TALLY_SPD_LOG("cudaDeviceSynchronize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceSynchronize();
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t), alignof(CUDA_API_ENUM))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICESYNCHRONIZE;
            header->client_id = TallyClient::client->client_id;
            
            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudaError_t);
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceSynchronize);
	return err;
}

cudaError_t cudaStreamSynchronize(cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaStreamSynchronize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamSynchronize(stream);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamSynchronizeArg), alignof(cudaStreamSynchronizeArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMSYNCHRONIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamSynchronizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudaError_t);
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamSynchronize);
	return err;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t*  handle)
{
	TALLY_SPD_LOG("cublasCreate_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasCreate_v2(handle);
    cublas_tracer.handle_cublasCreate_v2(*handle);
#else

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasCreate_v2Arg), alignof(cublasCreate_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASCREATE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasCreate_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasCreate_v2Response*>(responsePayload);
			if (handle) { *handle = response->handle; }

            cublas_tracer.handle_cublasCreate_v2(response->handle);
            cublas_tracer.handle_cublasSetStream_v2(response->handle, response->stream);

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasCreate_v2);
	return err;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t * handle)
{
	TALLY_SPD_LOG("cudnnCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreate(handle);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateResponse*>(responsePayload);
			if (handle) { *handle = response->handle; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreate);
	return err;
}

CUresult cuModuleLoadData(CUmodule * module, const void * image)
{
	TALLY_SPD_LOG("cuModuleLoadData hooked");
	TALLY_CLIENT_PROFILE_START;
    
#if defined(RUN_LOCALLY)
	auto err = lcuModuleLoadData(module, image);
#else

    CUresult err;

    // For any type of image, convert to fatbin format
    std::string fatbin_str;
    const char *fatbin_data = nullptr;
    size_t fatbin_size = -1;

    auto mod_type = get_cuda_module_type(image);
    if (mod_type == CUDA_MODULE_TYPE::PTX_STRING) {

        auto ptx_str = std::string((char *)image);
        auto res = get_fatbin_from_ptx(ptx_str);
        fatbin_data = res.first;
        fatbin_size = res.second;

    } else if (mod_type == CUDA_MODULE_TYPE::FATBIN) {

        auto fbh = (struct fatBinaryHeader *) image;
        fatbin_data = (char *) image;
        fatbin_size = fbh->headerSize + fbh->fatSize;

    } else if (mod_type == CUDA_MODULE_TYPE::ELF) {

        auto hdr = (Elf64_Ehdr *) image;

        // Compute elf size from last program header and last section header          
        auto elf_size_ph = hdr->e_phoff + hdr->e_phentsize * hdr->e_phnum;
        auto elf_size_sh = hdr->e_shoff + hdr->e_shentsize * hdr->e_shnum;
        auto elf_size = std::max(elf_size_ph, elf_size_sh);

        // Leave it unhandled at this moment
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Handling elf file is yet unimplemented.");
    }

    bool cached = TallyCache::cache->cubin_cache.contains(fatbin_data, fatbin_size);
    uint32_t cubin_uid = 0;

    size_t msg_len;

    if (!cached) {
        msg_len = sizeof(MessageHeader_t) + sizeof(cuModuleLoadDataArg) + fatbin_size;
    } else {
        msg_len = sizeof(MessageHeader_t) + sizeof(cuModuleLoadDataArg);
        cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(fatbin_data, fatbin_size);
    }

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMODULELOADDATA;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuModuleLoadDataArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			
            request->cached = cached;
            request->cubin_uid = cubin_uid;

            if (!cached) {
                memcpy(request->image, fatbin_data, fatbin_size);
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    std::map<std::string, std::vector<uint32_t>> kernel_args;
    std::string tmp_elf_file;

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuModuleLoadDataResponse*>(responsePayload);
			
            *module = response->module;
            if (!cached) {
                tmp_elf_file = std::string(response->tmp_elf_file);
            }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

    if (cached) {
        kernel_args = TallyCache::cache->cubin_cache.get_kernel_args(fatbin_data, fatbin_size);
    } else {
        kernel_args = get_kernel_names_and_param_sizes_from_elf(tmp_elf_file);

        // Delete elf file
        std::remove(tmp_elf_file.c_str());
    }

    for (auto &pair : kernel_args) {
        auto &kernel_name = pair.first;
        auto &param_sizes = pair.second;
        TallyClient::client->_kernel_name_to_args[kernel_name] = param_sizes;
    }

#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuModuleLoadData);
	return err;
}

CUresult cuModuleGetFunction(CUfunction * hfunc, CUmodule  hmod, const char * name)
{
	TALLY_SPD_LOG("cuModuleGetFunction hooked");
	TALLY_CLIENT_PROFILE_START;

    std::string kernel_name(name);
    TALLY_SPD_LOG(kernel_name);

#if defined(RUN_LOCALLY)
	auto err = lcuModuleGetFunction(hfunc, hmod, name);
#else

    CUresult err;

    size_t kernel_name_size = kernel_name.size() + 1;
    size_t msg_len = sizeof(MessageHeader_t) + sizeof(cuModuleGetFunctionArg) + kernel_name_size * sizeof(char);

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMODULEGETFUNCTION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuModuleGetFunctionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hmod = hmod;
            memcpy(request->name, name, kernel_name_size);
            request->name[kernel_name_size] = '\0';

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuModuleGetFunctionResponse*>(responsePayload);
			*hfunc = response->hfunc;

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

    TallyClient::client->_jit_kernel_addr_to_args[*hfunc] = TallyClient::client->_kernel_name_to_args[kernel_name];

#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuModuleGetFunction);
	return err;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t  stream, enum cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, cudaGraph_t * graph_out, const cudaGraphNode_t ** dependencies_out, size_t * numDependencies_out)
{
	TALLY_SPD_LOG("cudaStreamGetCaptureInfo_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamGetCaptureInfo_v2Arg), alignof(cudaStreamGetCaptureInfo_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMGETCAPTUREINFO_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamGetCaptureInfo_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;
			request->captureStatus_out = captureStatus_out;
			request->id_out = id_out;
			request->graph_out = graph_out;
			request->dependencies_out = (CUgraphNode_st***) dependencies_out;
			request->numDependencies_out = numDependencies_out;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamGetCaptureInfo_v2Response*>(responsePayload);
			if (captureStatus_out) { *captureStatus_out = response->captureStatus_out; }
			if (id_out) { *id_out = response->id_out; }
			if (graph_out) { *graph_out = response->graph_out; }
			if (dependencies_out) { *dependencies_out = response->dependencies_out; }
			if (numDependencies_out) { *numDependencies_out = response->numDependencies_out; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamGetCaptureInfo_v2);
	return err;
}

CUresult cuPointerGetAttribute(void * data, CUpointer_attribute  attribute, CUdeviceptr  ptr)
{
	TALLY_SPD_LOG("cuPointerGetAttribute hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcuPointerGetAttribute(data, attribute, ptr);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuPointerGetAttributeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUPOINTERGETATTRIBUTE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuPointerGetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->attribute = attribute;
            request->ptr = ptr;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuPointerGetAttributeResponse*>(responsePayload);
			
            size_t attribute_size = get_cupointer_attribute_size(attribute);
            memcpy(data, response->data, attribute_size);

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuPointerGetAttribute);
	return err;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t  graph, cudaGraphNode_t * nodes, size_t * numNodes)
{
	TALLY_SPD_LOG("cudaGraphGetNodes hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcudaGraphGetNodes(graph, nodes, numNodes);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphGetNodesArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGRAPHGETNODES;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGraphGetNodesArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->graph = graph;
            request->nodes = nodes;
            request->numNodes = *numNodes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaGraphGetNodesResponse*>(responsePayload);
			
            *numNodes = response->numNodes;

            if (nodes) {
                memcpy(nodes, response->nodes, sizeof(cudaGraphNode_t) * (*numNodes));
            }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};


#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphGetNodes);
	return err;
}

cudaError_t cudaFuncSetAttribute(const void * func, enum cudaFuncAttribute  attr, int  value)
{
	TALLY_SPD_LOG("cudaFuncSetAttribute hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaFuncSetAttribute(func, attr, value);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaFuncSetAttributeArg), alignof(cudaFuncSetAttributeArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAFUNCSETATTRIBUTE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaFuncSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->func = const_cast<void *>(func);
			request->attr = attr;
			request->value = value;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cudaError_t);

    err = lcudaFuncSetAttribute(func, attr, value);
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaFuncSetAttribute);
	
    return err;
}

CUresult cuGetProcAddress_v2(const char * symbol, void ** pfn, int  cudaVersion, cuuint64_t  flags, CUdriverProcAddressQueryResult * symbolStatus)
{
	TALLY_SPD_LOG("cuGetProcAddress_v2 hooked");

    std::string symbol_str(symbol);
    TALLY_SPD_LOG("cuGetProcAddress symbol: " + symbol_str);

    CUresult res = lcuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);

    if (res) {
        TALLY_SPD_LOG("cuGetProcAddress failed");
    }

    if (symbol_str == "") {
        // do nothing
        return res;
    }
    else if (symbol_str == "cuGetProcAddress") {
        return res;
    }
    else if (std::find(cuGetProcAddress_funcs.begin(), cuGetProcAddress_funcs.end(), symbol_str) != cuGetProcAddress_funcs.end()) {
        *pfn = dlsym(RTLD_DEFAULT, symbol_str.c_str());
        assert(pfn);
    }
    else if (std::find(cuGetProcAddress_v2funcs.begin(), cuGetProcAddress_v2funcs.end(), symbol_str) != cuGetProcAddress_v2funcs.end()) {
        auto symbol_v2_str = symbol_str + "_v2";
        *pfn = dlsym(RTLD_DEFAULT, symbol_v2_str.c_str());
        assert(pfn);
    }
    else if (std::find(cuGetProcAddress_v3funcs.begin(), cuGetProcAddress_v3funcs.end(), symbol_str) != cuGetProcAddress_v3funcs.end()) {
        auto symbol_v3_str = symbol_str + "_v3";
        *pfn = dlsym(RTLD_DEFAULT, symbol_v3_str.c_str());
        assert(pfn);
    }
    else if (std::find(cuGetProcAddress_direct_call_funcs.begin(), cuGetProcAddress_direct_call_funcs.end(), symbol_str) != cuGetProcAddress_direct_call_funcs.end()) {
        return lcuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
    }
    else {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented cuGetProcAddress_v2 lookup.");
    }

	return res;
}

CUresult cuMemcpy(CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpy hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcuMemcpy(dst, src, ByteCount);
#else

    CUresult err;

    uint32_t msg_len;

    // Copy data if `src` is at host
    if (is_dev_addr(dev_addr_map, (const void*) src)) {
        msg_len = sizeof(MessageHeader_t) + sizeof(cudaMemcpyArg);
    } else {
        msg_len = sizeof(MessageHeader_t) + sizeof(cudaMemcpyArg) + ByteCount;
    }

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMCPY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemcpyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dst = dst;
			request->src = src;
			request->ByteCount = ByteCount;

            if (!is_dev_addr(dev_addr_map, (const void*) src)) {
                memcpy(request->data, (const void*) src, ByteCount);
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cuMemcpyResponse*>(responsePayload);

            err = response->err;
            if (!is_dev_addr(dev_addr_map, (const void*) dst)) {
                memcpy((void*) dst, response->data, ByteCount);
            }

            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemcpy);
	return err;
}

CUresult cuMemcpyAsync(CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpyAsync hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemcpyAsync(dst, src, ByteCount, hStream);
#else

    CUresult err;

    uint32_t msg_len;

    // Copy data if `src` is at host
    if (is_dev_addr(dev_addr_map, (const void*) src)) {
        msg_len = sizeof(MessageHeader_t) + sizeof(cuMemcpyAsyncArg);
    } else {
        msg_len = sizeof(MessageHeader_t) + sizeof(cuMemcpyAsyncArg) + ByteCount;
    }

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMCPYASYNC;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemcpyAsyncArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dst = dst;
			request->src = src;
			request->ByteCount = ByteCount;
			request->hStream = hStream;

            if (!is_dev_addr(dev_addr_map, (const void*) src)) {
                memcpy(request->data, (const void*) src, ByteCount);
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cuMemcpyAsyncResponse*>(responsePayload);

            err = response->err;
            if (!is_dev_addr(dev_addr_map, (const void*) dst)) {
                memcpy((void*) dst, response->data, ByteCount);
            }

            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemcpyAsync);
	return err;
}

CUresult cuMemAllocAsync(CUdeviceptr * dptr, size_t  bytesize, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemAllocAsync hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemAllocAsync(dptr, bytesize, hStream);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemAllocAsyncArg), alignof(cuMemAllocAsyncArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMALLOCASYNC;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemAllocAsyncArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dptr = dptr;
			request->bytesize = bytesize;
			request->hStream = hStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuMemAllocAsyncResponse*>(responsePayload);
			if (dptr) { *dptr = response->dptr; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

    if (err == CUDA_SUCCESS) {
        dev_addr_map.push_back( DeviceMemoryKey((void *)*dptr, bytesize) );
    }
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemAllocAsync);

	return err;
}

CUresult cuMemFree_v2(CUdeviceptr  dptr)
{
	TALLY_SPD_LOG("cuMemFree_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemFree_v2(dptr);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemFree_v2Arg), alignof(cuMemFree_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMFREE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemFree_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dptr = dptr;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

    if (err == CUDA_SUCCESS) {
        free_dev_addr(dev_addr_map, (void *)dptr);
    }
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemFree_v2);
	return err;
}

cudaError_t cudaMemset(void * devPtr, int  value, size_t  count)
{
	TALLY_SPD_LOG("cudaMemset hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaMemset(devPtr, value, count);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaMemsetArg), alignof(cudaMemsetArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMEMSET;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaMemsetArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->devPtr = devPtr;
			request->value = value;
			request->count = count;

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
	TALLY_CLIENT_TRACE_API_CALL(cudaMemset);
	return err;
}

cudaError_t cudaStreamCreate(cudaStream_t * pStream)
{
	TALLY_SPD_LOG("cudaStreamCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamCreate(pStream);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamCreateArg), alignof(cudaStreamCreateArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pStream = pStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamCreateResponse*>(responsePayload);
			if (pStream) { *pStream = response->pStream; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamCreate);
	return err;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaStreamCreateWithFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamCreateWithFlags(pStream, flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamCreateWithFlagsArg), alignof(cudaStreamCreateWithFlagsArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMCREATEWITHFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamCreateWithFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pStream = pStream;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamCreateWithFlagsResponse*>(responsePayload);
			if (pStream) { *pStream = response->pStream; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamCreateWithFlags);
	return err;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned int  flags, int  priority)
{
	TALLY_SPD_LOG("cudaStreamCreateWithPriority hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamCreateWithPriority(pStream, flags, priority);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamCreateWithPriorityArg), alignof(cudaStreamCreateWithPriorityArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMCREATEWITHPRIORITY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamCreateWithPriorityArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pStream = pStream;
			request->flags = flags;
			request->priority = priority;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamCreateWithPriorityResponse*>(responsePayload);
			if (pStream) { *pStream = response->pStream; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamCreateWithPriority);
	return err;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t  stream, enum cudaStreamCaptureMode  mode)
{
	TALLY_SPD_LOG("cudaStreamBeginCapture hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamBeginCapture(stream, mode);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamBeginCaptureArg), alignof(cudaStreamBeginCaptureArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMBEGINCAPTURE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamBeginCaptureArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;
			request->mode = mode;

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
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamBeginCapture);
	return err;
}

CUresult cuStreamCreateWithPriority(CUstream * phStream, unsigned int  flags, int  priority)
{
	TALLY_SPD_LOG("cuStreamCreateWithPriority hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuStreamCreateWithPriority(phStream, flags, priority);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuStreamCreateWithPriorityArg), alignof(cuStreamCreateWithPriorityArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUSTREAMCREATEWITHPRIORITY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuStreamCreateWithPriorityArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->phStream = phStream;
			request->flags = flags;
			request->priority = priority;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuStreamCreateWithPriorityResponse*>(responsePayload);
			if (phStream) { *phStream = response->phStream; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuStreamCreateWithPriority);
	return err;
}

cublasStatus_t cublasGemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_SPD_LOG("cublasGemmEx hooked");
    TALLY_CLIENT_PROFILE_START;

    uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(struct cublasGemmExArg);

#if defined(RUN_LOCALLY)
    auto err = lcublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);

#else
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASGEMMEX;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cublasGemmExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
        request->handle = handle;
        request->transa = transa;
        request->transb = transb;
        request->m = m;
        request->n = n;
        request->k = k;
        request->alpha = *((uint64_t *) alpha);
        request->A = const_cast<void*>(A);
        request->Atype = Atype;
        request->lda = lda;
        request->B = const_cast<void*>(B);
        request->Btype = Btype;
        request->ldb = ldb;
        request->beta = *((uint64_t *) beta);
        request->C = C;
        request->Ctype = Ctype;
        request->ldc = ldc;
        request->computeType = computeType;
        request->algo = algo;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasGemmEx);
    
    return err;
}

CUresult cuFuncGetAttribute(int * pi, CUfunction_attribute  attrib, CUfunction  hfunc)
{
	TALLY_SPD_LOG("cuFuncGetAttribute hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuFuncGetAttribute(pi, attrib, hfunc);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuFuncGetAttributeArg), alignof(cuFuncGetAttributeArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUFUNCGETATTRIBUTE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuFuncGetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pi = pi;
			request->attrib = attrib;
			request->hfunc = hfunc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuFuncGetAttributeResponse*>(responsePayload);
			if (pi) { *pi = response->pi; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuFuncGetAttribute);
	return err;
}

CUresult cuFuncSetAttribute(CUfunction  hfunc, CUfunction_attribute  attrib, int  value)
{
	TALLY_SPD_LOG("cuFuncSetAttribute hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuFuncSetAttribute(hfunc, attrib, value);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuFuncSetAttributeArg), alignof(cuFuncSetAttributeArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUFUNCSETATTRIBUTE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuFuncSetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hfunc = hfunc;
			request->attrib = attrib;
			request->value = value;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuFuncSetAttribute);
	return err;
}

CUresult cuFuncSetCacheConfig(CUfunction  hfunc, CUfunc_cache  config)
{
	TALLY_SPD_LOG("cuFuncSetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuFuncSetCacheConfig(hfunc, config);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuFuncSetCacheConfigArg), alignof(cuFuncSetCacheConfigArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUFUNCSETCACHECONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuFuncSetCacheConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hfunc = hfunc;
			request->config = config;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuFuncSetCacheConfig);
	return err;
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, long long int  strideC, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_SPD_LOG("cublasGemmStridedBatchedEx hooked");
    TALLY_CLIENT_PROFILE_START;

	uint32_t msg_len =  sizeof(MessageHeader_t) + sizeof(cublasGemmStridedBatchedExArg);

#if defined(RUN_LOCALLY)
    auto err = lcublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);

#else
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(CUDA_API_ENUM))
    .and_then([&](auto& requestPayload) {
        auto header = static_cast<MessageHeader_t*>(requestPayload);
        header->api_id = CUDA_API_ENUM::CUBLASGEMMSTRIDEDBATCHEDEX;
        header->client_id = TallyClient::client->client_id;
        
        auto request = (cublasGemmStridedBatchedExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

        request->handle = handle;
        request->transa = transa;
        request->transb = transb;
        request->m = m;
        request->n = n;
        request->k = k;
        request->alpha = *((uint64_t *) alpha);    
        request->A = const_cast<void*>(A);
        request->Atype = Atype;
        request->lda = lda;
        request->strideA = strideA;
        request->B = const_cast<void*>(B);
        request->Btype = Btype;
        request->ldb = ldb;
        request->strideB = strideB;
        request->beta = *((uint64_t *) beta);
        request->C = C;
        request->Ctype = Ctype;
        request->ldc = ldc;
        request->strideC = strideC;
        request->batchCount = batchCount;
        request->computeType = computeType;
        request->algo = algo;

        TallyClient::client->iox_client->send(header).or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
    })
    .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif

    TALLY_CLIENT_PROFILE_END;
    TALLY_CLIENT_TRACE_API_CALL(cublasGemmStridedBatchedEx);
    
    return err;
}

CUresult cuMemsetD8_v2(CUdeviceptr  dstDevice, unsigned char  uc, size_t  N)
{
	TALLY_SPD_LOG("cuMemsetD8_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemsetD8_v2(dstDevice, uc, N);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemsetD8_v2Arg), alignof(cuMemsetD8_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMSETD8_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemsetD8_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dstDevice = dstDevice;
			request->uc = uc;
			request->N = N;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemsetD8_v2);
	return err;
}


CUresult cuStreamCreate(CUstream * phStream, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuStreamCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuStreamCreate(phStream, Flags);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuStreamCreateArg), alignof(cuStreamCreateArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUSTREAMCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuStreamCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->phStream = phStream;
			request->Flags = Flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuStreamCreateResponse*>(responsePayload);
			if (phStream) { *phStream = response->phStream; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuStreamCreate);
	return err;
}

CUresult cuMemAlloc_v2(CUdeviceptr * dptr, size_t  bytesize)
{
	TALLY_SPD_LOG("cuMemAlloc_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemAlloc_v2(dptr, bytesize);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemAlloc_v2Arg), alignof(cuMemAlloc_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMALLOC_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemAlloc_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dptr = dptr;
			request->bytesize = bytesize;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuMemAlloc_v2Response*>(responsePayload);
			if (dptr) { *dptr = response->dptr; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

    if (err == CUDA_SUCCESS) {
        dev_addr_map.push_back( DeviceMemoryKey((void *) *dptr, bytesize) );
    }
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemAlloc_v2);
	return err;
}

CUresult cuMemsetD32_v2(CUdeviceptr  dstDevice, unsigned int  ui, size_t  N)
{
	TALLY_SPD_LOG("cuMemsetD32_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemsetD32_v2(dstDevice, ui, N);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemsetD32_v2Arg), alignof(cuMemsetD32_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMSETD32_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemsetD32_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dstDevice = dstDevice;
			request->ui = ui;
			request->N = N;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemsetD32_v2);
	return err;
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpyHtoDAsync_v2 hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
#else

    CUresult err;

    uint32_t msg_len = sizeof(MessageHeader_t) + sizeof(cuMemcpyHtoDAsync_v2Arg) + ByteCount;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMCPYHTODASYNC_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemcpyHtoDAsync_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dstDevice = dstDevice;
			request->srcHost = const_cast<void *>(srcHost);
			request->ByteCount = ByteCount;
			request->hStream = hStream;
            memcpy(request->data, srcHost, ByteCount);

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(CUresult);

#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemcpyHtoDAsync_v2);
	return err;
}

CUresult cuMemcpyDtoHAsync_v2(void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpyDtoHAsync_v2 hooked");
    TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
#else
    CUresult err;

    uint32_t msg_len = sizeof(MessageHeader_t) + sizeof(cuMemcpyDtoHAsync_v2Arg);

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMCPYDTOHASYNC_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemcpyDtoHAsync_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dstHost = dstHost;
			request->srcDevice = srcDevice;
			request->ByteCount = ByteCount;
			request->hStream = hStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cuMemcpyDtoHAsync_v2Response*>(responsePayload);
            err = response->err;

            if (err == CUDA_SUCCESS) {
                memcpy(dstHost, response->data, ByteCount);
            }

            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemcpyDtoHAsync_v2);
	return err;
}

CUresult cuMemsetD32Async(CUdeviceptr  dstDevice, unsigned int  ui, size_t  N, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemsetD32Async hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemsetD32Async(dstDevice, ui, N, hStream);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemsetD32AsyncArg), alignof(cuMemsetD32AsyncArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMSETD32ASYNC;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemsetD32AsyncArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dstDevice = dstDevice;
			request->ui = ui;
			request->N = N;
            request->hStream = hStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(CUresult);
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemsetD32Async);
	return err;
}

CUresult cuModuleLoadFatBinary(CUmodule * module, const void * fatCubin)
{
	TALLY_SPD_LOG("cuModuleLoadFatBinary hooked");
	TALLY_CLIENT_PROFILE_START;
    
#if defined(RUN_LOCALLY)
	auto err = lcuModuleLoadFatBinary(module, fatCubin);
#else

    std::string fatbin_str;
    const char *fatbin_data = nullptr;
    size_t fatbin_size = -1;

    auto mod_type = get_cuda_module_type(fatCubin);
    if (mod_type == CUDA_MODULE_TYPE::PTX_STRING) {

        auto ptx_str = std::string((char *)fatCubin);
        auto res = get_fatbin_from_ptx(ptx_str);
        fatbin_data = res.first;
        fatbin_size = res.second;

    } else if (mod_type == CUDA_MODULE_TYPE::FATBIN) {

        auto fbh = (struct fatBinaryHeader *) fatCubin;
        fatbin_data = (char *) fatCubin;
        fatbin_size = fbh->headerSize + fbh->fatSize;

    } else if (mod_type == CUDA_MODULE_TYPE::ELF) {

        auto hdr = (Elf64_Ehdr *) fatCubin;

        // Compute elf size from last program header and last section header          
        auto elf_size_ph = hdr->e_phoff + hdr->e_phentsize * hdr->e_phnum;
        auto elf_size_sh = hdr->e_shoff + hdr->e_shentsize * hdr->e_shnum;
        auto elf_size = std::max(elf_size_ph, elf_size_sh);

        // Leave it unhandled at this moment
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Handling elf file is yet unimplemented.");
    }

    CUresult err;

    bool cached = TallyCache::cache->cubin_cache.contains(fatbin_data, fatbin_size);
    uint32_t cubin_uid = 0;

    size_t msg_len;

    if (!cached) {
        msg_len = sizeof(MessageHeader_t) + sizeof(cuModuleLoadFatBinaryArg) + fatbin_size;
    } else {
        msg_len = sizeof(MessageHeader_t) + sizeof(cuModuleLoadFatBinaryArg);
        cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(fatbin_data, fatbin_size);
    }

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMODULELOADFATBINARY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuModuleLoadFatBinaryArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			
            request->cached = cached;
            request->cubin_uid = cubin_uid;

            if (!cached) {
                memcpy(request->image, fatbin_data, fatbin_size);
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    std::map<std::string, std::vector<uint32_t>> kernel_args;
    std::string tmp_elf_file;

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuModuleLoadFatBinaryResponse*>(responsePayload);
			
            *module = response->module;
            if (!cached) {
                tmp_elf_file = std::string(response->tmp_elf_file);
            }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

    if (cached) {
        kernel_args = TallyCache::cache->cubin_cache.get_kernel_args(fatbin_data, fatbin_size);
    } else {
        kernel_args = get_kernel_names_and_param_sizes_from_elf(tmp_elf_file);

        // Delete elf file
        std::remove(tmp_elf_file.c_str());
    }

    for (auto &pair : kernel_args) {
        auto &kernel_name = pair.first;
        auto &param_sizes = pair.second;
        TallyClient::client->_kernel_name_to_args[kernel_name] = param_sizes;
    }

#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuModuleLoadFatBinary);
	return err;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr * dptr, size_t * bytes, CUmodule  hmod, const char * name)
{
	TALLY_SPD_LOG("cuModuleGetGlobal_v2 hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcuModuleGetGlobal_v2(dptr, bytes, hmod, name);
#else

    CUresult err;

    auto name_str = std::string(name);

    size_t msg_len = sizeof(cuModuleGetGlobal_v2Arg) + name_str.size() + 1;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(cuModuleGetGlobal_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMODULEGETGLOBAL_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuModuleGetGlobal_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hmod = hmod;
			memcpy(request->name, name, name_str.size() + 1);

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuModuleGetGlobal_v2Response*>(responsePayload);
			
            *dptr = response->dptr;
            *bytes = response->bytes;
            err = response->err;

            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuModuleGetGlobal_v2);
	return err;
}

CUresult cuModuleLoadDataEx(CUmodule * module, const void * image, unsigned int  numOptions, CUjit_option * options, void ** optionValues)
{
	TALLY_SPD_LOG("cuModuleLoadDataEx hooked");
	TALLY_CLIENT_PROFILE_START;
    
#if defined(RUN_LOCALLY)
	auto err = lcuModuleLoadDataEx(module, image, numOptions, options, optionValues);
#else

    CUresult err;

    // For any type of image, convert to fatbin format
    std::string fatbin_str;
    const char *fatbin_data = nullptr;
    size_t fatbin_size = -1;

    auto mod_type = get_cuda_module_type(image);
    if (mod_type == CUDA_MODULE_TYPE::PTX_STRING) {
        
        auto ptx_str = std::string((char *)image);
        auto res = get_fatbin_from_ptx(ptx_str);
        fatbin_data = res.first;
        fatbin_size = res.second;

    } else if (mod_type == CUDA_MODULE_TYPE::FATBIN) {
        auto fbh = (struct fatBinaryHeader *) image;
        fatbin_data = (char *) image;
        fatbin_size = fbh->headerSize + fbh->fatSize;

    } else if (mod_type == CUDA_MODULE_TYPE::ELF) {
        auto hdr = (Elf64_Ehdr *) image;

        // Compute elf size from last program header and last section header          
        auto elf_size_ph = hdr->e_phoff + hdr->e_phentsize * hdr->e_phnum;
        auto elf_size_sh = hdr->e_shoff + hdr->e_shentsize * hdr->e_shnum;
        auto elf_size = std::max(elf_size_ph, elf_size_sh);

        // Leave it unhandled at this moment
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Handling elf file is yet unimplemented.");
    }

    bool cached = TallyCache::cache->cubin_cache.contains(fatbin_data, fatbin_size);
    uint32_t cubin_uid = 0;

    size_t msg_len = sizeof(MessageHeader_t) + sizeof(cuModuleLoadDataExArg);

    if (!cached) {
        msg_len += fatbin_size;
    } else {
        cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(fatbin_data, fatbin_size);
    }

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(msg_len, alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMODULELOADDATAEX;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuModuleLoadDataExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			
            request->cached = cached;
            request->cubin_uid = cubin_uid;

            if (!cached) {
                memcpy(request->image, fatbin_data, fatbin_size);
            }

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    std::map<std::string, std::vector<uint32_t>> kernel_args;
    std::string tmp_elf_file;

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuModuleLoadDataExResponse*>(responsePayload);
			
            *module = response->module;
            if (!cached) {
                tmp_elf_file = std::string(response->tmp_elf_file);
            }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

    if (cached) {
        kernel_args = TallyCache::cache->cubin_cache.get_kernel_args(fatbin_data, fatbin_size);
    } else {
        kernel_args = get_kernel_names_and_param_sizes_from_elf(tmp_elf_file);

        // Delete elf file
        std::remove(tmp_elf_file.c_str());
    }

    for (auto &pair : kernel_args) {
        auto &kernel_name = pair.first;
        auto &param_sizes = pair.second;
        TallyClient::client->_kernel_name_to_args[kernel_name] = param_sizes;
    }

#endif

	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuModuleLoadData);

	return err;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpyDtoDAsync_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemcpyDtoDAsync_v2Arg), alignof(cuMemcpyDtoDAsync_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMCPYDTODASYNC_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemcpyDtoDAsync_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dstDevice = dstDevice;
			request->srcDevice = srcDevice;
			request->ByteCount = ByteCount;
			request->hStream = hStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(CUresult);
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemcpyDtoDAsync_v2);
	return err;
}

CUresult cuStreamSynchronize(CUstream  hStream)
{
	TALLY_SPD_LOG("cuStreamSynchronize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuStreamSynchronize(hStream);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuStreamSynchronizeArg), alignof(cuStreamSynchronizeArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUSTREAMSYNCHRONIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuStreamSynchronizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hStream = hStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(CUresult);
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuStreamSynchronize);
	return err;
}

CUresult cuCtxSynchronize()
{
	TALLY_SPD_LOG("cuCtxSynchronize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxSynchronize();
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxSynchronizeArg), alignof(cuCtxSynchronizeArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXSYNCHRONIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxSynchronizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(CUresult);
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxSynchronize);
	return err;
}

CUresult cuModuleUnload(CUmodule  hmod)
{
	TALLY_SPD_LOG("cuModuleUnload hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuModuleUnload(hmod);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuModuleUnloadArg), alignof(cuModuleUnloadArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMODULEUNLOAD;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuModuleUnloadArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hmod = hmod;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuModuleUnload);
	return err;
}

CUresult cuStreamEndCapture(CUstream  hStream, CUgraph * phGraph)
{
	TALLY_SPD_LOG("cuStreamEndCapture hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuStreamEndCapture(hStream, phGraph);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuStreamEndCaptureArg), alignof(cuStreamEndCaptureArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUSTREAMENDCAPTURE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuStreamEndCaptureArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hStream = hStream;
			request->phGraph = phGraph;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuStreamEndCaptureResponse*>(responsePayload);
			if (phGraph) { *phGraph = response->phGraph; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuStreamEndCapture);
	return err;
}

cudaError_t cudaStreamEndCapture(cudaStream_t  stream, cudaGraph_t * pGraph)
{
	TALLY_SPD_LOG("cudaStreamEndCapture hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamEndCapture(stream, pGraph);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamEndCaptureArg), alignof(cudaStreamEndCaptureArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMENDCAPTURE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamEndCaptureArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;
			request->pGraph = pGraph;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamEndCaptureResponse*>(responsePayload);
			if (pGraph) { *pGraph = response->pGraph; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamEndCapture);
	return err;
}

CUresult cuGraphLaunch(CUgraphExec  hGraphExec, CUstream  hStream)
{
	TALLY_SPD_LOG("cuGraphLaunch hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuGraphLaunch(hGraphExec, hStream);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuGraphLaunchArg), alignof(cuGraphLaunchArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUGRAPHLAUNCH;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuGraphLaunchArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hGraphExec = hGraphExec;
			request->hStream = hStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuGraphLaunch);
	return err;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t  handle, cublasMath_t  mode)
{
	TALLY_SPD_LOG("cublasSetMathMode hooked");
	TALLY_CLIENT_PROFILE_START;

    cublas_tracer.handle_cublasSetMathMode(handle, mode);

#if defined(RUN_LOCALLY)
	auto err = lcublasSetMathMode(handle, mode);
#else

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetMathModeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETMATHMODE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetMathModeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->mode = mode;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });
    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasSetMathMode);
	return err;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t  handle)
{
	TALLY_LOG("cublasDestroy_v2 hooked");
	TALLY_CLIENT_PROFILE_START;

    cublas_tracer.handle_cublasDestroy_v2(handle);

#if defined(RUN_LOCALLY)
	auto err = lcublasDestroy_v2(handle);
#else

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasDestroy_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASDESTROY_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasDestroy_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });
    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasDestroy_v2);
	return err;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t  handle, cudaStream_t  streamId)
{
	TALLY_SPD_LOG("cublasSetStream_v2 hooked");
	TALLY_CLIENT_PROFILE_START;

    cublas_tracer.handle_cublasSetStream_v2(handle, streamId);

#if defined(RUN_LOCALLY)
	auto err = lcublasSetStream_v2(handle, streamId);
#else

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetStream_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETSTREAM_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetStream_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->streamId = streamId;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });
    IOX_RECV_RETURN_STATUS(cublasStatus_t);
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasSetStream_v2);
	return err;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t  handle, void*  workspace, size_t  workspaceSizeInBytes)
{
	TALLY_SPD_LOG("cublasSetWorkspace_v2 hooked");
	TALLY_CLIENT_PROFILE_START;

    cublas_tracer.handle_cublasSetWorkspace_v2(handle, workspace, workspaceSizeInBytes);

#if defined(RUN_LOCALLY)
	auto err = lcublasSetWorkspace_v2(handle, workspace, workspaceSizeInBytes);
#else

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetWorkspace_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETWORKSPACE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetWorkspace_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->workspace = workspace;
			request->workspaceSizeInBytes = workspaceSizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    IOX_RECV_RETURN_STATUS(cublasStatus_t);

#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasSetWorkspace_v2);
	return err;
}

cublasStatus_t cublasLtCreate(cublasLtHandle_t*  lightHandle)
{
	TALLY_SPD_LOG("cublasLtCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtCreate(lightHandle);
    cublasLt_tracer.handle_cublasLtCreate(*lightHandle);
#else

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtCreateArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->lightHandle = lightHandle;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtCreateResponse*>(responsePayload);
			if (lightHandle) { *lightHandle = response->lightHandle; }

            err = response->err;
            cublasLt_tracer.handle_cublasLtCreate(response->lightHandle);

            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtCreate);
	return err;
}

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t*  matmulDesc, cublasComputeType_t  computeType, cudaDataType_t  scaleType)
{
	TALLY_SPD_LOG("cublasLtMatmulDescCreate hooked");
	TALLY_CLIENT_PROFILE_START;

#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatmulDescCreate(matmulDesc, computeType, scaleType);
    cublasLtMatmulDesc_tracer.handle_cublasLtMatmulDescCreate(*matmulDesc, computeType, scaleType);
#else

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatmulDescCreateArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATMULDESCCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatmulDescCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->matmulDesc = matmulDesc;
			request->computeType = computeType;
			request->scaleType = scaleType;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtMatmulDescCreateResponse*>(responsePayload);
			if (matmulDesc) { *matmulDesc = response->matmulDesc; }

            err = response->err;
            cublasLtMatmulDesc_tracer.handle_cublasLtMatmulDescCreate(response->matmulDesc, computeType, scaleType);

            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};

#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulDescCreate);
	return err;
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t*  matLayout, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld)
{
	TALLY_SPD_LOG("cublasLtMatrixLayoutCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatrixLayoutCreate(matLayout, type, rows, cols, ld);
    cublasLtMatrixLayout_tracer.handle_cublasLtMatrixLayoutCreate(*matLayout, type, rows, cols, ld);
#else

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatrixLayoutCreateArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatrixLayoutCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->matLayout = matLayout;
			request->type = type;
			request->rows = rows;
			request->cols = cols;
			request->ld = ld;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtMatrixLayoutCreateResponse*>(responsePayload);
			if (matLayout) { *matLayout = response->matLayout; }

            cublasLtMatrixLayout_tracer.handle_cublasLtMatrixLayoutCreate(response->matLayout, type, rows, cols, ld);

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatrixLayoutCreate);
	return err;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t*  pref)
{
	TALLY_SPD_LOG("cublasLtMatmulPreferenceCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatmulPreferenceCreate(pref);
#else

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatmulPreferenceCreateArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCECREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatmulPreferenceCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pref = pref;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtMatmulPreferenceCreateResponse*>(responsePayload);
			if (pref) { *pref = response->pref; }
            cublasLtMatmulPreference_tracer.handle_cublasLtMatmulPreferenceCreate(response->pref);

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulPreferenceCreate);
	return err;
}

}