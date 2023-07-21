#include <cstring>
#include <dlfcn.h>
#include <cassert>
#include <unordered_set>

#include "spdlog/spdlog.h"

#include <tally/transform.h>
#include <tally/util.h>
#include <tally/ipc_util.h>
#include <tally/cuda_util.h>
#include <tally/log.h>
#include <tally/msg_struct.h>
#include <tally/cache.h>
#include <tally/cache_util.h>
#include <tally/generated/cuda_api.h>
#include <tally/generated/msg_struct.h>
#include <tally/generated/server.h>

TallyServer *TallyServer::server = new TallyServer();

TallyServer::TallyServer()
{
    register_api_handler();

    __exit = [&](int sig_num) {
        is_quit__.store(true, std::memory_order_release);

        if (sig_num == SIGSEGV) {
            spdlog::info("Tally server received segfault signal.");
        }

        spdlog::info("Tally server shutting down ...");
        exit(0);
    };

    signal(SIGINT  , __exit_wrapper);
    signal(SIGABRT , __exit_wrapper);
    signal(SIGSEGV , __exit_wrapper);
    signal(SIGTERM , __exit_wrapper);
    signal(SIGHUP  , __exit_wrapper);
}

TallyServer::~TallyServer(){}

void TallyServer::start() {

    iox::runtime::PoshRuntime::initRuntime(APP_NAME);
    iox_server = new iox::popo::UntypedServer({"Example", "Request-Response", "Add"});

    load_cache();

    spdlog::info("Tally server is up ...");

    while (!iox::posix::hasTerminationRequested())
    {
        //! [take request]
        iox_server->take().and_then([&](auto& requestPayload) {
            auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
            auto handler = cuda_api_handler_map[msg_header->api_id];

            void *args = (void *) (static_cast<const uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
            handler(args, requestPayload);

            iox_server->releaseRequest(requestPayload);
        });
    }
}

void TallyServer::start_launcher()
{
    std::function<void()> kernel_partial;

    while (!iox::posix::hasTerminationRequested()) {
        if (launch_queue.wait_dequeue_timed(kernel_partial, std::chrono::milliseconds(100))) {
            kernel_partial();
            queue_size--;
        }
    }
}

void TallyServer::wait_until_launch_queue_empty()
{
    while(queue_size > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

void TallyServer::load_cache()
{
    std::map<size_t, std::vector<CubinData>> cubin_map = TallyCache::cache->cubin_cache.cubin_map;

    for (auto &pair : cubin_map) {
        uint32_t cubin_size = pair.first;
        auto &cubin_vec = pair.second;

        for (auto &cubin_data : cubin_vec) {
            const my__fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) = {
                cubin_data.magic,
                cubin_data.version,
                (const long long unsigned int*) cubin_data.cubin_data.c_str(),
                0
            };
            void **handle = l__cudaRegisterFatBinary((void *) &__fatDeviceText);

            auto kernel_args = cubin_data.kernel_args;

            for (auto &kernel_args_pair : cubin_data.kernel_args) {

                auto &kernel_name = kernel_args_pair.first;
                auto &param_sizes = kernel_args_pair.second;

                // allocate an address for the kernel
                void *kernel_server_addr = malloc(8);

                // Register the kernel with this address
                _kernel_name_to_addr[kernel_name] = kernel_server_addr;
                l__cudaRegisterFunction(handle, (const char*) kernel_server_addr, (char *)kernel_name.c_str(), kernel_name.c_str(), -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);
            
                _kernel_addr_to_args[kernel_server_addr] = param_sizes;
            }

            l__cudaRegisterFatBinaryEnd(handle);

            // For some reason, must call one cuda api call here. Otherwise it won't run.
            int *arr;
            cudaMalloc((void**)&arr, sizeof(int));
            cudaFree(arr);
        }
    }
}

void TallyServer::handle___cudaRegisterFatBinary(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFatBinary");
    auto args = (__cudaRegisterFatBinaryArg *) __args;
    bool cached = args->cached;
    magic = args->magic;
    version = args->version;

    struct fatBinaryHeader *header = (struct fatBinaryHeader *) args->data;
    size_t cubin_size = header->headerSize + header->fatSize;
    const char *cubin_data = (const char *) args->data;

    if (cached) {
        cubin_registered = true;
    } else {
        cubin_registered = TallyCache::cache->cubin_cache.contains(cubin_data, cubin_size);
    }

    if (cubin_registered) {
        TALLY_SPD_LOG("Fat binary exists in cache, skipping register");
    }

    // Free data from last time
    // TODO: change this to managed pointer
    if (fatbin_data) {
        free(fatbin_data);
        fatbin_data = nullptr;
    }

    register_queue.clear();

    if (!cubin_registered) {
        // Load necessary data into cache if not exists
        cache_cubin_data(cubin_data, cubin_size, magic, version);

        fatBinSize = cubin_size;
        fatbin_data = (unsigned long long *) malloc(cubin_size);
        memcpy(fatbin_data, args->data, cubin_size);

        std::string tmp_elf_file_name = get_tmp_file_path(".elf");
        auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
        iox_server->loan(requestHeader, tmp_elf_file_name.size() + 1, alignof(char[]))
            .and_then([&](auto& responsePayload) {

                auto response = static_cast<char *>(responsePayload);
                memcpy(response, tmp_elf_file_name.c_str(), tmp_elf_file_name.size());
                response[tmp_elf_file_name.size()] = '\0';

                iox_server->send(response).or_else(
                    [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
            })
            .or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
    }
}

void TallyServer::handle___cudaRegisterFunction(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFunction");
    auto args = (struct registerKernelArg *) __args;

    std::string kernel_name {args->data, args->kernel_func_len};
    register_queue.push_back( std::make_pair(args->host_func, kernel_name));
}

void TallyServer::handle___cudaRegisterFatBinaryEnd(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFatBinaryEnd");

    void **handle;
    void *kernel_server_addr;
    my__fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8)));
    std::map<std::string, std::vector<uint32_t>> kernel_names_and_param_sizes;

    if (!cubin_registered) {
        __fatDeviceText = { magic, version, fatbin_data, 0 };
        handle = l__cudaRegisterFatBinary((void *) &__fatDeviceText);
        kernel_names_and_param_sizes = TallyCache::cache->cubin_cache.get_kernel_args((const char*) fatbin_data, fatBinSize);
    }

    for (auto &kernel_pair : register_queue) {
        auto &client_addr = kernel_pair.first;
        auto &kernel_name = kernel_pair.second;
        
        if (!cubin_registered) {
            // allocate an address for the kernel
            kernel_server_addr = malloc(8);

            auto &param_sizes = kernel_names_and_param_sizes[kernel_name];

            // Register the kernel with this address
            _kernel_name_to_addr[kernel_name] = kernel_server_addr;
            _kernel_addr_to_args[kernel_server_addr] = param_sizes;
            l__cudaRegisterFunction(handle, (const char*) kernel_server_addr, (char *)kernel_name.c_str(), kernel_name.c_str(), -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);
        }

        // For now, hoping that the client addr does not appear more than once
        // TODO: fix this
        if (_kernel_client_addr_mapping.find(client_addr) != _kernel_client_addr_mapping.end()) {
            assert(_kernel_client_addr_mapping[client_addr] = _kernel_name_to_addr[kernel_name]);
        } else {
            // Associate this client addr with the server address
            _kernel_client_addr_mapping[client_addr] = _kernel_name_to_addr[kernel_name];
        }
    }

    if (!cubin_registered) {
        l__cudaRegisterFatBinaryEnd(handle);
    }

    // For some reason, must call one cuda api call here. Otherwise it won't run.
    int *arr;
    cudaMalloc((void**)&arr, sizeof(int));
    cudaFree(arr);
}

void TallyServer::handle_cudaMalloc(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaMalloc");
	auto args = (struct cudaMallocArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaMallocResponse), alignof(cudaMallocResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaMallocResponse*>(responsePayload);
            response->err = cudaMalloc(&(response->devPtr), args->size);

            // Keep track that this addr is device memory
            if (response->err == cudaSuccess) {
                dev_addr_map.push_back( DeviceMemoryKey(response->devPtr, args->size) );
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaFree(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaFree");
    auto args = (struct cudaMallocArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaFreeArg), alignof(cudaFreeArg))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaFree(args->devPtr);

            if (*response == cudaSuccess) {
                free_dev_addr(dev_addr_map, args->devPtr);
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaMemcpy(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaMemcpy");

    auto args = (struct cudaMemcpyArg *) __args;
    size_t res_size;

    if (args->kind == cudaMemcpyHostToDevice) {
        res_size = sizeof(cudaError_t);
    } else if (args->kind == cudaMemcpyDeviceToHost){
        res_size = sizeof(cudaError_t) + args->count;
    } else if (args->kind == cudaMemcpyDeviceToDevice) {
        res_size = sizeof(cudaError_t);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_size, alignof(cudaMemcpyResponse))
        .and_then([&](auto& responsePayload) {
            auto res = static_cast<cudaMemcpyResponse*>(responsePayload);

            if (args->kind == cudaMemcpyHostToDevice) {
                res->err = cudaMemcpy(args->dst, args->data, args->count, args->kind);
            } else if (args->kind == cudaMemcpyDeviceToHost){
                res->err = cudaMemcpy(res->data, args->src, args->count, args->kind);
            } else if (args->kind == cudaMemcpyDeviceToDevice) {
                res->err = cudaMemcpy(args->dst, args->src, args->count, args->kind);
            } else {
                throw std::runtime_error("Unknown memcpy kind!");
            }

            iox_server->send(res).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaMemcpyAsync(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaMemcpyAsync");
    
    auto args = (struct cudaMemcpyAsyncArg *) __args;
    size_t res_size;

    if (args->kind == cudaMemcpyHostToDevice) {
        res_size = sizeof(cudaError_t);
    } else if (args->kind == cudaMemcpyDeviceToHost){
        res_size = sizeof(cudaError_t) + args->count;
    } else if (args->kind == cudaMemcpyDeviceToDevice) {
        res_size = sizeof(cudaError_t);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_size, alignof(cudaMemcpyAsyncResponse))
        .and_then([&](auto& responsePayload) {
            auto res = static_cast<cudaMemcpyAsyncResponse*>(responsePayload);

            if (args->kind == cudaMemcpyHostToDevice) {
                res->err = cudaMemcpyAsync(args->dst, args->data, args->count, args->kind);
            } else if (args->kind == cudaMemcpyDeviceToHost){
                res->err = cudaMemcpyAsync(res->data, args->src, args->count, args->kind);
            } else if (args->kind == cudaMemcpyDeviceToDevice) {
                res->err = cudaMemcpyAsync(args->dst, args->src, args->count, args->kind);
            } else {
                throw std::runtime_error("Unknown memcpy kind!");
            }

            iox_server->send(res).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) {LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaLaunchKernel(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaLaunchKernel");
    auto args = (cudaLaunchKernelArg *) __args;
    auto partial = cudaLaunchKernel_Partial(args->host_func, args->gridDim, args->blockDim, args->sharedMem, args->stream, args->params);
    queue_size++;
    launch_queue.enqueue(partial);
    // partial();
}

void TallyServer::handle_cublasSgemm_v2(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasSgemm_v2");
    auto args = (struct cublasSgemm_v2Arg *) __args;

    const float alpha = args->alpha;
    const float beta = args->beta;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cublasSgemm_v2(
                args->handle,
                args->transa,
                args->transb,
                args->m,
                args->n,
                args->k,
                &alpha,
                args->A,
                args->lda,
                args->B,
                args->ldb,
                &beta,
                args->C,
                args->ldc
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatmul(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasLtMatmul");
    auto args = (struct cublasLtMatmulArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cublasLtMatmul(
                args->lightHandle,
                args->computeDesc,
                (void *) &(args->alpha),
                args->A,
                args->Adesc,
                args->B,
                args->Bdesc,
                (void *) &(args->beta),
                args->C,
                args->Cdesc,
                args->D,
                args->Ddesc,
                &(args->algo),
                args->workspace,
                args->workspaceSizeInBytes,
                args->stream
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatmulDescSetAttribute(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasLtMatmulDescSetAttribute");
    auto args = (struct cublasLtMatmulDescSetAttributeArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatmulDescSetAttribute(
                args->matmulDesc,
                args->attr,
                (void *)args->buf,
                args->sizeInBytes
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatrixLayoutSetAttribute(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasLtMatrixLayoutSetAttribute");
    auto args = (struct cublasLtMatrixLayoutSetAttributeArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatrixLayoutSetAttribute(
                args->matLayout,
                args->attr,
                (void *)args->buf,
                args->sizeInBytes
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatmulPreferenceSetAttribute(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasLtMatmulPreferenceSetAttribute");
    auto args = (struct cublasLtMatmulPreferenceSetAttributeArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatmulPreferenceSetAttribute(
                args->pref,
                args->attr,
                (void *)args->buf,
                args->sizeInBytes
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatmulAlgoGetHeuristic(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasLtMatmulAlgoGetHeuristic");
    auto args = (struct cublasLtMatmulAlgoGetHeuristicArg *) __args;

    size_t res_len = sizeof(cublasLtMatmulAlgoGetHeuristicResponse) + sizeof(cublasLtMatmulHeuristicResult_t) * args->requestedAlgoCount;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cublasLtMatmulAlgoGetHeuristicResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasLtMatmulAlgoGetHeuristicResponse*>(responsePayload);
            response->err = cublasLtMatmulAlgoGetHeuristic(
                args->lightHandle,
                args->operationDesc,
                args->Adesc,
                args->Bdesc,
                args->Cdesc,
                args->Ddesc,
                args->preference,
                args->requestedAlgoCount,
                response->heuristicResultsArray,
                &(response->returnAlgoCount)
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnBackendSetAttribute(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnBackendSetAttribute");
    auto args = (struct cudnnBackendSetAttributeArg *) __args;

    // In case the values contain CPU pointers
    // Then we will allocate again
    std::vector <void *> allocated_mem;

    if (args->attributeType == CUDNN_TYPE_VOID_PTR) {
        auto pointer_arr = (void **) (args->arrayOfElements);

        for (int i = 0; i < args->elementCount; i++) {
            auto pointer = pointer_arr[i];

            if (pointer == nullptr) {
                continue;
            }
            
            auto found = is_dev_addr(dev_addr_map, pointer);

            // pointer points to CPU memory
            if (!found) {
                uint64_t val = (uint64_t) pointer;

                // Store the value instead of addr
                pointer_arr[i] = std::malloc(sizeof(uint64_t));
                *((uint64_t *) (pointer_arr[i])) = val;
                allocated_mem.push_back(pointer_arr[i]);
            }
        }
    }

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnBackendSetAttribute(
                args->descriptor,
                args->attributeName,
                args->attributeType,
                args->elementCount,
                args->arrayOfElements
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });

    // These memory can be free once cudnnBackendSetAttribute returns
    for (auto &addr : allocated_mem) {
        std::free(addr);
    }
}

void TallyServer::handle_cudnnBackendGetAttribute(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnBackendGetAttribute");
    auto args = (struct cudnnBackendGetAttributeArg *) __args;

    int32_t type_size = get_cudnn_attribute_size(args->attributeType);
    int32_t buf_size = type_size * args->requestedElementCount;
    assert(buf_size >= 0);

    void *arrayOfElements;

    if (buf_size == 0) {
        arrayOfElements = NULL;
    } else {
        arrayOfElements = args->arrayOfElementsData;
    }

    int64_t elementCount = 0;
    size_t res_len = sizeof(cudnnBackendGetAttributeResponse) + buf_size;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnBackendGetAttributeResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnBackendGetAttributeResponse*>(responsePayload);
            response->err = cudnnBackendGetAttribute(
                args->descriptor,
                args->attributeName,
                args->attributeType,
                args->requestedElementCount,
                args->elementCount ? (&elementCount) : NULL,
                args->arrayOfElements ? (arrayOfElements) : NULL
            );

            int64_t arrayOfElementsSize;
            if (args->elementCount) {
                arrayOfElementsSize = std::min(elementCount, args->requestedElementCount);
            } else {
                arrayOfElementsSize = args->requestedElementCount;
            }

            response->arrayOfElementsSize = arrayOfElementsSize;
            response->elementCount = elementCount;
            if (arrayOfElements) {
                memcpy(response->arrayOfElements, arrayOfElements, type_size * arrayOfElementsSize);
            } else {
                assert(arrayOfElementsSize <= 0);
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnActivationForward(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnActivationForward");
    auto args = (struct cudnnActivationForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnActivationForward(
                args->handle,
                args->activationDesc,
                (void *) &(args->alpha),
                args->xDesc,
                args->x,
                (void *) &(args->beta),
                args->yDesc,
                args->y
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}


void TallyServer::handle_cudnnSetTensorNdDescriptor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnSetTensorNdDescriptor");
    auto args = (struct cudnnSetTensorNdDescriptorArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetTensorNdDescriptor(
                args->tensorDesc,
                args->dataType,
                args->nbDims,
                args->dimA_and_strideA,
                args->dimA_and_strideA + args->nbDims
            );

            assert(*response == CUDNN_STATUS_SUCCESS);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnSetConvolutionNdDescriptor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnSetConvolutionNdDescriptor");
    auto args = (struct cudnnSetConvolutionNdDescriptorArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetConvolutionNdDescriptor(
                args->convDesc,
                args->arrayLength,
                args->padA_and_filterStrideA_and_dilationA,
                args->padA_and_filterStrideA_and_dilationA + args->arrayLength,
                args->padA_and_filterStrideA_and_dilationA + args->arrayLength * 2,
                args->mode,
                args->computeType
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnSetFilterNdDescriptor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnSetFilterNdDescriptor");
    auto args = (struct cudnnSetFilterNdDescriptorArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetFilterNdDescriptor(
                args->filterDesc,
                args->dataType,
                args->format,
                args->nbDims,
                args->filterDimA
            );

            assert(*response == CUDNN_STATUS_SUCCESS);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnConvolutionForward(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnConvolutionForward");
    auto args = (struct cudnnConvolutionForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnConvolutionForward(
                args->handle,
                (void *) &(args->alpha),
                args->xDesc,
                args->x,
                args->wDesc,
                args->w,
                args->convDesc,
                args->algo,
                args->workSpace,
                args->workSpaceSizeInBytes,
                (void *) &(args->beta),
                args->yDesc,
                args->y
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetConvolutionNdForwardOutputDim(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnGetConvolutionNdForwardOutputDim");
    auto args = (struct cudnnGetConvolutionNdForwardOutputDimArg *) __args;

    uint32_t res_len = sizeof(cudnnGetConvolutionNdForwardOutputDimResponse) + sizeof(int) * args->nbDims;
    
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnGetConvolutionNdForwardOutputDimResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnGetConvolutionNdForwardOutputDimResponse*>(responsePayload);
            response->err = cudnnGetConvolutionNdForwardOutputDim(
                args->convDesc,
                args->inputTensorDesc,
                args->filterDesc,
                args->nbDims,
                response->tensorOuputDimA
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetConvolutionForwardAlgorithm_v7(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnGetConvolutionForwardAlgorithm_v7");
    auto args = (struct cudnnGetConvolutionForwardAlgorithm_v7Arg *) __args;

    uint32_t res_len = sizeof(cudnnGetConvolutionForwardAlgorithm_v7Response) + sizeof(cudnnConvolutionFwdAlgoPerf_t) * args->requestedAlgoCount;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnGetConvolutionForwardAlgorithm_v7Response))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnGetConvolutionForwardAlgorithm_v7Response*>(responsePayload);
            response->err = cudnnGetConvolutionForwardAlgorithm_v7(
                args->handle,
                args->srcDesc,
                args->filterDesc,
                args->convDesc,
                args->destDesc,
                args->requestedAlgoCount,
                &response->returnedAlgoCount,
                response->perfResults
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnFindConvolutionForwardAlgorithm(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnFindConvolutionForwardAlgorithm");
    auto args = (struct cudnnFindConvolutionForwardAlgorithmArg *) __args;

    uint32_t res_len = sizeof(cudnnFindConvolutionForwardAlgorithmResponse) + sizeof(cudnnConvolutionFwdAlgoPerf_t) * args->requestedAlgoCount;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnFindConvolutionForwardAlgorithmResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnFindConvolutionForwardAlgorithmResponse*>(responsePayload);
            response->err = cudnnFindConvolutionForwardAlgorithm(
                args->handle,
                args->xDesc,
                args->wDesc,
                args->convDesc,
                args->yDesc,
                args->requestedAlgoCount,
                &response->returnedAlgoCount,
                response->perfResults
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnAddTensor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnAddTensor");
    auto args = (struct cudnnAddTensorArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnAddTensor(
                args->handle,
                (void *) &(args->alpha),
                args->aDesc,
                args->A,
                (void *) &(args->beta),
                args->cDesc,
                args->C
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnSetPoolingNdDescriptor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnSetPoolingNdDescriptor");
    auto args = (struct cudnnSetPoolingNdDescriptorArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetPoolingNdDescriptor(
                args->poolingDesc,
                args->mode,
                args->maxpoolingNanOpt,
                args->nbDims,
                args->windowDimA_paddingA_strideA,
                args->windowDimA_paddingA_strideA + args->nbDims,
                args->windowDimA_paddingA_strideA + args->nbDims * 2
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetPoolingNdDescriptor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnGetPoolingNdDescriptor");
    auto args = (struct cudnnGetPoolingNdDescriptorArg *) __args;

    size_t res_len = sizeof(cudnnGetPoolingNdDescriptorResponse) + sizeof(int) * args->nbDimsRequested * 3;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnGetPoolingNdDescriptorResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnGetPoolingNdDescriptorResponse*>(responsePayload);
            response->err = cudnnGetPoolingNdDescriptor(
                args->poolingDesc,
                args->nbDimsRequested,
                &(response->mode),
                &(response->maxpoolingNanOpt),
                &(response->nbDims),
                response->windowDimA_paddingA_strideA,
                response->windowDimA_paddingA_strideA + args->nbDimsRequested,
                response->windowDimA_paddingA_strideA + args->nbDimsRequested * 2
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetPoolingNdForwardOutputDim(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnGetPoolingNdForwardOutputDim");
    auto args = (struct cudnnGetPoolingNdForwardOutputDimArg *) __args;

    uint32_t res_len = sizeof(cudnnGetPoolingNdForwardOutputDimResponse) + sizeof(int) * args->nbDims;
    
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnGetPoolingNdForwardOutputDimResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnGetPoolingNdForwardOutputDimResponse*>(responsePayload);
            response->err = cudnnGetPoolingNdForwardOutputDim(
                args->poolingDesc,
                args->inputTensorDesc,
                args->nbDims,
                response->outputTensorDimA
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnPoolingForward(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnPoolingForward");
    auto args = (struct cudnnPoolingForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnPoolingForward(
                args->handle,
                args->poolingDesc,
                (void *) &(args->alpha),
                args->xDesc,
                args->x,
                (void *) &(args->beta),
                args->yDesc,
                args->y
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasSgemv_v2(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasSgemv_v2");
    auto args = (struct cublasSgemv_v2Arg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cublasSgemv_v2(
                args->handle,
                args->trans,
                args->m,
                args->n,
                &args->alpha,
                args->A,
                args->lda,
                args->x,
                args->incx,
                &args->beta,
                args->y,
                args->incy
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnLRNCrossChannelForward(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnLRNCrossChannelForward");
    auto args = (struct cudnnLRNCrossChannelForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnLRNCrossChannelForward(
                args->handle,
                args->normDesc,
                args->lrnMode,
                &(args->alpha),
                args->xDesc,
                args->x,
                &(args->beta),
                args->yDesc,
                args->y
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnSoftmaxForward(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnSoftmaxForward");
    auto args = (struct cudnnSoftmaxForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnSoftmaxForward(
                args->handle,
                args->algo,
                args->mode,
                &(args->alpha),
                args->xDesc,
                args->x,
                &(args->beta),
                args->yDesc,
                args->y
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnTransformTensor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnTransformTensor");
    auto args = (struct cudnnTransformTensorArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnTransformTensor(
                args->handle,
                &(args->alpha),
                args->xDesc,
                args->x,
                &(args->beta),
                args->yDesc,
                args->y
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasSgemmEx(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasSgemmEx");
    auto args = (struct cublasSgemmExArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cublasSgemmEx(
                args->handle,
                args->transa,
                args->transb,
                args->m,
                args->n,
                args->k,
                &(args->alpha),
                args->A,
                args->Atype,
                args->lda,
                args->B,
                args->Btype,
                args->ldb,
                &(args->beta),
                args->C,
                args->Ctype,
                args->ldc
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnSetSeqDataDescriptor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnSetSeqDataDescriptor");
    auto args = (struct cudnnSetSeqDataDescriptorArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetSeqDataDescriptor(
                args->seqDataDesc,
                args->dataType,
                args->nbDims,
                args->dimA,
                args->axes,
                args->seqLengthArraySize,
                args->seqLengthArray,
                const_cast<void *>(args->paddingFill)
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetSeqDataDescriptor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnGetSeqDataDescriptor");
    auto args = (struct cudnnGetSeqDataDescriptorArg *) __args;

    uint32_t res_len = sizeof(cudnnGetSeqDataDescriptorResponse) + sizeof(int) * args->nbDimsRequested + sizeof(cudnnSeqDataAxis_t) * args->nbDimsRequested + sizeof(int) * args->seqLengthSizeRequested;
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnGetSeqDataDescriptorResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnGetSeqDataDescriptorResponse*>(responsePayload);
            response->err = cudnnGetSeqDataDescriptor(
                args->seqDataDesc,
                &(response->dataType),
                &(response->nbDims),
                args->nbDimsRequested,
                (int *) (response->dimA_axes_seqLengthArray),
                (cudnnSeqDataAxis_t*) (response->dimA_axes_seqLengthArray + sizeof(int) * args->nbDimsRequested),
                &(response->seqLengthArraySize),
                args->seqLengthSizeRequested,
                (int *) (response->dimA_axes_seqLengthArray + sizeof(int) * args->nbDimsRequested + sizeof(cudnnSeqDataAxis_t) * args->nbDimsRequested),
                const_cast<void *>(args->paddingFill)
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnMultiHeadAttnForward(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnMultiHeadAttnForward");
    auto args = (struct cudnnMultiHeadAttnForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnMultiHeadAttnForward(
                args->handle,
                args->attnDesc,
                args->currIdx,
                args->loWinIdx_hiWinIdx,
                args->loWinIdx_hiWinIdx + args->winIdxLen,
                args->devSeqLengthsQO,
                args->devSeqLengthsKV,
                args->qDesc,
                args->queries,
                args->residuals,
                args->kDesc,
                args->keys,
                args->vDesc,
                args->values,
                args->oDesc,
                args->out,
                args->weightSizeInBytes,
                args->weights,
                args->workSpaceSizeInBytes,
                args->workSpace,
                args->reserveSpaceSizeInBytes,
                args->reserveSpace
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnMultiHeadAttnBackwardData(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnMultiHeadAttnBackwardData");
    auto args = (struct cudnnMultiHeadAttnBackwardDataArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnMultiHeadAttnBackwardData(
                args->handle,
                args->attnDesc,
                args->loWinIdx_hiWinIdx,
                args->loWinIdx_hiWinIdx + args->winIdxLen,
                args->devSeqLengthsDQDO,
                args->devSeqLengthsDKDV,
                args->doDesc,
                args->dout,
                args->dqDesc,
                args->dqueries,
                args->queries,
                args->dkDesc,
                args->dkeys,
                args->keys,
                args->dvDesc,
                args->dvalues,
                args->values,
                args->weightSizeInBytes,
                args->weights,
                args->workSpaceSizeInBytes,
                args->workSpace,
                args->reserveSpaceSizeInBytes,
                args->reserveSpace
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); }); 
}

void TallyServer::handle_cudnnMultiHeadAttnBackwardWeights(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnMultiHeadAttnBackwardWeights");
    auto args = (struct cudnnMultiHeadAttnBackwardWeightsArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnMultiHeadAttnBackwardWeights(
                args->handle,
                args->attnDesc,
                args->addGrad,
                args->qDesc,
                args->queries,
                args->kDesc,
                args->keys,
                args->vDesc,
                args->values,
                args->doDesc,
                args->dout,
                args->weightSizeInBytes,
                args->weights,
                args->dweights,
                args->workSpaceSizeInBytes,
                args->workSpace,
                args->reserveSpaceSizeInBytes,
                args->reserveSpace
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnReorderFilterAndBias(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnReorderFilterAndBias");
    auto args = (struct cudnnReorderFilterAndBiasArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnReorderFilterAndBias(
                args->handle,
                args->filterDesc,
                args->reorderType,
                args->filterData,
                args->reorderedFilterData,
                args->reorderBias,
                args->biasData,
                args->reorderedBiasData
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetRNNWorkspaceSize(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnGetRNNWorkspaceSize");
    auto args = (struct cudnnGetRNNWorkspaceSizeArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnGetRNNWorkspaceSizeResponse), alignof(cudnnGetRNNWorkspaceSizeResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnGetRNNWorkspaceSizeResponse*>(responsePayload);
            response->err = cudnnGetRNNWorkspaceSize(
                args->handle,
                args->rnnDesc,
                args->seqLength,
                args->xDesc,
                &(response->sizeInBytes)
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetRNNTrainingReserveSize(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnGetRNNTrainingReserveSize");
    auto args = (struct cudnnGetRNNTrainingReserveSizeArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnGetRNNTrainingReserveSizeResponse), alignof(cudnnGetRNNTrainingReserveSizeResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnGetRNNTrainingReserveSizeResponse*>(responsePayload);

            response->err = cudnnGetRNNTrainingReserveSize(
                args->handle,
                args->rnnDesc,
                args->seqLength,
                args->xDesc,
                &(response->sizeInBytes)
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetFilterNdDescriptor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnGetFilterNdDescriptor");
    auto args = (struct cudnnGetFilterNdDescriptorArg *) __args;

    size_t res_len = sizeof(cudnnGetFilterNdDescriptorResponse) + args->nbDimsRequested * sizeof(int);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnGetFilterNdDescriptorResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnGetFilterNdDescriptorResponse*>(responsePayload);
            response->err = cudnnGetFilterNdDescriptor(
                args->filterDesc,
                args->nbDimsRequested,
                &response->dataType,
                &response->format,
                &response->nbDims,
                response->filterDimA
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnRNNForwardTraining(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnRNNForwardTraining");
    auto args = (struct cudnnRNNForwardTrainingArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnRNNForwardTraining(
                args->handle,
                args->rnnDesc,
                args->seqLength,
                args->xDesc_yDesc,
                args->x,
                args->hxDesc,
                args->hx,
                args->cxDesc,
                args->cx,
                args->wDesc,
                args->w,
                args->xDesc_yDesc + args->seqLength,
                args->y,
                args->hyDesc,
                args->hy,
                args->cyDesc,
                args->cy,
                args->workSpace,
                args->workSpaceSizeInBytes,
                args->reserveSpace,
                args->reserveSpaceSizeInBytes
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnRNNBackwardData(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnRNNBackwardData");
    auto args = (struct cudnnRNNBackwardDataArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnRNNBackwardData(
                args->handle,
                args->rnnDesc,
                args->seqLength,
                args->yDesc_dyDesc_dxDesc,
                args->y,
                args->yDesc_dyDesc_dxDesc + args->seqLength,
                args->dy,
                args->dhyDesc,
                args->dhy, 
                args->dcyDesc, 
                args->dcy, 
                args->wDesc, 
                args->w, 
                args->hxDesc, 
                args->hx, 
                args->cxDesc, 
                args->cx, 
                args->yDesc_dyDesc_dxDesc + args->seqLength * 2, 
                args->dx, 
                args->dhxDesc, 
                args->dhx, 
                args->dcxDesc, 
                args->dcx, 
                args->workSpace, 
                args->workSpaceSizeInBytes, 
                args->reserveSpace, 
                args->reserveSpaceSizeInBytes
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnRNNBackwardWeights(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnRNNBackwardWeights");
    auto args = (struct cudnnRNNBackwardWeightsArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnRNNBackwardWeights(
                args->handle,
                args->rnnDesc,
                args->seqLength,
                args->xDesc_yDesc,
                args->x,
                args->hxDesc,
                args->hx,
                args->xDesc_yDesc + args->seqLength,
                args->y,
                args->workSpace,
                args->workSpaceSizeInBytes,
                args->dwDesc,
                args->dw,
                args->reserveSpace,
                args->reserveSpaceSizeInBytes
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnSetRNNDataDescriptor(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetRNNDataDescriptor");
    auto args = (struct cudnnSetRNNDataDescriptorArg *) __args;

    void *paddingFill = NULL;
    uint64_t paddingFillVal = args->paddingFillVal;
    if (args->paddingFill) {
        paddingFill = (void *) &paddingFillVal;
    }

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetRNNDataDescriptor(
                args->rnnDataDesc,
                args->dataType,
                args->layout,
                args->maxSeqLength,
                args->batchSize,
                args->vectorSize,
                args->seqLengthArray,
                paddingFill
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetTensorNdDescriptor(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnGetTensorNdDescriptor");
    auto args = (struct cudnnGetTensorNdDescriptorArg *) __args;

    uint32_t res_len =  sizeof(cudnnGetTensorNdDescriptorResponse) + sizeof(int) * args->nbDimsRequested * 2;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnGetTensorNdDescriptorResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnGetTensorNdDescriptorResponse*>(responsePayload);
            response->err = cudnnGetTensorNdDescriptor(
                args->tensorDesc,
                args->nbDimsRequested,
                &response->dataType,
                &response->nbDims,
                response->dimA_strideA,
                response->dimA_strideA + args->nbDimsRequested
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnBatchNormalizationForwardTrainingEx(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnBatchNormalizationForwardTrainingEx");
    auto args = (struct cudnnBatchNormalizationForwardTrainingExArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnBatchNormalizationForwardTrainingEx(
                args->handle,
                args->mode,
                args->bnOps,
                (void *) &(args->alpha),
                (void *) &(args->beta),
                args->xDesc,
                args->xData,
                args->zDesc,
                args->zData,
                args->yDesc,
                args->yData,
                args->bnScaleBiasMeanVarDesc,
                args->bnScale,
                args->bnBias,
                args->exponentialAverageFactor,
                args->resultRunningMean,
                args->resultRunningVariance,
                args->epsilon,
                args->resultSaveMean,
                args->resultSaveInvVariance,
                args->activationDesc,
                args->workspace,
                args->workSpaceSizeInBytes,
                args->reserveSpace,
                args->reserveSpaceSizeInBytes
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnBatchNormalizationBackwardEx(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnBatchNormalizationBackwardEx");
    auto args = (struct cudnnBatchNormalizationBackwardExArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnBatchNormalizationBackwardEx(
                args->handle,
                args->mode,
                args->bnOps,
                (void *) &(args->alphaDataDiff),
                (void *) &(args->betaDataDiff),
                (void *) &(args->alphaParamDiff),
                (void *) &(args->betaParamDiff),
                args->xDesc,
                args->xData,
                args->yDesc,
                args->yData,
                args->dyDesc,
                args->dyData,
                args->dzDesc,
                args->dzData,
                args->dxDesc,
                args->dxData,
                args->dBnScaleBiasDesc,
                args->bnScaleData,
                args->bnBiasData,
                args->dBnScaleData,
                args->dBnBiasData,
                args->epsilon,
                args->savedMean,
                args->savedInvVariance,
                args->activationDesc,
                args->workSpace,
                args->workSpaceSizeInBytes,
                args->reserveSpace,
                args->reserveSpaceSizeInBytes
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });	
}


void TallyServer::handle_cublasSgemmStridedBatched(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasSgemmStridedBatched");
    auto args = (struct cublasSgemmStridedBatchedArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasSgemmStridedBatched(
                args->handle,
                args->transa,
                args->transb,
                args->m,
                args->n,
                args->k,
                &(args->alpha),
                args->A,
                args->lda,
                args->strideA,
                args->B,
                args->ldb,
                args->strideB,
                &(args->beta),
                args->C,
                args->ldc,
                args->strideC,
                args->batchCount
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaFuncGetAttributes(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaFuncGetAttributes");
	auto args = (struct cudaFuncGetAttributesArg *) __args;

    assert(_kernel_client_addr_mapping.find(args->func) != _kernel_client_addr_mapping.end());

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaFuncGetAttributesResponse), alignof(cudaFuncGetAttributesResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaFuncGetAttributesResponse*>(responsePayload);
            response->err = cudaFuncGetAttributes(&response->attr, _kernel_client_addr_mapping[args->func]);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *__args, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    auto args = (struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsArg *) __args;

    assert(_kernel_client_addr_mapping.find(args->func) != _kernel_client_addr_mapping.end());

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse), alignof(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse*>(responsePayload);
            response->err = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &response->numBlocks,
                _kernel_client_addr_mapping[args->func],
                args->blockSize,
                args->dynamicSMemSize,
                args->flags
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaChooseDevice(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaChooseDevice");
    auto args = (struct cudaChooseDeviceArg *) __args;
    
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaChooseDeviceResponse), alignof(cudaChooseDeviceResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaChooseDeviceResponse*>(responsePayload);
            response->err = cudaChooseDevice(
                &response->device,
                &args->prop
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaSetDevice(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaSetDevice");
	auto args = (struct cudaSetDeviceArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaSetDevice(
				args->device
            );
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnRNNBackwardWeights_v8(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNBackwardWeights_v8");
	auto args = (struct cudnnRNNBackwardWeights_v8Arg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnRNNBackwardWeights_v8(
				args->handle,
				args->rnnDesc,
				args->addGrad,
				args->devSeqLengths,
				args->xDesc,
				args->x,
				args->hDesc,
				args->hx,
				args->yDesc,
				args->y,
				args->weightSpaceSize,
				args->dweightSpace,
				args->workSpaceSize,
				args->workSpace,
				args->reserveSpaceSize,
				args->reserveSpace
            );
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnRNNBackwardData_v8(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNBackwardData_v8");
	auto args = (struct cudnnRNNBackwardData_v8Arg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnRNNBackwardData_v8(
				args->handle,
				args->rnnDesc,
				args->devSeqLengths,
				args->yDesc,
				args->y,
				args->dy,
				args->xDesc,
				args->dx,
				args->hDesc,
				args->hx,
				args->dhy,
				args->dhx,
				args->cDesc,
				args->cx,
				args->dcy,
				args->dcx,
				args->weightSpaceSize,
				args->weightSpace,
				args->workSpaceSize,
				args->workSpace,
				args->reserveSpaceSize,
				args->reserveSpace
            );
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnRNNForward(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNForward");
	auto args = (struct cudnnRNNForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnRNNForward(
				args->handle,
				args->rnnDesc,
				args->fwdMode,
				args->devSeqLengths,
				args->xDesc,
				args->x,
				args->yDesc,
				args->y,
				args->hDesc,
				args->hx,
				args->hy,
				args->cDesc,
				args->cx,
				args->cy,
				args->weightSpaceSize,
				args->weightSpace,
				args->workSpaceSize,
				args->workSpace,
				args->reserveSpaceSize,
				args->reserveSpace
            );
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnBackendExecute(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBackendExecute");
	auto args = (struct cudnnBackendExecuteArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudnnBackendExecute(
				args->handle,
				args->executionPlan,
				args->variantPack
            );
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaThreadSynchronize(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaThreadSynchronize");

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaError_t*>(responsePayload);
            wait_until_launch_queue_empty();
            *response = cudaThreadSynchronize();

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaEventRecord(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaEventRecord");
	auto args = (struct cudaEventRecordArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudaEventRecord(
				args->event,
				args->stream
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaDeviceSynchronize(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceSynchronize");
    
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaError_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudaDeviceSynchronize();

            if ((*response)) {
                throw std::runtime_error("cudaDeviceSynchronize fails");
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaStreamSynchronize(void *__args, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamSynchronize");
	auto args = (struct cudaStreamSynchronizeArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);

            wait_until_launch_queue_empty();

            *response = cudaStreamSynchronize(
				args->stream
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}