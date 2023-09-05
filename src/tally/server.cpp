#include <cstring>
#include <dlfcn.h>
#include <cassert>
#include <unordered_set>
#include <atomic>

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

void TallyServer::start_main_server() {

    iox::runtime::PoshRuntime::initRuntime(APP_NAME);
    iox::popo::UntypedServer handshake_server({"Tally", "handshake", "event"});

    implicit_init_cuda_ctx();

    load_cache();

    spdlog::info("Tally server is up ...");

    while (!iox::posix::hasTerminationRequested())
    {
        //! [take request]
        handshake_server.take().and_then([&](auto& requestPayload) {
            
            auto msg = static_cast<const HandshakeMessgae*>(requestPayload);
            int32_t client_id = msg->client_id;
            auto channel_desc_str = std::string("Tally-Communication") + std::to_string(client_id);
            char channel_desc[100];
            strcpy(channel_desc, channel_desc_str.c_str()); 

            worker_servers[client_id] = new iox::popo::UntypedServer({channel_desc, "tally", "tally"});
            std::thread t(&TallyServer::start_worker_server, TallyServer::server, client_id);
            t.detach();
            threads_running_map[client_id] = true;

            auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
            handshake_server.loan(requestHeader, sizeof(HandshakeResponse), alignof(HandshakeResponse))
                .and_then([&](auto& responsePayload) {

                    auto response = static_cast<HandshakeResponse*>(responsePayload);
                    response->success = true;

                    handshake_server.send(response).or_else(
                        [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
                })
                .or_else(
                    [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });

            handshake_server.releaseRequest(requestPayload);
        });

        // Check whether any worker thread has exited. Free its resources.
        for (auto it = threads_running_map.cbegin(); it != threads_running_map.cend() /* not hoisted */; /* no increment */)
        {
            auto client_id = it->first;
            auto &thread_running = it->second;

            if (!thread_running) {
                delete worker_servers[client_id];
                worker_servers.erase(client_id);

                client_data_all[client_id].has_exit = true;

                it = threads_running_map.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void TallyServer::start_worker_server(int32_t client_id) {

    implicit_init_cuda_ctx();

    auto &client_meta = client_data_all[client_id];

    CHECK_CUDA_ERROR(cudaStreamCreate(&client_meta.default_stream));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&client_meta.global_idx, sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&client_meta.retreat, sizeof(bool)));

    spdlog::info("Tally worker server is up ...");

    auto process_name = get_process_name(client_id);
    spdlog::info("Client process: " + process_name);

    auto worker_server = worker_servers[client_id];

    while (!iox::posix::hasTerminationRequested())
    {
        //! [take request]
        worker_server->take().and_then([&](auto& requestPayload) {

            auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
            auto handler = cuda_api_handler_map[msg_header->api_id];

            void *args = (void *) (static_cast<const uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
            handler(args, worker_server, requestPayload);

            worker_server->releaseRequest(requestPayload);
        });

        if (!is_process_running(client_id)) {
            break;
        }
    }

    threads_running_map[client_id] = false;

    spdlog::info("Tally worker server has exited ...");
}

void TallyServer::register_ptx_transform(const char* cubin_data, size_t cubin_size)
{
    using KERNEL_NAME_MAP_TYPE = folly::ConcurrentHashMap<std::string, const void *>;
    using KERNEL_MAP_TYPE = folly::ConcurrentHashMap<const void*, WrappedCUfunction>;

    auto original_data = TallyCache::cache->cubin_cache.get_original_data(cubin_data, cubin_size);
    auto ptb_data = TallyCache::cache->cubin_cache.get_ptb_data(cubin_data, cubin_size);
    auto dynamic_ptb_data = TallyCache::cache->cubin_cache.get_dynamic_ptb_data(cubin_data, cubin_size);
    auto preemptive_ptb_data = TallyCache::cache->cubin_cache.get_preemptive_ptb_data(cubin_data, cubin_size);

    auto cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);
    auto &kernel_name_to_host_func_map = cubin_to_kernel_name_to_host_func_map[cubin_uid];

    register_kernels_from_ptx_fatbin<KERNEL_NAME_MAP_TYPE, KERNEL_MAP_TYPE>(original_data, kernel_name_to_host_func_map, original_kernel_map);
    register_kernels_from_ptx_fatbin<KERNEL_NAME_MAP_TYPE, KERNEL_MAP_TYPE>(ptb_data, kernel_name_to_host_func_map, ptb_kernel_map);
    register_kernels_from_ptx_fatbin<KERNEL_NAME_MAP_TYPE, KERNEL_MAP_TYPE>(dynamic_ptb_data, kernel_name_to_host_func_map, dynamic_ptb_kernel_map);
    register_kernels_from_ptx_fatbin<KERNEL_NAME_MAP_TYPE, KERNEL_MAP_TYPE>(preemptive_ptb_data, kernel_name_to_host_func_map, preemptive_ptb_kernel_map);
}

void TallyServer::handle_cudaLaunchKernel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{

    TALLY_SPD_LOG("Received request: cudaLaunchKernel");
    auto args = (cudaLaunchKernelArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;
    
    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_uid].default_stream;
    }

    assert(client_data_all[client_uid]._kernel_client_addr_mapping.find(args->host_func) != client_data_all[client_uid]._kernel_client_addr_mapping.end());
    if (client_data_all[client_uid]._kernel_client_addr_mapping.find(args->host_func) == client_data_all[client_uid]._kernel_client_addr_mapping.end()) {
        throw std::runtime_error("client_data_all[client_uid]._kernel_client_addr_mapping.find(args->host_func) == client_data_all[client_uid]._kernel_client_addr_mapping.end()");
    }

    const void *server_func_addr = client_data_all[client_uid]._kernel_client_addr_mapping[args->host_func];

    auto kernel_name = host_func_to_demangled_kernel_name_map[server_func_addr];
    TALLY_SPD_LOG(kernel_name);
    
    auto partial = cudaLaunchKernel_Partial(server_func_addr, args->gridDim, args->blockDim, args->sharedMem, stream, args->params);

    while (client_data_all[client_uid].has_kernel) {}

    client_data_all[client_uid].dynamic_shmem_size_bytes = args->sharedMem;
    client_data_all[client_uid].launch_call = CudaLaunchCall(server_func_addr, args->gridDim, args->blockDim);
    client_data_all[client_uid].kernel_to_dispatch = partial;
    client_data_all[client_uid].launch_stream = stream;
    client_data_all[client_uid].has_kernel = true;

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaError_t*>(responsePayload);

            // Fool cudaGetLastError
            int device;
            cudaGetDevice(&device);

            *response = cudaSuccess;

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuLaunchKernel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLaunchKernel");
	auto args = (cuLaunchKernelArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_uid].default_stream;
    }

    dim3 gridDim(args->gridDimX, args->gridDimY, args->gridDimZ);
    dim3 blockDim(args->blockDimX, args->blockDimY, args->blockDimZ);

    assert(args->f);

    auto partial = cudaLaunchKernel_Partial(args->f, gridDim, blockDim, args->sharedMemBytes, stream, args->kernelParams);

    while (client_data_all[client_uid].has_kernel) {}

    client_data_all[client_uid].launch_call = CudaLaunchCall(0, 0, 0);
    client_data_all[client_uid].launch_stream = stream;
    client_data_all[client_uid].kernel_to_dispatch = partial;
    client_data_all[client_uid].has_kernel = true;

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<CUresult*>(responsePayload);
            // *response = client_data_all[client_uid].err;

            int device;
            cudaGetDevice(&device);

            *response = CUDA_SUCCESS;

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::load_cache()
{
    auto &cubin_map = TallyCache::cache->cubin_cache.cubin_map;

    for (auto &pair : cubin_map) {
        uint32_t cubin_size = pair.first;
        auto &cubin_vec = pair.second;

        for (auto &cubin_data : cubin_vec) {

            auto kernel_args = cubin_data.kernel_args;

            for (auto &kernel_args_pair : cubin_data.kernel_args) {

                auto &kernel_name = kernel_args_pair.first;
                auto &param_sizes = kernel_args_pair.second;
                auto demangled_kernel_name = demangleFunc(kernel_name);

                auto cubin_uid = cubin_data.cubin_uid;

                if (cubin_to_kernel_name_to_host_func_map[cubin_uid].find(kernel_name) == cubin_to_kernel_name_to_host_func_map[cubin_uid].end()) {

                    // allocate an address for the kernel
                    const void *kernel_server_addr = (const void *) malloc(8);

                    // Bookkeeping
                    _kernel_addr_to_args.insert(kernel_server_addr, param_sizes);
                    host_func_to_demangled_kernel_name_map.insert(kernel_server_addr, demangled_kernel_name);

                    host_func_to_cubin_uid_map.insert(kernel_server_addr, cubin_uid);

                    cubin_to_kernel_name_to_host_func_map[cubin_uid].insert(kernel_name, kernel_server_addr);
                    cubin_to_kernel_name_to_host_func_map[cubin_uid].insert(demangled_kernel_name, kernel_server_addr);

                    demangled_kernel_name_and_cubin_uid_to_host_func_map.insert(
                        std::make_pair<std::string, size_t>(std::move(demangled_kernel_name), std::move(cubin_uid)),
                        kernel_server_addr
                    );
                }
            }

            // Load the original and transformed PTX and register them as callable functions
            register_ptx_transform(cubin_data.cubin_data.c_str(), cubin_size);
        }
    }

    register_measurements();
}

void TallyServer::handle___cudaRegisterFatBinary(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFatBinary");
    auto args = (__cudaRegisterFatBinaryArg *) __args;
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    auto &client_meta = client_data_all[client_uid];

    client_meta.cubin_registered = args->cached;
    client_meta.cubin_uid = args->cubin_uid;

    // Free data from last time
    if (client_meta.fatbin_data) {
        free(client_meta.fatbin_data);
        client_meta.fatbin_data = nullptr;
    }

    client_meta.register_queue.clear();

    if (!client_meta.cubin_registered) {
        struct fatBinaryHeader *header = (struct fatBinaryHeader *) args->data;
        size_t cubin_size = header->headerSize + header->fatSize;
        const char *cubin_data = (const char *) args->data;

        // Load necessary data into cache if not exists
        cache_cubin_data(cubin_data, cubin_size, client_uid);

        client_meta.fatBinSize = cubin_size;
        client_meta.fatbin_data = (unsigned long long *) malloc(cubin_size);
        memcpy(client_meta.fatbin_data, args->data, cubin_size);

        std::string tmp_elf_file = get_tmp_file_path(".elf", client_uid);
        iox_server->loan(requestHeader, tmp_elf_file.size() + 1, alignof(char[]))
            .and_then([&](auto& responsePayload) {

                auto response = static_cast<char *>(responsePayload);
                memcpy(response, tmp_elf_file.c_str(), tmp_elf_file.size());
                response[tmp_elf_file.size()] = '\0';

                iox_server->send(response).or_else(
                    [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
            })
            .or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
    }
}

void TallyServer::handle___cudaRegisterFunction(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFunction");
    auto args = (struct __cudaRegisterFunctionArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    auto &client_meta = client_data_all[client_uid];

    std::string kernel_name {args->data, args->kernel_func_len};
    client_meta.register_queue.push_back( std::make_pair(args->host_func, kernel_name) );
}

void TallyServer::handle___cudaRegisterFatBinaryEnd(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFatBinaryEnd");
    
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    auto &client_meta = client_data_all[client_uid];

    void *kernel_server_addr;
    std::map<std::string, std::vector<uint32_t>> kernel_names_and_param_sizes;
    uint32_t cubin_uid;
    
    if (!client_meta.cubin_registered) {
        kernel_names_and_param_sizes = TallyCache::cache->cubin_cache.get_kernel_args((const char*) client_meta.fatbin_data, client_meta.fatBinSize);
        cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid((const char*) client_meta.fatbin_data, client_meta.fatBinSize);
    } else {
        cubin_uid = client_meta.cubin_uid;
    }
    
    auto cubin_str = TallyCache::cache->cubin_cache.get_cubin_data_str_from_cubin_uid(cubin_uid);

    for (auto &kernel_pair : client_meta.register_queue) {
        auto &client_addr = kernel_pair.first;
        auto &kernel_name = kernel_pair.second;

        // make sure client data has not registered this client addr already
        if (client_meta._kernel_client_addr_mapping.find(client_addr) == client_meta._kernel_client_addr_mapping.end()) {

            if (cubin_to_kernel_name_to_host_func_map[cubin_uid].find(kernel_name) == cubin_to_kernel_name_to_host_func_map[cubin_uid].end()) {

                auto demangled_kernel_name = demangleFunc(kernel_name);
        
                // allocate an address for the kernel
                // TODO: In fact we don't need malloc here
                // Just need a unique address for this purpose
                kernel_server_addr = malloc(8);

                auto &param_sizes = kernel_names_and_param_sizes[kernel_name];

                // Register the kernel with this address
                host_func_to_demangled_kernel_name_map.insert(kernel_server_addr, demangled_kernel_name);
                _kernel_addr_to_args.insert(kernel_server_addr, param_sizes);

                host_func_to_cubin_uid_map.insert(kernel_server_addr, cubin_uid);

                cubin_to_kernel_name_to_host_func_map[cubin_uid].insert(demangled_kernel_name, kernel_server_addr);
                cubin_to_kernel_name_to_host_func_map[cubin_uid].insert(kernel_name, kernel_server_addr);

                demangled_kernel_name_and_cubin_uid_to_host_func_map.insert(
                    std::make_pair<std::string, size_t>(std::move(demangled_kernel_name), std::move(cubin_uid)),
                    kernel_server_addr
                );
            }

            client_meta._kernel_client_addr_mapping[client_addr] = cubin_to_kernel_name_to_host_func_map[cubin_uid][kernel_name];
        }
    }

    // Load the transformed PTX and register them as callable functions
    if (!client_meta.cubin_registered) {
        register_ptx_transform((const char*) client_meta.fatbin_data, client_meta.fatBinSize);
    }
}

void TallyServer::handle_cudaMalloc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaMalloc");
	auto args = (struct cudaMallocArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaMallocResponse), alignof(cudaMallocResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaMallocResponse*>(responsePayload);
 
            response->err = cudaMalloc(&(response->devPtr), args->size);

            // Keep track that this addr is device memory
            if (response->err == cudaSuccess) {
                client_data_all[client_uid].dev_addr_map.push_back( DeviceMemoryKey(response->devPtr, args->size) );
            }
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaFree(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaFree");
    auto args = (struct cudaMallocArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaFreeArg), alignof(cudaFreeArg))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaError_t*>(responsePayload);

            *response = cudaFree(args->devPtr);

            if (*response == cudaSuccess) {
                free_dev_addr(client_data_all[client_uid].dev_addr_map, args->devPtr);
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaMemcpy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaMemcpy");

    auto args = (struct cudaMemcpyArg *) __args;
    size_t res_size;

    if (args->kind == cudaMemcpyHostToDevice) {
        res_size = sizeof(cudaMemcpyResponse);
    } else if (args->kind == cudaMemcpyDeviceToHost){
        res_size = sizeof(cudaMemcpyResponse) + args->count;
    } else if (args->kind == cudaMemcpyDeviceToDevice) {
        res_size = sizeof(cudaMemcpyResponse);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, res_size, alignof(cudaMemcpyResponse))
        .and_then([&](auto& responsePayload) {
            auto res = static_cast<cudaMemcpyResponse*>(responsePayload);

            while (client_data_all[client_uid].has_kernel) {}

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

void TallyServer::handle_cudaMemcpyAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaMemcpyAsync");
    
    auto args = (struct cudaMemcpyAsyncArg *) __args;
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_uid].default_stream;
    }

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

    iox_server->loan(requestHeader, res_size, alignof(cudaMemcpyAsyncResponse))
        .and_then([&](auto& responsePayload) {
            auto res = static_cast<cudaMemcpyAsyncResponse*>(responsePayload);

            while (client_data_all[client_uid].has_kernel) {}

            if (args->kind == cudaMemcpyHostToDevice) {
                res->err = cudaMemcpyAsync(args->dst, args->data, args->count, args->kind, stream);
            } else if (args->kind == cudaMemcpyDeviceToHost){
                res->err = cudaMemcpyAsync(res->data, args->src, args->count, args->kind, stream);
            } else if (args->kind == cudaMemcpyDeviceToDevice) {
                res->err = cudaMemcpyAsync(args->dst, args->src, args->count, args->kind, stream);
            } else {
                throw std::runtime_error("Unknown memcpy kind!");
            }

            iox_server->send(res).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) {LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasSgemm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasSgemm_v2");
    auto args = (struct cublasSgemm_v2Arg *) __args;

    const float alpha = args->alpha;
    const float beta = args->beta;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            while (client_data_all[client_uid].has_kernel) {}

            auto response = static_cast<cublasStatus_t*>(responsePayload);
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

void TallyServer::handle_cublasLtMatmul(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasLtMatmul");
    auto args = (struct cublasLtMatmulArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);

            while (client_data_all[client_uid].has_kernel) {}

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
                stream
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatmulDescSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cublasLtMatrixLayoutSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cublasLtMatmulPreferenceSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cublasLtMatmulAlgoGetHeuristic(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnBackendSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnBackendSetAttribute");
    auto args = (struct cudnnBackendSetAttributeArg *) __args;
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    // In case the values contain CPU pointers
    // Then we will allocate again
    std::vector <void *> allocated_mem;

    if (args->attributeType == CUDNN_TYPE_VOID_PTR) {
        auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
        int32_t client_uid = msg_header->client_id;

        auto pointer_arr = (void **) (args->arrayOfElements);

        for (int i = 0; i < args->elementCount; i++) {
            auto pointer = pointer_arr[i];

            if (pointer == nullptr) {
                continue;
            }
            
            auto found = is_dev_addr(client_data_all[client_uid].dev_addr_map, pointer);

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

void TallyServer::handle_cudnnBackendGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnActivationForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnActivationForward");
    auto args = (struct cudnnActivationForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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


void TallyServer::handle_cudnnSetTensorNdDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnSetConvolutionNdDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnSetFilterNdDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnConvolutionForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnConvolutionForward");
    auto args = (struct cudnnConvolutionForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnGetConvolutionNdForwardOutputDim(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnGetConvolutionForwardAlgorithm_v7(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnFindConvolutionForwardAlgorithm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnAddTensor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnAddTensor");
    auto args = (struct cudnnAddTensorArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnSetPoolingNdDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnGetPoolingNdDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnGetPoolingNdForwardOutputDim(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnPoolingForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnPoolingForward");
    auto args = (struct cudnnPoolingForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cublasSgemv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasSgemv_v2");
    auto args = (struct cublasSgemv_v2Arg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnLRNCrossChannelForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnLRNCrossChannelForward");
    auto args = (struct cudnnLRNCrossChannelForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnSoftmaxForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnSoftmaxForward");
    auto args = (struct cudnnSoftmaxForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnTransformTensor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnTransformTensor");
    auto args = (struct cudnnTransformTensorArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cublasSgemmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasSgemmEx");
    auto args = (struct cublasSgemmExArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnSetSeqDataDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnGetSeqDataDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnMultiHeadAttnForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnMultiHeadAttnForward");
    auto args = (struct cudnnMultiHeadAttnForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnMultiHeadAttnBackwardData(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnMultiHeadAttnBackwardData");
    auto args = (struct cudnnMultiHeadAttnBackwardDataArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnMultiHeadAttnBackwardWeights(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnMultiHeadAttnBackwardWeights");
    auto args = (struct cudnnMultiHeadAttnBackwardWeightsArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnReorderFilterAndBias(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnReorderFilterAndBias");
    auto args = (struct cudnnReorderFilterAndBiasArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnGetRNNWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnGetRNNTrainingReserveSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnGetFilterNdDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnRNNForwardTraining(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnRNNForwardTraining");
    auto args = (struct cudnnRNNForwardTrainingArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnRNNBackwardData(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnRNNBackwardData");
    auto args = (struct cudnnRNNBackwardDataArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnRNNBackwardWeights(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnRNNBackwardWeights");
    auto args = (struct cudnnRNNBackwardWeightsArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnSetRNNDataDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnGetTensorNdDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnBatchNormalizationForwardTrainingEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnBatchNormalizationForwardTrainingEx");
    auto args = (struct cudnnBatchNormalizationForwardTrainingExArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnBatchNormalizationBackwardEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudnnBatchNormalizationBackwardEx");
    auto args = (struct cudnnBatchNormalizationBackwardExArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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


void TallyServer::handle_cublasSgemmStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    auto args = (struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse), alignof(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse*>(responsePayload);
            response->err = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &response->numBlocks,
                client_data_all[client_uid]._kernel_client_addr_mapping[args->func],
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

void TallyServer::handle_cudaChooseDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudaSetDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
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

void TallyServer::handle_cudnnRNNBackwardWeights_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNBackwardWeights_v8");
	auto args = (struct cudnnRNNBackwardWeights_v8Arg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
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

void TallyServer::handle_cudnnRNNBackwardData_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNBackwardData_v8");
	auto args = (struct cudnnRNNBackwardData_v8Arg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);

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

void TallyServer::handle_cudnnRNNForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNForward");
	auto args = (struct cudnnRNNForwardArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);

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

void TallyServer::handle_cudnnBackendExecute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBackendExecute");
	auto args = (struct cudnnBackendExecuteArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);

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

void TallyServer::handle_cudaThreadSynchronize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaThreadSynchronize");

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            while (client_data_all[client_uid].has_kernel) {}

            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaThreadSynchronize();

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaEventRecord(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaEventRecord");
	auto args = (struct cudaEventRecordArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_uid].default_stream;
    }
    
    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);

            while (client_data_all[client_uid].has_kernel) {}

            *response = cudaEventRecord(
				args->event,
				stream
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaDeviceSynchronize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceSynchronize");
    
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaError_t*>(responsePayload);

            while (client_data_all[client_uid].has_kernel) {}
            
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

void TallyServer::handle_cudaStreamSynchronize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamSynchronize");
	auto args = (struct cudaStreamSynchronizeArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);

            while (client_data_all[client_uid].has_kernel) {}

            *response = cudaStreamSynchronize(
				stream
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasCreate_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCreate_v2");
	auto args = (struct cublasCreate_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    auto &client_meta = client_data_all[client_uid];

    iox_server->loan(requestHeader, sizeof(cublasCreate_v2Response), alignof(cublasCreate_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasCreate_v2Response*>(responsePayload);

            response->err = cublasCreate_v2(&(response->handle));

            // set stream to client's default stream
            cublasSetStream_v2(response->handle, client_meta.default_stream);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreate");
	auto args = (struct cudnnCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    auto &client_meta = client_data_all[client_uid];

    iox_server->loan(requestHeader, sizeof(cudnnCreateResponse), alignof(cudnnCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateResponse*>(responsePayload);
            response->err = cudnnCreate(&(response->handle));

            // set stream to client's default stream
            cudnnSetStream(response->handle, client_meta.default_stream);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuModuleLoadData(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleLoadData");
	auto args = (struct cuModuleLoadDataArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    bool cached = args->cached;
    std::string tmp_elf_file;
    uint32_t cubin_uid = 0;
    size_t msg_len = sizeof(cuModuleLoadDataResponse);

    size_t cubin_size = 0;
    const char *cubin_data;
    
    if (!cached) {
        auto header = (struct fatBinaryHeader *) args->image;
        cubin_size = header->headerSize + header->fatSize;
        cubin_data = (const char *) args->image;

        cache_cubin_data(cubin_data, cubin_size, client_uid);
        tmp_elf_file = get_tmp_file_path(".elf", client_uid);
        msg_len += tmp_elf_file.size() + 1;
    } else {
        cubin_uid = args->cubin_uid;
    }

    iox_server->loan(requestHeader, sizeof(cuModuleLoadDataResponse), alignof(cuModuleLoadDataResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuModuleLoadDataResponse*>(responsePayload);

            if (!cached) {
                memcpy(response->tmp_elf_file, tmp_elf_file.c_str(), tmp_elf_file.size());
                response->tmp_elf_file[tmp_elf_file.size()] = '\0';

                cubin_data = TallyCache::cache->cubin_cache.get_cubin_data_ptr(cubin_data, cubin_size);
            } else {
                cubin_data = TallyCache::cache->cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
                cubin_size = TallyCache::cache->cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);
            }

            response->err = cuModuleLoadData(&(response->module), cubin_data);

            jit_module_to_cubin_map.insert(response->module, std::make_pair<const char *, size_t>(
                std::move(cubin_data),
                std::move(cubin_size)
            ));

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuModuleGetFunction(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleGetFunction");
	auto args = (struct cuModuleGetFunctionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto cubin_data_size = jit_module_to_cubin_map[args->hmod];
    auto cubin_data = cubin_data_size.first;
    auto cubin_size = cubin_data_size.second;

    auto kernel_names_and_param_sizes = TallyCache::cache->cubin_cache.get_kernel_args(cubin_data, cubin_size);
    auto kernel_name = std::string(args->name);
    auto &param_sizes = kernel_names_and_param_sizes[kernel_name];

    auto ptb_data = TallyCache::cache->cubin_cache.get_ptb_data(cubin_data, cubin_size);
    auto dynamic_ptb_data = TallyCache::cache->cubin_cache.get_dynamic_ptb_data(cubin_data, cubin_size);
    auto preemptive_ptb_data = TallyCache::cache->cubin_cache.get_preemptive_ptb_data(cubin_data, cubin_size);

    iox_server->loan(requestHeader, sizeof(cuModuleGetFunctionResponse), alignof(cuModuleGetFunctionResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuModuleGetFunctionResponse*>(responsePayload);

            response->err = cuModuleGetFunction(&(response->hfunc), args->hmod, args->name);
            _jit_kernel_addr_to_args.insert(response->hfunc, param_sizes);

            register_jit_kernel_from_ptx_fatbin<folly::ConcurrentHashMap<CUfunction, CUfunction>>(ptb_data, response->hfunc, kernel_name, jit_ptb_kernel_map);
            register_jit_kernel_from_ptx_fatbin<folly::ConcurrentHashMap<CUfunction, CUfunction>>(dynamic_ptb_data, response->hfunc, kernel_name, jit_dynamic_ptb_kernel_map);
            register_jit_kernel_from_ptx_fatbin<folly::ConcurrentHashMap<CUfunction, CUfunction>>(preemptive_ptb_data, response->hfunc, kernel_name, jit_preemptive_ptb_kernel_map);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuPointerGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuPointerGetAttribute");
	auto args = (struct cuPointerGetAttributeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    size_t attribute_size = get_cupointer_attribute_size(args->attribute);

    iox_server->loan(requestHeader, sizeof(cuPointerGetAttributeResponse) + attribute_size, alignof(cuPointerGetAttributeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuPointerGetAttributeResponse*>(responsePayload);

            response->err = cuPointerGetAttribute(response->data, args->attribute, args->ptr);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaStreamGetCaptureInfo_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamGetCaptureInfo_v2");
	auto args = (struct cudaStreamGetCaptureInfo_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaStreamGetCaptureInfo_v2Response), alignof(cudaStreamGetCaptureInfo_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaStreamGetCaptureInfo_v2Response*>(responsePayload);
            response->err = cudaStreamGetCaptureInfo_v2(
				stream,
				(args->captureStatus_out ? &(response->captureStatus_out) : NULL),
				(args->id_out ? &(response->id_out) : NULL),
				(args->graph_out ? &(response->graph_out) : NULL),
				(args->dependencies_out ? ((CUgraphNode_st* const**) &(response->dependencies_out)) : NULL),
				(args->numDependencies_out ? &(response->numDependencies_out) : NULL)
			);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaGraphGetNodes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphGetNodes");
    auto args = (struct cudaGraphGetNodesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    size_t msg_len;
    if (args->nodes) {
        msg_len = sizeof(cudaGraphGetNodesResponse) + sizeof(cudaGraphNode_t) * args->numNodes;
    } else {
        msg_len = sizeof(cudaGraphGetNodesResponse);
    }

    iox_server->loan(requestHeader, sizeof(cudaGraphGetNodesResponse), alignof(cudaGraphGetNodesResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaGraphGetNodesResponse*>(responsePayload);

            if (args->nodes) {
                response->numNodes = args->numNodes;
            }

            response->err = cudaGraphGetNodes(
				args->graph,
				args->nodes ? response->nodes : NULL,
                &response->numNodes
			);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

// void TallyServer::handle_cuCtxCreate_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
// {
//     TALLY_SPD_LOG("Received request: cuCtxCreate_v2");
// 	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
// }

// TODO: This CUDA API is supposed to set attribute for a kernel function
// However, since the server register kernels using JIT APIs instead of host functions
// We need to figure out a way to set attribute for the CUfunctions
void TallyServer::handle_cudaFuncSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaFuncSetAttribute");
    
	auto args = (struct cudaFuncSetAttributeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;
    
    const void *server_func_addr = client_data_all[client_uid]._kernel_client_addr_mapping[args->func];
    auto cu_func = original_kernel_map[server_func_addr].func;
    auto cu_func_ptb = ptb_kernel_map[server_func_addr].func;
    auto cu_func_dynamic_ptb = dynamic_ptb_kernel_map[server_func_addr].func;
    auto cu_func_preemptive_ptb = preemptive_ptb_kernel_map[server_func_addr].func;

    auto cu_attr = convert_func_attribute(args->attr);

    cuFuncSetAttribute(cu_func, cu_attr, args->value);
    cuFuncSetAttribute(cu_func_ptb, cu_attr, args->value);
    cuFuncSetAttribute(cu_func_dynamic_ptb, cu_attr, args->value);
    cuFuncSetAttribute(cu_func_preemptive_ptb, cu_attr, args->value);

    std::string set_attr_log = "Setting attribute " + get_func_attr_str(cu_attr) + " to value " + std::to_string(args->value);
    TALLY_SPD_LOG(set_attr_log);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);

            *response = cudaSuccess;

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuGetProcAddress_v2(void*, iox::popo::UntypedServer*, void const*)
{
    TALLY_SPD_LOG("Received request: cuGetProcAddress_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

void TallyServer::handle_cuMemcpy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpy");
	auto args = (struct cuMemcpyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    size_t res_size;

    if (is_dev_addr(client_data_all[client_uid].dev_addr_map, (void *) args->dst)) {
        res_size = sizeof(cuMemcpyResponse);
    } else {
        res_size = sizeof(cuMemcpyResponse) + args->ByteCount;
    }

    iox_server->loan(requestHeader, res_size, alignof(cuMemcpyResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuMemcpyResponse*>(responsePayload);

            response->err = cuMemcpy(
				is_dev_addr(client_data_all[client_uid].dev_addr_map, (void *) args->dst) ? args->dst : (CUdeviceptr) response->data,
				is_dev_addr(client_data_all[client_uid].dev_addr_map, (void *) args->src) ? args->src : (CUdeviceptr) args->data,
				args->ByteCount
            );

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemcpyAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyAsync");
	auto args = (struct cuMemcpyAsyncArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    size_t res_size;

    if (is_dev_addr(client_data_all[client_uid].dev_addr_map, (void *) args->dst)) {
        res_size = sizeof(cuMemcpyAsyncResponse);
    } else {
        res_size = sizeof(cuMemcpyAsyncResponse) + args->ByteCount;
    }

    iox_server->loan(requestHeader, res_size, alignof(cuMemcpyAsyncResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuMemcpyAsyncResponse*>(responsePayload);

            response->err = cuMemcpyAsync(
				is_dev_addr(client_data_all[client_uid].dev_addr_map, (void *) args->dst) ? args->dst : (CUdeviceptr) response->data,
				is_dev_addr(client_data_all[client_uid].dev_addr_map, (void *) args->src) ? args->src : (CUdeviceptr) args->data,
				args->ByteCount,
				__stream
            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemAllocAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAllocAsync");
	auto args = (struct cuMemAllocAsyncArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cuMemAllocAsyncResponse), alignof(cuMemAllocAsyncResponse))
        .and_then([&](auto& responsePayload) {
            
            auto response = static_cast<cuMemAllocAsyncResponse*>(responsePayload);

            response->err = cuMemAllocAsync(
				&(response->dptr),
				args->bytesize,
				__stream
			);

            // Keep track that this addr is device memory
            if (response->err == CUDA_SUCCESS) {
                client_data_all[client_uid].dev_addr_map.push_back( DeviceMemoryKey((void *)response->dptr, args->bytesize) );
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemFree_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemFree_v2");
	auto args = (struct cuMemFree_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuMemFree_v2(
				args->dptr
            );

            if (*response == CUDA_SUCCESS) {
                free_dev_addr(client_data_all[client_uid].dev_addr_map, (void *)args->dptr);
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}