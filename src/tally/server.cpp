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
}

TallyServer::~TallyServer(){}

void TallyServer::start_main_server() {

    iox::runtime::PoshRuntime::initRuntime(APP_NAME);
    iox::popo::UntypedServer handshake_server({"Tally", "handshake", "event"});

    implicit_init_cuda_ctx();

    TALLY_SPD_LOG_ALWAYS("Tally server is up ...");

    std::vector<std::thread> worker_threads;

    while (!iox::posix::hasTerminationRequested())
    {
        //! [take request]
        handshake_server.take().and_then([&](auto& requestPayload) {
            
            auto msg = static_cast<const HandshakeMessgae*>(requestPayload);
            int32_t client_id = msg->client_id;
            client_data_all[client_id].client_id = client_id;

            ClientPriority client_priority(client_id, msg->priority);
            client_priority_map[client_priority] = client_id;

            auto channel_desc_str = std::string("Tally-Communication") + std::to_string(client_id);
            char channel_desc[100];
            strcpy(channel_desc, channel_desc_str.c_str()); 

            worker_servers[client_id] = new iox::popo::UntypedServer({channel_desc, "tally", "tally"});

            std::thread t(&TallyServer::start_worker_server, TallyServer::server, client_id);
            worker_threads.push_back(std::move(t));
            
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
                for (auto &key : client_data_all[client_id].dev_addr_map) {
                    cudaFree(key.addr);
                }

                it = threads_running_map.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (auto &t : worker_threads) {
        t.join();
    }

    TallyCache::cache->save_transform_cache();
}

int32_t TallyServer::get_client_priority(int32_t client_id)
{
    for (auto &client_priority_data : client_priority_map) {
        auto &client_priority = client_priority_data.first;
        if (client_priority.client_id == client_id) {
            return client_priority.priority;
        }
    }

    return 0;
}

int32_t TallyServer::get_client_stream_priority(int32_t client_id)
{
    static int leastPriority, greatestPriority;
    static bool inited = false;

    if (!inited) {
        cudaDeviceGetStreamPriorityRange (&leastPriority, &greatestPriority);
        inited = true;
    }

    auto priority = get_client_priority(client_id);

    // CUDA Stream priorities follow a convention where lower numbers imply greater priorities.
    // 0 -> 0, 1 -> -1, 2 -> -2, ...
    int stream_priority = - priority;
    stream_priority = std::max(stream_priority, greatestPriority);
    stream_priority = std::min(stream_priority, leastPriority);

    return stream_priority;
}

void TallyServer::increment_client_queue_size(int32_t client_id)
{
    client_data_all[client_id].queue_size++;
}

void TallyServer::start_worker_server(int32_t client_id) {

    implicit_init_cuda_ctx();

    auto &client_meta = client_data_all[client_id];

    auto policy = SCHEDULER_POLICY;
    if (policy == TALLY_SCHEDULER_POLICY::PRIORITY) {

        int stream_priority = get_client_stream_priority(client_id);
        CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&client_meta.default_stream, cudaStreamNonBlocking, stream_priority));
    
    } else {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&client_meta.default_stream, cudaStreamNonBlocking));
    }

    CHECK_CUDA_ERROR(cudaMalloc((void **)&client_meta.curr_idx_arr, sizeof(uint32_t) * CUDA_NUM_SM * 20));
    client_add_stream(client_id, client_meta.default_stream);

    TALLY_SPD_LOG_ALWAYS("Tally worker server is up ...");

    auto process_name = get_process_name(client_id);
    TALLY_SPD_LOG_ALWAYS("Client process: " + process_name);

    if (policy == TALLY_SCHEDULER_POLICY::PRIORITY) {
        int priority = get_client_priority(client_id);
        TALLY_SPD_LOG_ALWAYS("Client priority: " + std::to_string(priority));
    }

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
    TALLY_SPD_LOG_ALWAYS("Tally worker server has exited ...");
}

void TallyServer::launch_and_measure_kernel(KernelLaunchWrapper &kernel_wrapper, int32_t client_id,
                                            std::vector<CudaLaunchConfig> configs, float fallback_threshold,
                                            std::vector<CudaLaunchConfig> alternative_configs, bool is_preemptive)
{
    // Store temporary data here
    static std::unordered_map<CudaLaunchCallConfig, TempKernelProfileMetrics> temp_perf_data;
    static std::unordered_map<CudaLaunchCallConfig, uint32_t> warmup_perf_data;

    auto &launch_call = kernel_wrapper.launch_call;
    auto &client_data = client_data_all[client_id];
    auto kernel_name = host_func_to_demangled_kernel_name_map[launch_call.func];
    auto cubin_uid = host_func_to_cubin_uid_map[launch_call.func];
    auto kernel_str = kernel_name + "_" + launch_call.dim_str() + "_" + std::to_string(cubin_uid);

    // Measure single-run time
    float time_elapsed;
    float iters;

    auto base_config = CudaLaunchConfig::default_config;

    bool found;
    auto original_res = get_single_kernel_perf(launch_call, base_config, &found);

    if (!found) {
        CudaLaunchCallConfig original_call_config(launch_call, base_config);
        auto &metrics = temp_perf_data[original_call_config];

        cudaDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        kernel_wrapper.kernel_to_dispatch(base_config, nullptr, nullptr, nullptr, false, 0, nullptr, nullptr, 0, true);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        time_elapsed = elapsed.count();

        if (warmup_perf_data.find(original_call_config) == warmup_perf_data.end()) {
            warmup_perf_data[original_call_config] = 0;
        }

        warmup_perf_data[original_call_config]++;

        // first 10 is warm up
        if (warmup_perf_data[original_call_config] < 5) {
            return;
        }

        metrics.avg_latency_ms = (metrics.avg_latency_ms * metrics.count + time_elapsed) / (metrics.count + 1);
        metrics.count += 1;

        if (metrics.count >= KERNEL_PROFILE_ITERATIONS) {
            set_single_kernel_perf(launch_call, base_config, original_kernel_map[launch_call.func].meta_data, 1., metrics.avg_latency_ms, 0);
        }

        return;
    }

    auto &original_metrics = original_res.metrics;

    auto run_config = [&](CudaLaunchConfig &config, TempKernelProfileMetrics &metrics) {

        // prepare ptb args
        auto ptb_args = client_data.stream_to_ptb_args[kernel_wrapper.launch_stream];

        // prepare sliced args
        SlicedKernelArgs slice_args;
        if (config.use_sliced) {
            slice_args = get_sliced_kernel_args(launch_call.gridDim, config.num_slices);
        }

        cudaDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();

        if (config.use_dynamic_ptb || config.use_preemptive_ptb) {
            cudaMemsetAsync(ptb_args, 0, sizeof(PTBKernelArgs), kernel_wrapper.launch_stream);
        }

        kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, &slice_args, false, 0, nullptr, nullptr, -1, true);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        time_elapsed = elapsed.count();

        metrics.avg_latency_ms = (metrics.avg_latency_ms * metrics.count + time_elapsed) / (metrics.count + 1);
        metrics.count += 1;
    };

    CudaLaunchConfig best_config;
    float best_latency_ms = FLT_MAX;

    for (auto &config : configs) {

        auto res = get_single_kernel_perf(launch_call, config, &found);
        if (found) {

            if (res.metrics.latency_ms < best_latency_ms) {
                best_config = config;
                best_latency_ms = res.metrics.latency_ms;
            }

            continue;
        }

        CudaLaunchCallConfig call_config(launch_call, config);
        auto &metrics = temp_perf_data[call_config];

        run_config(config, metrics);

        if (metrics.count >= KERNEL_PROFILE_ITERATIONS) {
            float norm_speed = original_metrics.latency_ms / metrics.avg_latency_ms;

            // Likely the base config is not profiled correctly, so we delete it and do it again.
            if (norm_speed >= 2.) {
                delete_single_kernel_perf(launch_call, base_config);

                CudaLaunchCallConfig original_call_config(launch_call, base_config);
                temp_perf_data.erase(original_call_config);
                warmup_perf_data.erase(original_call_config);

                TALLY_SPD_WARN("Detecting abnormal norm_speed of " + std::to_string(norm_speed) + " of kernel " + kernel_str);
                TALLY_SPD_WARN("Deleting measurement for original config.");
                return;
            }

            set_single_kernel_perf(launch_call, config, ptb_kernel_map[launch_call.func].meta_data, norm_speed, metrics.avg_latency_ms, 0);

            if (metrics.avg_latency_ms < best_latency_ms) {
                best_config = config;
                best_latency_ms = metrics.avg_latency_ms;
            }

            if (config.max_worker_blocks < CUDA_NUM_SM) {
                auto blocks_per_sm = (launch_call.num_blocks + CUDA_NUM_SM - 1) / CUDA_NUM_SM;
                auto per_block_latency_ms = original_metrics.latency_ms / (float) blocks_per_sm;
                TALLY_SPD_WARN("Setting max_worker_blocks to " + std::to_string(config.max_worker_blocks) + " for " + kernel_str);
            }
        }

        // If not last config, return
        if (config != configs.back() || metrics.count < KERNEL_PROFILE_ITERATIONS) {
            return;
        }
    }

    float best_norm_speed = original_metrics.latency_ms / best_latency_ms;

    bool fallback_to_original = true;

    if (best_norm_speed >= fallback_threshold) {
        fallback_to_original = false;
    } else {
        if (best_config.max_worker_blocks < CUDA_NUM_SM) {
            auto adjusted_norm_speed = best_norm_speed / ((float) best_config.max_worker_blocks / (float) CUDA_NUM_SM);
            if (adjusted_norm_speed >= fallback_threshold) {
                fallback_to_original = false;
            }
        }
    }

    // if none of the configs satisfy `fallback_threshold`
    // consider the alternative_configs
    if (fallback_to_original && !alternative_configs.empty()) {

        for (auto &config : alternative_configs) {

            auto res = get_single_kernel_perf(launch_call, config, &found);
            if (found) {

                if (res.metrics.latency_ms < best_latency_ms) {
                    best_config = config;
                    best_latency_ms = res.metrics.latency_ms;
                }

                continue;
            }

            CudaLaunchCallConfig call_config(launch_call, config);
            auto &metrics = temp_perf_data[call_config];

            run_config(config, metrics);

            if (metrics.count >= KERNEL_PROFILE_ITERATIONS) {

                configs.push_back(config);

                float norm_speed = original_metrics.latency_ms / metrics.avg_latency_ms;
                set_single_kernel_perf(launch_call, config, ptb_kernel_map[launch_call.func].meta_data, norm_speed, metrics.avg_latency_ms, 0);

                if (metrics.avg_latency_ms < best_latency_ms) {
                    best_config = config;
                    best_latency_ms = metrics.avg_latency_ms;
                }

                // Will take that
                if (norm_speed >= fallback_threshold) {
                    TALLY_SPD_WARN("Fall back to use alternative config.")
                    break;
                }
            }

            // If not last config, return
            if (config != configs.back() || metrics.count < KERNEL_PROFILE_ITERATIONS) {
                return;
            }
        }

        // re-compute
        best_norm_speed = original_metrics.latency_ms / best_latency_ms;
        if (best_norm_speed >= fallback_threshold) {
            fallback_to_original = false;
        }
    }

    if (fallback_to_original || configs.empty()) {

        if (configs.empty()) {
            kernel_wrapper.kernel_to_dispatch(base_config, nullptr, nullptr, nullptr, false, 0, nullptr, nullptr, 0, true);
            TALLY_SPD_WARN("No transformed config is provided. Will Use original config.")
        } else {
            TALLY_SPD_WARN("Fall back to original config as transformed kernel norm speed is below threshold: " + std::to_string(best_norm_speed));
        }
        
        best_config = base_config;
        best_norm_speed = 1.;
        best_latency_ms = original_metrics.latency_ms;
    }

    auto res = get_single_kernel_perf(launch_call, best_config, &found);
    set_single_kernel_chosen_config(launch_call, res);

    TALLY_SPD_LOG_ALWAYS("Tuning complete for: " + kernel_str);

    for (auto &config : configs) {
        auto res = get_single_kernel_perf(launch_call, config, &found);
        auto norm_speed = res.metrics.norm_speed;
        auto latency_ms = res.metrics.latency_ms;
        TALLY_SPD_LOG_ALWAYS("\tLaunch config: " + config.str() + ". Latency: " + std::to_string(latency_ms) + " ms Norm speed: " + std::to_string(norm_speed));
    }
    TALLY_SPD_LOG_ALWAYS("Best config: " + best_config.str() + ". Latency: " + std::to_string(best_latency_ms) + " ms Norm speed: " + std::to_string(best_norm_speed) + "\n");

    return;
}

void TallyServer::tune_kernel_launch(KernelLaunchWrapper &kernel_wrapper, int32_t client_id, std::vector<CudaLaunchConfig> &configs)
{
    auto &launch_call = kernel_wrapper.launch_call;
    auto &client_data = client_data_all[client_id];
    auto kernel_name = host_func_to_demangled_kernel_name_map[launch_call.func];
    auto cubin_uid = host_func_to_cubin_uid_map[launch_call.func];

    TALLY_SPD_LOG_ALWAYS("Launch config not found for: " + kernel_name + "_" + launch_call.dim_str() + "_" + std::to_string(cubin_uid));

    auto start = std::chrono::high_resolution_clock::now();

    // Measure single-run time
    float time_elapsed;
    float iters;

    cudaDeviceSynchronize();

    kernel_wrapper.kernel_to_dispatch(CudaLaunchConfig::default_config, nullptr, nullptr, nullptr, true, 1000, &time_elapsed, nullptr, 1, true);

    // In seconds
    float profile_duration = (100 * time_elapsed) / 1000.f;

    // At least run for 0.01 sec
    profile_duration = std::max(profile_duration, 0.01f);

    // Maybe don't exceed 1 minute;
    profile_duration = std::min(profile_duration, 60.f);

    // Run default config first
    CudaLaunchConfig base_config = CudaLaunchConfig::default_config;

    // warmup
    kernel_wrapper.kernel_to_dispatch(base_config, nullptr, nullptr, nullptr, true, profile_duration, &time_elapsed, &iters, -1, true);

    // profile
    kernel_wrapper.kernel_to_dispatch(base_config, nullptr, nullptr, nullptr, true, profile_duration, &time_elapsed, &iters, -1, true);

    float base_latency_ms = time_elapsed / iters;

    // Save result to cache
    set_single_kernel_perf(launch_call, base_config, original_kernel_map[launch_call.func].meta_data, 1., base_latency_ms, iters);

    CudaLaunchConfig best_config;
    float best_latency_ms = FLT_MAX;

    for (auto &config : configs) {

        // prepare ptb args
        auto ptb_args = client_data.stream_to_ptb_args[kernel_wrapper.launch_stream];

        // prepare sliced args
        SlicedKernelArgs slice_args;
        if (config.use_sliced) {
            slice_args = get_sliced_kernel_args(launch_call.gridDim, config.num_slices);
        }

        auto err = kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, &slice_args, true, profile_duration, &time_elapsed, &iters, -1, true);

        if (err) {
            return;
        }

        float latency_ms = time_elapsed / iters;

        if (latency_ms < best_latency_ms) {
            best_config = config;
            best_latency_ms = latency_ms;
        }

        float norm_speed = base_latency_ms / latency_ms;

        set_single_kernel_perf(launch_call, config, ptb_kernel_map[launch_call.func].meta_data, norm_speed, base_latency_ms, iters);
        TALLY_SPD_LOG_ALWAYS("Launch config: " + config.str() + ". Norm speed: " + std::to_string(norm_speed));
    }

    float best_norm_speed = base_latency_ms / best_latency_ms;

    bool found_in_cache;
    auto res = get_single_kernel_perf(launch_call, best_config, &found_in_cache);
    set_single_kernel_chosen_config(launch_call, res);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    TALLY_SPD_LOG_ALWAYS("Tuning complete ("+ std::to_string(elapsed.count()) + " ms). Launch config: " + best_config.str() + ". Norm speed: " + std::to_string(best_norm_speed) + "\n");
}

void TallyServer::wait_until_launch_queue_empty(int32_t client_id)
{
    int attempt = 0;
    while (client_data_all[client_id].queue_size > 0) {
        attempt++;

        if (attempt == 10000000) {
            if (iox::posix::hasTerminationRequested() || signal_exit) {
                break;
            }
        }
    }
}

void TallyServer::client_add_stream(int32_t client_id, cudaStream_t stream)
{
    auto &client_meta = client_data_all[client_id];
    client_meta.streams.push_back(stream);

    auto &ptb_args = client_meta.stream_to_ptb_args[stream];
    cudaMalloc(&ptb_args, sizeof(PTBKernelArgs));
}

const void *TallyServer::get_server_addr_from_client_addr(uint32_t client_id, const void *client_addr)
{
    auto &client_addr_mapping = client_data_all[client_id]._kernel_client_addr_mapping;

    if (client_addr_mapping.find(client_addr) == client_addr_mapping.end()) {
        throw std::runtime_error("Fail to look up server addr from client addr");
    }

    const void *server_addr = client_addr_mapping[client_addr];

    // register the kernel of this server addr if not done already
    if (original_kernel_map.find(server_addr) == original_kernel_map.end()) {
        register_kernel(server_addr);
    }

    return server_addr;
}

const void *TallyServer::get_server_addr_from_cu_func(CUfunction cu_func)
{
    if (cu_func_addr_mapping.find(cu_func) == cu_func_addr_mapping.end()) {
        throw std::runtime_error("Fail to look up server addr from cu_func");
    }

    const void *server_addr = cu_func_addr_mapping[cu_func];

    // register the kernel of this server addr if not done already
    if (original_kernel_map.find(server_addr) == original_kernel_map.end()) {
        register_kernel(server_addr);
    }

    return server_addr;
}

void TallyServer::register_kernel(const void *server_func_addr)
{
    auto &cubin_cache = TallyCache::cache->get_cubin_cache();
    auto cubin_uid = host_func_to_cubin_uid_map[server_func_addr];
    
    register_cu_module(cubin_uid);
    register_ptx_transform(cubin_uid);
    load_measurements(cubin_uid);
}

void TallyServer::register_ptx_transform(uint32_t cubin_uid)
{
    using KERNEL_NAME_MAP_TYPE = folly::ConcurrentHashMap<std::string, const void *>;
    using KERNEL_MAP_TYPE = folly::ConcurrentHashMap<const void*, WrappedCUfunction>;

    auto &cubin_cache = TallyCache::cache->get_cubin_cache();
    auto cubin_size = cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);
    auto cubin_data = cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
    auto &kernel_params = cubin_cache.get_kernel_args(cubin_data, cubin_size);

    // no transform is available if ptx is not available
    auto &transform_ptx_str = cubin_cache.get_transform_ptx_str(cubin_data, cubin_size);
    bool has_transform = transform_ptx_str != "";
    
    auto &kernel_name_to_host_func_map = cubin_to_kernel_name_to_host_func_map[cubin_uid];

    if (cubin_to_cu_module.find(cubin_uid) != cubin_to_cu_module.end()) {
        CUmodule tranform_module = cubin_to_cu_module[cubin_uid];

        register_kernels_from_ptx_fatbin<KERNEL_NAME_MAP_TYPE, KERNEL_MAP_TYPE>(
            tranform_module, kernel_params, has_transform, kernel_name_to_host_func_map, original_kernel_map,
            ptb_kernel_map, dynamic_ptb_kernel_map, preemptive_ptb_kernel_map, sliced_kernel_map
        );
    }
}

void TallyServer::handle_cudaLaunchKernel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{

    TALLY_SPD_LOG("Received request: cudaLaunchKernel");
    auto args = (cudaLaunchKernelArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;
    
    // Make sure what is called on the default stream has finished
    // For some reason it will cause some process to wait for no event, don't know why
    // Therefore, I will try to make sure no one uses the default stream.
    // cudaStreamSynchronize(NULL);

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_id].default_stream;
    }

    const void *server_func_addr = get_server_addr_from_client_addr(client_id, args->host_func);

    auto kernel_name = host_func_to_demangled_kernel_name_map[server_func_addr];
    TALLY_SPD_LOG(kernel_name);
    
    auto partial_and_args = cudaLaunchKernel_Partial(server_func_addr, args->gridDim, args->blockDim, args->sharedMem, stream, args->params);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            false,
            CudaLaunchCall(server_func_addr, args->gridDim, args->blockDim),
            stream,
            args->sharedMem
        )
    );

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

void TallyServer::handle_cuLaunchKernel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLaunchKernel");
	auto args = (cuLaunchKernelArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    cudaStream_t stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_id].default_stream;
    }

    auto server_func_addr = get_server_addr_from_cu_func(args->f);
    auto kernel_name = host_func_to_demangled_kernel_name_map[server_func_addr];
    TALLY_SPD_LOG(kernel_name);

    dim3 gridDim(args->gridDimX, args->gridDimY, args->gridDimZ);
    dim3 blockDim(args->blockDimX, args->blockDimY, args->blockDimZ);

    assert(args->f);

    auto partial_and_args = cudaLaunchKernel_Partial(server_func_addr, gridDim, blockDim, args->sharedMemBytes, stream, args->kernelParams);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            false,
            CudaLaunchCall(server_func_addr, gridDim, blockDim),
            stream,
            args->sharedMemBytes
        )
    );

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<CUresult*>(responsePayload);
            *response = CUDA_SUCCESS;

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::register_cu_module(uint32_t cubin_uid)
{   
    if (cubin_to_cu_module.find(cubin_uid) == cubin_to_cu_module.end()) {

        auto &cubin_cache = TallyCache::cache->get_cubin_cache();
        auto cubin_data = cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
        auto cubin_size = cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);
        auto &transform_fatbin_str = cubin_cache.get_transform_fatbin_str(cubin_data, cubin_size);
        
        CUmodule transform_module;
        CHECK_CUDA_ERROR(cuModuleLoadData(&transform_module, transform_fatbin_str.c_str()));

        cubin_to_cu_module.insert(cubin_uid, transform_module);
    }
}

void TallyServer::handle___cudaRegisterFatBinary(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFatBinary");
    auto args = (__cudaRegisterFatBinaryArg *) __args;
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto &client_meta = client_data_all[client_id];

    client_meta.cubin_registered = args->cached;
    client_meta.cubin_uid = args->cubin_uid;

    // Free data from last time
    if (client_meta.fatbin_data) {
        free(client_meta.fatbin_data);
        client_meta.fatbin_data = nullptr;
    }

    client_meta.register_queue.clear();

    size_t cubin_size = args->cubin_size;

    // load cubin cache
    auto &cubin_cache = TallyCache::cache->get_cubin_cache();
    cubin_cache.load_cubin_cache(cubin_size);

    std::string tmp_elf_file = "";

    if (!client_meta.cubin_registered) {
        auto cubin_data = (const char *) args->data;

        // Load necessary data into cache if not exists
        cache_cubin_data(cubin_data, cubin_size, client_id);

        client_meta.fatBinSize = cubin_size;
        client_meta.fatbin_data = (unsigned long long *) malloc(cubin_size);
        memcpy(client_meta.fatbin_data, args->data, cubin_size);

        tmp_elf_file = get_tmp_file_path(".elf", client_id);
    }
    
    auto msg_len = sizeof(__cudaRegisterFatBinaryResponse) + tmp_elf_file.size();

    iox_server->loan(requestHeader, msg_len, alignof(__cudaRegisterFatBinaryResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<__cudaRegisterFatBinaryResponse *>(responsePayload);
            
            response->str_len = tmp_elf_file.size();
            memcpy(response->tmp_elf_file, tmp_elf_file.c_str(), tmp_elf_file.size());

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle___cudaRegisterFunction(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFunction");

    auto args = (struct __cudaRegisterFunctionArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto &client_meta = client_data_all[client_id];

    std::string kernel_name {args->data, args->kernel_func_len};
    client_meta.register_queue.push_back( std::make_pair(args->host_func, kernel_name) );

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaError_t *>(responsePayload);
            
            *response = cudaSuccess;

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle___cudaRegisterFatBinaryEnd(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: __cudaRegisterFatBinaryEnd");
    
    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto &client_meta = client_data_all[client_id];
    auto &cubin_cache = TallyCache::cache->get_cubin_cache();
    
    const char* fatbin_data;
    size_t fatbin_size;
    uint32_t cubin_uid;

    if (!client_meta.cubin_registered) {
        fatbin_data = (const char*) client_meta.fatbin_data;
        fatbin_size = client_meta.fatBinSize;
        cubin_uid = cubin_cache.get_cubin_data_uid(fatbin_data, fatbin_size);
    } else {
        cubin_uid = client_meta.cubin_uid;
        fatbin_data = cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
        fatbin_size = cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);
    }

    auto kernel_names_and_param_sizes = cubin_cache.get_kernel_args(fatbin_data, fatbin_size);

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
                auto kernel_server_addr = malloc(8);

                auto &param_sizes = kernel_names_and_param_sizes[kernel_name];

                // Register the kernel with this address
                host_func_to_demangled_kernel_name_map.insert(kernel_server_addr, demangled_kernel_name);
                _kernel_addr_to_args.insert(kernel_server_addr, param_sizes);

                host_func_to_cubin_uid_map.insert(kernel_server_addr, cubin_uid);

                cubin_to_kernel_name_to_host_func_map[cubin_uid].insert(demangled_kernel_name, kernel_server_addr);
                cubin_to_kernel_name_to_host_func_map[cubin_uid].insert(kernel_name, kernel_server_addr);

                kernel_cubin_uid_to_host_func_map.insert(
                    std::make_pair(demangled_kernel_name, cubin_uid),
                    kernel_server_addr
                );
            }

            client_meta._kernel_client_addr_mapping[client_addr] = cubin_to_kernel_name_to_host_func_map[cubin_uid][kernel_name];
        }
    }
}

void TallyServer::handle_cudaMalloc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cudaMalloc");
	auto args = (struct cudaMallocArg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaMallocResponse), alignof(cudaMallocResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaMallocResponse*>(responsePayload);
 
            response->err = cudaMalloc(&(response->devPtr), args->size);

            // Keep track that this addr is device memory
            if (response->err == cudaSuccess) {
                client_data_all[client_id].dev_addr_map.push_back( mem_region(response->devPtr, args->size) );
            }

            if (response->err == cudaErrorMemoryAllocation) {
                TALLY_SPD_WARN("Encountered cudaErrorMemoryAllocation " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
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
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaFreeArg), alignof(cudaFreeArg))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaError_t*>(responsePayload);

            wait_until_launch_queue_empty(client_id);
            *response = cudaFree(args->devPtr);

            if (*response == cudaSuccess) {
                free_mem_region(client_data_all[client_id].dev_addr_map, args->devPtr);
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
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, res_size, alignof(cudaMemcpyResponse))
        .and_then([&](auto& responsePayload) {
            auto res = static_cast<cudaMemcpyResponse*>(responsePayload);

            // Make sure all kernels have been dispatched
            wait_until_launch_queue_empty(client_id);

            if (args->kind == cudaMemcpyHostToDevice) {
                res->err = cudaMemcpyAsync(args->dst, args->data, args->count, args->kind, client_data_all[client_id].default_stream);
            } else if (args->kind == cudaMemcpyDeviceToHost){
                res->err = cudaMemcpyAsync(res->data, args->src, args->count, args->kind, client_data_all[client_id].default_stream);
            } else if (args->kind == cudaMemcpyDeviceToDevice) {
                res->err = cudaMemcpyAsync(args->dst, args->src, args->count, args->kind, client_data_all[client_id].default_stream);
            } else {
                throw std::runtime_error("Unknown memcpy kind!");
            }

            // Make sure wait until cudaMemcpy complete, because cudaMemcpy is synchronous
            cudaStreamSynchronize(client_data_all[client_id].default_stream);

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
    int32_t client_id = msg_header->client_id;

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_id].default_stream;
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

            wait_until_launch_queue_empty(client_id);

            if (args->kind == cudaMemcpyHostToDevice) {
                res->err = cudaMemcpyAsync(args->dst, args->data, args->count, args->kind, stream);
            } else if (args->kind == cudaMemcpyDeviceToHost){
                res->err = cudaMemcpyAsync(res->data, args->src, args->count, args->kind, stream);
            } else if (args->kind == cudaMemcpyDeviceToDevice) {
                res->err = cudaMemcpyAsync(args->dst, args->src, args->count, args->kind, stream);
            } else {
                throw std::runtime_error("Unknown memcpy kind!");
            }

            // cudaStreamSynchronize(stream);

            iox_server->send(res).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) {LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasSgemm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasSgemm_v2");
    auto args = (cublasSgemm_v2Arg *) __args;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cublasSgemm_v2_Partial(args);

    cudaStream_t stream;
    cublasGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = CUBLAS_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    // If client submits to default stream, set to a re-assigned stream
    if (args->stream == nullptr) {
        args->stream = client_data_all[client_id].default_stream;
    }

    auto partial_and_args = cublasLtMatmul_Partial(args);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            args->stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);

            *response = CUBLAS_STATUS_SUCCESS;

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
        int32_t client_id = msg_header->client_id;

        auto pointer_arr = (void **) (args->arrayOfElements);

        for (int i = 0; i < args->elementCount; i++) {
            auto pointer = pointer_arr[i];

            if (pointer == nullptr) {
                continue;
            }
            
            auto found = is_registered_addr(client_data_all[client_id].dev_addr_map, pointer);

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnActivationForward_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnConvolutionForward_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnAddTensor_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnPoolingForward_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cublasSgemv_v2_Partial(args);

    cudaStream_t stream;
    cublasGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = CUBLAS_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnLRNCrossChannelForward_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnSoftmaxForward_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnTransformTensor_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cublasSgemmEx_Partial(args);

    cudaStream_t stream;
    cublasGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = CUBLAS_STATUS_SUCCESS;

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasGemmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
    TALLY_SPD_LOG("Received request: cublasGemmEx");
    auto args = (struct cublasGemmExArg *) __args;
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cublasGemmEx_Partial(args);

    cudaStream_t stream;
    cublasGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = CUBLAS_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnMultiHeadAttnForward_Partial(args);

    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnMultiHeadAttnBackwardData_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnMultiHeadAttnBackwardWeights_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnReorderFilterAndBias_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnRNNForwardTraining_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnRNNBackwardData_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnRNNBackwardWeights_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnBatchNormalizationForwardTrainingEx_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnBatchNormalizationBackwardEx_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cublasSgemmStridedBatched_Partial(args);

    cudaStream_t stream;
    cublasGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = CUBLAS_STATUS_SUCCESS;

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
    int32_t client_id = msg_header->client_id;

    auto server_addr = get_server_addr_from_client_addr(client_id, args->func);
    auto cu_func = original_kernel_map[server_addr].func;

    iox_server->loan(requestHeader, sizeof(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse), alignof(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse*>(responsePayload);
            
            auto err = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &response->numBlocks,
                cu_func,
                args->blockSize,
                args->dynamicSMemSize,
                args->flags
            );
            
            CHECK_CUDA_ERROR(err);

            if (!err) {
                response->err = cudaSuccess;
            } else {
                response->err = cudaErrorInvalidValue;
            }

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
            CHECK_CUDA_ERROR(response->err);

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnRNNBackwardWeights_v8_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnRNNBackwardData_v8_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cudnnRNNForward_Partial(args);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = CUDNN_STATUS_SUCCESS;

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
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    cudnnStatus_t err;
    auto partial_and_args = cudnnBackendExecute_Partial(args, &err);
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);

            *response = err;
            // CHECK_CUDA_ERROR(*response);

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
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            wait_until_launch_queue_empty(client_id);

            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaThreadSynchronize();
            CHECK_CUDA_ERROR(*response);

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
    int32_t client_id = msg_header->client_id;

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_id].default_stream;
    }
    
    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);

            wait_until_launch_queue_empty(client_id);

            *response = cudaEventRecord(
				args->event,
				stream
            );
            CHECK_CUDA_ERROR(*response);

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
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cudaError_t*>(responsePayload);

            wait_until_launch_queue_empty(client_id);

            // Instead of calling cudaDeviceSynchronize, only synchronize all streams of the client
            auto &client_streams = client_data_all[client_id].streams;
            
            for (auto &stream : client_streams) {
                *response = cudaStreamSynchronize(stream);
                CHECK_CUDA_ERROR(*response);
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
    int32_t client_id = msg_header->client_id;

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_id].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            wait_until_launch_queue_empty(client_id);
            
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaStreamSynchronize(
				stream
            );
            CHECK_CUDA_ERROR(*response);

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
    int32_t client_id = msg_header->client_id;

    auto &client_meta = client_data_all[client_id];

    iox_server->loan(requestHeader, sizeof(cublasCreate_v2Response), alignof(cublasCreate_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasCreate_v2Response*>(responsePayload);

            response->err = cublasCreate_v2(&(response->handle));
            response->stream = client_meta.default_stream;
            CHECK_CUDA_ERROR(response->err);

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
    int32_t client_id = msg_header->client_id;

    auto &client_meta = client_data_all[client_id];

    iox_server->loan(requestHeader, sizeof(cudnnCreateResponse), alignof(cudnnCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateResponse*>(responsePayload);
            response->err = cudnnCreate(&(response->handle));
            CHECK_CUDA_ERROR(response->err);

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
    int32_t client_id = msg_header->client_id;

    bool cached = args->cached;
    std::string tmp_elf_file;
    uint32_t cubin_uid = 0;
    size_t cubin_size = args->cubin_size;
    const char *cubin_data;
    size_t msg_len = sizeof(cuModuleLoadDataResponse);

    // load cubin cache
    auto &cubin_cache = TallyCache::cache->get_cubin_cache();
    cubin_cache.load_cubin_cache(cubin_size);
    
    if (!cached) {
        cubin_data = (const char *) args->image;
        cache_cubin_data(cubin_data, cubin_size, client_id);
        tmp_elf_file = get_tmp_file_path(".elf", client_id);
        msg_len += tmp_elf_file.size() + 1;
    } else {
        cubin_uid = args->cubin_uid;
    }

    iox_server->loan(requestHeader, sizeof(cuModuleLoadDataResponse), alignof(cuModuleLoadDataResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuModuleLoadDataResponse*>(responsePayload);

            auto &cubin_cache = TallyCache::cache->get_cubin_cache();
            if (!cached) {
                memcpy(response->tmp_elf_file, tmp_elf_file.c_str(), tmp_elf_file.size());
                response->tmp_elf_file[tmp_elf_file.size()] = '\0';

                cubin_data = cubin_cache.get_cubin_data_ptr(cubin_data, cubin_size);
                cubin_uid = cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);
            } else {
                cubin_data = cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
                cubin_size = cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);
            }

            // Register cu module for this cubin
            register_cu_module(cubin_uid);

            if (cubin_to_cu_module.find(cubin_uid) == cubin_to_cu_module.end()) {
                throw std::runtime_error("Cannot find cu module");
            }

            response->module = cubin_to_cu_module[cubin_uid];
            response->err = CUDA_SUCCESS;

            jit_module_to_cubin_map.insert(response->module, std::make_tuple(
                cubin_data,
                cubin_size,
                cubin_uid
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

    bool cached = false;
    auto kernel_name = std::string(args->name);

    if (jit_module_to_function_map.find(args->hmod) != jit_module_to_function_map.end()) {
        if (jit_module_to_function_map[args->hmod].find(kernel_name) != jit_module_to_function_map[args->hmod].end()) {
            cached = true;
        }
    }

    uint32_t cubin_uid = 0;

    if (!cached) {
        auto cubin_data_size_id = jit_module_to_cubin_map[args->hmod];
        auto cubin_data = std::get<0>(cubin_data_size_id);
        auto cubin_size = std::get<1>(cubin_data_size_id);
        cubin_uid = std::get<2>(cubin_data_size_id);

        auto &cubin_cache = TallyCache::cache->get_cubin_cache();
        auto kernel_names_and_param_sizes = cubin_cache.get_kernel_args(cubin_data, cubin_size);
        auto &param_sizes = kernel_names_and_param_sizes[kernel_name];

        if (cubin_to_kernel_name_to_host_func_map[cubin_uid].find(kernel_name) == cubin_to_kernel_name_to_host_func_map[cubin_uid].end()) {
            void *kernel_server_addr = malloc(8);
            cubin_to_kernel_name_to_host_func_map[cubin_uid].insert(kernel_name, kernel_server_addr);

            host_func_to_demangled_kernel_name_map.insert(kernel_server_addr, kernel_name);
            _kernel_addr_to_args.insert(kernel_server_addr, param_sizes);

            host_func_to_cubin_uid_map.insert(kernel_server_addr, cubin_uid);

            kernel_cubin_uid_to_host_func_map.insert(
                std::make_pair(kernel_name, cubin_uid),
                kernel_server_addr
            );
        }

        register_ptx_transform(cubin_uid);
    }

    iox_server->loan(requestHeader, sizeof(cuModuleGetFunctionResponse), alignof(cuModuleGetFunctionResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuModuleGetFunctionResponse*>(responsePayload);

            if (!cached) {
                response->err = cuModuleGetFunction(&(response->hfunc), args->hmod, args->name);
                CHECK_CUDA_ERROR(response->err);

                jit_module_to_function_map[args->hmod][kernel_name] = response->hfunc;

                // Map CUFunction to host func
                cu_func_addr_mapping.insert(response->hfunc, cubin_to_kernel_name_to_host_func_map[cubin_uid][kernel_name]);
            } else {
                response->err = CUDA_SUCCESS;
                response->hfunc = jit_module_to_function_map[args->hmod][kernel_name];
            }

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
            CHECK_CUDA_ERROR(response->err);

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
    int32_t client_id = msg_header->client_id;

    cudaStream_t stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (stream == nullptr) {
        stream = client_data_all[client_id].default_stream;
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
            CHECK_CUDA_ERROR(response->err);

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
            CHECK_CUDA_ERROR(response->err);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaFuncSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaFuncSetAttribute");
    
	auto args = (struct cudaFuncSetAttributeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;
    
    auto server_func_addr = get_server_addr_from_client_addr(client_id, args->func);
    auto cu_func = original_kernel_map[server_func_addr].func;
    auto cu_func_ptb = ptb_kernel_map[server_func_addr].func;
    auto cu_func_dynamic_ptb = dynamic_ptb_kernel_map[server_func_addr].func;
    auto cu_func_preemptive_ptb = preemptive_ptb_kernel_map[server_func_addr].func;
    auto cu_func_sliced = sliced_kernel_map[server_func_addr].func;

    auto cu_attr = convert_func_attribute(args->attr);

    cuFuncSetAttribute(cu_func, cu_attr, args->value);
    cuFuncSetAttribute(cu_func_ptb, cu_attr, args->value);
    cuFuncSetAttribute(cu_func_dynamic_ptb, cu_attr, args->value);
    cuFuncSetAttribute(cu_func_preemptive_ptb, cu_attr, args->value);
    cuFuncSetAttribute(cu_func_sliced, cu_attr, args->value);

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
    int32_t client_id = msg_header->client_id;

    size_t res_size;

    if (is_registered_addr(client_data_all[client_id].dev_addr_map, (void *) args->dst)) {
        res_size = sizeof(cuMemcpyResponse);
    } else {
        res_size = sizeof(cuMemcpyResponse) + args->ByteCount;
    }

    iox_server->loan(requestHeader, res_size, alignof(cuMemcpyResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuMemcpyResponse*>(responsePayload);

            wait_until_launch_queue_empty(client_id);

            response->err = cuMemcpyAsync(
				is_registered_addr(client_data_all[client_id].dev_addr_map, (void *) args->dst) ? args->dst : (CUdeviceptr) response->data,
				is_registered_addr(client_data_all[client_id].dev_addr_map, (void *) args->src) ? args->src : (CUdeviceptr) args->data,
				args->ByteCount,
                client_data_all[client_id].default_stream
            );
            CHECK_CUDA_ERROR(response->err);

            // Make sure wait until cudaMemcpy complete, because cudaMemcpy is synchronous
            cudaStreamSynchronize(client_data_all[client_id].default_stream);

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
    int32_t client_id = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_id].default_stream;
    }

    size_t res_size;

    if (is_registered_addr(client_data_all[client_id].dev_addr_map, (void *) args->dst)) {
        res_size = sizeof(cuMemcpyAsyncResponse);
    } else {
        res_size = sizeof(cuMemcpyAsyncResponse) + args->ByteCount;
    }

    iox_server->loan(requestHeader, res_size, alignof(cuMemcpyAsyncResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuMemcpyAsyncResponse*>(responsePayload);

            wait_until_launch_queue_empty(client_id);

            response->err = cuMemcpyAsync(
				is_registered_addr(client_data_all[client_id].dev_addr_map, (void *) args->dst) ? args->dst : (CUdeviceptr) response->data,
				is_registered_addr(client_data_all[client_id].dev_addr_map, (void *) args->src) ? args->src : (CUdeviceptr) args->data,
				args->ByteCount,
				__stream
            );
            CHECK_CUDA_ERROR(response->err);

            // cudaStreamSynchronize(__stream);
            
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
    int32_t client_id = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_id].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cuMemAllocAsyncResponse), alignof(cuMemAllocAsyncResponse))
        .and_then([&](auto& responsePayload) {
            
            auto response = static_cast<cuMemAllocAsyncResponse*>(responsePayload);

            response->err = cuMemAllocAsync(
				&(response->dptr),
				args->bytesize,
				__stream
			);
            CHECK_CUDA_ERROR(response->err);

            // Keep track that this addr is device memory
            if (response->err == CUDA_SUCCESS) {
                client_data_all[client_id].dev_addr_map.push_back( mem_region((void *)response->dptr, args->bytesize) );
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
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuMemFree_v2(
				args->dptr
            );
            CHECK_CUDA_ERROR(*response);

            if (*response == CUDA_SUCCESS) {
                free_mem_region(client_data_all[client_id].dev_addr_map, (void *)args->dptr);
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaMemset(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemset");
	auto args = (struct cudaMemsetArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
	auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            // Make sure all kernels have been dispatched
            wait_until_launch_queue_empty(client_id);

            auto response = static_cast<cudaError_t*>(responsePayload);			
            *response = cudaMemsetAsync(
				args->devPtr,
				args->value,
				args->count,
                client_data_all[client_id].default_stream
            );
            CHECK_CUDA_ERROR(*response);

            // The cudaMemset functions are asynchronous with respect to the host except when the
            // target memory is pinned host memory.

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaStreamCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamCreate");
	auto args = (struct cudaStreamCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaStreamCreateResponse), alignof(cudaStreamCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaStreamCreateResponse*>(responsePayload);

            auto policy = SCHEDULER_POLICY;
            if (policy == TALLY_SCHEDULER_POLICY::PRIORITY) {

                int stream_priority = get_client_stream_priority(client_uid);
                response->err = cudaStreamCreateWithPriority(
                    (args->pStream ? &(response->pStream) : NULL),
                    cudaStreamNonBlocking,
                    stream_priority
                );
            } else {
                response->err = cudaStreamCreateWithFlags(
                    (args->pStream ? &(response->pStream) : NULL),
                    cudaStreamNonBlocking
                );
            }

            CHECK_CUDA_ERROR(response->err);

            // Bookkeep the newly created stream
            client_add_stream(client_uid, response->pStream);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaStreamCreateWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamCreateWithFlags");
	auto args = (struct cudaStreamCreateWithFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaStreamCreateWithFlagsResponse), alignof(cudaStreamCreateWithFlagsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaStreamCreateWithFlagsResponse*>(responsePayload);
            
            auto policy = SCHEDULER_POLICY;
            if (policy == TALLY_SCHEDULER_POLICY::PRIORITY) {

                int stream_priority = get_client_stream_priority(client_uid);
                response->err = cudaStreamCreateWithPriority(
                    (args->pStream ? &(response->pStream) : NULL),
                    cudaStreamNonBlocking,
                    stream_priority
                );
            } else {
                response->err = cudaStreamCreateWithFlags(
                    (args->pStream ? &(response->pStream) : NULL),
                    cudaStreamNonBlocking
                );
            }
            CHECK_CUDA_ERROR(response->err);

            // Bookkeep the newly created stream
            client_add_stream(client_uid, response->pStream);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaStreamCreateWithPriority(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamCreateWithPriority");
	auto args = (struct cudaStreamCreateWithPriorityArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cudaStreamCreateWithPriorityResponse), alignof(cudaStreamCreateWithPriorityResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaStreamCreateWithPriorityResponse*>(responsePayload);
            
            auto policy = SCHEDULER_POLICY;
            if (policy == TALLY_SCHEDULER_POLICY::PRIORITY) {

                int stream_priority = get_client_stream_priority(client_uid);
                response->err = cudaStreamCreateWithPriority(
                    (args->pStream ? &(response->pStream) : NULL),
                    cudaStreamNonBlocking,
                    stream_priority
                );
                
            } else {
                response->err = cudaStreamCreateWithFlags(
                    (args->pStream ? &(response->pStream) : NULL),
                    cudaStreamNonBlocking
                );
            }
            CHECK_CUDA_ERROR(response->err);

            // Bookkeep the newly created stream
            client_add_stream(client_uid, response->pStream);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaStreamBeginCapture(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamBeginCapture");
	auto args = (struct cudaStreamBeginCaptureArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

			wait_until_launch_queue_empty(client_uid);

            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaStreamBeginCapture(
				__stream,
				args->mode
            );
            CHECK_CUDA_ERROR(*response);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuStreamCreateWithPriority(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamCreateWithPriority");
	auto args = (struct cuStreamCreateWithPriorityArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cuStreamCreateWithPriorityResponse), alignof(cuStreamCreateWithPriorityResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuStreamCreateWithPriorityResponse*>(responsePayload);

            auto policy = SCHEDULER_POLICY;
            if (policy == TALLY_SCHEDULER_POLICY::PRIORITY) {

                int stream_priority = get_client_stream_priority(client_uid);
                response->err = cuStreamCreateWithPriority(
                    (args->phStream ? &(response->phStream) : NULL),
                    CU_STREAM_NON_BLOCKING,
                    stream_priority
                );
            } else {
                response->err = cuStreamCreate(
                    (args->phStream ? &(response->phStream) : NULL),
                    CU_STREAM_NON_BLOCKING
                );
            }
            CHECK_CUDA_ERROR(response->err);

            // Bookkeep the newly created stream
            client_add_stream(client_uid, response->phStream);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuFuncGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuFuncGetAttribute");
	auto args = (struct cuFuncGetAttributeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

	auto server_func_addr = get_server_addr_from_cu_func(args->hfunc);
	auto cu_func_original = TallyServer::server->original_kernel_map[server_func_addr].func;

    iox_server->loan(requestHeader, sizeof(cuFuncGetAttributeResponse), alignof(cuFuncGetAttributeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuFuncGetAttributeResponse*>(responsePayload);
            response->err = cuFuncGetAttribute(
				(args->pi ? &(response->pi) : NULL),
				args->attrib,
				cu_func_original
			);
            CHECK_CUDA_ERROR(response->err);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuFuncSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuFuncSetAttribute");
	auto args = (struct cuFuncSetAttributeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

	auto server_func_addr = get_server_addr_from_cu_func(args->hfunc);
	auto cu_func_original = TallyServer::server->original_kernel_map[server_func_addr].func;
	auto cu_func_ptb = TallyServer::server->ptb_kernel_map[server_func_addr].func;
	auto cu_func_dynamic = TallyServer::server->dynamic_ptb_kernel_map[server_func_addr].func;
	auto cu_func_preemptive = TallyServer::server->preemptive_ptb_kernel_map[server_func_addr].func;
    auto cu_func_sliced = TallyServer::server->sliced_kernel_map[server_func_addr].func;

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);

            *response = cuFuncSetAttribute(cu_func_original, args->attrib, args->value);
			cuFuncSetAttribute(cu_func_ptb, args->attrib, args->value);
			cuFuncSetAttribute(cu_func_dynamic, args->attrib, args->value);
            cuFuncSetAttribute(cu_func_preemptive, args->attrib, args->value);
			cuFuncSetAttribute(cu_func_sliced, args->attrib, args->value);
            CHECK_CUDA_ERROR(*response);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuFuncSetCacheConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuFuncSetCacheConfig");
	auto args = (struct cuFuncSetCacheConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto server_func_addr = get_server_addr_from_cu_func(args->hfunc);
	auto cu_func_original = TallyServer::server->original_kernel_map[server_func_addr].func;
	auto cu_func_ptb = TallyServer::server->ptb_kernel_map[server_func_addr].func;
	auto cu_func_dynamic = TallyServer::server->dynamic_ptb_kernel_map[server_func_addr].func;
	auto cu_func_preemptive = TallyServer::server->preemptive_ptb_kernel_map[server_func_addr].func;
    auto cu_func_sliced = TallyServer::server->sliced_kernel_map[server_func_addr].func;

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);

            *response = cuFuncSetCacheConfig(cu_func_original, args->config);
            cuFuncSetCacheConfig(cu_func_ptb, args->config);
            cuFuncSetCacheConfig(cu_func_dynamic, args->config);
            cuFuncSetCacheConfig(cu_func_preemptive, args->config);
            cuFuncSetCacheConfig(cu_func_sliced, args->config);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasGemmStridedBatchedEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGemmStridedBatchedEx");
	auto args = (struct cublasGemmStridedBatchedExArg *) __args;
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    auto partial_and_args = cublasGemmStridedBatchedEx_Partial(args);

    cudaStream_t stream;
    cublasGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = CUBLAS_STATUS_SUCCESS;

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemsetD8_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD8_v2");
	auto args = (struct cuMemsetD8_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {

            // Make sure all kernels have been dispatched
            wait_until_launch_queue_empty(client_id);

            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuMemsetD8Async(
				args->dstDevice,
				args->uc,
				args->N,
                client_data_all[client_id].default_stream
            );
            CHECK_CUDA_ERROR(*response);

            // The cudaMemset functions are asynchronous with respect to the host except when the
            // target memory is pinned host memory.

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuStreamCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamCreate");
	auto args = (struct cuStreamCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cuStreamCreateResponse), alignof(cuStreamCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuStreamCreateResponse*>(responsePayload);

            auto policy = SCHEDULER_POLICY;
            if (policy == TALLY_SCHEDULER_POLICY::PRIORITY) {

                int stream_priority = get_client_stream_priority(client_uid);
                response->err = cuStreamCreateWithPriority(
                    (args->phStream ? &(response->phStream) : NULL),
                    CU_STREAM_NON_BLOCKING,
                    stream_priority
                );
            } else {
                response->err = cuStreamCreate(
                    (args->phStream ? &(response->phStream) : NULL),
                    CU_STREAM_NON_BLOCKING
                );
            }
            CHECK_CUDA_ERROR(response->err);

            // Bookkeep the newly created stream
            client_add_stream(client_uid, response->phStream);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemAlloc_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAlloc_v2");
	auto args = (struct cuMemAlloc_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cuMemAlloc_v2Response), alignof(cuMemAlloc_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuMemAlloc_v2Response*>(responsePayload);
            response->err = cuMemAlloc_v2(
				(args->dptr ? &(response->dptr) : NULL),
				args->bytesize
			);
            CHECK_CUDA_ERROR(response->err);

            // Keep track that this addr is device memory
            if (response->err == CUDA_SUCCESS) {
                client_data_all[client_id].dev_addr_map.push_back( mem_region((void *)response->dptr, args->bytesize) );
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemsetD32_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD32_v2");
	auto args = (struct cuMemsetD32_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuMemsetD32Async(
				args->dstDevice,
				args->ui,
				args->N,
                client_data_all[client_id].default_stream
            );
            CHECK_CUDA_ERROR(*response);

            // The cudaMemset functions are asynchronous with respect to the host except when the
            // target memory is pinned host memory.

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemcpyHtoDAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyHtoDAsync_v2");
	auto args = (struct cuMemcpyHtoDAsync_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    CUstream  stream = args->hStream;

    if (stream == nullptr) {
        stream = client_data_all[client_id].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(cuMemcpyResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);

            wait_until_launch_queue_empty(client_id);

            *response = cuMemcpyHtoDAsync_v2(
				args->dstDevice,
				args->data,
				args->ByteCount,
                stream
            );
            CHECK_CUDA_ERROR(*response);

            // cudaStreamSynchronize(stream);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemcpyDtoHAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyDtoHAsync_v2");
	auto args = (struct cuMemcpyDtoHAsync_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_id].default_stream;
    }

    size_t res_size = sizeof(cuMemcpyDtoHAsync_v2Response) + args->ByteCount;

    iox_server->loan(requestHeader, res_size, alignof(MessageHeader_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuMemcpyDtoHAsync_v2Response*>(responsePayload);

            wait_until_launch_queue_empty(client_id);

            response->err = cuMemcpyDtoHAsync_v2(
				response->data,
				args->srcDevice,
				args->ByteCount,
				__stream
            );
            CHECK_CUDA_ERROR(response->err);

            // cudaStreamSynchronize(__stream);
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemsetD32Async(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD32Async");
	auto args = (struct cuMemsetD32AsyncArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_id].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {

            // Make sure all kernels have been dispatched
            wait_until_launch_queue_empty(client_id);

            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuMemsetD32Async(
				args->dstDevice,
				args->ui,
				args->N,
                __stream
            );
            CHECK_CUDA_ERROR(*response);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuMemcpyDtoDAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyDtoDAsync_v2");
	auto args = (struct cuMemcpyDtoDAsync_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);

            wait_until_launch_queue_empty(client_uid);

            *response = cuMemcpyDtoDAsync_v2(
				args->dstDevice,
				args->srcDevice,
				args->ByteCount,
				__stream
            );
            CHECK_CUDA_ERROR(*response);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuModuleLoadFatBinary(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleLoadFatBinary");
	auto args = (struct cuModuleLoadFatBinaryArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    bool cached = args->cached;
    std::string tmp_elf_file;
    uint32_t cubin_uid = 0;
    size_t msg_len = sizeof(cuModuleLoadFatBinaryResponse);

    size_t cubin_size = args->cubin_size;
    const char *cubin_data;

    // load cubin cache
    auto &cubin_cache = TallyCache::cache->get_cubin_cache();
    cubin_cache.load_cubin_cache(cubin_size);
    
    if (!cached) {
        cubin_data = (const char *) args->image;

        cache_cubin_data(cubin_data, cubin_size, client_id);
        tmp_elf_file = get_tmp_file_path(".elf", client_id);
        msg_len += tmp_elf_file.size() + 1;
    } else {
        cubin_uid = args->cubin_uid;
    }

    iox_server->loan(requestHeader, sizeof(cuModuleLoadFatBinaryResponse), alignof(cuModuleLoadFatBinaryResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuModuleLoadFatBinaryResponse*>(responsePayload);

            auto &cubin_cache = TallyCache::cache->get_cubin_cache();
            if (!cached) {
                memcpy(response->tmp_elf_file, tmp_elf_file.c_str(), tmp_elf_file.size());
                response->tmp_elf_file[tmp_elf_file.size()] = '\0';

                cubin_data = cubin_cache.get_cubin_data_ptr(cubin_data, cubin_size);
                cubin_uid = cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);
            } else {
                cubin_data = cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
                cubin_size = cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);
            }

            // Register cu module for this cubin
            register_cu_module(cubin_uid);

            response->module = cubin_to_cu_module[cubin_uid];
            response->err = CUDA_SUCCESS;

            jit_module_to_cubin_map.insert(response->module, std::make_tuple(
                cubin_data,
                cubin_size,
                cubin_uid
            ));

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuModuleLoadDataEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleLoadDataEx");
	auto args = (struct cuModuleLoadDataExArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    bool cached = args->cached;
    std::string tmp_elf_file;
    uint32_t cubin_uid = 0;
    size_t msg_len = sizeof(cuModuleLoadDataExResponse);

    size_t cubin_size = args->cubin_size;
    const char *cubin_data;

    // load cubin cache
    auto &cubin_cache = TallyCache::cache->get_cubin_cache();
    cubin_cache.load_cubin_cache(cubin_size);

    if (!cached) {
        cubin_data = (const char *) args->image;
        cache_cubin_data(cubin_data, cubin_size, client_id);
        tmp_elf_file = get_tmp_file_path(".elf", client_id);
        msg_len += tmp_elf_file.size() + 1;
    } else {
        cubin_uid = args->cubin_uid;
    }

    iox_server->loan(requestHeader, sizeof(cuModuleLoadDataExResponse), alignof(cuModuleLoadDataExResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuModuleLoadDataExResponse*>(responsePayload);

            auto &cubin_cache = TallyCache::cache->get_cubin_cache();
            if (!cached) {
                memcpy(response->tmp_elf_file, tmp_elf_file.c_str(), tmp_elf_file.size());
                response->tmp_elf_file[tmp_elf_file.size()] = '\0';

                cubin_data = cubin_cache.get_cubin_data_ptr(cubin_data, cubin_size);
                cubin_uid = cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);
            } else {
                cubin_data = cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
                cubin_size = cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);
            }

            // Register cu module for this cubin
            register_cu_module(cubin_uid);

            if (cubin_to_cu_module.find(cubin_uid) == cubin_to_cu_module.end()) {
                throw std::runtime_error("Cannot find cu module");
            }

            response->module = cubin_to_cu_module[cubin_uid];
            response->err = CUDA_SUCCESS;

            jit_module_to_cubin_map.insert(response->module, std::make_tuple(
                cubin_data,
                cubin_size,
                cubin_uid
            ));

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuModuleGetGlobal_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleGetGlobal_v2");
	auto args = (struct cuModuleGetGlobal_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(cuModuleGetGlobal_v2Response), alignof(CUresult))
        .and_then([&](auto& responsePayload) {

            auto response = static_cast<cuModuleGetGlobal_v2Response*>(responsePayload);

            response->err = cuModuleGetGlobal_v2(
				&response->dptr,
				&response->bytes,
				args->hmod,
                args->name
            );

            CHECK_CUDA_ERROR(response->err);
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuCtxSynchronize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxSynchronize");
	auto args = (struct cuCtxSynchronizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);

            wait_until_launch_queue_empty(client_id);

            // Instead of calling cudaDeviceSynchronize, only synchronize all streams of the client
            auto &client_streams = client_data_all[client_id].streams;
            
            for (auto &stream : client_streams) {
                auto err = cudaStreamSynchronize(stream);
                CHECK_CUDA_ERROR(err);
            }

            *response = CUDA_SUCCESS;
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuStreamSynchronize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamSynchronize");
	auto args = (struct cuStreamSynchronizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);

			wait_until_launch_queue_empty(client_uid);

            *response = cuStreamSynchronize(
				__stream
            );
            CHECK_CUDA_ERROR(*response);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuModuleUnload(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleUnload");
	auto args = (struct cuModuleUnloadArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            // *response = cuModuleUnload(
			// 	args->hmod
            // );

            *response = CUDA_SUCCESS;

            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaStreamEndCapture(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamEndCapture");
	auto args = (struct cudaStreamEndCaptureArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaStreamEndCaptureResponse), alignof(cudaStreamEndCaptureResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaStreamEndCaptureResponse*>(responsePayload);

            wait_until_launch_queue_empty(client_uid);

            response->err = cudaStreamEndCapture(
				__stream,
				(args->pGraph ? &(response->pGraph) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuStreamEndCapture(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamEndCapture");
	auto args = (struct cuStreamEndCaptureArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cuStreamEndCaptureResponse), alignof(cuStreamEndCaptureResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuStreamEndCaptureResponse*>(responsePayload);

			wait_until_launch_queue_empty(client_uid);

            response->err = cuStreamEndCapture(
				__stream,
				(args->phGraph ? &(response->phGraph) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

// TODO: make graph launch a partial too
void TallyServer::handle_cuGraphLaunch(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphLaunch");
	auto args = (struct cuGraphLaunchArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);

            wait_until_launch_queue_empty(client_uid);
			cudaDeviceSynchronize();

            *response = cuGraphLaunch(
				args->hGraphExec,
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasSetMathMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetMathMode");
	auto args = (struct cublasSetMathModeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasSetMathMode(
				args->handle,
				args->mode
            );

            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasDestroy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDestroy_v2");
	auto args = (struct cublasDestroy_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasDestroy_v2(
				args->handle
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasSetStream_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetStream_v2");
	auto args = (struct cublasSetStream_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->streamId;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasSetStream_v2(
				args->handle,
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasSetWorkspace_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetWorkspace_v2");
	auto args = (struct cublasSetWorkspace_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasSetWorkspace_v2(
				args->handle,
				args->workspace,
				args->workspaceSizeInBytes
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtCreate");
	auto args = (struct cublasLtCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;
    auto &client_meta = client_data_all[client_id];

    iox_server->loan(requestHeader, sizeof(cublasLtCreateResponse), alignof(cublasLtCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasLtCreateResponse*>(responsePayload);
            response->err = cublasLtCreate(
				(args->lightHandle ? &(response->lightHandle) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatmulDescCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulDescCreate");
	auto args = (struct cublasLtMatmulDescCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasLtMatmulDescCreateResponse), alignof(cublasLtMatmulDescCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasLtMatmulDescCreateResponse*>(responsePayload);
            response->err = cublasLtMatmulDescCreate(
				(args->matmulDesc ? &(response->matmulDesc) : NULL),
				args->computeType,
				args->scaleType
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatrixLayoutCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixLayoutCreate");
	auto args = (struct cublasLtMatrixLayoutCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasLtMatrixLayoutCreateResponse), alignof(cublasLtMatrixLayoutCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasLtMatrixLayoutCreateResponse*>(responsePayload);
            response->err = cublasLtMatrixLayoutCreate(
				(args->matLayout ? &(response->matLayout) : NULL),
				args->type,
				args->rows,
				args->cols,
				args->ld
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatmulPreferenceCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulPreferenceCreate");
	auto args = (struct cublasLtMatmulPreferenceCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasLtMatmulPreferenceCreateResponse), alignof(cublasLtMatmulPreferenceCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasLtMatmulPreferenceCreateResponse*>(responsePayload);
            response->err = cublasLtMatmulPreferenceCreate(
				(args->pref ? &(response->pref) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaPointerGetAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaPointerGetAttributes");
	auto args = (struct cudaPointerGetAttributesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaPointerGetAttributesResponse), alignof(cudaPointerGetAttributesResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaPointerGetAttributesResponse*>(responsePayload);
            response->err = cudaPointerGetAttributes(
				(args->attributes ? &(response->attributes) : NULL),
				args->ptr
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuDevicePrimaryCtxSetFlags_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDevicePrimaryCtxSetFlags_v2");
	auto args = (struct cuDevicePrimaryCtxSetFlags_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuDevicePrimaryCtxSetFlags_v2(
				args->dev,
				args->flags
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaMallocHost(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMallocHost");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

void TallyServer::handle_cuMemHostAlloc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemHostAlloc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

void TallyServer::handle_cudaHostAlloc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaHostAlloc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

void TallyServer::handle_cudaFreeHost(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaFreeHost");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

void TallyServer::handle_cuDeviceGetName(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetName");
	auto args = (struct cuDeviceGetNameArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    size_t msg_len = sizeof(cuDeviceGetNameResponse) + args->len;

    iox_server->loan(requestHeader, msg_len, alignof(cuDeviceGetNameResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetNameResponse*>(responsePayload);
            response->err = cuDeviceGetName(
                response->name,
				args->len,
				args->dev
            );
            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaFuncGetAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaFuncGetAttributes");
	auto args = (struct cudaFuncGetAttributesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;
    
    auto server_func_addr = get_server_addr_from_client_addr(client_id, args->func);
    auto cu_func = original_kernel_map[server_func_addr].func;

    iox_server->loan(requestHeader, sizeof(cudaFuncGetAttributesResponse), alignof(cudaFuncGetAttributesResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaFuncGetAttributesResponse*>(responsePayload);

            int constSizeBytes;
            int localSizeBytes;
            int sharedSizeBytes;

            auto err = cuFuncGetAttribute (&(response->attr.binaryVersion), CU_FUNC_ATTRIBUTE_BINARY_VERSION, cu_func);
            cuFuncGetAttribute (&(response->attr.cacheModeCA), CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, cu_func);
            cuFuncGetAttribute (&(response->attr.clusterDimMustBeSet), CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, cu_func);
            cuFuncGetAttribute (&(response->attr.clusterSchedulingPolicyPreference), CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, cu_func);
            cuFuncGetAttribute (&(response->attr.maxDynamicSharedSizeBytes), CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, cu_func);
            cuFuncGetAttribute (&(response->attr.maxThreadsPerBlock), CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cu_func);
            cuFuncGetAttribute (&(response->attr.nonPortableClusterSizeAllowed), CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, cu_func);
            cuFuncGetAttribute (&(response->attr.numRegs), CU_FUNC_ATTRIBUTE_NUM_REGS, cu_func);
            cuFuncGetAttribute (&(response->attr.preferredShmemCarveout), CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, cu_func);
            cuFuncGetAttribute (&(response->attr.ptxVersion), CU_FUNC_ATTRIBUTE_PTX_VERSION, cu_func);
            cuFuncGetAttribute (&(response->attr.requiredClusterWidth), CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, cu_func);
            cuFuncGetAttribute (&(response->attr.requiredClusterHeight), CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, cu_func);
            cuFuncGetAttribute (&(response->attr.requiredClusterDepth), CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, cu_func);
            

            cuFuncGetAttribute (&constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, cu_func);
            cuFuncGetAttribute (&localSizeBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, cu_func);
            cuFuncGetAttribute (&sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, cu_func);

            response->attr.constSizeBytes = (size_t) constSizeBytes;
            response->attr.localSizeBytes = (size_t) localSizeBytes;
            response->attr.sharedSizeBytes = (size_t) sharedSizeBytes;

            if (err) {
                response->err = cudaErrorInvalidValue;
            } else {
                response->err = cudaSuccess;
            }

            CHECK_CUDA_ERROR(response->err);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnGetErrorString(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetErrorString");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

void TallyServer::handle_ncclCommInitRankConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommInitRankConfig");
	auto args = (struct ncclCommInitRankConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    // Check if netName is set
    if (args->config.netName) {
        args->config.netName = args->netName;
    }

    iox_server->loan(requestHeader, sizeof(ncclCommInitRankConfigResponse), alignof(ncclCommInitRankConfigResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<ncclCommInitRankConfigResponse*>(responsePayload);
            response->err = ncclCommInitRankConfig(
				(args->comm ? &(response->comm) : NULL),
				args->nranks,
				args->commId,
				args->rank,
				&(args->config)
			);

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaGetDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetDevice");
	auto args = (struct cudaGetDeviceArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaGetDeviceResponse), alignof(cudaGetDeviceResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaGetDeviceResponse*>(responsePayload);
            response->err = cudaGetDevice(
				(args->device ? &(response->device) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuDevicePrimaryCtxGetState(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDevicePrimaryCtxGetState");
	auto args = (struct cuDevicePrimaryCtxGetStateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDevicePrimaryCtxGetStateResponse), alignof(cuDevicePrimaryCtxGetStateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDevicePrimaryCtxGetStateResponse*>(responsePayload);
            response->err = cuDevicePrimaryCtxGetState(
				args->dev,
				(args->flags ? &(response->flags) : NULL),
				(args->active ? &(response->active) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatrixLayoutDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixLayoutDestroy");
	auto args = (struct cublasLtMatrixLayoutDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatrixLayoutDestroy(
				args->matLayout
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatmulDescDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulDescDestroy");
	auto args = (struct cublasLtMatmulDescDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatmulDescDestroy(
				args->matmulDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaGetLastError(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetLastError");
	auto args = (struct cudaGetLastErrorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaGetLastError(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtDestroy");
	auto args = (struct cublasLtDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtDestroy(
				args->lightHandle
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasGetMathMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetMathMode");
	auto args = (struct cublasGetMathModeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasGetMathModeResponse), alignof(cublasGetMathModeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasGetMathModeResponse*>(responsePayload);
            response->err = cublasGetMathMode(
				args->handle,
				(args->mode ? &(response->mode) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cublasLtMatmulPreferenceDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulPreferenceDestroy");
	auto args = (struct cublasLtMatmulPreferenceDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatmulPreferenceDestroy(
				args->pref
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaMemsetAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemsetAsync");
	auto args = (struct cudaMemsetAsyncArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {

            wait_until_launch_queue_empty(client_uid);

            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaMemsetAsync(
				args->devPtr,
				args->value,
				args->count,
				__stream
            );
            
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaPeekAtLastError(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaPeekAtLastError");
	auto args = (struct cudaPeekAtLastErrorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaPeekAtLastError(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuCtxGetApiVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetApiVersion");
	auto args = (struct cuCtxGetApiVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetApiVersionResponse), alignof(cuCtxGetApiVersionResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetApiVersionResponse*>(responsePayload);
            response->err = cuCtxGetApiVersion(
				args->ctx,
				(args->version ? &(response->version) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuCtxGetDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetDevice");
	auto args = (struct cuCtxGetDeviceArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetDeviceResponse), alignof(cuCtxGetDeviceResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetDeviceResponse*>(responsePayload);
            response->err = cuCtxGetDevice(
				(args->device ? &(response->device) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuCtxGetCurrent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetCurrent");
	auto args = (struct cuCtxGetCurrentArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetCurrentResponse), alignof(cuCtxGetCurrentResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetCurrentResponse*>(responsePayload);
            response->err = cuCtxGetCurrent(
				(args->pctx ? &(response->pctx) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuCtxCreate_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxCreate_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

void TallyServer::handle_cudaDeviceGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetAttribute");
	auto args = (struct cudaDeviceGetAttributeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDeviceGetAttributeResponse), alignof(cudaDeviceGetAttributeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDeviceGetAttributeResponse*>(responsePayload);
            response->err = cudaDeviceGetAttribute(
				(args->value ? &(response->value) : NULL),
				args->attr,
				args->device
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuGraphInstantiateWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphInstantiateWithFlags");
	auto args = (struct cuGraphInstantiateWithFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuGraphInstantiateWithFlagsResponse), alignof(cuGraphInstantiateWithFlagsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuGraphInstantiateWithFlagsResponse*>(responsePayload);
            // response->err = cuGraphInstantiateWithFlags(
			// 	(args->phGraphExec ? &(response->phGraphExec) : NULL),
			// 	args->hGraph,
			// 	args->flags
			// );

            TALLY_SPD_WARN("Failing cuGraphInstantiateWithFlags intentionally");
            response->err = CUDA_ERROR_INVALID_VALUE;

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuGetExportTable(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGetExportTable");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

void TallyServer::handle_cudaStreamIsCapturing(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamIsCapturing");
	auto args = (struct cudaStreamIsCapturingArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaStreamIsCapturingResponse), alignof(cudaStreamIsCapturingResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaStreamIsCapturingResponse*>(responsePayload);
            response->err = cudaStreamIsCapturing(
				__stream,
				(args->pCaptureStatus ? &(response->pCaptureStatus) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuStreamBeginCapture_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamBeginCapture_v2");
	auto args = (struct cuStreamBeginCapture_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuStreamBeginCapture_v2(
				__stream,
				args->mode
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cuStreamIsCapturing(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamIsCapturing");
	auto args = (struct cuStreamIsCapturingArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cuStreamIsCapturingResponse), alignof(cuStreamIsCapturingResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuStreamIsCapturingResponse*>(responsePayload);
            response->err = cuStreamIsCapturing(
				__stream,
				(args->captureStatus ? &(response->captureStatus) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudnnReduceTensor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnReduceTensor");
	auto args = (cudnnReduceTensorArg *) __args;
    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_id = msg_header->client_id;

    void *indices = nullptr;
    if (args->indices) {
        indices = malloc(args->indicesSizeInBytes);
    }

    auto partial_and_args = cudnnReduceTensor_Partial(args, indices);
    
    cudaStream_t stream;
    cudnnGetStream(args->handle, &stream);

    increment_client_queue_size(client_id);
    client_data_all[client_id].kernel_dispatch_queue.enqueue(
        KernelLaunchWrapper(
            partial_and_args.first,
            partial_and_args.second,
            true,
            CudaLaunchCall(0, 0, 0),
            stream,
            0
        )
    );

    wait_until_launch_queue_empty(client_id);

    size_t res_len = sizeof(cudnnStatus_t) + args->workspaceSizeInBytes;

    auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);
    iox_server->loan(requestHeader, res_len, alignof(cudnnReduceTensorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnReduceTensorResponse*>(responsePayload);
            response->err = CUDNN_STATUS_SUCCESS;
            if (indices) {
                memcpy(response->indices, indices, args->indicesSizeInBytes);
                free(indices);
            }

            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}

void TallyServer::handle_cudaLaunchHostFunc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaLaunchHostFunc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}