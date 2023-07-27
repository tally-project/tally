#include <cassert>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>

void TallyServer::start_scheduler()
{

#ifdef PROFILE_KERNEL_WISE

    while (!iox::posix::hasTerminationRequested()) {

        int kernel_count = 0;

        for (auto &pair : client_data) {
            auto &info = pair.second;
            if (info.has_kernel) {
                kernel_count++;
                (*info.kernel_to_dispatch)(CudaLaunchConfig::default_config, false, 0, nullptr, nullptr);
            }
        }

        if (kernel_count < 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        assert(kernel_count == 2);

        CudaLaunchCall launch_calls[2];
        std::function<cudaError_t(CudaLaunchConfig, bool, float, float*, float*)> kernel_partials[2];
        std::string kernel_names[2];

        int index = 0;
        for (auto &pair : client_data) {

            auto &info = pair.second;
            launch_calls[index] = info.launch_call; 
            kernel_partials[index] = *info.kernel_to_dispatch;
            kernel_names[index] = host_func_to_demangled_kernel_name_map[info.launch_call.func];
            index++;
        }

        float iters[2];
        float time_elapsed[2];
        cudaError_t errs[2];

        auto launch_kernel_func = [kernel_partials, &iters, &time_elapsed, &errs](int idx, CudaLaunchConfig config) {
            errs[idx] = (kernel_partials[idx])(config, true, 5, &(time_elapsed[idx]), &(iters[idx]));
        };

        float baseline[2];

        launch_kernel_func(0, CudaLaunchConfig::default_config);
        launch_kernel_func(1, CudaLaunchConfig::default_config);
        baseline[0] = iters[0] / time_elapsed[0];
        baseline[1] = iters[1] / time_elapsed[1];

        auto k1_blockDim = launch_calls[0].blockDim;
        auto k2_blockDim = launch_calls[1].blockDim;

        auto k1_configs = CudaLaunchConfig::get_configs(k1_blockDim.x * k1_blockDim.y * k1_blockDim.z);
        auto k2_configs = CudaLaunchConfig::get_configs(k2_blockDim.x * k2_blockDim.y * k2_blockDim.z);

        for (auto &k1_config : k1_configs) {
            for (auto &k2_config : k2_configs) {
                cudaDeviceSynchronize();

                std::thread launch_t_1(launch_kernel_func, 0, k1_config);
                std::thread launch_t_2(launch_kernel_func, 1, k2_config);

                launch_t_1.join();
                launch_t_2.join();

                float k1_thrupt = iters[0] / time_elapsed[0];
                float k2_thrupt = iters[1] / time_elapsed[1];

                std::cout << "=========== Begin Profiling ============" << std::endl;

                std::cout << "Kernel 1: " << std::endl;
                std::cout << "\t" << kernel_names[0] << std::endl;
                std::cout << "\t" << k1_config << std::endl;
                std::cout << "\tTime: " << time_elapsed[0] << " Iters: " << iters[0] << std::endl;

                std::cout << "Kernel 2: " << std::endl;
                std::cout << "\t" << kernel_names[1] << std::endl;
                std::cout << "\t" << k2_config << std::endl;
                std::cout << "\tTime: " << time_elapsed[1] << " Iters: " << iters[1] << std::endl;

                std::cout << "Kernel 1 normalized throughput: " << k1_thrupt / baseline[0] << std::endl;
                std::cout << "Kernel 2 normalized throughput: " << k2_thrupt / baseline[1] << std::endl;
            }
        }

        // clear the flags
        index = 0;
        for (auto &pair : client_data) {
            auto &info = pair.second;
            info.err = errs[index];
            info.has_kernel = false;
            index++;
        }
    }

#else
    CudaLaunchConfig config = CudaLaunchConfig::default_config;
    // CudaLaunchConfig config(false, false, true, false, 0, 4);

    while (!iox::posix::hasTerminationRequested()) {

        for (auto &pair : client_data) {

            auto &info = pair.second;

            if (info.has_kernel) {
                info.err = (*info.kernel_to_dispatch)(config, false, 0, nullptr, nullptr);
                info.has_kernel = false;
            }

        }
    }

#endif
}