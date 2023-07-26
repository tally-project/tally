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

        std::string kernel_names[2];
        float iters[2];
        float time_elapsed[2];
        
        cudaError_t errs[2];
        cudaError_t *errs_ptr = errs;

        auto launch_kernel_func = [&, errs_ptr](int idx, CudaLaunchConfig config) {

            int index = 0;
            std::function<cudaError_t(CudaLaunchConfig, bool, float, float*, float*)> kernel_partial;

            for (auto &pair : client_data) {

                if (index == idx) {
                    auto &info = pair.second;
                    kernel_partial = *info.kernel_to_dispatch;
                    kernel_names[idx] = host_func_to_demangled_kernel_name_map[info.launch_call.func];
                    break;
                }
                index++;
            }

            errs_ptr[idx] = kernel_partial(config, true, 5, &(time_elapsed[idx]), &(iters[idx]));
        };

        cudaDeviceSynchronize();

        std::thread launch_t_1(launch_kernel_func, 0, CudaLaunchConfig::default_config);
        std::thread launch_t_2(launch_kernel_func, 1, CudaLaunchConfig::default_config);

        launch_t_1.join();
        launch_t_2.join();

        std::cout << "Kernel 1: " << kernel_names[0] << ": Time: " << time_elapsed[0] << " Iterations: " << iters[0] << std::endl;
        std::cout << "Kernel 2: " << kernel_names[1] << ": Time: " << time_elapsed[1] << " Iterations: " << iters[1] << std::endl;

        // clear the flags
        int index = 0;
        for (auto &pair : client_data) {
            auto &info = pair.second;
            info.err = errs[index];
            info.has_kernel = false;
            index++;
        }
    }

#else
    while (!iox::posix::hasTerminationRequested()) {

        for (auto &pair : client_data) {

            auto &info = pair.second;

            if (info.has_kernel) {
                info.err = (*info.kernel_to_dispatch)(CudaLaunchConfig::default_config, false, 0, nullptr, nullptr);
                info.has_kernel = false;
            }

        }
    }

#endif
}