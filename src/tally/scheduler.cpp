#include <cassert>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>

void TallyServer::start_scheduler()
{

#ifdef PROFILE_KERNEL_WISE

    while (!iox::posix::hasTerminationRequested()) {

        int kernel_count = 0;
        int client_count = 0;

        for (auto &pair : client_data) {

            client_count++;
            auto &info = pair.second;

            if (info.has_kernel) {
                kernel_count++;
            }

        }

        if (kernel_count < 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        assert(kernel_count == 2);

        uint32_t iters[2] = { 0, 0 };
        double time_elapsed[2] = { 0., 0. };

        uint32_t *iters_ptr = iters;
        double *time_elapsed_ptr = time_elapsed;

        auto launch_kernel_func = [&, iters_ptr, time_elapsed_ptr](int idx) {

            int index = 0;
            std::function<void()> kernel_partial;

            for (auto &pair : client_data) {

                if (index == idx) {
                    auto &info = pair.second;
                    kernel_partial = *info.kernel_to_dispatch;
                    break;
                }
                index++;
            }

            double elapsedSeconds;
            const int durationSeconds = 10;
            auto startTime = std::chrono::high_resolution_clock::now();

            while (true) {

                kernel_partial();
                iters_ptr[idx]++;

                // Get the current time
                auto currentTime = std::chrono::high_resolution_clock::now();

                // Calculate the elapsed time in seconds
                std::chrono::duration<double> elapsedTime = currentTime - startTime;
                elapsedSeconds = elapsedTime.count();

                // Check if the desired duration has passed
                if (elapsedSeconds >= durationSeconds) {
                    cudaDeviceSynchronize();

                    currentTime = std::chrono::high_resolution_clock::now();
                    elapsedTime = currentTime - startTime;
                    elapsedSeconds = elapsedTime.count();

                    break;
                }
            }

            time_elapsed_ptr[idx] = elapsedSeconds;
        };

        std::thread launch_t_1(launch_kernel_func, 0);
        std::thread launch_t_2(launch_kernel_func, 1);

        launch_t_1.join();
        launch_t_2.join();

        std::cout << "Kernel 1: Time: " << time_elapsed[0] << " Iterations: " << iters[0] << std::endl;
        std::cout << "Kernel 2: Time: " << time_elapsed[1] << " Iterations: " << iters[1] << std::endl;

        // clear the flags
        for (auto &pair : client_data) {
            auto &info = pair.second;
            info.has_kernel = false;
        }
    }

#else

    std::function<void()> kernel_partial;

    while (!iox::posix::hasTerminationRequested()) {

        for (auto &pair : client_data) {

            auto &info = pair.second;

            if (info.has_kernel) {
                (*info.kernel_to_dispatch)();
                info.has_kernel = false;
            }

        }
    }

#endif
}