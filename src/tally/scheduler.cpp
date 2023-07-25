#include <cassert>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>

void TallyServer::start_scheduler()
{

#ifdef PROFILE_KERNEL_WISE

    std::function<void()> kernel_partial;

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

        // launch both kernels
        const int durationSeconds = 10;

        auto startTime = std::chrono::high_resolution_clock::now();
        double elapsedSeconds = 0.;
        uint32_t iters[2] = { 0, 0 };

        while (true) {

            int index = 0;
            for (auto &pair : client_data) {
                auto &info = pair.second;
                (*info.kernel_to_dispatch)();
                iters[index]++;
                index++;
            }

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

        std::cout << "elapsedSeconds: " << elapsedSeconds << " iters 1: " << iters[0] << " iters 2: " <<  iters[1] << std::endl;

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