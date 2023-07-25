#include <cassert>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>

void TallyServer::start_scheduler()
{
    std::function<void()> kernel_partial;

    while (!iox::posix::hasTerminationRequested()) {

        for (auto &pair : client_data) {

            auto &_client_data = pair.second;

            if (_client_data.has_kernel) {
                (*_client_data.kernel_to_dispatch)();
                _client_data.has_kernel = false;
            }
        }
    }
}