#include <thread>

#include "tally/generated/server.h"

int main(int argc, char ** argv) {

    // Server thread, handling CUDA requests
    std::thread server_t(&TallyServer::start_server, TallyServer::server);

    // Kernel scheduler, scheduling kernel launches
    std::thread scheduler_t(&TallyServer::start_scheduler, TallyServer::server);

    server_t.join();
    scheduler_t.join();

    return 0;
}