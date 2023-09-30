#include <thread>

#include "tally/generated/server.h"

int main(int argc, char ** argv) {

    // Main server thread, will spawn one server thread for each client
    auto main_server_t = new std::thread(&TallyServer::start_main_server, TallyServer::server);

    // Kernel scheduler, scheduling kernel launches from all clients
    auto scheduler_t = new std::thread(&TallyServer::start_scheduler, TallyServer::server);

    main_server_t->join();
    scheduler_t->join();

    spdlog::info("Tally server shutting down ...");

    return 0;
}