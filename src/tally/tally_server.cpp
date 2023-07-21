#include <thread>

#include "tally/generated/server.h"

int main(int argc, char ** argv) {

    std::thread server_t(&TallyServer::start, TallyServer::server);
    std::thread server_launcher_t(&TallyServer::start_launcher, TallyServer::server);

    server_t.join();
    server_launcher_t.join();

    return 0;
}