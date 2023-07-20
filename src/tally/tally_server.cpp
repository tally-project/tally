#include "tally/generated/server.h"

int main(int argc, char ** argv) {

    TallyServer::server->start(1000);

    return 0;
}