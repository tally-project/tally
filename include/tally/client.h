#ifndef TALLY_CLIENT_H
#define TALLY_CLIENT_H

#include <signal.h>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <iostream>

#include "libipc/ipc.h"

static std::function<void(int)> __exit;

static void __exit_wrapper(int signal) {
    __exit(signal);
}

class TallyClient {

public:

    static std::unique_ptr<TallyClient> client;

    std::map<std::string, std::vector<uint32_t>> _kernel_name_to_args;
    std::map<const void *, std::vector<uint32_t>> _kernel_addr_to_args;
    std::map<const void *, std::string> _kernel_map;
    ipc::channel *send_ipc;
    ipc::channel *recv_ipc;

    TallyClient()
    {
        __exit = [&](int sig_num) {

            if (sig_num == SIGSEGV) {
                std::cout << "Encountered segfault. Shutting down... " << std::endl;
            }

            if (send_ipc != nullptr) send_ipc->disconnect();
            if (recv_ipc != nullptr) recv_ipc->disconnect();
            exit(0);
        };

        signal(SIGINT  , __exit_wrapper);
        signal(SIGABRT , __exit_wrapper);
        signal(SIGSEGV , __exit_wrapper);
        signal(SIGTERM , __exit_wrapper);
        signal(SIGHUP  , __exit_wrapper);

        send_ipc = new ipc::channel("client-to-server-230000", ipc::sender);
        recv_ipc = new ipc::channel("server-to-client-230000", ipc::receiver);
    }

    ~TallyClient(){}
};

#endif // TALLY_CLIENT_H