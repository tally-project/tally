#ifndef TALLY_CLIENT_H
#define TALLY_CLIENT_H

#include <map>
#include <string>
#include <vector>
#include <memory>

#include "libipc/ipc.h"

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
        send_ipc = new ipc::channel("client-to-server", ipc::sender);
        recv_ipc = new ipc::channel("server-to-client", ipc::receiver);
    }

    ~TallyClient()
    {
        if (send_ipc != nullptr) send_ipc->disconnect();
        if (recv_ipc != nullptr) recv_ipc->disconnect();
    }
};

#endif // TALLY_CLIENT_H