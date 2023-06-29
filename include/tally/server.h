#ifndef TALLY_SERVER_H
#define TALLY_SERVER_H

#include <signal.h>
#include <string>
#include <atomic>
#include <map>
#include <iostream>
#include <functional>

#include <cuda_runtime.h>
#include <cuda.h>

#include "libipc/ipc.h"

#include <tally/msg_struct.h>

static std::function<void(int)> __exit;

static void __exit_wrapper(int signal) {
    __exit(signal);
}

class TallyServer {

public:

    static TallyServer *server;

    int magic;
    int version;
    unsigned long long* fatbin_data;
    uint32_t fatBinSize;

    std::atomic<bool> is_quit__ {false};
    ipc::channel *send_ipc = nullptr;
    ipc::channel *recv_ipc = nullptr;
    std::map<void *, std::vector<uint32_t>> _kernel_addr_to_args;
    std::map<std::string, void *> _kernel_name_to_addr;
    std::map<void *, void *> _kernel_client_addr_mapping;
    std::vector<std::pair<void *, std::string>> register_queue;
    std::unordered_map<CUDA_API_ENUM, std::function<void(void *)>> cuda_api_handler_map;

    TallyServer()
    {
        __exit = [&](int) {
            is_quit__.store(true, std::memory_order_release);
            if (send_ipc != nullptr) send_ipc->disconnect();
            if (recv_ipc != nullptr) recv_ipc->disconnect();
            exit(0);
        };

        signal(SIGINT  , __exit_wrapper);
        signal(SIGABRT , __exit_wrapper);
        signal(SIGSEGV , __exit_wrapper);
        signal(SIGTERM , __exit_wrapper);
        signal(SIGHUP  , __exit_wrapper);

        cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOC] = std::bind(&TallyServer::handle_cudaMalloc, this, std::placeholders::_1);
        cuda_api_handler_map[CUDA_API_ENUM::CUDAFREE] = std::bind(&TallyServer::handle_cudaFree, this, std::placeholders::_1);
        cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY] = std::bind(&TallyServer::handle_cudaMemcpy, this, std::placeholders::_1);
        cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHKERNEL] = std::bind(&TallyServer::handle_cudaLaunchKernel, this, std::placeholders::_1);
        cuda_api_handler_map[CUDA_API_ENUM::__CUDAREGISTERFUNCTION] = std::bind(&TallyServer::handle___cudaRegisterFunction, this, std::placeholders::_1);
        cuda_api_handler_map[CUDA_API_ENUM::__CUDAREGISTERFATBINARY] = std::bind(&TallyServer::handle___cudaRegisterFatBinary, this, std::placeholders::_1);
        cuda_api_handler_map[CUDA_API_ENUM::__CUDAREGISTERFATBINARYEND] = std::bind(&TallyServer::handle___cudaRegisterFatBinaryEnd, this, std::placeholders::_1);
    }

    void start(uint32_t interval) {
        send_ipc = new ipc::channel("server-to-client", ipc::sender);
        recv_ipc = new ipc::channel("client-to-server", ipc::receiver);

        while (!is_quit__.load(std::memory_order_acquire)) {
            ipc::buff_t buf;
            while (buf.empty()) {
                buf = recv_ipc->recv(interval);
                if (is_quit__.load(std::memory_order_acquire)) return;
            }

            char const *dat = buf.get<char const *>();
            MessageHeader_t *msg_header = (MessageHeader_t *) dat;
            void *args = (void *) (dat + sizeof(CUDA_API_ENUM));

            auto handler = cuda_api_handler_map[msg_header->api_id];
            handler(args);
        }
    }

    void handle_cudaMalloc(void *args);
    void handle_cudaFree(void *args);
    void handle_cudaMemcpy(void *args);
    void handle_cudaLaunchKernel(void *args);

    void handle___cudaRegisterFatBinary(void *args);
    void handle___cudaRegisterFunction(void *args);
    void handle___cudaRegisterFatBinaryEnd(void *args);
};

#endif // TALLY_SERVER_H