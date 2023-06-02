#include <signal.h>

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <stdlib.h>

#include "libipc/ipc.h"

// nvcc -I/usr/local/cuda/include -I/home/zhaowe58/exp/cpp-ipc/include -L/home/zhaowe58/exp/cpp-ipc/build/lib -lipc server.cu -o server

namespace {

std::atomic<bool> is_quit__ {false};
ipc::channel *ipc__ = nullptr;

void do_recv(int interval) {
    ipc::channel ipc {"ipc", ipc::receiver};
    ipc__ = &ipc;
    while (!is_quit__.load(std::memory_order_acquire)) {
        ipc::buff_t recv;
        for (int k = 1; recv.empty(); ++k) {
            std::cout << "recv waiting... " << k << "\n";
            recv = ipc.recv(interval);
            if (is_quit__.load(std::memory_order_acquire)) return;
        }
        std::cout << "recv size: " << recv.size() << "\n";
    }
}

} // namespace

int main(int argc, char ** argv) {

    auto _exit = [](int) {
        is_quit__.store(true, std::memory_order_release);
        if (ipc__ != nullptr) ipc__->disconnect();
        exit(0);
    };

    ::signal(SIGINT  , _exit);
    ::signal(SIGABRT , _exit);
    ::signal(SIGSEGV , _exit);
    ::signal(SIGTERM , _exit);
    ::signal(SIGHUP  , _exit);

    do_recv(1000);

    return 0;
}