#ifndef TALLY_CLIENT_H
#define TALLY_CLIENT_H

#include <signal.h>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <functional>
#include <iostream>
#include <cassert>
#include <sstream>

#include "libipc/ipc.h"

#ifdef ENABLE_PROFILING
    #define TALLY_CLIENT_PROFILE_START \
        auto __tally_call_start = std::chrono::high_resolution_clock::now();

    #define TALLY_CLIENT_PROFILE_END \
        auto __tally_call_end = std::chrono::high_resolution_clock::now(); \
        TallyClient::client->_profile_cpu_timestamps.push_back({ __tally_call_start, __tally_call_end });

    #define TALLY_CLIENT_TRACE_API_CALL(CLIENT_API_CALL) \
        TallyClient::client->_profile_kernel_seq.push_back((void *) l##CLIENT_API_CALL);

    #define TALLY_CLIENT_TRACE_KERNEL_CALL(FUNC) \
        TallyClient::client->_profile_kernel_seq.push_back((void *) FUNC);
#else
    #define TALLY_CLIENT_PROFILE_START
    #define TALLY_CLIENT_PROFILE_END
    #define TALLY_CLIENT_TRACE_API_CALL(CLIENT_API_CALL)
    #define TALLY_CLIENT_TRACE_KERNEL_CALL(FUNC)
#endif

static std::function<void(int)> __exit;

static void __exit_wrapper(int signal) {
    __exit(signal);
}

class TallyClient {

typedef std::chrono::time_point<std::chrono::system_clock> time_point_t;

public:

    static std::unique_ptr<TallyClient> client;

    // For performance measurements
    std::vector<const void *> _profile_kernel_seq;
    std::vector<std::pair<time_point_t, time_point_t>> _profile_cpu_timestamps;
    std::map<const void *, std::string> _profile_kernel_map;

    std::map<const void *, std::string> host_func_to_demangled_kernel_name_map;
    std::map<std::string, std::vector<uint32_t>> _kernel_name_to_args;
    std::map<const void *, std::vector<uint32_t>> _kernel_addr_to_args;
    ipc::channel *send_ipc;
    ipc::channel *recv_ipc;

    const static size_t msg_size = 1024 * 1024 * 1024;
    uint8_t *msg;

    void register_profile_kernel_map();
    void print_profile_trace()
    {
        assert(_profile_cpu_timestamps.size() == _profile_kernel_seq.size());
        for (size_t i = 0; i < _profile_kernel_seq.size(); i++) {
            auto _trace_addr = _profile_kernel_seq[i];
            std::string _trace_name;
            if (_profile_kernel_map.find(_trace_addr) != _profile_kernel_map.end()) {
                _trace_name = _profile_kernel_map[_trace_addr];
            } else if (host_func_to_demangled_kernel_name_map.find(_trace_addr) != host_func_to_demangled_kernel_name_map.end()) {
                _trace_name = host_func_to_demangled_kernel_name_map[_trace_addr];
            } else {
                std::cerr << "Cannot find _trace_addr in _profile_kernel_map" << std::endl;
                continue;
            }

            std::ostringstream stream;

            stream << _trace_name;
            auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    _profile_cpu_timestamps[i].first.time_since_epoch()).count();
            auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    _profile_cpu_timestamps[i].second.time_since_epoch()).count();

            stream << " Duration: " << end_ns - start_ns << "ns";

            std::cout << stream.str() << std::endl;
        }
    }

    TallyClient()
    {
        // Allocate 1GB memory for message passing 
        msg = (uint8_t *) malloc(msg_size);
        register_profile_kernel_map();

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

        send_ipc = new ipc::channel("client-to-server-350000", ipc::sender);
        recv_ipc = new ipc::channel("server-to-client-350000", ipc::receiver);
    }

    ~TallyClient(){
        free(msg);
        print_profile_trace();
    }
};

#endif // TALLY_CLIENT_H