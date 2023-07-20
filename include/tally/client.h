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

#include "iceoryx_dust/posix_wrapper/signal_watcher.hpp"
#include "iceoryx_posh/popo/untyped_client.hpp"
#include "iceoryx_posh/runtime/posh_runtime.hpp"

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

    static TallyClient *client;

    // For performance measurements
    std::vector<const void *> _profile_kernel_seq;
    std::vector<std::pair<time_point_t, time_point_t>> _profile_cpu_timestamps;
    std::map<const void *, std::string> _profile_kernel_map;

    std::map<const void *, std::string> host_func_to_demangled_kernel_name_map;
    std::map<std::string, std::vector<uint32_t>> _kernel_name_to_args;

    std::unordered_map<const void *, std::vector<uint32_t>> _kernel_addr_to_args;

#ifndef RUN_LOCALLY
    static constexpr char APP_NAME[] = "iox-cpp-request-response-client-untyped";
    iox::popo::UntypedClient *iox_client;
#endif

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
        register_profile_kernel_map();

        __exit = [&](int sig_num) {

            if (sig_num == SIGSEGV) {
                std::cout << "Encountered segfault. Shutting down... " << std::endl;
            }
            exit(0);
        };

        signal(SIGINT  , __exit_wrapper);
        signal(SIGABRT , __exit_wrapper);
        signal(SIGSEGV , __exit_wrapper);
        signal(SIGTERM , __exit_wrapper);
        signal(SIGHUP  , __exit_wrapper);

#ifndef RUN_LOCALLY
        iox::runtime::PoshRuntime::initRuntime(APP_NAME);
        iox_client = new iox::popo::UntypedClient({"Example", "Request-Response", "Add"});
#endif
    }

    ~TallyClient(){
        print_profile_trace();
    }
};

#endif // TALLY_CLIENT_H