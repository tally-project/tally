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
#include <unistd.h>

#include "iceoryx_dust/posix_wrapper/signal_watcher.hpp"
#include "iceoryx_posh/popo/untyped_client.hpp"
#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "iox/detail/unique_id.hpp"

#include "tally/msg_struct.h"

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

    int32_t client_id;

    // For performance measurements
    std::vector<const void *> _profile_kernel_seq;
    std::vector<std::pair<time_point_t, time_point_t>> _profile_cpu_timestamps;
    std::map<const void *, std::string> _profile_kernel_map;

    std::map<const void *, std::string> host_func_to_demangled_kernel_name_map;
    std::map<std::string, std::vector<uint32_t>> _kernel_name_to_args;

    std::unordered_map<const void *, std::vector<uint32_t>> _kernel_addr_to_args;

#ifndef RUN_LOCALLY
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
        client_id = getpid();
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
        auto app_name_str_base = std::string("tally-client-app");
        auto app_name_str = app_name_str_base + std::to_string(client_id);

        char APP_NAME[100];
        strcpy(APP_NAME, app_name_str.c_str()); 

        iox::runtime::PoshRuntime::initRuntime(APP_NAME);

        iox::popo::UntypedClient client_handshake({"Tally", "handshake", "event"});

        // Send handshake to server
        client_handshake.loan(sizeof(HandshakeMessgae), alignof(HandshakeMessgae))
            .and_then([&](auto& requestPayload) {

                auto request = static_cast<HandshakeMessgae*>(requestPayload);
                request->client_id = client_id;

                client_handshake.send(request).or_else(
                    [&](auto& error) { std::cout << "Could not send Request! Error: " << error << std::endl; });
            })
            .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });

        while (!client_handshake.take().and_then([&](const auto& responsePayload) {

            auto response = static_cast<const HandshakeResponse*>(responsePayload);
            
            bool success = response->success;
            if (!success) {
                std::cout << "Handshake with tally server failed. Exiting ..." << std::endl;
                exit(1);
            }

            client_handshake.releaseResponse(responsePayload);

        })) {};

        auto channel_desc_str = std::string("Tally-Communication") + std::to_string(client_id);
        char channel_desc[100];
        strcpy(channel_desc, channel_desc_str.c_str()); 
        iox_client = new iox::popo::UntypedClient({channel_desc, "tally", "tally"});
#endif
    }

    ~TallyClient(){
        print_profile_trace();
    }
};

#endif // TALLY_CLIENT_H