#ifndef TALLY_LOG_H
#define TALLY_LOG_H

#include <string>
#include <iostream>

#include "spdlog/spdlog.h"

#ifdef ENABLE_LOGGING
    #define TALLY_LOG(msg) \
        std::cout << msg << std::endl
#else
    #define TALLY_LOG(msg)
#endif

#define TALLY_SPD_WARN(msg) \
    spdlog::warn(msg);

#define TALLY_SPD_LOG_ALWAYS(msg) \
    spdlog::info(msg);

#ifdef ENABLE_LOGGING
    #define TALLY_SPD_LOG(msg) \
        spdlog::info(msg);
#else
    #define TALLY_SPD_LOG(msg)
#endif

#if defined(ENABLE_LOGGING) || defined(ENABLE_PERFORMANCE_LOGGING)
    #define TALLY_SPD_LOG_PROFILE(msg) \
        spdlog::info(msg);
    #define TALLY_LOG_PROFILE(msg) \
        std::cout << msg << std::endl
#else
    #define TALLY_SPD_LOG_PROFILE(msg)
    #define TALLY_LOG_PROFILE(msg)
#endif

#define CHECK_ERR_LOG_AND_EXIT(ERR, MSG) \
    if (ERR) { \
        std::cerr << MSG << " " << std::string(__FILE__) + ":" + std::to_string(__LINE__) << std::endl; \
        pid_t pid = getpid(); \
        int sig_num = SIGTERM; \
        kill(pid, sig_num); \
        signal_exit = true; \
    }
    
#ifdef ENABLE_PROFILING
    #define TALLY_CLIENT_PROFILE_START \
        auto __tally_call_start = std::chrono::high_resolution_clock::now();

    #define TALLY_CLIENT_PROFILE_END \
        auto __tally_call_end = std::chrono::high_resolution_clock::now(); \
        auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(__tally_call_start.time_since_epoch()).count();    \
        auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(__tally_call_end.time_since_epoch()).count();    \
        std::cout << "Duration: " << end_ns - start_ns << "ns" << std::endl;
#else
    #define TALLY_CLIENT_PROFILE_START
    #define TALLY_CLIENT_PROFILE_END
#endif


#endif // TALLY_LOG_H