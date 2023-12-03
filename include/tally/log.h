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


#endif // TALLY_LOG_H