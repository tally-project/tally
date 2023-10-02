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

#ifdef ENABLE_LOGGING
    #define TALLY_SPD_LOG(msg) \
        spdlog::info(msg);
#else
    #define TALLY_SPD_LOG(msg)
#endif

#define CHECK_ERR_LOG_AND_EXIT(ERR, MSG) \
    if (ERR) { \
        std::cerr << MSG << " " << std::string(__FILE__) + ":" + std::to_string(__LINE__) << std::endl; \
        exit(1); \
    }


#endif // TALLY_LOG_H