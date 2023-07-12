#ifndef TALLY_LOG_H
#define TALLY_LOG_H

#include <string>
#include <iostream>

// #undef ENABLE_LOGGING

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

#endif // TALLY_LOG_H