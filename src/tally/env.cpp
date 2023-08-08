#include <string>
#include <iostream>

#include <tally/env.h>

#define REGISTER_UINT32_ENV_VAR(NAME, DEFAULT_VAL) \
    NAME = std::getenv(#NAME) ? std::stoi(std::getenv(#NAME)) : DEFAULT_VAL;

#define REGISTER_BOOL_ENV_VAR(NAME, DEFAULT_VAL) \
    NAME = std::getenv(#NAME) ? (std::string(std::getenv(#NAME)) == "TRUE") : DEFAULT_VAL;

bool TALLY_INITIALIZED = false;

uint32_t PROFILE_DURATION;

void __attribute__((constructor)) register_env_vars()
{
    if (!TALLY_INITIALIZED) {

        REGISTER_UINT32_ENV_VAR(PROFILE_DURATION, 10);

        TALLY_INITIALIZED = true;
    }
}