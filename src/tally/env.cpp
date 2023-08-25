#include <string>
#include <iostream>

#include <tally/env.h>

#define REGISTER_UINT32_ENV_VAR(NAME, DEFAULT_VAL) \
    NAME = std::getenv(#NAME) ? std::stoi(std::getenv(#NAME)) : DEFAULT_VAL;

#define REGISTER_BOOL_ENV_VAR(NAME, DEFAULT_VAL) \
    NAME = std::getenv(#NAME) ? (std::string(std::getenv(#NAME)) == "TRUE") : DEFAULT_VAL;

bool TALLY_INITIALIZED = false;

TALLY_SCHEDULER_POLICY SCHEDULER_POLICY;

void __attribute__((constructor)) register_env_vars()
{
    if (!TALLY_INITIALIZED) {

        if (std::getenv("SCHEDULER_POLICY")) {
            auto policy_str = std::string(std::getenv("SCHEDULER_POLICY"));
            if (policy_str == "naive") {
                SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::NAIVE; 
            } else if (policy_str == "profile") {
                SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::PROFILE; 
            } else {
                SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::NAIVE;
            }
        } else {
            SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::NAIVE;
        }

        TALLY_INITIALIZED = true;
    }
}