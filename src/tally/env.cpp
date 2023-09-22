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
            if (policy_str == "NAIVE") {
                SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::NAIVE; 
            } else if (policy_str == "PROFILE") {
                SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::PROFILE; 
            } else if (policy_str == "PRIORITY") {
                SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::PRIORITY; 
            } else if (policy_str == "WORKLOAD_AGNOSTIC_SHARING") {
                SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::WORKLOAD_AGNOSTIC_SHARING; 
            } else if (policy_str == "WORKLOAD_AWARE_SHARING") {
                SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::WORKLOAD_AWARE_SHARING; 
            } else {
                SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::NAIVE;
            }
        } else {
            SCHEDULER_POLICY = TALLY_SCHEDULER_POLICY::NAIVE;
        }

        TALLY_INITIALIZED = true;
    }
}