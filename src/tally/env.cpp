#include <string>
#include <iostream>

#include <tally/env.h>

#define REGISTER_UINT32_ENV_VAR(NAME, DEFAULT_VAL) \
    NAME = std::getenv(#NAME) ? std::stoi(std::getenv(#NAME)) : DEFAULT_VAL;

#define REGISTER_BOOL_ENV_VAR(NAME, DEFAULT_VAL) \
    NAME = std::getenv(#NAME) ? (std::string(std::getenv(#NAME)) == "TRUE") : DEFAULT_VAL;

bool TALLY_INITIALIZED = false;

TALLY_SCHEDULER_POLICY SCHEDULER_POLICY;

uint32_t PRIORITY_PTB_MAX_NUM_THREADS_PER_SM = 1024;
float PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS = 0.1f;
float PRIORITY_FALL_BACK_TO_KERNEL_SLICING_THRESHOLD = 0.3f;
float PRIORITY_FALL_BACK_TO_ORIGINAL_THRESHOLD = 0.1f;

void set_max_allowed_preemption_latency(float latency_ms)
{
    PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS = latency_ms;
}

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

        if (std::getenv("PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS")) {
            auto max_latency_ms = std::stof(std::getenv("PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS"));
            set_max_allowed_preemption_latency(max_latency_ms);
        }

        TALLY_INITIALIZED = true;
    }
}