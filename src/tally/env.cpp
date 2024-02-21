#include <string>
#include <iostream>

#include <cuda_runtime.h>

#include <tally/env.h>

#define REGISTER_UINT32_ENV_VAR(NAME, DEFAULT_VAL) \
    NAME = std::getenv(#NAME) ? std::stoi(std::getenv(#NAME)) : DEFAULT_VAL;

#define REGISTER_FLOAT_ENV_VAR(NAME, DEFAULT_VAL) \
    NAME = std::getenv(#NAME) ? std::stof(std::getenv(#NAME)) : DEFAULT_VAL;

#define REGISTER_BOOL_ENV_VAR(NAME, DEFAULT_VAL) \
    NAME = std::getenv(#NAME) ? (std::string(std::getenv(#NAME)) == "TRUE") : DEFAULT_VAL;

bool TALLY_INITIALIZED = false;
TALLY_SCHEDULER_POLICY SCHEDULER_POLICY;

float TIME_SHARE_THRESHOLD = 1.;

// number of times to run a kernel to get performance metrics
uint32_t KERNEL_PROFILE_ITERATIONS;

uint32_t PRIORITY_PTB_MAX_NUM_THREADS_PER_SM;
uint32_t PRIORITY_MIN_WORKER_BLOCKS;
float PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS;
float PRIORITY_FALL_BACK_TO_ORIGINAL_THRESHOLD;
float PRIORITY_PTB_PREEMPTION_LATENCY_CALCULATION_FACTOR;

uint32_t SHARING_PTB_MAX_NUM_THREADS_PER_SM;
float SHARING_USE_PTB_THRESHOLD;

// Later it should be adjustable at runtime
void set_max_allowed_preemption_latency(float latency_ms)
{
    PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS = latency_ms;
}

void __attribute__((constructor)) register_env_vars()
{
    if (!TALLY_INITIALIZED) {

        // Init scheduler policy
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

        // init max allowed preemption latency for priority scheduler
        REGISTER_FLOAT_ENV_VAR(PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS, 0.1f);
        REGISTER_FLOAT_ENV_VAR(PRIORITY_FALL_BACK_TO_ORIGINAL_THRESHOLD, 0.1f);
        REGISTER_FLOAT_ENV_VAR(SHARING_USE_PTB_THRESHOLD, 0.9f);
        REGISTER_FLOAT_ENV_VAR(PRIORITY_PTB_PREEMPTION_LATENCY_CALCULATION_FACTOR, 2.f);

        REGISTER_UINT32_ENV_VAR(KERNEL_PROFILE_ITERATIONS, 5);
        REGISTER_UINT32_ENV_VAR(PRIORITY_PTB_MAX_NUM_THREADS_PER_SM, 1024);
        REGISTER_UINT32_ENV_VAR(PRIORITY_MIN_WORKER_BLOCKS, 24);
        REGISTER_UINT32_ENV_VAR(SHARING_PTB_MAX_NUM_THREADS_PER_SM, 1024);

        TALLY_INITIALIZED = true;
    }
}