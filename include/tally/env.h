#ifndef TALLY_ENV_H
#define TALLY_ENV_H

#include <csignal>
#include <string>

extern float TIME_SHARE_THRESHOLD;

// number of times to run a kernel to get performance metrics
extern uint32_t KERNEL_PROFILE_ITERATIONS;

// Priority scheduler parameters
extern uint32_t PRIORITY_PTB_MAX_NUM_THREADS_PER_SM;
extern float PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS;
extern float PRIORITY_FALL_BACK_TO_ORIGINAL_THRESHOLD;
extern uint32_t PRIORITY_MIN_WORKER_BLOCKS;
extern float PRIORITY_PTB_PREEMPTION_LATENCY_CALCULATION_FACTOR;

// Sharing scheduler parameters
extern uint32_t SHARING_PTB_MAX_NUM_THREADS_PER_SM;
extern float SHARING_USE_PTB_THRESHOLD;

extern bool TALLY_INITIALIZED;

enum TALLY_SCHEDULER_POLICY {
    NAIVE,
    PROFILE,
    PRIORITY,
    WORKLOAD_AGNOSTIC_SHARING,
    WORKLOAD_AWARE_SHARING
};

extern TALLY_SCHEDULER_POLICY SCHEDULER_POLICY;

void __attribute__((constructor)) register_env_vars();
void set_max_allowed_preemption_latency(float latency_ms);

#endif // TALLY_ENV_H