#include <cassert>
#include <cfloat>
#include <random>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_util.h>
#include <tally/util.h>
#include <tally/env.h>

void TallyServer::start_scheduler()
{
    implicit_init_cuda_ctx();

    auto policy = SCHEDULER_POLICY;

    if (policy == TALLY_SCHEDULER_POLICY::NAIVE) {
        run_naive_scheduler();
    } else if (policy == TALLY_SCHEDULER_POLICY::PROFILE) {
        run_profile_scheduler();
    } else if (policy == TALLY_SCHEDULER_POLICY::PRIORITY) {
        run_priority_scheduler();
    } else if (policy == TALLY_SCHEDULER_POLICY::WORKLOAD_AGNOSTIC_SHARING) {
        run_workload_agnostic_sharing_scheduler();
    } else if (policy == TALLY_SCHEDULER_POLICY::WORKLOAD_AWARE_SHARING) {
        run_workload_aware_sharing_scheduler();
    } else {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unknown policy enum.");
    }
}