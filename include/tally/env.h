#ifndef TALLY_ENV_H
#define TALLY_ENV_H

#include <csignal>
#include <string>

static const char* LIBCUDART_PATH = "/usr/local/cuda/lib64/libcudart.so";
static const char* LIBCUDA_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
static const char* LIBCUDNN_PATH = "/usr/local/cuda/lib64/libcudnn.so";
static const char* LIBCUBLAS_PATH = "/usr/local/cuda/lib64/libcublas.so";
static const char* LIBCUBLASLT_PATH = "/usr/local/cuda/lib64/libcublasLt.so";

// These should be queried by CUDA API 
// But just set them here for now
static const char* CUDA_COMPUTE_VERSION = "86";
static uint32_t CUDA_NUM_SM = 82;
static uint32_t CUDA_MAX_NUM_THREADS_PER_SM = 1536;
static uint32_t CUDA_MAX_NUM_REGISTERS_PER_SM = 64 * 1024;
static uint32_t CUDA_MAX_SHM_BYTES_PER_SM = 100 * 1024;

static uint32_t FATBIN_MAGIC_NUMBER = 3126193488;

// These can be tuned
static uint32_t PTB_MAX_NUM_THREADS_PER_SM = 1024;

// Time share Threshold
static uint32_t TIME_SHARE_THRESHOLD = 1;
static uint32_t USE_PTB_THRESHOLD = 0.5;
static uint32_t USE_PREEMPTIVE_LATENCY_THRESHOLD = 0.5;

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

#endif // TALLY_ENV_H