#ifndef TALLY_ENV_H
#define TALLY_ENV_H

#include <csignal>
#include <string>

static const char* LIBCUDART_PATH = "/usr/local/cuda/lib64/libcudart.so";
static const char* LIBCUDA_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
static const char* LIBCUDNN_PATH = "/usr/local/cuda/lib/libcudnn.so";
static const char* LIBCUBLAS_PATH = "/usr/local/cuda/lib64/libcublas.so";
static const char* LIBCUBLASLT_PATH = "/usr/local/cuda/lib64/libcublasLt.so";
static const char* LIBNVRTC_PATH = "/usr/local/cuda/lib64/libnvrtc.so";
static const char* LIBCURAND_PATH = "/usr/local/cuda/lib64/libcurand.so";
static const char* LIBCUSPARSE_PATH = "/usr/local/cuda/lib64/libcusparse.so";

// These should be queried by CUDA API 
// But just set them here for now
static const char* CUDA_COMPUTE_VERSION = "80";
static uint32_t CUDA_NUM_SM = 82;
static uint32_t CUDA_MAX_NUM_THREADS_PER_SM = 1536;
static uint32_t CUDA_MAX_NUM_REGISTERS_PER_SM = 64 * 1024;
static uint32_t CUDA_MAX_SHM_BYTES_PER_SM = 100 * 1024;

static uint32_t FATBIN_MAGIC_NUMBER = 3126193488;

static float TIME_SHARE_THRESHOLD = 1.;

// number of times to run a kernel to get performance metrics
static uint32_t KERNEL_PROFILE_ITERATIONS = 5;

// Priority scheduler parameters
extern uint32_t PRIORITY_PTB_MAX_NUM_THREADS_PER_SM;
extern float PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS;
extern float PRIORITY_FALL_BACK_TO_KERNEL_SLICING_THRESHOLD;
extern float PRIORITY_FALL_BACK_TO_ORIGINAL_THRESHOLD;

// Sharing scheduler parameters
static uint32_t SHARING_PTB_MAX_NUM_THREADS_PER_SM = 1024;
static float SHARING_USE_PTB_THRESHOLD = 0.9f;

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