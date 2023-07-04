#ifndef TALLY_CONST_H
#define TALLY_CONST_H

#include <csignal>

extern bool TALLY_INITIALIZED;

static const char* LIBCUDART_PATH = "/usr/local/cuda/lib64/libcudart.so";
static const char* LIBCUDA_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
static const char* LIBCUDNN_PATH = "/usr/local/cuda/lib64/libcudnn.so";

extern uint32_t THREADS_PER_SLICE;
extern bool USE_CUDA_GRAPH;
extern uint32_t USE_TRANSFORM_THRESHOLD;
extern uint32_t TRANSFORM_THREADS_THRESHOLD;

// For kernel-to-kernel profiling purposes
extern bool PROFILE_KERNEL_TO_KERNEL_PERF;
extern uint32_t PROFILE_KERNEL_IDX;
extern uint32_t PROFILE_DURATION_SECONDS;
extern bool PROFILE_WARMED_UP; // Set this to true when warmed up

// Sepcify which config to profile
extern bool PROFILE_USE_ORIGINAL;
extern bool PROFILE_USE_SLICED;
extern bool PROFILE_USE_PTB;
extern bool PROFILE_USE_CUDA_GRAPH;
extern uint32_t PROFILE_THREADS_PER_SLICE;
extern uint32_t PROFILE_NUM_BLOCKS_PER_SM;

void __attribute__((constructor)) register_env_vars();

#endif // TALLY_CONST_H