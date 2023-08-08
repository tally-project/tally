#ifndef TALLY_CONST_H
#define TALLY_CONST_H

#include <csignal>

extern bool TALLY_INITIALIZED;

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

static uint32_t PTB_MAX_NUM_THREADS_PER_SM = 1024;

// For kernel-wise profiling experiment
extern uint32_t PROFILE_DURATION;

void __attribute__((constructor)) register_env_vars();

#endif // TALLY_CONST_H