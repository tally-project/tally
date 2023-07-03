#ifndef TALLY_CONST_H
#define TALLY_CONST_H

static const char* LIBCUDART_PATH = "/usr/local/cuda/lib64/libcudart.so";
static const char* LIBCUDA_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
static const char* LIBCUDNN_PATH = "/usr/local/cuda/lib64/libcudnn.so";

extern uint32_t THREADS_PER_SLICE;
extern bool USE_CUDA_GRAPH;
extern uint32_t USE_TRANSFORM_THRESHOLD;
extern uint32_t TRANSFORM_THREADS_THRESHOLD;

#endif // TALLY_CONST_H