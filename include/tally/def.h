#ifndef TALLY_DEF_H
#define TALLY_DEF_H

struct __align__(8) fatBinaryHeader
{
    unsigned int           magic;
    unsigned short         version;
    unsigned short         headerSize;
    unsigned long long int fatSize;
};

struct cudaMallocArg {
    void ** devPtr;
    size_t  size;
};

struct cudaMallocResponse {
    void *ptr;
    cudaError_t err;
};

struct cudaMemcpyArg {
    void *dst;
    void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    char data[];
};

struct cudaLaunchKernelArg {
    const void *host_func;
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    char params[];
};

struct cudaMemcpyResponse {
    cudaError_t err;
    char data[];
};

struct __cudaRegisterFatBinaryArg {
    int magic;
    int version;
    char data[];
};

struct registerKernelArg {
    void *host_func;
    uint32_t kernel_func_len; 
    char data[]; // kernel_func_name
};

typedef struct {
    int magic;
    int version;
    unsigned long long data[];
} fatBinArg;

typedef struct {
    int magic;
    int version;
    const unsigned long long* data;
    void *filename_or_fatbins;  /* version 1: offline filename,
                                * version 2: array of prelinked fatbins */
} my__fatBinC_Wrapper_t;

#endif // TALLY_DEF_H