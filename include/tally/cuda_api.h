#ifndef TALLY_CUDA_API_H
#define TALLY_CUDA_API_H

#include <cuda.h>
#include <cuda_runtime.h>

extern void *cuda_handle;
extern void *cudart_handle;

extern CUresult (*lcuModuleLoadDataEx) (CUmodule *, const void *, unsigned int, CUjit_option *, void **);
extern CUresult (*lcuModuleGetFunction) (CUfunction*, CUmodule, const char*);
extern CUresult (*lcuLaunchKernel) (CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);

extern cudaError_t (*lcudaStreamCreate) ( cudaStream_t* pStream );
extern cudaError_t (*lcudaLaunchKernel) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t);

extern void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
extern void** (*l__cudaRegisterFatBinary) (void *);
extern void (*l__cudaRegisterFatBinaryEnd) (void **);

#endif // TALLY_CUDA_API_H