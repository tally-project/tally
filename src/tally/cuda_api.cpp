
#include <dlfcn.h>

#include <cuda.h>

#include <tally/cuda_api.h>
#include <tally/const.h>

void *cuda_handle = dlopen(LIBCUDA_PATH, RTLD_LAZY);
void *cudart_handle = dlopen(LIBCUDART_PATH, RTLD_LAZY);

CUresult (*lcuModuleLoadDataEx) (CUmodule *, const void *, unsigned int, CUjit_option *, void **) =
    (CUresult (*) (CUmodule *, const void *, unsigned int, CUjit_option *, void **)) dlsym(cuda_handle, "cuModuleLoadDataEx");

CUresult (*lcuModuleGetFunction) (CUfunction*, CUmodule, const char*) = 
    (CUresult (*) (CUfunction*, CUmodule, const char*)) dlsym(cuda_handle, "cuModuleGetFunction");

CUresult (*lcuLaunchKernel) (CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) =
    (CUresult (*) (CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**)) dlsym(cuda_handle, "cuLaunchKernel");

cudaError_t (*lcudaLaunchKernel) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t ) = 
    (cudaError_t (*) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t )) dlsym(cudart_handle, "cudaLaunchKernel");


void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)
    = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(cudart_handle, "__cudaRegisterFunction");

void** (*l__cudaRegisterFatBinary) (void *) = 
    (void** (*) (void *)) dlsym(cudart_handle, "__cudaRegisterFatBinary");

void (*l__cudaRegisterFatBinaryEnd) (void **) =
	(void (*) (void **)) dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd");