

func_sig_must_contain = ["cu", "(", ")"]
func_sig_must_not_contain = ["noexcept", "{", "}", "return"]

ignore_keywords = [
    "\"C\"", "CUDARTAPI", "extern", "__host__", "__cudart_builtin__",
    "__attribute__((deprecated))"
]

preload_template = """
#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <cxxabi.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <map>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>

// g++ -I/usr/local/cuda/include -fPIC -shared -o preload.so preload.cpp

static void *cudnn_handle;
static std::vector<const void *> _kernel_seq;
static std::map<const void *, char*> _kernel_map;

void __attribute__((constructor)) preload_init()
{
    cudnn_handle = dlopen("/usr/local/cuda/lib64/libcudnn.so", RTLD_LAZY);
    assert(cudnn_handle);
}

char *demangleFunc(const char* mangledName)
{
    int status;
    return abi::__cxa_demangle(mangledName, nullptr, nullptr, &status);
}

"""

special_preload_funcs = {
    "__cudaRegisterFunction": """
void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    static void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
    if (!l__cudaRegisterFunction) {
        l__cudaRegisterFunction = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    }
    assert(l__cudaRegisterFunction);

    // store kernal names
    _kernel_map[(const void *)hostFun] = deviceFun;

    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}
    """,
    "cudaLaunchKernel": """
cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    static cudaError_t (*lcudaLaunchKernel) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t );
    if (!lcudaLaunchKernel) {
        lcudaLaunchKernel = (cudaError_t (*) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t )) dlsym(RTLD_NEXT, "cudaLaunchKernel");
    }
    assert(lcudaLaunchKernel);

    _kernel_seq.push_back(func);

    // if (_kernel_map.find((char *)func) != _kernel_map.end()) {
    //     char *demangled_name = demangleFunc(_kernel_map[(char *)func]);
    //     printf("%s\\n", demangled_name);
    //     free(demangled_name);
    // }

    return lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
    """
}