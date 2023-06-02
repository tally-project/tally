
#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <cxxabi.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>
#include <thread>

#include <cuda_runtime.h>
#include <cuda.h>

#include "libipc/ipc.h"

// g++ -I/usr/local/cuda/include -fPIC -shared -o preload.so preload.cpp

typedef std::chrono::time_point<std::chrono::system_clock> time_point_t;

class Preload {

public:

    std::map<const void *, std::string> _kernel_map;
    int shm_fd;
    void *shm;
    ipc::channel *ipc;

    Preload()
    {
        printf("Initialize\n");
        shm_fd = shm_open("shared_mem", O_RDWR | O_CREAT, 0666);

        // Allocate 100MB shared memory
        int size = 100 * 1024 * 1024;
        ftruncate(shm_fd, size);

        // Map address space to shared memory
        shm = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

        ipc = new ipc::channel("channel", ipc::sender);

        std::string test_str("string");
        while (true) {
            
            bool suc = ipc->send(test_str, 1000 /* time out = 1000 ms*/);
            if (suc) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    ~Preload()
    {
       printf("Post processing\n");
    }
};

std::string demangleFunc(std::string mangledName)
{
    int status;
    char *demangled_name = abi::__cxa_demangle(mangledName.c_str(), nullptr, nullptr, &status);
    
    if (status == 0) {
        std::string demangled_name_str(demangled_name);
        free(demangled_name);
        return demangled_name_str;
    } else {
        return mangledName;
    }
}

Preload tracer;

extern "C" { 


cudaError_t cudaMalloc(void ** devPtr, size_t  size)
{
	static cudaError_t (*lcudaMalloc) (void **, size_t );
	if (!lcudaMalloc) {
		lcudaMalloc = (cudaError_t (*) (void **, size_t )) dlsym(RTLD_NEXT, "cudaMalloc");
		tracer._kernel_map[(void *) lcudaMalloc] = std::string("cudaMalloc");
	}
	assert(lcudaMalloc);

	cudaError_t res = lcudaMalloc(devPtr, size);

    std::string _kernel_name = tracer._kernel_map[(void *) lcudaMalloc];
    std::cout << _kernel_name << std::endl;

	return res;
}

cudaError_t cudaMemcpy(void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind)
{
	static cudaError_t (*lcudaMemcpy) (void *, const void *, size_t , enum cudaMemcpyKind );
	if (!lcudaMemcpy) {
		lcudaMemcpy = (cudaError_t (*) (void *, const void *, size_t , enum cudaMemcpyKind )) dlsym(RTLD_NEXT, "cudaMemcpy");
		tracer._kernel_map[(void *) lcudaMemcpy] = std::string("cudaMemcpy");
	}
	assert(lcudaMemcpy);

	cudaError_t res = lcudaMemcpy(dst, src, count, kind);

    std::string _kernel_name = tracer._kernel_map[(void *) lcudaMemcpy];
    std::cout << _kernel_name << std::endl;

	return res;
}

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    static cudaError_t (*lcudaLaunchKernel) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t );
    if (!lcudaLaunchKernel) {
        lcudaLaunchKernel = (cudaError_t (*) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t )) dlsym(RTLD_NEXT, "cudaLaunchKernel");
    }
    assert(lcudaLaunchKernel);

    std::string _kernel_name = tracer._kernel_map[func];
    std::cout << _kernel_name << std::endl;

    cudaError_t err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    return err;
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    static void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
    if (!l__cudaRegisterFunction) {
        l__cudaRegisterFunction = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    }
    assert(l__cudaRegisterFunction);

    // store kernal names
    std::string deviceFun_str(deviceFun);
    tracer._kernel_map[(const void *)hostFun] = demangleFunc(deviceFun_str);

    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}
        

}