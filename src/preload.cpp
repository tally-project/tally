
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
#include <cstring>
#include <fstream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <fatbinary_section.h>

#include "libipc/ipc.h"

// g++ -I/usr/local/cuda/include -fPIC -shared -o preload.so preload.cpp

class Preload {

public:

    std::map<const void *, std::string> _kernel_map;
    int shm_fd;
    void *shm;
    ipc::channel *ipc;

    Preload()
    {
        shm_fd = shm_open("shared_mem", O_RDWR | O_CREAT, 0666);

        // Allocate 100MB shared memory
        int size = 100 * 1024 * 1024;
        ftruncate(shm_fd, size);

        // Map address space to shared memory
        shm = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

        ipc = new ipc::channel("channel-1", ipc::sender | ipc::receiver);
    }

    ~Preload(){}
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

struct cudaMallocArg {
    void ** devPtr;
    size_t  size;
};

struct __align__(8) fatBinaryHeader
{
    unsigned int           magic;
    unsigned short         version;
    unsigned short         headerSize;
    unsigned long long int fatSize;
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

cudaError_t cudaMalloc(void ** devPtr, size_t  size)
{
    static const char *func_name = "cudaMalloc";
    static const size_t func_name_len = 10;
    size_t msg_len = sizeof(size_t) + func_name_len + sizeof(void **) + sizeof(size_t);

    void *msg = malloc(msg_len);
    *((int *) msg) = func_name_len;
    memcpy(msg + 4, func_name, func_name_len);
    
    struct cudaMallocArg *arg_ptr = (struct cudaMallocArg *)(msg + 4 + func_name_len);
    arg_ptr->devPtr = devPtr;
    arg_ptr->size = size;

    while (true) {
        bool success = tracer.ipc->send(msg, msg_len, 1000 /* time out = 1000 ms*/);
        if (success) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    ipc::buff_t buf;
    while (buf.empty()) {
        buf = tracer.ipc->recv(1000);
    }

    const char *dat = buf.get<const char *>();
    struct cudaMallocResponse *res = (struct cudaMallocResponse *) dat;

    *devPtr = res->ptr;
	return res->err;
}

cudaError_t cudaMemcpy(void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind)
{
    static const char *func_name = "cudaMemcpy";
    static const size_t func_name_len = 10;
    size_t msg_len;
    void *msg;
    
    if (kind == cudaMemcpyHostToDevice) {
        msg_len = sizeof(size_t) + func_name_len + sizeof(void *) + sizeof(const void *) + sizeof(size_t) + sizeof(enum cudaMemcpyKind) + count;
    } else if (kind == cudaMemcpyDeviceToHost){
        msg_len = sizeof(size_t) + func_name_len + sizeof(void *) + sizeof(const void *) + sizeof(size_t) + sizeof(enum cudaMemcpyKind);
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    msg = malloc(msg_len);
    *((int *) msg) = func_name_len;
    memcpy(msg + 4, func_name, func_name_len);
    
    struct cudaMemcpyArg *arg_ptr = (struct cudaMemcpyArg *)(msg + 4 + func_name_len);
    arg_ptr->dst = dst;
    arg_ptr->src = (void *)src;
    arg_ptr->count = count;
    arg_ptr->kind = kind;

    // Copy data to the message
    if (kind == cudaMemcpyHostToDevice) {
        memcpy(arg_ptr->data, src, count);
    }

    while (true) {
        bool success = tracer.ipc->send(msg, msg_len, 1000 /* time out = 1000 ms*/);
        if (success) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    ipc::buff_t buf;
    while (buf.empty()) {
        buf = tracer.ipc->recv(1000);
    }

    const char *dat = buf.get<const char *>();

    struct cudaMemcpyResponse {
        cudaError_t err;
        char data[];
    };

    struct cudaMemcpyResponse *res = (struct cudaMemcpyResponse *) dat;

    // Copy data to the host ptr
    if (kind == cudaMemcpyDeviceToHost) {
        memcpy(dst, res->data, count);
    }

	return res->err;
}

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    static cudaError_t (*lcudaLaunchKernel) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t );
    if (!lcudaLaunchKernel) {
        lcudaLaunchKernel = (cudaError_t (*) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t )) dlsym(RTLD_NEXT, "cudaLaunchKernel");
    }
    assert(lcudaLaunchKernel);

    cudaError_t err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    return err;
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    static const char *func_name = "__cudaRegisterFunction";
    static const size_t func_name_len = 22;

    struct registerKernelArg {
        void *host_func;
        uint32_t kernel_func_len; 
        char data[]; // kernel_func_name
    };

    std::string deviceFunName (deviceFun);
    uint32_t kernel_func_len = deviceFunName.size();

    size_t msg_len = sizeof(size_t) + func_name_len + sizeof(void *) + sizeof(uint32_t) + kernel_func_len * sizeof(char);

    void *msg = malloc(msg_len);
    *((int *) msg) = func_name_len;
    memcpy(msg + 4, func_name, func_name_len);

    struct registerKernelArg *arg_ptr = (struct registerKernelArg *)(msg + 4 + func_name_len);
    arg_ptr->host_func = (void*) hostFun;
    arg_ptr->kernel_func_len = kernel_func_len;
    memcpy(arg_ptr->data, deviceFun, kernel_func_len * sizeof(char));

    while (true) {
        bool success = tracer.ipc->send(msg, msg_len, 1000 /* time out = 1000 ms*/);
        if (success) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    static void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
    if (!l__cudaRegisterFunction) {
        l__cudaRegisterFunction = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    }
    assert(l__cudaRegisterFunction);

    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void** __cudaRegisterFatBinary( void *fatCubin ) {

    static const char *func_name = "__cudaRegisterFatBinary";
    static const size_t func_name_len = 23;

    __fatBinC_Wrapper_t *wp = (__fatBinC_Wrapper_t *) fatCubin;

    int magic = wp->magic;
    int version = wp->version;

    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) wp->data;
    size_t fatCubin_data_size_bytes = fbh->headerSize + fbh->fatSize;
    size_t msg_len = sizeof(size_t) + func_name_len + sizeof(int) + sizeof(int) + fatCubin_data_size_bytes;

    void *msg = malloc(msg_len);
    *((int *) msg) = func_name_len;
    memcpy(msg + 4, func_name, func_name_len);

    struct __cudaRegisterFatBinaryArg *arg_ptr = (struct __cudaRegisterFatBinaryArg *)(msg + 4 + func_name_len);
    arg_ptr->magic = magic;
    arg_ptr->version = version;
    memcpy(arg_ptr->data, wp->data, fatCubin_data_size_bytes);

    while (true) {
        bool success = tracer.ipc->send(msg, msg_len, 1000 /* time out = 1000 ms*/);
        if (success) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    static void** (*l__cudaRegisterFatBinary) (void *);
    if (!l__cudaRegisterFatBinary) {
        l__cudaRegisterFatBinary = (void** (*) (void *)) dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
    }
    assert(l__cudaRegisterFatBinary);

    return l__cudaRegisterFatBinary(fatCubin);
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
	static void (*l__cudaRegisterFatBinaryEnd) (void **);
	if (!l__cudaRegisterFatBinaryEnd) {
		l__cudaRegisterFatBinaryEnd = (void (*) (void **)) dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd");
	}
	assert(l__cudaRegisterFatBinaryEnd);

    static const char *func_name = "__cudaRegisterFatBinaryEnd";
    static const size_t func_name_len = 26;

    size_t msg_len = sizeof(size_t) + func_name_len;

    void *msg = malloc(msg_len);
    *((int *) msg) = func_name_len;
    memcpy(msg + 4, func_name, func_name_len);

    while (true) {
        bool success = tracer.ipc->send(msg, msg_len, 1000 /* time out = 1000 ms*/);
        if (success) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

	l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}
        

}