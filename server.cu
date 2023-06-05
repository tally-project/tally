#include <signal.h>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <stdlib.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>
#include <cassert>
#include <functional>
#include <unordered_map>
#include <cxxabi.h>
#include <map>

#include <cuda_runtime.h>
#include <cuda.h>

#include "libipc/ipc.h"

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

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

struct __align__(8) fatBinaryHeader
{
  unsigned int           magic;
  unsigned short         version;
  unsigned short         headerSize;
  unsigned long long int fatSize;
};

int magic;
int version;
unsigned long long* fatbin_data;
void **handle;

void addOneKernel(int * a, int b) {
    return;
}


std::atomic<bool> is_quit__ {false};
ipc::channel *ipc__ = nullptr;
std::map<std::string, void *> kernel_map;
std::vector<std::string> register_queue;

struct cudaMallocArg {
    void ** devPtr;
    size_t  size;
};

struct cudaMallocResponse {
    void * ptr;
    cudaError_t err;
};

struct cudaMemcpyResponse {
    cudaError_t err;
    char data[];
};

struct cudaMemcpyArg {
    void *dst;
    void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    char data[];
};

struct registerKernelArg {
    void *host_func;
    uint32_t kernel_func_len; 
    char data[]; // kernel_func_name
};

void handle_cudaMalloc(struct cudaMallocArg *arg)
{
    static cudaError_t (*lcudaMalloc) (void **, size_t );
	if (!lcudaMalloc) {
		lcudaMalloc = (cudaError_t (*) (void **, size_t )) dlsym(RTLD_NEXT, "cudaMalloc");
	}
	assert(lcudaMalloc);

    void *devPtr;
    cudaError_t err = cudaMalloc(&devPtr, arg->size);

    struct cudaMallocResponse res { devPtr, err };
    ipc__->send((void *) &res, sizeof(struct cudaMallocResponse), 1000 /* time out = 1000 ms*/);
}

void handle_cudaMemcpy(struct cudaMemcpyArg *arg)
{
    struct cudaMemcpyResponse *res;
    size_t res_size = 0;

    if (arg->kind == cudaMemcpyHostToDevice) {

        // Only care about dst (pointer to device memory) from the client call
        cudaError_t err = cudaMemcpy(arg->dst, arg->data, arg->count, arg->kind);

        res_size = sizeof(cudaError_t);
        res = (struct cudaMemcpyResponse *) malloc(res_size);
        res->err = err;
    } else if (arg->kind == cudaMemcpyDeviceToHost){
        res_size = sizeof(cudaError_t) + arg->count;
        res = (struct cudaMemcpyResponse *) malloc(res_size);

        // Only care about src (pointer to device memory) from the client call
        cudaError_t err = cudaMemcpy(res->data, arg->src, arg->count, arg->kind);

        res->err = err;
    } else {
        throw std::runtime_error("Unknown memcpy kind!");
    }

    ipc__->send((void *) res, res_size, 1000 /* time out = 1000 ms*/);
}

void handle_fatCubin(fatBinArg *arg)
{
    magic = arg->magic;
    version = arg->version;

    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) arg->data;
    uint32_t fatBinSize = fbh->headerSize + fbh->fatSize;

    fatbin_data = (unsigned long long *) malloc(fatBinSize);
    memcpy(fatbin_data, arg->data, fatBinSize);
}

void handle_register_kernel(struct registerKernelArg *arg)
{
    std::string __device_fun {arg->data, arg->kernel_func_len};
    register_queue.push_back(__device_fun);
}

void handle_fatCubin_end()
{
    static void (*l__cudaRegisterFatBinaryEnd) (void **);
	if (!l__cudaRegisterFatBinaryEnd) {
		l__cudaRegisterFatBinaryEnd = (void (*) (void **)) dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd");
	}
	assert(l__cudaRegisterFatBinaryEnd);
    static void** (*l__cudaRegisterFatBinary) (void *);
    if (!l__cudaRegisterFatBinary) {
        l__cudaRegisterFatBinary = (void** (*) (void *)) dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
    }
    assert(l__cudaRegisterFatBinary);
    static void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
    if (!l__cudaRegisterFunction) {
        l__cudaRegisterFunction = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    }
    assert(l__cudaRegisterFunction);

    const my__fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) = { magic, version, fatbin_data, 0 };
    
    handle = l__cudaRegisterFatBinary((void *)&__fatDeviceText);

    for (auto &func_name : register_queue) {
        void *func_addr = malloc(8);
        kernel_map[func_name] = func_addr;
        l__cudaRegisterFunction(handle, (const char*) func_addr, (char *)func_name.c_str(), func_name.c_str(), -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);
    }

    l__cudaRegisterFatBinaryEnd(handle);

    // For some reason, must call one cuda api call here. Otherwise it won't run.
    int *devArray;
    cudaMalloc((void**)&devArray, sizeof(int));
    cudaFree(devArray);
}

void do_recv(int interval) {

    int shm_fd = shm_open("shared_mem", O_RDWR | O_CREAT, 0666);

    // Allocate 100MB shared memory
    int size = 100 * 1024 * 1024;
    ftruncate(shm_fd, size);

    // Map address space to shared memory
    void *shm = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    ipc::channel ipc {"channel-1", ipc::receiver | ipc::sender};
    ipc__ = &ipc;

    while (!is_quit__.load(std::memory_order_acquire)) {
        ipc::buff_t buf;
        while (buf.empty()) {
            buf = ipc.recv(interval);
            if (is_quit__.load(std::memory_order_acquire)) return;
        }

        char const *dat = buf.get<char const *>();
        
        int func_name_length = *((int *)dat);
        std::string func_name(dat + 4, func_name_length);
        std::cout << func_name << std::endl;

        void *args = (void *) (dat + 4 + func_name_length);

        if (func_name == "cudaMalloc") {
            handle_cudaMalloc((cudaMallocArg *) args);
        } else if (func_name == "cudaMemcpy") {
            handle_cudaMemcpy((cudaMemcpyArg *) args);
        } else if (func_name == "__cudaRegisterFunction") {
            handle_register_kernel((registerKernelArg *) args);
        } else if (func_name == "__cudaRegisterFatBinary") {
            handle_fatCubin((fatBinArg *) args);
        } else if (func_name == "__cudaRegisterFatBinaryEnd") {
            handle_fatCubin_end();
        }
    }
}


int main(int argc, char ** argv) {

    auto _exit = [](int) {
        is_quit__.store(true, std::memory_order_release);
        if (ipc__ != nullptr) ipc__->disconnect();
        exit(0);
    };

    signal(SIGINT  , _exit);
    signal(SIGABRT , _exit);
    signal(SIGSEGV , _exit);
    signal(SIGTERM , _exit);
    signal(SIGHUP  , _exit);

    do_recv(1000);

    return 0;
}