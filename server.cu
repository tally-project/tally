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
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda.h>

#include "libipc/ipc.h"

#include "tally/util.h"
#include "tally/def.h"

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

int magic;
int version;
unsigned long long* fatbin_data;
uint32_t fatBinSize;

std::atomic<bool> is_quit__ {false};
ipc::channel *send_ipc = nullptr;
ipc::channel *recv_ipc = nullptr;
std::map<void *, std::vector<uint32_t>> _kernel_addr_to_args;
std::map<std::string, void *> _kernel_name_to_addr;
std::map<void *, void *> _kernel_client_addr_mapping;
std::vector<std::pair<void *, std::string>> register_queue;

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
    while(!send_ipc->send((void *) &res, sizeof(struct cudaMallocResponse))) {
        send_ipc->wait_for_recv(1);
    }
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

    while(!send_ipc->send((void *) res, res_size)) {
        send_ipc->wait_for_recv(1);
    }
}

void handle_cudaLaunchKernel(cudaLaunchKernelArg *arg)
{
    static cudaError_t (*lcudaLaunchKernel) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t );
    if (!lcudaLaunchKernel) {
        lcudaLaunchKernel = (cudaError_t (*) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t )) dlsym(RTLD_NEXT, "cudaLaunchKernel");
    }
    assert(lcudaLaunchKernel);

    void *kernel_server_addr = _kernel_client_addr_mapping[(void *) arg->host_func];
    auto &arg_sizes = _kernel_addr_to_args[kernel_server_addr];
    auto argc = arg_sizes.size();

    void *__args_arr[argc];
    int __args_idx = 0;
    int offset = 0;

    for (size_t i = 0; i < argc; i++) {
        __args_arr[__args_idx] = (void *) (arg->params + offset);
        ++__args_idx;
        offset += arg_sizes[i];
    }

    auto err = lcudaLaunchKernel((const void *) kernel_server_addr, arg->gridDim, arg->blockDim, &__args_arr[0], arg->sharedMem, cudaStreamDefault);

    while (!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void handle_fatCubin(fatBinArg *arg)
{
    magic = arg->magic;
    version = arg->version;

    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) arg->data;
    fatBinSize = fbh->headerSize + fbh->fatSize;

    fatbin_data = (unsigned long long *) malloc(fatBinSize);
    memcpy(fatbin_data, arg->data, fatBinSize);
}

void handle_register_kernel(struct registerKernelArg *arg)
{
    std::string kernel_name {arg->data, arg->kernel_func_len};
    register_queue.push_back( std::make_pair(arg->host_func, kernel_name));
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
    
    void **handle = l__cudaRegisterFatBinary((void *)&__fatDeviceText);

    void *kernel_server_addr;

    for (auto &kernel_pair : register_queue) {
        auto &client_addr = kernel_pair.first;
        auto &kernel_name = kernel_pair.second;

        // allocate an address for the purpose
        kernel_server_addr = malloc(8);

        // Bookkeeping the mapping between clinet kernel addr and server kernel addr
        _kernel_name_to_addr[kernel_name] = kernel_server_addr;
        _kernel_client_addr_mapping[client_addr] = kernel_server_addr;
        l__cudaRegisterFunction(handle, (const char*) kernel_server_addr, (char *)kernel_name.c_str(), kernel_name.c_str(), -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0);
    }

    l__cudaRegisterFatBinaryEnd(handle);

    std::ofstream cubin_file("/tmp/tmp.cubin", std::ios::binary); // Open the file in binary mode
    cubin_file.write(reinterpret_cast<const char*>(fatbin_data), fatBinSize);
    cubin_file.close();

    const char* command = "cuobjdump /tmp/tmp.cubin -elf > /tmp/tmp_cubin.elf";
    system(command);

    std::string filename = "/tmp/tmp_cubin.elf";
    std::ifstream elf_file(filename);

    // key: func_name, val: [ <ordinal, size> ]
    using ordinal_size_pair = std::pair<uint32_t, uint32_t>;

    std::string line;
    while (std::getline(elf_file, line)) {
        if (startsWith(line, ".nv.info.")) {
            std::string kernel_name = line.substr(9);
            std::vector<ordinal_size_pair> params_info;

            while (std::getline(elf_file, line)) {
                if (containsSubstring(line, "EIATTR_KPARAM_INFO")) {
                    
                } else if (containsSubstring(line, "Ordinal :")) {
                    auto split_by_ordinal = splitOnce(line, "Ordinal :");
                    auto split_by_offset = splitOnce(split_by_ordinal.second, "Offset  :");
                    auto split_by_size = splitOnce(split_by_offset.second, "Size    :");

                    auto ordinal_str = strip(split_by_offset.first);
                    auto size_str = strip(split_by_size.second);

                    uint32_t arg_ordinal = std::stoi(ordinal_str, nullptr, 16);
                    uint32_t arg_size = std::stoi(size_str, nullptr, 16);

                    params_info.push_back(std::make_pair(arg_ordinal, arg_size));

                } else if (line.empty()) {
                    break;
                }
            }

            // Sort by ordinal
            std::sort(
                params_info.begin(),
                params_info.end(),
                [](ordinal_size_pair a, ordinal_size_pair b) {
                    return a.first < b.first;
                }
            );

            // Store the size
            for (auto &pair : params_info) {
                _kernel_addr_to_args[_kernel_name_to_addr[kernel_name]].push_back(pair.second);
            }
        }
    }    

    elf_file.close();

    // For some reason, must call one cuda api call here. Otherwise it won't run.
    int *arr;
    cudaMalloc((void**)&arr, sizeof(int));
    cudaFree(arr);
}

void do_recv(int interval) {
    send_ipc = new ipc::channel("server-to-client", ipc::sender);
    recv_ipc = new ipc::channel("client-to-server", ipc::receiver);

    while (!is_quit__.load(std::memory_order_acquire)) {
        ipc::buff_t buf;
        while (buf.empty()) {
            buf = recv_ipc->recv(interval);
            if (is_quit__.load(std::memory_order_acquire)) return;
        }

        char const *dat = buf.get<char const *>();
        
        int func_name_length = *((int *)dat);
        std::string func_name(dat + 4, func_name_length);
        std::cout << func_name << std::endl;

        void *arg_sizes = (void *) (dat + 4 + func_name_length);

        if (func_name == "cudaMalloc") {
            handle_cudaMalloc((cudaMallocArg *) arg_sizes);
        } else if (func_name == "cudaMemcpy") {
            handle_cudaMemcpy((cudaMemcpyArg *) arg_sizes);
        } else if (func_name == "cudaLaunchKernel") {
            handle_cudaLaunchKernel((cudaLaunchKernelArg *) arg_sizes);
        } else if (func_name == "__cudaRegisterFunction") {
            handle_register_kernel((registerKernelArg *) arg_sizes);
        } else if (func_name == "__cudaRegisterFatBinary") {
            handle_fatCubin((fatBinArg *) arg_sizes);
        } else if (func_name == "__cudaRegisterFatBinaryEnd") {
            handle_fatCubin_end();
        }
    }
}

int main(int argc, char ** argv) {

    auto _exit = [](int) {
        is_quit__.store(true, std::memory_order_release);
        if (send_ipc != nullptr) send_ipc->disconnect();
        if (recv_ipc != nullptr) recv_ipc->disconnect();
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