
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
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda.h>
#include <fatbinary_section.h>

#include "libipc/ipc.h"

// g++ -I/usr/local/cuda/include -fPIC -shared -o preload.so preload.cpp

class Preload {

public:

    std::map<std::string, std::vector<uint32_t>> _kernel_name_to_args;
    std::map<const void *, std::vector<uint32_t>> _kernel_addr_to_args;
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

        ipc = new ipc::channel("chan", ipc::sender | ipc::receiver);
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

bool startsWith(const std::string& str, const std::string& prefix) {
    return str.compare(0, prefix.length(), prefix) == 0;
}

bool containsSubstring(const std::string& str, const std::string& substring) {
    return str.find(substring) != std::string::npos;
}

std::pair<std::string, std::string> splitOnce(const std::string& str, const std::string& delimiter) {
    std::size_t pos = str.find(delimiter);
    if (pos != std::string::npos) {
        std::string first = str.substr(0, pos);
        std::string second = str.substr(pos + delimiter.length());
        return {first, second};
    }
    return {str, ""};
}

std::string strip(const std::string& str) {
    std::string result = str;
    
    // Remove leading whitespace
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    
    // Remove trailing whitespace
    result.erase(std::find_if(result.rbegin(), result.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), result.end());

    return result;
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

    struct cudaMemcpyResponse *res = (struct cudaMemcpyResponse *) dat;

    // Copy data to the host ptr
    if (kind == cudaMemcpyDeviceToHost) {
        memcpy(dst, res->data, count);
    }

	return res->err;
}

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    auto &params_info = tracer._kernel_addr_to_args[func];
    uint32_t params_size =  std::accumulate(params_info.begin(), params_info.end(), 0);

    size_t offset = 0;
    char params_data[params_size];

    for (size_t i = 0; i < params_info.size(); i++) {
        memcpy(params_data + offset, args[i], params_info[i]);
        offset += params_info[i];
    }

    static const char *func_name = "cudaLaunchKernel";
    static const size_t func_name_len = 16;
    size_t msg_len = sizeof(size_t) + func_name_len + sizeof(void *) + sizeof(dim3) + sizeof(dim3) + sizeof(size_t) + params_size;
    void *msg;

    msg = malloc(msg_len);
    *((int *) msg) = func_name_len;
    memcpy(msg + 4, func_name, func_name_len);

    struct cudaLaunchKernelArg *arg_ptr = (struct cudaLaunchKernelArg *)(msg + 4 + func_name_len);

    arg_ptr->host_func = func;
    arg_ptr->gridDim = gridDim;
    arg_ptr->blockDim = blockDim;
    arg_ptr->sharedMem = sharedMem;
    memcpy(arg_ptr->params, params_data, params_size);

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
    cudaError_t *err = (cudaError_t *) dat;

    return *err;
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    static const char *func_name = "__cudaRegisterFunction";
    static const size_t func_name_len = 22;

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

    tracer._kernel_addr_to_args[hostFun] = tracer._kernel_name_to_args[deviceFunName];

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

    std::ofstream cubin_file("/tmp/tmp.cubin", std::ios::binary); // Open the file in binary mode
    cubin_file.write(reinterpret_cast<const char*>(wp->data), fatCubin_data_size_bytes);
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
                tracer._kernel_name_to_args[kernel_name].push_back(pair.second);
            }
        }
    }    

    elf_file.close();

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