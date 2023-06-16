
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

#include <util.h>
#include <def.h>

class Preload {

public:

    std::map<std::string, const void *> kernel_name_map;
    std::map<const void *, CUfunction> kernel_map;
    std::vector<std::string> sliced_ptx_files;
    void *cuda_handle;
    bool registered = false;

    void register_sliced_kernels()
    {
        static CUresult (*lcuModuleLoadDataEx) (CUmodule *, const void *, unsigned int, CUjit_option *, void **);
        if (!lcuModuleLoadDataEx) {
            lcuModuleLoadDataEx = (CUresult (*) (CUmodule *, const void *, unsigned int, CUjit_option *, void **)) dlsym(cuda_handle, "cuModuleLoadDataEx");
        }
        assert(lcuModuleLoadDataEx);

        static CUresult (*lcuModuleGetFunction) (CUfunction*, CUmodule, const char*);
        if (!lcuModuleGetFunction) {
            lcuModuleGetFunction = (CUresult (*) (CUfunction*, CUmodule, const char*)) dlsym(cuda_handle, "cuModuleGetFunction");
        }
        assert(lcuModuleGetFunction);

        for (auto &sliced_ptx_file : sliced_ptx_files) {

            std::ifstream t(sliced_ptx_file);
            std::string sliced_ptx_file_str((std::istreambuf_iterator<char>(t)),
                                std::istreambuf_iterator<char>());

            CUmodule cudaModule;
            lcuModuleLoadDataEx(&cudaModule, sliced_ptx_file_str.c_str(), 0, 0, 0);

            std::string kernel_names_str = exec("python3 ../scripts/run_slice.py --input-file " + sliced_ptx_file + " --get-names");
            
            std::stringstream ss(kernel_names_str);
            std::string kernel_name;

            while (std::getline(ss, kernel_name, '\n')) {
                if (kernel_name != "") {
                    CUfunction function;
                    lcuModuleGetFunction(&function, cudaModule, kernel_name.c_str());

                    const void *hostFunc = kernel_name_map[kernel_name];
                    kernel_map[hostFunc] = function;
                }
            }
        }
        registered = true;
    }

    Preload(){
        cuda_handle = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_LAZY);
        assert(cuda_handle);
    }

    ~Preload(){}
};

Preload tracer;

extern "C" { 

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    static cudaError_t (*lcudaLaunchKernel) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t );
    if (!lcudaLaunchKernel) {
        lcudaLaunchKernel = (cudaError_t (*) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t )) dlsym(RTLD_NEXT, "cudaLaunchKernel");
    }
    assert(lcudaLaunchKernel);

    static CUresult (*lcuLaunchKernel) (CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
    if (!lcuLaunchKernel) {
        lcuLaunchKernel = (CUresult (*) (CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**)) dlsym(tracer.cuda_handle, "cuLaunchKernel");
    }
    assert(lcuLaunchKernel);

    if (!tracer.registered) {
        tracer.register_sliced_kernels();
    }

    auto cu_func = tracer.kernel_map[func];
    assert(cu_func);

    dim3 new_grid_dim(8, 1, 1);
    dim3 blockOffset(0, 0, 0);

    CUresult res;

    while (blockOffset.x < gridDim.x && blockOffset.y < gridDim.y && blockOffset.z < gridDim.z) {

        void *KernelParams[] = { args[0], args[1], args[2], args[3], &blockOffset };

        res = lcuLaunchKernel(cu_func, new_grid_dim.x, new_grid_dim.y, new_grid_dim.z,
                        blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        if (res != CUDA_SUCCESS) {
            return cudaErrorInvalidValue;
        }

        blockOffset.x += new_grid_dim.x;

        if (blockOffset.x >= gridDim.x) {
            blockOffset.x = 0;
            blockOffset.y += new_grid_dim.y;

            if (blockOffset.y >= gridDim.y) {
                blockOffset.y = 0;
                blockOffset.z += new_grid_dim.z;
            }
        }
    }

    return cudaSuccess;
    // lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    static void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
    if (!l__cudaRegisterFunction) {
        l__cudaRegisterFunction = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    }
    assert(l__cudaRegisterFunction);

    tracer.kernel_name_map[std::string(deviceFun)] = hostFun;

    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void** __cudaRegisterFatBinary( void *fatCubin ) {

    static CUresult (*lcuModuleLoadDataEx) (CUmodule *, const void *, unsigned int, CUjit_option *, void **);
    if (!lcuModuleLoadDataEx) {
        lcuModuleLoadDataEx = (CUresult (*) (CUmodule *, const void *, unsigned int, CUjit_option *, void **)) dlsym(tracer.cuda_handle, "cuModuleLoadDataEx");
    }
    assert(lcuModuleLoadDataEx);

    static CUresult (*lcuModuleGetFunction) (CUfunction*, CUmodule, const char*);
    if (!lcuModuleGetFunction) {
        lcuModuleGetFunction = (CUresult (*) (CUfunction*, CUmodule, const char*)) dlsym(tracer.cuda_handle, "cuModuleGetFunction");
    }
    assert(lcuModuleGetFunction);

    __fatBinC_Wrapper_t *wp = (__fatBinC_Wrapper_t *) fatCubin;

    int magic = wp->magic;
    int version = wp->version;

    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) wp->data;
    size_t fatCubin_data_size_bytes = fbh->headerSize + fbh->fatSize;

    int i = 0;
    while (true) {
        std::string file_name = "output" + std::to_string(i) + ".cubin";
        std::ifstream infile(file_name);
        if (infile.good()) {
            i++;
            continue;
        }
        infile.close();

        std::ofstream file(file_name, std::ios::binary); // Open the file in binary mode
        file.write(reinterpret_cast<const char*>(wp->data), fatCubin_data_size_bytes);
        file.close();

        exec("cuobjdump -xptx all " + file_name);
        auto output = exec("cuobjdump " + file_name + " -lptx");
        std::stringstream ss(output);
        std::vector<std::string> lines;
        std::string line;

        while (std::getline(ss, line, '\n')) {
            lines.push_back(line);
        }

        // Print the split lines
        for (const auto& line : lines) {
            auto split_str = splitOnce(line, ":");
            auto ptx_file_name = strip(split_str.second);
            auto sliced_ptx_file_name = "sliced_" + ptx_file_name;

            exec("python3 ../scripts/run_slice.py --input-file " + ptx_file_name + " --output-file " + sliced_ptx_file_name);
            tracer.sliced_ptx_files.push_back(sliced_ptx_file_name);
        }

        break;
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

	l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

}