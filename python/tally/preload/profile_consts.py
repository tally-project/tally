CUDA_API_HEADER_FILES = [
    "/usr/local/cuda/include/cuda.h",
    "/usr/local/cuda/include/cuda_runtime.h",
    "/usr/local/cuda/include/cudnn.h",
    "/usr/local/cuda/include/cublas_v2.h",
    "/usr/local/cuda/include/cuda_profiler_api.h"
]

FUNC_SIG_MUST_CONTAIN = ["cu", "(", ")"]
FUNC_SIG_MUST_NOT_CONTAIN = ["noexcept", "{", "}", "return", "for", "throw"]

IGNORE_KEYWORDS = [
    "\"C\"", "CUDARTAPI", "extern", "__host__", "__cudart_builtin__",
    "__attribute__((deprecated))"
]

# These most likely won't trigger work on GPU
EXCLUDE_TRACE_FUNCTIONS = [
    # from CUDA Runtime
    "cuInit",
    "cudaGetDevice",
    "cudaGetDeviceCount",
    "cudaStreamIsCapturing",
    "cudaGetDeviceProperties",
    "cudaGetLastError",
    "cudnnSetStream",
    "cudaEventCreate",
    "cudaDeviceSynchronize",
    "cudaEventRecord",
    "cudaEventDestroy",
    "cudaEventSynchronize",
    "cudaEventElapsedTime",
    "cudaStreamSynchronize",

    # from cuDNN
    "cudnnCreate",
    "cudnnBackendGetAttribute",
    "cudnnBackendSetAttribute",
    "cudnnBackendFinalize",
    "cudnnBackendInitialize",
    "cudnnBackendDestroyDescriptor",
    "cudnnBackendCreateDescriptor",
    "cudnnCreateTensorDescriptor",
    "cudnnSetTensorNdDescriptor",
    "cudnnDestroyTensorDescriptor",
    "cudnnGetBatchNormalizationBackwardExWorkspaceSize",
    "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize",
    "cudnnGetBatchNormalizationTrainingExReserveSpaceSize",
    "cudnnGetRNNWorkspaceSize",
    "cudnnCreateRNNDescriptor",
    "cudnnSetRNNDescriptor_v6",
    "cudnnSetRNNMatrixMathType",
    "cudnnCreateFilterDescriptor",
    "cudnnDestroyFilterDescriptor",
    "cudnnDestroyDropoutDescriptor",
    "cudnnDestroyRNNDescriptor",
    "cudnnCreateDropoutDescriptor",
    "cudnnRestoreDropoutDescriptor",
    "cudnnRNNGetClip",
    "cudnnGetRNNTrainingReserveSize",
    "cudnnGetRNNLinLayerMatrixParams",
    "cudnnGetRNNLinLayerBiasParams",

    # from cuBlas
    "cublasCreate_v2",
    "cublasSetStream_v2",
    "cublasSetWorkspace_v2",
    "cublasSetMathMode",
    "cublasGetMathMode",

    # from profiler
    "cudaProfilerInitialize",
]

PROFILE_KERNEL_START = """
    float _time_ms = 0.0f;

    cudaEvent_t _start, _stop;
    if (tracer.profile_start) {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
        cudaDeviceSynchronize();

        cudaEventRecord(_start);
    }
"""

PROFILE_KERNEL_END = """
    if (tracer.profile_start) {
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);
        cudaEventElapsedTime(&_time_ms, _start, _stop);

        tracer._kernel_time.push_back(_time_ms);
    }
"""

PROFILE_CPU_START = """
    time_point_t _start;
    if (tracer.profile_start) {
        _start = std::chrono::high_resolution_clock::now();
    }
"""

PROFILE_CPU_END = """
    if (tracer.profile_start) {
        auto _end = std::chrono::high_resolution_clock::now();
        tracer._cpu_timestamps.push_back({ _start, _end });
    }
"""

API_DECL_TEMPLATE_TOP = """

#ifndef TALLY_CUDA_API_H
#define TALLY_CUDA_API_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <nvrtc.h>

"""

API_DECL_TEMPLATE_BUTTOM = """

extern void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
extern void** (*l__cudaRegisterFatBinary) (void *);
extern void (*l__cudaRegisterFatBinaryEnd) (void **);

#endif // TALLY_CUDA_API_H

"""

API_DEF_TEMPLATE_TOP = """

#include <dlfcn.h>

#include <tally/cuda_api.h>
#include <tally/env.h>

void *cuda_handle = dlopen(LIBCUDA_PATH, RTLD_LAZY);
void *cudart_handle = dlopen(LIBCUDART_PATH, RTLD_LAZY);
void *cudnn_handle = dlopen(LIBCUDNN_PATH, RTLD_LAZY);

"""

API_DEF_TEMPLATE_BUTTOM = """

void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)
    = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(cudart_handle, "__cudaRegisterFunction");

void** (*l__cudaRegisterFatBinary) (void *) = 
    (void** (*) (void *)) dlsym(cudart_handle, "__cudaRegisterFatBinary");

void (*l__cudaRegisterFatBinaryEnd) (void **) =
	(void (*) (void **)) dlsym(cudart_handle, "__cudaRegisterFatBinaryEnd");

"""

PRELOAD_TEMPLATE = """
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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <nvrtc.h>

// g++ -I/usr/local/cuda/include -fPIC -shared -o preload.so preload.cpp

typedef std::chrono::time_point<std::chrono::system_clock> time_point_t;

class PreloadTracer {

public:
    bool profile_start = false;
    bool print_trace;
    void *cudnn_handle;
    std::vector<const void *> _kernel_seq;
    std::vector<float> _kernel_time;
    std::vector<std::pair<time_point_t, time_point_t>> _cpu_timestamps;
    std::map<const void *, std::string> _kernel_map;

    PreloadTracer(bool print_trace=true) :
        print_trace(print_trace)
    {
        cudnn_handle = dlopen("/usr/local/cuda/lib64/libcudnn.so", RTLD_LAZY);
        assert(cudnn_handle);
    }

    ~PreloadTracer() {
        std::cout << "preload_post_process" << std::endl;
        if (_kernel_time.size() > 0) {
            assert(_kernel_time.size() == _kernel_seq.size());
        }
        if (_cpu_timestamps.size() > 0) {
            assert(_cpu_timestamps.size() == _kernel_seq.size());
        }
        for (size_t i = 0; i < _kernel_seq.size(); i++) {
            auto _kernel = _kernel_seq[i];
            if (_kernel_map.find(_kernel) != _kernel_map.end()) {

                std::ostringstream stream;

                stream << _kernel_map[_kernel];

                if (_kernel_time.size() > 0) {
                    stream << " Kernel Time: " << _kernel_time[i];
                }

                if (_cpu_timestamps.size() > 0) {
                    stream << " Start: " <<
						std::chrono::duration_cast<std::chrono::nanoseconds>(
                   			_cpu_timestamps[i].first.time_since_epoch()).count();
                    stream << " End: " <<
						std::chrono::duration_cast<std::chrono::nanoseconds>(
                   			_cpu_timestamps[i].second.time_since_epoch()).count();
                }

                if (print_trace) {
                    std::cout << stream.str() << std::endl;
                }
            } else {
                std::cerr << "Cannot find _kernel in _kernel_map" << std::endl;
            }
        }
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

"""

def get_trace_initialize_code(print_trace=True):
    return f"""
PreloadTracer tracer({"true" if print_trace else "false"});\n
    """

def special_preload_funcs(profile_kernel=False):

    _special_preload_funcs = {
    "cudaStreamSynchronize": f"""
cudaError_t cudaStreamSynchronize(cudaStream_t  stream)
{{
	static cudaError_t (*lcudaStreamSynchronize) (cudaStream_t );
	if (!lcudaStreamSynchronize) {{
		lcudaStreamSynchronize = (cudaError_t (*) (cudaStream_t )) dlsym(RTLD_NEXT, "cudaStreamSynchronize");
		tracer._kernel_map[(void *) lcudaStreamSynchronize] = std::string("cudaStreamSynchronize");
	}}
	assert(lcudaStreamSynchronize);

    // Only matters when collecting CPU trace
    {PROFILE_CPU_START if not profile_kernel else ""}
	cudaError_t res = lcudaStreamSynchronize(stream);
    {PROFILE_CPU_END if not profile_kernel else ""}

    {"if (tracer.profile_start) { tracer._kernel_seq.push_back((void *) lcudaStreamSynchronize); }" if not profile_kernel else ""}

	return res;
}}
""",
        "cudaDeviceSynchronize": f"""
cudaError_t cudaDeviceSynchronize()
{{
	static cudaError_t (*lcudaDeviceSynchronize) ();
	if (!lcudaDeviceSynchronize) {{
		lcudaDeviceSynchronize = (cudaError_t (*) ()) dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
		tracer._kernel_map[(void *) lcudaDeviceSynchronize] = std::string("cudaDeviceSynchronize");
	}}
	assert(lcudaDeviceSynchronize);
    
    // Only matters when collecting CPU trace
    {PROFILE_CPU_START if not profile_kernel else ""}
    auto err = lcudaDeviceSynchronize();
    {PROFILE_CPU_END if not profile_kernel else ""}

    {"if (tracer.profile_start) { tracer._kernel_seq.push_back((void *) lcudaDeviceSynchronize); }" if not profile_kernel else ""}

    return lcudaDeviceSynchronize();
}}
        """,

        "__cudaRegisterFunction": """
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
        """,

        "cudaLaunchKernel": f"""
cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{{
    static cudaError_t (*lcudaLaunchKernel) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t );
    if (!lcudaLaunchKernel) {{
        lcudaLaunchKernel = (cudaError_t (*) (const void *, dim3 , dim3 , void **, size_t , cudaStream_t )) dlsym(RTLD_NEXT, "cudaLaunchKernel");
    }}
    assert(lcudaLaunchKernel);

    {PROFILE_KERNEL_START if profile_kernel else PROFILE_CPU_START}
    cudaError_t err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    {PROFILE_KERNEL_END if profile_kernel else PROFILE_CPU_END}

    if (tracer.profile_start) {{
        tracer._kernel_seq.push_back(func);
    }}

    return err;
}}
        """,

        "cudaProfilerStart": """
cudaError_t cudaProfilerStart()
{
	static cudaError_t (*lcudaProfilerStart) ();
	if (!lcudaProfilerStart) {
		lcudaProfilerStart = (cudaError_t (*) ()) dlsym(RTLD_NEXT, "cudaProfilerStart");
	}
	assert(lcudaProfilerStart);

    tracer.profile_start = true;
    return lcudaProfilerStart();
}
        """,

        "cudaProfilerStop": """
cudaError_t cudaProfilerStop()
{
	static cudaError_t (*lcudaProfilerStop) ();
	if (!lcudaProfilerStop) {
		lcudaProfilerStop = (cudaError_t (*) ()) dlsym(RTLD_NEXT, "cudaProfilerStop");
	}
	assert(lcudaProfilerStop);

    tracer.profile_start = false;
    return lcudaProfilerStop();
}
        """
}
    return _special_preload_funcs