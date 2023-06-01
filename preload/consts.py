import os

# Set these variables
PROFILE_KERNEL = os.getenv('PROFILE_KERNEL', 'False') == 'True'
PRINT_TRACE = os.getenv('PRINT_TRACE', 'True') == 'True'

func_sig_must_contain = ["cu", "(", ")"]
func_sig_must_not_contain = ["noexcept", "{", "}", "return", "for", "throw"]

ignore_keywords = [
    "\"C\"", "CUDARTAPI", "extern", "__host__", "__cudart_builtin__",
    "__attribute__((deprecated))"
]

# These most likely won't trigger work on GPU
exclude_trace_functions = [
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
    "cudnnGetBatchNormalizationTrainingExReserveSpaceSize"    

    # from cuBlas
    "cublasCreate_v2",
    "cublasSetStream_v2",
    "cublasSetWorkspace_v2",
    "cublasSetMathMode",
    "cublasGetMathMode"

    # from profiler
    "cudaProfilerInitialize",
]

profile_kernel_start = """
    float _time_ms = 0.0f;

    cudaEvent_t _start, _stop;
    if (tracer.profile_start) {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
        cudaDeviceSynchronize();

        cudaEventRecord(_start);
    }
"""

profile_kernel_end = """
    if (tracer.profile_start) {
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);
        cudaEventElapsedTime(&_time_ms, _start, _stop);

        tracer._kernel_time.push_back(_time_ms);
    }
"""

profile_cpu_start = """
    time_point_t _start;
    if (tracer.profile_start) {
        _start = std::chrono::high_resolution_clock::now();
    }
"""

profile_cpu_end = """
    if (tracer.profile_start) {
        auto _end = std::chrono::high_resolution_clock::now();
        tracer._cpu_timestamps.push_back({ _start, _end });
    }
"""


preload_template = """
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

// g++ -I/usr/local/cuda/include -fPIC -shared -o preload.so preload.cpp

typedef std::chrono::time_point<std::chrono::system_clock> time_point_t;

class PreloadTracer {

public:
    bool profile_start = false;
    bool print_trace;
    void *cudnn_handle;
    time_point_t null_time_point;
    std::vector<const void *> _kernel_seq;
    std::vector<float> _kernel_time;
    std::vector<std::pair<time_point_t, time_point_t>> _cpu_timestamps;
    std::map<const void *, std::string> _kernel_map;

    PreloadTracer(bool print_trace=true) :
        print_trace(print_trace),
        null_time_point(std::chrono::system_clock::now())
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

trace_initialize_code = f"""
PreloadTracer tracer({"true" if PRINT_TRACE else "false"});
"""

def special_preload_funcs(profile_kernel=False):

    _special_preload_funcs = {
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
    if (tracer.profile_start) {{
        {"tracer._cpu_timestamps.push_back({ tracer.null_time_point, tracer.null_time_point });" if not profile_kernel else ""}
        {"tracer._kernel_seq.push_back((void *) lcudaDeviceSynchronize);" if not profile_kernel else ""}
    }}

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

    {profile_kernel_start if profile_kernel else profile_cpu_start}
    cudaError_t err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    {profile_kernel_end if profile_kernel else profile_cpu_end}

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