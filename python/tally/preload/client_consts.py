
API_ENUM_TEMPLATE_TOP = """
#ifndef TALLY_CUDA_API_ENUM_H
#define TALLY_CUDA_API_ENUM_H
"""

API_ENUM_TEMPLATE_BUTTOM = """
#endif // TALLY_CUDA_API_ENUM_H
"""

API_SPECIAL_ENUM = [
    "__CUDAREGISTERFUNCTION",
    "__CUDAREGISTERFATBINARY",
    "__CUDAREGISTERFATBINARYEND"
]

API_DECL_TEMPLATE_TOP = """
#ifndef TALLY_CUDA_API_H
#define TALLY_CUDA_API_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>

"""

API_DECL_TEMPLATE_BUTTOM = """

extern void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
extern void** (*l__cudaRegisterFatBinary) (void *);
extern void (*l__cudaRegisterFatBinaryEnd) (void **);

#endif // TALLY_CUDA_API_H

"""

API_DEF_TEMPLATE_TOP = """
#include <dlfcn.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>

#include <tally/generated/cuda_api.h>
#include <tally/env.h>

void *cuda_handle = dlopen(LIBCUDA_PATH, RTLD_LAZY);
void *cudart_handle = dlopen(LIBCUDART_PATH, RTLD_LAZY);
void *cudnn_handle = dlopen(LIBCUDNN_PATH, RTLD_LAZY);
void *cublas_handle = dlopen(LIBCUBLAS_PATH, RTLD_LAZY);
void *cublasLt_handle = dlopen(LIBCUBLASLT_PATH, RTLD_LAZY);

"""

API_DEF_TEMPLATE_BUTTOM = """

void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)
    = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(cudart_handle, "__cudaRegisterFunction");

void** (*l__cudaRegisterFatBinary) (void *) = 
    (void** (*) (void *)) dlsym(cudart_handle, "__cudaRegisterFatBinary");

void (*l__cudaRegisterFatBinaryEnd) (void **) =
	(void (*) (void **)) dlsym(cudart_handle, "__cudaRegisterFatBinaryEnd");

"""

MSG_STRUCT_TEMPLATE_TOP = """
#ifndef TALLY_GENERATED_MSG_STRUCT_H
#define TALLY_GENERATED_MSG_STRUCT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>


"""

MSG_STRUCT_TEMPLATE_BUTTOM = """

#endif // TALLY_GENERATED_MSG_STRUCT_H
"""

CLIENT_PRELOAD_TEMPLATE = """
#include <dlfcn.h>
#include <stdio.h>
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
#include <unistd.h>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>

#include "tally/cuda_util.h"
#include "tally/msg_struct.h"
#include "tally/client.h"
#include "tally/log.h"
#include "tally/ipc_util.h"
#include "tally/generated/cuda_api.h"
#include "tally/generated/cuda_api_enum.h"
#include "tally/generated/msg_struct.h"

"""

TALLY_SERVER_HEADER_TEMPLATE_TOP = """
#ifndef TALLY_SERVER_H
#define TALLY_SERVER_H

#include <signal.h>
#include <string>
#include <atomic>
#include <map>
#include <iostream>
#include <functional>
#include <memory>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>

#include <readerwriterqueue.h>

#include "spdlog/spdlog.h"

#include <folly/concurrency/ConcurrentHashMap.h>

#include "iceoryx_dust/posix_wrapper/signal_watcher.hpp"
#include "iceoryx_posh/popo/untyped_server.hpp"
#include "iceoryx_posh/runtime/posh_runtime.hpp"

#include <tally/log.h>
#include <tally/cuda_launch.h>
#include <tally/msg_struct.h>
#include <tally/cuda_util.h>
#include <tally/cache_struct.h>

static std::function<void(int)> __exit;

static void __exit_wrapper(int signal) {
    __exit(signal);
}

typedef std::function<CUresult(CudaLaunchConfig, uint32_t *, bool *, bool, float, float*, float*, int32_t)> kernel_partial_t;


class KernelLaunchWrapper {

public:
	// Callable to launch kernel
	kernel_partial_t kernel_to_dispatch;

	// whether it is blackbox kernel from nvidia libraries
	bool is_library_call;

	// unique identification of the kernel
	CudaLaunchCall launch_call;

	// Stream to launch to
	cudaStream_t launch_stream;

	// Useful info
	int dynamic_shmem_size_bytes = 0;
};

class ClientData {

public:
	// For registering kernels at the start:
    unsigned long long* fatbin_data = nullptr;
    uint32_t fatBinSize;
    bool cubin_registered = false;
	uint32_t cubin_uid = 0;
    std::vector<std::pair<void *, std::string>> register_queue;

	// Following are used at runtime:

	// Used to check whether an address points to device memory
	std::vector<DeviceMemoryKey> dev_addr_map;

	std::unordered_map<const void *, const void *> _kernel_client_addr_mapping;

	moodycamel::ReaderWriterQueue<KernelLaunchWrapper> kernel_dispatch_queue;
	std::atomic<uint32_t> queue_size = 0;

	uint32_t *global_idx;
	bool *retreat;

    cudaStream_t default_stream = nullptr;
	std::atomic<bool> has_exit = false;
};

class TallyServer {

public:

    static TallyServer *server;

	std::atomic<bool> is_quit__ {false};

	// ================== Per-client state ===================
	std::map<int32_t, ClientData> client_data_all;
    std::map<int32_t, iox::popo::UntypedServer *> worker_servers;
	std::map<int32_t, std::atomic<bool>> threads_running_map;
    
	// ==================== Global state =====================
	std::unordered_map<CUDA_API_ENUM, std::function<void(void *, iox::popo::UntypedServer *, const void* const)>> cuda_api_handler_map;

	folly::ConcurrentHashMap<CUmodule, std::pair<const char *, size_t>> jit_module_to_cubin_map;
	folly::ConcurrentHashMap<CUfunction, std::vector<uint32_t>> _jit_kernel_addr_to_args;
	folly::ConcurrentHashMap<const void *, std::vector<uint32_t>> _kernel_addr_to_args;
	
	// Map func addr to kernel name and cubin hash
	folly::ConcurrentHashMap<const void *, std::string> host_func_to_demangled_kernel_name_map;
	folly::ConcurrentHashMap<const void *, uint32_t> host_func_to_cubin_uid_map;

	// Map kernel name and cubin hash to a host func
	folly::ConcurrentHashMap<std::pair<std::string, uint32_t>, const void *> demangled_kernel_name_and_cubin_uid_to_host_func_map;

	// Use cubin as unique id of a kernel
	// { Cubin uid: { Kernel Name: host func addr } }
	std::map<uint32_t, folly::ConcurrentHashMap<std::string, const void *>> cubin_to_kernel_name_to_host_func_map;

	// Register original and transformed kernels here
	folly::ConcurrentHashMap<const void *, WrappedCUfunction> original_kernel_map;
    folly::ConcurrentHashMap<const void *, WrappedCUfunction> ptb_kernel_map;
	folly::ConcurrentHashMap<const void *, WrappedCUfunction> dynamic_ptb_kernel_map;
	folly::ConcurrentHashMap<const void *, WrappedCUfunction> preemptive_ptb_kernel_map;

    folly::ConcurrentHashMap<CUfunction, CUfunction> jit_ptb_kernel_map;
	folly::ConcurrentHashMap<CUfunction, CUfunction> jit_dynamic_ptb_kernel_map;
	folly::ConcurrentHashMap<CUfunction, CUfunction> jit_preemptive_ptb_kernel_map;

	// Performance cache to use at runtime
	std::unordered_map<CudaLaunchCallConfig, CudaLaunchCallConfigResult> single_kernel_perf_map;
	std::unordered_map<CudaLaunchCall, CudaLaunchCallConfigResult> single_kernel_best_config_map;

    std::unordered_map<CudaLaunchCallPair, std::unordered_map<CudaLaunchCallConfigPair, CudaLaunchCallConfigPairResult>> kernel_pair_perf_map;
	std::unordered_map<CudaLaunchCallPair, CudaLaunchCallConfigPairResult> kernel_pair_best_config_map;

	// Set and Get performance cache

	CudaLaunchCallConfigResult get_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config, bool *found);
	void set_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config, CudaLaunchMetadata meta_data, float norm_speed, float latency, uint32_t iters);

	CudaLaunchCallConfigResult get_single_kernel_best_config(CudaLaunchCall &launch_call, bool *found);
	void set_single_kernel_best_config(CudaLaunchCall &launch_call, CudaLaunchCallConfigResult &best_config);

    CudaLaunchCallConfigPairResult get_kernel_pair_perf(CudaLaunchCall &launch_call_1, CudaLaunchCall &launch_call_2,
														CudaLaunchConfig &launch_config_1, CudaLaunchConfig &launch_config_2,
														bool *found);
	
	void set_kernel_pair_perf(CudaLaunchCall &launch_call_1, CudaLaunchCall &launch_call_2,
							  CudaLaunchConfig &launch_config_1, CudaLaunchConfig &launch_config_2,
							  CudaLaunchMetadata meta_data_1, CudaLaunchMetadata meta_data_2,
							  float norm_speed_1, float norm_speed_2, float latency_1, float latency_2,
							  float fixed_workload_latency, float fixed_workload_speedup,
							  float unfair_workload_latency, float unfair_workload_speedup);

	CudaLaunchCallConfigPairResult get_kernel_pair_best_config(CudaLaunchCall &launch_call_1, CudaLaunchCall &launch_call_2, bool *found);
	void set_kernel_pair_best_config(CudaLaunchCall &launch_call_1, CudaLaunchCall &launch_call_2, CudaLaunchCallConfigPairResult best_config);

	// Utility functions for measurement data
	CudaLaunchCall convert_key_to_call(CudaLaunchKey key);
	CudaLaunchKey convert_call_to_key(CudaLaunchCall call);

	CudaLaunchCallConfig convert_key_config_to_call_config(CudaLaunchKeyConfig key_config);
	CudaLaunchKeyConfig convert_call_config_to_key_config(CudaLaunchCallConfig call_config);

	CudaLaunchCallConfigPairResult convert_pair_res_to_runtime_res(CudaLaunchKeyConfigPairResult res);
	CudaLaunchKeyConfigPairResult convert_pair_res_to_cache_res(CudaLaunchCallConfigPairResult res);

	CudaLaunchCallConfigPair convert_key_config_pair_to_call_config_pair(CudaLaunchKeyConfigPair key_config_pair);
	CudaLaunchKeyConfigPair convert_call_config_pair_to_key_config_pair(CudaLaunchCallConfigPair call_config_pair);

    void save_performance_cache();

    static constexpr char APP_NAME[] = "iox-cpp-request-response-server-untyped";

    TallyServer();
    ~TallyServer();

	// Scheduler options
	void run_naive_scheduler();
	void run_profile_scheduler();

    void register_api_handler();
    void load_cache();

	void register_kernels();
    void register_measurements();
    void register_ptx_transform(const char* cubin_data, size_t cubin_size);

    void start_scheduler();
    void start_main_server();
    void start_worker_server(int32_t client_id);

	void wait_until_launch_queue_empty(int32_t client_uid);

	template<typename T>
    std::function<CUresult(CudaLaunchConfig, uint32_t *, bool *, bool, float, float*, float*, int32_t)>
	cudaLaunchKernel_Partial(T, dim3, dim3, size_t, cudaStream_t, char *);

	std::function<CUresult(CudaLaunchConfig, uint32_t *, bool *, bool, float, float*, float*, int32_t)>
	cublasSgemm_v2_Partial(cublasSgemm_v2Arg *args);
    
"""

TALLY_SERVER_HEADER_TEMPLATE_BUTTOM = """
};

#endif // TALLY_SERVER_H
"""

TALLY_CLIENT_SRC_TEMPLATE_TOP = """
#include <cstring>
#include <memory>
#include <vector>

#include <tally/util.h>
#include <tally/client.h>
#include <tally/consts.h>
#include <tally/generated/cuda_api.h>

TallyClient *TallyClient::client;

__attribute__((__constructor__)) void init_client()
{
    NO_INIT_PROCESS_KEYWORDS_VEC;

    auto process_name = get_process_name(getpid());

    for (auto &keyword : no_init_process_keywords) {

        if (containsSubstring(process_name, keyword)) {
            return;
        }
    }

    TallyClient::client = new TallyClient;
}

"""

REGISTER_FUNCS = [
    "__cudaRegisterFatBinary",
    "__cudaRegisterFunction",
    "__cudaRegisterFatBinaryEnd"
]

# API calls that will launch a kernel
# For these, we will delay the actual dispatch to the hardware
# as we intend to schedule ourselves.
KERNEL_LAUNCH_CALLS = [
    "cudaLaunchKernel",
    "cudnnRNNBackwardWeights",
    "cudnnRNNBackwardData",
    "cudnnRNNForwardTraining",
    "cudnnMultiHeadAttnBackwardData",
    "cudnnMultiHeadAttnForward",
    "cublasSgemmEx",
    "cudnnTransformTensor",
    "cublasSgemv_v2",
    "cudnnLRNCrossChannelForward",
    "cublasSgemm_v2",
    "cudnnSoftmaxForward",
    "cudnnAddTensor",
    "cublasLtMatmul",
    "cudnnActivationForward",
    "cudnnConvolutionForward",
    "cudnnPoolingForward",
    "cudnnMultiHeadAttnBackwardWeights",
    "cudnnReorderFilterAndBias",
    "cudnnBatchNormalizationForwardTrainingEx",
    "cudnnBatchNormalizationBackwardEx",
    "cudnnRNNBackwardWeights_v8",
    "cudnnRNNBackwardData_v8",
    "cudnnRNNForward",
    "cudnnBackendExecute"
]

# let the client call the APIs directly
DIRECT_CALLS = [
    "cuCtxDestroy_v2",
    "cuInit",
    "cuDeviceGetName",
    "cudaGetErrorString",
    "cuGetErrorString",
    "cuGetErrorName",
    "cudnnGetErrorString",
    "cudaMallocHost",
    "cudaFreeHost",
    "cudaGetDevice",
    "cudaGetDeviceCount",
    "cudaDriverGetVersion",
    "cudaRuntimeGetVersion",
    "cudaFuncGetAttributes",
    "cudaFuncGetAttribute",
    "cuGetExportTable",
    "cuCtxCreate_v2",
]

# implement manually
SPECIAL_CLIENT_PRELOAD_FUNCS = [
    "cudaMemset",
    "cuMemFree_v2",
    "cuMemAllocAsync",
    "cuMemcpyAsync",
    "cuMemcpy",
    "cuGetProcAddress_v2",
    "cudaFuncSetAttribute",
    "cudaStreamGetCaptureInfo_v2",
    "cudaGraphGetNodes",
    "cuLaunchKernel",
    "cuPointerGetAttribute",
    "cuModuleGetFunction",
    "cuModuleLoadData",
    "cublasCreate_v2",
    "cudnnCreate",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventRecord",
    "cudaThreadSynchronize",
    "cudnnBackendExecute",
    "cudnnRNNForward",
    "cudnnRNNBackwardData_v8",
    "cudnnRNNBackwardWeights_v8",
    "cudaSetDevice",
    "cudaChooseDevice",
    "cublasSgemmStridedBatched",
    "cudnnGetTensorNdDescriptor",
    "cudnnSetRNNDataDescriptor",
    "cudnnRNNBackwardWeights",
    "cudnnRNNBackwardData",
    "cudnnRNNForwardTraining",
    "cudnnGetFilterNdDescriptor",
    "cudnnGetRNNTrainingReserveSize",
    "cudnnGetRNNWorkspaceSize",
    "cudaFree",
    "cudaMalloc",
    "cudnnMultiHeadAttnBackwardData",
    "cudnnMultiHeadAttnForward",
    "cudnnGetSeqDataDescriptor",
    "cublasSgemmEx",
    "cudnnTransformTensor",
    "cublasSgemv_v2",
    "cudnnLRNCrossChannelForward",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaLaunchKernel",
    "cublasSgemm_v2",
    "cudnnSoftmaxForward",
    "cublasLtMatmulDescSetAttribute",
    "cublasLtMatrixLayoutSetAttribute",
    "cublasLtMatmulPreferenceSetAttribute",
    "cublasLtMatmulAlgoGetHeuristic",
    "cudnnFindConvolutionForwardAlgorithm",
    "cudnnAddTensor",
    "cudnnSetPoolingNdDescriptor",
    "cudnnGetPoolingNdDescriptor",
    "cudnnGetPoolingNdForwardOutputDim",
    "cublasLtMatmul",
    "cudnnBackendSetAttribute",
    "cudnnBackendGetAttribute",
    "cudnnSetTensorNdDescriptor",
    "cudnnSetConvolutionNdDescriptor",
    "cudnnGetConvolutionNdForwardOutputDim",
    "cudnnGetConvolutionForwardAlgorithm_v7",
    "cudnnSetFilterNdDescriptor",
    "cudnnActivationForward",
    "cudnnConvolutionForward",
    "cudnnPoolingForward",
    "cudnnSetSeqDataDescriptor",
    "cudnnMultiHeadAttnBackwardWeights",
    "cudnnReorderFilterAndBias",
    "cudnnBatchNormalizationForwardTrainingEx",
    "cudnnBatchNormalizationBackwardEx",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "__cudaRegisterFunction",
    "__cudaRegisterFatBinary",
    "__cudaRegisterFatBinaryEnd"
]

# These api calls can be directly forwarded to the server without addtional logic
# this means no value needs to be assigned
FORWARD_API_CALLS = [
    "cuMemsetD8_v2",
    "cuEventDestroy_v2",
    "cuStreamWaitEvent",
    "cuEventRecord",
    "cuGraphLaunch",
    "cuStreamBeginCapture_v2",
    "cuStreamSynchronize",
    "cudaGraphUpload",
    "cudaGraphLaunch",
    "cudaGraphExecDestroy",
    "cudaGraphDestroy",
    "cudnnRestoreDropoutDescriptor",
    "cudnnRNNSetClip",
    "cudnnRNNSetClip_v8",
    "cudnnSetRNNMatrixMathType",
    "cudaMemsetAsync",
    "cublasSetSmCountTarget",
    "cublasSetLoggerCallback",
    "cudnnGetFoldedConvBackwardDataDescriptors",
    "cudnnSetRNNAlgorithmDescriptor",
    "cudnnSetRNNDescriptor_v8",
    "cublasLtLoggerForceDisable",
    "cublasLtGetCudartVersion",
    "cublasLtGetVersion",
    "cudnnCnnInferVersionCheck",
    "cudnnAdvTrainVersionCheck",
    "cudnnAdvInferVersionCheck",
    "cudnnOpsTrainVersionCheck",
    "cudnnOpsInferVersionCheck",
    "cudaMemPoolTrimTo"
    "cudaFreeArray",
    "cudnnSetAttnDescriptor",
    "cudnnSetDropoutDescriptor",
    "cudnnDestroySeqDataDescriptor",
    "cudnnDestroyTensorTransformDescriptor",
    "cudnnDestroyLRNDescriptor",
    "cudnnSetFilter4dDescriptor",
    "cudnnCreateActivationDescriptor",
    "cudnnDestroyPoolingDescriptor",
    "cudaProfilerStart",
    "cudaProfilerStop",
    "cudaDeviceReset",
    "cudaDeviceSetLimit",
    "cudaDeviceSetCacheConfig",
    "cudaDeviceSetSharedMemConfig",
    "cudaThreadExit",
    "cudaThreadSetLimit",
    "cudaThreadSetCacheConfig",
    "cudaGetLastError",
    "cudaPeekAtLastError",
    "cudaSetDeviceFlags",
    "cudaCtxResetPersistingL2Cache",
    "cudaEventRecordWithFlags",
    "cudaEventQuery",
    "cudaEventSynchronize",
    "cudaEventDestroy",
    "cudaDeviceSetMemPool",
    "cudaStreamCopyAttributes",
    "cudaStreamDestroy",
    "cudaStreamWaitEvent",
    "cudaStreamQuery",
    "cudaStreamBeginCapture",
    "cublasDestroy_v2",
    "cublasGetCudartVersion",
    "cuDeviceSetMemPool",
    "cuFlushGPUDirectRDMAWrites",
    "cuDevicePrimaryCtxRelease_v2",
    "cuDevicePrimaryCtxSetFlags_v2",
    "cuDevicePrimaryCtxReset_v2",
    "cublasSetStream_v2",
    "cublasSetWorkspace_v2",
    "cublasSetMathMode",
    "cublasLtMatmulDescDestroy",
    "cublasLtMatrixLayoutDestroy",
    "cublasLtMatmulPreferenceDestroy",
    "cublasLtDestroy",
    "cuCtxPushCurrent_v2",
    "cuCtxSetCurrent",
    "cuCtxSynchronize",
    "cuCtxSetLimit",
    "cuCtxSetCacheConfig",
    "cuCtxSetSharedMemConfig",
    "cuCtxResetPersistingL2Cache",
    "cuCtxDetach",
    "cuModuleUnload",
    "cudnnDestroy",
    "cudnnSetStream",
    "cudnnSetTensor4dDescriptor",
    "cudnnSetTensor4dDescriptorEx",
    "cudnnBackendDestroyDescriptor",
    "cudnnBackendInitialize",
    "cudnnBackendFinalize",
    "cublasSetPointerMode_v2",
    "cudnnDestroyFilterDescriptor",
    "cudnnGetVersion",
    "cudnnGetMaxDeviceVersion",
    "cudnnGetCudartVersion",
    "cudnnDestroyTensorDescriptor",
    "cudnnCnnTrainVersionCheck",
    "cudnnSetActivationDescriptor",
    "cudnnDestroyConvolutionDescriptor",
    "cudnnDestroyActivationDescriptor",
    "cudnnSetLRNDescriptor",
    "cudnnDestroyAttnDescriptor",
    "cudnnDestroyDropoutDescriptor",
    "cudnnDestroyRNNDataDescriptor",
    "cudnnDestroyRNNDescriptor",
    "cudnnSetRNNDescriptor_v6",
    "cudnnBuildRNNDynamic",
    "cuDestroyExternalMemory",
    "cudaIpcCloseMemHandle",
    "cudaDeviceFlushGPUDirectRDMAWrites",
    "cudnnSetOpTensorDescriptor",
    "cublasSetVector",
    "cudnnSetRNNBiasMode"
]

# API calls that has the first argument set
# by CUDA API call, such as cudaStreamCreate
CUDA_GET_1_PARAM_FUNCS = [
    "cuEventCreate",
    "cuGraphInstantiateWithFlags",
    "cuStreamCreateWithPriority",
    "cuModuleGetLoadingMode",
    "cuFuncGetAttribute",
    "cudaGraphInstantiateWithFlags",
    "cublasGetLoggerCallback",
    "cudnnCreateOpTensorDescriptor",
    "cudaIpcGetMemHandle",
    "cudaIpcOpenMemHandle",
    "cudaIpcGetEventHandle",
    "cudaDeviceGetPCIBusId",
    "cuMemAllocFromPoolAsync",
    "cudnnCreateRNNDescriptor",
    "cudnnCreateRNNDataDescriptor",
    "cudnnCreateDropoutDescriptor",
    "cudnnCreateAttnDescriptor",
    "cudnnCreateSeqDataDescriptor",
    "cudnnCreatePoolingDescriptor",
    "cudnnCreateLRNDescriptor",
    "cudaStreamCreate",
    "cudaStreamCreateWithFlags",
    "cudaStreamCreateWithPriority",
    "cudaEventCreate",
    "cudaEventCreateWithFlags",
    "cudaEventElapsedTime",
    "cudaDeviceGetLimit",
    "cudaDeviceGetCacheConfig",
    "cudaDeviceGetSharedMemConfig"
    "cudaIpcGetEventHandle",
    "cudaIpcOpenEventHandle",
    "cudaThreadGetLimit",
    "cudaThreadGetCacheConfig",
    "cudaGetDeviceProperties",
    "cudaGetDeviceProperties_v2",
    "cudaDeviceGetAttribute",
    "cudaDeviceGetDefaultMemPool",
    "cudaDeviceGetMemPool",
    "cudaDeviceGetP2PAttribute",
    "cudaGetDeviceFlags",
    "cudaGraphCreate",
    "cuDriverGetVersion",
    "cuDeviceGet",
    "cuDeviceGetCount",
    "cuDeviceGetUuid",
    "cuDeviceGetUuid_v2",
    "cuDeviceTotalMem_v2",
    "cuDeviceGetAttribute",
    "cuDeviceGetMemPool",
    "cuDeviceGetDefaultMemPool",
    "cuDeviceGetProperties",
    "cuDevicePrimaryCtxRetain",
    "cublasLtMatmulDescCreate",
    "cublasLtMatrixLayoutCreate",
    "cublasLtMatmulPreferenceCreate",
    "cublasLtCreate",
    "cuDeviceGetTexture1DLinearMaxWidth",
    "cuDeviceGetExecAffinitySupport",
    "cuCtxPopCurrent_v2",
    "cuCtxGetCurrent",
    "cuCtxGetDevice",
    "cuCtxGetFlags",
    "cuCtxGetLimit",
    "cuCtxGetCacheConfig",
    "cuCtxGetSharedMemConfig",
    "cuCtxGetExecAffinity",
    "cuCtxAttach",
    "cudnnCreateTensorDescriptor",
    "cudnnCreateTensorTransformDescriptor",
    "cudnnCreateActivationDescriptor",
    "cudnnCreateFilterDescriptor",
    "cudnnCreateConvolutionDescriptor",
]

# these should just throw unsupported error
UNSUPPORTED_FUNCS = [
    "cudaMallocManaged"
]

CUDA_GET_2_PARAM_FUNCS = [
    "cuStreamEndCapture",
    "cudaStreamEndCapture",
    "cudnnGetRNNBiasMode",
    "cudnnGetRNNMatrixMathType",
    "cublasGetSmCountTarget",
    "cublasGetProperty",
    "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount",
    "cudnnGetConvolutionBackwardDataAlgorithmMaxCount",
    "cudaStreamGetFlags",
    "cudaStreamGetPriority",
    "cudnnDropoutGetStatesSize",
    "cudnnGetFilterSizeInBytes",
    "cudaStreamIsCapturing",
    "cublasGetVersion_v2",
    "cuCtxGetApiVersion",
    "cudnnGetStream",
    "cudnnBackendCreateDescriptor",
    "cublasGetStream_v2",
    "cublasGetPointerMode_v2",
    "cudnnGetProperty",
    "cudnnGetTensorSizeInBytes"
]

CUDA_GET_2_3_PARAM_FUNCS = [
    "cudaStreamGetCaptureInfo",
    "cuDevicePrimaryCtxGetState"
]

CUDA_GET_1_2_PARAM_FUNCS = [
    "cudaDeviceGetStreamPriorityRange",
    "cuDeviceGetLuid",
    "cuDeviceComputeCapability",
    "cuCtxGetStreamPriorityRange",
    "cudaMemGetInfo"
]

CUDA_GET_2_3_4_5_6_7_8_9_10_PARAM_FUNCS = [
    "cudnnGetTensor4dDescriptor"
]

CUDA_GET_4_PARAM_FUNCS = [
    "cudnnInitTransformDest",
    "cudnnGetRNNParamsSize"
]

CUDA_GET_7_PARAM_FUNCS = [
    "cudnnGetConvolutionForwardWorkspaceSize",
    "cudnnGetConvolutionBackwardDataWorkspaceSize"
]

CUDA_GET_3_4_5_PARAM_FUNCS = [
    "cudnnGetMultiHeadAttnBuffers"
]

CUDA_GET_9_PARAM_FUNCS = [
    "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize",
    "cudnnGetRNNLinLayerMatrixParams",
    "cudnnGetRNNLinLayerBiasParams"    
]

CUDA_GET_2_3_4_PARAM_FUNCS = [
    "cudnnGetOpTensorDescriptor"
]

CUDA_GET_3_PARAM_FUNCS = [
    "cudnnGetRNNWeightSpaceSize",
    "cudnnGetRNNForwardInferenceAlgorithmMaxCount"
]

CUDA_GET_5_6_PARAM_FUNCS = [
    "cudnnGetRNNTempSpaceSizes"
]

CUDA_GET_8_10_PARAM_FUNCS = [
    "cudnnGetRNNWeightParams"
]

CUDA_GET_6_PARAM_FUNCS = [
    "cudnnGetBatchNormalizationTrainingExReserveSpaceSize"
]

CUDA_GET_11_PARAM_FUNCS = [
    "cudnnGetBatchNormalizationBackwardExWorkspaceSize"
]

CUDA_GET_2_3_4_5_6_PARAM_FUNCS = [

]

CUDA_GET_1_PARAM_FUNC_KEY = 1
CUDA_GET_2_3_PARAM_FUNC_KEY = 2
CUDA_GET_1_2_PARAM_FUNC_KEY = 3
CUDA_GET_2_PARAM_FUNC_KEY = 4
CUDA_GET_2_3_4_5_6_7_8_9_10_PARAM_FUNC_KEY = 5
CUDA_GET_4_PARAM_FUNC_KEY = 6
CUDA_GET_7_PARAM_FUNC_KEY = 7
CUDA_GET_3_4_5_PARAM_FUNC_KEY = 8
CUDA_GET_9_PARAM_FUNC_KEY = 9
CUDA_GET_2_3_4_PARAM_FUNC_KEY = 10
CUDA_GET_3_PARAM_FUNC_KEY = 11
CUDA_GET_5_6_PARAM_FUNC_KEY = 12
CUDA_GET_8_10_PARAM_FUNC_KEY = 13
CUDA_GET_6_PARAM_FUNC_KEY = 14
CUDA_GET_11_PARAM_FUNC_KEY = 15
CUDA_GET_2_3_4_5_6_PARAM_FUNC_KEY = 16

PARAM_INDICES = {
    CUDA_GET_1_PARAM_FUNC_KEY: [0],
    CUDA_GET_2_PARAM_FUNC_KEY: [1],
    CUDA_GET_2_3_PARAM_FUNC_KEY: [1, 2],
    CUDA_GET_1_2_PARAM_FUNC_KEY: [0, 1],
    CUDA_GET_2_3_4_5_6_7_8_9_10_PARAM_FUNC_KEY: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    CUDA_GET_4_PARAM_FUNC_KEY: [3],
    CUDA_GET_7_PARAM_FUNC_KEY: [6],
    CUDA_GET_3_4_5_PARAM_FUNC_KEY: [2, 3, 4],
    CUDA_GET_9_PARAM_FUNC_KEY: [8],
    CUDA_GET_2_3_4_PARAM_FUNC_KEY: [1, 2, 3],
    CUDA_GET_3_PARAM_FUNC_KEY: [2],
    CUDA_GET_5_6_PARAM_FUNC_KEY: [4, 5],
    CUDA_GET_8_10_PARAM_FUNC_KEY: [7, 9],
    CUDA_GET_6_PARAM_FUNC_KEY: [5],
    CUDA_GET_11_PARAM_FUNC_KEY: [10],
    CUDA_GET_2_3_4_5_6_PARAM_FUNC_KEY: [1, 2, 3, 4, 5]
}

def is_get_param_func(func_name):

    for funcs in [
        CUDA_GET_1_PARAM_FUNCS,
        CUDA_GET_2_3_PARAM_FUNCS,
        CUDA_GET_1_2_PARAM_FUNCS,
        CUDA_GET_2_PARAM_FUNCS,
        CUDA_GET_2_3_PARAM_FUNCS,
        CUDA_GET_1_2_PARAM_FUNCS,
        CUDA_GET_2_3_4_5_6_7_8_9_10_PARAM_FUNCS,
        CUDA_GET_4_PARAM_FUNCS,
        CUDA_GET_7_PARAM_FUNCS,
        CUDA_GET_3_4_5_PARAM_FUNCS,
        CUDA_GET_9_PARAM_FUNCS,
        CUDA_GET_2_3_4_PARAM_FUNCS,
        CUDA_GET_3_PARAM_FUNCS,
        CUDA_GET_5_6_PARAM_FUNCS,
        CUDA_GET_8_10_PARAM_FUNCS,
        CUDA_GET_6_PARAM_FUNCS,
        CUDA_GET_11_PARAM_FUNCS,
        CUDA_GET_2_3_4_5_6_PARAM_FUNCS
    ]:
        if func_name in funcs:
            return True
    return False

def get_param_group(func_name):
    if func_name in CUDA_GET_1_PARAM_FUNCS:
        return CUDA_GET_1_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_2_3_PARAM_FUNCS:
        return CUDA_GET_2_3_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_1_2_PARAM_FUNCS:
        return CUDA_GET_1_2_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_2_PARAM_FUNCS:
        return CUDA_GET_2_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_2_3_4_5_6_7_8_9_10_PARAM_FUNCS:
        return CUDA_GET_2_3_4_5_6_7_8_9_10_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_4_PARAM_FUNCS:
        return CUDA_GET_4_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_7_PARAM_FUNCS:
        return CUDA_GET_7_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_3_4_5_PARAM_FUNCS:
        return CUDA_GET_3_4_5_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_9_PARAM_FUNCS:
        return CUDA_GET_9_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_2_3_4_PARAM_FUNCS:
        return CUDA_GET_2_3_4_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_3_PARAM_FUNCS:
        return CUDA_GET_3_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_5_6_PARAM_FUNCS:
        return CUDA_GET_5_6_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_8_10_PARAM_FUNCS:
        return CUDA_GET_8_10_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_6_PARAM_FUNCS:
        return CUDA_GET_6_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_11_PARAM_FUNCS:
        return CUDA_GET_11_PARAM_FUNC_KEY
    elif func_name in CUDA_GET_2_3_4_5_6_PARAM_FUNCS:
        return CUDA_GET_2_3_4_5_6_PARAM_FUNC_KEY
    else:
        assert(False)


def get_preload_func_template_iox(func_name, arg_names, arg_types, ret_type):
    func_preload_builder = f"""
    {ret_type} err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof({func_name}Arg), alignof({func_name}Arg))
        .and_then([&](auto& requestPayload) {{

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::{func_name.upper()};
            header->client_id = TallyClient::client->client_id;
            
            auto request = ({func_name}Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
"""
    for idx, arg_name in enumerate(arg_names):
        arg_type = arg_types[idx]
        if arg_type.strip() == "const void *" or arg_type.strip() == "const void*":
            func_preload_builder += f"\t\t\trequest->{arg_name} = const_cast<void *>({arg_name});\n"
        elif arg_type.strip() == "const int32_t []":
            func_preload_builder += f"\t\t\trequest->{arg_name} = const_cast<int32_t *>({arg_name});\n"
        else:
            func_preload_builder += f"\t\t\trequest->{arg_name} = {arg_name};\n"

    func_preload_builder += f"""
            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) {{ LOG_ERR_AND_EXIT("Could not send Request: ", error); }});
        }})
        .or_else([](auto& error) {{ LOG_ERR_AND_EXIT("Could not allocate Request: ", error); }});
"""
    return func_preload_builder