
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
    "__CUDAREGISTERFATBINARYEND",
    "__CUDAPUSHCALLCONFIGURATION",
    "__CUDAPOPCALLCONFIGURATION"
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
#include <nccl.h>
#include <curand.h>
#include <cusparse_v2.h>

"""

API_DECL_TEMPLATE_BUTTOM = """

extern void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *);
extern void** (*l__cudaRegisterFatBinary) (void *);
extern void (*l__cudaRegisterFatBinaryEnd) (void **);
extern unsigned (*l__cudaPushCallConfiguration)(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream);
extern cudaError_t (*l__cudaPopCallConfiguration)(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream);

#endif // TALLY_CUDA_API_H

"""

API_DEF_TEMPLATE_TOP = """
#include <dlfcn.h>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>
#include <nccl.h>
#include <curand.h>
#include <cusparse_v2.h>

#include <tally/generated/cuda_api.h>
#include <tally/env.h>
#include <tally/util.h>

void *cuda_handle;
void *cudart_handle;
void *cudnn_handle;
void *cublas_handle;
void *cublasLt_handle;
void *nvrtc_handle;
void *nccl_handle;
void *curand_handle;
void *cusparse_handle;

#define REGISTER_HANDLE(HANDLE, LIB_NAME)									\\
	for (auto &path : CUDA_SEARCH_PATHS) {									\\
		auto lib_path = path + LIB_NAME;									\\
		std::ifstream f(lib_path.c_str());									\\
		if (f.good()) {														\\
			HANDLE = dlopen(lib_path.c_str(), RTLD_LAZY);					\\
			if (HANDLE)	break;												\\
		}																	\\
	}																		\\
	if (!HANDLE) throw std::runtime_error("Fail to load " + LIB_NAME);


void __attribute__((constructor)) register_cuda_handles()
{
	const std::string LIB_CUDART_NAME = "libcudart.so";
	const std::string LIB_CUDA_NAME = "libcuda.so.1";
	const std::string LIB_CUDNN_NAME = "libcudnn.so";
	const std::string LIB_CUBLAS_NAME = "libcublas.so";
	const std::string LIB_CUBLASLT_NAME = "libcublasLt.so";
	const std::string LIB_NVRTC_NAME = "libnvrtc.so";
	const std::string LIB_CURAND_NAME = "libcurand.so";
	const std::string LIB_CUSPARSE_NAME = "libcusparse.so";

	const std::vector<std::string> CUDA_SEARCH_PATHS = {
		"/usr/local/cuda/lib64/",
		"/usr/lib/x86_64-linux-gnu/",
		"/usr/local/cuda/lib/",
	};
	
	auto tally_home_dir = get_tally_home_dir();
	auto lib_nccl_path = tally_home_dir / "third_party/nccl/build/lib/libnccl.so";
	nccl_handle = dlopen(lib_nccl_path.string().c_str(), RTLD_LAZY);

	REGISTER_HANDLE(cuda_handle, LIB_CUDA_NAME);
	REGISTER_HANDLE(cudart_handle, LIB_CUDART_NAME);
	REGISTER_HANDLE(cudnn_handle, LIB_CUDNN_NAME);
	REGISTER_HANDLE(cublas_handle, LIB_CUBLAS_NAME);
	REGISTER_HANDLE(cublasLt_handle, LIB_CUBLASLT_NAME);
	REGISTER_HANDLE(nvrtc_handle, LIB_NVRTC_NAME);
	REGISTER_HANDLE(curand_handle, LIB_CURAND_NAME);
	REGISTER_HANDLE(cusparse_handle, LIB_CUSPARSE_NAME);
}

"""

API_DEF_TEMPLATE_BUTTOM = """

void (*l__cudaRegisterFunction) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)
    = (void (*) (void **, const char *, char *, const char *, int , uint3 *, uint3 *, dim3 *, dim3 *, int *)) dlsym(cudart_handle, "__cudaRegisterFunction");

void** (*l__cudaRegisterFatBinary) (void *) = 
    (void** (*) (void *)) dlsym(cudart_handle, "__cudaRegisterFatBinary");

void (*l__cudaRegisterFatBinaryEnd) (void **) =
	(void (*) (void **)) dlsym(cudart_handle, "__cudaRegisterFatBinaryEnd");

unsigned (*l__cudaPushCallConfiguration)(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream) = 
	(unsigned (*) (dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream)) dlsym(cudart_handle, "__cudaPushCallConfiguration");

cudaError_t (*l__cudaPopCallConfiguration)(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) = 
	(cudaError_t (*) (dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream)) dlsym(cudart_handle, "__cudaPopCallConfiguration");
    
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
#include <nccl.h>
#include <curand.h>
#include <cusparse_v2.h>


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
#include <nccl.h>
#include <curand.h>
#include <cusparse_v2.h>

#include <tally/consts.h>
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
#include <cfloat>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>
#include <nccl.h>
#include <curand.h>
#include <cusparse_v2.h>

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

class ClientData {

public:
	int32_t client_id;

	// For registering kernels at the start:
    unsigned long long* fatbin_data = nullptr;
    uint32_t fatBinSize;
    bool cubin_registered = false;
	uint32_t cubin_uid = 0;
    std::vector<std::pair<void *, std::string>> register_queue;

	// Following are used at runtime:

	// Used to check whether an address points to device memory
	std::vector<mem_region> dev_addr_map;

	std::unordered_map<const void *, const void *> _kernel_client_addr_mapping;

	moodycamel::ReaderWriterQueue<KernelLaunchWrapper> kernel_dispatch_queue;
	std::atomic<uint32_t> queue_size = 0;

	uint32_t *curr_idx_arr;

    cudaStream_t default_stream = nullptr;
	std::atomic<bool> has_exit = false;

	std::vector<cudaStream_t> streams;
	std::map<cudaStream_t, PTBKernelArgs*> stream_to_ptb_args;
};

struct ClientPriority {

	int32_t client_id;
	int32_t priority;

	bool operator>(const ClientPriority& other) const {

		if (priority != other.priority) {
			return priority > other.priority;
		}

		return client_id > other.client_id;
    }

};

class TallyServer {

public:

    static TallyServer *server;

	bool signal_exit = false;

	// ================== Per-client state ===================
	std::map<int32_t, ClientData> client_data_all;
	std::map<ClientPriority, int32_t, std::greater<ClientPriority>> client_priority_map;

    std::map<int32_t, iox::popo::UntypedServer *> worker_servers;
	std::map<int32_t, std::atomic<bool>> threads_running_map;
    
	// ==================== Global state =====================
	std::unordered_map<CUDA_API_ENUM, std::function<void(void *, iox::popo::UntypedServer *, const void* const)>> cuda_api_handler_map;

	// Map CUfunction to host func, similar to _kernel_client_addr_mapping
	folly::ConcurrentHashMap<CUfunction, const void *> cu_func_addr_mapping;

	folly::ConcurrentHashMap<uint32_t, CUmodule> cubin_to_cu_module;

	// CUmodule: { cubin_data_ptr, cubin_size, cubin_uid }
	folly::ConcurrentHashMap<CUmodule, std::tuple<const char *, size_t, uint32_t>> jit_module_to_cubin_map;

	// CUmodule : 
	std::unordered_map<CUmodule, std::unordered_map<std::string, CUfunction>> jit_module_to_function_map;

	folly::ConcurrentHashMap<const void *, std::vector<uint32_t>> _kernel_addr_to_args;

	// Map CUfunction to kernel name and cubin hash
	folly::ConcurrentHashMap<CUfunction, std::string> cu_func_to_kernel_name_map;
	folly::ConcurrentHashMap<CUfunction, uint32_t> cu_func_to_cubin_uid_map;

	// Map func addr to kernel name and cubin hash
	folly::ConcurrentHashMap<const void *, std::string> host_func_to_demangled_kernel_name_map;
	folly::ConcurrentHashMap<const void *, uint32_t> host_func_to_cubin_uid_map;

	// Map kernel name and cubin hash to a host func
	folly::ConcurrentHashMap<std::pair<std::string, uint32_t>, const void *> kernel_cubin_uid_to_host_func_map;

	// Use cubin as unique id of a kernel
	// { Cubin uid: { Kernel Name: host func addr } }
	std::map<uint32_t, folly::ConcurrentHashMap<std::string, const void *>> cubin_to_kernel_name_to_host_func_map;

	// Register original and transformed kernels here
	folly::ConcurrentHashMap<const void *, WrappedCUfunction> original_kernel_map;
    folly::ConcurrentHashMap<const void *, WrappedCUfunction> ptb_kernel_map;
	folly::ConcurrentHashMap<const void *, WrappedCUfunction> dynamic_ptb_kernel_map;
	folly::ConcurrentHashMap<const void *, WrappedCUfunction> preemptive_ptb_kernel_map;
	folly::ConcurrentHashMap<const void *, WrappedCUfunction> sliced_kernel_map;

	// Performance cache to use at runtime
	std::unordered_map<CudaLaunchCallConfig, CudaLaunchCallConfigResult> single_kernel_perf_map;
	std::unordered_map<CudaLaunchCall, CudaLaunchCallConfigResult> single_kernel_chosen_config_map;

	int32_t get_client_priority(int32_t client_id);
	int32_t get_client_stream_priority(int32_t client_id);

	void tune_kernel_launch(KernelLaunchWrapper &kernel_wrapper, int32_t client_id, std::vector<CudaLaunchConfig> &configs);

	void launch_and_measure_kernel(KernelLaunchWrapper &kernel_wrapper, int32_t client_id, std::vector<CudaLaunchConfig> configs,
								   float fallback_threshold, std::vector<CudaLaunchConfig> alternative_configs={}, bool is_preemptive=false);

	void priority_launch_and_measure_kernel(KernelLaunchWrapper &kernel_wrapper, int32_t client_id);
	void priority_launch_and_measure_space_share_kernel(KernelLaunchWrapper &kernel_wrapper, int32_t client_id);

	// Set and Get performance cache
	CudaLaunchCallConfigResult get_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config, bool *found);
	void delete_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config);
	void set_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config,
								CudaLaunchMetadata meta_data, float norm_speed, float latency,
								uint32_t iters=0, float preempt_latency_ms_est=0.);

	CudaLaunchCallConfigResult get_single_kernel_chosen_config(CudaLaunchCall &launch_call, bool *found);
	void set_single_kernel_chosen_config(CudaLaunchCall &launch_call, CudaLaunchCallConfigResult &best_config);
	void clear_single_kernel_chosen_configs();

	// Utility functions for measurement data
	CudaLaunchCall convert_key_to_call(CudaLaunchKey key);
	CudaLaunchKey convert_call_to_key(CudaLaunchCall call);

	CudaLaunchCallConfig convert_key_config_to_call_config(CudaLaunchKeyConfig key_config);
	CudaLaunchKeyConfig convert_call_config_to_key_config(CudaLaunchCallConfig call_config);

    void save_performance_cache();

    static constexpr char APP_NAME[] = "iox-cpp-request-response-server-untyped";

    TallyServer();
    ~TallyServer();

	// Scheduler options
	void run_naive_scheduler();
	void run_priority_scheduler();
	void run_profile_scheduler();
	void run_workload_agnostic_sharing_scheduler();
	void run_workload_aware_sharing_scheduler();

    void register_api_handler();

	const void *get_server_addr_from_client_addr(uint32_t client_id, const void *client_addr);
	const void *get_server_addr_from_cu_func(CUfunction cu_func);

	void register_kernel(const void *addr);
	void register_cu_module(uint32_t cubin_uid);
    void load_measurements(uint32_t cubin_uid);
    void register_ptx_transform(uint32_t cubin_uid);

    void start_scheduler();
    void start_main_server();
    void start_worker_server(int32_t client_id);

	void increment_client_queue_size(int32_t client_id);
	void wait_until_launch_queue_empty(int32_t client_id);
	void client_add_stream(int32_t client_id, cudaStream_t stream);

    // Return a partial function to be scheduled by scheduler
    std::pair<partial_t, void *> cudaLaunchKernel_Partial(const void *, dim3, dim3, size_t, cudaStream_t, char *);
	std::pair<partial_t, void *> cublasSgemm_v2_Partial(cublasSgemm_v2Arg *);
	std::pair<partial_t, void *> cudnnRNNBackwardWeights_Partial(cudnnRNNBackwardWeightsArg *);
	std::pair<partial_t, void *> cudnnRNNBackwardData_Partial(cudnnRNNBackwardDataArg *);
    std::pair<partial_t, void *> cudnnRNNForwardTraining_Partial(cudnnRNNForwardTrainingArg *);
	std::pair<partial_t, void *> cudnnMultiHeadAttnBackwardData_Partial(cudnnMultiHeadAttnBackwardDataArg *);
	std::pair<partial_t, void *> cudnnMultiHeadAttnForward_Partial(cudnnMultiHeadAttnForwardArg *);
	std::pair<partial_t, void *> cublasSgemmEx_Partial(cublasSgemmExArg *);
	std::pair<partial_t, void *> cudnnTransformTensor_Partial(cudnnTransformTensorArg *);
	std::pair<partial_t, void *> cublasSgemv_v2_Partial(cublasSgemv_v2Arg *);
	std::pair<partial_t, void *> cudnnLRNCrossChannelForward_Partial(cudnnLRNCrossChannelForwardArg *);
	std::pair<partial_t, void *> cudnnSoftmaxForward_Partial(cudnnSoftmaxForwardArg *);
	std::pair<partial_t, void *> cudnnAddTensor_Partial(cudnnAddTensorArg *);
	std::pair<partial_t, void *> cublasLtMatmul_Partial(cublasLtMatmulArg *);
	std::pair<partial_t, void *> cudnnActivationForward_Partial(cudnnActivationForwardArg *);
	std::pair<partial_t, void *> cudnnConvolutionForward_Partial(cudnnConvolutionForwardArg *);
	std::pair<partial_t, void *> cudnnPoolingForward_Partial(cudnnPoolingForwardArg *);
	std::pair<partial_t, void *> cudnnMultiHeadAttnBackwardWeights_Partial(cudnnMultiHeadAttnBackwardWeightsArg *);
	std::pair<partial_t, void *> cudnnReorderFilterAndBias_Partial(cudnnReorderFilterAndBiasArg *);
	std::pair<partial_t, void *> cudnnBatchNormalizationForwardTrainingEx_Partial(cudnnBatchNormalizationForwardTrainingExArg *);
	std::pair<partial_t, void *> cudnnBatchNormalizationBackwardEx_Partial(cudnnBatchNormalizationBackwardExArg *);
	std::pair<partial_t, void *> cudnnRNNBackwardWeights_v8_Partial(cudnnRNNBackwardWeights_v8Arg *);
	std::pair<partial_t, void *> cudnnRNNBackwardData_v8_Partial(cudnnRNNBackwardData_v8Arg *);
	std::pair<partial_t, void *> cudnnRNNForward_Partial(cudnnRNNForwardArg *);
	std::pair<partial_t, void *> cudnnBackendExecute_Partial(cudnnBackendExecuteArg *, cudnnStatus_t *err);
	std::pair<partial_t, void *> cublasGemmEx_Partial(cublasGemmExArg *);
	std::pair<partial_t, void *> cublasGemmStridedBatchedEx_Partial(cublasGemmStridedBatchedExArg *);
    std::pair<partial_t, void *> cublasSgemmStridedBatched_Partial(cublasSgemmStridedBatchedArg *);
	std::pair<partial_t, void *> cudnnReduceTensor_Partial(cudnnReduceTensorArg *, void *indices);

"""

TALLY_SERVER_HEADER_TEMPLATE_BUTTOM = """
};

#endif // TALLY_SERVER_H
"""

REGISTER_FUNCS = [
    "__cudaRegisterFatBinary",
    "__cudaRegisterFunction",
    "__cudaRegisterFatBinaryEnd"
]

# just do nothing for these
IGNORE_CALLS = [
    "cuCtxDestroy_v2",
    "cuInit",
    "cuCtxSetCurrent",
]

# implement manually
SPECIAL_CLIENT_PRELOAD_FUNCS = [
    "cudaLaunchHostFunc",
    "cudnnReduceTensor",
    "cuStreamIsCapturing",
    "cuStreamBeginCapture_v2",
    "cudaStreamBeginCapture",
    "cudaStreamIsCapturing",
    "cuGetExportTable",
    "cuGraphInstantiateWithFlags",
    "cudaDeviceGetAttribute",
    "cuCtxCreate_v2",
    "cuCtxGetCurrent",
    "cuCtxGetDevice",
    "cuCtxGetApiVersion",
    "cudaPeekAtLastError",
    "cudaMemsetAsync",
    "cublasLtMatmulPreferenceDestroy",
    "cublasGetMathMode",
    "cublasLtDestroy",
    "cudaGetLastError",
    "cublasLtMatmulDescDestroy",
    "cublasLtMatrixLayoutDestroy",
    "cuDevicePrimaryCtxGetState",
    "cudaGetDevice",
    "ncclCommInitRankConfig",
    "cudnnGetErrorString",
    "cudaFuncGetAttributes",
    "cuDeviceGetName",
    "cudaFreeHost",
    "cuMemHostAlloc",
    "cudaMallocHost",
    "cudaHostAlloc",
    "cuDevicePrimaryCtxSetFlags_v2",
    "cudaPointerGetAttributes",
    "cublasLtMatmulPreferenceCreate",
    "cublasLtMatrixLayoutCreate",
    "cublasLtMatmulDescCreate",
    "cublasLtCreate",
    "cublasSetWorkspace_v2",
    "cublasSetStream_v2",
    "cublasDestroy_v2",
    "cublasSetMathMode",
    "cudaStreamEndCapture",
    "cuGraphLaunch",
    "cuStreamEndCapture",
    "cuModuleUnload",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuModuleLoadDataEx",
    "cuModuleGetGlobal_v2",
    "cuMemcpyDtoDAsync_v2",
    "cuModuleLoadFatBinary",
    "cuMemsetD32Async",
    "cuMemcpyDtoHAsync_v2",
    "cuMemcpyHtoDAsync_v2",
    "cuMemsetD32_v2",
    "cuMemAlloc_v2",
    "cuStreamCreate",
    "cuMemsetD8_v2",
    "cublasGemmStridedBatchedEx",
    "cublasGemmEx",
    "cuStreamCreateWithPriority",
    "cudaStreamCreate",
    "cudaStreamCreateWithFlags",
    "cudaStreamCreateWithPriority",
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
    "cuFuncSetAttribute",
    "cuFuncSetCacheConfig",
    "cuFuncGetAttribute",
    "__cudaRegisterFunction",
    "__cudaRegisterFatBinary",
    "__cudaRegisterFatBinaryEnd"
]

# These api calls can be directly forwarded to the server without addtional logic
# this means no value needs to be assigned
FORWARD_API_CALLS = [
    "cudnnSetReduceTensorDescriptor",
    "ncclAllGather",
    "curandSetPseudoRandomGeneratorSeed",
    "ncclCommAbort",
    "ncclBcast",
    "ncclGroupEnd",
    "ncclGroupStart",
    "ncclCommDestroy",
    "ncclAllReduce",
    "cublasLtMatrixTransformDescDestroy",
    "cuGraphExecDestroy",
    "cuGraphDestroy",
    "cuEventSynchronize",
    "cuEventQuery",
    "cuEventDestroy_v2",
    "cuStreamWaitEvent",
    "cuEventRecord",
    "cudaGraphUpload",
    # "cudaGraphLaunch",
    "cudaGraphExecDestroy",
    "cudaGraphDestroy",
    "cudnnRestoreDropoutDescriptor",
    "cudnnRNNSetClip",
    "cudnnRNNSetClip_v8",
    "cudnnSetRNNMatrixMathType",
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
    "cudaMemPoolTrimTo",
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
    "cublasGetCudartVersion",
    "cuDeviceSetMemPool",
    "cuFlushGPUDirectRDMAWrites",
    "cuDevicePrimaryCtxRelease_v2",
    "cuDevicePrimaryCtxReset_v2",
    "cuCtxPushCurrent_v2",
    "cuCtxSetLimit",
    "cuCtxSetCacheConfig",
    "cuCtxSetSharedMemConfig",
    "cuCtxResetPersistingL2Cache",
    "cuCtxDetach",
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
    "cudnnSetRNNBiasMode",
    "cudnnDestroyReduceTensorDescriptor"
]

# API calls that has the first argument set
# by CUDA API call, such as cudaStreamCreate
CUDA_GET_1_PARAM_FUNCS = [
    "cudnnCreateReduceTensorDescriptor",
    "curandCreateGenerator",
    "cudaDriverGetVersion",
    "cudaRuntimeGetVersion",
    "cuDeviceGetPCIBusId",
    "cudaGetDeviceCount",
    "cuThreadExchangeStreamCaptureMode",
    "ncclCommInitRank",
    "ncclGetUniqueId",
    "cudaThreadExchangeStreamCaptureMode",
    "cublasLtMatrixTransformDescCreate",
    "cuEventElapsedTime",
    "cuEventCreate",
    "cuModuleGetLoadingMode",
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
    "cuDeviceGetTexture1DLinearMaxWidth",
    "cuDeviceGetExecAffinitySupport",
    "cuCtxPopCurrent_v2",
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
    "ncclCommGetAsyncError",
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
    "cublasGetVersion_v2",
    "cudnnGetStream",
    "cudnnBackendCreateDescriptor",
    "cublasGetStream_v2",
    "cublasGetPointerMode_v2",
    "cudnnGetProperty",
    "cudnnGetTensorSizeInBytes"
]

CUDA_GET_2_3_PARAM_FUNCS = [
    "cudaStreamGetCaptureInfo"
]

CUDA_GET_1_2_PARAM_FUNCS = [
    "cuMemGetInfo_v2",
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
    "cuGraphExecUpdate_v2",
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

CUDA_GET_5_PARAM_FUNCS = [
    "cudnnGetReductionIndicesSize",
    "cudnnGetReductionWorkspaceSize"
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
CUDA_GET_5_PARAM_FUNC_KEY = 17

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
    CUDA_GET_2_3_4_5_6_PARAM_FUNC_KEY: [1, 2, 3, 4, 5],
    CUDA_GET_5_PARAM_FUNC_KEY: [4],
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
        CUDA_GET_2_3_4_5_6_PARAM_FUNCS,
        CUDA_GET_5_PARAM_FUNCS
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
    elif func_name in CUDA_GET_5_PARAM_FUNCS:
        return CUDA_GET_5_PARAM_FUNC_KEY
    else:
        assert(False)


def get_preload_func_template_iox(func_name, arg_names, arg_types, ret_type):
    func_preload_builder = f"""
    {ret_type} err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof({func_name}Arg), alignof(MessageHeader_t))
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


def should_check_cuda_err(ret_type, func_name):
    if ret_type not in ["cudaError_t", "CUresult", "cudnnStatus_t", "cublasStatus_t"]:
        return False
    
    if "eventquery" in func_name.lower():
        return False
    
    if func_name == "cudnnBackendFinalize":
        return False
    
    return True