
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

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <nvrtc.h>
#include <cublasLt.h>

#include "iceoryx_dust/posix_wrapper/signal_watcher.hpp"
#include "iceoryx_posh/popo/untyped_server.hpp"
#include "iceoryx_posh/runtime/posh_runtime.hpp"

#include "concurrentqueue.h"

#include <tally/log.h>
#include <tally/msg_struct.h>
#include <tally/cuda_util.h>

static std::function<void(int)> __exit;

static void __exit_wrapper(int signal) {
    __exit(signal);
}

class TallyServer {

public:

    static TallyServer *server;

    int magic;
    int version;
    unsigned long long* fatbin_data = nullptr;
    uint32_t fatBinSize;
    bool cubin_registered = false;

    std::atomic<bool> is_quit__ {false};
    std::map<std::string, void *> _kernel_name_to_addr;
    std::vector<std::pair<void *, std::string>> register_queue;

    // Used to check whether an address points to device memory
	std::vector<DeviceMemoryKey> dev_addr_map;

    std::unordered_map<void *, std::vector<uint32_t>> _kernel_addr_to_args;
    std::unordered_map<void *, void *> _kernel_client_addr_mapping;
    std::unordered_map<CUDA_API_ENUM, std::function<void(void *, const void* const)>> cuda_api_handler_map;

    moodycamel::ConcurrentQueue<std::function<void()>> launch_queue;

    static constexpr char APP_NAME[] = "iox-cpp-request-response-server-untyped";
	iox::popo::UntypedServer *iox_server;

    TallyServer();
    ~TallyServer();

    void start(uint32_t interval);
    void register_api_handler();
    void load_cache();

    std::function<void()> cudaLaunchKernel_Partial(const void *, dim3, dim3, size_t, cudaStream_t, char *);
    std::function<void()> cublasSgemm_v2_Partial(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);
    std::function<void()> cudnnRNNBackwardWeights_Partial(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int  seqLength, cudnnTensorDescriptor_t * xDesc, void * x, cudnnTensorDescriptor_t  hxDesc, void * hx, cudnnTensorDescriptor_t * yDesc, void * y, void * workSpace, size_t  workSpaceSizeInBytes, cudnnFilterDescriptor_t  dwDesc, void * dw, void * reserveSpace, size_t  reserveSpaceSizeInBytes);
    std::function<void()> cudnnRNNBackwardData_Partial(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes);
    std::function<void()> cudnnRNNForwardTraining_Partial(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes);
    std::function<void()> cudnnMultiHeadAttnBackwardData_Partial (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsDQDO[], const int  devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t  doDesc, const void * dout, const cudnnSeqDataDescriptor_t  dqDesc, void * dqueries, const void * queries, const cudnnSeqDataDescriptor_t  dkDesc, void * dkeys, const void * keys, const cudnnSeqDataDescriptor_t  dvDesc, void * dvalues, const void * values, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace, int winIdxLen);
    std::function<void()> cudnnMultiHeadAttnForward_Partial (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, int  currIdx, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsQO[], const int  devSeqLengthsKV[], const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const void * residuals, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  oDesc, void * out, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace, int winIdxLen);
    std::function<void()> cublasSgemmEx_Partial (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const float  beta, void*  C, cudaDataType  Ctype, int  ldc);
    std::function<void()> cudnnTransformTensor_Partial (cudnnHandle_t  handle, uint64_t alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, uint64_t beta, const cudnnTensorDescriptor_t  yDesc, void * y);
"""

TALLY_SERVER_HEADER_TEMPLATE_BUTTOM = """
};

#endif // TALLY_SERVER_H
"""

TALLY_CLIENT_SRC_TEMPLATE_TOP = f"""
#include <cstring>
#include <memory>

#include <tally/client.h>
#include <tally/generated/cuda_api.h>

TallyClient *TallyClient::client = new TallyClient;

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
    "cudaGetErrorString",
    "cuGetProcAddress",
    "cuGetErrorString",
    "cuGetErrorName",
    "cudnnGetErrorString",
    "cudaMallocHost",
    "cudaFreeHost",
    "cudaGetDevice",
    "cudaGetDeviceCount"
]

# implement manually
SPECIAL_CLIENT_PRELOAD_FUNCS = [
    "cudaSetDevice",
    "cudaChooseDevice",
    "cudaFuncGetAttributes",
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
    "cudnnRestoreDropoutDescriptor",
    "cudnnRNNSetClip",
    "cudnnRNNSetClip_v8",
    "cudnnSetRNNMatrixMathType",
    "cudaMemsetAsync",
    "cublasSetSmCountTarget",
    "cublasSetLoggerCallback",
    "cudnnGetFoldedConvBackwardDataDescriptors",
    "cudnnSetRNNAlgorithmDescriptor",
    "cudnnRNNBackwardWeights_v8",
    "cudnnRNNBackwardData_v8",
    "cudnnRNNForward",
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
    "cuMemFree_v2",
    "cudaMemset",
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
    "cuInit",
    "cudaDeviceReset",
    "cudaDeviceSynchronize",
    "cudaDeviceSetLimit",
    "cudaDeviceSetCacheConfig",
    "cudaDeviceSetSharedMemConfig",
    "cudaThreadExit",
    "cudaThreadSynchronize",
    "cudaThreadSetLimit",
    "cudaThreadSetCacheConfig",
    "cudaGetLastError",
    "cudaPeekAtLastError",
    "cudaSetDeviceFlags",
    "cudaCtxResetPersistingL2Cache",
    "cudaEventRecord",
    "cudaEventRecordWithFlags",
    "cudaEventQuery",
    "cudaEventSynchronize",
    "cudaEventDestroy",
    "cudaDeviceSetMemPool",
    "cudaStreamCopyAttributes",
    "cudaStreamDestroy",
    "cudaStreamWaitEvent",
    "cudaStreamSynchronize",
    "cudaStreamQuery",
    "cudaStreamBeginCapture",
    "cudaStreamEndCapture",
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
    "cuCtxDestroy_v2",
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
    "cudnnBackendExecute",
    "cublasSetPointerMode_v2",
    "cudnnDestroyFilterDescriptor",
    "cudnnGetVersion",
    "cudnnGetMaxDeviceVersion",
    "cudnnGetCudartVersion",
    "cudnnDestroyTensorDescriptor",
    "cuMemcpyAsync",
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
    "cudaDeviceGetAttribute",
    "cudaDeviceGetDefaultMemPool",
    "cudaDeviceGetMemPool",
    "cudaDeviceGetP2PAttribute",
    "cudaGetDeviceFlags",
    "cudaGraphCreate",
    "cublasCreate_v2",
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
    "cuCtxCreate_v2",
    "cuCtxPopCurrent_v2",
    "cuCtxGetCurrent",
    "cuCtxGetDevice",
    "cuCtxGetFlags",
    "cuCtxGetLimit",
    "cuCtxGetCacheConfig",
    "cuCtxGetSharedMemConfig",
    "cuCtxGetExecAffinity",
    "cuCtxAttach",
    "cudnnCreate",
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
    CUDA_GET_11_PARAM_FUNC_KEY: [10]
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
        CUDA_GET_11_PARAM_FUNCS
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
    else:
        assert(False)

def get_preload_func_template(func_name, arg_names, arg_types):
    arg_struct = f"{func_name}Arg"

    preload_body = ""
    preload_body += f"""
    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct {arg_struct});

    uint8_t *msg = (msg_len <= TallyClient::msg_size) ? TallyClient::client->msg : (uint8_t *) malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::{func_name.upper()};
    
    struct {arg_struct} *arg_ptr = (struct {arg_struct} *)(msg + sizeof(CUDA_API_ENUM));
"""

    for idx, arg_name in enumerate(arg_names):
        arg_type = arg_types[idx]
        if arg_type.strip() == "const void *" or arg_type.strip() == "const void*":
            preload_body += f"\targ_ptr->{arg_name} = const_cast<void *>({arg_name});\n"
        elif arg_type.strip() == "const int32_t []":
            preload_body += f"\targ_ptr->{arg_name} = const_cast<int32_t *>({arg_name});\n"
        else:
            preload_body += f"\targ_ptr->{arg_name} = {arg_name};\n"
    
    preload_body += "\tCLIENT_SEND_MSG_AND_FREE;\n"
    preload_body += "\tCLIENT_RECV_MSG;\n"

    return preload_body


def get_preload_func_template_iox(func_name, arg_names, arg_types, ret_type):
    func_preload_builder = f"""
    {ret_type} err;

    TallyClient::client->iox_client->loan(sizeof(CUDA_API_ENUM) + sizeof({func_name}Arg), alignof({func_name}Arg))
        .and_then([&](auto& requestPayload) {{

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::{func_name.upper()};
            
            auto request = ({func_name}Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(CUDA_API_ENUM));
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