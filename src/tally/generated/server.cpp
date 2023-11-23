
#include <cstring>
                
#include "spdlog/spdlog.h"

#include <tally/util.h>
#include <tally/ipc_util.h>
#include <tally/msg_struct.h>
#include <tally/generated/cuda_api.h>
#include <tally/generated/msg_struct.h>
#include <tally/generated/server.h>
        
void TallyServer::register_api_handler() {
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDIVISIVENORMALIZATIONFORWARD] = std::bind(&TallyServer::handle_cudnnDivisiveNormalizationForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTPSV_V2_64] = std::bind(&TallyServer::handle_cublasZtpsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESYNCHRONIZE] = std::bind(&TallyServer::handle_cudaDeviceSynchronize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2D] = std::bind(&TallyServer::handle_cudaMemcpy2D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSIGNALEXTERNALSEMAPHORESASYNC] = std::bind(&TallyServer::handle_cuSignalExternalSemaphoresAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETBORDERCOLOR] = std::bind(&TallyServer::handle_cuTexRefSetBorderColor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETF] = std::bind(&TallyServer::handle_cuParamSetf, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDDATAALGORITHM_V7] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardDataAlgorithm_v7, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTGETDEVICEPOINTER] = std::bind(&TallyServer::handle_cudaHostGetDevicePointer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMMEX] = std::bind(&TallyServer::handle_cublasCgemmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIAMAXEX_64] = std::bind(&TallyServer::handle_cublasIamaxEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMIPMAPLEVELCLAMP] = std::bind(&TallyServer::handle_cuTexRefGetMipmapLevelClamp, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNADVINFERVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnAdvInferVersionCheck, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETNORMALIZATIONTRAININGRESERVESPACESIZE] = std::bind(&TallyServer::handle_cudnnGetNormalizationTrainingReserveSpaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGELSBATCHED] = std::bind(&TallyServer::handle_cublasCgelsBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTION2DFORWARDOUTPUTDIM] = std::bind(&TallyServer::handle_cudnnGetConvolution2dForwardOutputDim, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLDESTROY] = std::bind(&TallyServer::handle_cuMemPoolDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNPOOLINGBACKWARD] = std::bind(&TallyServer::handle_cudnnPoolingBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETCOUNT] = std::bind(&TallyServer::handle_cuDeviceGetCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHEMM_V2] = std::bind(&TallyServer::handle_cublasZhemm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCOPYEX_64] = std::bind(&TallyServer::handle_cublasCopyEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYR2K_V2] = std::bind(&TallyServer::handle_cublasCsyr2k_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLGROUPSTART] = std::bind(&TallyServer::handle_pncclGroupStart, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCFROMPOOLASYNC] = std::bind(&TallyServer::handle_cuMemAllocFromPoolAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3M_64] = std::bind(&TallyServer::handle_cublasCgemm3m_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSETCACHECONFIG] = std::bind(&TallyServer::handle_cuCtxSetCacheConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTION2DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetConvolution2dDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNOPSINFERVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnOpsInferVersionCheck, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDSCAL_V2_64] = std::bind(&TallyServer::handle_cublasZdscal_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHER2_V2_64] = std::bind(&TallyServer::handle_cublasCher2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMIPMAPPEDARRAYGETSPARSEPROPERTIES] = std::bind(&TallyServer::handle_cudaMipmappedArrayGetSparseProperties, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULELOADDATAEX] = std::bind(&TallyServer::handle_cuModuleLoadDataEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSPMV_V2] = std::bind(&TallyServer::handle_cublasDspmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUWAITEXTERNALSEMAPHORESASYNC] = std::bind(&TallyServer::handle_cuWaitExternalSemaphoresAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMBEGINCAPTURE] = std::bind(&TallyServer::handle_cudaStreamBeginCapture, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDOTU_V2] = std::bind(&TallyServer::handle_cublasZdotu_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCOPYEX] = std::bind(&TallyServer::handle_cublasCopyEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNADVTRAINVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnAdvTrainVersionCheck, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDROT_V2] = std::bind(&TallyServer::handle_cublasZdrot_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNTRANSFORMFILTER] = std::bind(&TallyServer::handle_cudnnTransformFilter, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYSPATIALTRANSFORMERDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroySpatialTransformerDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETALGORITHMSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetAlgorithmSpaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECGETFLAGS] = std::bind(&TallyServer::handle_cudaGraphExecGetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEFUSEDOPSVARIANTPARAMPACK] = std::bind(&TallyServer::handle_cudnnCreateFusedOpsVariantParamPack, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMV_V2] = std::bind(&TallyServer::handle_cublasSgemv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETPTX] = std::bind(&TallyServer::handle_nvrtcGetPTX, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMMAP] = std::bind(&TallyServer::handle_cuMemMap, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHPMV_V2] = std::bind(&TallyServer::handle_cublasChpmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXTRANSFORMDESCINIT_INTERNAL] = std::bind(&TallyServer::handle_cublasLtMatrixTransformDescInit_internal, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCADDNAMEEXPRESSION] = std::bind(&TallyServer::handle_nvrtcAddNameExpression, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYFROMSYMBOLASYNC] = std::bind(&TallyServer::handle_cudaMemcpyFromSymbolAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM_V2] = std::bind(&TallyServer::handle_cublasCgemm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFREFSETARRAY] = std::bind(&TallyServer::handle_cuSurfRefSetArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETMATRIXASYNC] = std::bind(&TallyServer::handle_cublasGetMatrixAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHGETNODES] = std::bind(&TallyServer::handle_cuGraphGetNodes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEREDUCETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateReduceTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHCLONE] = std::bind(&TallyServer::handle_cuGraphClone, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTSETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatrixLayoutSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMADDCALLBACK] = std::bind(&TallyServer::handle_cuStreamAddCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFOBJECTCREATE] = std::bind(&TallyServer::handle_cuSurfObjectCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDKERNELNODE] = std::bind(&TallyServer::handle_cudaGraphAddKernelNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAPOINTERGETATTRIBUTES] = std::bind(&TallyServer::handle_cudaPointerGetAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYMM_V2_64] = std::bind(&TallyServer::handle_cublasZsymm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETHANDLEFORADDRESSRANGE] = std::bind(&TallyServer::handle_cuMemGetHandleForAddressRange, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPREFETCHASYNC] = std::bind(&TallyServer::handle_cuMemPrefetchAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRSV_V2] = std::bind(&TallyServer::handle_cublasStrsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTIONGROUPCOUNT] = std::bind(&TallyServer::handle_cudnnSetConvolutionGroupCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETVECTOR_64] = std::bind(&TallyServer::handle_cublasSetVector_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY] = std::bind(&TallyServer::handle_cudaMemcpy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTCREATE] = std::bind(&TallyServer::handle_cublasLtMatrixLayoutCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMUNMAP] = std::bind(&TallyServer::handle_cuMemUnmap, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODESETPARAMS1D] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeSetParams1D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSBMV_V2_64] = std::bind(&TallyServer::handle_cublasSsbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETMATRIXASYNC] = std::bind(&TallyServer::handle_cublasSetMatrixAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMVSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasZgemvStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYRKX_64] = std::bind(&TallyServer::handle_cublasZsyrkx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMMSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasCgemmStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3MEX] = std::bind(&TallyServer::handle_cublasCgemm3mEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEAM_64] = std::bind(&TallyServer::handle_cublasSgeam_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODEGETATTRIBUTE] = std::bind(&TallyServer::handle_cuGraphKernelNodeGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASUINT8GEMMBIAS] = std::bind(&TallyServer::handle_cublasUint8gemmBias, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRTTP] = std::bind(&TallyServer::handle_cublasDtrttp, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMGETASYNCERROR] = std::bind(&TallyServer::handle_ncclCommGetAsyncError, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHLAUNCH] = std::bind(&TallyServer::handle_cuGraphLaunch, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNDESCRIPTOR_V6] = std::bind(&TallyServer::handle_cudnnGetRNNDescriptor_v6, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETLOWEREDNAME] = std::bind(&TallyServer::handle_nvrtcGetLoweredName, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMCPYNODE] = std::bind(&TallyServer::handle_cudaGraphAddMemcpyNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYGETDESCRIPTOR_V2] = std::bind(&TallyServer::handle_cuArrayGetDescriptor_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADGETCACHECONFIG] = std::bind(&TallyServer::handle_cudaThreadGetCacheConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTMGEX] = std::bind(&TallyServer::handle_cublasRotmgEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSSUBRESOURCEGETMAPPEDARRAY] = std::bind(&TallyServer::handle_cuGraphicsSubResourceGetMappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDGETATTRIBUTE] = std::bind(&TallyServer::handle_cudnnBackendGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCALEX] = std::bind(&TallyServer::handle_cublasScalEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETREDUCTIONWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetReductionWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATESPATIALTRANSFORMERDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateSpatialTransformerDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDGMM_64] = std::bind(&TallyServer::handle_cublasZdgmm_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULALGOCAPGETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatmulAlgoCapGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTLOGGERFORCEDISABLE] = std::bind(&TallyServer::handle_cublasLtLoggerForceDisable, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLREDUCESCATTER] = std::bind(&TallyServer::handle_ncclReduceScatter, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSCAL_V2_64] = std::bind(&TallyServer::handle_cublasDscal_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNPROJECTIONLAYERS] = std::bind(&TallyServer::handle_cudnnSetRNNProjectionLayers, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYR2_V2_64] = std::bind(&TallyServer::handle_cublasSsyr2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDDATAEX] = std::bind(&TallyServer::handle_cudnnRNNBackwardDataEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMULTICASTGETGRANULARITY] = std::bind(&TallyServer::handle_cuMulticastGetGranularity, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETCACHECONFIG] = std::bind(&TallyServer::handle_cudaDeviceSetCacheConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYR_V2_64] = std::bind(&TallyServer::handle_cublasCsyr_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSAXPY_V2_64] = std::bind(&TallyServer::handle_cublasSaxpy_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIDAMAX_V2] = std::bind(&TallyServer::handle_cublasIdamax_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetRNNWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNREDUCETENSOR] = std::bind(&TallyServer::handle_cudnnReduceTensor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDERIVEBNTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDeriveBNTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADEXIT] = std::bind(&TallyServer::handle_cudaThreadExit, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAARRAYGETINFO] = std::bind(&TallyServer::handle_cudaArrayGetInfo, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cuCtxGetSharedMemConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETTENSORNDDESCRIPTOREX] = std::bind(&TallyServer::handle_cudnnSetTensorNdDescriptorEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRSM_V2] = std::bind(&TallyServer::handle_cublasCtrsm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETPROPERTIES] = std::bind(&TallyServer::handle_cuDeviceGetProperties, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETGRAPHMEMATTRIBUTE] = std::bind(&TallyServer::handle_cuDeviceGetGraphMemAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCUDARTVERSION] = std::bind(&TallyServer::handle_cudnnGetCudartVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDASUM_V2_64] = std::bind(&TallyServer::handle_cublasDasum_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTBMV_V2] = std::bind(&TallyServer::handle_cublasCtbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROTMG_V2] = std::bind(&TallyServer::handle_cublasSrotmg_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSPMV_V2_64] = std::bind(&TallyServer::handle_cublasDspmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERK_V2] = std::bind(&TallyServer::handle_cublasCherk_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGERU_V2_64] = std::bind(&TallyServer::handle_cublasCgeru_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHBMV_V2_64] = std::bind(&TallyServer::handle_cublasZhbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECEVENTWAITNODESETEVENT] = std::bind(&TallyServer::handle_cuGraphExecEventWaitNodeSetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETFLAGS] = std::bind(&TallyServer::handle_cuStreamGetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLGROUPSTART] = std::bind(&TallyServer::handle_ncclGroupStart, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHCHILDGRAPHNODEGETGRAPH] = std::bind(&TallyServer::handle_cudaGraphChildGraphNodeGetGraph, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETTENSORNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetTensorNdDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEDISABLEPEERACCESS] = std::bind(&TallyServer::handle_cudaDeviceDisablePeerAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetRNNDataDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHER_V2_64] = std::bind(&TallyServer::handle_cublasCher_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTPSV_V2_64] = std::bind(&TallyServer::handle_cublasStpsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCNRM2_V2_64] = std::bind(&TallyServer::handle_cublasScnrm2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNMAKEFUSEDOPSPLAN] = std::bind(&TallyServer::handle_cudnnMakeFusedOpsPlan, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMIPMAPPEDARRAYGETLEVEL] = std::bind(&TallyServer::handle_cuMipmappedArrayGetLevel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGBMV_V2] = std::bind(&TallyServer::handle_cublasCgbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICECOMPUTECAPABILITY] = std::bind(&TallyServer::handle_cuDeviceComputeCapability, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOC3DARRAY] = std::bind(&TallyServer::handle_cudaMalloc3DArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETSTATUSNAME] = std::bind(&TallyServer::handle_cublasGetStatusName, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLRECV] = std::bind(&TallyServer::handle_pncclRecv, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLALLREDUCE] = std::bind(&TallyServer::handle_ncclAllReduce, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETSTREAM_V2] = std::bind(&TallyServer::handle_cublasGetStream_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNOPTENSOR] = std::bind(&TallyServer::handle_cudnnOpTensor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDZNRM2_V2_64] = std::bind(&TallyServer::handle_cublasDznrm2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHPR_V2_64] = std::bind(&TallyServer::handle_cublasZhpr_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTHEURISTICSCACHEGETCAPACITY] = std::bind(&TallyServer::handle_cublasLtHeuristicsCacheGetCapacity, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETCACHECONFIG] = std::bind(&TallyServer::handle_cudaDeviceGetCacheConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTPSV_V2_64] = std::bind(&TallyServer::handle_cublasDtpsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGEMMSTRIDEDBATCHEDEX_64] = std::bind(&TallyServer::handle_cublasGemmStridedBatchedEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETLIMIT] = std::bind(&TallyServer::handle_cudaDeviceSetLimit, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASTSSGEMVBATCHED] = std::bind(&TallyServer::handle_cublasTSSgemvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYR2_V2] = std::bind(&TallyServer::handle_cublasCsyr2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTUNREGISTER] = std::bind(&TallyServer::handle_cudaHostUnregister, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSPATIALTFGRIDGENERATORFORWARD] = std::bind(&TallyServer::handle_cudnnSpatialTfGridGeneratorForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROT_V2] = std::bind(&TallyServer::handle_cublasDrot_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYMV_V2] = std::bind(&TallyServer::handle_cublasZsymv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROT_V2_64] = std::bind(&TallyServer::handle_cublasSrot_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHRETAINUSEROBJECT] = std::bind(&TallyServer::handle_cuGraphRetainUserObject, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHEMM_V2] = std::bind(&TallyServer::handle_cublasChemm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETADDRESS2D_V3] = std::bind(&TallyServer::handle_cuTexRefSetAddress2D_v3, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYFUSEDOPSPLAN] = std::bind(&TallyServer::handle_cudnnDestroyFusedOpsPlan, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTBMV_V2] = std::bind(&TallyServer::handle_cublasDtbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTHEURISTICSCACHESETCAPACITY] = std::bind(&TallyServer::handle_cublasLtHeuristicsCacheSetCapacity, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYR_V2] = std::bind(&TallyServer::handle_cublasSsyr_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMADDCALLBACK] = std::bind(&TallyServer::handle_cudaStreamAddCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNDESCRIPTOR_V8] = std::bind(&TallyServer::handle_cudnnSetRNNDescriptor_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICERESET] = std::bind(&TallyServer::handle_cudaDeviceReset, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRSMBATCHED_64] = std::bind(&TallyServer::handle_cublasDtrsmBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYCREATE_V2] = std::bind(&TallyServer::handle_cuArrayCreate_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGETRIBATCHED] = std::bind(&TallyServer::handle_cublasCgetriBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSPATIALTFGRIDGENERATORBACKWARD] = std::bind(&TallyServer::handle_cudnnSpatialTfGridGeneratorBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULALGOCHECK] = std::bind(&TallyServer::handle_cublasLtMatmulAlgoCheck, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFCREATE] = std::bind(&TallyServer::handle_cuTexRefCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLGETUNIQUEID] = std::bind(&TallyServer::handle_pncclGetUniqueId, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMADVISE] = std::bind(&TallyServer::handle_cudaMemAdvise, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASICAMAX_V2] = std::bind(&TallyServer::handle_cublasIcamax_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DTOARRAYASYNC] = std::bind(&TallyServer::handle_cudaMemcpy2DToArrayAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLGETERRORSTRING] = std::bind(&TallyServer::handle_pncclGetErrorString, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACHOOSEDEVICE] = std::bind(&TallyServer::handle_cudaChooseDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYHTOD_V2] = std::bind(&TallyServer::handle_cuMemcpyHtoD_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYMV_V2_64] = std::bind(&TallyServer::handle_cublasDsymv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXRETAIN] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxRetain, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERK3MEX_64] = std::bind(&TallyServer::handle_cublasCherk3mEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDASUM_V2] = std::bind(&TallyServer::handle_cublasDasum_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMVBATCHED] = std::bind(&TallyServer::handle_cublasDgemvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODEFINDINCLONE] = std::bind(&TallyServer::handle_cuGraphNodeFindInClone, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGERU_V2] = std::bind(&TallyServer::handle_cublasZgeru_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMSYNCHRONIZE] = std::bind(&TallyServer::handle_cuStreamSynchronize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYRKX_64] = std::bind(&TallyServer::handle_cublasDsyrkx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCDGMM_64] = std::bind(&TallyServer::handle_cublasCdgmm_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCTCLOSS_V8] = std::bind(&TallyServer::handle_cudnnCTCLoss_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECEVENTWAITNODESETEVENT] = std::bind(&TallyServer::handle_cudaGraphExecEventWaitNodeSetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAINITDEVICE] = std::bind(&TallyServer::handle_cudaInitDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAUSEROBJECTRELEASE] = std::bind(&TallyServer::handle_cudaUserObjectRelease, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCCLOSEMEMHANDLE] = std::bind(&TallyServer::handle_cuIpcCloseMemHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCOPENEVENTHANDLE] = std::bind(&TallyServer::handle_cudaIpcOpenEventHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETLOGGERCALLBACK] = std::bind(&TallyServer::handle_cublasGetLoggerCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMMSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasZgemmStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETFOLDEDCONVBACKWARDDATADESCRIPTORS] = std::bind(&TallyServer::handle_cudnnGetFoldedConvBackwardDataDescriptors, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMCPYNODESETPARAMSTOSYMBOL] = std::bind(&TallyServer::handle_cudaGraphExecMemcpyNodeSetParamsToSymbol, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGELSBATCHED] = std::bind(&TallyServer::handle_cublasSgelsBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYOPTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyOpTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTBSV_V2_64] = std::bind(&TallyServer::handle_cublasZtbsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCOPENMEMHANDLE] = std::bind(&TallyServer::handle_cudaIpcOpenMemHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROTM_V2_64] = std::bind(&TallyServer::handle_cublasDrotm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHER_V2_64] = std::bind(&TallyServer::handle_cublasZher_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMMSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasSgemmStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSDESCRIPTOR_V8] = std::bind(&TallyServer::handle_cudnnGetCTCLossDescriptor_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYMM_V2] = std::bind(&TallyServer::handle_cublasCsymm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRSMBATCHED] = std::bind(&TallyServer::handle_cublasStrsmBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYRK_V2_64] = std::bind(&TallyServer::handle_cublasDsyrk_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETFLAGS] = std::bind(&TallyServer::handle_cuCtxGetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETALLOCATIONPROPERTIESFROMHANDLE] = std::bind(&TallyServer::handle_cuMemGetAllocationPropertiesFromHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSPR_V2_64] = std::bind(&TallyServer::handle_cublasSspr_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULALGOINIT] = std::bind(&TallyServer::handle_cublasLtMatmulAlgoInit, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXPOTENTIALBLOCKSIZEWITHFLAGS] = std::bind(&TallyServer::handle_cuOccupancyMaxPotentialBlockSizeWithFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFUNCSETCACHECONFIG] = std::bind(&TallyServer::handle_cudaFuncSetCacheConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODEGETTYPE] = std::bind(&TallyServer::handle_cuGraphNodeGetType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETDEVICEFLAGS] = std::bind(&TallyServer::handle_cudaSetDeviceFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZROT_V2] = std::bind(&TallyServer::handle_cublasZrot_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERKEX_64] = std::bind(&TallyServer::handle_cublasCherkEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMVBATCHED_64] = std::bind(&TallyServer::handle_cublasZgemvBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMCPYNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemcpyNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTGETRESOURCEDESC] = std::bind(&TallyServer::handle_cuTexObjectGetResourceDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASTSTGEMVBATCHED] = std::bind(&TallyServer::handle_cublasTSTgemvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasHgemmStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRMM_V2_64] = std::bind(&TallyServer::handle_cublasCtrmm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASAXPYEX_64] = std::bind(&TallyServer::handle_cublasAxpyEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMIPMAPPEDARRAYCREATE] = std::bind(&TallyServer::handle_cuMipmappedArrayCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLBCAST] = std::bind(&TallyServer::handle_ncclBcast, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASXERBLA] = std::bind(&TallyServer::handle_cublasXerbla, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACTXRESETPERSISTINGL2CACHE] = std::bind(&TallyServer::handle_cudaCtxResetPersistingL2Cache, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMMAPARRAYASYNC] = std::bind(&TallyServer::handle_cuMemMapArrayAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETTEXREF] = std::bind(&TallyServer::handle_cuParamSetTexRef, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLSETACCESS] = std::bind(&TallyServer::handle_cudaMemPoolSetAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEAM_64] = std::bind(&TallyServer::handle_cublasZgeam_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGBMV_V2] = std::bind(&TallyServer::handle_cublasSgbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCGETEVENTHANDLE] = std::bind(&TallyServer::handle_cudaIpcGetEventHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDHOSTNODE] = std::bind(&TallyServer::handle_cuGraphAddHostNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLBROADCAST] = std::bind(&TallyServer::handle_ncclBroadcast, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCALLBACK] = std::bind(&TallyServer::handle_cudnnSetCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRMM_V2] = std::bind(&TallyServer::handle_cublasCtrmm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSPATIALTFSAMPLERBACKWARD] = std::bind(&TallyServer::handle_cudnnSpatialTfSamplerBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAY3DCREATE_V2] = std::bind(&TallyServer::handle_cuArray3DCreate_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHPR_V2_64] = std::bind(&TallyServer::handle_cublasChpr_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGBMV_V2] = std::bind(&TallyServer::handle_cublasDgbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDNRM2_V2_64] = std::bind(&TallyServer::handle_cublasDnrm2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEENABLEPEERACCESS] = std::bind(&TallyServer::handle_cudaDeviceEnablePeerAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMM_V2_64] = std::bind(&TallyServer::handle_cublasZgemm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasSgemmStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDRNNFORWARDTRAININGALGORITHMEX] = std::bind(&TallyServer::handle_cudnnFindRNNForwardTrainingAlgorithmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXRESET_V2] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxReset_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3MBATCHED] = std::bind(&TallyServer::handle_cublasCgemm3mBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNLRNCROSSCHANNELBACKWARD] = std::bind(&TallyServer::handle_cudnnLRNCrossChannelBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYGETSPARSEPROPERTIES] = std::bind(&TallyServer::handle_cuArrayGetSparseProperties, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCNRM2_V2] = std::bind(&TallyServer::handle_cublasScnrm2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHER2K_V2_64] = std::bind(&TallyServer::handle_cublasZher2k_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUIMPORTEXTERNALSEMAPHORE] = std::bind(&TallyServer::handle_cuImportExternalSemaphore, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXTERNALSEMAPHORESWAITNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExternalSemaphoresWaitNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSNRM2_V2] = std::bind(&TallyServer::handle_cublasSnrm2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTDESTROY] = std::bind(&TallyServer::handle_cublasLtMatrixLayoutDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHERKX] = std::bind(&TallyServer::handle_cublasZherkx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAPROFILERSTOP] = std::bind(&TallyServer::handle_cudaProfilerStop, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMINITRANK] = std::bind(&TallyServer::handle_ncclCommInitRank, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADSETCACHECONFIG] = std::bind(&TallyServer::handle_cudaThreadSetCacheConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRSM_V2] = std::bind(&TallyServer::handle_cublasStrsm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRKEX_64] = std::bind(&TallyServer::handle_cublasCsyrkEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEVENTRECORDNODEGETEVENT] = std::bind(&TallyServer::handle_cuGraphEventRecordNodeGetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDHOSTNODE] = std::bind(&TallyServer::handle_cudaGraphAddHostNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D16_V2] = std::bind(&TallyServer::handle_cuMemsetD2D16_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphKernelNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEVENTRECORDNODEGETEVENT] = std::bind(&TallyServer::handle_cudaGraphEventRecordNodeGetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCTCLOSSDESCRIPTOR_V8] = std::bind(&TallyServer::handle_cudnnSetCTCLossDescriptor_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETEXECAFFINITYSUPPORT] = std::bind(&TallyServer::handle_cuDeviceGetExecAffinitySupport, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCGETATTRIBUTE] = std::bind(&TallyServer::handle_cuFuncGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNMATRIXMATHTYPE] = std::bind(&TallyServer::handle_cudnnSetRNNMatrixMathType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPROPERTY] = std::bind(&TallyServer::handle_cudnnGetProperty, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTREGISTER] = std::bind(&TallyServer::handle_cudaHostRegister, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDCONVOLUTIONFORWARDALGORITHMEX] = std::bind(&TallyServer::handle_cudnnFindConvolutionForwardAlgorithmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTHREADEXCHANGESTREAMCAPTUREMODE] = std::bind(&TallyServer::handle_cuThreadExchangeStreamCaptureMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTLOGGERSETFILE] = std::bind(&TallyServer::handle_cublasLtLoggerSetFile, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYMM_V2] = std::bind(&TallyServer::handle_cublasZsymm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOC_V2] = std::bind(&TallyServer::handle_cuMemAlloc_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHERK_V2] = std::bind(&TallyServer::handle_cublasZherk_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXTRANSFORM] = std::bind(&TallyServer::handle_cublasLtMatrixTransform, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODEGETPARAMS_V2] = std::bind(&TallyServer::handle_cuGraphKernelNodeGetParams_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPREFETCHASYNC_V2] = std::bind(&TallyServer::handle_cuMemPrefetchAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDALGORITHM_V7] = std::bind(&TallyServer::handle_cudnnGetConvolutionForwardAlgorithm_v7, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSUNREGISTERRESOURCE] = std::bind(&TallyServer::handle_cuGraphicsUnregisterResource, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTBMV_V2_64] = std::bind(&TallyServer::handle_cublasDtbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY3DPEER] = std::bind(&TallyServer::handle_cudaMemcpy3DPeer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRTTP] = std::bind(&TallyServer::handle_cublasZtrttp, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKADDFILE_V2] = std::bind(&TallyServer::handle_cuLinkAddFile_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHER2K_V2] = std::bind(&TallyServer::handle_cublasCher2k_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRSV_V2] = std::bind(&TallyServer::handle_cublasDtrsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCGETMEMHANDLE] = std::bind(&TallyServer::handle_cudaIpcGetMemHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEOPTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateOpTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMCPYNODETOSYMBOL] = std::bind(&TallyServer::handle_cudaGraphAddMemcpyNodeToSymbol, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYR_V2] = std::bind(&TallyServer::handle_cublasZsyr_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHCHILDGRAPHNODEGETGRAPH] = std::bind(&TallyServer::handle_cuGraphChildGraphNodeGetGraph, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETMATRIX] = std::bind(&TallyServer::handle_cublasSetMatrix, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETSYMBOLSIZE] = std::bind(&TallyServer::handle_cudaGetSymbolSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWAITVALUE64_V2] = std::bind(&TallyServer::handle_cuStreamWaitValue64_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETSYMBOLADDRESS] = std::bind(&TallyServer::handle_cudaGetSymbolAddress, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETFORMAT] = std::bind(&TallyServer::handle_cuTexRefGetFormat, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGERC_V2] = std::bind(&TallyServer::handle_cublasZgerc_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECHOSTNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecHostNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSETFLAGS] = std::bind(&TallyServer::handle_cuCtxSetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETACCESS] = std::bind(&TallyServer::handle_cuMemSetAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaDeviceGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREEASYNC] = std::bind(&TallyServer::handle_cudaFreeAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNPADDINGMODE] = std::bind(&TallyServer::handle_cudnnSetRNNPaddingMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMMBATCHED_64] = std::bind(&TallyServer::handle_cublasCgemmBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETATTRIBUTE] = std::bind(&TallyServer::handle_cuStreamGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETI] = std::bind(&TallyServer::handle_cuParamSeti, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYPEERASYNC] = std::bind(&TallyServer::handle_cudaMemcpyPeerAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMFREENODE] = std::bind(&TallyServer::handle_cudaGraphAddMemFreeNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDEVICEFLAGS] = std::bind(&TallyServer::handle_cudaGetDeviceFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetConvolutionNdDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYLRNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyLRNDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSCAL_V2_64] = std::bind(&TallyServer::handle_cublasZscal_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEVENTWAITNODESETEVENT] = std::bind(&TallyServer::handle_cudaGraphEventWaitNodeSetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAY3DGETDESCRIPTOR_V2] = std::bind(&TallyServer::handle_cuArray3DGetDescriptor_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULIBRARYGETMANAGED] = std::bind(&TallyServer::handle_cuLibraryGetManaged, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::__CUDAREGISTERFATBINARYEND] = std::bind(&TallyServer::handle___cudaRegisterFatBinaryEnd, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETSURFACEOBJECTRESOURCEDESC] = std::bind(&TallyServer::handle_cudaGetSurfaceObjectResourceDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDDEPENDENCIES] = std::bind(&TallyServer::handle_cuGraphAddDependencies, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTPMV_V2] = std::bind(&TallyServer::handle_cublasStpmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWAITEVENT] = std::bind(&TallyServer::handle_cuStreamWaitEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMRANGEGETATTRIBUTE] = std::bind(&TallyServer::handle_cuMemRangeGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYMV_V2_64] = std::bind(&TallyServer::handle_cublasSsymv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMADVISE_V2] = std::bind(&TallyServer::handle_cudaMemAdvise_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETLIMIT] = std::bind(&TallyServer::handle_cuCtxGetLimit, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHRETAINUSEROBJECT] = std::bind(&TallyServer::handle_cudaGraphRetainUserObject, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMM_V2] = std::bind(&TallyServer::handle_cublasZgemm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETNVSCISYNCATTRIBUTES] = std::bind(&TallyServer::handle_cudaDeviceGetNvSciSyncAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETPROPERTY] = std::bind(&TallyServer::handle_cublasGetProperty, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMCPYNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemcpyNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETLIMIT] = std::bind(&TallyServer::handle_cudaDeviceGetLimit, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyAlgorithmDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHEMV_V2_64] = std::bind(&TallyServer::handle_cublasZhemv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMIPMAPLEVELBIAS] = std::bind(&TallyServer::handle_cuTexRefGetMipmapLevelBias, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETERRORSTRING] = std::bind(&TallyServer::handle_cudnnGetErrorString, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTPMV_V2] = std::bind(&TallyServer::handle_cublasZtpmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSBMV_V2] = std::bind(&TallyServer::handle_cublasDsbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMIPMAPPEDARRAYDESTROY] = std::bind(&TallyServer::handle_cuMipmappedArrayDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSWAP_V2_64] = std::bind(&TallyServer::handle_cublasZswap_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDGMM] = std::bind(&TallyServer::handle_cublasZdgmm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYGETPLANE] = std::bind(&TallyServer::handle_cuArrayGetPlane, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D32_V2] = std::bind(&TallyServer::handle_cuMemsetD2D32_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSWAP_V2_64] = std::bind(&TallyServer::handle_cublasDswap_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRSV_V2_64] = std::bind(&TallyServer::handle_cublasStrsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIAMINEX] = std::bind(&TallyServer::handle_cublasIaminEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DFROMARRAY] = std::bind(&TallyServer::handle_cudaMemcpy2DFromArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIZAMIN_V2_64] = std::bind(&TallyServer::handle_cublasIzamin_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETTENSORTRANSFORMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetTensorTransformDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTCREATE] = std::bind(&TallyServer::handle_cudaEventCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMMBATCHED_64] = std::bind(&TallyServer::handle_cublasSgemmBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYATOHASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyAtoHAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETMEMPOOL] = std::bind(&TallyServer::handle_cudaDeviceGetMemPool, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDOTEX_64] = std::bind(&TallyServer::handle_cublasDotEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DARRAYTOARRAY] = std::bind(&TallyServer::handle_cudaMemcpy2DArrayToArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSUNREGISTERRESOURCE] = std::bind(&TallyServer::handle_cudaGraphicsUnregisterResource, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHDEBUGDOTPRINT] = std::bind(&TallyServer::handle_cuGraphDebugDotPrint, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATERNNDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateRNNDataDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMRANGEGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaMemRangeGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHSSGEMVSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasHSSgemvStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCTCLOSS] = std::bind(&TallyServer::handle_cudnnCTCLoss, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNTRAININGRESERVESIZE] = std::bind(&TallyServer::handle_cudnnGetRNNTrainingReserveSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTRECORDWITHFLAGS] = std::bind(&TallyServer::handle_cuEventRecordWithFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEAM] = std::bind(&TallyServer::handle_cublasDgeam, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGETERRORSTRING] = std::bind(&TallyServer::handle_cuGetErrorString, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATETENSORTRANSFORMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateTensorTransformDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMINITALL] = std::bind(&TallyServer::handle_ncclCommInitAll, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDOTC_V2] = std::bind(&TallyServer::handle_cublasZdotc_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYR_V2_64] = std::bind(&TallyServer::handle_cublasSsyr_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMMEX] = std::bind(&TallyServer::handle_cublasSgemmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMFINALIZE] = std::bind(&TallyServer::handle_ncclCommFinalize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROTG_V2] = std::bind(&TallyServer::handle_cublasDrotg_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYR_V2_64] = std::bind(&TallyServer::handle_cublasZsyr_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSPR2_V2_64] = std::bind(&TallyServer::handle_cublasDspr2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPOINTERSETATTRIBUTE] = std::bind(&TallyServer::handle_cuPointerSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETDOUBLEFORHOST] = std::bind(&TallyServer::handle_cudaSetDoubleForHost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLALLGATHER] = std::bind(&TallyServer::handle_ncclAllGather, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDFILTERWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardFilterWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCCOPY_V2_64] = std::bind(&TallyServer::handle_cublasCcopy_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCASUM_V2] = std::bind(&TallyServer::handle_cublasScasum_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETID] = std::bind(&TallyServer::handle_cudaStreamGetId, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRMV_V2] = std::bind(&TallyServer::handle_cublasDtrmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRMM_V2_64] = std::bind(&TallyServer::handle_cublasDtrmm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECKERNELNODESETPARAMS_V2] = std::bind(&TallyServer::handle_cuGraphExecKernelNodeSetParams_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHCREATE] = std::bind(&TallyServer::handle_cuGraphCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDMEMALLOCNODE] = std::bind(&TallyServer::handle_cuGraphAddMemAllocNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETFILTERMODE] = std::bind(&TallyServer::handle_cuTexRefGetFilterMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYR2_V2] = std::bind(&TallyServer::handle_cublasDsyr2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXRELEASE_V2] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxRelease_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKCREATE_V2] = std::bind(&TallyServer::handle_cuLinkCreate_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODEGETENABLED] = std::bind(&TallyServer::handle_cuGraphNodeGetEnabled, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRMV_V2_64] = std::bind(&TallyServer::handle_cublasZtrmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDDGMM_64] = std::bind(&TallyServer::handle_cublasDdgmm_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERKEX] = std::bind(&TallyServer::handle_cublasCherkEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEVENTRECORDNODE] = std::bind(&TallyServer::handle_cuGraphAddEventRecordNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONMATHTYPE] = std::bind(&TallyServer::handle_cudnnGetConvolutionMathType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMV_V2] = std::bind(&TallyServer::handle_cublasDgemv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMFINALIZE] = std::bind(&TallyServer::handle_pncclCommFinalize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGBMV_V2_64] = std::bind(&TallyServer::handle_cublasDgbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULALGOGETIDS] = std::bind(&TallyServer::handle_cublasLtMatmulAlgoGetIds, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCREATE] = std::bind(&TallyServer::handle_cuMemCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETCTX] = std::bind(&TallyServer::handle_cuStreamGetCtx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETSTREAMPRIORITYRANGE] = std::bind(&TallyServer::handle_cuCtxGetStreamPriorityRange, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASAXPYEX] = std::bind(&TallyServer::handle_cublasAxpyEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMM_V2] = std::bind(&TallyServer::handle_cublasSgemm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAUSEROBJECTCREATE] = std::bind(&TallyServer::handle_cudaUserObjectCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETTENSOR4DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetTensor4dDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNNORMALIZATIONFORWARDTRAINING] = std::bind(&TallyServer::handle_cudnnNormalizationForwardTraining, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECDESTROY] = std::bind(&TallyServer::handle_cuGraphExecDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DASYNC] = std::bind(&TallyServer::handle_cudaMemcpy2DAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGETRSBATCHED] = std::bind(&TallyServer::handle_cublasZgetrsBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETSPATIALTRANSFORMERNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetSpatialTransformerNdDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXACTIVECLUSTERS] = std::bind(&TallyServer::handle_cuOccupancyMaxActiveClusters, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDROPOUTGETRESERVESPACESIZE] = std::bind(&TallyServer::handle_cudnnDropoutGetReserveSpaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFUNCSETATTRIBUTE] = std::bind(&TallyServer::handle_cudaFuncSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY3DASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpy3DAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMQUERY] = std::bind(&TallyServer::handle_cudaStreamQuery, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRSM_V2] = std::bind(&TallyServer::handle_cublasDtrsm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETFORMAT] = std::bind(&TallyServer::handle_cuTexRefSetFormat, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLEXPORTTOSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cudaMemPoolExportToShareableHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMVSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasCgemvStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETACCESS] = std::bind(&TallyServer::handle_cuMemGetAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYRKX] = std::bind(&TallyServer::handle_cublasZsyrkx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGETRFBATCHED] = std::bind(&TallyServer::handle_cublasDgetrfBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasCgemmStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLSEND] = std::bind(&TallyServer::handle_ncclSend, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMV_V2_64] = std::bind(&TallyServer::handle_cublasDgemv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULELOADFATBINARY] = std::bind(&TallyServer::handle_cuModuleLoadFatBinary, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXTRANSFORMDESCSETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatrixTransformDescSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSUNMAPRESOURCES] = std::bind(&TallyServer::handle_cudaGraphicsUnmapResources, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETPRIORITY] = std::bind(&TallyServer::handle_cuStreamGetPriority, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDROPOUTGETSTATESSIZE] = std::bind(&TallyServer::handle_cudnnDropoutGetStatesSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUUSEROBJECTRELEASE] = std::bind(&TallyServer::handle_cuUserObjectRelease, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDOTCEX_64] = std::bind(&TallyServer::handle_cublasDotcEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSORWITHFLAGS] = std::bind(&TallyServer::handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMRANGEGETATTRIBUTES] = std::bind(&TallyServer::handle_cudaMemRangeGetAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTPMV_V2] = std::bind(&TallyServer::handle_cublasCtpmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMMEX_64] = std::bind(&TallyServer::handle_cublasSgemmEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDZNRM2_V2] = std::bind(&TallyServer::handle_cublasDznrm2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXTERNALSEMAPHORESSIGNALNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExternalSemaphoresSignalNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCOPENMEMHANDLE_V2] = std::bind(&TallyServer::handle_cuIpcOpenMemHandle_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODEGETENABLED] = std::bind(&TallyServer::handle_cudaGraphNodeGetEnabled, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMCREATE] = std::bind(&TallyServer::handle_cuStreamCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR] = std::bind(&TallyServer::handle_cuOccupancyMaxActiveBlocksPerMultiprocessor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLSETACCESS] = std::bind(&TallyServer::handle_cuMemPoolSetAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECCHILDGRAPHNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecChildGraphNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYR2_V2_64] = std::bind(&TallyServer::handle_cublasDsyr2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDWEIGHTS] = std::bind(&TallyServer::handle_cudnnMultiHeadAttnBackwardWeights, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHHOSTFUNC] = std::bind(&TallyServer::handle_cuLaunchHostFunc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET3D] = std::bind(&TallyServer::handle_cudaMemset3D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTGETSTATUSSTRING] = std::bind(&TallyServer::handle_cublasLtGetStatusString, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDNODE] = std::bind(&TallyServer::handle_cudaGraphAddNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETFUNCBYSYMBOL] = std::bind(&TallyServer::handle_cudaGetFuncBySymbol, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTRECORD] = std::bind(&TallyServer::handle_cuEventRecord, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGER_V2_64] = std::bind(&TallyServer::handle_cublasSger_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETCACHECONFIG] = std::bind(&TallyServer::handle_cuCtxGetCacheConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNACTIVATIONFORWARD] = std::bind(&TallyServer::handle_cudnnActivationForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYMM_V2] = std::bind(&TallyServer::handle_cublasDsymm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTSYNCHRONIZE] = std::bind(&TallyServer::handle_cuEventSynchronize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODECOPYATTRIBUTES] = std::bind(&TallyServer::handle_cuGraphKernelNodeCopyAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHSHGEMVBATCHED] = std::bind(&TallyServer::handle_cublasHSHgemvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXTERNALSEMAPHORESSIGNALNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphExternalSemaphoresSignalNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSWAP_V2] = std::bind(&TallyServer::handle_cublasSswap_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETSIZE] = std::bind(&TallyServer::handle_cuParamSetSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETREDUCTIONINDICESSIZE] = std::bind(&TallyServer::handle_cudnnGetReductionIndicesSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDDOT_V2_64] = std::bind(&TallyServer::handle_cublasDdot_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLTRIMTO] = std::bind(&TallyServer::handle_cuMemPoolTrimTo, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNMULTIHEADATTNFORWARD] = std::bind(&TallyServer::handle_cudnnMultiHeadAttnForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDNODE] = std::bind(&TallyServer::handle_cuGraphAddNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTRECORD] = std::bind(&TallyServer::handle_cudaEventRecord, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHGEMM] = std::bind(&TallyServer::handle_cublasHgemm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDRIVERGETVERSION] = std::bind(&TallyServer::handle_cuDriverGetVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYHTOA_V2] = std::bind(&TallyServer::handle_cuMemcpyHtoA_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULDESCGETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatmulDescGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONBACKWARDEX] = std::bind(&TallyServer::handle_cudnnBatchNormalizationBackwardEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYASYNC] = std::bind(&TallyServer::handle_cudaMemcpyAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEPERSISTENTRNNPLAN] = std::bind(&TallyServer::handle_cudnnCreatePersistentRNNPlan, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRSV_V2] = std::bind(&TallyServer::handle_cublasZtrsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY3D_V2] = std::bind(&TallyServer::handle_cuMemcpy3D_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMIPMAPPEDARRAYGETSPARSEPROPERTIES] = std::bind(&TallyServer::handle_cuMipmappedArrayGetSparseProperties, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETTENSOR4DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetTensor4dDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMVBATCHED_64] = std::bind(&TallyServer::handle_cublasCgemvBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETBYPCIBUSID] = std::bind(&TallyServer::handle_cudaDeviceGetByPCIBusId, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDNRM2_V2] = std::bind(&TallyServer::handle_cublasDnrm2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNADDTENSOR] = std::bind(&TallyServer::handle_cudnnAddTensor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETMEMPOOL] = std::bind(&TallyServer::handle_cuDeviceGetMemPool, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLGETVERSION] = std::bind(&TallyServer::handle_ncclGetVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRMM_V2_64] = std::bind(&TallyServer::handle_cublasStrmm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTOA_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoA_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTQUERY] = std::bind(&TallyServer::handle_cudaEventQuery, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODESETATTRIBUTE] = std::bind(&TallyServer::handle_cudaGraphKernelNodeSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSASUM_V2] = std::bind(&TallyServer::handle_cublasSasum_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECUPDATE] = std::bind(&TallyServer::handle_cudaGraphExecUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGERU_V2] = std::bind(&TallyServer::handle_cublasCgeru_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONFORWARDINFERENCE] = std::bind(&TallyServer::handle_cudnnBatchNormalizationForwardInference, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSWAPEX_64] = std::bind(&TallyServer::handle_cublasSwapEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTLOGGERSETMASK] = std::bind(&TallyServer::handle_cublasLtLoggerSetMask, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETTENSORTRANSFORMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetTensorTransformDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYPOOLINGDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyPoolingDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRK3MEX] = std::bind(&TallyServer::handle_cublasCsyrk3mEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYR2_V2] = std::bind(&TallyServer::handle_cublasZsyr2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETMATHMODE] = std::bind(&TallyServer::handle_cublasSetMathMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHERK_V2_64] = std::bind(&TallyServer::handle_cublasZherk_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODEGETDEPENDENCIES] = std::bind(&TallyServer::handle_cudaGraphNodeGetDependencies, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRMV_V2_64] = std::bind(&TallyServer::handle_cublasDtrmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHGETEDGES] = std::bind(&TallyServer::handle_cuGraphGetEdges, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETSTREAMPRIORITYRANGE] = std::bind(&TallyServer::handle_cudaDeviceGetStreamPriorityRange, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETPRIORITY] = std::bind(&TallyServer::handle_cudaStreamGetPriority, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSBMV_V2] = std::bind(&TallyServer::handle_cublasSsbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGER_V2] = std::bind(&TallyServer::handle_cublasSger_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYFILTERDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyFilterDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETACTIVATIONDESCRIPTORSWISHBETA] = std::bind(&TallyServer::handle_cudnnSetActivationDescriptorSwishBeta, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCOREDUMPSETATTRIBUTEGLOBAL] = std::bind(&TallyServer::handle_cuCoredumpSetAttributeGlobal, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASISAMAX_V2] = std::bind(&TallyServer::handle_cublasIsamax_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCGETEVENTHANDLE] = std::bind(&TallyServer::handle_cuIpcGetEventHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDAXPY_V2_64] = std::bind(&TallyServer::handle_cublasDaxpy_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTRECORDWITHFLAGS] = std::bind(&TallyServer::handle_cudaEventRecordWithFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMVSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasSgemvStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETVECTOR_64] = std::bind(&TallyServer::handle_cublasGetVector_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMMSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasDgemmStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHCOOPERATIVEKERNELMULTIDEVICE] = std::bind(&TallyServer::handle_cuLaunchCooperativeKernelMultiDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULPREFERENCECREATE] = std::bind(&TallyServer::handle_cublasLtMatmulPreferenceCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTALLOC] = std::bind(&TallyServer::handle_cudaHostAlloc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETSEQDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetSeqDataDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYMV_V2] = std::bind(&TallyServer::handle_cublasCsymv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLREDUCESCATTER] = std::bind(&TallyServer::handle_pncclReduceScatter, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETCUDARTVERSION] = std::bind(&TallyServer::handle_cublasGetCudartVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYR2K_V2_64] = std::bind(&TallyServer::handle_cublasSsyr2k_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMCPYNODEFROMSYMBOL] = std::bind(&TallyServer::handle_cudaGraphAddMemcpyNodeFromSymbol, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRESTOREDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnRestoreDropoutDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNBIASMODE] = std::bind(&TallyServer::handle_cudnnGetRNNBiasMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D8_V2] = std::bind(&TallyServer::handle_cuMemsetD2D8_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHCREATE] = std::bind(&TallyServer::handle_cudaGraphCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCCLOSEMEMHANDLE] = std::bind(&TallyServer::handle_cudaIpcCloseMemHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFDESTROY] = std::bind(&TallyServer::handle_cuTexRefDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYREDUCETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyReduceTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSROT_V2_64] = std::bind(&TallyServer::handle_cublasCsrot_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXENABLEPEERACCESS] = std::bind(&TallyServer::handle_cuCtxEnablePeerAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYATOA_V2] = std::bind(&TallyServer::handle_cuMemcpyAtoA_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSCAL_V2] = std::bind(&TallyServer::handle_cublasCscal_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETFILTERMODE] = std::bind(&TallyServer::handle_cuTexRefSetFilterMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHERKX_64] = std::bind(&TallyServer::handle_cublasZherkx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGETRSBATCHED] = std::bind(&TallyServer::handle_cublasDgetrsBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIMPORTEXTERNALMEMORY] = std::bind(&TallyServer::handle_cudaImportExternalMemory, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMVSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasDgemvStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNDESCRIPTOR_V6] = std::bind(&TallyServer::handle_cudnnSetRNNDescriptor_v6, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEATTNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateAttnDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRK3MEX_64] = std::bind(&TallyServer::handle_cublasCsyrk3mEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACREATESURFACEOBJECT] = std::bind(&TallyServer::handle_cudaCreateSurfaceObject, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHHOSTNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphHostNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMCPYNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecMemcpyNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGETRSBATCHED] = std::bind(&TallyServer::handle_cublasCgetrsBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY2D_V2] = std::bind(&TallyServer::handle_cuMemcpy2D_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAPROFILERSTART] = std::bind(&TallyServer::handle_cudaProfilerStart, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETDEVICE] = std::bind(&TallyServer::handle_cudaSetDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY2DASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpy2DAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLGETUNIQUEID] = std::bind(&TallyServer::handle_ncclGetUniqueId, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATECTCLOSSDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateCTCLossDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHHOSTNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphHostNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETLASTERROR] = std::bind(&TallyServer::handle_cudaGetLastError, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETPROGRAMLOGSIZE] = std::bind(&TallyServer::handle_nvrtcGetProgramLogSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXCREATE_V3] = std::bind(&TallyServer::handle_cuCtxCreate_v3, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETBATCHNORMALIZATIONTRAININGEXRESERVESPACESIZE] = std::bind(&TallyServer::handle_cudnnGetBatchNormalizationTrainingExReserveSpaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCPITCH] = std::bind(&TallyServer::handle_cudaMallocPitch, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYRNNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyRNNDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDCOPY_V2] = std::bind(&TallyServer::handle_cublasDcopy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDEVICE] = std::bind(&TallyServer::handle_cudaGetDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGRAPHMEMTRIM] = std::bind(&TallyServer::handle_cudaDeviceGraphMemTrim, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGETRFBATCHED] = std::bind(&TallyServer::handle_cublasCgetrfBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHBMV_V2_64] = std::bind(&TallyServer::handle_cublasChbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHSHGEMVSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasHSHgemvStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETFILTER4DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetFilter4dDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGRAPHMEMTRIM] = std::bind(&TallyServer::handle_cuDeviceGraphMemTrim, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCOREDUMPGETATTRIBUTE] = std::bind(&TallyServer::handle_cuCoredumpGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNDESCRIPTOR_V8] = std::bind(&TallyServer::handle_cudnnGetRNNDescriptor_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXDETACH] = std::bind(&TallyServer::handle_cuCtxDetach, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTPMV_V2_64] = std::bind(&TallyServer::handle_cublasDtpmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTBSV_V2] = std::bind(&TallyServer::handle_cublasDtbsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMBATCHMEMOP_V2] = std::bind(&TallyServer::handle_cuStreamBatchMemOp_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMINITRANKCONFIG] = std::bind(&TallyServer::handle_ncclCommInitRankConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDBATCHMEMOPNODE] = std::bind(&TallyServer::handle_cuGraphAddBatchMemOpNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHRELEASEUSEROBJECT] = std::bind(&TallyServer::handle_cuGraphReleaseUserObject, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYFUSEDOPSCONSTPARAMPACK] = std::bind(&TallyServer::handle_cudnnDestroyFusedOpsConstParamPack, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEQRFBATCHED] = std::bind(&TallyServer::handle_cublasZgeqrfBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTGETCUDARTVERSION] = std::bind(&TallyServer::handle_cublasLtGetCudartVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUFLUSHGPUDIRECTRDMAWRITES] = std::bind(&TallyServer::handle_cuFlushGPUDirectRDMAWrites, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNLRNCROSSCHANNELFORWARD] = std::bind(&TallyServer::handle_cudnnLRNCrossChannelForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMVSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasZgemvStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLGETLASTERROR] = std::bind(&TallyServer::handle_ncclGetLastError, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUIMPORTEXTERNALMEMORY] = std::bind(&TallyServer::handle_cuImportExternalMemory, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECEXTERNALSEMAPHORESSIGNALNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecExternalSemaphoresSignalNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMSETNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemsetNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMCOUNT] = std::bind(&TallyServer::handle_ncclCommCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETTEXTUREOBJECTRESOURCEVIEWDESC] = std::bind(&TallyServer::handle_cudaGetTextureObjectResourceViewDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYR2K_V2_64] = std::bind(&TallyServer::handle_cublasZsyr2k_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFOBJECTDESTROY] = std::bind(&TallyServer::handle_cuSurfObjectDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDCONVOLUTIONBACKWARDFILTERALGORITHMEX] = std::bind(&TallyServer::handle_cudnnFindConvolutionBackwardFilterAlgorithmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHSHGEMVSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasHSHgemvStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTIONREORDERTYPE] = std::bind(&TallyServer::handle_cudnnSetConvolutionReorderType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLGETACCESS] = std::bind(&TallyServer::handle_cuMemPoolGetAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDEXECUTE] = std::bind(&TallyServer::handle_cudnnBackendExecute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRMM_V2] = std::bind(&TallyServer::handle_cublasDtrmm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEXTERNALSEMAPHORESWAITNODE] = std::bind(&TallyServer::handle_cudaGraphAddExternalSemaphoresWaitNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMISCAPTURING] = std::bind(&TallyServer::handle_cudaStreamIsCapturing, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTS] = std::bind(&TallyServer::handle_cudnnRNNBackwardWeights, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRMM_V2_64] = std::bind(&TallyServer::handle_cublasZtrmm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETLOGGERCALLBACK] = std::bind(&TallyServer::handle_cublasSetLoggerCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFOBJECTGETRESOURCEDESC] = std::bind(&TallyServer::handle_cuSurfObjectGetResourceDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHSSGEMVBATCHED_64] = std::bind(&TallyServer::handle_cublasHSSgemvBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCMATINVBATCHED] = std::bind(&TallyServer::handle_cublasCmatinvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHSSGEMVSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasHSSgemvStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRTTP] = std::bind(&TallyServer::handle_cublasStrttp, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNWEIGHTPARAMS] = std::bind(&TallyServer::handle_cudnnGetRNNWeightParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3MEX_64] = std::bind(&TallyServer::handle_cublasCgemm3mEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNNORMALIZATIONBACKWARD] = std::bind(&TallyServer::handle_cudnnNormalizationBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETEXECAFFINITY] = std::bind(&TallyServer::handle_cuCtxGetExecAffinity, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRSMBATCHED] = std::bind(&TallyServer::handle_cublasDtrsmBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D32ASYNC] = std::bind(&TallyServer::handle_cuMemsetD2D32Async, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLREDUCE] = std::bind(&TallyServer::handle_ncclReduce, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCASUM_V2_64] = std::bind(&TallyServer::handle_cublasScasum_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLTRIMTO] = std::bind(&TallyServer::handle_cudaMemPoolTrimTo, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHPMV_V2_64] = std::bind(&TallyServer::handle_cublasChpmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREEHOST] = std::bind(&TallyServer::handle_cudaFreeHost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEAM] = std::bind(&TallyServer::handle_cublasZgeam, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMMEX_64] = std::bind(&TallyServer::handle_cublasCgemmEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCROT_V2] = std::bind(&TallyServer::handle_cublasCrot_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMRETAINALLOCATIONHANDLE] = std::bind(&TallyServer::handle_cuMemRetainAllocationHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICECANACCESSPEER] = std::bind(&TallyServer::handle_cudaDeviceCanAccessPeer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetRNNAlgorithmDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDFINALIZE] = std::bind(&TallyServer::handle_cudnnBackendFinalize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATECONVOLUTIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateConvolutionDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEPOOLINGDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreatePoolingDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETSMCOUNTTARGET] = std::bind(&TallyServer::handle_cublasGetSmCountTarget, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTBSV_V2] = std::bind(&TallyServer::handle_cublasZtbsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCOPYALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCopyAlgorithmDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSSCAL_V2] = std::bind(&TallyServer::handle_cublasCsscal_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTS_V8] = std::bind(&TallyServer::handle_cudnnRNNBackwardWeights_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETOPTIXIR] = std::bind(&TallyServer::handle_nvrtcGetOptiXIR, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETADDRESS_V2] = std::bind(&TallyServer::handle_cuTexRefGetAddress_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRSV_V2_64] = std::bind(&TallyServer::handle_cublasDtrsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAPEEKATLASTERROR] = std::bind(&TallyServer::handle_cudaPeekAtLastError, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCCREATEPROGRAM] = std::bind(&TallyServer::handle_nvrtcCreateProgram, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDROPOUTBACKWARD] = std::bind(&TallyServer::handle_cudnnDropoutBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETDOUBLEFORDEVICE] = std::bind(&TallyServer::handle_cudaSetDoubleForDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMULTICASTBINDMEM] = std::bind(&TallyServer::handle_cuMulticastBindMem, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICESETGRAPHMEMATTRIBUTE] = std::bind(&TallyServer::handle_cuDeviceSetGraphMemAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMCOUNT] = std::bind(&TallyServer::handle_pncclCommCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULELOAD] = std::bind(&TallyServer::handle_cuModuleLoad, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHSHGEMVBATCHED_64] = std::bind(&TallyServer::handle_cublasHSHgemvBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHEMM_V2_64] = std::bind(&TallyServer::handle_cublasZhemm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETGRAPHMEMATTRIBUTE] = std::bind(&TallyServer::handle_cudaDeviceSetGraphMemAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEGETTEXREF] = std::bind(&TallyServer::handle_cuModuleGetTexRef, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCOPENEVENTHANDLE] = std::bind(&TallyServer::handle_cuIpcOpenEventHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEQRFBATCHED] = std::bind(&TallyServer::handle_cublasSgeqrfBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetConvolutionForwardAlgorithmMaxCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHCLONE] = std::bind(&TallyServer::handle_cudaGraphClone, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEACTIVATIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateActivationDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTPTTR] = std::bind(&TallyServer::handle_cublasDtpttr, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZROT_V2_64] = std::bind(&TallyServer::handle_cublasZrot_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMABORT] = std::bind(&TallyServer::handle_pncclCommAbort, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYHTOAASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyHtoAAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYRKX_64] = std::bind(&TallyServer::handle_cublasSsyrkx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETVERSION] = std::bind(&TallyServer::handle_cudnnGetVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGBMV_V2_64] = std::bind(&TallyServer::handle_cublasSgbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETPOINTERMODE_V2] = std::bind(&TallyServer::handle_cublasGetPointerMode_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETALLOCATIONGRANULARITY] = std::bind(&TallyServer::handle_cuMemGetAllocationGranularity, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMSETNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemsetNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASISAMAX_V2_64] = std::bind(&TallyServer::handle_cublasIsamax_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDESTROYEXTERNALSEMAPHORE] = std::bind(&TallyServer::handle_cuDestroyExternalSemaphore, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCFROMPOOLASYNC] = std::bind(&TallyServer::handle_cudaMallocFromPoolAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZMATINVBATCHED] = std::bind(&TallyServer::handle_cublasZmatinvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODEGETDEPENDENCIES] = std::bind(&TallyServer::handle_cuGraphNodeGetDependencies, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETFUSEDOPSVARIANTPARAMPACKATTRIBUTE] = std::bind(&TallyServer::handle_cudnnSetFusedOpsVariantParamPackAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETOPTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetOpTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHEMM_V2_64] = std::bind(&TallyServer::handle_cublasChemm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCDOTU_V2_64] = std::bind(&TallyServer::handle_cublasCdotu_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASTSSGEMVSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasTSSgemvStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCREATE_V2] = std::bind(&TallyServer::handle_cublasCreate_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLEXPORTPOINTER] = std::bind(&TallyServer::handle_cudaMemPoolExportPointer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSPR_V2] = std::bind(&TallyServer::handle_cublasSspr_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROTMG_V2] = std::bind(&TallyServer::handle_cublasDrotmg_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCNNTRAINVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnCnnTrainVersionCheck, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDDOT_V2] = std::bind(&TallyServer::handle_cublasDdot_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRSM_V2_64] = std::bind(&TallyServer::handle_cublasCtrsm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGBMV_V2] = std::bind(&TallyServer::handle_cublasZgbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHBMV_V2] = std::bind(&TallyServer::handle_cublasChbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETPCIBUSID] = std::bind(&TallyServer::handle_cuDeviceGetPCIBusId, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETTEXTURE1DLINEARMAXWIDTH] = std::bind(&TallyServer::handle_cudaDeviceGetTexture1DLinearMaxWidth, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXTERNALSEMAPHORESWAITNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExternalSemaphoresWaitNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTLOGGERSETLEVEL] = std::bind(&TallyServer::handle_cublasLtLoggerSetLevel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGETRFBATCHED] = std::bind(&TallyServer::handle_cublasZgetrfBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYR2_V2_64] = std::bind(&TallyServer::handle_cublasZsyr2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMM_V2] = std::bind(&TallyServer::handle_cublasDgemm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETADDRESSMODE] = std::bind(&TallyServer::handle_cuTexRefSetAddressMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTBSV_V2] = std::bind(&TallyServer::handle_cublasCtbsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLDESTROY] = std::bind(&TallyServer::handle_cudaMemPoolDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYR_V2_64] = std::bind(&TallyServer::handle_cublasDsyr_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONFORWARDTRAININGEX] = std::bind(&TallyServer::handle_cudnnBatchNormalizationForwardTrainingEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULDESCINIT_INTERNAL] = std::bind(&TallyServer::handle_cublasLtMatmulDescInit_internal, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETWORKSPACE_V2] = std::bind(&TallyServer::handle_cublasSetWorkspace_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCPITCH_V2] = std::bind(&TallyServer::handle_cuMemAllocPitch_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXTRANSFORMDESCCREATE] = std::bind(&TallyServer::handle_cublasLtMatrixTransformDescCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFREFGETARRAY] = std::bind(&TallyServer::handle_cuSurfRefGetArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETPOOLINGNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetPoolingNdDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDROT_V2_64] = std::bind(&TallyServer::handle_cublasZdrot_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTPSV_V2] = std::bind(&TallyServer::handle_cublasCtpsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETPCIBUSID] = std::bind(&TallyServer::handle_cudaDeviceGetPCIBusId, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDOTU_V2_64] = std::bind(&TallyServer::handle_cublasZdotu_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMVBATCHED] = std::bind(&TallyServer::handle_cublasZgemvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASICAMAX_V2_64] = std::bind(&TallyServer::handle_cublasIcamax_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMISCAPTURING] = std::bind(&TallyServer::handle_cuStreamIsCapturing, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHKERNEL] = std::bind(&TallyServer::handle_cudaLaunchKernel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaStreamGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXPOTENTIALCLUSTERSIZE] = std::bind(&TallyServer::handle_cuOccupancyMaxPotentialClusterSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSRESOURCESETMAPFLAGS] = std::bind(&TallyServer::handle_cudaGraphicsResourceSetMapFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDEVICECOUNT] = std::bind(&TallyServer::handle_cudaGetDeviceCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCALEX_64] = std::bind(&TallyServer::handle_cublasScalEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULIBRARYGETUNIFIEDFUNCTION] = std::bind(&TallyServer::handle_cuLibraryGetUnifiedFunction, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTBMV_V2_64] = std::bind(&TallyServer::handle_cublasStbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTOH_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoH_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODESETENABLED] = std::bind(&TallyServer::handle_cuGraphNodeSetEnabled, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMCREATEWITHFLAGS] = std::bind(&TallyServer::handle_cudaStreamCreateWithFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIZAMIN_V2] = std::bind(&TallyServer::handle_cublasIzamin_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTLOGGEROPENFILE] = std::bind(&TallyServer::handle_cublasLtLoggerOpenFile, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNLINLAYERMATRIXPARAMS] = std::bind(&TallyServer::handle_cudnnGetRNNLinLayerMatrixParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSMATINVBATCHED] = std::bind(&TallyServer::handle_cublasSmatinvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEMPTYNODE] = std::bind(&TallyServer::handle_cudaGraphAddEmptyNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHRELEASEUSEROBJECT] = std::bind(&TallyServer::handle_cudaGraphReleaseUserObject, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3M] = std::bind(&TallyServer::handle_cublasCgemm3m, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEVENTWAITNODESETEVENT] = std::bind(&TallyServer::handle_cuGraphEventWaitNodeSetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROY] = std::bind(&TallyServer::handle_cudnnDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETUUID_V2] = std::bind(&TallyServer::handle_cuDeviceGetUuid_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSDOT_V2] = std::bind(&TallyServer::handle_cublasSdot_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRMV_V2_64] = std::bind(&TallyServer::handle_cublasStrmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSCAL_V2] = std::bind(&TallyServer::handle_cublasSscal_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMM_V2_64] = std::bind(&TallyServer::handle_cublasSgemm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMVSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasCgemvStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERKX] = std::bind(&TallyServer::handle_cublasCherkx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDCONVOLUTIONBACKWARDDATAALGORITHM] = std::bind(&TallyServer::handle_cudnnFindConvolutionBackwardDataAlgorithm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPROFILERSTART] = std::bind(&TallyServer::handle_cuProfilerStart, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULIBRARYUNLOAD] = std::bind(&TallyServer::handle_cuLibraryUnload, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMADDRESSRESERVE] = std::bind(&TallyServer::handle_cuMemAddressReserve, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetAlgorithmDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECUPDATE_V2] = std::bind(&TallyServer::handle_cuGraphExecUpdate_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYPEER] = std::bind(&TallyServer::handle_cudaMemcpyPeer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSOFTMAXBACKWARD] = std::bind(&TallyServer::handle_cudnnSoftmaxBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHGRIDASYNC] = std::bind(&TallyServer::handle_cuLaunchGridAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYTOARRAYASYNC] = std::bind(&TallyServer::handle_cudaMemcpyToArrayAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAOCCUPANCYMAXPOTENTIALCLUSTERSIZE] = std::bind(&TallyServer::handle_cudaOccupancyMaxPotentialClusterSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMATTACHMEMASYNC] = std::bind(&TallyServer::handle_cudaStreamAttachMemAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIDAMIN_V2] = std::bind(&TallyServer::handle_cublasIdamin_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETALGORITHMPERFORMANCE] = std::bind(&TallyServer::handle_cudnnGetAlgorithmPerformance, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDDESTROYDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnBackendDestroyDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXTERNALSEMAPHORESWAITNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExternalSemaphoresWaitNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDSETATTRIBUTE] = std::bind(&TallyServer::handle_cudnnBackendSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUUSEROBJECTRETAIN] = std::bind(&TallyServer::handle_cuUserObjectRetain, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCH] = std::bind(&TallyServer::handle_cuLaunch, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROT_V2] = std::bind(&TallyServer::handle_cublasSrot_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDMEMFREENODE] = std::bind(&TallyServer::handle_cuGraphAddMemFreeNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHINSTANTIATEWITHPARAMS] = std::bind(&TallyServer::handle_cudaGraphInstantiateWithParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYMV_V2] = std::bind(&TallyServer::handle_cublasDsymv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTGETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatrixLayoutGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETLUID] = std::bind(&TallyServer::handle_cuDeviceGetLuid, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHEMV_V2] = std::bind(&TallyServer::handle_cublasZhemv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD8ASYNC] = std::bind(&TallyServer::handle_cuMemsetD8Async, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEVENTRECORDNODE] = std::bind(&TallyServer::handle_cudaGraphAddEventRecordNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCROT_V2_64] = std::bind(&TallyServer::handle_cublasCrot_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETTENSORNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetTensorNdDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSCAL_V2_64] = std::bind(&TallyServer::handle_cublasCscal_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEVENTWAITNODEGETEVENT] = std::bind(&TallyServer::handle_cuGraphEventWaitNodeGetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMUSERRANK] = std::bind(&TallyServer::handle_pncclCommUserRank, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMSETATTRIBUTE] = std::bind(&TallyServer::handle_cudaStreamSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCTCLOSSDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetCTCLossDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMMBATCHED_64] = std::bind(&TallyServer::handle_cublasZgemmBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECKERNELNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecKernelNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTUNREGISTER] = std::bind(&TallyServer::handle_cuMemHostUnregister, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTPMV_V2_64] = std::bind(&TallyServer::handle_cublasZtpmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHCOOPERATIVEKERNEL] = std::bind(&TallyServer::handle_cuLaunchCooperativeKernel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNWEIGHTSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetRNNWeightSpaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSPR2_V2_64] = std::bind(&TallyServer::handle_cublasSspr2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMADDRESSFREE] = std::bind(&TallyServer::handle_cuMemAddressFree, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETMATRIXASYNC_64] = std::bind(&TallyServer::handle_cublasGetMatrixAsync_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXTRANSFORMDESCGETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatrixTransformDescGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DTOARRAY] = std::bind(&TallyServer::handle_cudaMemcpy2DToArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSWAP_V2_64] = std::bind(&TallyServer::handle_cublasSswap_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLCREATE] = std::bind(&TallyServer::handle_cudaMemPoolCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROTM_V2] = std::bind(&TallyServer::handle_cublasDrotm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXTERNALSEMAPHORESWAITNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphExternalSemaphoresWaitNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETLTOIRSIZE] = std::bind(&TallyServer::handle_nvrtcGetLTOIRSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTPSV_V2] = std::bind(&TallyServer::handle_cublasZtpsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDEVICEPROPERTIES_V2] = std::bind(&TallyServer::handle_cudaGetDeviceProperties_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREEMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaFreeMipmappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYR_V2] = std::bind(&TallyServer::handle_cublasDsyr_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRSV_V2_64] = std::bind(&TallyServer::handle_cublasZtrsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLEXPORTTOSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cuMemPoolExportToShareableHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHDESTROYNODE] = std::bind(&TallyServer::handle_cudaGraphDestroyNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETCAPTUREINFO_V2] = std::bind(&TallyServer::handle_cuStreamGetCaptureInfo_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMIPMAPPEDARRAYGETMEMORYREQUIREMENTS] = std::bind(&TallyServer::handle_cuMipmappedArrayGetMemoryRequirements, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETFUSEDOPSCONSTPARAMPACKATTRIBUTE] = std::bind(&TallyServer::handle_cudnnSetFusedOpsConstParamPackAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDMEMCPYNODE] = std::bind(&TallyServer::handle_cuGraphAddMemcpyNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLIMPORTFROMSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cuMemPoolImportFromShareableHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSWAP_V2] = std::bind(&TallyServer::handle_cublasZswap_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYRK_V2] = std::bind(&TallyServer::handle_cublasDsyrk_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSRESOURCEGETMAPPEDPOINTER] = std::bind(&TallyServer::handle_cudaGraphicsResourceGetMappedPointer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETCAPTUREINFO_V2] = std::bind(&TallyServer::handle_cudaStreamGetCaptureInfo_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODESETPARAMS_V2] = std::bind(&TallyServer::handle_cuGraphKernelNodeSetParams_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARDINFERENCEEX] = std::bind(&TallyServer::handle_cudnnRNNForwardInferenceEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDROPOUTFORWARD] = std::bind(&TallyServer::handle_cudnnDropoutForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFUNCGETATTRIBUTES] = std::bind(&TallyServer::handle_cudaFuncGetAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARDINFERENCE] = std::bind(&TallyServer::handle_cudnnRNNForwardInference, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULIBRARYGETMODULE] = std::bind(&TallyServer::handle_cuLibraryGetModule, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDDATA] = std::bind(&TallyServer::handle_cudnnRNNBackwardData, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETSTREAM_V2] = std::bind(&TallyServer::handle_cublasSetStream_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRESTOREALGORITHM] = std::bind(&TallyServer::handle_cudnnRestoreAlgorithm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUEXTERNALMEMORYGETMAPPEDBUFFER] = std::bind(&TallyServer::handle_cuExternalMemoryGetMappedBuffer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY3D] = std::bind(&TallyServer::handle_cudaMemcpy3D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHINSTANTIATE] = std::bind(&TallyServer::handle_cudaGraphInstantiate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXATTACH] = std::bind(&TallyServer::handle_cuCtxAttach, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMVSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasDgemvStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRSMBATCHED] = std::bind(&TallyServer::handle_cublasZtrsmBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTGETDEVICEPOINTER_V2] = std::bind(&TallyServer::handle_cuMemHostGetDevicePointer_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHHOSTFUNC] = std::bind(&TallyServer::handle_cudaLaunchHostFunc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYRK_V2_64] = std::bind(&TallyServer::handle_cublasSsyrk_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETPTXSIZE] = std::bind(&TallyServer::handle_nvrtcGetPTXSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEXTERNALSEMAPHORESWAITNODE] = std::bind(&TallyServer::handle_cuGraphAddExternalSemaphoresWaitNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHINSTANTIATEWITHPARAMS] = std::bind(&TallyServer::handle_cuGraphInstantiateWithParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHSSGEMVBATCHED] = std::bind(&TallyServer::handle_cublasHSSgemvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHER2_V2] = std::bind(&TallyServer::handle_cublasCher2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMABORT] = std::bind(&TallyServer::handle_ncclCommAbort, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAARRAYGETPLANE] = std::bind(&TallyServer::handle_cudaArrayGetPlane, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYHTODASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyHtoDAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHGETEDGES] = std::bind(&TallyServer::handle_cudaGraphGetEdges, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMSETATTRIBUTE] = std::bind(&TallyServer::handle_cuStreamSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARDTRAINING] = std::bind(&TallyServer::handle_cudnnRNNForwardTraining, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRSMBATCHED_64] = std::bind(&TallyServer::handle_cublasZtrsmBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEXTERNALSEMAPHORESSIGNALNODE] = std::bind(&TallyServer::handle_cudaGraphAddExternalSemaphoresSignalNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYRK_V2] = std::bind(&TallyServer::handle_cublasZsyrk_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMBEGINCAPTURE_V2] = std::bind(&TallyServer::handle_cuStreamBeginCapture_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNTRANSFORMTENSOR] = std::bind(&TallyServer::handle_cudnnTransformTensor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEALGORITHMPERFORMANCE] = std::bind(&TallyServer::handle_cudnnCreateAlgorithmPerformance, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETPROGRAMLOG] = std::bind(&TallyServer::handle_nvrtcGetProgramLog, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSCAL_V2] = std::bind(&TallyServer::handle_cublasZscal_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIMPORTEXTERNALSEMAPHORE] = std::bind(&TallyServer::handle_cudaImportExternalSemaphore, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHDESTROY] = std::bind(&TallyServer::handle_cudaGraphDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD8_V2] = std::bind(&TallyServer::handle_cuMemsetD8_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPOOLINGNDFORWARDOUTPUTDIM] = std::bind(&TallyServer::handle_cudnnGetPoolingNdForwardOutputDim, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETMATHMODE] = std::bind(&TallyServer::handle_cublasGetMathMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETVECTOR] = std::bind(&TallyServer::handle_cublasGetVector, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECEXTERNALSEMAPHORESWAITNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecExternalSemaphoresWaitNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEGETLOADINGMODE] = std::bind(&TallyServer::handle_cuModuleGetLoadingMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHPMV_V2_64] = std::bind(&TallyServer::handle_cublasZhpmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPOOLING2DFORWARDOUTPUTDIM] = std::bind(&TallyServer::handle_cudnnGetPooling2dForwardOutputDim, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGELSBATCHED] = std::bind(&TallyServer::handle_cublasZgelsBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTDESTROY] = std::bind(&TallyServer::handle_cudaEventDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETFILTERSIZEINBYTES] = std::bind(&TallyServer::handle_cudnnGetFilterSizeInBytes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODEGETTYPE] = std::bind(&TallyServer::handle_cudaGraphNodeGetType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSPR_V2] = std::bind(&TallyServer::handle_cublasDspr_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLGETACCESS] = std::bind(&TallyServer::handle_cudaMemPoolGetAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMVSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasSgemvStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKDESTROY] = std::bind(&TallyServer::handle_cuLinkDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDMEMSETNODE] = std::bind(&TallyServer::handle_cuGraphAddMemsetNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARD] = std::bind(&TallyServer::handle_cudnnRNNForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGETRIBATCHED] = std::bind(&TallyServer::handle_cublasSgetriBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETATOMICSMODE] = std::bind(&TallyServer::handle_cublasSetAtomicsMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETACTIVATIONDESCRIPTORSWISHBETA] = std::bind(&TallyServer::handle_cudnnGetActivationDescriptorSwishBeta, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRMV_V2] = std::bind(&TallyServer::handle_cublasCtrmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDOTEX] = std::bind(&TallyServer::handle_cublasDotEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAUSEROBJECTRETAIN] = std::bind(&TallyServer::handle_cudaUserObjectRetain, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETMATRIX_64] = std::bind(&TallyServer::handle_cublasSetMatrix_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETDEFAULTMEMPOOL] = std::bind(&TallyServer::handle_cuDeviceGetDefaultMemPool, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETSHAREDSIZE] = std::bind(&TallyServer::handle_cuFuncSetSharedSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMUPDATECAPTUREDEPENDENCIES] = std::bind(&TallyServer::handle_cudaStreamUpdateCaptureDependencies, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR] = std::bind(&TallyServer::handle_cudaOccupancyMaxActiveBlocksPerMultiprocessor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYACTIVATIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyActivationDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETSEQDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetSeqDataDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHER_V2] = std::bind(&TallyServer::handle_cublasZher_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cudaDeviceSetSharedMemConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSCALETENSOR] = std::bind(&TallyServer::handle_cudnnScaleTensor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTBSV_V2_64] = std::bind(&TallyServer::handle_cublasDtbsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMINITRANKCONFIG] = std::bind(&TallyServer::handle_pncclCommInitRankConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNPARAMSSIZE] = std::bind(&TallyServer::handle_cudnnGetRNNParamsSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCMANAGED] = std::bind(&TallyServer::handle_cudaMallocManaged, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMENDCAPTURE] = std::bind(&TallyServer::handle_cudaStreamEndCapture, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLOGGERCONFIGURE] = std::bind(&TallyServer::handle_cublasLoggerConfigure, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGERC_V2_64] = std::bind(&TallyServer::handle_cublasCgerc_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPREFETCHASYNC] = std::bind(&TallyServer::handle_cudaMemPrefetchAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRSM_V2] = std::bind(&TallyServer::handle_cublasZtrsm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLCREATE] = std::bind(&TallyServer::handle_cuMemPoolCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMVBATCHED_64] = std::bind(&TallyServer::handle_cublasSgemvBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEGETFUNCTION] = std::bind(&TallyServer::handle_cuModuleGetFunction, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMSETNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecMemsetNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIDAMIN_V2_64] = std::bind(&TallyServer::handle_cublasIdamin_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCDOTC_V2] = std::bind(&TallyServer::handle_cublasCdotc_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETNVSCISYNCATTRIBUTES] = std::bind(&TallyServer::handle_cuDeviceGetNvSciSyncAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSETLIMIT] = std::bind(&TallyServer::handle_cuCtxSetLimit, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHCOOPERATIVEKERNELMULTIDEVICE] = std::bind(&TallyServer::handle_cudaLaunchCooperativeKernelMultiDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTPMV_V2_64] = std::bind(&TallyServer::handle_cublasCtpmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYTOSYMBOLASYNC] = std::bind(&TallyServer::handle_cudaMemcpyToSymbolAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETAPIVERSION] = std::bind(&TallyServer::handle_cuCtxGetApiVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTEX_64] = std::bind(&TallyServer::handle_cublasRotEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETCHANNELDESC] = std::bind(&TallyServer::handle_cudaGetChannelDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMATTACHMEMASYNC] = std::bind(&TallyServer::handle_cuStreamAttachMemAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYRK_V2] = std::bind(&TallyServer::handle_cublasSsyrk_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETVECTORASYNC] = std::bind(&TallyServer::handle_cublasSetVectorAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETDEFAULTMEMPOOL] = std::bind(&TallyServer::handle_cudaDeviceGetDefaultMemPool, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETCACHECONFIG] = std::bind(&TallyServer::handle_cuFuncSetCacheConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTCREATE] = std::bind(&TallyServer::handle_cuEventCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNMATRIXMATHTYPE] = std::bind(&TallyServer::handle_cudnnGetRNNMatrixMathType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADESTROYEXTERNALMEMORY] = std::bind(&TallyServer::handle_cudaDestroyExternalMemory, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEGETSURFREF] = std::bind(&TallyServer::handle_cuModuleGetSurfRef, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWAITVALUE32_V2] = std::bind(&TallyServer::handle_cuStreamWaitValue32_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCHOST_V2] = std::bind(&TallyServer::handle_cuMemAllocHost_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUUSEROBJECTCREATE] = std::bind(&TallyServer::handle_cuUserObjectCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOC] = std::bind(&TallyServer::handle_cudaMalloc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONNDFORWARDOUTPUTDIM] = std::bind(&TallyServer::handle_cudnnGetConvolutionNdForwardOutputDim, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGER_V2] = std::bind(&TallyServer::handle_cublasDger_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMSETNODE] = std::bind(&TallyServer::handle_cudaGraphAddMemsetNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTDESTROY] = std::bind(&TallyServer::handle_cublasLtDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYTENSORTRANSFORMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyTensorTransformDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONBACKWARDBIAS] = std::bind(&TallyServer::handle_cudnnConvolutionBackwardBias, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLALLREDUCE] = std::bind(&TallyServer::handle_pncclAllReduce, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYSEQDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroySeqDataDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXPOTENTIALBLOCKSIZE] = std::bind(&TallyServer::handle_cuOccupancyMaxPotentialBlockSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHEMV_V2_64] = std::bind(&TallyServer::handle_cublasChemv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNINITTRANSFORMDEST] = std::bind(&TallyServer::handle_cudnnInitTransformDest, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODEGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaGraphKernelNodeGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMGETASYNCERROR] = std::bind(&TallyServer::handle_pncclCommGetAsyncError, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTPSV_V2_64] = std::bind(&TallyServer::handle_cublasCtpsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLGROUPEND] = std::bind(&TallyServer::handle_ncclGroupEnd, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETNORMALIZATIONFORWARDTRAININGWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetNormalizationForwardTrainingWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMM3M] = std::bind(&TallyServer::handle_cublasZgemm3m, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DFROMARRAYASYNC] = std::bind(&TallyServer::handle_cudaMemcpy2DFromArrayAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCHOST] = std::bind(&TallyServer::handle_cudaMallocHost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASICAMIN_V2_64] = std::bind(&TallyServer::handle_cublasIcamin_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDRNNBACKWARDWEIGHTSALGORITHMEX] = std::bind(&TallyServer::handle_cudnnFindRNNBackwardWeightsAlgorithmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECEXTERNALSEMAPHORESSIGNALNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecExternalSemaphoresSignalNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETPERSISTENTRNNPLAN] = std::bind(&TallyServer::handle_cudnnSetPersistentRNNPlan, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNQUERYRUNTIMEERROR] = std::bind(&TallyServer::handle_cudnnQueryRuntimeError, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMINITALL] = std::bind(&TallyServer::handle_pncclCommInitAll, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETEXPORTTABLE] = std::bind(&TallyServer::handle_cudaGetExportTable, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLREDOPDESTROY] = std::bind(&TallyServer::handle_ncclRedOpDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASISAMIN_V2] = std::bind(&TallyServer::handle_cublasIsamin_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXDISABLEPEERACCESS] = std::bind(&TallyServer::handle_cuCtxDisablePeerAccess, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMALLOCNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemAllocNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHHOSTNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphHostNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDDATA_V8] = std::bind(&TallyServer::handle_cudnnRNNBackwardData_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateDropoutDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYDESTROY] = std::bind(&TallyServer::handle_cuArrayDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSCAL_V2] = std::bind(&TallyServer::handle_cublasDscal_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTCREATE] = std::bind(&TallyServer::handle_cublasLtCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYRKX] = std::bind(&TallyServer::handle_cublasDsyrkx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRSMBATCHED] = std::bind(&TallyServer::handle_cublasCtrsmBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONGROUPCOUNT] = std::bind(&TallyServer::handle_cudnnGetConvolutionGroupCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLSETATTRIBUTE] = std::bind(&TallyServer::handle_cudaMemPoolSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTPSV_V2] = std::bind(&TallyServer::handle_cublasDtpsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHREMOVEDEPENDENCIES] = std::bind(&TallyServer::handle_cuGraphRemoveDependencies, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEFILTERDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateFilterDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETBATCHNORMALIZATIONBACKWARDEXWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetBatchNormalizationBackwardExWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETADDRESSRANGE_V2] = std::bind(&TallyServer::handle_cuMemGetAddressRange_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTENSORMAPENCODEIM2COL] = std::bind(&TallyServer::handle_cuTensorMapEncodeIm2col, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTBMV_V2_64] = std::bind(&TallyServer::handle_cublasZtbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRMM_V2] = std::bind(&TallyServer::handle_cublasStrmm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMENDCAPTURE] = std::bind(&TallyServer::handle_cuStreamEndCapture, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSBMV_V2_64] = std::bind(&TallyServer::handle_cublasDsbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFUNCSETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cudaFuncSetSharedMemConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSETASYNC] = std::bind(&TallyServer::handle_cudaMemsetAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETTENSOR4DDESCRIPTOREX] = std::bind(&TallyServer::handle_cudnnSetTensor4dDescriptorEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEVENTWAITNODEGETEVENT] = std::bind(&TallyServer::handle_cudaGraphEventWaitNodeGetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHGEMMBATCHED_64] = std::bind(&TallyServer::handle_cublasHgemmBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHPR2_V2_64] = std::bind(&TallyServer::handle_cublasZhpr2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEXTERNALSEMAPHORESSIGNALNODE] = std::bind(&TallyServer::handle_cuGraphAddExternalSemaphoresSignalNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET3DASYNC] = std::bind(&TallyServer::handle_cudaMemset3DAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERK3MEX] = std::bind(&TallyServer::handle_cublasCherk3mEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTGETRESOURCEVIEWDESC] = std::bind(&TallyServer::handle_cuTexObjectGetResourceViewDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYALGORITHMPERFORMANCE] = std::bind(&TallyServer::handle_cudnnDestroyAlgorithmPerformance, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEAM_64] = std::bind(&TallyServer::handle_cublasDgeam_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyDropoutDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYMM_V2_64] = std::bind(&TallyServer::handle_cublasSsymm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONBACKWARD] = std::bind(&TallyServer::handle_cudnnBatchNormalizationBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET2D] = std::bind(&TallyServer::handle_cudaMemset2D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETVALIDDEVICES] = std::bind(&TallyServer::handle_cudaSetValidDevices, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZROTG_V2] = std::bind(&TallyServer::handle_cublasZrotg_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTBSV_V2_64] = std::bind(&TallyServer::handle_cublasStbsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASNRM2EX_64] = std::bind(&TallyServer::handle_cublasNrm2Ex_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASIGNALEXTERNALSEMAPHORESASYNC_V2] = std::bind(&TallyServer::handle_cudaSignalExternalSemaphoresAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMCUDEVICE] = std::bind(&TallyServer::handle_ncclCommCuDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATESEQDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateSeqDataDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHPR_V2] = std::bind(&TallyServer::handle_cublasChpr_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEXTERNALMEMORYGETMAPPEDBUFFER] = std::bind(&TallyServer::handle_cudaExternalMemoryGetMappedBuffer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMCPYNODESETPARAMS1D] = std::bind(&TallyServer::handle_cudaGraphExecMemcpyNodeSetParams1D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDDATAWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardDataWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYPERSISTENTRNNPLAN] = std::bind(&TallyServer::handle_cudnnDestroyPersistentRNNPlan, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUKERNELGETFUNCTION] = std::bind(&TallyServer::handle_cuKernelGetFunction, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD16_V2] = std::bind(&TallyServer::handle_cuMemsetD16_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEFUSEDOPSCONSTPARAMPACK] = std::bind(&TallyServer::handle_cudnnCreateFusedOpsConstParamPack, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSAVEALGORITHM] = std::bind(&TallyServer::handle_cudnnSaveAlgorithm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTENSORMAPREPLACEADDRESS] = std::bind(&TallyServer::handle_cuTensorMapReplaceAddress, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDCOPY_V2_64] = std::bind(&TallyServer::handle_cublasDcopy_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETGRAPHMEMATTRIBUTE] = std::bind(&TallyServer::handle_cudaDeviceGetGraphMemAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADESTROYTEXTUREOBJECT] = std::bind(&TallyServer::handle_cudaDestroyTextureObject, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPOINTERGETATTRIBUTES] = std::bind(&TallyServer::handle_cuPointerGetAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEUNLOAD] = std::bind(&TallyServer::handle_cuModuleUnload, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D16ASYNC] = std::bind(&TallyServer::handle_cuMemsetD2D16Async, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHDEBUGDOTPRINT] = std::bind(&TallyServer::handle_cudaGraphDebugDotPrint, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERKX_64] = std::bind(&TallyServer::handle_cublasCherkx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSPR_V2_64] = std::bind(&TallyServer::handle_cublasDspr_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3MBATCHED_64] = std::bind(&TallyServer::handle_cublasCgemm3mBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMDESTROY] = std::bind(&TallyServer::handle_pncclCommDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETKERNEL] = std::bind(&TallyServer::handle_cudaGetKernel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEAM] = std::bind(&TallyServer::handle_cublasCgeam, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetDropoutDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCDESTROYPROGRAM] = std::bind(&TallyServer::handle_nvrtcDestroyProgram, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETREDUCETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetReduceTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLGETERROR] = std::bind(&TallyServer::handle_pncclGetError, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWRITEVALUE64_V2] = std::bind(&TallyServer::handle_cuStreamWriteValue64_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAARRAYGETSPARSEPROPERTIES] = std::bind(&TallyServer::handle_cudaArrayGetSparseProperties, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETFLAGS] = std::bind(&TallyServer::handle_cuTexRefGetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASICAMIN_V2] = std::bind(&TallyServer::handle_cublasIcamin_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETMIPMAPPEDARRAYLEVEL] = std::bind(&TallyServer::handle_cudaGetMipmappedArrayLevel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTGETSTATUSNAME] = std::bind(&TallyServer::handle_cublasLtGetStatusName, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASTSSGEMVSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasTSSgemvStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTALLOC] = std::bind(&TallyServer::handle_cuMemHostAlloc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLIMPORTPOINTER] = std::bind(&TallyServer::handle_cuMemPoolImportPointer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETATTNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetAttnDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAARRAYGETMEMORYREQUIREMENTS] = std::bind(&TallyServer::handle_cudaArrayGetMemoryRequirements, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYATOD_V2] = std::bind(&TallyServer::handle_cuMemcpyAtoD_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDRNNFORWARDINFERENCEALGORITHMEX] = std::bind(&TallyServer::handle_cudnnFindRNNForwardInferenceAlgorithmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETSTATUSSTRING] = std::bind(&TallyServer::handle_cublasGetStatusString, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETNUMSUPPORTEDARCHS] = std::bind(&TallyServer::handle_nvrtcGetNumSupportedArchs, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRSM_V2_64] = std::bind(&TallyServer::handle_cublasZtrsm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTGETFLAGS] = std::bind(&TallyServer::handle_cudaHostGetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFUSEDOPSEXECUTE] = std::bind(&TallyServer::handle_cudnnFusedOpsExecute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZCOPY_V2_64] = std::bind(&TallyServer::handle_cublasZcopy_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMIPMAPFILTERMODE] = std::bind(&TallyServer::handle_cuTexRefSetMipmapFilterMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRKX] = std::bind(&TallyServer::handle_cublasCsyrkx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODEFINDINCLONE] = std::bind(&TallyServer::handle_cudaGraphNodeFindInClone, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETV] = std::bind(&TallyServer::handle_cuParamSetv, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSMAPRESOURCES] = std::bind(&TallyServer::handle_cudaGraphicsMapResources, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDCONVOLUTIONBACKWARDFILTERALGORITHM] = std::bind(&TallyServer::handle_cudnnFindConvolutionBackwardFilterAlgorithm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHDESTROY] = std::bind(&TallyServer::handle_cuGraphDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYATOH_V2] = std::bind(&TallyServer::handle_cuMemcpyAtoH_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROT_V2_64] = std::bind(&TallyServer::handle_cublasDrot_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETBORDERCOLOR] = std::bind(&TallyServer::handle_cuTexRefGetBorderColor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRKEX] = std::bind(&TallyServer::handle_cublasCsyrkEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPOOLINGNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetPoolingNdDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULIBRARYGETKERNEL] = std::bind(&TallyServer::handle_cuLibraryGetKernel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYGETMEMORYREQUIREMENTS] = std::bind(&TallyServer::handle_cuArrayGetMemoryRequirements, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM_V2_64] = std::bind(&TallyServer::handle_cublasCgemm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETCUBIN] = std::bind(&TallyServer::handle_nvrtcGetCUBIN, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASTSTGEMVSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasTSTgemvStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYFROMSYMBOL] = std::bind(&TallyServer::handle_cudaMemcpyFromSymbol, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTLOGGERSETCALLBACK] = std::bind(&TallyServer::handle_cublasLtLoggerSetCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGET] = std::bind(&TallyServer::handle_cuDeviceGet, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSWAP_V2] = std::bind(&TallyServer::handle_cublasCswap_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMFREENODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemFreeNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTBSV_V2_64] = std::bind(&TallyServer::handle_cublasCtbsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKCOMPLETE] = std::bind(&TallyServer::handle_cuLinkComplete, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMRELEASE] = std::bind(&TallyServer::handle_cuMemRelease, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMIPMAPLEVELCLAMP] = std::bind(&TallyServer::handle_cuTexRefSetMipmapLevelClamp, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETFLAGS] = std::bind(&TallyServer::handle_cudaStreamGetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULIBRARYLOADFROMFILE] = std::bind(&TallyServer::handle_cuLibraryLoadFromFile, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUEXTERNALMEMORYGETMAPPEDMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cuExternalMemoryGetMappedMipmappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDMATINVBATCHED] = std::bind(&TallyServer::handle_cublasDmatinvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYR2K_V2_64] = std::bind(&TallyServer::handle_cublasDsyr2k_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLREDOPCREATEPREMULSUM] = std::bind(&TallyServer::handle_ncclRedOpCreatePreMulSum, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMCREATE] = std::bind(&TallyServer::handle_cudaStreamCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREEARRAY] = std::bind(&TallyServer::handle_cudaFreeArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGETERRORNAME] = std::bind(&TallyServer::handle_cuGetErrorName, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATERNNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateRNNDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEFUSEDOPSPLAN] = std::bind(&TallyServer::handle_cudnnCreateFusedOpsPlan, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROTM_V2_64] = std::bind(&TallyServer::handle_cublasSrotm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRSV_V2_64] = std::bind(&TallyServer::handle_cublasCtrsv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETSMCOUNTTARGET] = std::bind(&TallyServer::handle_cublasSetSmCountTarget, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHINSTANTIATEWITHFLAGS] = std::bind(&TallyServer::handle_cudaGraphInstantiateWithFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETFILTER4DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetFilter4dDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETSTREAM] = std::bind(&TallyServer::handle_cudnnSetStream, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGETRIBATCHED] = std::bind(&TallyServer::handle_cublasDgetriBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLGETATTRIBUTE] = std::bind(&TallyServer::handle_cuMemPoolGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMCPYNODESETPARAMSFROMSYMBOL] = std::bind(&TallyServer::handle_cudaGraphExecMemcpyNodeSetParamsFromSymbol, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMINITRANK] = std::bind(&TallyServer::handle_pncclCommInitRank, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTIONMATHTYPE] = std::bind(&TallyServer::handle_cudnnSetConvolutionMathType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSPR2_V2] = std::bind(&TallyServer::handle_cublasSspr2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXPOPCURRENT_V2] = std::bind(&TallyServer::handle_cuCtxPopCurrent_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASTSTGEMVBATCHED_64] = std::bind(&TallyServer::handle_cublasTSTgemvBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETSUPPORTEDARCHS] = std::bind(&TallyServer::handle_nvrtcGetSupportedArchs, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLRECV] = std::bind(&TallyServer::handle_ncclRecv, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCALLBACK] = std::bind(&TallyServer::handle_cudnnGetCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSUNMAPRESOURCES] = std::bind(&TallyServer::handle_cuGraphicsUnmapResources, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCASYNC] = std::bind(&TallyServer::handle_cuMemAllocAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRKX_64] = std::bind(&TallyServer::handle_cublasCsyrkx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYATTNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyAttnDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROTG_V2] = std::bind(&TallyServer::handle_cublasSrotg_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cuTexRefGetMipmappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDRIVERENTRYPOINT] = std::bind(&TallyServer::handle_cudaGetDriverEntryPoint, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYMV_V2] = std::bind(&TallyServer::handle_cublasSsymv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDCONVOLUTIONBACKWARDDATAALGORITHMEX] = std::bind(&TallyServer::handle_cudnnFindConvolutionBackwardDataAlgorithmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYFROMARRAY] = std::bind(&TallyServer::handle_cudaMemcpyFromArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETBYPCIBUSID] = std::bind(&TallyServer::handle_cuDeviceGetByPCIBusId, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLCOMMCUDEVICE] = std::bind(&TallyServer::handle_pncclCommCuDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXCREATE_V2] = std::bind(&TallyServer::handle_cuCtxCreate_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSWORKSPACESIZE_V8] = std::bind(&TallyServer::handle_cudnnGetCTCLossWorkspaceSize_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWRITEVALUE32_V2] = std::bind(&TallyServer::handle_cuStreamWriteValue32_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECCHILDGRAPHNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecChildGraphNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZCOPY_V2] = std::bind(&TallyServer::handle_cublasZcopy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASISAMIN_V2_64] = std::bind(&TallyServer::handle_cublasIsamin_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODESETATTRIBUTE] = std::bind(&TallyServer::handle_cuGraphKernelNodeSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHHOSTNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphHostNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMIMPORTFROMSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cuMemImportFromShareableHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTOD_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoD_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETFILTERNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetFilterNdDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYMV_V2_64] = std::bind(&TallyServer::handle_cublasCsymv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNFORWARDINFERENCEALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetRNNForwardInferenceAlgorithmMaxCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUKERNELSETCACHECONFIG] = std::bind(&TallyServer::handle_cuKernelSetCacheConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXTERNALSEMAPHORESSIGNALNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExternalSemaphoresSignalNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMCREATEWITHPRIORITY] = std::bind(&TallyServer::handle_cuStreamCreateWithPriority, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYPEERASYNC] = std::bind(&TallyServer::handle_cuMemcpyPeerAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUKERNELGETATTRIBUTE] = std::bind(&TallyServer::handle_cuKernelGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECEXTERNALSEMAPHORESWAITNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecExternalSemaphoresWaitNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTPMV_V2] = std::bind(&TallyServer::handle_cublasDtpmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDESTROYEXTERNALMEMORY] = std::bind(&TallyServer::handle_cuDestroyExternalMemory, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTCREATE] = std::bind(&TallyServer::handle_cuTexObjectCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETERRORSTRING] = std::bind(&TallyServer::handle_cudaGetErrorString, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERK_V2_64] = std::bind(&TallyServer::handle_cublasCherk_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONFORWARD] = std::bind(&TallyServer::handle_cudnnConvolutionForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMUPDATECAPTUREDEPENDENCIES] = std::bind(&TallyServer::handle_cuStreamUpdateCaptureDependencies, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETADDRESS_V2] = std::bind(&TallyServer::handle_cuTexRefSetAddress_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETVERSION_V2] = std::bind(&TallyServer::handle_cublasGetVersion_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAOCCUPANCYMAXACTIVECLUSTERS] = std::bind(&TallyServer::handle_cudaOccupancyMaxActiveClusters, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDFILTERALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLBCAST] = std::bind(&TallyServer::handle_pncclBcast, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET2DASYNC] = std::bind(&TallyServer::handle_cudaMemset2DAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNBACKWARDWEIGHTSALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetRNNBackwardWeightsAlgorithmMaxCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLIMPORTFROMSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cudaMemPoolImportFromShareableHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDDATA] = std::bind(&TallyServer::handle_cudnnMultiHeadAttnBackwardData, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPOINTERGETATTRIBUTE] = std::bind(&TallyServer::handle_cuPointerGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXSETFLAGS_V2] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxSetFlags_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaMallocMipmappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYR2_V2_64] = std::bind(&TallyServer::handle_cublasCsyr2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSOFTMAXFORWARD] = std::bind(&TallyServer::handle_cudnnSoftmaxForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCNNINFERVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnCnnInferVersionCheck, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDDATAALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardDataAlgorithmMaxCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETACTIVATIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetActivationDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTDISABLECPUINSTRUCTIONSSETMASK] = std::bind(&TallyServer::handle_cublasLtDisableCpuInstructionsSetMask, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCGETMODULE] = std::bind(&TallyServer::handle_cuFuncGetModule, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHUPLOAD] = std::bind(&TallyServer::handle_cuGraphUpload, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSSUBRESOURCEGETMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaGraphicsSubResourceGetMappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETMEMPOOL] = std::bind(&TallyServer::handle_cudaDeviceSetMemPool, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTPSV_V2] = std::bind(&TallyServer::handle_cublasStpsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONBIASACTIVATIONFORWARD] = std::bind(&TallyServer::handle_cudnnConvolutionBiasActivationForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIZAMAX_V2] = std::bind(&TallyServer::handle_cublasIzamax_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCAXPY_V2] = std::bind(&TallyServer::handle_cublasCaxpy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYR2K_V2] = std::bind(&TallyServer::handle_cublasDsyr2k_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEVENTRECORDNODESETEVENT] = std::bind(&TallyServer::handle_cuGraphEventRecordNodeSetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHER2K_V2_64] = std::bind(&TallyServer::handle_cublasCher2k_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMMBATCHED] = std::bind(&TallyServer::handle_cublasDgemmBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCOPY_V2_64] = std::bind(&TallyServer::handle_cublasScopy_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHCOOPERATIVEKERNEL] = std::bind(&TallyServer::handle_cudaLaunchCooperativeKernel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNBIASMODE] = std::bind(&TallyServer::handle_cudnnSetRNNBiasMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODESETPARAMSFROMSYMBOL] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeSetParamsFromSymbol, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETATTRIBUTE] = std::bind(&TallyServer::handle_cuFuncSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODESETPARAMSTOSYMBOL] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeSetParamsToSymbol, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETATTRIBUTE] = std::bind(&TallyServer::handle_cuDeviceGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICETOTALMEM_V2] = std::bind(&TallyServer::handle_cuDeviceTotalMem_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTGETTEXTUREDESC] = std::bind(&TallyServer::handle_cuTexObjectGetTextureDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYR2K_V2_64] = std::bind(&TallyServer::handle_cublasCsyr2k_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMMBATCHED] = std::bind(&TallyServer::handle_cublasSgemmBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNOPSTRAINVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnOpsTrainVersionCheck, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETACTIVATIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetActivationDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSETCURRENT] = std::bind(&TallyServer::handle_cuCtxSetCurrent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETP2PATTRIBUTE] = std::bind(&TallyServer::handle_cuDeviceGetP2PAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMULTICASTBINDADDR] = std::bind(&TallyServer::handle_cuMulticastBindAddr, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRSMBATCHED_64] = std::bind(&TallyServer::handle_cublasStrsmBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETID] = std::bind(&TallyServer::handle_cuCtxGetId, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPOOLING2DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetPooling2dDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASNRM2EX] = std::bind(&TallyServer::handle_cublasNrm2Ex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHPMV_V2] = std::bind(&TallyServer::handle_cublasZhpmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHGEMM_64] = std::bind(&TallyServer::handle_cublasHgemm_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMALLOCNODE] = std::bind(&TallyServer::handle_cudaGraphAddMemAllocNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasZgemmStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHKERNELEX] = std::bind(&TallyServer::handle_cuLaunchKernelEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDZASUM_V2_64] = std::bind(&TallyServer::handle_cublasDzasum_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETID] = std::bind(&TallyServer::handle_cuStreamGetId, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGEMMEX] = std::bind(&TallyServer::handle_cublasGemmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMCOPYATTRIBUTES] = std::bind(&TallyServer::handle_cudaStreamCopyAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETTENSOR] = std::bind(&TallyServer::handle_cudnnSetTensor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMULTICASTADDDEVICE] = std::bind(&TallyServer::handle_cuMulticastAddDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCGETMEMHANDLE] = std::bind(&TallyServer::handle_cuIpcGetMemHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLGETERRORSTRING] = std::bind(&TallyServer::handle_ncclGetErrorString, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETLRNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetLRNDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONBACKWARDDATA] = std::bind(&TallyServer::handle_cudnnConvolutionBackwardData, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY] = std::bind(&TallyServer::handle_cuMemcpy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSDGMM] = std::bind(&TallyServer::handle_cublasSdgmm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHER2_V2] = std::bind(&TallyServer::handle_cublasZher2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYMM_V2] = std::bind(&TallyServer::handle_cublasSsymm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTENSORMAPENCODETILED] = std::bind(&TallyServer::handle_cuTensorMapEncodeTiled, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTGETVERSION] = std::bind(&TallyServer::handle_cublasLtGetVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETFUSEDOPSVARIANTPARAMPACKATTRIBUTE] = std::bind(&TallyServer::handle_cudnnGetFusedOpsVariantParamPackAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYCONVOLUTIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyConvolutionDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETATOMICSMODE] = std::bind(&TallyServer::handle_cublasGetAtomicsMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADEXCHANGESTREAMCAPTUREMODE] = std::bind(&TallyServer::handle_cudaThreadExchangeStreamCaptureMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARDTRAININGEX] = std::bind(&TallyServer::handle_cudnnRNNForwardTrainingEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETVECTOR] = std::bind(&TallyServer::handle_cublasSetVector, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMFREENODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemFreeNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNPOOLINGFORWARD] = std::bind(&TallyServer::handle_cudnnPoolingForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETMATRIX_64] = std::bind(&TallyServer::handle_cublasGetMatrix_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEMPTYNODE] = std::bind(&TallyServer::handle_cuGraphAddEmptyNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREE] = std::bind(&TallyServer::handle_cudaFree, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGEMMBATCHEDEX] = std::bind(&TallyServer::handle_cublasGemmBatchedEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULDESCDESTROY] = std::bind(&TallyServer::handle_cublasLtMatmulDescDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMALLOCNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemAllocNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMIPMAPPEDARRAYGETMEMORYREQUIREMENTS] = std::bind(&TallyServer::handle_cudaMipmappedArrayGetMemoryRequirements, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetRNNDataDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOC3D] = std::bind(&TallyServer::handle_cudaMalloc3D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETLRNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetLRNDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADGETLIMIT] = std::bind(&TallyServer::handle_cudaThreadGetLimit, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDAXPY_V2] = std::bind(&TallyServer::handle_cublasDaxpy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMRANGEGETATTRIBUTES] = std::bind(&TallyServer::handle_cuMemRangeGetAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYRNNDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyRNNDataDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSDGMM_64] = std::bind(&TallyServer::handle_cublasSdgmm_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADSETLIMIT] = std::bind(&TallyServer::handle_cudaThreadSetLimit, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETFLAGS] = std::bind(&TallyServer::handle_cuTexRefSetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLSETATTRIBUTE] = std::bind(&TallyServer::handle_cuMemPoolSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTDESTROY] = std::bind(&TallyServer::handle_cuTexObjectDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECGETFLAGS] = std::bind(&TallyServer::handle_cuGraphExecGetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHINSTANTIATEWITHFLAGS] = std::bind(&TallyServer::handle_cuGraphInstantiateWithFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATE] = std::bind(&TallyServer::handle_cudnnCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECDESTROY] = std::bind(&TallyServer::handle_cudaGraphExecDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD16ASYNC] = std::bind(&TallyServer::handle_cuMemsetD16Async, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYARRAYTOARRAY] = std::bind(&TallyServer::handle_cudaMemcpyArrayToArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCTCLOSSDESCRIPTOREX] = std::bind(&TallyServer::handle_cudnnSetCTCLossDescriptorEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDCHILDGRAPHNODE] = std::bind(&TallyServer::handle_cudaGraphAddChildGraphNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDRNNBACKWARDDATAALGORITHMEX] = std::bind(&TallyServer::handle_cudnnFindRNNBackwardDataAlgorithmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULPREFERENCESETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatmulPreferenceSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEXTERNALMEMORYGETMAPPEDMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaExternalMemoryGetMappedMipmappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSWAP_V2] = std::bind(&TallyServer::handle_cublasDswap_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASASUMEX_64] = std::bind(&TallyServer::handle_cublasAsumEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYR_V2] = std::bind(&TallyServer::handle_cublasCsyr_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSWAPEX] = std::bind(&TallyServer::handle_cublasSwapEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADRIVERGETVERSION] = std::bind(&TallyServer::handle_cudaDriverGetVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTREGISTER_V2] = std::bind(&TallyServer::handle_cuMemHostRegister_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEVENTWAITNODE] = std::bind(&TallyServer::handle_cuGraphAddEventWaitNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasDgemmStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRSMBATCHED_64] = std::bind(&TallyServer::handle_cublasCtrsmBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHGEMMBATCHED] = std::bind(&TallyServer::handle_cublasHgemmBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMWAITEVENT] = std::bind(&TallyServer::handle_cudaStreamWaitEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDDEPENDENCIES] = std::bind(&TallyServer::handle_cudaGraphAddDependencies, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULIBRARYLOADDATA] = std::bind(&TallyServer::handle_cuLibraryLoadData, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMV_V2] = std::bind(&TallyServer::handle_cublasZgemv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACREATETEXTUREOBJECT] = std::bind(&TallyServer::handle_cudaCreateTextureObject, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTGEX] = std::bind(&TallyServer::handle_cublasRotgEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMSETNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemsetNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNACTIVATIONBACKWARD] = std::bind(&TallyServer::handle_cudnnActivationBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGETRFBATCHED] = std::bind(&TallyServer::handle_cublasSgetrfBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKADDDATA_V2] = std::bind(&TallyServer::handle_cuLinkAddData_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDDGMM] = std::bind(&TallyServer::handle_cublasDdgmm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULIBRARYGETGLOBAL] = std::bind(&TallyServer::handle_cuLibraryGetGlobal, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIDAMAX_V2_64] = std::bind(&TallyServer::handle_cublasIdamax_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADESTROYSURFACEOBJECT] = std::bind(&TallyServer::handle_cudaDestroySurfaceObject, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETATTNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetAttnDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRTTP] = std::bind(&TallyServer::handle_cublasCtrttp, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCCOMPILEPROGRAM] = std::bind(&TallyServer::handle_nvrtcCompileProgram, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHLAUNCH] = std::bind(&TallyServer::handle_cudaGraphLaunch, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXDESTROY_V2] = std::bind(&TallyServer::handle_cuCtxDestroy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETVECTORASYNC_64] = std::bind(&TallyServer::handle_cublasGetVectorAsync_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMFREEHOST] = std::bind(&TallyServer::handle_cuMemFreeHost, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDERIVENORMTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDeriveNormTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAWAITEXTERNALSEMAPHORESASYNC_V2] = std::bind(&TallyServer::handle_cudaWaitExternalSemaphoresAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSPMV_V2] = std::bind(&TallyServer::handle_cublasSspmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETFILTERNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetFilterNdDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMGETINFO] = std::bind(&TallyServer::handle_cudaMemGetInfo, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETINFO_V2] = std::bind(&TallyServer::handle_cuMemGetInfo_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMMBATCHED] = std::bind(&TallyServer::handle_cublasCgemmBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMADVISE_V2] = std::bind(&TallyServer::handle_cuMemAdvise_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTSYNCHRONIZE] = std::bind(&TallyServer::handle_cudaEventSynchronize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHGETNODES] = std::bind(&TallyServer::handle_cudaGraphGetNodes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATELRNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateLRNDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSPR2_V2] = std::bind(&TallyServer::handle_cublasDspr2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMVBATCHED] = std::bind(&TallyServer::handle_cublasCgemvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTGETPROPERTY] = std::bind(&TallyServer::handle_cublasLtGetProperty, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACREATECHANNELDESC] = std::bind(&TallyServer::handle_cudaCreateChannelDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMCPYNODE1D] = std::bind(&TallyServer::handle_cudaGraphAddMemcpyNode1D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYRK_V2_64] = std::bind(&TallyServer::handle_cublasZsyrk_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYMM_V2_64] = std::bind(&TallyServer::handle_cublasDsymm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGBMV_V2_64] = std::bind(&TallyServer::handle_cublasCgbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIAMAXEX] = std::bind(&TallyServer::handle_cublasIamaxEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAOCCUPANCYAVAILABLEDYNAMICSMEMPERBLOCK] = std::bind(&TallyServer::handle_cudaOccupancyAvailableDynamicSMemPerBlock, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHER2_V2_64] = std::bind(&TallyServer::handle_cublasZher2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRK_V2] = std::bind(&TallyServer::handle_cublasCsyrk_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNLINLAYERBIASPARAMS] = std::bind(&TallyServer::handle_cudnnGetRNNLinLayerBiasParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADESTROYEXTERNALSEMAPHORE] = std::bind(&TallyServer::handle_cudaDestroyExternalSemaphore, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETALGORITHMPERFORMANCE] = std::bind(&TallyServer::handle_cudnnSetAlgorithmPerformance, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTELAPSEDTIME] = std::bind(&TallyServer::handle_cudaEventElapsedTime, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYASYNC] = std::bind(&TallyServer::handle_cuMemcpyAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSAXPY_V2] = std::bind(&TallyServer::handle_cublasSaxpy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETBATCHNORMALIZATIONFORWARDTRAININGEXWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHBATCHMEMOPNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphBatchMemOpNodeGetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONBACKWARDFILTER] = std::bind(&TallyServer::handle_cudnnConvolutionBackwardFilter, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULALGOGETHEURISTIC] = std::bind(&TallyServer::handle_cublasLtMatmulAlgoGetHeuristic, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETMATRIXASYNC_64] = std::bind(&TallyServer::handle_cublasSetMatrixAsync_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDOTCEX] = std::bind(&TallyServer::handle_cublasDotcEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCAXPY_V2_64] = std::bind(&TallyServer::handle_cublasCaxpy_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDESTROY_V2] = std::bind(&TallyServer::handle_cublasDestroy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGEMMEX_64] = std::bind(&TallyServer::handle_cublasGemmEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCMANAGED] = std::bind(&TallyServer::handle_cuMemAllocManaged, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHBMV_V2] = std::bind(&TallyServer::handle_cublasZhbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCOREDUMPSETATTRIBUTE] = std::bind(&TallyServer::handle_cuCoredumpSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHGETROOTNODES] = std::bind(&TallyServer::handle_cudaGraphGetRootNodes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETDEVICE] = std::bind(&TallyServer::handle_cuCtxGetDevice, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHBATCHMEMOPNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphBatchMemOpNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLSEND] = std::bind(&TallyServer::handle_pncclSend, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTQUERY] = std::bind(&TallyServer::handle_cuEventQuery, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCDOTC_V2_64] = std::bind(&TallyServer::handle_cublasCdotc_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMIPMAPLEVELBIAS] = std::bind(&TallyServer::handle_cuTexRefSetMipmapLevelBias, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXPUSHCURRENT_V2] = std::bind(&TallyServer::handle_cuCtxPushCurrent_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEAM_64] = std::bind(&TallyServer::handle_cublasCgeam_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCDGMM] = std::bind(&TallyServer::handle_cublasCdgmm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONREORDERTYPE] = std::bind(&TallyServer::handle_cudnnGetConvolutionReorderType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGETEXPORTTABLE] = std::bind(&TallyServer::handle_cuGetExportTable, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASASUMEX] = std::bind(&TallyServer::handle_cublasAsumEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULDESCSETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatmulDescSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMVBATCHED] = std::bind(&TallyServer::handle_cublasSgemvBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSRESOURCEGETMAPPEDMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaGraphicsResourceGetMappedMipmappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTEX] = std::bind(&TallyServer::handle_cublasRotEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNSETCLIP_V8] = std::bind(&TallyServer::handle_cudnnRNNSetClip_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDCREATEDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnBackendCreateDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETMAXDEVICEVERSION] = std::bind(&TallyServer::handle_cudnnGetMaxDeviceVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPROFILERSTOP] = std::bind(&TallyServer::handle_cuProfilerStop, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMFREE_V2] = std::bind(&TallyServer::handle_cuMemFree_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXTERNALSEMAPHORESSIGNALNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExternalSemaphoresSignalNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTDESTROY_V2] = std::bind(&TallyServer::handle_cuEventDestroy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDINITIALIZE] = std::bind(&TallyServer::handle_cudnnBackendInitialize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSMAPRESOURCES] = std::bind(&TallyServer::handle_cuGraphicsMapResources, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNREORDERFILTERANDBIAS] = std::bind(&TallyServer::handle_cudnnReorderFilterAndBias, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULPREFERENCEINIT_INTERNAL] = std::bind(&TallyServer::handle_cublasLtMatmulPreferenceInit_internal, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTGETFLAGS] = std::bind(&TallyServer::handle_cuMemHostGetFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMUSERRANK] = std::bind(&TallyServer::handle_ncclCommUserRank, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMCOPYATTRIBUTES] = std::bind(&TallyServer::handle_cuStreamCopyAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEFLUSHGPUDIRECTRDMAWRITES] = std::bind(&TallyServer::handle_cudaDeviceFlushGPUDirectRDMAWrites, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODECOPYATTRIBUTES] = std::bind(&TallyServer::handle_cudaGraphKernelNodeCopyAttributes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSRESOURCESETMAPFLAGS_V2] = std::bind(&TallyServer::handle_cuGraphicsResourceSetMapFlags_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSDOT_V2_64] = std::bind(&TallyServer::handle_cublasSdot_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASTSSGEMVBATCHED_64] = std::bind(&TallyServer::handle_cublasTSSgemvBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTIONNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetConvolutionNdDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULALGOCONFIGSETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatmulAlgoConfigSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROTM_V2] = std::bind(&TallyServer::handle_cublasSrotm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHGRID] = std::bind(&TallyServer::handle_cuLaunchGrid, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTCREATEWITHFLAGS] = std::bind(&TallyServer::handle_cudaEventCreateWithFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHKERNELEXC] = std::bind(&TallyServer::handle_cudaLaunchKernelExC, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODESETENABLED] = std::bind(&TallyServer::handle_cudaGraphNodeSetEnabled, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETREDUCETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetReduceTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETTEXTUREOBJECTTEXTUREDESC] = std::bind(&TallyServer::handle_cudaGetTextureObjectTextureDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLIMPORTPOINTER] = std::bind(&TallyServer::handle_cudaMemPoolImportPointer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCDOTU_V2] = std::bind(&TallyServer::handle_cublasCdotu_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETTEXTURE1DLINEARMAXWIDTH] = std::bind(&TallyServer::handle_cuDeviceGetTexture1DLinearMaxWidth, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCOPY_V2] = std::bind(&TallyServer::handle_cublasScopy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHGEMMSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasHgemmStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY3DPEER] = std::bind(&TallyServer::handle_cuMemcpy3DPeer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYFUSEDOPSVARIANTPARAMPACK] = std::bind(&TallyServer::handle_cudnnDestroyFusedOpsVariantParamPack, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNTEMPSPACESIZES] = std::bind(&TallyServer::handle_cudnnGetRNNTempSpaceSizes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMSYNCHRONIZE] = std::bind(&TallyServer::handle_cudaStreamSynchronize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLREDUCE] = std::bind(&TallyServer::handle_pncclReduce, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMQUERY] = std::bind(&TallyServer::handle_cuStreamQuery, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMIPMAPFILTERMODE] = std::bind(&TallyServer::handle_cuTexRefGetMipmapFilterMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTBSV_V2] = std::bind(&TallyServer::handle_cublasStbsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETP2PATTRIBUTE] = std::bind(&TallyServer::handle_cudaDeviceGetP2PAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNPROJECTIONLAYERS] = std::bind(&TallyServer::handle_cudnnGetRNNProjectionLayers, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADSYNCHRONIZE] = std::bind(&TallyServer::handle_cudaThreadSynchronize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMAXANISOTROPY] = std::bind(&TallyServer::handle_cuTexRefGetMaxAnisotropy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCCOPY_V2] = std::bind(&TallyServer::handle_cublasCcopy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDCHILDGRAPHNODE] = std::bind(&TallyServer::handle_cuGraphAddChildGraphNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEQRFBATCHED] = std::bind(&TallyServer::handle_cublasDgeqrfBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSYNCHRONIZE] = std::bind(&TallyServer::handle_cuCtxSynchronize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDOTC_V2_64] = std::bind(&TallyServer::handle_cublasZdotc_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMDESTROY_V2] = std::bind(&TallyServer::handle_cuStreamDestroy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD32ASYNC] = std::bind(&TallyServer::handle_cuMemsetD32Async, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHPR2_V2_64] = std::bind(&TallyServer::handle_cublasChpr2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYTOSYMBOL] = std::bind(&TallyServer::handle_cudaMemcpyToSymbol, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRSV_V2] = std::bind(&TallyServer::handle_cublasCtrsv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICECANACCESSPEER] = std::bind(&TallyServer::handle_cuDeviceCanAccessPeer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGERC_V2_64] = std::bind(&TallyServer::handle_cublasZgerc_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTSEX] = std::bind(&TallyServer::handle_cudnnRNNBackwardWeightsEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETOPTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetOpTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETBLOCKSHAPE] = std::bind(&TallyServer::handle_cuFuncSetBlockShape, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD32_V2] = std::bind(&TallyServer::handle_cuMemsetD32_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSDESCRIPTOREX] = std::bind(&TallyServer::handle_cudnnGetCTCLossDescriptorEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMFREEASYNC] = std::bind(&TallyServer::handle_cuMemFreeAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHKERNEL] = std::bind(&TallyServer::handle_cuLaunchKernel, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSPATIALTFSAMPLERFORWARD] = std::bind(&TallyServer::handle_cudnnSpatialTfSamplerForward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYRKX] = std::bind(&TallyServer::handle_cublasSsyrkx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUPROFILERINITIALIZE] = std::bind(&TallyServer::handle_cuProfilerInitialize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY2DUNALIGNED_V2] = std::bind(&TallyServer::handle_cuMemcpy2DUnaligned_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETOPTIXIRSIZE] = std::bind(&TallyServer::handle_nvrtcGetOptiXIRSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDKERNELNODE_V2] = std::bind(&TallyServer::handle_cuGraphAddKernelNode_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDIVISIVENORMALIZATIONBACKWARD] = std::bind(&TallyServer::handle_cudnnDivisiveNormalizationBackward, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDZASUM_V2] = std::bind(&TallyServer::handle_cublasDzasum_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEGETGLOBAL_V2] = std::bind(&TallyServer::handle_cuModuleGetGlobal_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETARRAY] = std::bind(&TallyServer::handle_cuTexRefSetArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETVECTORASYNC_64] = std::bind(&TallyServer::handle_cublasSetVectorAsync_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSNRM2_V2_64] = std::bind(&TallyServer::handle_cublasSnrm2_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMV_V2_64] = std::bind(&TallyServer::handle_cublasZgemv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGELSBATCHED] = std::bind(&TallyServer::handle_cublasDgelsBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICESETMEMPOOL] = std::bind(&TallyServer::handle_cuDeviceSetMemPool, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTPTTR] = std::bind(&TallyServer::handle_cublasZtpttr, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTPTTR] = std::bind(&TallyServer::handle_cublasCtpttr, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCOREDUMPGETATTRIBUTEGLOBAL] = std::bind(&TallyServer::handle_cuCoredumpGetAttributeGlobal, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMMBATCHED_64] = std::bind(&TallyServer::handle_cublasDgemmBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECEVENTRECORDNODESETEVENT] = std::bind(&TallyServer::handle_cuGraphExecEventRecordNodeSetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYCTCLOSSDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyCTCLossDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRMV_V2] = std::bind(&TallyServer::handle_cublasStrmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSRESOURCEGETMAPPEDPOINTER_V2] = std::bind(&TallyServer::handle_cuGraphicsResourceGetMappedPointer_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETADDRESSMODE] = std::bind(&TallyServer::handle_cuTexRefGetAddressMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASTSTGEMVSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasTSTgemvStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateAlgorithmDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRMM_V2] = std::bind(&TallyServer::handle_cublasZtrmm_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETCUBINSIZE] = std::bind(&TallyServer::handle_nvrtcGetCUBINSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLREDOPDESTROY] = std::bind(&TallyServer::handle_pncclRedOpDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRSM_V2_64] = std::bind(&TallyServer::handle_cublasDtrsm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULDESCCREATE] = std::bind(&TallyServer::handle_cublasLtMatmulDescCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHUPLOAD] = std::bind(&TallyServer::handle_cudaGraphUpload, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetCTCLossWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cuFuncSetSharedMemConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCASYNC] = std::bind(&TallyServer::handle_cudaMallocAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSWAP_V2_64] = std::bind(&TallyServer::handle_cublasCswap_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSCAL_V2_64] = std::bind(&TallyServer::handle_cublasSscal_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHER_V2] = std::bind(&TallyServer::handle_cublasCher_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTOHASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoHAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRMV_V2_64] = std::bind(&TallyServer::handle_cublasCtrmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDARUNTIMEGETVERSION] = std::bind(&TallyServer::handle_cudaRuntimeGetVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMV_V2_64] = std::bind(&TallyServer::handle_cublasSgemv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMAXANISOTROPY] = std::bind(&TallyServer::handle_cuTexRefSetMaxAnisotropy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHER2K_V2] = std::bind(&TallyServer::handle_cublasZher2k_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3MSTRIDEDBATCHED_64] = std::bind(&TallyServer::handle_cublasCgemm3mStridedBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTELAPSEDTIME] = std::bind(&TallyServer::handle_cuEventElapsedTime, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMMBATCHED] = std::bind(&TallyServer::handle_cublasZgemmBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHGETROOTNODES] = std::bind(&TallyServer::handle_cuGraphGetRootNodes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZAXPY_V2_64] = std::bind(&TallyServer::handle_cublasZaxpy_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateTensorDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHDESTROYNODE] = std::bind(&TallyServer::handle_cuGraphDestroyNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGERC_V2] = std::bind(&TallyServer::handle_cublasCgerc_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEAM] = std::bind(&TallyServer::handle_cublasSgeam, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBUILDRNNDYNAMIC] = std::bind(&TallyServer::handle_cudnnBuildRNNDynamic, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGETPROCADDRESS_V2] = std::bind(&TallyServer::handle_cuGetProcAddress_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D8ASYNC] = std::bind(&TallyServer::handle_cuMemsetD2D8Async, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYTOARRAY] = std::bind(&TallyServer::handle_cudaMemcpyToArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMV_V2_64] = std::bind(&TallyServer::handle_cublasCgemv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTBMV_V2] = std::bind(&TallyServer::handle_cublasStbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODEGETDEPENDENTNODES] = std::bind(&TallyServer::handle_cuGraphNodeGetDependentNodes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHPR_V2] = std::bind(&TallyServer::handle_cublasZhpr_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETLTOIR] = std::bind(&TallyServer::handle_nvrtcGetLTOIR, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNSETCLIP] = std::bind(&TallyServer::handle_cudnnRNNSetClip, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNIM2COL] = std::bind(&TallyServer::handle_cudnnIm2Col, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHPR2_V2] = std::bind(&TallyServer::handle_cublasZhpr2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRSM_V2_64] = std::bind(&TallyServer::handle_cublasStrsm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYMM_V2_64] = std::bind(&TallyServer::handle_cublasCsymm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECMEMCPYNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecMemcpyNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTMEX_64] = std::bind(&TallyServer::handle_cublasRotmEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMV_V2] = std::bind(&TallyServer::handle_cublasCgemv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetConvolutionForwardWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUINIT] = std::bind(&TallyServer::handle_cuInit, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIAMINEX_64] = std::bind(&TallyServer::handle_cublasIaminEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTBMV_V2_64] = std::bind(&TallyServer::handle_cublasCtbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCROTG_V2] = std::bind(&TallyServer::handle_cublasCrotg_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULALGOCONFIGGETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatmulAlgoConfigGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMUL] = std::bind(&TallyServer::handle_cublasLtMatmul, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSASUM_V2_64] = std::bind(&TallyServer::handle_cublasSasum_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETUUID] = std::bind(&TallyServer::handle_cuDeviceGetUuid, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYR2K_V2] = std::bind(&TallyServer::handle_cublasZsyr2k_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMM3M_64] = std::bind(&TallyServer::handle_cublasZgemm3m_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSSCAL_V2_64] = std::bind(&TallyServer::handle_cublasCsscal_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGERU_V2_64] = std::bind(&TallyServer::handle_cublasZgeru_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETTEXTUREOBJECTRESOURCEDESC] = std::bind(&TallyServer::handle_cudaGetTextureObjectResourceDesc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULPREFERENCEDESTROY] = std::bind(&TallyServer::handle_cublasLtMatmulPreferenceDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NCCLCOMMDESTROY] = std::bind(&TallyServer::handle_ncclCommDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLBROADCAST] = std::bind(&TallyServer::handle_pncclBroadcast, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETMULTIHEADATTNWEIGHTS] = std::bind(&TallyServer::handle_cudnnGetMultiHeadAttnWeights, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGER_V2_64] = std::bind(&TallyServer::handle_cublasDger_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNFORWARDTRAININGALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetRNNForwardTrainingAlgorithmMaxCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMULTICASTCREATE] = std::bind(&TallyServer::handle_cuMulticastCreate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECBATCHMEMOPNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecBatchMemOpNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDSCAL_V2] = std::bind(&TallyServer::handle_cublasZdscal_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDFILTERALGORITHM_V7] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardFilterAlgorithm_v7, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGETRSBATCHED] = std::bind(&TallyServer::handle_cublasSgetrsBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPREFETCHASYNC_V2] = std::bind(&TallyServer::handle_cudaMemPrefetchAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETSTREAM] = std::bind(&TallyServer::handle_cudnnGetStream, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLGROUPEND] = std::bind(&TallyServer::handle_pncclGroupEnd, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphKernelNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETPOOLING2DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetPooling2dDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODEGETDEPENDENTNODES] = std::bind(&TallyServer::handle_cudaGraphNodeGetDependentNodes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRK_V2_64] = std::bind(&TallyServer::handle_cublasCsyrk_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMSETNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemsetNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTINIT_INTERNAL] = std::bind(&TallyServer::handle_cublasLtMatrixLayoutInit_internal, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMULTICASTUNBIND] = std::bind(&TallyServer::handle_cuMulticastUnbind, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNTRANSFORMTENSOREX] = std::bind(&TallyServer::handle_cudnnTransformTensorEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHEMV_V2] = std::bind(&TallyServer::handle_cublasChemv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLGETVERSION] = std::bind(&TallyServer::handle_pncclGetVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIZAMAX_V2_64] = std::bind(&TallyServer::handle_cublasIzamax_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETPOINTERMODE_V2] = std::bind(&TallyServer::handle_cublasSetPointerMode_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETERRORSTRING] = std::bind(&TallyServer::handle_nvrtcGetErrorString, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTPTTR] = std::bind(&TallyServer::handle_cublasStpttr, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGEMMBATCHEDEX_64] = std::bind(&TallyServer::handle_cublasGemmBatchedEx_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEVENTWAITNODE] = std::bind(&TallyServer::handle_cudaGraphAddEventWaitNode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETERRORNAME] = std::bind(&TallyServer::handle_cudaGetErrorName, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNPADDINGMODE] = std::bind(&TallyServer::handle_cudnnGetRNNPaddingMode, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETVECTORASYNC] = std::bind(&TallyServer::handle_cublasGetVectorAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMDESTROY] = std::bind(&TallyServer::handle_cudaStreamDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYR2_V2] = std::bind(&TallyServer::handle_cublasSsyr2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMEXPORTTOSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cuMemExportToShareableHandle, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY3DASYNC] = std::bind(&TallyServer::handle_cudaMemcpy3DAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cuTexRefSetMipmappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETNAME] = std::bind(&TallyServer::handle_cuDeviceGetName, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYMV_V2_64] = std::bind(&TallyServer::handle_cublasZsymv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETCURRENT] = std::bind(&TallyServer::handle_cuCtxGetCurrent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCVERSION] = std::bind(&TallyServer::handle_nvrtcVersion, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLEXPORTPOINTER] = std::bind(&TallyServer::handle_cuMemPoolExportPointer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGBMV_V2_64] = std::bind(&TallyServer::handle_cublasZgbmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET] = std::bind(&TallyServer::handle_cudaMemset, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSRESOURCEGETMAPPEDMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cuGraphicsResourceGetMappedMipmappedArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXRESETPERSISTINGL2CACHE] = std::bind(&TallyServer::handle_cuCtxResetPersistingL2Cache, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::__CUDAREGISTERFUNCTION] = std::bind(&TallyServer::handle___cudaRegisterFunction, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNNORMALIZATIONFORWARDINFERENCE] = std::bind(&TallyServer::handle_cudnnNormalizationForwardInference, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXGETSTATE] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxGetState, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3MSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasCgemm3mStridedBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cudaDeviceGetSharedMemConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFINDCONVOLUTIONFORWARDALGORITHM] = std::bind(&TallyServer::handle_cudnnFindConvolutionForwardAlgorithm, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECEVENTRECORDNODESETEVENT] = std::bind(&TallyServer::handle_cudaGraphExecEventRecordNodeSetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMCREATEWITHPRIORITY] = std::bind(&TallyServer::handle_cudaStreamCreateWithPriority, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetCTCLossDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCARRAY] = std::bind(&TallyServer::handle_cudaMallocArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMM_V2_64] = std::bind(&TallyServer::handle_cublasDgemm_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetAlgorithmDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEVENTRECORDNODESETEVENT] = std::bind(&TallyServer::handle_cudaGraphEventRecordNodeSetEvent, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGETRIBATCHED] = std::bind(&TallyServer::handle_cublasZgetriBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetDropoutDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTMEX] = std::bind(&TallyServer::handle_cublasRotmEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYAVAILABLEDYNAMICSMEMPERBLOCK] = std::bind(&TallyServer::handle_cuOccupancyAvailableDynamicSMemPerBlock, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSORWITHFLAGS] = std::bind(&TallyServer::handle_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONFORWARDTRAINING] = std::bind(&TallyServer::handle_cudnnBatchNormalizationForwardTraining, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYR2K_V2] = std::bind(&TallyServer::handle_cublasSsyr2k_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNBACKWARDDATAALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetRNNBackwardDataAlgorithmMaxCount, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMVBATCHED_64] = std::bind(&TallyServer::handle_cublasDgemvBatched_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETNORMALIZATIONBACKWARDWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetNormalizationBackwardWorkspaceSize, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSPMV_V2_64] = std::bind(&TallyServer::handle_cublasSspmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETMULTIHEADATTNBUFFERS] = std::bind(&TallyServer::handle_cudnnGetMultiHeadAttnBuffers, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNGETCLIP] = std::bind(&TallyServer::handle_cudnnRNNGetClip, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETTENSORSIZEINBYTES] = std::bind(&TallyServer::handle_cudnnGetTensorSizeInBytes, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSROT_V2] = std::bind(&TallyServer::handle_cublasCsrot_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLALLGATHER] = std::bind(&TallyServer::handle_pncclAllGather, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTPMV_V2_64] = std::bind(&TallyServer::handle_cublasStpmv_v2_64, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECHOSTNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecHostNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETFUSEDOPSCONSTPARAMPACKATTRIBUTE] = std::bind(&TallyServer::handle_cudnnGetFusedOpsConstParamPackAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHREMOVEDEPENDENCIES] = std::bind(&TallyServer::handle_cudaGraphRemoveDependencies, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULELOADDATA] = std::bind(&TallyServer::handle_cuModuleLoadData, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cuCtxSetSharedMemConfig, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTBMV_V2] = std::bind(&TallyServer::handle_cublasZtbmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::__CUDAREGISTERFATBINARY] = std::bind(&TallyServer::handle___cudaRegisterFatBinary, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::PNCCLREDOPCREATEPREMULSUM] = std::bind(&TallyServer::handle_pncclRedOpCreatePreMulSum, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTION2DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetConvolution2dDescriptor, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZAXPY_V2] = std::bind(&TallyServer::handle_cublasZaxpy_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYFROMARRAYASYNC] = std::bind(&TallyServer::handle_cudaMemcpyFromArrayAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATMULPREFERENCEGETATTRIBUTE] = std::bind(&TallyServer::handle_cublasLtMatmulPreferenceGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY3DPEERASYNC] = std::bind(&TallyServer::handle_cudaMemcpy3DPeerAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaMemPoolGetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNGETCLIP_V8] = std::bind(&TallyServer::handle_cudnnRNNGetClip_v8, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGEMMSTRIDEDBATCHEDEX] = std::bind(&TallyServer::handle_cublasGemmStridedBatchedEx, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHPR2_V2] = std::bind(&TallyServer::handle_cublasChpr2_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTODASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoDAsync_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY3DPEERASYNC] = std::bind(&TallyServer::handle_cuMemcpy3DPeerAsync, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETARRAY] = std::bind(&TallyServer::handle_cuTexRefGetArray, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLTMATRIXTRANSFORMDESCDESTROY] = std::bind(&TallyServer::handle_cublasLtMatrixTransformDescDestroy, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRMV_V2] = std::bind(&TallyServer::handle_cublasZtrmv_v2, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETMATRIX] = std::bind(&TallyServer::handle_cublasGetMatrix, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMADVISE] = std::bind(&TallyServer::handle_cuMemAdvise, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECMEMSETNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecMemsetNodeSetParams, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYPEER] = std::bind(&TallyServer::handle_cuMemcpyPeer, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEQRFBATCHED] = std::bind(&TallyServer::handle_cublasCgeqrfBatched, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	cuda_api_handler_map[CUDA_API_ENUM::CUKERNELSETATTRIBUTE] = std::bind(&TallyServer::handle_cuKernelSetAttribute, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
}
void TallyServer::handle_cuGetErrorString(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGetErrorString");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGetErrorName(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGetErrorName");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuInit(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuInit");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDriverGetVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDriverGetVersion");
	auto args = (struct cuDriverGetVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDriverGetVersionResponse), alignof(cuDriverGetVersionResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDriverGetVersionResponse*>(responsePayload);
            response->err = cuDriverGetVersion(
				(args->driverVersion ? &(response->driverVersion) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGet(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGet");
	auto args = (struct cuDeviceGetArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetResponse), alignof(cuDeviceGetResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetResponse*>(responsePayload);
            response->err = cuDeviceGet(
				(args->device ? &(response->device) : NULL),
				args->ordinal
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetCount");
	auto args = (struct cuDeviceGetCountArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetCountResponse), alignof(cuDeviceGetCountResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetCountResponse*>(responsePayload);
            response->err = cuDeviceGetCount(
				(args->count ? &(response->count) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetUuid(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetUuid");
	auto args = (struct cuDeviceGetUuidArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetUuidResponse), alignof(cuDeviceGetUuidResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetUuidResponse*>(responsePayload);
            response->err = cuDeviceGetUuid(
				(args->uuid ? &(response->uuid) : NULL),
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetUuid_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetUuid_v2");
	auto args = (struct cuDeviceGetUuid_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetUuid_v2Response), alignof(cuDeviceGetUuid_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetUuid_v2Response*>(responsePayload);
            response->err = cuDeviceGetUuid_v2(
				(args->uuid ? &(response->uuid) : NULL),
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetLuid(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetLuid");
	auto args = (struct cuDeviceGetLuidArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetLuidResponse), alignof(cuDeviceGetLuidResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetLuidResponse*>(responsePayload);
            response->err = cuDeviceGetLuid(
				(args->luid ? &(response->luid) : NULL),
				(args->deviceNodeMask ? &(response->deviceNodeMask) : NULL),
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceTotalMem_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceTotalMem_v2");
	auto args = (struct cuDeviceTotalMem_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceTotalMem_v2Response), alignof(cuDeviceTotalMem_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceTotalMem_v2Response*>(responsePayload);
            response->err = cuDeviceTotalMem_v2(
				(args->bytes ? &(response->bytes) : NULL),
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetTexture1DLinearMaxWidth(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetTexture1DLinearMaxWidth");
	auto args = (struct cuDeviceGetTexture1DLinearMaxWidthArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetTexture1DLinearMaxWidthResponse), alignof(cuDeviceGetTexture1DLinearMaxWidthResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetTexture1DLinearMaxWidthResponse*>(responsePayload);
            response->err = cuDeviceGetTexture1DLinearMaxWidth(
				(args->maxWidthInElements ? &(response->maxWidthInElements) : NULL),
				args->format,
				args->numChannels,
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetAttribute");
	auto args = (struct cuDeviceGetAttributeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetAttributeResponse), alignof(cuDeviceGetAttributeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetAttributeResponse*>(responsePayload);
            response->err = cuDeviceGetAttribute(
				(args->pi ? &(response->pi) : NULL),
				args->attrib,
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetNvSciSyncAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetNvSciSyncAttributes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDeviceSetMemPool(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceSetMemPool");
	auto args = (struct cuDeviceSetMemPoolArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuDeviceSetMemPool(
				args->dev,
				args->pool
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetMemPool(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetMemPool");
	auto args = (struct cuDeviceGetMemPoolArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetMemPoolResponse), alignof(cuDeviceGetMemPoolResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetMemPoolResponse*>(responsePayload);
            response->err = cuDeviceGetMemPool(
				(args->pool ? &(response->pool) : NULL),
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetDefaultMemPool(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetDefaultMemPool");
	auto args = (struct cuDeviceGetDefaultMemPoolArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetDefaultMemPoolResponse), alignof(cuDeviceGetDefaultMemPoolResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetDefaultMemPoolResponse*>(responsePayload);
            response->err = cuDeviceGetDefaultMemPool(
				(args->pool_out ? &(response->pool_out) : NULL),
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetExecAffinitySupport(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetExecAffinitySupport");
	auto args = (struct cuDeviceGetExecAffinitySupportArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetExecAffinitySupportResponse), alignof(cuDeviceGetExecAffinitySupportResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetExecAffinitySupportResponse*>(responsePayload);
            response->err = cuDeviceGetExecAffinitySupport(
				(args->pi ? &(response->pi) : NULL),
				args->type,
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuFlushGPUDirectRDMAWrites(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuFlushGPUDirectRDMAWrites");
	auto args = (struct cuFlushGPUDirectRDMAWritesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuFlushGPUDirectRDMAWrites(
				args->target,
				args->scope
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceGetProperties(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetProperties");
	auto args = (struct cuDeviceGetPropertiesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetPropertiesResponse), alignof(cuDeviceGetPropertiesResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetPropertiesResponse*>(responsePayload);
            response->err = cuDeviceGetProperties(
				(args->prop ? &(response->prop) : NULL),
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDeviceComputeCapability(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceComputeCapability");
	auto args = (struct cuDeviceComputeCapabilityArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceComputeCapabilityResponse), alignof(cuDeviceComputeCapabilityResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceComputeCapabilityResponse*>(responsePayload);
            response->err = cuDeviceComputeCapability(
				(args->major ? &(response->major) : NULL),
				(args->minor ? &(response->minor) : NULL),
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDevicePrimaryCtxRetain(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDevicePrimaryCtxRetain");
	auto args = (struct cuDevicePrimaryCtxRetainArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDevicePrimaryCtxRetainResponse), alignof(cuDevicePrimaryCtxRetainResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDevicePrimaryCtxRetainResponse*>(responsePayload);
            response->err = cuDevicePrimaryCtxRetain(
				(args->pctx ? &(response->pctx) : NULL),
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDevicePrimaryCtxRelease_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDevicePrimaryCtxRelease_v2");
	auto args = (struct cuDevicePrimaryCtxRelease_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuDevicePrimaryCtxRelease_v2(
				args->dev
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDevicePrimaryCtxGetState(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDevicePrimaryCtxGetState");
	auto args = (struct cuDevicePrimaryCtxGetStateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDevicePrimaryCtxGetStateResponse), alignof(cuDevicePrimaryCtxGetStateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDevicePrimaryCtxGetStateResponse*>(responsePayload);
            response->err = cuDevicePrimaryCtxGetState(
				args->dev,
				(args->flags ? &(response->flags) : NULL),
				(args->active ? &(response->active) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuDevicePrimaryCtxReset_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDevicePrimaryCtxReset_v2");
	auto args = (struct cuDevicePrimaryCtxReset_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuDevicePrimaryCtxReset_v2(
				args->dev
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxCreate_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxCreate_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCtxCreate_v3(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxCreate_v3");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCtxDestroy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxDestroy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCtxPushCurrent_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxPushCurrent_v2");
	auto args = (struct cuCtxPushCurrent_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuCtxPushCurrent_v2(
				args->ctx
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxPopCurrent_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxPopCurrent_v2");
	auto args = (struct cuCtxPopCurrent_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxPopCurrent_v2Response), alignof(cuCtxPopCurrent_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxPopCurrent_v2Response*>(responsePayload);
            response->err = cuCtxPopCurrent_v2(
				(args->pctx ? &(response->pctx) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxSetCurrent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxSetCurrent");
	auto args = (struct cuCtxSetCurrentArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuCtxSetCurrent(
				args->ctx
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxGetCurrent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetCurrent");
	auto args = (struct cuCtxGetCurrentArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetCurrentResponse), alignof(cuCtxGetCurrentResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetCurrentResponse*>(responsePayload);
            response->err = cuCtxGetCurrent(
				(args->pctx ? &(response->pctx) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxGetDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetDevice");
	auto args = (struct cuCtxGetDeviceArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetDeviceResponse), alignof(cuCtxGetDeviceResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetDeviceResponse*>(responsePayload);
            response->err = cuCtxGetDevice(
				(args->device ? &(response->device) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxGetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetFlags");
	auto args = (struct cuCtxGetFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetFlagsResponse), alignof(cuCtxGetFlagsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetFlagsResponse*>(responsePayload);
            response->err = cuCtxGetFlags(
				(args->flags ? &(response->flags) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxSetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxSetFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCtxGetId(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetId");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCtxSetLimit(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxSetLimit");
	auto args = (struct cuCtxSetLimitArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuCtxSetLimit(
				args->limit,
				args->value
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxGetLimit(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetLimit");
	auto args = (struct cuCtxGetLimitArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetLimitResponse), alignof(cuCtxGetLimitResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetLimitResponse*>(responsePayload);
            response->err = cuCtxGetLimit(
				(args->pvalue ? &(response->pvalue) : NULL),
				args->limit
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxGetCacheConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetCacheConfig");
	auto args = (struct cuCtxGetCacheConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetCacheConfigResponse), alignof(cuCtxGetCacheConfigResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetCacheConfigResponse*>(responsePayload);
            response->err = cuCtxGetCacheConfig(
				(args->pconfig ? &(response->pconfig) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxSetCacheConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxSetCacheConfig");
	auto args = (struct cuCtxSetCacheConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuCtxSetCacheConfig(
				args->config
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxGetSharedMemConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetSharedMemConfig");
	auto args = (struct cuCtxGetSharedMemConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetSharedMemConfigResponse), alignof(cuCtxGetSharedMemConfigResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetSharedMemConfigResponse*>(responsePayload);
            response->err = cuCtxGetSharedMemConfig(
				(args->pConfig ? &(response->pConfig) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxSetSharedMemConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxSetSharedMemConfig");
	auto args = (struct cuCtxSetSharedMemConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuCtxSetSharedMemConfig(
				args->config
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxGetApiVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetApiVersion");
	auto args = (struct cuCtxGetApiVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetApiVersionResponse), alignof(cuCtxGetApiVersionResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetApiVersionResponse*>(responsePayload);
            response->err = cuCtxGetApiVersion(
				args->ctx,
				(args->version ? &(response->version) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxGetStreamPriorityRange(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetStreamPriorityRange");
	auto args = (struct cuCtxGetStreamPriorityRangeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetStreamPriorityRangeResponse), alignof(cuCtxGetStreamPriorityRangeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetStreamPriorityRangeResponse*>(responsePayload);
            response->err = cuCtxGetStreamPriorityRange(
				(args->leastPriority ? &(response->leastPriority) : NULL),
				(args->greatestPriority ? &(response->greatestPriority) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxResetPersistingL2Cache(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxResetPersistingL2Cache");
	auto args = (struct cuCtxResetPersistingL2CacheArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuCtxResetPersistingL2Cache(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxGetExecAffinity(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxGetExecAffinity");
	auto args = (struct cuCtxGetExecAffinityArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxGetExecAffinityResponse), alignof(cuCtxGetExecAffinityResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxGetExecAffinityResponse*>(responsePayload);
            response->err = cuCtxGetExecAffinity(
				(args->pExecAffinity ? &(response->pExecAffinity) : NULL),
				args->type
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxAttach(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxAttach");
	auto args = (struct cuCtxAttachArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuCtxAttachResponse), alignof(cuCtxAttachResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuCtxAttachResponse*>(responsePayload);
            response->err = cuCtxAttach(
				(args->pctx ? &(response->pctx) : NULL),
				args->flags
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuCtxDetach(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxDetach");
	auto args = (struct cuCtxDetachArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuCtxDetach(
				args->ctx
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuModuleLoad(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleLoad");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuModuleGetLoadingMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleGetLoadingMode");
	auto args = (struct cuModuleGetLoadingModeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuModuleGetLoadingModeResponse), alignof(cuModuleGetLoadingModeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuModuleGetLoadingModeResponse*>(responsePayload);
            response->err = cuModuleGetLoadingMode(
				(args->mode ? &(response->mode) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuLinkCreate_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLinkCreate_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLinkAddData_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLinkAddData_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLinkAddFile_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLinkAddFile_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLinkComplete(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLinkComplete");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLinkDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLinkDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuModuleGetTexRef(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleGetTexRef");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuModuleGetSurfRef(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuModuleGetSurfRef");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLibraryLoadData(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLibraryLoadData");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLibraryLoadFromFile(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLibraryLoadFromFile");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLibraryUnload(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLibraryUnload");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLibraryGetKernel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLibraryGetKernel");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLibraryGetModule(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLibraryGetModule");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuKernelGetFunction(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuKernelGetFunction");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLibraryGetGlobal(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLibraryGetGlobal");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLibraryGetManaged(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLibraryGetManaged");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLibraryGetUnifiedFunction(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLibraryGetUnifiedFunction");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuKernelGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuKernelGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuKernelSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuKernelSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuKernelSetCacheConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuKernelSetCacheConfig");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemGetInfo_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemGetInfo_v2");
	auto args = (struct cuMemGetInfo_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuMemGetInfo_v2Response), alignof(cuMemGetInfo_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuMemGetInfo_v2Response*>(responsePayload);
            response->err = cuMemGetInfo_v2(
				(args->free ? &(response->free) : NULL),
				(args->total ? &(response->total) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuMemAllocPitch_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAllocPitch_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemGetAddressRange_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemGetAddressRange_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemAllocHost_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAllocHost_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemFreeHost(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemFreeHost");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemHostGetDevicePointer_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemHostGetDevicePointer_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemHostGetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemHostGetFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemAllocManaged(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAllocManaged");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDeviceGetByPCIBusId(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetByPCIBusId");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDeviceGetPCIBusId(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetPCIBusId");
	auto args = (struct cuDeviceGetPCIBusIdArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuDeviceGetPCIBusIdResponse), alignof(cuDeviceGetPCIBusIdResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuDeviceGetPCIBusIdResponse*>(responsePayload);
            response->err = cuDeviceGetPCIBusId(
				(args->pciBusId ? &(response->pciBusId) : NULL),
				args->len,
				args->dev
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuIpcGetEventHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuIpcGetEventHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuIpcOpenEventHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuIpcOpenEventHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuIpcGetMemHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuIpcGetMemHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuIpcOpenMemHandle_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuIpcOpenMemHandle_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuIpcCloseMemHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuIpcCloseMemHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemHostRegister_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemHostRegister_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemHostUnregister(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemHostUnregister");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyPeer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyPeer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyHtoD_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyHtoD_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyDtoH_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyDtoH_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyDtoD_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyDtoD_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyDtoA_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyDtoA_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyAtoD_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyAtoD_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyHtoA_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyHtoA_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyAtoH_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyAtoH_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyAtoA_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyAtoA_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpy2D_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpy2D_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpy2DUnaligned_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpy2DUnaligned_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpy3D_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpy3D_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpy3DPeer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpy3DPeer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyPeerAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyPeerAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyHtoAAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyHtoAAsync_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpyAtoHAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpyAtoHAsync_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpy2DAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpy2DAsync_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpy3DAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpy3DAsync_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemcpy3DPeerAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemcpy3DPeerAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemsetD16_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD16_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemsetD2D8_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD2D8_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemsetD2D16_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD2D16_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemsetD2D32_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD2D32_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemsetD8Async(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD8Async");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemsetD16Async(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD16Async");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemsetD2D8Async(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD2D8Async");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemsetD2D16Async(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD2D16Async");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemsetD2D32Async(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemsetD2D32Async");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuArrayCreate_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuArrayCreate_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuArrayGetDescriptor_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuArrayGetDescriptor_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuArrayGetSparseProperties(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuArrayGetSparseProperties");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMipmappedArrayGetSparseProperties(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMipmappedArrayGetSparseProperties");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuArrayGetMemoryRequirements(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuArrayGetMemoryRequirements");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMipmappedArrayGetMemoryRequirements(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMipmappedArrayGetMemoryRequirements");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuArrayGetPlane(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuArrayGetPlane");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuArrayDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuArrayDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuArray3DCreate_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuArray3DCreate_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuArray3DGetDescriptor_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuArray3DGetDescriptor_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMipmappedArrayCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMipmappedArrayCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMipmappedArrayGetLevel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMipmappedArrayGetLevel");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMipmappedArrayDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMipmappedArrayDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemGetHandleForAddressRange(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemGetHandleForAddressRange");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemAddressReserve(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAddressReserve");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemAddressFree(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAddressFree");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemRelease(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemRelease");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemMap(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemMap");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemMapArrayAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemMapArrayAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemUnmap(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemUnmap");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemSetAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemSetAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemGetAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemGetAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemExportToShareableHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemExportToShareableHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemImportFromShareableHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemImportFromShareableHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemGetAllocationGranularity(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemGetAllocationGranularity");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemGetAllocationPropertiesFromHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemGetAllocationPropertiesFromHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemRetainAllocationHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemRetainAllocationHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemFreeAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemFreeAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolTrimTo(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolTrimTo");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolSetAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolSetAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolGetAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolGetAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemAllocFromPoolAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAllocFromPoolAsync");
	auto args = (struct cuMemAllocFromPoolAsyncArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cuMemAllocFromPoolAsyncResponse), alignof(cuMemAllocFromPoolAsyncResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuMemAllocFromPoolAsyncResponse*>(responsePayload);
            response->err = cuMemAllocFromPoolAsync(
				(args->dptr ? &(response->dptr) : NULL),
				args->bytesize,
				args->pool,
				__stream
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuMemPoolExportToShareableHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolExportToShareableHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolImportFromShareableHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolImportFromShareableHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolExportPointer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolExportPointer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPoolImportPointer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPoolImportPointer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMulticastCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMulticastCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMulticastAddDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMulticastAddDevice");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMulticastBindMem(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMulticastBindMem");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMulticastBindAddr(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMulticastBindAddr");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMulticastUnbind(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMulticastUnbind");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMulticastGetGranularity(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMulticastGetGranularity");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPrefetchAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPrefetchAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemPrefetchAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemPrefetchAsync_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemAdvise(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAdvise");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemAdvise_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemAdvise_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemRangeGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemRangeGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuMemRangeGetAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuMemRangeGetAttributes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuPointerSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuPointerSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuPointerGetAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuPointerGetAttributes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamGetPriority(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamGetPriority");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamGetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamGetFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamGetId(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamGetId");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamGetCtx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamGetCtx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamWaitEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamWaitEvent");
	auto args = (struct cuStreamWaitEventArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuStreamWaitEvent(
				__stream,
				args->hEvent,
				args->Flags
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuStreamAddCallback(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamAddCallback");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamBeginCapture_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamBeginCapture_v2");
	auto args = (struct cuStreamBeginCapture_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuStreamBeginCapture_v2(
				__stream,
				args->mode
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuThreadExchangeStreamCaptureMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuThreadExchangeStreamCaptureMode");
	auto args = (struct cuThreadExchangeStreamCaptureModeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuThreadExchangeStreamCaptureModeResponse), alignof(cuThreadExchangeStreamCaptureModeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuThreadExchangeStreamCaptureModeResponse*>(responsePayload);
            response->err = cuThreadExchangeStreamCaptureMode(
				(args->mode ? &(response->mode) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuStreamIsCapturing(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamIsCapturing");
	auto args = (struct cuStreamIsCapturingArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cuStreamIsCapturingResponse), alignof(cuStreamIsCapturingResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuStreamIsCapturingResponse*>(responsePayload);
            response->err = cuStreamIsCapturing(
				__stream,
				(args->captureStatus ? &(response->captureStatus) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuStreamGetCaptureInfo_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamGetCaptureInfo_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamUpdateCaptureDependencies(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamUpdateCaptureDependencies");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamAttachMemAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamAttachMemAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamQuery(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamQuery");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamDestroy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamDestroy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamCopyAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamCopyAttributes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuEventCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuEventCreate");
	auto args = (struct cuEventCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuEventCreateResponse), alignof(cuEventCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuEventCreateResponse*>(responsePayload);
            response->err = cuEventCreate(
				(args->phEvent ? &(response->phEvent) : NULL),
				args->Flags
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuEventRecord(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuEventRecord");
	auto args = (struct cuEventRecordArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    CUstream __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuEventRecord(
				args->hEvent,
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuEventRecordWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuEventRecordWithFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuEventQuery(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuEventQuery");
	auto args = (struct cuEventQueryArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuEventQuery(
				args->hEvent
            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuEventSynchronize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuEventSynchronize");
	auto args = (struct cuEventSynchronizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuEventSynchronize(
				args->hEvent
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuEventDestroy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuEventDestroy_v2");
	auto args = (struct cuEventDestroy_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuEventDestroy_v2(
				args->hEvent
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuEventElapsedTime(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuEventElapsedTime");
	auto args = (struct cuEventElapsedTimeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuEventElapsedTimeResponse), alignof(cuEventElapsedTimeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuEventElapsedTimeResponse*>(responsePayload);
            response->err = cuEventElapsedTime(
				(args->pMilliseconds ? &(response->pMilliseconds) : NULL),
				args->hStart,
				args->hEnd
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuImportExternalMemory(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuImportExternalMemory");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuExternalMemoryGetMappedBuffer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuExternalMemoryGetMappedBuffer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuExternalMemoryGetMappedMipmappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuExternalMemoryGetMappedMipmappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDestroyExternalMemory(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDestroyExternalMemory");
	auto args = (struct cuDestroyExternalMemoryArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuDestroyExternalMemory(
				args->extMem
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuImportExternalSemaphore(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuImportExternalSemaphore");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuSignalExternalSemaphoresAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuSignalExternalSemaphoresAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuWaitExternalSemaphoresAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuWaitExternalSemaphoresAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDestroyExternalSemaphore(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDestroyExternalSemaphore");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamWaitValue32_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamWaitValue32_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamWaitValue64_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamWaitValue64_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamWriteValue32_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamWriteValue32_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamWriteValue64_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamWriteValue64_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuStreamBatchMemOp_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuStreamBatchMemOp_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuFuncSetSharedMemConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuFuncSetSharedMemConfig");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuFuncGetModule(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuFuncGetModule");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLaunchKernelEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLaunchKernelEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLaunchCooperativeKernel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLaunchCooperativeKernel");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLaunchCooperativeKernelMultiDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLaunchCooperativeKernelMultiDevice");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLaunchHostFunc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLaunchHostFunc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuFuncSetBlockShape(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuFuncSetBlockShape");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuFuncSetSharedSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuFuncSetSharedSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuParamSetSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuParamSetSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuParamSeti(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuParamSeti");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuParamSetf(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuParamSetf");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuParamSetv(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuParamSetv");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLaunch(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLaunch");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLaunchGrid(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLaunchGrid");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuLaunchGridAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuLaunchGridAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuParamSetTexRef(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuParamSetTexRef");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddKernelNode_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddKernelNode_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphKernelNodeGetParams_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphKernelNodeGetParams_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphKernelNodeSetParams_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphKernelNodeSetParams_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddMemcpyNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddMemcpyNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphMemcpyNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphMemcpyNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphMemcpyNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphMemcpyNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddMemsetNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddMemsetNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphMemsetNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphMemsetNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphMemsetNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphMemsetNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddHostNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddHostNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphHostNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphHostNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphHostNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphHostNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddChildGraphNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddChildGraphNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphChildGraphNodeGetGraph(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphChildGraphNodeGetGraph");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddEmptyNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddEmptyNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddEventRecordNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddEventRecordNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphEventRecordNodeGetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphEventRecordNodeGetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphEventRecordNodeSetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphEventRecordNodeSetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddEventWaitNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddEventWaitNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphEventWaitNodeGetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphEventWaitNodeGetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphEventWaitNodeSetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphEventWaitNodeSetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddExternalSemaphoresSignalNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddExternalSemaphoresSignalNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExternalSemaphoresSignalNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExternalSemaphoresSignalNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExternalSemaphoresSignalNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExternalSemaphoresSignalNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddExternalSemaphoresWaitNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddExternalSemaphoresWaitNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExternalSemaphoresWaitNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExternalSemaphoresWaitNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExternalSemaphoresWaitNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExternalSemaphoresWaitNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddBatchMemOpNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddBatchMemOpNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphBatchMemOpNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphBatchMemOpNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphBatchMemOpNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphBatchMemOpNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecBatchMemOpNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecBatchMemOpNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddMemAllocNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddMemAllocNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphMemAllocNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphMemAllocNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddMemFreeNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddMemFreeNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphMemFreeNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphMemFreeNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDeviceGraphMemTrim(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGraphMemTrim");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDeviceGetGraphMemAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetGraphMemAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDeviceSetGraphMemAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceSetGraphMemAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphClone(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphClone");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphNodeFindInClone(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphNodeFindInClone");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphNodeGetType(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphNodeGetType");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphGetNodes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphGetNodes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphGetRootNodes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphGetRootNodes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphGetEdges(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphGetEdges");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphNodeGetDependencies(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphNodeGetDependencies");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphNodeGetDependentNodes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphNodeGetDependentNodes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddDependencies(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddDependencies");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphRemoveDependencies(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphRemoveDependencies");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphDestroyNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphDestroyNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphInstantiateWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphInstantiateWithFlags");
	auto args = (struct cuGraphInstantiateWithFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuGraphInstantiateWithFlagsResponse), alignof(cuGraphInstantiateWithFlagsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuGraphInstantiateWithFlagsResponse*>(responsePayload);
            response->err = cuGraphInstantiateWithFlags(
				(args->phGraphExec ? &(response->phGraphExec) : NULL),
				args->hGraph,
				args->flags
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuGraphInstantiateWithParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphInstantiateWithParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecGetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecGetFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecKernelNodeSetParams_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecKernelNodeSetParams_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecMemcpyNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecMemcpyNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecMemsetNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecMemsetNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecHostNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecHostNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecChildGraphNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecChildGraphNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecEventRecordNodeSetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecEventRecordNodeSetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecEventWaitNodeSetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecEventWaitNodeSetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecExternalSemaphoresSignalNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecExternalSemaphoresSignalNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecExternalSemaphoresWaitNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecExternalSemaphoresWaitNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphNodeSetEnabled(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphNodeSetEnabled");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphNodeGetEnabled(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphNodeGetEnabled");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphUpload(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphUpload");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecDestroy");
	auto args = (struct cuGraphExecDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuGraphExecDestroy(
				args->hGraphExec
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuGraphDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphDestroy");
	auto args = (struct cuGraphDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(CUresult), alignof(CUresult))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<CUresult*>(responsePayload);
            *response = cuGraphDestroy(
				args->hGraph
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuGraphExecUpdate_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecUpdate_v2");
	auto args = (struct cuGraphExecUpdate_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cuGraphExecUpdate_v2Response), alignof(cuGraphExecUpdate_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cuGraphExecUpdate_v2Response*>(responsePayload);
            response->err = cuGraphExecUpdate_v2(
				args->hGraphExec,
				args->hGraph,
				(args->resultInfo ? &(response->resultInfo) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuGraphKernelNodeCopyAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphKernelNodeCopyAttributes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphKernelNodeGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphKernelNodeGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphKernelNodeSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphKernelNodeSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphDebugDotPrint(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphDebugDotPrint");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuUserObjectCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuUserObjectCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuUserObjectRetain(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuUserObjectRetain");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuUserObjectRelease(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuUserObjectRelease");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphRetainUserObject(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphRetainUserObject");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphReleaseUserObject(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphReleaseUserObject");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphAddNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphAddNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphExecNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphExecNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuOccupancyMaxActiveBlocksPerMultiprocessor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuOccupancyMaxActiveBlocksPerMultiprocessor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuOccupancyMaxPotentialBlockSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuOccupancyMaxPotentialBlockSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuOccupancyMaxPotentialBlockSizeWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuOccupancyMaxPotentialBlockSizeWithFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuOccupancyAvailableDynamicSMemPerBlock(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuOccupancyAvailableDynamicSMemPerBlock");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuOccupancyMaxPotentialClusterSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuOccupancyMaxPotentialClusterSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuOccupancyMaxActiveClusters(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuOccupancyMaxActiveClusters");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetMipmappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetMipmappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetAddress_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetAddress_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetAddress2D_v3(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetAddress2D_v3");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetFormat(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetFormat");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetAddressMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetAddressMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetFilterMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetFilterMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetMipmapFilterMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetMipmapFilterMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetMipmapLevelBias(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetMipmapLevelBias");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetMipmapLevelClamp(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetMipmapLevelClamp");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetMaxAnisotropy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetMaxAnisotropy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetBorderColor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetBorderColor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefSetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefSetFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetAddress_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetAddress_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetMipmappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetMipmappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetAddressMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetAddressMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetFilterMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetFilterMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetFormat(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetFormat");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetMipmapFilterMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetMipmapFilterMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetMipmapLevelBias(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetMipmapLevelBias");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetMipmapLevelClamp(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetMipmapLevelClamp");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetMaxAnisotropy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetMaxAnisotropy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetBorderColor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetBorderColor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefGetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefGetFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexRefDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexRefDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuSurfRefSetArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuSurfRefSetArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuSurfRefGetArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuSurfRefGetArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexObjectCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexObjectCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexObjectDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexObjectDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexObjectGetResourceDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexObjectGetResourceDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexObjectGetTextureDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexObjectGetTextureDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTexObjectGetResourceViewDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTexObjectGetResourceViewDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuSurfObjectCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuSurfObjectCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuSurfObjectDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuSurfObjectDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuSurfObjectGetResourceDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuSurfObjectGetResourceDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTensorMapEncodeTiled(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTensorMapEncodeTiled");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTensorMapEncodeIm2col(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTensorMapEncodeIm2col");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuTensorMapReplaceAddress(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuTensorMapReplaceAddress");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDeviceCanAccessPeer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceCanAccessPeer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCtxEnablePeerAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxEnablePeerAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCtxDisablePeerAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCtxDisablePeerAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuDeviceGetP2PAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuDeviceGetP2PAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphicsUnregisterResource(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphicsUnregisterResource");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphicsSubResourceGetMappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphicsSubResourceGetMappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphicsResourceGetMappedMipmappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphicsResourceGetMappedMipmappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphicsResourceGetMappedPointer_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphicsResourceGetMappedPointer_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphicsResourceSetMapFlags_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphicsResourceSetMapFlags_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphicsMapResources(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphicsMapResources");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGraphicsUnmapResources(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGraphicsUnmapResources");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCoredumpGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCoredumpGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCoredumpGetAttributeGlobal(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCoredumpGetAttributeGlobal");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCoredumpSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCoredumpSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuCoredumpSetAttributeGlobal(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuCoredumpSetAttributeGlobal");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuGetExportTable(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuGetExportTable");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceReset(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceReset");
	auto args = (struct cudaDeviceResetArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaDeviceReset(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceSetLimit(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceSetLimit");
	auto args = (struct cudaDeviceSetLimitArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaDeviceSetLimit(
				args->limit,
				args->value
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceGetLimit(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetLimit");
	auto args = (struct cudaDeviceGetLimitArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDeviceGetLimitResponse), alignof(cudaDeviceGetLimitResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDeviceGetLimitResponse*>(responsePayload);
            response->err = cudaDeviceGetLimit(
				(args->pValue ? &(response->pValue) : NULL),
				args->limit
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceGetTexture1DLinearMaxWidth(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetTexture1DLinearMaxWidth");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceGetCacheConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetCacheConfig");
	auto args = (struct cudaDeviceGetCacheConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDeviceGetCacheConfigResponse), alignof(cudaDeviceGetCacheConfigResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDeviceGetCacheConfigResponse*>(responsePayload);
            response->err = cudaDeviceGetCacheConfig(
				(args->pCacheConfig ? &(response->pCacheConfig) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceGetStreamPriorityRange(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetStreamPriorityRange");
	auto args = (struct cudaDeviceGetStreamPriorityRangeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDeviceGetStreamPriorityRangeResponse), alignof(cudaDeviceGetStreamPriorityRangeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDeviceGetStreamPriorityRangeResponse*>(responsePayload);
            response->err = cudaDeviceGetStreamPriorityRange(
				(args->leastPriority ? &(response->leastPriority) : NULL),
				(args->greatestPriority ? &(response->greatestPriority) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceSetCacheConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceSetCacheConfig");
	auto args = (struct cudaDeviceSetCacheConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaDeviceSetCacheConfig(
				args->cacheConfig
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceGetSharedMemConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetSharedMemConfig");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceSetSharedMemConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceSetSharedMemConfig");
	auto args = (struct cudaDeviceSetSharedMemConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaDeviceSetSharedMemConfig(
				args->config
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceGetByPCIBusId(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetByPCIBusId");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceGetPCIBusId(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetPCIBusId");
	auto args = (struct cudaDeviceGetPCIBusIdArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDeviceGetPCIBusIdResponse), alignof(cudaDeviceGetPCIBusIdResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDeviceGetPCIBusIdResponse*>(responsePayload);
            response->err = cudaDeviceGetPCIBusId(
				(args->pciBusId ? &(response->pciBusId) : NULL),
				args->len,
				args->device
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaIpcGetEventHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaIpcGetEventHandle");
	auto args = (struct cudaIpcGetEventHandleArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaIpcGetEventHandleResponse), alignof(cudaIpcGetEventHandleResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaIpcGetEventHandleResponse*>(responsePayload);
            response->err = cudaIpcGetEventHandle(
				(args->handle ? &(response->handle) : NULL),
				args->event
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaIpcOpenEventHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaIpcOpenEventHandle");
	auto args = (struct cudaIpcOpenEventHandleArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaIpcOpenEventHandleResponse), alignof(cudaIpcOpenEventHandleResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaIpcOpenEventHandleResponse*>(responsePayload);
            response->err = cudaIpcOpenEventHandle(
				(args->event ? &(response->event) : NULL),
				args->handle
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaIpcGetMemHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaIpcGetMemHandle");
	auto args = (struct cudaIpcGetMemHandleArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaIpcGetMemHandleResponse), alignof(cudaIpcGetMemHandleResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaIpcGetMemHandleResponse*>(responsePayload);
            response->err = cudaIpcGetMemHandle(
				(args->handle ? &(response->handle) : NULL),
				args->devPtr
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaIpcOpenMemHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaIpcOpenMemHandle");
	auto args = (struct cudaIpcOpenMemHandleArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaIpcOpenMemHandleResponse), alignof(cudaIpcOpenMemHandleResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaIpcOpenMemHandleResponse*>(responsePayload);
            response->err = cudaIpcOpenMemHandle(
				(args->devPtr ? &(response->devPtr) : NULL),
				args->handle,
				args->flags
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaIpcCloseMemHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaIpcCloseMemHandle");
	auto args = (struct cudaIpcCloseMemHandleArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaIpcCloseMemHandle(
				args->devPtr
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceFlushGPUDirectRDMAWrites(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceFlushGPUDirectRDMAWrites");
	auto args = (struct cudaDeviceFlushGPUDirectRDMAWritesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaDeviceFlushGPUDirectRDMAWrites(
				args->target,
				args->scope
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaThreadExit(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaThreadExit");
	auto args = (struct cudaThreadExitArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaThreadExit(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaThreadSetLimit(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaThreadSetLimit");
	auto args = (struct cudaThreadSetLimitArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaThreadSetLimit(
				args->limit,
				args->value
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaThreadGetLimit(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaThreadGetLimit");
	auto args = (struct cudaThreadGetLimitArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaThreadGetLimitResponse), alignof(cudaThreadGetLimitResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaThreadGetLimitResponse*>(responsePayload);
            response->err = cudaThreadGetLimit(
				(args->pValue ? &(response->pValue) : NULL),
				args->limit
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaThreadGetCacheConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaThreadGetCacheConfig");
	auto args = (struct cudaThreadGetCacheConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaThreadGetCacheConfigResponse), alignof(cudaThreadGetCacheConfigResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaThreadGetCacheConfigResponse*>(responsePayload);
            response->err = cudaThreadGetCacheConfig(
				(args->pCacheConfig ? &(response->pCacheConfig) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaThreadSetCacheConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaThreadSetCacheConfig");
	auto args = (struct cudaThreadSetCacheConfigArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaThreadSetCacheConfig(
				args->cacheConfig
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGetLastError(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetLastError");
	auto args = (struct cudaGetLastErrorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaGetLastError(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaPeekAtLastError(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaPeekAtLastError");
	auto args = (struct cudaPeekAtLastErrorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaPeekAtLastError(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGetErrorName(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetErrorName");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetErrorString(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetErrorString");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetDeviceCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetDeviceCount");
	auto args = (struct cudaGetDeviceCountArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaGetDeviceCountResponse), alignof(cudaGetDeviceCountResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaGetDeviceCountResponse*>(responsePayload);
            response->err = cudaGetDeviceCount(
				(args->count ? &(response->count) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGetDeviceProperties_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetDeviceProperties_v2");
	auto args = (struct cudaGetDeviceProperties_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaGetDeviceProperties_v2Response), alignof(cudaGetDeviceProperties_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaGetDeviceProperties_v2Response*>(responsePayload);
            response->err = cudaGetDeviceProperties_v2(
				(args->prop ? &(response->prop) : NULL),
				args->device
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetAttribute");
	auto args = (struct cudaDeviceGetAttributeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDeviceGetAttributeResponse), alignof(cudaDeviceGetAttributeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDeviceGetAttributeResponse*>(responsePayload);
            response->err = cudaDeviceGetAttribute(
				(args->value ? &(response->value) : NULL),
				args->attr,
				args->device
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceGetDefaultMemPool(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetDefaultMemPool");
	auto args = (struct cudaDeviceGetDefaultMemPoolArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDeviceGetDefaultMemPoolResponse), alignof(cudaDeviceGetDefaultMemPoolResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDeviceGetDefaultMemPoolResponse*>(responsePayload);
            response->err = cudaDeviceGetDefaultMemPool(
				(args->memPool ? &(response->memPool) : NULL),
				args->device
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceSetMemPool(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceSetMemPool");
	auto args = (struct cudaDeviceSetMemPoolArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaDeviceSetMemPool(
				args->device,
				args->memPool
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceGetMemPool(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetMemPool");
	auto args = (struct cudaDeviceGetMemPoolArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDeviceGetMemPoolResponse), alignof(cudaDeviceGetMemPoolResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDeviceGetMemPoolResponse*>(responsePayload);
            response->err = cudaDeviceGetMemPool(
				(args->memPool ? &(response->memPool) : NULL),
				args->device
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaDeviceGetNvSciSyncAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetNvSciSyncAttributes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceGetP2PAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetP2PAttribute");
	auto args = (struct cudaDeviceGetP2PAttributeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDeviceGetP2PAttributeResponse), alignof(cudaDeviceGetP2PAttributeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDeviceGetP2PAttributeResponse*>(responsePayload);
            response->err = cudaDeviceGetP2PAttribute(
				(args->value ? &(response->value) : NULL),
				args->attr,
				args->srcDevice,
				args->dstDevice
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaInitDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaInitDevice");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetDevice");
	auto args = (struct cudaGetDeviceArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaGetDeviceResponse), alignof(cudaGetDeviceResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaGetDeviceResponse*>(responsePayload);
            response->err = cudaGetDevice(
				(args->device ? &(response->device) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaSetValidDevices(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaSetValidDevices");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaSetDeviceFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaSetDeviceFlags");
	auto args = (struct cudaSetDeviceFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaSetDeviceFlags(
				args->flags
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGetDeviceFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetDeviceFlags");
	auto args = (struct cudaGetDeviceFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaGetDeviceFlagsResponse), alignof(cudaGetDeviceFlagsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaGetDeviceFlagsResponse*>(responsePayload);
            response->err = cudaGetDeviceFlags(
				(args->flags ? &(response->flags) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamGetPriority(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamGetPriority");
	auto args = (struct cudaStreamGetPriorityArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaStreamGetPriorityResponse), alignof(cudaStreamGetPriorityResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaStreamGetPriorityResponse*>(responsePayload);
            response->err = cudaStreamGetPriority(
				__stream,
				(args->priority ? &(response->priority) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamGetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamGetFlags");
	auto args = (struct cudaStreamGetFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->hStream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaStreamGetFlagsResponse), alignof(cudaStreamGetFlagsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaStreamGetFlagsResponse*>(responsePayload);
            response->err = cudaStreamGetFlags(
				__stream,
				(args->flags ? &(response->flags) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamGetId(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamGetId");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaCtxResetPersistingL2Cache(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaCtxResetPersistingL2Cache");
	auto args = (struct cudaCtxResetPersistingL2CacheArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaCtxResetPersistingL2Cache(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamCopyAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamCopyAttributes");
	auto args = (struct cudaStreamCopyAttributesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->dst;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaStreamCopyAttributes(
				__stream,
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaStreamSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaStreamDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamDestroy");
	auto args = (struct cudaStreamDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaStreamDestroy(
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamWaitEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamWaitEvent");
	auto args = (struct cudaStreamWaitEventArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaStreamWaitEvent(
				__stream,
				args->event,
				args->flags
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamAddCallback(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamAddCallback");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaStreamQuery(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamQuery");
	auto args = (struct cudaStreamQueryArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaStreamQuery(
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamAttachMemAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamAttachMemAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaThreadExchangeStreamCaptureMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaThreadExchangeStreamCaptureMode");
	auto args = (struct cudaThreadExchangeStreamCaptureModeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaThreadExchangeStreamCaptureModeResponse), alignof(cudaThreadExchangeStreamCaptureModeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaThreadExchangeStreamCaptureModeResponse*>(responsePayload);
            response->err = cudaThreadExchangeStreamCaptureMode(
				(args->mode ? &(response->mode) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamIsCapturing(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamIsCapturing");
	auto args = (struct cudaStreamIsCapturingArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaStreamIsCapturingResponse), alignof(cudaStreamIsCapturingResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaStreamIsCapturingResponse*>(responsePayload);
            response->err = cudaStreamIsCapturing(
				__stream,
				(args->pCaptureStatus ? &(response->pCaptureStatus) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaStreamUpdateCaptureDependencies(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaStreamUpdateCaptureDependencies");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaEventCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaEventCreate");
	auto args = (struct cudaEventCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaEventCreateResponse), alignof(cudaEventCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaEventCreateResponse*>(responsePayload);
            response->err = cudaEventCreate(
				(args->event ? &(response->event) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaEventCreateWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaEventCreateWithFlags");
	auto args = (struct cudaEventCreateWithFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaEventCreateWithFlagsResponse), alignof(cudaEventCreateWithFlagsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaEventCreateWithFlagsResponse*>(responsePayload);
            response->err = cudaEventCreateWithFlags(
				(args->event ? &(response->event) : NULL),
				args->flags
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaEventRecordWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaEventRecordWithFlags");
	auto args = (struct cudaEventRecordWithFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaEventRecordWithFlags(
				args->event,
				__stream,
				args->flags
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaEventQuery(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaEventQuery");
	auto args = (struct cudaEventQueryArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaEventQuery(
				args->event
            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaEventSynchronize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaEventSynchronize");
	auto args = (struct cudaEventSynchronizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaEventSynchronize(
				args->event
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaEventDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaEventDestroy");
	auto args = (struct cudaEventDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaEventDestroy(
				args->event
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaEventElapsedTime(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaEventElapsedTime");
	auto args = (struct cudaEventElapsedTimeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaEventElapsedTimeResponse), alignof(cudaEventElapsedTimeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaEventElapsedTimeResponse*>(responsePayload);
            response->err = cudaEventElapsedTime(
				(args->ms ? &(response->ms) : NULL),
				args->start,
				args->end
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaImportExternalMemory(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaImportExternalMemory");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaExternalMemoryGetMappedBuffer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaExternalMemoryGetMappedBuffer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaExternalMemoryGetMappedMipmappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaExternalMemoryGetMappedMipmappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDestroyExternalMemory(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDestroyExternalMemory");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaImportExternalSemaphore(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaImportExternalSemaphore");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaSignalExternalSemaphoresAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaSignalExternalSemaphoresAsync_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaWaitExternalSemaphoresAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaWaitExternalSemaphoresAsync_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDestroyExternalSemaphore(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDestroyExternalSemaphore");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaLaunchKernelExC(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaLaunchKernelExC");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaLaunchCooperativeKernel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaLaunchCooperativeKernel");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaLaunchCooperativeKernelMultiDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaLaunchCooperativeKernelMultiDevice");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaFuncSetCacheConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaFuncSetCacheConfig");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaFuncSetSharedMemConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaFuncSetSharedMemConfig");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaSetDoubleForDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaSetDoubleForDevice");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaSetDoubleForHost(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaSetDoubleForHost");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaLaunchHostFunc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaLaunchHostFunc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaOccupancyMaxActiveBlocksPerMultiprocessor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaOccupancyMaxActiveBlocksPerMultiprocessor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaOccupancyAvailableDynamicSMemPerBlock(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaOccupancyAvailableDynamicSMemPerBlock");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaOccupancyMaxPotentialClusterSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaOccupancyMaxPotentialClusterSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaOccupancyMaxActiveClusters(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaOccupancyMaxActiveClusters");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMallocManaged(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMallocManaged");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMallocPitch(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMallocPitch");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMallocArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMallocArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaFreeArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaFreeArray");
	auto args = (struct cudaFreeArrayArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaFreeArray(
				args->array
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaFreeMipmappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaFreeMipmappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaHostRegister(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaHostRegister");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaHostUnregister(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaHostUnregister");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaHostGetDevicePointer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaHostGetDevicePointer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaHostGetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaHostGetFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMalloc3D(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMalloc3D");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMalloc3DArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMalloc3DArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMallocMipmappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMallocMipmappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetMipmappedArrayLevel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetMipmappedArrayLevel");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy3D(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy3D");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy3DPeer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy3DPeer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy3DAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy3DAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy3DPeerAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy3DPeerAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemGetInfo(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemGetInfo");
	auto args = (struct cudaMemGetInfoArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaMemGetInfoResponse), alignof(cudaMemGetInfoResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaMemGetInfoResponse*>(responsePayload);
            response->err = cudaMemGetInfo(
				(args->free ? &(response->free) : NULL),
				(args->total ? &(response->total) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaArrayGetInfo(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaArrayGetInfo");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaArrayGetPlane(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaArrayGetPlane");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaArrayGetMemoryRequirements(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaArrayGetMemoryRequirements");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMipmappedArrayGetMemoryRequirements(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMipmappedArrayGetMemoryRequirements");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaArrayGetSparseProperties(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaArrayGetSparseProperties");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMipmappedArrayGetSparseProperties(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMipmappedArrayGetSparseProperties");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyPeer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyPeer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy2D(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy2D");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy2DToArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy2DToArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy2DFromArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy2DFromArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy2DArrayToArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy2DArrayToArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyToSymbol(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyToSymbol");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyFromSymbol(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyFromSymbol");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyPeerAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyPeerAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy2DAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy2DAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy2DToArrayAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy2DToArrayAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpy2DFromArrayAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpy2DFromArrayAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyToSymbolAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyToSymbolAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyFromSymbolAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyFromSymbolAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemset2D(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemset2D");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemset3D(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemset3D");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemsetAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemsetAsync");
	auto args = (struct cudaMemsetAsyncArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaMemsetAsync(
				args->devPtr,
				args->value,
				args->count,
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaMemset2DAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemset2DAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemset3DAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemset3DAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetSymbolAddress(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetSymbolAddress");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetSymbolSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetSymbolSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPrefetchAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPrefetchAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPrefetchAsync_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPrefetchAsync_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemAdvise(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemAdvise");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemAdvise_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemAdvise_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemRangeGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemRangeGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemRangeGetAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemRangeGetAttributes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyToArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyToArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyFromArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyFromArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyArrayToArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyArrayToArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyToArrayAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyToArrayAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemcpyFromArrayAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemcpyFromArrayAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMallocAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMallocAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaFreeAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaFreeAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolTrimTo(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolTrimTo");
	auto args = (struct cudaMemPoolTrimToArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaMemPoolTrimTo(
				args->memPool,
				args->minBytesToKeep
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaMemPoolSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolSetAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolSetAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolGetAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolGetAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMallocFromPoolAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMallocFromPoolAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolExportToShareableHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolExportToShareableHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolImportFromShareableHandle(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolImportFromShareableHandle");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolExportPointer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolExportPointer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaMemPoolImportPointer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaMemPoolImportPointer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceCanAccessPeer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceCanAccessPeer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceEnablePeerAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceEnablePeerAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceDisablePeerAccess(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceDisablePeerAccess");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphicsUnregisterResource(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphicsUnregisterResource");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphicsResourceSetMapFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphicsResourceSetMapFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphicsMapResources(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphicsMapResources");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphicsUnmapResources(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphicsUnmapResources");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphicsResourceGetMappedPointer(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphicsResourceGetMappedPointer");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphicsSubResourceGetMappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphicsSubResourceGetMappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphicsResourceGetMappedMipmappedArray(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphicsResourceGetMappedMipmappedArray");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetChannelDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetChannelDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaCreateChannelDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaCreateChannelDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaCreateTextureObject(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaCreateTextureObject");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDestroyTextureObject(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDestroyTextureObject");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetTextureObjectResourceDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetTextureObjectResourceDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetTextureObjectTextureDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetTextureObjectTextureDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetTextureObjectResourceViewDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetTextureObjectResourceViewDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaCreateSurfaceObject(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaCreateSurfaceObject");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDestroySurfaceObject(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDestroySurfaceObject");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetSurfaceObjectResourceDesc(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetSurfaceObjectResourceDesc");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDriverGetVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDriverGetVersion");
	auto args = (struct cudaDriverGetVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaDriverGetVersionResponse), alignof(cudaDriverGetVersionResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaDriverGetVersionResponse*>(responsePayload);
            response->err = cudaDriverGetVersion(
				(args->driverVersion ? &(response->driverVersion) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaRuntimeGetVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaRuntimeGetVersion");
	auto args = (struct cudaRuntimeGetVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaRuntimeGetVersionResponse), alignof(cudaRuntimeGetVersionResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaRuntimeGetVersionResponse*>(responsePayload);
            response->err = cudaRuntimeGetVersion(
				(args->runtimeVersion ? &(response->runtimeVersion) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGraphCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphCreate");
	auto args = (struct cudaGraphCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaGraphCreateResponse), alignof(cudaGraphCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaGraphCreateResponse*>(responsePayload);
            response->err = cudaGraphCreate(
				(args->pGraph ? &(response->pGraph) : NULL),
				args->flags
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGraphAddKernelNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddKernelNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphKernelNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphKernelNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphKernelNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphKernelNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphKernelNodeCopyAttributes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphKernelNodeCopyAttributes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphKernelNodeGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphKernelNodeGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphKernelNodeSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphKernelNodeSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddMemcpyNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddMemcpyNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddMemcpyNodeToSymbol(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddMemcpyNodeToSymbol");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddMemcpyNodeFromSymbol(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddMemcpyNodeFromSymbol");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddMemcpyNode1D(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddMemcpyNode1D");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphMemcpyNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphMemcpyNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphMemcpyNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphMemcpyNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphMemcpyNodeSetParamsToSymbol(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphMemcpyNodeSetParamsToSymbol");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphMemcpyNodeSetParamsFromSymbol(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphMemcpyNodeSetParamsFromSymbol");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphMemcpyNodeSetParams1D(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphMemcpyNodeSetParams1D");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddMemsetNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddMemsetNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphMemsetNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphMemsetNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphMemsetNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphMemsetNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddHostNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddHostNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphHostNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphHostNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphHostNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphHostNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddChildGraphNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddChildGraphNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphChildGraphNodeGetGraph(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphChildGraphNodeGetGraph");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddEmptyNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddEmptyNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddEventRecordNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddEventRecordNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphEventRecordNodeGetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphEventRecordNodeGetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphEventRecordNodeSetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphEventRecordNodeSetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddEventWaitNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddEventWaitNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphEventWaitNodeGetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphEventWaitNodeGetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphEventWaitNodeSetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphEventWaitNodeSetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddExternalSemaphoresSignalNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddExternalSemaphoresSignalNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExternalSemaphoresSignalNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExternalSemaphoresSignalNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExternalSemaphoresSignalNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExternalSemaphoresSignalNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddExternalSemaphoresWaitNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddExternalSemaphoresWaitNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExternalSemaphoresWaitNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExternalSemaphoresWaitNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExternalSemaphoresWaitNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExternalSemaphoresWaitNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddMemAllocNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddMemAllocNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphMemAllocNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphMemAllocNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddMemFreeNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddMemFreeNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphMemFreeNodeGetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphMemFreeNodeGetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceGraphMemTrim(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGraphMemTrim");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceGetGraphMemAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceGetGraphMemAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaDeviceSetGraphMemAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaDeviceSetGraphMemAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphClone(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphClone");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphNodeFindInClone(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphNodeFindInClone");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphNodeGetType(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphNodeGetType");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphGetRootNodes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphGetRootNodes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphGetEdges(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphGetEdges");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphNodeGetDependencies(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphNodeGetDependencies");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphNodeGetDependentNodes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphNodeGetDependentNodes");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddDependencies(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddDependencies");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphRemoveDependencies(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphRemoveDependencies");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphDestroyNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphDestroyNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphInstantiate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphInstantiate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphInstantiateWithFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphInstantiateWithFlags");
	auto args = (struct cudaGraphInstantiateWithFlagsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaGraphInstantiateWithFlagsResponse), alignof(cudaGraphInstantiateWithFlagsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaGraphInstantiateWithFlagsResponse*>(responsePayload);
            response->err = cudaGraphInstantiateWithFlags(
				(args->pGraphExec ? &(response->pGraphExec) : NULL),
				args->graph,
				args->flags
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGraphInstantiateWithParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphInstantiateWithParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecGetFlags(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecGetFlags");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecKernelNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecKernelNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecMemcpyNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecMemcpyNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecMemcpyNodeSetParamsToSymbol(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecMemcpyNodeSetParamsToSymbol");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecMemcpyNodeSetParamsFromSymbol(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecMemcpyNodeSetParamsFromSymbol");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecMemcpyNodeSetParams1D(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecMemcpyNodeSetParams1D");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecMemsetNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecMemsetNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecHostNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecHostNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecChildGraphNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecChildGraphNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecEventRecordNodeSetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecEventRecordNodeSetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecEventWaitNodeSetEvent(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecEventWaitNodeSetEvent");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecExternalSemaphoresSignalNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecExternalSemaphoresSignalNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecExternalSemaphoresWaitNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecExternalSemaphoresWaitNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphNodeSetEnabled(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphNodeSetEnabled");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphNodeGetEnabled(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphNodeGetEnabled");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecUpdate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecUpdate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphUpload(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphUpload");
	auto args = (struct cudaGraphUploadArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaGraphUpload(
				args->graphExec,
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGraphLaunch(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphLaunch");
	auto args = (struct cudaGraphLaunchArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaGraphLaunch(
				args->graphExec,
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGraphExecDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecDestroy");
	auto args = (struct cudaGraphExecDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaGraphExecDestroy(
				args->graphExec
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGraphDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphDestroy");
	auto args = (struct cudaGraphDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaGraphDestroy(
				args->graph
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaGraphDebugDotPrint(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphDebugDotPrint");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaUserObjectCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaUserObjectCreate");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaUserObjectRetain(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaUserObjectRetain");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaUserObjectRelease(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaUserObjectRelease");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphRetainUserObject(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphRetainUserObject");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphReleaseUserObject(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphReleaseUserObject");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphAddNode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphAddNode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGraphExecNodeSetParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGraphExecNodeSetParams");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetDriverEntryPoint(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetDriverEntryPoint");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetExportTable(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetExportTable");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetFuncBySymbol(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetFuncBySymbol");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaGetKernel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaGetKernel");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetVersion");
	auto args = (struct cudnnGetVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(size_t), alignof(size_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<size_t*>(responsePayload);
            *response = cudnnGetVersion(

            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetMaxDeviceVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetMaxDeviceVersion");
	auto args = (struct cudnnGetMaxDeviceVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(size_t), alignof(size_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<size_t*>(responsePayload);
            *response = cudnnGetMaxDeviceVersion(

            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetCudartVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetCudartVersion");
	auto args = (struct cudnnGetCudartVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(size_t), alignof(size_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<size_t*>(responsePayload);
            *response = cudnnGetCudartVersion(

            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnQueryRuntimeError(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnQueryRuntimeError");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetProperty(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetProperty");
	auto args = (struct cudnnGetPropertyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetPropertyResponse), alignof(cudnnGetPropertyResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetPropertyResponse*>(responsePayload);
            response->err = cudnnGetProperty(
				args->type,
				(args->value ? &(response->value) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroy");
	auto args = (struct cudnnDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroy(
				args->handle
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetStream(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetStream");
	auto args = (struct cudnnSetStreamArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->streamId;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetStream(
				args->handle,
				__stream
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetStream(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetStream");
	auto args = (struct cudnnGetStreamArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetStreamResponse), alignof(cudnnGetStreamResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetStreamResponse*>(responsePayload);
            response->err = cudnnGetStream(
				args->handle,
				(args->streamId ? &(response->streamId) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnCreateTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateTensorDescriptor");
	auto args = (struct cudnnCreateTensorDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateTensorDescriptorResponse), alignof(cudnnCreateTensorDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateTensorDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateTensorDescriptor(
				(args->tensorDesc ? &(response->tensorDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetTensor4dDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetTensor4dDescriptor");
	auto args = (struct cudnnSetTensor4dDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetTensor4dDescriptor(
				args->tensorDesc,
				args->format,
				args->dataType,
				args->n,
				args->c,
				args->h,
				args->w
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetTensor4dDescriptorEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetTensor4dDescriptorEx");
	auto args = (struct cudnnSetTensor4dDescriptorExArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetTensor4dDescriptorEx(
				args->tensorDesc,
				args->dataType,
				args->n,
				args->c,
				args->h,
				args->w,
				args->nStride,
				args->cStride,
				args->hStride,
				args->wStride
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetTensor4dDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetTensor4dDescriptor");
	auto args = (struct cudnnGetTensor4dDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetTensor4dDescriptorResponse), alignof(cudnnGetTensor4dDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetTensor4dDescriptorResponse*>(responsePayload);
            response->err = cudnnGetTensor4dDescriptor(
				args->tensorDesc,
				(args->dataType ? &(response->dataType) : NULL),
				(args->n ? &(response->n) : NULL),
				(args->c ? &(response->c) : NULL),
				(args->h ? &(response->h) : NULL),
				(args->w ? &(response->w) : NULL),
				(args->nStride ? &(response->nStride) : NULL),
				(args->cStride ? &(response->cStride) : NULL),
				(args->hStride ? &(response->hStride) : NULL),
				(args->wStride ? &(response->wStride) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetTensorNdDescriptorEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetTensorNdDescriptorEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetTensorSizeInBytes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetTensorSizeInBytes");
	auto args = (struct cudnnGetTensorSizeInBytesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetTensorSizeInBytesResponse), alignof(cudnnGetTensorSizeInBytesResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetTensorSizeInBytesResponse*>(responsePayload);
            response->err = cudnnGetTensorSizeInBytes(
				args->tensorDesc,
				(args->size ? &(response->size) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDestroyTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyTensorDescriptor");
	auto args = (struct cudnnDestroyTensorDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyTensorDescriptor(
				args->tensorDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnInitTransformDest(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnInitTransformDest");
	auto args = (struct cudnnInitTransformDestArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnInitTransformDestResponse), alignof(cudnnInitTransformDestResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnInitTransformDestResponse*>(responsePayload);
            response->err = cudnnInitTransformDest(
				args->transformDesc,
				args->srcDesc,
				args->destDesc,
				(args->destSizeInBytes ? &(response->destSizeInBytes) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnCreateTensorTransformDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateTensorTransformDescriptor");
	auto args = (struct cudnnCreateTensorTransformDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateTensorTransformDescriptorResponse), alignof(cudnnCreateTensorTransformDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateTensorTransformDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateTensorTransformDescriptor(
				(args->transformDesc ? &(response->transformDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetTensorTransformDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetTensorTransformDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetTensorTransformDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetTensorTransformDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyTensorTransformDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyTensorTransformDescriptor");
	auto args = (struct cudnnDestroyTensorTransformDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyTensorTransformDescriptor(
				args->transformDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnTransformTensorEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnTransformTensorEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateOpTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateOpTensorDescriptor");
	auto args = (struct cudnnCreateOpTensorDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateOpTensorDescriptorResponse), alignof(cudnnCreateOpTensorDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateOpTensorDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateOpTensorDescriptor(
				(args->opTensorDesc ? &(response->opTensorDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetOpTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetOpTensorDescriptor");
	auto args = (struct cudnnSetOpTensorDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetOpTensorDescriptor(
				args->opTensorDesc,
				args->opTensorOp,
				args->opTensorCompType,
				args->opTensorNanOpt
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetOpTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetOpTensorDescriptor");
	auto args = (struct cudnnGetOpTensorDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetOpTensorDescriptorResponse), alignof(cudnnGetOpTensorDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetOpTensorDescriptorResponse*>(responsePayload);
            response->err = cudnnGetOpTensorDescriptor(
				args->opTensorDesc,
				(args->opTensorOp ? &(response->opTensorOp) : NULL),
				(args->opTensorCompType ? &(response->opTensorCompType) : NULL),
				(args->opTensorNanOpt ? &(response->opTensorNanOpt) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDestroyOpTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyOpTensorDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnOpTensor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnOpTensor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateReduceTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateReduceTensorDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetReduceTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetReduceTensorDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetReduceTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetReduceTensorDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyReduceTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyReduceTensorDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetReductionIndicesSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetReductionIndicesSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetReductionWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetReductionWorkspaceSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnReduceTensor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnReduceTensor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetTensor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetTensor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnScaleTensor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnScaleTensor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateFilterDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateFilterDescriptor");
	auto args = (struct cudnnCreateFilterDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateFilterDescriptorResponse), alignof(cudnnCreateFilterDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateFilterDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateFilterDescriptor(
				(args->filterDesc ? &(response->filterDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetFilter4dDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetFilter4dDescriptor");
	auto args = (struct cudnnSetFilter4dDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetFilter4dDescriptor(
				args->filterDesc,
				args->dataType,
				args->format,
				args->k,
				args->c,
				args->h,
				args->w
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetFilter4dDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetFilter4dDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetFilterSizeInBytes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetFilterSizeInBytes");
	auto args = (struct cudnnGetFilterSizeInBytesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetFilterSizeInBytesResponse), alignof(cudnnGetFilterSizeInBytesResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetFilterSizeInBytesResponse*>(responsePayload);
            response->err = cudnnGetFilterSizeInBytes(
				args->filterDesc,
				(args->size ? &(response->size) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnTransformFilter(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnTransformFilter");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyFilterDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyFilterDescriptor");
	auto args = (struct cudnnDestroyFilterDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyFilterDescriptor(
				args->filterDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnCreatePoolingDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreatePoolingDescriptor");
	auto args = (struct cudnnCreatePoolingDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreatePoolingDescriptorResponse), alignof(cudnnCreatePoolingDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreatePoolingDescriptorResponse*>(responsePayload);
            response->err = cudnnCreatePoolingDescriptor(
				(args->poolingDesc ? &(response->poolingDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetPooling2dDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetPooling2dDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetPooling2dDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetPooling2dDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetPooling2dForwardOutputDim(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetPooling2dForwardOutputDim");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyPoolingDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyPoolingDescriptor");
	auto args = (struct cudnnDestroyPoolingDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyPoolingDescriptor(
				args->poolingDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnCreateActivationDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateActivationDescriptor");
	auto args = (struct cudnnCreateActivationDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateActivationDescriptorResponse), alignof(cudnnCreateActivationDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateActivationDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateActivationDescriptor(
				(args->activationDesc ? &(response->activationDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetActivationDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetActivationDescriptor");
	auto args = (struct cudnnSetActivationDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetActivationDescriptor(
				args->activationDesc,
				args->mode,
				args->reluNanOpt,
				args->coef
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetActivationDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetActivationDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetActivationDescriptorSwishBeta(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetActivationDescriptorSwishBeta");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetActivationDescriptorSwishBeta(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetActivationDescriptorSwishBeta");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyActivationDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyActivationDescriptor");
	auto args = (struct cudnnDestroyActivationDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyActivationDescriptor(
				args->activationDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnCreateLRNDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateLRNDescriptor");
	auto args = (struct cudnnCreateLRNDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateLRNDescriptorResponse), alignof(cudnnCreateLRNDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateLRNDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateLRNDescriptor(
				(args->normDesc ? &(response->normDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetLRNDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetLRNDescriptor");
	auto args = (struct cudnnSetLRNDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetLRNDescriptor(
				args->normDesc,
				args->lrnN,
				args->lrnAlpha,
				args->lrnBeta,
				args->lrnK
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetLRNDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetLRNDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyLRNDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyLRNDescriptor");
	auto args = (struct cudnnDestroyLRNDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyLRNDescriptor(
				args->lrnDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDivisiveNormalizationForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDivisiveNormalizationForward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDeriveBNTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDeriveBNTensorDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnBatchNormalizationForwardInference(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBatchNormalizationForwardInference");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDeriveNormTensorDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDeriveNormTensorDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnNormalizationForwardInference(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnNormalizationForwardInference");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateSpatialTransformerDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateSpatialTransformerDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetSpatialTransformerNdDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetSpatialTransformerNdDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroySpatialTransformerDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroySpatialTransformerDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSpatialTfGridGeneratorForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSpatialTfGridGeneratorForward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSpatialTfSamplerForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSpatialTfSamplerForward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateDropoutDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateDropoutDescriptor");
	auto args = (struct cudnnCreateDropoutDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateDropoutDescriptorResponse), alignof(cudnnCreateDropoutDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateDropoutDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateDropoutDescriptor(
				(args->dropoutDesc ? &(response->dropoutDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDestroyDropoutDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyDropoutDescriptor");
	auto args = (struct cudnnDestroyDropoutDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyDropoutDescriptor(
				args->dropoutDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDropoutGetStatesSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDropoutGetStatesSize");
	auto args = (struct cudnnDropoutGetStatesSizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnDropoutGetStatesSizeResponse), alignof(cudnnDropoutGetStatesSizeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnDropoutGetStatesSizeResponse*>(responsePayload);
            response->err = cudnnDropoutGetStatesSize(
				args->handle,
				(args->sizeInBytes ? &(response->sizeInBytes) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDropoutGetReserveSpaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDropoutGetReserveSpaceSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetDropoutDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetDropoutDescriptor");
	auto args = (struct cudnnSetDropoutDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetDropoutDescriptor(
				args->dropoutDesc,
				args->handle,
				args->dropout,
				args->states,
				args->stateSizeInBytes,
				args->seed
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnRestoreDropoutDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRestoreDropoutDescriptor");
	auto args = (struct cudnnRestoreDropoutDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnRestoreDropoutDescriptor(
				args->dropoutDesc,
				args->handle,
				args->dropout,
				args->states,
				args->stateSizeInBytes,
				args->seed
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetDropoutDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetDropoutDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDropoutForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDropoutForward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateAlgorithmDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateAlgorithmDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetAlgorithmDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetAlgorithmDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetAlgorithmDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetAlgorithmDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCopyAlgorithmDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCopyAlgorithmDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyAlgorithmDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyAlgorithmDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateAlgorithmPerformance(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateAlgorithmPerformance");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetAlgorithmPerformance(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetAlgorithmPerformance");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetAlgorithmPerformance(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetAlgorithmPerformance");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyAlgorithmPerformance(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyAlgorithmPerformance");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetAlgorithmSpaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetAlgorithmSpaceSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSaveAlgorithm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSaveAlgorithm");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnRestoreAlgorithm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRestoreAlgorithm");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetCallback(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetCallback");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetCallback(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetCallback");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnOpsInferVersionCheck(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnOpsInferVersionCheck");
	auto args = (struct cudnnOpsInferVersionCheckArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnOpsInferVersionCheck(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSoftmaxBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSoftmaxBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnPoolingBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnPoolingBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnActivationBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnActivationBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnLRNCrossChannelBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnLRNCrossChannelBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDivisiveNormalizationBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDivisiveNormalizationBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
	auto args = (struct cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeResponse), alignof(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeResponse*>(responsePayload);
            response->err = cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
				args->handle,
				args->mode,
				args->bnOps,
				args->xDesc,
				args->zDesc,
				args->yDesc,
				args->bnScaleBiasMeanVarDesc,
				args->activationDesc,
				(args->sizeInBytes ? &(response->sizeInBytes) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetBatchNormalizationBackwardExWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetBatchNormalizationBackwardExWorkspaceSize");
	auto args = (struct cudnnGetBatchNormalizationBackwardExWorkspaceSizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetBatchNormalizationBackwardExWorkspaceSizeResponse), alignof(cudnnGetBatchNormalizationBackwardExWorkspaceSizeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetBatchNormalizationBackwardExWorkspaceSizeResponse*>(responsePayload);
            response->err = cudnnGetBatchNormalizationBackwardExWorkspaceSize(
				args->handle,
				args->mode,
				args->bnOps,
				args->xDesc,
				args->yDesc,
				args->dyDesc,
				args->dzDesc,
				args->dxDesc,
				args->dBnScaleBiasDesc,
				args->activationDesc,
				(args->sizeInBytes ? &(response->sizeInBytes) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetBatchNormalizationTrainingExReserveSpaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
	auto args = (struct cudnnGetBatchNormalizationTrainingExReserveSpaceSizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetBatchNormalizationTrainingExReserveSpaceSizeResponse), alignof(cudnnGetBatchNormalizationTrainingExReserveSpaceSizeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetBatchNormalizationTrainingExReserveSpaceSizeResponse*>(responsePayload);
            response->err = cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
				args->handle,
				args->mode,
				args->bnOps,
				args->activationDesc,
				args->xDesc,
				(args->sizeInBytes ? &(response->sizeInBytes) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnBatchNormalizationForwardTraining(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBatchNormalizationForwardTraining");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnBatchNormalizationBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBatchNormalizationBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetNormalizationForwardTrainingWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetNormalizationForwardTrainingWorkspaceSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetNormalizationBackwardWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetNormalizationBackwardWorkspaceSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetNormalizationTrainingReserveSpaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetNormalizationTrainingReserveSpaceSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnNormalizationForwardTraining(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnNormalizationForwardTraining");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnNormalizationBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnNormalizationBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSpatialTfGridGeneratorBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSpatialTfGridGeneratorBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSpatialTfSamplerBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSpatialTfSamplerBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDropoutBackward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDropoutBackward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnOpsTrainVersionCheck(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnOpsTrainVersionCheck");
	auto args = (struct cudnnOpsTrainVersionCheckArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnOpsTrainVersionCheck(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnCreateRNNDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateRNNDescriptor");
	auto args = (struct cudnnCreateRNNDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateRNNDescriptorResponse), alignof(cudnnCreateRNNDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateRNNDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateRNNDescriptor(
				(args->rnnDesc ? &(response->rnnDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDestroyRNNDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyRNNDescriptor");
	auto args = (struct cudnnDestroyRNNDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyRNNDescriptor(
				args->rnnDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetRNNDescriptor_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetRNNDescriptor_v8");
	auto args = (struct cudnnSetRNNDescriptor_v8Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetRNNDescriptor_v8(
				args->rnnDesc,
				args->algo,
				args->cellMode,
				args->biasMode,
				args->dirMode,
				args->inputMode,
				args->dataType,
				args->mathPrec,
				args->mathType,
				args->inputSize,
				args->hiddenSize,
				args->projSize,
				args->numLayers,
				args->dropoutDesc,
				args->auxFlags
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNDescriptor_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNDescriptor_v8");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetRNNDescriptor_v6(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetRNNDescriptor_v6");
	auto args = (struct cudnnSetRNNDescriptor_v6Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetRNNDescriptor_v6(
				args->handle,
				args->rnnDesc,
				args->hiddenSize,
				args->numLayers,
				args->dropoutDesc,
				args->inputMode,
				args->direction,
				args->cellMode,
				args->algo,
				args->mathPrec
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNDescriptor_v6(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNDescriptor_v6");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetRNNMatrixMathType(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetRNNMatrixMathType");
	auto args = (struct cudnnSetRNNMatrixMathTypeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetRNNMatrixMathType(
				args->rnnDesc,
				args->mType
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNMatrixMathType(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNMatrixMathType");
	auto args = (struct cudnnGetRNNMatrixMathTypeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetRNNMatrixMathTypeResponse), alignof(cudnnGetRNNMatrixMathTypeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetRNNMatrixMathTypeResponse*>(responsePayload);
            response->err = cudnnGetRNNMatrixMathType(
				args->rnnDesc,
				(args->mType ? &(response->mType) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetRNNBiasMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetRNNBiasMode");
	auto args = (struct cudnnSetRNNBiasModeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetRNNBiasMode(
				args->rnnDesc,
				args->biasMode
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNBiasMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNBiasMode");
	auto args = (struct cudnnGetRNNBiasModeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetRNNBiasModeResponse), alignof(cudnnGetRNNBiasModeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetRNNBiasModeResponse*>(responsePayload);
            response->err = cudnnGetRNNBiasMode(
				args->rnnDesc,
				(args->biasMode ? &(response->biasMode) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnRNNSetClip_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNSetClip_v8");
	auto args = (struct cudnnRNNSetClip_v8Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnRNNSetClip_v8(
				args->rnnDesc,
				args->clipMode,
				args->clipNanOpt,
				args->lclip,
				args->rclip
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnRNNGetClip_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNGetClip_v8");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnRNNSetClip(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNSetClip");
	auto args = (struct cudnnRNNSetClipArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnRNNSetClip(
				args->handle,
				args->rnnDesc,
				args->clipMode,
				args->clipNanOpt,
				args->lclip,
				args->rclip
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnRNNGetClip(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNGetClip");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetRNNProjectionLayers(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetRNNProjectionLayers");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetRNNProjectionLayers(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNProjectionLayers");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreatePersistentRNNPlan(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreatePersistentRNNPlan");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyPersistentRNNPlan(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyPersistentRNNPlan");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetPersistentRNNPlan(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetPersistentRNNPlan");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnBuildRNNDynamic(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBuildRNNDynamic");
	auto args = (struct cudnnBuildRNNDynamicArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnBuildRNNDynamic(
				args->handle,
				args->rnnDesc,
				args->miniBatch
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNTempSpaceSizes(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNTempSpaceSizes");
	auto args = (struct cudnnGetRNNTempSpaceSizesArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetRNNTempSpaceSizesResponse), alignof(cudnnGetRNNTempSpaceSizesResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetRNNTempSpaceSizesResponse*>(responsePayload);
            response->err = cudnnGetRNNTempSpaceSizes(
				args->handle,
				args->rnnDesc,
				args->fwdMode,
				args->xDesc,
				(args->workSpaceSize ? &(response->workSpaceSize) : NULL),
				(args->reserveSpaceSize ? &(response->reserveSpaceSize) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNParamsSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNParamsSize");
	auto args = (struct cudnnGetRNNParamsSizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetRNNParamsSizeResponse), alignof(cudnnGetRNNParamsSizeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetRNNParamsSizeResponse*>(responsePayload);
            response->err = cudnnGetRNNParamsSize(
				args->handle,
				args->rnnDesc,
				args->xDesc,
				(args->sizeInBytes ? &(response->sizeInBytes) : NULL),
				args->dataType
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNWeightSpaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNWeightSpaceSize");
	auto args = (struct cudnnGetRNNWeightSpaceSizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetRNNWeightSpaceSizeResponse), alignof(cudnnGetRNNWeightSpaceSizeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetRNNWeightSpaceSizeResponse*>(responsePayload);
            response->err = cudnnGetRNNWeightSpaceSize(
				args->handle,
				args->rnnDesc,
				(args->weightSpaceSize ? &(response->weightSpaceSize) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNLinLayerMatrixParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNLinLayerMatrixParams");
	auto args = (struct cudnnGetRNNLinLayerMatrixParamsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetRNNLinLayerMatrixParamsResponse), alignof(cudnnGetRNNLinLayerMatrixParamsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetRNNLinLayerMatrixParamsResponse*>(responsePayload);
            response->err = cudnnGetRNNLinLayerMatrixParams(
				args->handle,
				args->rnnDesc,
				args->pseudoLayer,
				args->xDesc,
				args->wDesc,
				args->w,
				args->linLayerID,
				args->linLayerMatDesc,
				(args->linLayerMat ? &(response->linLayerMat) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNLinLayerBiasParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNLinLayerBiasParams");
	auto args = (struct cudnnGetRNNLinLayerBiasParamsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetRNNLinLayerBiasParamsResponse), alignof(cudnnGetRNNLinLayerBiasParamsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetRNNLinLayerBiasParamsResponse*>(responsePayload);
            response->err = cudnnGetRNNLinLayerBiasParams(
				args->handle,
				args->rnnDesc,
				args->pseudoLayer,
				args->xDesc,
				args->wDesc,
				args->w,
				args->linLayerID,
				args->linLayerBiasDesc,
				(args->linLayerBias ? &(response->linLayerBias) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNWeightParams(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNWeightParams");
	auto args = (struct cudnnGetRNNWeightParamsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetRNNWeightParamsResponse), alignof(cudnnGetRNNWeightParamsResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetRNNWeightParamsResponse*>(responsePayload);
            response->err = cudnnGetRNNWeightParams(
				args->handle,
				args->rnnDesc,
				args->pseudoLayer,
				args->weightSpaceSize,
				args->weightSpace,
				args->linLayerID,
				args->mDesc,
				(args->mAddr ? &(response->mAddr) : NULL),
				args->bDesc,
				(args->bAddr ? &(response->bAddr) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnRNNForwardInference(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNForwardInference");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetRNNPaddingMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetRNNPaddingMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetRNNPaddingMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNPaddingMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateRNNDataDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateRNNDataDescriptor");
	auto args = (struct cudnnCreateRNNDataDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateRNNDataDescriptorResponse), alignof(cudnnCreateRNNDataDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateRNNDataDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateRNNDataDescriptor(
				(args->rnnDataDesc ? &(response->rnnDataDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDestroyRNNDataDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyRNNDataDescriptor");
	auto args = (struct cudnnDestroyRNNDataDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyRNNDataDescriptor(
				args->rnnDataDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNDataDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNDataDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnRNNForwardInferenceEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNForwardInferenceEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetRNNAlgorithmDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetRNNAlgorithmDescriptor");
	auto args = (struct cudnnSetRNNAlgorithmDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetRNNAlgorithmDescriptor(
				args->handle,
				args->rnnDesc,
				args->algoDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetRNNForwardInferenceAlgorithmMaxCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNForwardInferenceAlgorithmMaxCount");
	auto args = (struct cudnnGetRNNForwardInferenceAlgorithmMaxCountArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetRNNForwardInferenceAlgorithmMaxCountResponse), alignof(cudnnGetRNNForwardInferenceAlgorithmMaxCountResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetRNNForwardInferenceAlgorithmMaxCountResponse*>(responsePayload);
            response->err = cudnnGetRNNForwardInferenceAlgorithmMaxCount(
				args->handle,
				args->rnnDesc,
				(args->count ? &(response->count) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnFindRNNForwardInferenceAlgorithmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFindRNNForwardInferenceAlgorithmEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateSeqDataDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateSeqDataDescriptor");
	auto args = (struct cudnnCreateSeqDataDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateSeqDataDescriptorResponse), alignof(cudnnCreateSeqDataDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateSeqDataDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateSeqDataDescriptor(
				(args->seqDataDesc ? &(response->seqDataDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDestroySeqDataDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroySeqDataDescriptor");
	auto args = (struct cudnnDestroySeqDataDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroySeqDataDescriptor(
				args->seqDataDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnCreateAttnDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateAttnDescriptor");
	auto args = (struct cudnnCreateAttnDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateAttnDescriptorResponse), alignof(cudnnCreateAttnDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateAttnDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateAttnDescriptor(
				(args->attnDesc ? &(response->attnDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDestroyAttnDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyAttnDescriptor");
	auto args = (struct cudnnDestroyAttnDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyAttnDescriptor(
				args->attnDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetAttnDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetAttnDescriptor");
	auto args = (struct cudnnSetAttnDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnSetAttnDescriptor(
				args->attnDesc,
				args->attnMode,
				args->nHeads,
				args->smScaler,
				args->dataType,
				args->computePrec,
				args->mathType,
				args->attnDropoutDesc,
				args->postDropoutDesc,
				args->qSize,
				args->kSize,
				args->vSize,
				args->qProjSize,
				args->kProjSize,
				args->vProjSize,
				args->oProjSize,
				args->qoMaxSeqLength,
				args->kvMaxSeqLength,
				args->maxBatchSize,
				args->maxBeamSize
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetAttnDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetAttnDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetMultiHeadAttnBuffers(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetMultiHeadAttnBuffers");
	auto args = (struct cudnnGetMultiHeadAttnBuffersArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetMultiHeadAttnBuffersResponse), alignof(cudnnGetMultiHeadAttnBuffersResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetMultiHeadAttnBuffersResponse*>(responsePayload);
            response->err = cudnnGetMultiHeadAttnBuffers(
				args->handle,
				args->attnDesc,
				(args->weightSizeInBytes ? &(response->weightSizeInBytes) : NULL),
				(args->workSpaceSizeInBytes ? &(response->workSpaceSizeInBytes) : NULL),
				(args->reserveSpaceSizeInBytes ? &(response->reserveSpaceSizeInBytes) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetMultiHeadAttnWeights(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetMultiHeadAttnWeights");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnAdvInferVersionCheck(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnAdvInferVersionCheck");
	auto args = (struct cudnnAdvInferVersionCheckArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnAdvInferVersionCheck(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnRNNForwardTrainingEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNForwardTrainingEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnRNNBackwardDataEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNBackwardDataEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnRNNBackwardWeightsEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnRNNBackwardWeightsEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetRNNForwardTrainingAlgorithmMaxCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNForwardTrainingAlgorithmMaxCount");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnFindRNNForwardTrainingAlgorithmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFindRNNForwardTrainingAlgorithmEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetRNNBackwardDataAlgorithmMaxCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNBackwardDataAlgorithmMaxCount");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnFindRNNBackwardDataAlgorithmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFindRNNBackwardDataAlgorithmEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetRNNBackwardWeightsAlgorithmMaxCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnFindRNNBackwardWeightsAlgorithmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFindRNNBackwardWeightsAlgorithmEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateCTCLossDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateCTCLossDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetCTCLossDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetCTCLossDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetCTCLossDescriptorEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetCTCLossDescriptorEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetCTCLossDescriptor_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetCTCLossDescriptor_v8");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetCTCLossDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetCTCLossDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetCTCLossDescriptorEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetCTCLossDescriptorEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetCTCLossDescriptor_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetCTCLossDescriptor_v8");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyCTCLossDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyCTCLossDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCTCLoss(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCTCLoss");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCTCLoss_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCTCLoss_v8");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetCTCLossWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetCTCLossWorkspaceSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetCTCLossWorkspaceSize_v8(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetCTCLossWorkspaceSize_v8");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnAdvTrainVersionCheck(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnAdvTrainVersionCheck");
	auto args = (struct cudnnAdvTrainVersionCheckArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnAdvTrainVersionCheck(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnCreateConvolutionDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateConvolutionDescriptor");
	auto args = (struct cudnnCreateConvolutionDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnCreateConvolutionDescriptorResponse), alignof(cudnnCreateConvolutionDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnCreateConvolutionDescriptorResponse*>(responsePayload);
            response->err = cudnnCreateConvolutionDescriptor(
				(args->convDesc ? &(response->convDesc) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnDestroyConvolutionDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyConvolutionDescriptor");
	auto args = (struct cudnnDestroyConvolutionDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnDestroyConvolutionDescriptor(
				args->convDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnSetConvolutionMathType(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetConvolutionMathType");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionMathType(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionMathType");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetConvolutionGroupCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetConvolutionGroupCount");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionGroupCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionGroupCount");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetConvolutionReorderType(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetConvolutionReorderType");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionReorderType(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionReorderType");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetConvolution2dDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetConvolution2dDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolution2dDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolution2dDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionNdDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionNdDescriptor");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolution2dForwardOutputDim(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolution2dForwardOutputDim");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionForwardAlgorithmMaxCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionForwardAlgorithmMaxCount");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnFindConvolutionForwardAlgorithmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFindConvolutionForwardAlgorithmEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnIm2Col(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnIm2Col");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionForwardWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionForwardWorkspaceSize");
	auto args = (struct cudnnGetConvolutionForwardWorkspaceSizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetConvolutionForwardWorkspaceSizeResponse), alignof(cudnnGetConvolutionForwardWorkspaceSizeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetConvolutionForwardWorkspaceSizeResponse*>(responsePayload);
            response->err = cudnnGetConvolutionForwardWorkspaceSize(
				args->handle,
				args->xDesc,
				args->wDesc,
				args->convDesc,
				args->yDesc,
				args->algo,
				(args->sizeInBytes ? &(response->sizeInBytes) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnConvolutionBiasActivationForward(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnConvolutionBiasActivationForward");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionBackwardDataAlgorithmMaxCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
	auto args = (struct cudnnGetConvolutionBackwardDataAlgorithmMaxCountArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetConvolutionBackwardDataAlgorithmMaxCountResponse), alignof(cudnnGetConvolutionBackwardDataAlgorithmMaxCountResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetConvolutionBackwardDataAlgorithmMaxCountResponse*>(responsePayload);
            response->err = cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
				args->handle,
				(args->count ? &(response->count) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnFindConvolutionBackwardDataAlgorithm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFindConvolutionBackwardDataAlgorithm");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnFindConvolutionBackwardDataAlgorithmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFindConvolutionBackwardDataAlgorithmEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionBackwardDataAlgorithm_v7(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionBackwardDataAlgorithm_v7");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionBackwardDataWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionBackwardDataWorkspaceSize");
	auto args = (struct cudnnGetConvolutionBackwardDataWorkspaceSizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetConvolutionBackwardDataWorkspaceSizeResponse), alignof(cudnnGetConvolutionBackwardDataWorkspaceSizeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetConvolutionBackwardDataWorkspaceSizeResponse*>(responsePayload);
            response->err = cudnnGetConvolutionBackwardDataWorkspaceSize(
				args->handle,
				args->wDesc,
				args->dyDesc,
				args->convDesc,
				args->dxDesc,
				args->algo,
				(args->sizeInBytes ? &(response->sizeInBytes) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnConvolutionBackwardData(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnConvolutionBackwardData");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetFoldedConvBackwardDataDescriptors(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetFoldedConvBackwardDataDescriptors");
	auto args = (struct cudnnGetFoldedConvBackwardDataDescriptorsArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnGetFoldedConvBackwardDataDescriptors(
				args->handle,
				args->filterDesc,
				args->diffDesc,
				args->convDesc,
				args->gradDesc,
				args->transformFormat,
				args->foldedFilterDesc,
				args->paddedDiffDesc,
				args->foldedConvDesc,
				args->foldedGradDesc,
				args->filterFoldTransDesc,
				args->diffPadTransDesc,
				args->gradFoldTransDesc,
				args->gradUnfoldTransDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnCnnInferVersionCheck(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCnnInferVersionCheck");
	auto args = (struct cudnnCnnInferVersionCheckArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnCnnInferVersionCheck(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
	auto args = (struct cudnnGetConvolutionBackwardFilterAlgorithmMaxCountArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnGetConvolutionBackwardFilterAlgorithmMaxCountResponse), alignof(cudnnGetConvolutionBackwardFilterAlgorithmMaxCountResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnGetConvolutionBackwardFilterAlgorithmMaxCountResponse*>(responsePayload);
            response->err = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
				args->handle,
				(args->count ? &(response->count) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnFindConvolutionBackwardFilterAlgorithm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFindConvolutionBackwardFilterAlgorithm");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnFindConvolutionBackwardFilterAlgorithmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFindConvolutionBackwardFilterAlgorithmEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionBackwardFilterAlgorithm_v7(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionBackwardFilterAlgorithm_v7");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetConvolutionBackwardFilterWorkspaceSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetConvolutionBackwardFilterWorkspaceSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnConvolutionBackwardFilter(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnConvolutionBackwardFilter");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnConvolutionBackwardBias(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnConvolutionBackwardBias");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateFusedOpsConstParamPack(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateFusedOpsConstParamPack");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyFusedOpsConstParamPack(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyFusedOpsConstParamPack");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetFusedOpsConstParamPackAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetFusedOpsConstParamPackAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetFusedOpsConstParamPackAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetFusedOpsConstParamPackAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateFusedOpsVariantParamPack(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateFusedOpsVariantParamPack");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyFusedOpsVariantParamPack(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyFusedOpsVariantParamPack");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnSetFusedOpsVariantParamPackAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnSetFusedOpsVariantParamPackAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnGetFusedOpsVariantParamPackAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnGetFusedOpsVariantParamPackAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCreateFusedOpsPlan(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCreateFusedOpsPlan");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnDestroyFusedOpsPlan(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnDestroyFusedOpsPlan");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnMakeFusedOpsPlan(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnMakeFusedOpsPlan");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnFusedOpsExecute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnFusedOpsExecute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudnnCnnTrainVersionCheck(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnCnnTrainVersionCheck");
	auto args = (struct cudnnCnnTrainVersionCheckArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnCnnTrainVersionCheck(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnBackendCreateDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBackendCreateDescriptor");
	auto args = (struct cudnnBackendCreateDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnBackendCreateDescriptorResponse), alignof(cudnnBackendCreateDescriptorResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnBackendCreateDescriptorResponse*>(responsePayload);
            response->err = cudnnBackendCreateDescriptor(
				args->descriptorType,
				(args->descriptor ? &(response->descriptor) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnBackendDestroyDescriptor(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBackendDestroyDescriptor");
	auto args = (struct cudnnBackendDestroyDescriptorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnBackendDestroyDescriptor(
				args->descriptor
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnBackendInitialize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBackendInitialize");
	auto args = (struct cudnnBackendInitializeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnBackendInitialize(
				args->descriptor
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudnnBackendFinalize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudnnBackendFinalize");
	auto args = (struct cudnnBackendFinalizeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudnnStatus_t), alignof(cudnnStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudnnStatus_t*>(responsePayload);
            *response = cudnnBackendFinalize(
				args->descriptor
            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasGetVersion_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetVersion_v2");
	auto args = (struct cublasGetVersion_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasGetVersion_v2Response), alignof(cublasGetVersion_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasGetVersion_v2Response*>(responsePayload);
            response->err = cublasGetVersion_v2(
				args->handle,
				(args->version ? &(response->version) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasGetProperty(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetProperty");
	auto args = (struct cublasGetPropertyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasGetPropertyResponse), alignof(cublasGetPropertyResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasGetPropertyResponse*>(responsePayload);
            response->err = cublasGetProperty(
				args->type,
				(args->value ? &(response->value) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasGetCudartVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetCudartVersion");
	auto args = (struct cublasGetCudartVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(size_t), alignof(size_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<size_t*>(responsePayload);
            *response = cublasGetCudartVersion(

            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasGetStream_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetStream_v2");
	auto args = (struct cublasGetStream_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasGetStream_v2Response), alignof(cublasGetStream_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasGetStream_v2Response*>(responsePayload);
            response->err = cublasGetStream_v2(
				args->handle,
				(args->streamId ? &(response->streamId) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasGetPointerMode_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetPointerMode_v2");
	auto args = (struct cublasGetPointerMode_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasGetPointerMode_v2Response), alignof(cublasGetPointerMode_v2Response))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasGetPointerMode_v2Response*>(responsePayload);
            response->err = cublasGetPointerMode_v2(
				args->handle,
				(args->mode ? &(response->mode) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasSetPointerMode_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetPointerMode_v2");
	auto args = (struct cublasSetPointerMode_v2Arg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasSetPointerMode_v2(
				args->handle,
				args->mode
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasGetAtomicsMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetAtomicsMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSetAtomicsMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetAtomicsMode");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetMathMode(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetMathMode");
	auto args = (struct cublasGetMathModeArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasGetMathModeResponse), alignof(cublasGetMathModeResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasGetMathModeResponse*>(responsePayload);
            response->err = cublasGetMathMode(
				args->handle,
				(args->mode ? &(response->mode) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasGetSmCountTarget(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetSmCountTarget");
	auto args = (struct cublasGetSmCountTargetArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasGetSmCountTargetResponse), alignof(cublasGetSmCountTargetResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasGetSmCountTargetResponse*>(responsePayload);
            response->err = cublasGetSmCountTarget(
				args->handle,
				(args->smCountTarget ? &(response->smCountTarget) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasSetSmCountTarget(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetSmCountTarget");
	auto args = (struct cublasSetSmCountTargetArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasSetSmCountTarget(
				args->handle,
				args->smCountTarget
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasGetStatusName(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetStatusName");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetStatusString(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetStatusString");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLoggerConfigure(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLoggerConfigure");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSetLoggerCallback(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetLoggerCallback");
	auto args = (struct cublasSetLoggerCallbackArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasSetLoggerCallback(
				args->userCallback
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasGetLoggerCallback(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetLoggerCallback");
	auto args = (struct cublasGetLoggerCallbackArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasGetLoggerCallbackResponse), alignof(cublasGetLoggerCallbackResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasGetLoggerCallbackResponse*>(responsePayload);
            response->err = cublasGetLoggerCallback(
				(args->userCallback ? &(response->userCallback) : NULL)
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasSetVector(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetVector");
	auto args = (struct cublasSetVectorArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasSetVector(
				args->n,
				args->elemSize,
				args->x,
				args->incx,
				args->devicePtr,
				args->incy
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasSetVector_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetVector_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetVector(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetVector");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetVector_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetVector_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSetMatrix(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetMatrix");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSetMatrix_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetMatrix_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetMatrix(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetMatrix");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetMatrix_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetMatrix_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSetVectorAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetVectorAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSetVectorAsync_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetVectorAsync_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetVectorAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetVectorAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetVectorAsync_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetVectorAsync_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSetMatrixAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetMatrixAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSetMatrixAsync_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSetMatrixAsync_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetMatrixAsync(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetMatrixAsync");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGetMatrixAsync_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGetMatrixAsync_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasXerbla(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasXerbla");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasNrm2Ex(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasNrm2Ex");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasNrm2Ex_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasNrm2Ex_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSnrm2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSnrm2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSnrm2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSnrm2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDnrm2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDnrm2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDnrm2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDnrm2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasScnrm2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasScnrm2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasScnrm2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasScnrm2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDznrm2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDznrm2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDznrm2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDznrm2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDotEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDotEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDotEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDotEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDotcEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDotcEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDotcEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDotcEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSdot_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSdot_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSdot_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSdot_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDdot_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDdot_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDdot_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDdot_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCdotu_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCdotu_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCdotu_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCdotu_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCdotc_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCdotc_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCdotc_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCdotc_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdotu_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdotu_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdotu_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdotu_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdotc_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdotc_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdotc_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdotc_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasScalEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasScalEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasScalEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasScalEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSscal_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSscal_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSscal_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSscal_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDscal_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDscal_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDscal_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDscal_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCscal_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCscal_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCscal_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCscal_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsscal_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsscal_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsscal_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsscal_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZscal_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZscal_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZscal_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZscal_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdscal_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdscal_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdscal_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdscal_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasAxpyEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasAxpyEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasAxpyEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasAxpyEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSaxpy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSaxpy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSaxpy_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSaxpy_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDaxpy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDaxpy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDaxpy_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDaxpy_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCaxpy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCaxpy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCaxpy_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCaxpy_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZaxpy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZaxpy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZaxpy_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZaxpy_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCopyEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCopyEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCopyEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCopyEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasScopy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasScopy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasScopy_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasScopy_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDcopy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDcopy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDcopy_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDcopy_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCcopy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCcopy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCcopy_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCcopy_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZcopy_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZcopy_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZcopy_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZcopy_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSswap_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSswap_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSswap_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSswap_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDswap_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDswap_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDswap_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDswap_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCswap_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCswap_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCswap_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCswap_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZswap_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZswap_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZswap_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZswap_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSwapEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSwapEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSwapEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSwapEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIsamax_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIsamax_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIsamax_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIsamax_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIdamax_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIdamax_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIdamax_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIdamax_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIcamax_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIcamax_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIcamax_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIcamax_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIzamax_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIzamax_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIzamax_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIzamax_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIamaxEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIamaxEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIamaxEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIamaxEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIsamin_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIsamin_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIsamin_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIsamin_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIdamin_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIdamin_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIdamin_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIdamin_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIcamin_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIcamin_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIcamin_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIcamin_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIzamin_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIzamin_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIzamin_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIzamin_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIaminEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIaminEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasIaminEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasIaminEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasAsumEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasAsumEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasAsumEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasAsumEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSasum_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSasum_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSasum_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSasum_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDasum_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDasum_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDasum_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDasum_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasScasum_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasScasum_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasScasum_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasScasum_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDzasum_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDzasum_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDzasum_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDzasum_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSrot_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSrot_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSrot_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSrot_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDrot_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDrot_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDrot_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDrot_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCrot_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCrot_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCrot_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCrot_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsrot_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsrot_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsrot_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsrot_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZrot_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZrot_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZrot_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZrot_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdrot_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdrot_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdrot_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdrot_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasRotEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasRotEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasRotEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasRotEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSrotg_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSrotg_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDrotg_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDrotg_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCrotg_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCrotg_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZrotg_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZrotg_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasRotgEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasRotgEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSrotm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSrotm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSrotm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSrotm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDrotm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDrotm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDrotm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDrotm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasRotmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasRotmEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasRotmEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasRotmEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSrotmg_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSrotmg_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDrotmg_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDrotmg_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasRotmgEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasRotmgEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStpmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStpmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStpmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStpmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtpmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtpmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtpmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtpmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtpmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtpmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtpmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtpmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtpmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtpmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtpmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtpmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStpsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStpsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStpsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStpsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtpsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtpsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtpsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtpsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtpsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtpsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtpsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtpsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtpsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtpsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtpsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtpsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStbsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStbsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStbsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStbsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtbsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtbsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtbsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtbsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtbsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtbsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtbsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtbsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtbsv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtbsv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtbsv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtbsv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsymv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsymv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsymv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsymv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsymv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsymv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsymv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsymv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsymv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsymv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsymv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsymv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsymv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsymv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsymv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsymv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChemv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChemv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChemv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChemv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhemv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhemv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhemv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhemv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhbmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhbmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhbmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhbmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSspmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSspmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSspmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSspmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDspmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDspmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDspmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDspmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChpmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChpmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChpmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChpmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhpmv_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhpmv_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhpmv_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhpmv_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSger_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSger_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSger_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSger_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDger_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDger_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDger_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDger_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgeru_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgeru_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgeru_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgeru_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgerc_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgerc_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgerc_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgerc_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgeru_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgeru_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgeru_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgeru_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgerc_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgerc_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgerc_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgerc_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyr_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyr_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyr_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyr_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyr_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyr_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyr_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyr_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyr_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyr_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyr_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyr_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyr_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyr_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyr_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyr_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCher_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCher_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCher_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCher_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZher_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZher_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZher_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZher_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSspr_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSspr_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSspr_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSspr_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDspr_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDspr_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDspr_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDspr_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChpr_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChpr_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChpr_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChpr_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhpr_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhpr_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhpr_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhpr_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyr2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyr2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyr2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyr2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyr2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyr2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyr2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyr2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyr2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyr2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyr2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyr2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyr2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyr2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyr2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyr2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCher2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCher2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCher2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCher2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZher2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZher2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZher2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZher2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSspr2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSspr2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSspr2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSspr2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDspr2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDspr2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDspr2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDspr2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChpr2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChpr2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChpr2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChpr2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhpr2_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhpr2_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhpr2_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhpr2_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemvBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemvBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemvBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemvBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemvBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemvBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemvBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemvBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHSHgemvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHSHgemvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHSHgemvBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHSHgemvBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHSSgemvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHSSgemvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHSSgemvBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHSSgemvBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasTSTgemvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasTSTgemvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasTSTgemvBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasTSTgemvBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasTSSgemvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasTSSgemvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasTSSgemvBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasTSSgemvBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemvStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemvStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemvStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemvStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemvStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemvStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemvStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemvStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemvStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemvStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemvStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemvStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemvStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemvStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemvStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemvStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHSHgemvStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHSHgemvStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHSHgemvStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHSHgemvStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHSSgemvStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHSSgemvStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHSSgemvStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHSSgemvStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasTSTgemvStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasTSTgemvStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasTSTgemvStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasTSTgemvStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasTSSgemvStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasTSSgemvStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasTSSgemvStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasTSSgemvStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm3m(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm3m");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm3m_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm3m_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm3mEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm3mEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm3mEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm3mEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemm3m(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemm3m");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemm3m_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemm3m_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHgemm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHgemm");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHgemm_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHgemm_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemmEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemmEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGemmEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGemmEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemmEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemmEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemmEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemmEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyrk_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyrk_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyrk_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyrk_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyrk_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyrk_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyrk_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyrk_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyrk_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyrk_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyrk_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyrk_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyrk_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyrk_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyrk_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyrk_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyrkEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyrkEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyrkEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyrkEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyrk3mEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyrk3mEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyrk3mEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyrk3mEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCherk_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCherk_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCherk_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCherk_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZherk_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZherk_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZherk_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZherk_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCherkEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCherkEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCherkEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCherkEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCherk3mEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCherk3mEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCherk3mEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCherk3mEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyr2k_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyr2k_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyr2k_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyr2k_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyr2k_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyr2k_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyr2k_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyr2k_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyr2k_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyr2k_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyr2k_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyr2k_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyr2k_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyr2k_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyr2k_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyr2k_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCher2k_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCher2k_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCher2k_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCher2k_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZher2k_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZher2k_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZher2k_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZher2k_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyrkx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyrkx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsyrkx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsyrkx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyrkx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyrkx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsyrkx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsyrkx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyrkx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyrkx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsyrkx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsyrkx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyrkx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyrkx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsyrkx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsyrkx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCherkx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCherkx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCherkx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCherkx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZherkx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZherkx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZherkx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZherkx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsymm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsymm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSsymm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSsymm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsymm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsymm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDsymm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDsymm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsymm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsymm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCsymm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCsymm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsymm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsymm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZsymm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZsymm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChemm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChemm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasChemm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasChemm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhemm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhemm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZhemm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZhemm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrsm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrsm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrsm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrsm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrsm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrsm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrsm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrsm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrsm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrsm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrsm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrsm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrsm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrsm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrsm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrsm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrmm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrmm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrmm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrmm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrmm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrmm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrmm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrmm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrmm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrmm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrmm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrmm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrmm_v2(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrmm_v2");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrmm_v2_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrmm_v2_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHgemmBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHgemmBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHgemmBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHgemmBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemmBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemmBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemmBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemmBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemmBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemmBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemmBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemmBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemmBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemmBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemmBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemmBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm3mBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm3mBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm3mBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm3mBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemmBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemmBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemmBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemmBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHgemmStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHgemmStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasHgemmStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasHgemmStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgemmStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgemmStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemmStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemmStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgemmStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgemmStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemmStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemmStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemmStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemmStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm3mStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm3mStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgemm3mStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgemm3mStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemmStridedBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemmStridedBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgemmStridedBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgemmStridedBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGemmBatchedEx(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGemmBatchedEx");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGemmBatchedEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGemmBatchedEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasGemmStridedBatchedEx_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasGemmStridedBatchedEx_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgeam(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgeam");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgeam_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgeam_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgeam(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgeam");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgeam_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgeam_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgeam(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgeam");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgeam_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgeam_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgeam(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgeam");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgeam_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgeam_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrsmBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrsmBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrsmBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrsmBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrsmBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrsmBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrsmBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrsmBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrsmBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrsmBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrsmBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrsmBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrsmBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrsmBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrsmBatched_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrsmBatched_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSdgmm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSdgmm");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSdgmm_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSdgmm_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDdgmm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDdgmm");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDdgmm_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDdgmm_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCdgmm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCdgmm");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCdgmm_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCdgmm_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdgmm(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdgmm");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZdgmm_64(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZdgmm_64");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSmatinvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSmatinvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDmatinvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDmatinvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCmatinvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCmatinvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZmatinvBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZmatinvBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgeqrfBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgeqrfBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgeqrfBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgeqrfBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgeqrfBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgeqrfBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgeqrfBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgeqrfBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgelsBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgelsBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgelsBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgelsBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgelsBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgelsBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgelsBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgelsBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStpttr(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStpttr");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtpttr(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtpttr");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtpttr(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtpttr");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtpttr(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtpttr");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasStrttp(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasStrttp");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDtrttp(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDtrttp");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCtrttp(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCtrttp");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZtrttp(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZtrttp");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgetrfBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgetrfBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgetrfBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgetrfBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgetrfBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgetrfBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgetrfBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgetrfBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgetriBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgetriBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgetriBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgetriBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgetriBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgetriBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgetriBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgetriBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasSgetrsBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasSgetrsBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasDgetrsBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasDgetrsBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasCgetrsBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasCgetrsBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasZgetrsBatched(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasZgetrsBatched");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasUint8gemmBias(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasUint8gemmBias");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cudaProfilerStart(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaProfilerStart");
	auto args = (struct cudaProfilerStartArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaProfilerStart(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cudaProfilerStop(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cudaProfilerStop");
	auto args = (struct cudaProfilerStopArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cudaError_t), alignof(cudaError_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cudaError_t*>(responsePayload);
            *response = cudaProfilerStop(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cuProfilerInitialize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuProfilerInitialize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuProfilerStart(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuProfilerStart");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cuProfilerStop(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cuProfilerStop");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetErrorString(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetErrorString");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcVersion");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetNumSupportedArchs(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetNumSupportedArchs");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetSupportedArchs(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetSupportedArchs");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcCreateProgram(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcCreateProgram");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcDestroyProgram(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcDestroyProgram");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcCompileProgram(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcCompileProgram");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetPTXSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetPTXSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetPTX(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetPTX");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetCUBINSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetCUBINSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetCUBIN(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetCUBIN");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetLTOIRSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetLTOIRSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetLTOIR(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetLTOIR");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetOptiXIRSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetOptiXIRSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetOptiXIR(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetOptiXIR");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetProgramLogSize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetProgramLogSize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetProgramLog(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetProgramLog");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcAddNameExpression(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcAddNameExpression");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_nvrtcGetLoweredName(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: nvrtcGetLoweredName");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtDestroy");
	auto args = (struct cublasLtDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtDestroy(
				args->lightHandle
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasLtGetStatusName(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtGetStatusName");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtGetStatusString(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtGetStatusString");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtGetVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtGetVersion");
	auto args = (struct cublasLtGetVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(size_t), alignof(size_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<size_t*>(responsePayload);
            *response = cublasLtGetVersion(

            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasLtGetCudartVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtGetCudartVersion");
	auto args = (struct cublasLtGetCudartVersionArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(size_t), alignof(size_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<size_t*>(responsePayload);
            *response = cublasLtGetCudartVersion(

            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasLtGetProperty(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtGetProperty");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtHeuristicsCacheGetCapacity(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtHeuristicsCacheGetCapacity");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtHeuristicsCacheSetCapacity(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtHeuristicsCacheSetCapacity");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtDisableCpuInstructionsSetMask(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtDisableCpuInstructionsSetMask");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatrixTransform(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixTransform");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatrixLayoutInit_internal(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixLayoutInit_internal");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatrixLayoutDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixLayoutDestroy");
	auto args = (struct cublasLtMatrixLayoutDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatrixLayoutDestroy(
				args->matLayout
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasLtMatrixLayoutGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixLayoutGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulDescInit_internal(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulDescInit_internal");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulDescDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulDescDestroy");
	auto args = (struct cublasLtMatmulDescDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatmulDescDestroy(
				args->matmulDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasLtMatmulDescGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulDescGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatrixTransformDescInit_internal(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixTransformDescInit_internal");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatrixTransformDescCreate(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixTransformDescCreate");
	auto args = (struct cublasLtMatrixTransformDescCreateArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasLtMatrixTransformDescCreateResponse), alignof(cublasLtMatrixTransformDescCreateResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasLtMatrixTransformDescCreateResponse*>(responsePayload);
            response->err = cublasLtMatrixTransformDescCreate(
				(args->transformDesc ? &(response->transformDesc) : NULL),
				args->scaleType
			);

            CHECK_CUDA_ERROR(response->err);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasLtMatrixTransformDescDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixTransformDescDestroy");
	auto args = (struct cublasLtMatrixTransformDescDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatrixTransformDescDestroy(
				args->transformDesc
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasLtMatrixTransformDescSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixTransformDescSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatrixTransformDescGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatrixTransformDescGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulPreferenceInit_internal(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulPreferenceInit_internal");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulPreferenceDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulPreferenceDestroy");
	auto args = (struct cublasLtMatmulPreferenceDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtMatmulPreferenceDestroy(
				args->pref
            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_cublasLtMatmulPreferenceGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulPreferenceGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulAlgoGetIds(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulAlgoGetIds");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulAlgoInit(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulAlgoInit");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulAlgoCheck(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulAlgoCheck");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulAlgoCapGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulAlgoCapGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulAlgoConfigSetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulAlgoConfigSetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtMatmulAlgoConfigGetAttribute(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtMatmulAlgoConfigGetAttribute");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtLoggerSetCallback(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtLoggerSetCallback");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtLoggerSetFile(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtLoggerSetFile");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtLoggerOpenFile(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtLoggerOpenFile");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtLoggerSetLevel(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtLoggerSetLevel");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtLoggerSetMask(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtLoggerSetMask");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_cublasLtLoggerForceDisable(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: cublasLtLoggerForceDisable");
	auto args = (struct cublasLtLoggerForceDisableArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(cublasStatus_t), alignof(cublasStatus_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<cublasStatus_t*>(responsePayload);
            *response = cublasLtLoggerForceDisable(

            );
            CHECK_CUDA_ERROR(*response);
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_ncclGetVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclGetVersion");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclGetVersion(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclGetVersion");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclGetUniqueId(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclGetUniqueId");
	auto args = (struct ncclGetUniqueIdArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(ncclGetUniqueIdResponse), alignof(ncclGetUniqueIdResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<ncclGetUniqueIdResponse*>(responsePayload);
            response->err = ncclGetUniqueId(
				(args->uniqueId ? &(response->uniqueId) : NULL)
			);

            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_pncclGetUniqueId(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclGetUniqueId");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommInitRankConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommInitRankConfig");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclCommInitRankConfig(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommInitRankConfig");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommInitRank(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommInitRank");
	auto args = (struct ncclCommInitRankArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(ncclCommInitRankResponse), alignof(ncclCommInitRankResponse))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<ncclCommInitRankResponse*>(responsePayload);
            response->err = ncclCommInitRank(
				(args->comm ? &(response->comm) : NULL),
				args->nranks,
				args->commId,
				args->rank
			);

            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_pncclCommInitRank(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommInitRank");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommInitAll(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommInitAll");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclCommInitAll(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommInitAll");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommFinalize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommFinalize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclCommFinalize(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommFinalize");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommDestroy");
	auto args = (struct ncclCommDestroyArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    iox_server->loan(requestHeader, sizeof(ncclResult_t), alignof(ncclResult_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<ncclResult_t*>(responsePayload);
            *response = ncclCommDestroy(
				args->comm
            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_pncclCommDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommAbort(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommAbort");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclCommAbort(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommAbort");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclGetErrorString(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclGetErrorString");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclGetErrorString(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclGetErrorString");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclGetLastError(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclGetLastError");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclGetError(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclGetError");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommGetAsyncError(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommGetAsyncError");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclCommGetAsyncError(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommGetAsyncError");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommCount");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclCommCount(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommCount");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommCuDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommCuDevice");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclCommCuDevice(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommCuDevice");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclCommUserRank(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclCommUserRank");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclCommUserRank(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclCommUserRank");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclRedOpCreatePreMulSum(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclRedOpCreatePreMulSum");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclRedOpCreatePreMulSum(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclRedOpCreatePreMulSum");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclRedOpDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclRedOpDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclRedOpDestroy(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclRedOpDestroy");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclReduce(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclReduce");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclReduce(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclReduce");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclBcast(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclBcast");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclBcast(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclBcast");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclBroadcast(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclBroadcast");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclBroadcast(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclBroadcast");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclAllReduce(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclAllReduce");
	auto args = (struct ncclAllReduceArg *) __args;
	auto requestHeader = iox::popo::RequestHeader::fromPayload(requestPayload);

    auto msg_header = static_cast<const MessageHeader_t*>(requestPayload);
    int32_t client_uid = msg_header->client_id;

    cudaStream_t __stream = args->stream;

    // If client submits to default stream, set to a re-assigned stream
    if (__stream == nullptr) {
        __stream = client_data_all[client_uid].default_stream;
    }

    iox_server->loan(requestHeader, sizeof(ncclResult_t), alignof(ncclResult_t))
        .and_then([&](auto& responsePayload) {
            auto response = static_cast<ncclResult_t*>(responsePayload);
            *response = ncclAllReduce(
				args->sendbuff,
				args->recvbuff,
				args->count,
				args->datatype,
				args->op,
				args->comm,
				__stream
            );
            
            iox_server->send(response).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Response: ", error); });
        })
        .or_else(
            [&](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Response: ", error); });
}
void TallyServer::handle_pncclAllReduce(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclAllReduce");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclReduceScatter(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclReduceScatter");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclReduceScatter(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclReduceScatter");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclAllGather(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclAllGather");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclAllGather(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclAllGather");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclSend(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclSend");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclSend(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclSend");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclRecv(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclRecv");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclRecv(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclRecv");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclGroupStart(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclGroupStart");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclGroupStart(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclGroupStart");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_ncclGroupEnd(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: ncclGroupEnd");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
void TallyServer::handle_pncclGroupEnd(void *__args, iox::popo::UntypedServer *iox_server, const void* const requestPayload)
{
	TALLY_SPD_LOG("Received request: pncclGroupEnd");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}
