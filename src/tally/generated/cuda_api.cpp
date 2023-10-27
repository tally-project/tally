
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

void *cuda_handle;
void *cudart_handle;
void *cudnn_handle;
void *cublas_handle;
void *cublasLt_handle;

void __attribute__((constructor)) register_cuda_handles()
{
	cuda_handle = dlopen(LIBCUDA_PATH, RTLD_LAZY);
	cudart_handle = dlopen(LIBCUDART_PATH, RTLD_LAZY);
	cudnn_handle = dlopen(LIBCUDNN_PATH, RTLD_LAZY);
	cublas_handle = dlopen(LIBCUBLAS_PATH, RTLD_LAZY);
	cublasLt_handle = dlopen(LIBCUBLASLT_PATH, RTLD_LAZY);
}

CUresult (*lcuGetErrorString) (CUresult  error, const char ** pStr) =
	(CUresult (*) (CUresult  error, const char ** pStr)) dlsym(cuda_handle, "cuGetErrorString");

CUresult (*lcuGetErrorName) (CUresult  error, const char ** pStr) =
	(CUresult (*) (CUresult  error, const char ** pStr)) dlsym(cuda_handle, "cuGetErrorName");

CUresult (*lcuInit) (unsigned int  Flags) =
	(CUresult (*) (unsigned int  Flags)) dlsym(cuda_handle, "cuInit");

CUresult (*lcuDriverGetVersion) (int * driverVersion) =
	(CUresult (*) (int * driverVersion)) dlsym(cuda_handle, "cuDriverGetVersion");

CUresult (*lcuDeviceGet) (CUdevice * device, int  ordinal) =
	(CUresult (*) (CUdevice * device, int  ordinal)) dlsym(cuda_handle, "cuDeviceGet");

CUresult (*lcuDeviceGetCount) (int * count) =
	(CUresult (*) (int * count)) dlsym(cuda_handle, "cuDeviceGetCount");

CUresult (*lcuDeviceGetName) (char * name, int  len, CUdevice  dev) =
	(CUresult (*) (char * name, int  len, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetName");

CUresult (*lcuDeviceGetUuid) (CUuuid * uuid, CUdevice  dev) =
	(CUresult (*) (CUuuid * uuid, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetUuid");

CUresult (*lcuDeviceGetUuid_v2) (CUuuid * uuid, CUdevice  dev) =
	(CUresult (*) (CUuuid * uuid, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetUuid_v2");

CUresult (*lcuDeviceGetLuid) (char * luid, unsigned int * deviceNodeMask, CUdevice  dev) =
	(CUresult (*) (char * luid, unsigned int * deviceNodeMask, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetLuid");

CUresult (*lcuDeviceTotalMem_v2) (size_t * bytes, CUdevice  dev) =
	(CUresult (*) (size_t * bytes, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceTotalMem_v2");

CUresult (*lcuDeviceGetTexture1DLinearMaxWidth) (size_t * maxWidthInElements, CUarray_format  format, unsigned  numChannels, CUdevice  dev) =
	(CUresult (*) (size_t * maxWidthInElements, CUarray_format  format, unsigned  numChannels, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetTexture1DLinearMaxWidth");

CUresult (*lcuDeviceGetAttribute) (int * pi, CUdevice_attribute  attrib, CUdevice  dev) =
	(CUresult (*) (int * pi, CUdevice_attribute  attrib, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetAttribute");

CUresult (*lcuDeviceGetNvSciSyncAttributes) (void * nvSciSyncAttrList, CUdevice  dev, int  flags) =
	(CUresult (*) (void * nvSciSyncAttrList, CUdevice  dev, int  flags)) dlsym(cuda_handle, "cuDeviceGetNvSciSyncAttributes");

CUresult (*lcuDeviceSetMemPool) (CUdevice  dev, CUmemoryPool  pool) =
	(CUresult (*) (CUdevice  dev, CUmemoryPool  pool)) dlsym(cuda_handle, "cuDeviceSetMemPool");

CUresult (*lcuDeviceGetMemPool) (CUmemoryPool * pool, CUdevice  dev) =
	(CUresult (*) (CUmemoryPool * pool, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetMemPool");

CUresult (*lcuDeviceGetDefaultMemPool) (CUmemoryPool * pool_out, CUdevice  dev) =
	(CUresult (*) (CUmemoryPool * pool_out, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetDefaultMemPool");

CUresult (*lcuDeviceGetExecAffinitySupport) (int * pi, CUexecAffinityType  type, CUdevice  dev) =
	(CUresult (*) (int * pi, CUexecAffinityType  type, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetExecAffinitySupport");

CUresult (*lcuFlushGPUDirectRDMAWrites) (CUflushGPUDirectRDMAWritesTarget  target, CUflushGPUDirectRDMAWritesScope  scope) =
	(CUresult (*) (CUflushGPUDirectRDMAWritesTarget  target, CUflushGPUDirectRDMAWritesScope  scope)) dlsym(cuda_handle, "cuFlushGPUDirectRDMAWrites");

CUresult (*lcuDeviceGetProperties) (CUdevprop * prop, CUdevice  dev) =
	(CUresult (*) (CUdevprop * prop, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetProperties");

CUresult (*lcuDeviceComputeCapability) (int * major, int * minor, CUdevice  dev) =
	(CUresult (*) (int * major, int * minor, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceComputeCapability");

CUresult (*lcuDevicePrimaryCtxRetain) (CUcontext * pctx, CUdevice  dev) =
	(CUresult (*) (CUcontext * pctx, CUdevice  dev)) dlsym(cuda_handle, "cuDevicePrimaryCtxRetain");

CUresult (*lcuDevicePrimaryCtxRelease_v2) (CUdevice  dev) =
	(CUresult (*) (CUdevice  dev)) dlsym(cuda_handle, "cuDevicePrimaryCtxRelease_v2");

CUresult (*lcuDevicePrimaryCtxSetFlags_v2) (CUdevice  dev, unsigned int  flags) =
	(CUresult (*) (CUdevice  dev, unsigned int  flags)) dlsym(cuda_handle, "cuDevicePrimaryCtxSetFlags_v2");

CUresult (*lcuDevicePrimaryCtxGetState) (CUdevice  dev, unsigned int * flags, int * active) =
	(CUresult (*) (CUdevice  dev, unsigned int * flags, int * active)) dlsym(cuda_handle, "cuDevicePrimaryCtxGetState");

CUresult (*lcuDevicePrimaryCtxReset_v2) (CUdevice  dev) =
	(CUresult (*) (CUdevice  dev)) dlsym(cuda_handle, "cuDevicePrimaryCtxReset_v2");

CUresult (*lcuCtxCreate_v2) (CUcontext * pctx, unsigned int  flags, CUdevice  dev) =
	(CUresult (*) (CUcontext * pctx, unsigned int  flags, CUdevice  dev)) dlsym(cuda_handle, "cuCtxCreate_v2");

CUresult (*lcuCtxCreate_v3) (CUcontext * pctx, CUexecAffinityParam * paramsArray, int  numParams, unsigned int  flags, CUdevice  dev) =
	(CUresult (*) (CUcontext * pctx, CUexecAffinityParam * paramsArray, int  numParams, unsigned int  flags, CUdevice  dev)) dlsym(cuda_handle, "cuCtxCreate_v3");

CUresult (*lcuCtxDestroy_v2) (CUcontext  ctx) =
	(CUresult (*) (CUcontext  ctx)) dlsym(cuda_handle, "cuCtxDestroy_v2");

CUresult (*lcuCtxPushCurrent_v2) (CUcontext  ctx) =
	(CUresult (*) (CUcontext  ctx)) dlsym(cuda_handle, "cuCtxPushCurrent_v2");

CUresult (*lcuCtxPopCurrent_v2) (CUcontext * pctx) =
	(CUresult (*) (CUcontext * pctx)) dlsym(cuda_handle, "cuCtxPopCurrent_v2");

CUresult (*lcuCtxSetCurrent) (CUcontext  ctx) =
	(CUresult (*) (CUcontext  ctx)) dlsym(cuda_handle, "cuCtxSetCurrent");

CUresult (*lcuCtxGetCurrent) (CUcontext * pctx) =
	(CUresult (*) (CUcontext * pctx)) dlsym(cuda_handle, "cuCtxGetCurrent");

CUresult (*lcuCtxGetDevice) (CUdevice * device) =
	(CUresult (*) (CUdevice * device)) dlsym(cuda_handle, "cuCtxGetDevice");

CUresult (*lcuCtxGetFlags) (unsigned int * flags) =
	(CUresult (*) (unsigned int * flags)) dlsym(cuda_handle, "cuCtxGetFlags");

CUresult (*lcuCtxSetFlags) (unsigned int  flags) =
	(CUresult (*) (unsigned int  flags)) dlsym(cuda_handle, "cuCtxSetFlags");

CUresult (*lcuCtxGetId) (CUcontext  ctx, unsigned long long * ctxId) =
	(CUresult (*) (CUcontext  ctx, unsigned long long * ctxId)) dlsym(cuda_handle, "cuCtxGetId");

CUresult (*lcuCtxSynchronize) () =
	(CUresult (*) ()) dlsym(cuda_handle, "cuCtxSynchronize");

CUresult (*lcuCtxSetLimit) (CUlimit  limit, size_t  value) =
	(CUresult (*) (CUlimit  limit, size_t  value)) dlsym(cuda_handle, "cuCtxSetLimit");

CUresult (*lcuCtxGetLimit) (size_t * pvalue, CUlimit  limit) =
	(CUresult (*) (size_t * pvalue, CUlimit  limit)) dlsym(cuda_handle, "cuCtxGetLimit");

CUresult (*lcuCtxGetCacheConfig) (CUfunc_cache * pconfig) =
	(CUresult (*) (CUfunc_cache * pconfig)) dlsym(cuda_handle, "cuCtxGetCacheConfig");

CUresult (*lcuCtxSetCacheConfig) (CUfunc_cache  config) =
	(CUresult (*) (CUfunc_cache  config)) dlsym(cuda_handle, "cuCtxSetCacheConfig");

CUresult (*lcuCtxGetSharedMemConfig) (CUsharedconfig * pConfig) =
	(CUresult (*) (CUsharedconfig * pConfig)) dlsym(cuda_handle, "cuCtxGetSharedMemConfig");

CUresult (*lcuCtxSetSharedMemConfig) (CUsharedconfig  config) =
	(CUresult (*) (CUsharedconfig  config)) dlsym(cuda_handle, "cuCtxSetSharedMemConfig");

CUresult (*lcuCtxGetApiVersion) (CUcontext  ctx, unsigned int * version) =
	(CUresult (*) (CUcontext  ctx, unsigned int * version)) dlsym(cuda_handle, "cuCtxGetApiVersion");

CUresult (*lcuCtxGetStreamPriorityRange) (int * leastPriority, int * greatestPriority) =
	(CUresult (*) (int * leastPriority, int * greatestPriority)) dlsym(cuda_handle, "cuCtxGetStreamPriorityRange");

CUresult (*lcuCtxResetPersistingL2Cache) () =
	(CUresult (*) ()) dlsym(cuda_handle, "cuCtxResetPersistingL2Cache");

CUresult (*lcuCtxGetExecAffinity) (CUexecAffinityParam * pExecAffinity, CUexecAffinityType  type) =
	(CUresult (*) (CUexecAffinityParam * pExecAffinity, CUexecAffinityType  type)) dlsym(cuda_handle, "cuCtxGetExecAffinity");

CUresult (*lcuCtxAttach) (CUcontext * pctx, unsigned int  flags) =
	(CUresult (*) (CUcontext * pctx, unsigned int  flags)) dlsym(cuda_handle, "cuCtxAttach");

CUresult (*lcuCtxDetach) (CUcontext  ctx) =
	(CUresult (*) (CUcontext  ctx)) dlsym(cuda_handle, "cuCtxDetach");

CUresult (*lcuModuleLoad) (CUmodule * module, const char * fname) =
	(CUresult (*) (CUmodule * module, const char * fname)) dlsym(cuda_handle, "cuModuleLoad");

CUresult (*lcuModuleLoadData) (CUmodule * module, const void * image) =
	(CUresult (*) (CUmodule * module, const void * image)) dlsym(cuda_handle, "cuModuleLoadData");

CUresult (*lcuModuleLoadDataEx) (CUmodule * module, const void * image, unsigned int  numOptions, CUjit_option * options, void ** optionValues) =
	(CUresult (*) (CUmodule * module, const void * image, unsigned int  numOptions, CUjit_option * options, void ** optionValues)) dlsym(cuda_handle, "cuModuleLoadDataEx");

CUresult (*lcuModuleLoadFatBinary) (CUmodule * module, const void * fatCubin) =
	(CUresult (*) (CUmodule * module, const void * fatCubin)) dlsym(cuda_handle, "cuModuleLoadFatBinary");

CUresult (*lcuModuleUnload) (CUmodule  hmod) =
	(CUresult (*) (CUmodule  hmod)) dlsym(cuda_handle, "cuModuleUnload");

CUresult (*lcuModuleGetLoadingMode) (CUmoduleLoadingMode * mode) =
	(CUresult (*) (CUmoduleLoadingMode * mode)) dlsym(cuda_handle, "cuModuleGetLoadingMode");

CUresult (*lcuModuleGetFunction) (CUfunction * hfunc, CUmodule  hmod, const char * name) =
	(CUresult (*) (CUfunction * hfunc, CUmodule  hmod, const char * name)) dlsym(cuda_handle, "cuModuleGetFunction");

CUresult (*lcuModuleGetGlobal_v2) (CUdeviceptr * dptr, size_t * bytes, CUmodule  hmod, const char * name) =
	(CUresult (*) (CUdeviceptr * dptr, size_t * bytes, CUmodule  hmod, const char * name)) dlsym(cuda_handle, "cuModuleGetGlobal_v2");

CUresult (*lcuLinkCreate_v2) (unsigned int  numOptions, CUjit_option * options, void ** optionValues, CUlinkState * stateOut) =
	(CUresult (*) (unsigned int  numOptions, CUjit_option * options, void ** optionValues, CUlinkState * stateOut)) dlsym(cuda_handle, "cuLinkCreate_v2");

CUresult (*lcuLinkAddData_v2) (CUlinkState  state, CUjitInputType  type, void * data, size_t  size, const char * name, unsigned int  numOptions, CUjit_option * options, void ** optionValues) =
	(CUresult (*) (CUlinkState  state, CUjitInputType  type, void * data, size_t  size, const char * name, unsigned int  numOptions, CUjit_option * options, void ** optionValues)) dlsym(cuda_handle, "cuLinkAddData_v2");

CUresult (*lcuLinkAddFile_v2) (CUlinkState  state, CUjitInputType  type, const char * path, unsigned int  numOptions, CUjit_option * options, void ** optionValues) =
	(CUresult (*) (CUlinkState  state, CUjitInputType  type, const char * path, unsigned int  numOptions, CUjit_option * options, void ** optionValues)) dlsym(cuda_handle, "cuLinkAddFile_v2");

CUresult (*lcuLinkComplete) (CUlinkState  state, void ** cubinOut, size_t * sizeOut) =
	(CUresult (*) (CUlinkState  state, void ** cubinOut, size_t * sizeOut)) dlsym(cuda_handle, "cuLinkComplete");

CUresult (*lcuLinkDestroy) (CUlinkState  state) =
	(CUresult (*) (CUlinkState  state)) dlsym(cuda_handle, "cuLinkDestroy");

CUresult (*lcuModuleGetTexRef) (CUtexref * pTexRef, CUmodule  hmod, const char * name) =
	(CUresult (*) (CUtexref * pTexRef, CUmodule  hmod, const char * name)) dlsym(cuda_handle, "cuModuleGetTexRef");

CUresult (*lcuModuleGetSurfRef) (CUsurfref * pSurfRef, CUmodule  hmod, const char * name) =
	(CUresult (*) (CUsurfref * pSurfRef, CUmodule  hmod, const char * name)) dlsym(cuda_handle, "cuModuleGetSurfRef");

CUresult (*lcuLibraryLoadData) (CUlibrary * library, const void * code, CUjit_option * jitOptions, void ** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void**  libraryOptionValues, unsigned int  numLibraryOptions) =
	(CUresult (*) (CUlibrary * library, const void * code, CUjit_option * jitOptions, void ** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void**  libraryOptionValues, unsigned int  numLibraryOptions)) dlsym(cuda_handle, "cuLibraryLoadData");

CUresult (*lcuLibraryLoadFromFile) (CUlibrary * library, const char * fileName, CUjit_option * jitOptions, void ** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void ** libraryOptionValues, unsigned int  numLibraryOptions) =
	(CUresult (*) (CUlibrary * library, const char * fileName, CUjit_option * jitOptions, void ** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void ** libraryOptionValues, unsigned int  numLibraryOptions)) dlsym(cuda_handle, "cuLibraryLoadFromFile");

CUresult (*lcuLibraryUnload) (CUlibrary  library) =
	(CUresult (*) (CUlibrary  library)) dlsym(cuda_handle, "cuLibraryUnload");

CUresult (*lcuLibraryGetKernel) (CUkernel * pKernel, CUlibrary  library, const char * name) =
	(CUresult (*) (CUkernel * pKernel, CUlibrary  library, const char * name)) dlsym(cuda_handle, "cuLibraryGetKernel");

CUresult (*lcuLibraryGetModule) (CUmodule * pMod, CUlibrary  library) =
	(CUresult (*) (CUmodule * pMod, CUlibrary  library)) dlsym(cuda_handle, "cuLibraryGetModule");

CUresult (*lcuKernelGetFunction) (CUfunction * pFunc, CUkernel  kernel) =
	(CUresult (*) (CUfunction * pFunc, CUkernel  kernel)) dlsym(cuda_handle, "cuKernelGetFunction");

CUresult (*lcuLibraryGetGlobal) (CUdeviceptr * dptr, size_t * bytes, CUlibrary  library, const char * name) =
	(CUresult (*) (CUdeviceptr * dptr, size_t * bytes, CUlibrary  library, const char * name)) dlsym(cuda_handle, "cuLibraryGetGlobal");

CUresult (*lcuLibraryGetManaged) (CUdeviceptr * dptr, size_t * bytes, CUlibrary  library, const char * name) =
	(CUresult (*) (CUdeviceptr * dptr, size_t * bytes, CUlibrary  library, const char * name)) dlsym(cuda_handle, "cuLibraryGetManaged");

CUresult (*lcuLibraryGetUnifiedFunction) (void ** fptr, CUlibrary  library, const char * symbol) =
	(CUresult (*) (void ** fptr, CUlibrary  library, const char * symbol)) dlsym(cuda_handle, "cuLibraryGetUnifiedFunction");

CUresult (*lcuKernelGetAttribute) (int * pi, CUfunction_attribute  attrib, CUkernel  kernel, CUdevice  dev) =
	(CUresult (*) (int * pi, CUfunction_attribute  attrib, CUkernel  kernel, CUdevice  dev)) dlsym(cuda_handle, "cuKernelGetAttribute");

CUresult (*lcuKernelSetAttribute) (CUfunction_attribute  attrib, int  val, CUkernel  kernel, CUdevice  dev) =
	(CUresult (*) (CUfunction_attribute  attrib, int  val, CUkernel  kernel, CUdevice  dev)) dlsym(cuda_handle, "cuKernelSetAttribute");

CUresult (*lcuKernelSetCacheConfig) (CUkernel  kernel, CUfunc_cache  config, CUdevice  dev) =
	(CUresult (*) (CUkernel  kernel, CUfunc_cache  config, CUdevice  dev)) dlsym(cuda_handle, "cuKernelSetCacheConfig");

CUresult (*lcuMemGetInfo_v2) (size_t * free, size_t * total) =
	(CUresult (*) (size_t * free, size_t * total)) dlsym(cuda_handle, "cuMemGetInfo_v2");

CUresult (*lcuMemAlloc_v2) (CUdeviceptr * dptr, size_t  bytesize) =
	(CUresult (*) (CUdeviceptr * dptr, size_t  bytesize)) dlsym(cuda_handle, "cuMemAlloc_v2");

CUresult (*lcuMemAllocPitch_v2) (CUdeviceptr * dptr, size_t * pPitch, size_t  WidthInBytes, size_t  Height, unsigned int  ElementSizeBytes) =
	(CUresult (*) (CUdeviceptr * dptr, size_t * pPitch, size_t  WidthInBytes, size_t  Height, unsigned int  ElementSizeBytes)) dlsym(cuda_handle, "cuMemAllocPitch_v2");

CUresult (*lcuMemFree_v2) (CUdeviceptr  dptr) =
	(CUresult (*) (CUdeviceptr  dptr)) dlsym(cuda_handle, "cuMemFree_v2");

CUresult (*lcuMemGetAddressRange_v2) (CUdeviceptr * pbase, size_t * psize, CUdeviceptr  dptr) =
	(CUresult (*) (CUdeviceptr * pbase, size_t * psize, CUdeviceptr  dptr)) dlsym(cuda_handle, "cuMemGetAddressRange_v2");

CUresult (*lcuMemAllocHost_v2) (void ** pp, size_t  bytesize) =
	(CUresult (*) (void ** pp, size_t  bytesize)) dlsym(cuda_handle, "cuMemAllocHost_v2");

CUresult (*lcuMemFreeHost) (void * p) =
	(CUresult (*) (void * p)) dlsym(cuda_handle, "cuMemFreeHost");

CUresult (*lcuMemHostAlloc) (void ** pp, size_t  bytesize, unsigned int  Flags) =
	(CUresult (*) (void ** pp, size_t  bytesize, unsigned int  Flags)) dlsym(cuda_handle, "cuMemHostAlloc");

CUresult (*lcuMemHostGetDevicePointer_v2) (CUdeviceptr * pdptr, void * p, unsigned int  Flags) =
	(CUresult (*) (CUdeviceptr * pdptr, void * p, unsigned int  Flags)) dlsym(cuda_handle, "cuMemHostGetDevicePointer_v2");

CUresult (*lcuMemHostGetFlags) (unsigned int * pFlags, void * p) =
	(CUresult (*) (unsigned int * pFlags, void * p)) dlsym(cuda_handle, "cuMemHostGetFlags");

CUresult (*lcuMemAllocManaged) (CUdeviceptr * dptr, size_t  bytesize, unsigned int  flags) =
	(CUresult (*) (CUdeviceptr * dptr, size_t  bytesize, unsigned int  flags)) dlsym(cuda_handle, "cuMemAllocManaged");

CUresult (*lcuDeviceGetByPCIBusId) (CUdevice * dev, const char * pciBusId) =
	(CUresult (*) (CUdevice * dev, const char * pciBusId)) dlsym(cuda_handle, "cuDeviceGetByPCIBusId");

CUresult (*lcuDeviceGetPCIBusId) (char * pciBusId, int  len, CUdevice  dev) =
	(CUresult (*) (char * pciBusId, int  len, CUdevice  dev)) dlsym(cuda_handle, "cuDeviceGetPCIBusId");

CUresult (*lcuIpcGetEventHandle) (CUipcEventHandle * pHandle, CUevent  event) =
	(CUresult (*) (CUipcEventHandle * pHandle, CUevent  event)) dlsym(cuda_handle, "cuIpcGetEventHandle");

CUresult (*lcuIpcOpenEventHandle) (CUevent * phEvent, CUipcEventHandle  handle) =
	(CUresult (*) (CUevent * phEvent, CUipcEventHandle  handle)) dlsym(cuda_handle, "cuIpcOpenEventHandle");

CUresult (*lcuIpcGetMemHandle) (CUipcMemHandle * pHandle, CUdeviceptr  dptr) =
	(CUresult (*) (CUipcMemHandle * pHandle, CUdeviceptr  dptr)) dlsym(cuda_handle, "cuIpcGetMemHandle");

CUresult (*lcuIpcOpenMemHandle_v2) (CUdeviceptr * pdptr, CUipcMemHandle  handle, unsigned int  Flags) =
	(CUresult (*) (CUdeviceptr * pdptr, CUipcMemHandle  handle, unsigned int  Flags)) dlsym(cuda_handle, "cuIpcOpenMemHandle_v2");

CUresult (*lcuIpcCloseMemHandle) (CUdeviceptr  dptr) =
	(CUresult (*) (CUdeviceptr  dptr)) dlsym(cuda_handle, "cuIpcCloseMemHandle");

CUresult (*lcuMemHostRegister_v2) (void * p, size_t  bytesize, unsigned int  Flags) =
	(CUresult (*) (void * p, size_t  bytesize, unsigned int  Flags)) dlsym(cuda_handle, "cuMemHostRegister_v2");

CUresult (*lcuMemHostUnregister) (void * p) =
	(CUresult (*) (void * p)) dlsym(cuda_handle, "cuMemHostUnregister");

CUresult (*lcuMemcpy) (CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount) =
	(CUresult (*) (CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpy");

CUresult (*lcuMemcpyPeer) (CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount) =
	(CUresult (*) (CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpyPeer");

CUresult (*lcuMemcpyHtoD_v2) (CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount) =
	(CUresult (*) (CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpyHtoD_v2");

CUresult (*lcuMemcpyDtoH_v2) (void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount) =
	(CUresult (*) (void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpyDtoH_v2");

CUresult (*lcuMemcpyDtoD_v2) (CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount) =
	(CUresult (*) (CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpyDtoD_v2");

CUresult (*lcuMemcpyDtoA_v2) (CUarray  dstArray, size_t  dstOffset, CUdeviceptr  srcDevice, size_t  ByteCount) =
	(CUresult (*) (CUarray  dstArray, size_t  dstOffset, CUdeviceptr  srcDevice, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpyDtoA_v2");

CUresult (*lcuMemcpyAtoD_v2) (CUdeviceptr  dstDevice, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount) =
	(CUresult (*) (CUdeviceptr  dstDevice, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpyAtoD_v2");

CUresult (*lcuMemcpyHtoA_v2) (CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount) =
	(CUresult (*) (CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpyHtoA_v2");

CUresult (*lcuMemcpyAtoH_v2) (void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount) =
	(CUresult (*) (void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpyAtoH_v2");

CUresult (*lcuMemcpyAtoA_v2) (CUarray  dstArray, size_t  dstOffset, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount) =
	(CUresult (*) (CUarray  dstArray, size_t  dstOffset, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)) dlsym(cuda_handle, "cuMemcpyAtoA_v2");

CUresult (*lcuMemcpy2D_v2) (const CUDA_MEMCPY2D * pCopy) =
	(CUresult (*) (const CUDA_MEMCPY2D * pCopy)) dlsym(cuda_handle, "cuMemcpy2D_v2");

CUresult (*lcuMemcpy2DUnaligned_v2) (const CUDA_MEMCPY2D * pCopy) =
	(CUresult (*) (const CUDA_MEMCPY2D * pCopy)) dlsym(cuda_handle, "cuMemcpy2DUnaligned_v2");

CUresult (*lcuMemcpy3D_v2) (const CUDA_MEMCPY3D * pCopy) =
	(CUresult (*) (const CUDA_MEMCPY3D * pCopy)) dlsym(cuda_handle, "cuMemcpy3D_v2");

CUresult (*lcuMemcpy3DPeer) (const CUDA_MEMCPY3D_PEER * pCopy) =
	(CUresult (*) (const CUDA_MEMCPY3D_PEER * pCopy)) dlsym(cuda_handle, "cuMemcpy3DPeer");

CUresult (*lcuMemcpyAsync) (CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpyAsync");

CUresult (*lcuMemcpyPeerAsync) (CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpyPeerAsync");

CUresult (*lcuMemcpyHtoDAsync_v2) (CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpyHtoDAsync_v2");

CUresult (*lcuMemcpyDtoHAsync_v2) (void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream) =
	(CUresult (*) (void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpyDtoHAsync_v2");

CUresult (*lcuMemcpyDtoDAsync_v2) (CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpyDtoDAsync_v2");

CUresult (*lcuMemcpyHtoAAsync_v2) (CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount, CUstream  hStream) =
	(CUresult (*) (CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpyHtoAAsync_v2");

CUresult (*lcuMemcpyAtoHAsync_v2) (void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount, CUstream  hStream) =
	(CUresult (*) (void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpyAtoHAsync_v2");

CUresult (*lcuMemcpy2DAsync_v2) (const CUDA_MEMCPY2D * pCopy, CUstream  hStream) =
	(CUresult (*) (const CUDA_MEMCPY2D * pCopy, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpy2DAsync_v2");

CUresult (*lcuMemcpy3DAsync_v2) (const CUDA_MEMCPY3D * pCopy, CUstream  hStream) =
	(CUresult (*) (const CUDA_MEMCPY3D * pCopy, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpy3DAsync_v2");

CUresult (*lcuMemcpy3DPeerAsync) (const CUDA_MEMCPY3D_PEER * pCopy, CUstream  hStream) =
	(CUresult (*) (const CUDA_MEMCPY3D_PEER * pCopy, CUstream  hStream)) dlsym(cuda_handle, "cuMemcpy3DPeerAsync");

CUresult (*lcuMemsetD8_v2) (CUdeviceptr  dstDevice, unsigned char  uc, size_t  N) =
	(CUresult (*) (CUdeviceptr  dstDevice, unsigned char  uc, size_t  N)) dlsym(cuda_handle, "cuMemsetD8_v2");

CUresult (*lcuMemsetD16_v2) (CUdeviceptr  dstDevice, unsigned short  us, size_t  N) =
	(CUresult (*) (CUdeviceptr  dstDevice, unsigned short  us, size_t  N)) dlsym(cuda_handle, "cuMemsetD16_v2");

CUresult (*lcuMemsetD32_v2) (CUdeviceptr  dstDevice, unsigned int  ui, size_t  N) =
	(CUresult (*) (CUdeviceptr  dstDevice, unsigned int  ui, size_t  N)) dlsym(cuda_handle, "cuMemsetD32_v2");

CUresult (*lcuMemsetD2D8_v2) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height) =
	(CUresult (*) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height)) dlsym(cuda_handle, "cuMemsetD2D8_v2");

CUresult (*lcuMemsetD2D16_v2) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height) =
	(CUresult (*) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height)) dlsym(cuda_handle, "cuMemsetD2D16_v2");

CUresult (*lcuMemsetD2D32_v2) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height) =
	(CUresult (*) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height)) dlsym(cuda_handle, "cuMemsetD2D32_v2");

CUresult (*lcuMemsetD8Async) (CUdeviceptr  dstDevice, unsigned char  uc, size_t  N, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dstDevice, unsigned char  uc, size_t  N, CUstream  hStream)) dlsym(cuda_handle, "cuMemsetD8Async");

CUresult (*lcuMemsetD16Async) (CUdeviceptr  dstDevice, unsigned short  us, size_t  N, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dstDevice, unsigned short  us, size_t  N, CUstream  hStream)) dlsym(cuda_handle, "cuMemsetD16Async");

CUresult (*lcuMemsetD32Async) (CUdeviceptr  dstDevice, unsigned int  ui, size_t  N, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dstDevice, unsigned int  ui, size_t  N, CUstream  hStream)) dlsym(cuda_handle, "cuMemsetD32Async");

CUresult (*lcuMemsetD2D8Async) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height, CUstream  hStream)) dlsym(cuda_handle, "cuMemsetD2D8Async");

CUresult (*lcuMemsetD2D16Async) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height, CUstream  hStream)) dlsym(cuda_handle, "cuMemsetD2D16Async");

CUresult (*lcuMemsetD2D32Async) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height, CUstream  hStream)) dlsym(cuda_handle, "cuMemsetD2D32Async");

CUresult (*lcuArrayCreate_v2) (CUarray * pHandle, const CUDA_ARRAY_DESCRIPTOR * pAllocateArray) =
	(CUresult (*) (CUarray * pHandle, const CUDA_ARRAY_DESCRIPTOR * pAllocateArray)) dlsym(cuda_handle, "cuArrayCreate_v2");

CUresult (*lcuArrayGetDescriptor_v2) (CUDA_ARRAY_DESCRIPTOR * pArrayDescriptor, CUarray  hArray) =
	(CUresult (*) (CUDA_ARRAY_DESCRIPTOR * pArrayDescriptor, CUarray  hArray)) dlsym(cuda_handle, "cuArrayGetDescriptor_v2");

CUresult (*lcuArrayGetSparseProperties) (CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUarray  array) =
	(CUresult (*) (CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUarray  array)) dlsym(cuda_handle, "cuArrayGetSparseProperties");

CUresult (*lcuMipmappedArrayGetSparseProperties) (CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUmipmappedArray  mipmap) =
	(CUresult (*) (CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUmipmappedArray  mipmap)) dlsym(cuda_handle, "cuMipmappedArrayGetSparseProperties");

CUresult (*lcuArrayGetMemoryRequirements) (CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUarray  array, CUdevice  device) =
	(CUresult (*) (CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUarray  array, CUdevice  device)) dlsym(cuda_handle, "cuArrayGetMemoryRequirements");

CUresult (*lcuMipmappedArrayGetMemoryRequirements) (CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUmipmappedArray  mipmap, CUdevice  device) =
	(CUresult (*) (CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUmipmappedArray  mipmap, CUdevice  device)) dlsym(cuda_handle, "cuMipmappedArrayGetMemoryRequirements");

CUresult (*lcuArrayGetPlane) (CUarray * pPlaneArray, CUarray  hArray, unsigned int  planeIdx) =
	(CUresult (*) (CUarray * pPlaneArray, CUarray  hArray, unsigned int  planeIdx)) dlsym(cuda_handle, "cuArrayGetPlane");

CUresult (*lcuArrayDestroy) (CUarray  hArray) =
	(CUresult (*) (CUarray  hArray)) dlsym(cuda_handle, "cuArrayDestroy");

CUresult (*lcuArray3DCreate_v2) (CUarray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pAllocateArray) =
	(CUresult (*) (CUarray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pAllocateArray)) dlsym(cuda_handle, "cuArray3DCreate_v2");

CUresult (*lcuArray3DGetDescriptor_v2) (CUDA_ARRAY3D_DESCRIPTOR * pArrayDescriptor, CUarray  hArray) =
	(CUresult (*) (CUDA_ARRAY3D_DESCRIPTOR * pArrayDescriptor, CUarray  hArray)) dlsym(cuda_handle, "cuArray3DGetDescriptor_v2");

CUresult (*lcuMipmappedArrayCreate) (CUmipmappedArray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int  numMipmapLevels) =
	(CUresult (*) (CUmipmappedArray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int  numMipmapLevels)) dlsym(cuda_handle, "cuMipmappedArrayCreate");

CUresult (*lcuMipmappedArrayGetLevel) (CUarray * pLevelArray, CUmipmappedArray  hMipmappedArray, unsigned int  level) =
	(CUresult (*) (CUarray * pLevelArray, CUmipmappedArray  hMipmappedArray, unsigned int  level)) dlsym(cuda_handle, "cuMipmappedArrayGetLevel");

CUresult (*lcuMipmappedArrayDestroy) (CUmipmappedArray  hMipmappedArray) =
	(CUresult (*) (CUmipmappedArray  hMipmappedArray)) dlsym(cuda_handle, "cuMipmappedArrayDestroy");

CUresult (*lcuMemGetHandleForAddressRange) (void * handle, CUdeviceptr  dptr, size_t  size, CUmemRangeHandleType  handleType, unsigned long long  flags) =
	(CUresult (*) (void * handle, CUdeviceptr  dptr, size_t  size, CUmemRangeHandleType  handleType, unsigned long long  flags)) dlsym(cuda_handle, "cuMemGetHandleForAddressRange");

CUresult (*lcuMemAddressReserve) (CUdeviceptr * ptr, size_t  size, size_t  alignment, CUdeviceptr  addr, unsigned long long  flags) =
	(CUresult (*) (CUdeviceptr * ptr, size_t  size, size_t  alignment, CUdeviceptr  addr, unsigned long long  flags)) dlsym(cuda_handle, "cuMemAddressReserve");

CUresult (*lcuMemAddressFree) (CUdeviceptr  ptr, size_t  size) =
	(CUresult (*) (CUdeviceptr  ptr, size_t  size)) dlsym(cuda_handle, "cuMemAddressFree");

CUresult (*lcuMemCreate) (CUmemGenericAllocationHandle * handle, size_t  size, const CUmemAllocationProp * prop, unsigned long long  flags) =
	(CUresult (*) (CUmemGenericAllocationHandle * handle, size_t  size, const CUmemAllocationProp * prop, unsigned long long  flags)) dlsym(cuda_handle, "cuMemCreate");

CUresult (*lcuMemRelease) (CUmemGenericAllocationHandle  handle) =
	(CUresult (*) (CUmemGenericAllocationHandle  handle)) dlsym(cuda_handle, "cuMemRelease");

CUresult (*lcuMemMap) (CUdeviceptr  ptr, size_t  size, size_t  offset, CUmemGenericAllocationHandle  handle, unsigned long long  flags) =
	(CUresult (*) (CUdeviceptr  ptr, size_t  size, size_t  offset, CUmemGenericAllocationHandle  handle, unsigned long long  flags)) dlsym(cuda_handle, "cuMemMap");

CUresult (*lcuMemMapArrayAsync) (CUarrayMapInfo * mapInfoList, unsigned int  count, CUstream  hStream) =
	(CUresult (*) (CUarrayMapInfo * mapInfoList, unsigned int  count, CUstream  hStream)) dlsym(cuda_handle, "cuMemMapArrayAsync");

CUresult (*lcuMemUnmap) (CUdeviceptr  ptr, size_t  size) =
	(CUresult (*) (CUdeviceptr  ptr, size_t  size)) dlsym(cuda_handle, "cuMemUnmap");

CUresult (*lcuMemSetAccess) (CUdeviceptr  ptr, size_t  size, const CUmemAccessDesc * desc, size_t  count) =
	(CUresult (*) (CUdeviceptr  ptr, size_t  size, const CUmemAccessDesc * desc, size_t  count)) dlsym(cuda_handle, "cuMemSetAccess");

CUresult (*lcuMemGetAccess) (unsigned long long * flags, const CUmemLocation * location, CUdeviceptr  ptr) =
	(CUresult (*) (unsigned long long * flags, const CUmemLocation * location, CUdeviceptr  ptr)) dlsym(cuda_handle, "cuMemGetAccess");

CUresult (*lcuMemExportToShareableHandle) (void * shareableHandle, CUmemGenericAllocationHandle  handle, CUmemAllocationHandleType  handleType, unsigned long long  flags) =
	(CUresult (*) (void * shareableHandle, CUmemGenericAllocationHandle  handle, CUmemAllocationHandleType  handleType, unsigned long long  flags)) dlsym(cuda_handle, "cuMemExportToShareableHandle");

CUresult (*lcuMemImportFromShareableHandle) (CUmemGenericAllocationHandle * handle, void * osHandle, CUmemAllocationHandleType  shHandleType) =
	(CUresult (*) (CUmemGenericAllocationHandle * handle, void * osHandle, CUmemAllocationHandleType  shHandleType)) dlsym(cuda_handle, "cuMemImportFromShareableHandle");

CUresult (*lcuMemGetAllocationGranularity) (size_t * granularity, const CUmemAllocationProp * prop, CUmemAllocationGranularity_flags  option) =
	(CUresult (*) (size_t * granularity, const CUmemAllocationProp * prop, CUmemAllocationGranularity_flags  option)) dlsym(cuda_handle, "cuMemGetAllocationGranularity");

CUresult (*lcuMemGetAllocationPropertiesFromHandle) (CUmemAllocationProp * prop, CUmemGenericAllocationHandle  handle) =
	(CUresult (*) (CUmemAllocationProp * prop, CUmemGenericAllocationHandle  handle)) dlsym(cuda_handle, "cuMemGetAllocationPropertiesFromHandle");

CUresult (*lcuMemRetainAllocationHandle) (CUmemGenericAllocationHandle * handle, void * addr) =
	(CUresult (*) (CUmemGenericAllocationHandle * handle, void * addr)) dlsym(cuda_handle, "cuMemRetainAllocationHandle");

CUresult (*lcuMemFreeAsync) (CUdeviceptr  dptr, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  dptr, CUstream  hStream)) dlsym(cuda_handle, "cuMemFreeAsync");

CUresult (*lcuMemAllocAsync) (CUdeviceptr * dptr, size_t  bytesize, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr * dptr, size_t  bytesize, CUstream  hStream)) dlsym(cuda_handle, "cuMemAllocAsync");

CUresult (*lcuMemPoolTrimTo) (CUmemoryPool  pool, size_t  minBytesToKeep) =
	(CUresult (*) (CUmemoryPool  pool, size_t  minBytesToKeep)) dlsym(cuda_handle, "cuMemPoolTrimTo");

CUresult (*lcuMemPoolSetAttribute) (CUmemoryPool  pool, CUmemPool_attribute  attr, void * value) =
	(CUresult (*) (CUmemoryPool  pool, CUmemPool_attribute  attr, void * value)) dlsym(cuda_handle, "cuMemPoolSetAttribute");

CUresult (*lcuMemPoolGetAttribute) (CUmemoryPool  pool, CUmemPool_attribute  attr, void * value) =
	(CUresult (*) (CUmemoryPool  pool, CUmemPool_attribute  attr, void * value)) dlsym(cuda_handle, "cuMemPoolGetAttribute");

CUresult (*lcuMemPoolSetAccess) (CUmemoryPool  pool, const CUmemAccessDesc * map, size_t  count) =
	(CUresult (*) (CUmemoryPool  pool, const CUmemAccessDesc * map, size_t  count)) dlsym(cuda_handle, "cuMemPoolSetAccess");

CUresult (*lcuMemPoolGetAccess) (CUmemAccess_flags * flags, CUmemoryPool  memPool, CUmemLocation * location) =
	(CUresult (*) (CUmemAccess_flags * flags, CUmemoryPool  memPool, CUmemLocation * location)) dlsym(cuda_handle, "cuMemPoolGetAccess");

CUresult (*lcuMemPoolCreate) (CUmemoryPool * pool, const CUmemPoolProps * poolProps) =
	(CUresult (*) (CUmemoryPool * pool, const CUmemPoolProps * poolProps)) dlsym(cuda_handle, "cuMemPoolCreate");

CUresult (*lcuMemPoolDestroy) (CUmemoryPool  pool) =
	(CUresult (*) (CUmemoryPool  pool)) dlsym(cuda_handle, "cuMemPoolDestroy");

CUresult (*lcuMemAllocFromPoolAsync) (CUdeviceptr * dptr, size_t  bytesize, CUmemoryPool  pool, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr * dptr, size_t  bytesize, CUmemoryPool  pool, CUstream  hStream)) dlsym(cuda_handle, "cuMemAllocFromPoolAsync");

CUresult (*lcuMemPoolExportToShareableHandle) (void * handle_out, CUmemoryPool  pool, CUmemAllocationHandleType  handleType, unsigned long long  flags) =
	(CUresult (*) (void * handle_out, CUmemoryPool  pool, CUmemAllocationHandleType  handleType, unsigned long long  flags)) dlsym(cuda_handle, "cuMemPoolExportToShareableHandle");

CUresult (*lcuMemPoolImportFromShareableHandle) (CUmemoryPool * pool_out, void * handle, CUmemAllocationHandleType  handleType, unsigned long long  flags) =
	(CUresult (*) (CUmemoryPool * pool_out, void * handle, CUmemAllocationHandleType  handleType, unsigned long long  flags)) dlsym(cuda_handle, "cuMemPoolImportFromShareableHandle");

CUresult (*lcuMemPoolExportPointer) (CUmemPoolPtrExportData * shareData_out, CUdeviceptr  ptr) =
	(CUresult (*) (CUmemPoolPtrExportData * shareData_out, CUdeviceptr  ptr)) dlsym(cuda_handle, "cuMemPoolExportPointer");

CUresult (*lcuMemPoolImportPointer) (CUdeviceptr * ptr_out, CUmemoryPool  pool, CUmemPoolPtrExportData * shareData) =
	(CUresult (*) (CUdeviceptr * ptr_out, CUmemoryPool  pool, CUmemPoolPtrExportData * shareData)) dlsym(cuda_handle, "cuMemPoolImportPointer");

CUresult (*lcuMulticastCreate) (CUmemGenericAllocationHandle * mcHandle, const CUmulticastObjectProp * prop) =
	(CUresult (*) (CUmemGenericAllocationHandle * mcHandle, const CUmulticastObjectProp * prop)) dlsym(cuda_handle, "cuMulticastCreate");

CUresult (*lcuMulticastAddDevice) (CUmemGenericAllocationHandle  mcHandle, CUdevice  dev) =
	(CUresult (*) (CUmemGenericAllocationHandle  mcHandle, CUdevice  dev)) dlsym(cuda_handle, "cuMulticastAddDevice");

CUresult (*lcuMulticastBindMem) (CUmemGenericAllocationHandle  mcHandle, size_t  mcOffset, CUmemGenericAllocationHandle  memHandle, size_t  memOffset, size_t  size, unsigned long long  flags) =
	(CUresult (*) (CUmemGenericAllocationHandle  mcHandle, size_t  mcOffset, CUmemGenericAllocationHandle  memHandle, size_t  memOffset, size_t  size, unsigned long long  flags)) dlsym(cuda_handle, "cuMulticastBindMem");

CUresult (*lcuMulticastBindAddr) (CUmemGenericAllocationHandle  mcHandle, size_t  mcOffset, CUdeviceptr  memptr, size_t  size, unsigned long long  flags) =
	(CUresult (*) (CUmemGenericAllocationHandle  mcHandle, size_t  mcOffset, CUdeviceptr  memptr, size_t  size, unsigned long long  flags)) dlsym(cuda_handle, "cuMulticastBindAddr");

CUresult (*lcuMulticastUnbind) (CUmemGenericAllocationHandle  mcHandle, CUdevice  dev, size_t  mcOffset, size_t  size) =
	(CUresult (*) (CUmemGenericAllocationHandle  mcHandle, CUdevice  dev, size_t  mcOffset, size_t  size)) dlsym(cuda_handle, "cuMulticastUnbind");

CUresult (*lcuMulticastGetGranularity) (size_t * granularity, const CUmulticastObjectProp * prop, CUmulticastGranularity_flags  option) =
	(CUresult (*) (size_t * granularity, const CUmulticastObjectProp * prop, CUmulticastGranularity_flags  option)) dlsym(cuda_handle, "cuMulticastGetGranularity");

CUresult (*lcuPointerGetAttribute) (void * data, CUpointer_attribute  attribute, CUdeviceptr  ptr) =
	(CUresult (*) (void * data, CUpointer_attribute  attribute, CUdeviceptr  ptr)) dlsym(cuda_handle, "cuPointerGetAttribute");

CUresult (*lcuMemPrefetchAsync) (CUdeviceptr  devPtr, size_t  count, CUdevice  dstDevice, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  devPtr, size_t  count, CUdevice  dstDevice, CUstream  hStream)) dlsym(cuda_handle, "cuMemPrefetchAsync");

CUresult (*lcuMemPrefetchAsync_v2) (CUdeviceptr  devPtr, size_t  count, CUmemLocation  location, unsigned int  flags, CUstream  hStream) =
	(CUresult (*) (CUdeviceptr  devPtr, size_t  count, CUmemLocation  location, unsigned int  flags, CUstream  hStream)) dlsym(cuda_handle, "cuMemPrefetchAsync_v2");

CUresult (*lcuMemAdvise) (CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUdevice  device) =
	(CUresult (*) (CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUdevice  device)) dlsym(cuda_handle, "cuMemAdvise");

CUresult (*lcuMemAdvise_v2) (CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUmemLocation  location) =
	(CUresult (*) (CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUmemLocation  location)) dlsym(cuda_handle, "cuMemAdvise_v2");

CUresult (*lcuMemRangeGetAttribute) (void * data, size_t  dataSize, CUmem_range_attribute  attribute, CUdeviceptr  devPtr, size_t  count) =
	(CUresult (*) (void * data, size_t  dataSize, CUmem_range_attribute  attribute, CUdeviceptr  devPtr, size_t  count)) dlsym(cuda_handle, "cuMemRangeGetAttribute");

CUresult (*lcuMemRangeGetAttributes) (void ** data, size_t * dataSizes, CUmem_range_attribute * attributes, size_t  numAttributes, CUdeviceptr  devPtr, size_t  count) =
	(CUresult (*) (void ** data, size_t * dataSizes, CUmem_range_attribute * attributes, size_t  numAttributes, CUdeviceptr  devPtr, size_t  count)) dlsym(cuda_handle, "cuMemRangeGetAttributes");

CUresult (*lcuPointerSetAttribute) (const void * value, CUpointer_attribute  attribute, CUdeviceptr  ptr) =
	(CUresult (*) (const void * value, CUpointer_attribute  attribute, CUdeviceptr  ptr)) dlsym(cuda_handle, "cuPointerSetAttribute");

CUresult (*lcuPointerGetAttributes) (unsigned int  numAttributes, CUpointer_attribute * attributes, void ** data, CUdeviceptr  ptr) =
	(CUresult (*) (unsigned int  numAttributes, CUpointer_attribute * attributes, void ** data, CUdeviceptr  ptr)) dlsym(cuda_handle, "cuPointerGetAttributes");

CUresult (*lcuStreamCreate) (CUstream * phStream, unsigned int  Flags) =
	(CUresult (*) (CUstream * phStream, unsigned int  Flags)) dlsym(cuda_handle, "cuStreamCreate");

CUresult (*lcuStreamCreateWithPriority) (CUstream * phStream, unsigned int  flags, int  priority) =
	(CUresult (*) (CUstream * phStream, unsigned int  flags, int  priority)) dlsym(cuda_handle, "cuStreamCreateWithPriority");

CUresult (*lcuStreamGetPriority) (CUstream  hStream, int * priority) =
	(CUresult (*) (CUstream  hStream, int * priority)) dlsym(cuda_handle, "cuStreamGetPriority");

CUresult (*lcuStreamGetFlags) (CUstream  hStream, unsigned int * flags) =
	(CUresult (*) (CUstream  hStream, unsigned int * flags)) dlsym(cuda_handle, "cuStreamGetFlags");

CUresult (*lcuStreamGetId) (CUstream  hStream, unsigned long long * streamId) =
	(CUresult (*) (CUstream  hStream, unsigned long long * streamId)) dlsym(cuda_handle, "cuStreamGetId");

CUresult (*lcuStreamGetCtx) (CUstream  hStream, CUcontext * pctx) =
	(CUresult (*) (CUstream  hStream, CUcontext * pctx)) dlsym(cuda_handle, "cuStreamGetCtx");

CUresult (*lcuStreamWaitEvent) (CUstream  hStream, CUevent  hEvent, unsigned int  Flags) =
	(CUresult (*) (CUstream  hStream, CUevent  hEvent, unsigned int  Flags)) dlsym(cuda_handle, "cuStreamWaitEvent");

CUresult (*lcuStreamAddCallback) (CUstream  hStream, CUstreamCallback  callback, void * userData, unsigned int  flags) =
	(CUresult (*) (CUstream  hStream, CUstreamCallback  callback, void * userData, unsigned int  flags)) dlsym(cuda_handle, "cuStreamAddCallback");

CUresult (*lcuStreamBeginCapture_v2) (CUstream  hStream, CUstreamCaptureMode  mode) =
	(CUresult (*) (CUstream  hStream, CUstreamCaptureMode  mode)) dlsym(cuda_handle, "cuStreamBeginCapture_v2");

CUresult (*lcuThreadExchangeStreamCaptureMode) (CUstreamCaptureMode * mode) =
	(CUresult (*) (CUstreamCaptureMode * mode)) dlsym(cuda_handle, "cuThreadExchangeStreamCaptureMode");

CUresult (*lcuStreamEndCapture) (CUstream  hStream, CUgraph * phGraph) =
	(CUresult (*) (CUstream  hStream, CUgraph * phGraph)) dlsym(cuda_handle, "cuStreamEndCapture");

CUresult (*lcuStreamIsCapturing) (CUstream  hStream, CUstreamCaptureStatus * captureStatus) =
	(CUresult (*) (CUstream  hStream, CUstreamCaptureStatus * captureStatus)) dlsym(cuda_handle, "cuStreamIsCapturing");

CUresult (*lcuStreamGetCaptureInfo_v2) (CUstream  hStream, CUstreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, CUgraph * graph_out, const CUgraphNode ** dependencies_out, size_t * numDependencies_out) =
	(CUresult (*) (CUstream  hStream, CUstreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, CUgraph * graph_out, const CUgraphNode ** dependencies_out, size_t * numDependencies_out)) dlsym(cuda_handle, "cuStreamGetCaptureInfo_v2");

CUresult (*lcuStreamUpdateCaptureDependencies) (CUstream  hStream, CUgraphNode * dependencies, size_t  numDependencies, unsigned int  flags) =
	(CUresult (*) (CUstream  hStream, CUgraphNode * dependencies, size_t  numDependencies, unsigned int  flags)) dlsym(cuda_handle, "cuStreamUpdateCaptureDependencies");

CUresult (*lcuStreamAttachMemAsync) (CUstream  hStream, CUdeviceptr  dptr, size_t  length, unsigned int  flags) =
	(CUresult (*) (CUstream  hStream, CUdeviceptr  dptr, size_t  length, unsigned int  flags)) dlsym(cuda_handle, "cuStreamAttachMemAsync");

CUresult (*lcuStreamQuery) (CUstream  hStream) =
	(CUresult (*) (CUstream  hStream)) dlsym(cuda_handle, "cuStreamQuery");

CUresult (*lcuStreamSynchronize) (CUstream  hStream) =
	(CUresult (*) (CUstream  hStream)) dlsym(cuda_handle, "cuStreamSynchronize");

CUresult (*lcuStreamDestroy_v2) (CUstream  hStream) =
	(CUresult (*) (CUstream  hStream)) dlsym(cuda_handle, "cuStreamDestroy_v2");

CUresult (*lcuStreamCopyAttributes) (CUstream  dst, CUstream  src) =
	(CUresult (*) (CUstream  dst, CUstream  src)) dlsym(cuda_handle, "cuStreamCopyAttributes");

CUresult (*lcuStreamGetAttribute) (CUstream  hStream, CUstreamAttrID  attr, CUstreamAttrValue * value_out) =
	(CUresult (*) (CUstream  hStream, CUstreamAttrID  attr, CUstreamAttrValue * value_out)) dlsym(cuda_handle, "cuStreamGetAttribute");

CUresult (*lcuStreamSetAttribute) (CUstream  hStream, CUstreamAttrID  attr, const CUstreamAttrValue * value) =
	(CUresult (*) (CUstream  hStream, CUstreamAttrID  attr, const CUstreamAttrValue * value)) dlsym(cuda_handle, "cuStreamSetAttribute");

CUresult (*lcuEventCreate) (CUevent * phEvent, unsigned int  Flags) =
	(CUresult (*) (CUevent * phEvent, unsigned int  Flags)) dlsym(cuda_handle, "cuEventCreate");

CUresult (*lcuEventRecord) (CUevent  hEvent, CUstream  hStream) =
	(CUresult (*) (CUevent  hEvent, CUstream  hStream)) dlsym(cuda_handle, "cuEventRecord");

CUresult (*lcuEventRecordWithFlags) (CUevent  hEvent, CUstream  hStream, unsigned int  flags) =
	(CUresult (*) (CUevent  hEvent, CUstream  hStream, unsigned int  flags)) dlsym(cuda_handle, "cuEventRecordWithFlags");

CUresult (*lcuEventQuery) (CUevent  hEvent) =
	(CUresult (*) (CUevent  hEvent)) dlsym(cuda_handle, "cuEventQuery");

CUresult (*lcuEventSynchronize) (CUevent  hEvent) =
	(CUresult (*) (CUevent  hEvent)) dlsym(cuda_handle, "cuEventSynchronize");

CUresult (*lcuEventDestroy_v2) (CUevent  hEvent) =
	(CUresult (*) (CUevent  hEvent)) dlsym(cuda_handle, "cuEventDestroy_v2");

CUresult (*lcuEventElapsedTime) (float * pMilliseconds, CUevent  hStart, CUevent  hEnd) =
	(CUresult (*) (float * pMilliseconds, CUevent  hStart, CUevent  hEnd)) dlsym(cuda_handle, "cuEventElapsedTime");

CUresult (*lcuImportExternalMemory) (CUexternalMemory * extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC * memHandleDesc) =
	(CUresult (*) (CUexternalMemory * extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC * memHandleDesc)) dlsym(cuda_handle, "cuImportExternalMemory");

CUresult (*lcuExternalMemoryGetMappedBuffer) (CUdeviceptr * devPtr, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC * bufferDesc) =
	(CUresult (*) (CUdeviceptr * devPtr, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC * bufferDesc)) dlsym(cuda_handle, "cuExternalMemoryGetMappedBuffer");

CUresult (*lcuExternalMemoryGetMappedMipmappedArray) (CUmipmappedArray * mipmap, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC * mipmapDesc) =
	(CUresult (*) (CUmipmappedArray * mipmap, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC * mipmapDesc)) dlsym(cuda_handle, "cuExternalMemoryGetMappedMipmappedArray");

CUresult (*lcuDestroyExternalMemory) (CUexternalMemory  extMem) =
	(CUresult (*) (CUexternalMemory  extMem)) dlsym(cuda_handle, "cuDestroyExternalMemory");

CUresult (*lcuImportExternalSemaphore) (CUexternalSemaphore * extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC * semHandleDesc) =
	(CUresult (*) (CUexternalSemaphore * extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC * semHandleDesc)) dlsym(cuda_handle, "cuImportExternalSemaphore");

CUresult (*lcuSignalExternalSemaphoresAsync) (const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream) =
	(CUresult (*) (const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream)) dlsym(cuda_handle, "cuSignalExternalSemaphoresAsync");

CUresult (*lcuWaitExternalSemaphoresAsync) (const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream) =
	(CUresult (*) (const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream)) dlsym(cuda_handle, "cuWaitExternalSemaphoresAsync");

CUresult (*lcuDestroyExternalSemaphore) (CUexternalSemaphore  extSem) =
	(CUresult (*) (CUexternalSemaphore  extSem)) dlsym(cuda_handle, "cuDestroyExternalSemaphore");

CUresult (*lcuStreamWaitValue32_v2) (CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags) =
	(CUresult (*) (CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)) dlsym(cuda_handle, "cuStreamWaitValue32_v2");

CUresult (*lcuStreamWaitValue64_v2) (CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags) =
	(CUresult (*) (CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)) dlsym(cuda_handle, "cuStreamWaitValue64_v2");

CUresult (*lcuStreamWriteValue32_v2) (CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags) =
	(CUresult (*) (CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)) dlsym(cuda_handle, "cuStreamWriteValue32_v2");

CUresult (*lcuStreamWriteValue64_v2) (CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags) =
	(CUresult (*) (CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)) dlsym(cuda_handle, "cuStreamWriteValue64_v2");

CUresult (*lcuStreamBatchMemOp_v2) (CUstream  stream, unsigned int  count, CUstreamBatchMemOpParams * paramArray, unsigned int  flags) =
	(CUresult (*) (CUstream  stream, unsigned int  count, CUstreamBatchMemOpParams * paramArray, unsigned int  flags)) dlsym(cuda_handle, "cuStreamBatchMemOp_v2");

CUresult (*lcuFuncGetAttribute) (int * pi, CUfunction_attribute  attrib, CUfunction  hfunc) =
	(CUresult (*) (int * pi, CUfunction_attribute  attrib, CUfunction  hfunc)) dlsym(cuda_handle, "cuFuncGetAttribute");

CUresult (*lcuFuncSetAttribute) (CUfunction  hfunc, CUfunction_attribute  attrib, int  value) =
	(CUresult (*) (CUfunction  hfunc, CUfunction_attribute  attrib, int  value)) dlsym(cuda_handle, "cuFuncSetAttribute");

CUresult (*lcuFuncSetCacheConfig) (CUfunction  hfunc, CUfunc_cache  config) =
	(CUresult (*) (CUfunction  hfunc, CUfunc_cache  config)) dlsym(cuda_handle, "cuFuncSetCacheConfig");

CUresult (*lcuFuncSetSharedMemConfig) (CUfunction  hfunc, CUsharedconfig  config) =
	(CUresult (*) (CUfunction  hfunc, CUsharedconfig  config)) dlsym(cuda_handle, "cuFuncSetSharedMemConfig");

CUresult (*lcuFuncGetModule) (CUmodule * hmod, CUfunction  hfunc) =
	(CUresult (*) (CUmodule * hmod, CUfunction  hfunc)) dlsym(cuda_handle, "cuFuncGetModule");

CUresult (*lcuLaunchKernel) (CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams, void ** extra) =
	(CUresult (*) (CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams, void ** extra)) dlsym(cuda_handle, "cuLaunchKernel");

CUresult (*lcuLaunchKernelEx) (const CUlaunchConfig * config, CUfunction  f, void ** kernelParams, void ** extra) =
	(CUresult (*) (const CUlaunchConfig * config, CUfunction  f, void ** kernelParams, void ** extra)) dlsym(cuda_handle, "cuLaunchKernelEx");

CUresult (*lcuLaunchCooperativeKernel) (CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams) =
	(CUresult (*) (CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams)) dlsym(cuda_handle, "cuLaunchCooperativeKernel");

CUresult (*lcuLaunchCooperativeKernelMultiDevice) (CUDA_LAUNCH_PARAMS * launchParamsList, unsigned int  numDevices, unsigned int  flags) =
	(CUresult (*) (CUDA_LAUNCH_PARAMS * launchParamsList, unsigned int  numDevices, unsigned int  flags)) dlsym(cuda_handle, "cuLaunchCooperativeKernelMultiDevice");

CUresult (*lcuLaunchHostFunc) (CUstream  hStream, CUhostFn  fn, void * userData) =
	(CUresult (*) (CUstream  hStream, CUhostFn  fn, void * userData)) dlsym(cuda_handle, "cuLaunchHostFunc");

CUresult (*lcuFuncSetBlockShape) (CUfunction  hfunc, int  x, int  y, int  z) =
	(CUresult (*) (CUfunction  hfunc, int  x, int  y, int  z)) dlsym(cuda_handle, "cuFuncSetBlockShape");

CUresult (*lcuFuncSetSharedSize) (CUfunction  hfunc, unsigned int  bytes) =
	(CUresult (*) (CUfunction  hfunc, unsigned int  bytes)) dlsym(cuda_handle, "cuFuncSetSharedSize");

CUresult (*lcuParamSetSize) (CUfunction  hfunc, unsigned int  numbytes) =
	(CUresult (*) (CUfunction  hfunc, unsigned int  numbytes)) dlsym(cuda_handle, "cuParamSetSize");

CUresult (*lcuParamSeti) (CUfunction  hfunc, int  offset, unsigned int  value) =
	(CUresult (*) (CUfunction  hfunc, int  offset, unsigned int  value)) dlsym(cuda_handle, "cuParamSeti");

CUresult (*lcuParamSetf) (CUfunction  hfunc, int  offset, float  value) =
	(CUresult (*) (CUfunction  hfunc, int  offset, float  value)) dlsym(cuda_handle, "cuParamSetf");

CUresult (*lcuParamSetv) (CUfunction  hfunc, int  offset, void * ptr, unsigned int  numbytes) =
	(CUresult (*) (CUfunction  hfunc, int  offset, void * ptr, unsigned int  numbytes)) dlsym(cuda_handle, "cuParamSetv");

CUresult (*lcuLaunch) (CUfunction  f) =
	(CUresult (*) (CUfunction  f)) dlsym(cuda_handle, "cuLaunch");

CUresult (*lcuLaunchGrid) (CUfunction  f, int  grid_width, int  grid_height) =
	(CUresult (*) (CUfunction  f, int  grid_width, int  grid_height)) dlsym(cuda_handle, "cuLaunchGrid");

CUresult (*lcuLaunchGridAsync) (CUfunction  f, int  grid_width, int  grid_height, CUstream  hStream) =
	(CUresult (*) (CUfunction  f, int  grid_width, int  grid_height, CUstream  hStream)) dlsym(cuda_handle, "cuLaunchGridAsync");

CUresult (*lcuParamSetTexRef) (CUfunction  hfunc, int  texunit, CUtexref  hTexRef) =
	(CUresult (*) (CUfunction  hfunc, int  texunit, CUtexref  hTexRef)) dlsym(cuda_handle, "cuParamSetTexRef");

CUresult (*lcuGraphCreate) (CUgraph * phGraph, unsigned int  flags) =
	(CUresult (*) (CUgraph * phGraph, unsigned int  flags)) dlsym(cuda_handle, "cuGraphCreate");

CUresult (*lcuGraphAddKernelNode_v2) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_KERNEL_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_KERNEL_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphAddKernelNode_v2");

CUresult (*lcuGraphKernelNodeGetParams_v2) (CUgraphNode  hNode, CUDA_KERNEL_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, CUDA_KERNEL_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphKernelNodeGetParams_v2");

CUresult (*lcuGraphKernelNodeSetParams_v2) (CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphKernelNodeSetParams_v2");

CUresult (*lcuGraphAddMemcpyNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx)) dlsym(cuda_handle, "cuGraphAddMemcpyNode");

CUresult (*lcuGraphMemcpyNodeGetParams) (CUgraphNode  hNode, CUDA_MEMCPY3D * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, CUDA_MEMCPY3D * nodeParams)) dlsym(cuda_handle, "cuGraphMemcpyNodeGetParams");

CUresult (*lcuGraphMemcpyNodeSetParams) (CUgraphNode  hNode, const CUDA_MEMCPY3D * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, const CUDA_MEMCPY3D * nodeParams)) dlsym(cuda_handle, "cuGraphMemcpyNodeSetParams");

CUresult (*lcuGraphAddMemsetNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx)) dlsym(cuda_handle, "cuGraphAddMemsetNode");

CUresult (*lcuGraphMemsetNodeGetParams) (CUgraphNode  hNode, CUDA_MEMSET_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, CUDA_MEMSET_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphMemsetNodeGetParams");

CUresult (*lcuGraphMemsetNodeSetParams) (CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphMemsetNodeSetParams");

CUresult (*lcuGraphAddHostNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_HOST_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_HOST_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphAddHostNode");

CUresult (*lcuGraphHostNodeGetParams) (CUgraphNode  hNode, CUDA_HOST_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, CUDA_HOST_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphHostNodeGetParams");

CUresult (*lcuGraphHostNodeSetParams) (CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphHostNodeSetParams");

CUresult (*lcuGraphAddChildGraphNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUgraph  childGraph) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUgraph  childGraph)) dlsym(cuda_handle, "cuGraphAddChildGraphNode");

CUresult (*lcuGraphChildGraphNodeGetGraph) (CUgraphNode  hNode, CUgraph * phGraph) =
	(CUresult (*) (CUgraphNode  hNode, CUgraph * phGraph)) dlsym(cuda_handle, "cuGraphChildGraphNodeGetGraph");

CUresult (*lcuGraphAddEmptyNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies)) dlsym(cuda_handle, "cuGraphAddEmptyNode");

CUresult (*lcuGraphAddEventRecordNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event)) dlsym(cuda_handle, "cuGraphAddEventRecordNode");

CUresult (*lcuGraphEventRecordNodeGetEvent) (CUgraphNode  hNode, CUevent * event_out) =
	(CUresult (*) (CUgraphNode  hNode, CUevent * event_out)) dlsym(cuda_handle, "cuGraphEventRecordNodeGetEvent");

CUresult (*lcuGraphEventRecordNodeSetEvent) (CUgraphNode  hNode, CUevent  event) =
	(CUresult (*) (CUgraphNode  hNode, CUevent  event)) dlsym(cuda_handle, "cuGraphEventRecordNodeSetEvent");

CUresult (*lcuGraphAddEventWaitNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event)) dlsym(cuda_handle, "cuGraphAddEventWaitNode");

CUresult (*lcuGraphEventWaitNodeGetEvent) (CUgraphNode  hNode, CUevent * event_out) =
	(CUresult (*) (CUgraphNode  hNode, CUevent * event_out)) dlsym(cuda_handle, "cuGraphEventWaitNodeGetEvent");

CUresult (*lcuGraphEventWaitNodeSetEvent) (CUgraphNode  hNode, CUevent  event) =
	(CUresult (*) (CUgraphNode  hNode, CUevent  event)) dlsym(cuda_handle, "cuGraphEventWaitNodeSetEvent");

CUresult (*lcuGraphAddExternalSemaphoresSignalNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphAddExternalSemaphoresSignalNode");

CUresult (*lcuGraphExternalSemaphoresSignalNodeGetParams) (CUgraphNode  hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * params_out) =
	(CUresult (*) (CUgraphNode  hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * params_out)) dlsym(cuda_handle, "cuGraphExternalSemaphoresSignalNodeGetParams");

CUresult (*lcuGraphExternalSemaphoresSignalNodeSetParams) (CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphExternalSemaphoresSignalNodeSetParams");

CUresult (*lcuGraphAddExternalSemaphoresWaitNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphAddExternalSemaphoresWaitNode");

CUresult (*lcuGraphExternalSemaphoresWaitNodeGetParams) (CUgraphNode  hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS * params_out) =
	(CUresult (*) (CUgraphNode  hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS * params_out)) dlsym(cuda_handle, "cuGraphExternalSemaphoresWaitNodeGetParams");

CUresult (*lcuGraphExternalSemaphoresWaitNodeSetParams) (CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphExternalSemaphoresWaitNodeSetParams");

CUresult (*lcuGraphAddBatchMemOpNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphAddBatchMemOpNode");

CUresult (*lcuGraphBatchMemOpNodeGetParams) (CUgraphNode  hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams_out) =
	(CUresult (*) (CUgraphNode  hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams_out)) dlsym(cuda_handle, "cuGraphBatchMemOpNodeGetParams");

CUresult (*lcuGraphBatchMemOpNodeSetParams) (CUgraphNode  hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphBatchMemOpNodeSetParams");

CUresult (*lcuGraphExecBatchMemOpNodeSetParams) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphExecBatchMemOpNodeSetParams");

CUresult (*lcuGraphAddMemAllocNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphAddMemAllocNode");

CUresult (*lcuGraphMemAllocNodeGetParams) (CUgraphNode  hNode, CUDA_MEM_ALLOC_NODE_PARAMS * params_out) =
	(CUresult (*) (CUgraphNode  hNode, CUDA_MEM_ALLOC_NODE_PARAMS * params_out)) dlsym(cuda_handle, "cuGraphMemAllocNodeGetParams");

CUresult (*lcuGraphAddMemFreeNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUdeviceptr  dptr) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUdeviceptr  dptr)) dlsym(cuda_handle, "cuGraphAddMemFreeNode");

CUresult (*lcuGraphMemFreeNodeGetParams) (CUgraphNode  hNode, CUdeviceptr * dptr_out) =
	(CUresult (*) (CUgraphNode  hNode, CUdeviceptr * dptr_out)) dlsym(cuda_handle, "cuGraphMemFreeNodeGetParams");

CUresult (*lcuDeviceGraphMemTrim) (CUdevice  device) =
	(CUresult (*) (CUdevice  device)) dlsym(cuda_handle, "cuDeviceGraphMemTrim");

CUresult (*lcuDeviceGetGraphMemAttribute) (CUdevice  device, CUgraphMem_attribute  attr, void*  value) =
	(CUresult (*) (CUdevice  device, CUgraphMem_attribute  attr, void*  value)) dlsym(cuda_handle, "cuDeviceGetGraphMemAttribute");

CUresult (*lcuDeviceSetGraphMemAttribute) (CUdevice  device, CUgraphMem_attribute  attr, void*  value) =
	(CUresult (*) (CUdevice  device, CUgraphMem_attribute  attr, void*  value)) dlsym(cuda_handle, "cuDeviceSetGraphMemAttribute");

CUresult (*lcuGraphClone) (CUgraph * phGraphClone, CUgraph  originalGraph) =
	(CUresult (*) (CUgraph * phGraphClone, CUgraph  originalGraph)) dlsym(cuda_handle, "cuGraphClone");

CUresult (*lcuGraphNodeFindInClone) (CUgraphNode * phNode, CUgraphNode  hOriginalNode, CUgraph  hClonedGraph) =
	(CUresult (*) (CUgraphNode * phNode, CUgraphNode  hOriginalNode, CUgraph  hClonedGraph)) dlsym(cuda_handle, "cuGraphNodeFindInClone");

CUresult (*lcuGraphNodeGetType) (CUgraphNode  hNode, CUgraphNodeType * type) =
	(CUresult (*) (CUgraphNode  hNode, CUgraphNodeType * type)) dlsym(cuda_handle, "cuGraphNodeGetType");

CUresult (*lcuGraphGetNodes) (CUgraph  hGraph, CUgraphNode * nodes, size_t * numNodes) =
	(CUresult (*) (CUgraph  hGraph, CUgraphNode * nodes, size_t * numNodes)) dlsym(cuda_handle, "cuGraphGetNodes");

CUresult (*lcuGraphGetRootNodes) (CUgraph  hGraph, CUgraphNode * rootNodes, size_t * numRootNodes) =
	(CUresult (*) (CUgraph  hGraph, CUgraphNode * rootNodes, size_t * numRootNodes)) dlsym(cuda_handle, "cuGraphGetRootNodes");

CUresult (*lcuGraphGetEdges) (CUgraph  hGraph, CUgraphNode * from, CUgraphNode * to, size_t * numEdges) =
	(CUresult (*) (CUgraph  hGraph, CUgraphNode * from, CUgraphNode * to, size_t * numEdges)) dlsym(cuda_handle, "cuGraphGetEdges");

CUresult (*lcuGraphNodeGetDependencies) (CUgraphNode  hNode, CUgraphNode * dependencies, size_t * numDependencies) =
	(CUresult (*) (CUgraphNode  hNode, CUgraphNode * dependencies, size_t * numDependencies)) dlsym(cuda_handle, "cuGraphNodeGetDependencies");

CUresult (*lcuGraphNodeGetDependentNodes) (CUgraphNode  hNode, CUgraphNode * dependentNodes, size_t * numDependentNodes) =
	(CUresult (*) (CUgraphNode  hNode, CUgraphNode * dependentNodes, size_t * numDependentNodes)) dlsym(cuda_handle, "cuGraphNodeGetDependentNodes");

CUresult (*lcuGraphAddDependencies) (CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies) =
	(CUresult (*) (CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies)) dlsym(cuda_handle, "cuGraphAddDependencies");

CUresult (*lcuGraphRemoveDependencies) (CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies) =
	(CUresult (*) (CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies)) dlsym(cuda_handle, "cuGraphRemoveDependencies");

CUresult (*lcuGraphDestroyNode) (CUgraphNode  hNode) =
	(CUresult (*) (CUgraphNode  hNode)) dlsym(cuda_handle, "cuGraphDestroyNode");

CUresult (*lcuGraphInstantiateWithFlags) (CUgraphExec * phGraphExec, CUgraph  hGraph, unsigned long long  flags) =
	(CUresult (*) (CUgraphExec * phGraphExec, CUgraph  hGraph, unsigned long long  flags)) dlsym(cuda_handle, "cuGraphInstantiateWithFlags");

CUresult (*lcuGraphInstantiateWithParams) (CUgraphExec * phGraphExec, CUgraph  hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS * instantiateParams) =
	(CUresult (*) (CUgraphExec * phGraphExec, CUgraph  hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS * instantiateParams)) dlsym(cuda_handle, "cuGraphInstantiateWithParams");

CUresult (*lcuGraphExecGetFlags) (CUgraphExec  hGraphExec, cuuint64_t * flags) =
	(CUresult (*) (CUgraphExec  hGraphExec, cuuint64_t * flags)) dlsym(cuda_handle, "cuGraphExecGetFlags");

CUresult (*lcuGraphExecKernelNodeSetParams_v2) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphExecKernelNodeSetParams_v2");

CUresult (*lcuGraphExecMemcpyNodeSetParams) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx)) dlsym(cuda_handle, "cuGraphExecMemcpyNodeSetParams");

CUresult (*lcuGraphExecMemsetNodeSetParams) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx)) dlsym(cuda_handle, "cuGraphExecMemsetNodeSetParams");

CUresult (*lcuGraphExecHostNodeSetParams) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphExecHostNodeSetParams");

CUresult (*lcuGraphExecChildGraphNodeSetParams) (CUgraphExec  hGraphExec, CUgraphNode  hNode, CUgraph  childGraph) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, CUgraph  childGraph)) dlsym(cuda_handle, "cuGraphExecChildGraphNodeSetParams");

CUresult (*lcuGraphExecEventRecordNodeSetEvent) (CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event)) dlsym(cuda_handle, "cuGraphExecEventRecordNodeSetEvent");

CUresult (*lcuGraphExecEventWaitNodeSetEvent) (CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event)) dlsym(cuda_handle, "cuGraphExecEventWaitNodeSetEvent");

CUresult (*lcuGraphExecExternalSemaphoresSignalNodeSetParams) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphExecExternalSemaphoresSignalNodeSetParams");

CUresult (*lcuGraphExecExternalSemaphoresWaitNodeSetParams) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)) dlsym(cuda_handle, "cuGraphExecExternalSemaphoresWaitNodeSetParams");

CUresult (*lcuGraphNodeSetEnabled) (CUgraphExec  hGraphExec, CUgraphNode  hNode, unsigned int  isEnabled) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, unsigned int  isEnabled)) dlsym(cuda_handle, "cuGraphNodeSetEnabled");

CUresult (*lcuGraphNodeGetEnabled) (CUgraphExec  hGraphExec, CUgraphNode  hNode, unsigned int * isEnabled) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, unsigned int * isEnabled)) dlsym(cuda_handle, "cuGraphNodeGetEnabled");

CUresult (*lcuGraphUpload) (CUgraphExec  hGraphExec, CUstream  hStream) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUstream  hStream)) dlsym(cuda_handle, "cuGraphUpload");

CUresult (*lcuGraphLaunch) (CUgraphExec  hGraphExec, CUstream  hStream) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUstream  hStream)) dlsym(cuda_handle, "cuGraphLaunch");

CUresult (*lcuGraphExecDestroy) (CUgraphExec  hGraphExec) =
	(CUresult (*) (CUgraphExec  hGraphExec)) dlsym(cuda_handle, "cuGraphExecDestroy");

CUresult (*lcuGraphDestroy) (CUgraph  hGraph) =
	(CUresult (*) (CUgraph  hGraph)) dlsym(cuda_handle, "cuGraphDestroy");

CUresult (*lcuGraphExecUpdate_v2) (CUgraphExec  hGraphExec, CUgraph  hGraph, CUgraphExecUpdateResultInfo * resultInfo) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraph  hGraph, CUgraphExecUpdateResultInfo * resultInfo)) dlsym(cuda_handle, "cuGraphExecUpdate_v2");

CUresult (*lcuGraphKernelNodeCopyAttributes) (CUgraphNode  dst, CUgraphNode  src) =
	(CUresult (*) (CUgraphNode  dst, CUgraphNode  src)) dlsym(cuda_handle, "cuGraphKernelNodeCopyAttributes");

CUresult (*lcuGraphKernelNodeGetAttribute) (CUgraphNode  hNode, CUkernelNodeAttrID  attr, CUkernelNodeAttrValue * value_out) =
	(CUresult (*) (CUgraphNode  hNode, CUkernelNodeAttrID  attr, CUkernelNodeAttrValue * value_out)) dlsym(cuda_handle, "cuGraphKernelNodeGetAttribute");

CUresult (*lcuGraphKernelNodeSetAttribute) (CUgraphNode  hNode, CUkernelNodeAttrID  attr, const CUkernelNodeAttrValue * value) =
	(CUresult (*) (CUgraphNode  hNode, CUkernelNodeAttrID  attr, const CUkernelNodeAttrValue * value)) dlsym(cuda_handle, "cuGraphKernelNodeSetAttribute");

CUresult (*lcuGraphDebugDotPrint) (CUgraph  hGraph, const char * path, unsigned int  flags) =
	(CUresult (*) (CUgraph  hGraph, const char * path, unsigned int  flags)) dlsym(cuda_handle, "cuGraphDebugDotPrint");

CUresult (*lcuUserObjectCreate) (CUuserObject * object_out, void * ptr, CUhostFn  destroy, unsigned int  initialRefcount, unsigned int  flags) =
	(CUresult (*) (CUuserObject * object_out, void * ptr, CUhostFn  destroy, unsigned int  initialRefcount, unsigned int  flags)) dlsym(cuda_handle, "cuUserObjectCreate");

CUresult (*lcuUserObjectRetain) (CUuserObject  object, unsigned int  count) =
	(CUresult (*) (CUuserObject  object, unsigned int  count)) dlsym(cuda_handle, "cuUserObjectRetain");

CUresult (*lcuUserObjectRelease) (CUuserObject  object, unsigned int  count) =
	(CUresult (*) (CUuserObject  object, unsigned int  count)) dlsym(cuda_handle, "cuUserObjectRelease");

CUresult (*lcuGraphRetainUserObject) (CUgraph  graph, CUuserObject  object, unsigned int  count, unsigned int  flags) =
	(CUresult (*) (CUgraph  graph, CUuserObject  object, unsigned int  count, unsigned int  flags)) dlsym(cuda_handle, "cuGraphRetainUserObject");

CUresult (*lcuGraphReleaseUserObject) (CUgraph  graph, CUuserObject  object, unsigned int  count) =
	(CUresult (*) (CUgraph  graph, CUuserObject  object, unsigned int  count)) dlsym(cuda_handle, "cuGraphReleaseUserObject");

CUresult (*lcuGraphAddNode) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUgraphNodeParams * nodeParams) =
	(CUresult (*) (CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUgraphNodeParams * nodeParams)) dlsym(cuda_handle, "cuGraphAddNode");

CUresult (*lcuGraphNodeSetParams) (CUgraphNode  hNode, CUgraphNodeParams * nodeParams) =
	(CUresult (*) (CUgraphNode  hNode, CUgraphNodeParams * nodeParams)) dlsym(cuda_handle, "cuGraphNodeSetParams");

CUresult (*lcuGraphExecNodeSetParams) (CUgraphExec  hGraphExec, CUgraphNode  hNode, CUgraphNodeParams * nodeParams) =
	(CUresult (*) (CUgraphExec  hGraphExec, CUgraphNode  hNode, CUgraphNodeParams * nodeParams)) dlsym(cuda_handle, "cuGraphExecNodeSetParams");

CUresult (*lcuOccupancyMaxActiveBlocksPerMultiprocessor) (int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize) =
	(CUresult (*) (int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize)) dlsym(cuda_handle, "cuOccupancyMaxActiveBlocksPerMultiprocessor");

CUresult (*lcuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) (int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags) =
	(CUresult (*) (int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)) dlsym(cuda_handle, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");

CUresult (*lcuOccupancyMaxPotentialBlockSize) (int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit) =
	(CUresult (*) (int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit)) dlsym(cuda_handle, "cuOccupancyMaxPotentialBlockSize");

CUresult (*lcuOccupancyMaxPotentialBlockSizeWithFlags) (int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit, unsigned int  flags) =
	(CUresult (*) (int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit, unsigned int  flags)) dlsym(cuda_handle, "cuOccupancyMaxPotentialBlockSizeWithFlags");

CUresult (*lcuOccupancyAvailableDynamicSMemPerBlock) (size_t * dynamicSmemSize, CUfunction  func, int  numBlocks, int  blockSize) =
	(CUresult (*) (size_t * dynamicSmemSize, CUfunction  func, int  numBlocks, int  blockSize)) dlsym(cuda_handle, "cuOccupancyAvailableDynamicSMemPerBlock");

CUresult (*lcuOccupancyMaxPotentialClusterSize) (int * clusterSize, CUfunction  func, const CUlaunchConfig * config) =
	(CUresult (*) (int * clusterSize, CUfunction  func, const CUlaunchConfig * config)) dlsym(cuda_handle, "cuOccupancyMaxPotentialClusterSize");

CUresult (*lcuOccupancyMaxActiveClusters) (int * numClusters, CUfunction  func, const CUlaunchConfig * config) =
	(CUresult (*) (int * numClusters, CUfunction  func, const CUlaunchConfig * config)) dlsym(cuda_handle, "cuOccupancyMaxActiveClusters");

CUresult (*lcuTexRefSetArray) (CUtexref  hTexRef, CUarray  hArray, unsigned int  Flags) =
	(CUresult (*) (CUtexref  hTexRef, CUarray  hArray, unsigned int  Flags)) dlsym(cuda_handle, "cuTexRefSetArray");

CUresult (*lcuTexRefSetMipmappedArray) (CUtexref  hTexRef, CUmipmappedArray  hMipmappedArray, unsigned int  Flags) =
	(CUresult (*) (CUtexref  hTexRef, CUmipmappedArray  hMipmappedArray, unsigned int  Flags)) dlsym(cuda_handle, "cuTexRefSetMipmappedArray");

CUresult (*lcuTexRefSetAddress_v2) (size_t * ByteOffset, CUtexref  hTexRef, CUdeviceptr  dptr, size_t  bytes) =
	(CUresult (*) (size_t * ByteOffset, CUtexref  hTexRef, CUdeviceptr  dptr, size_t  bytes)) dlsym(cuda_handle, "cuTexRefSetAddress_v2");

CUresult (*lcuTexRefSetAddress2D_v3) (CUtexref  hTexRef, const CUDA_ARRAY_DESCRIPTOR * desc, CUdeviceptr  dptr, size_t  Pitch) =
	(CUresult (*) (CUtexref  hTexRef, const CUDA_ARRAY_DESCRIPTOR * desc, CUdeviceptr  dptr, size_t  Pitch)) dlsym(cuda_handle, "cuTexRefSetAddress2D_v3");

CUresult (*lcuTexRefSetFormat) (CUtexref  hTexRef, CUarray_format  fmt, int  NumPackedComponents) =
	(CUresult (*) (CUtexref  hTexRef, CUarray_format  fmt, int  NumPackedComponents)) dlsym(cuda_handle, "cuTexRefSetFormat");

CUresult (*lcuTexRefSetAddressMode) (CUtexref  hTexRef, int  dim, CUaddress_mode  am) =
	(CUresult (*) (CUtexref  hTexRef, int  dim, CUaddress_mode  am)) dlsym(cuda_handle, "cuTexRefSetAddressMode");

CUresult (*lcuTexRefSetFilterMode) (CUtexref  hTexRef, CUfilter_mode  fm) =
	(CUresult (*) (CUtexref  hTexRef, CUfilter_mode  fm)) dlsym(cuda_handle, "cuTexRefSetFilterMode");

CUresult (*lcuTexRefSetMipmapFilterMode) (CUtexref  hTexRef, CUfilter_mode  fm) =
	(CUresult (*) (CUtexref  hTexRef, CUfilter_mode  fm)) dlsym(cuda_handle, "cuTexRefSetMipmapFilterMode");

CUresult (*lcuTexRefSetMipmapLevelBias) (CUtexref  hTexRef, float  bias) =
	(CUresult (*) (CUtexref  hTexRef, float  bias)) dlsym(cuda_handle, "cuTexRefSetMipmapLevelBias");

CUresult (*lcuTexRefSetMipmapLevelClamp) (CUtexref  hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp) =
	(CUresult (*) (CUtexref  hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp)) dlsym(cuda_handle, "cuTexRefSetMipmapLevelClamp");

CUresult (*lcuTexRefSetMaxAnisotropy) (CUtexref  hTexRef, unsigned int  maxAniso) =
	(CUresult (*) (CUtexref  hTexRef, unsigned int  maxAniso)) dlsym(cuda_handle, "cuTexRefSetMaxAnisotropy");

CUresult (*lcuTexRefSetBorderColor) (CUtexref  hTexRef, float * pBorderColor) =
	(CUresult (*) (CUtexref  hTexRef, float * pBorderColor)) dlsym(cuda_handle, "cuTexRefSetBorderColor");

CUresult (*lcuTexRefSetFlags) (CUtexref  hTexRef, unsigned int  Flags) =
	(CUresult (*) (CUtexref  hTexRef, unsigned int  Flags)) dlsym(cuda_handle, "cuTexRefSetFlags");

CUresult (*lcuTexRefGetAddress_v2) (CUdeviceptr * pdptr, CUtexref  hTexRef) =
	(CUresult (*) (CUdeviceptr * pdptr, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetAddress_v2");

CUresult (*lcuTexRefGetArray) (CUarray * phArray, CUtexref  hTexRef) =
	(CUresult (*) (CUarray * phArray, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetArray");

CUresult (*lcuTexRefGetMipmappedArray) (CUmipmappedArray * phMipmappedArray, CUtexref  hTexRef) =
	(CUresult (*) (CUmipmappedArray * phMipmappedArray, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetMipmappedArray");

CUresult (*lcuTexRefGetAddressMode) (CUaddress_mode * pam, CUtexref  hTexRef, int  dim) =
	(CUresult (*) (CUaddress_mode * pam, CUtexref  hTexRef, int  dim)) dlsym(cuda_handle, "cuTexRefGetAddressMode");

CUresult (*lcuTexRefGetFilterMode) (CUfilter_mode * pfm, CUtexref  hTexRef) =
	(CUresult (*) (CUfilter_mode * pfm, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetFilterMode");

CUresult (*lcuTexRefGetFormat) (CUarray_format * pFormat, int * pNumChannels, CUtexref  hTexRef) =
	(CUresult (*) (CUarray_format * pFormat, int * pNumChannels, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetFormat");

CUresult (*lcuTexRefGetMipmapFilterMode) (CUfilter_mode * pfm, CUtexref  hTexRef) =
	(CUresult (*) (CUfilter_mode * pfm, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetMipmapFilterMode");

CUresult (*lcuTexRefGetMipmapLevelBias) (float * pbias, CUtexref  hTexRef) =
	(CUresult (*) (float * pbias, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetMipmapLevelBias");

CUresult (*lcuTexRefGetMipmapLevelClamp) (float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, CUtexref  hTexRef) =
	(CUresult (*) (float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetMipmapLevelClamp");

CUresult (*lcuTexRefGetMaxAnisotropy) (int * pmaxAniso, CUtexref  hTexRef) =
	(CUresult (*) (int * pmaxAniso, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetMaxAnisotropy");

CUresult (*lcuTexRefGetBorderColor) (float * pBorderColor, CUtexref  hTexRef) =
	(CUresult (*) (float * pBorderColor, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetBorderColor");

CUresult (*lcuTexRefGetFlags) (unsigned int * pFlags, CUtexref  hTexRef) =
	(CUresult (*) (unsigned int * pFlags, CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefGetFlags");

CUresult (*lcuTexRefCreate) (CUtexref * pTexRef) =
	(CUresult (*) (CUtexref * pTexRef)) dlsym(cuda_handle, "cuTexRefCreate");

CUresult (*lcuTexRefDestroy) (CUtexref  hTexRef) =
	(CUresult (*) (CUtexref  hTexRef)) dlsym(cuda_handle, "cuTexRefDestroy");

CUresult (*lcuSurfRefSetArray) (CUsurfref  hSurfRef, CUarray  hArray, unsigned int  Flags) =
	(CUresult (*) (CUsurfref  hSurfRef, CUarray  hArray, unsigned int  Flags)) dlsym(cuda_handle, "cuSurfRefSetArray");

CUresult (*lcuSurfRefGetArray) (CUarray * phArray, CUsurfref  hSurfRef) =
	(CUresult (*) (CUarray * phArray, CUsurfref  hSurfRef)) dlsym(cuda_handle, "cuSurfRefGetArray");

CUresult (*lcuTexObjectCreate) (CUtexObject * pTexObject, const CUDA_RESOURCE_DESC * pResDesc, const CUDA_TEXTURE_DESC * pTexDesc, const CUDA_RESOURCE_VIEW_DESC * pResViewDesc) =
	(CUresult (*) (CUtexObject * pTexObject, const CUDA_RESOURCE_DESC * pResDesc, const CUDA_TEXTURE_DESC * pTexDesc, const CUDA_RESOURCE_VIEW_DESC * pResViewDesc)) dlsym(cuda_handle, "cuTexObjectCreate");

CUresult (*lcuTexObjectDestroy) (CUtexObject  texObject) =
	(CUresult (*) (CUtexObject  texObject)) dlsym(cuda_handle, "cuTexObjectDestroy");

CUresult (*lcuTexObjectGetResourceDesc) (CUDA_RESOURCE_DESC * pResDesc, CUtexObject  texObject) =
	(CUresult (*) (CUDA_RESOURCE_DESC * pResDesc, CUtexObject  texObject)) dlsym(cuda_handle, "cuTexObjectGetResourceDesc");

CUresult (*lcuTexObjectGetTextureDesc) (CUDA_TEXTURE_DESC * pTexDesc, CUtexObject  texObject) =
	(CUresult (*) (CUDA_TEXTURE_DESC * pTexDesc, CUtexObject  texObject)) dlsym(cuda_handle, "cuTexObjectGetTextureDesc");

CUresult (*lcuTexObjectGetResourceViewDesc) (CUDA_RESOURCE_VIEW_DESC * pResViewDesc, CUtexObject  texObject) =
	(CUresult (*) (CUDA_RESOURCE_VIEW_DESC * pResViewDesc, CUtexObject  texObject)) dlsym(cuda_handle, "cuTexObjectGetResourceViewDesc");

CUresult (*lcuSurfObjectCreate) (CUsurfObject * pSurfObject, const CUDA_RESOURCE_DESC * pResDesc) =
	(CUresult (*) (CUsurfObject * pSurfObject, const CUDA_RESOURCE_DESC * pResDesc)) dlsym(cuda_handle, "cuSurfObjectCreate");

CUresult (*lcuSurfObjectDestroy) (CUsurfObject  surfObject) =
	(CUresult (*) (CUsurfObject  surfObject)) dlsym(cuda_handle, "cuSurfObjectDestroy");

CUresult (*lcuSurfObjectGetResourceDesc) (CUDA_RESOURCE_DESC * pResDesc, CUsurfObject  surfObject) =
	(CUresult (*) (CUDA_RESOURCE_DESC * pResDesc, CUsurfObject  surfObject)) dlsym(cuda_handle, "cuSurfObjectGetResourceDesc");

CUresult (*lcuTensorMapEncodeTiled) (CUtensorMap * tensorMap, CUtensorMapDataType  tensorDataType, cuuint32_t  tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const cuuint32_t * boxDim, const cuuint32_t * elementStrides, CUtensorMapInterleave  interleave, CUtensorMapSwizzle  swizzle, CUtensorMapL2promotion  l2Promotion, CUtensorMapFloatOOBfill  oobFill) =
	(CUresult (*) (CUtensorMap * tensorMap, CUtensorMapDataType  tensorDataType, cuuint32_t  tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const cuuint32_t * boxDim, const cuuint32_t * elementStrides, CUtensorMapInterleave  interleave, CUtensorMapSwizzle  swizzle, CUtensorMapL2promotion  l2Promotion, CUtensorMapFloatOOBfill  oobFill)) dlsym(cuda_handle, "cuTensorMapEncodeTiled");

CUresult (*lcuTensorMapEncodeIm2col) (CUtensorMap * tensorMap, CUtensorMapDataType  tensorDataType, cuuint32_t  tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const int * pixelBoxLowerCorner, const int * pixelBoxUpperCorner, cuuint32_t  channelsPerPixel, cuuint32_t  pixelsPerColumn, const cuuint32_t * elementStrides, CUtensorMapInterleave  interleave, CUtensorMapSwizzle  swizzle, CUtensorMapL2promotion  l2Promotion, CUtensorMapFloatOOBfill  oobFill) =
	(CUresult (*) (CUtensorMap * tensorMap, CUtensorMapDataType  tensorDataType, cuuint32_t  tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const int * pixelBoxLowerCorner, const int * pixelBoxUpperCorner, cuuint32_t  channelsPerPixel, cuuint32_t  pixelsPerColumn, const cuuint32_t * elementStrides, CUtensorMapInterleave  interleave, CUtensorMapSwizzle  swizzle, CUtensorMapL2promotion  l2Promotion, CUtensorMapFloatOOBfill  oobFill)) dlsym(cuda_handle, "cuTensorMapEncodeIm2col");

CUresult (*lcuTensorMapReplaceAddress) (CUtensorMap * tensorMap, void * globalAddress) =
	(CUresult (*) (CUtensorMap * tensorMap, void * globalAddress)) dlsym(cuda_handle, "cuTensorMapReplaceAddress");

CUresult (*lcuDeviceCanAccessPeer) (int * canAccessPeer, CUdevice  dev, CUdevice  peerDev) =
	(CUresult (*) (int * canAccessPeer, CUdevice  dev, CUdevice  peerDev)) dlsym(cuda_handle, "cuDeviceCanAccessPeer");

CUresult (*lcuCtxEnablePeerAccess) (CUcontext  peerContext, unsigned int  Flags) =
	(CUresult (*) (CUcontext  peerContext, unsigned int  Flags)) dlsym(cuda_handle, "cuCtxEnablePeerAccess");

CUresult (*lcuCtxDisablePeerAccess) (CUcontext  peerContext) =
	(CUresult (*) (CUcontext  peerContext)) dlsym(cuda_handle, "cuCtxDisablePeerAccess");

CUresult (*lcuDeviceGetP2PAttribute) (int*  value, CUdevice_P2PAttribute  attrib, CUdevice  srcDevice, CUdevice  dstDevice) =
	(CUresult (*) (int*  value, CUdevice_P2PAttribute  attrib, CUdevice  srcDevice, CUdevice  dstDevice)) dlsym(cuda_handle, "cuDeviceGetP2PAttribute");

CUresult (*lcuGraphicsUnregisterResource) (CUgraphicsResource  resource) =
	(CUresult (*) (CUgraphicsResource  resource)) dlsym(cuda_handle, "cuGraphicsUnregisterResource");

CUresult (*lcuGraphicsSubResourceGetMappedArray) (CUarray * pArray, CUgraphicsResource  resource, unsigned int  arrayIndex, unsigned int  mipLevel) =
	(CUresult (*) (CUarray * pArray, CUgraphicsResource  resource, unsigned int  arrayIndex, unsigned int  mipLevel)) dlsym(cuda_handle, "cuGraphicsSubResourceGetMappedArray");

CUresult (*lcuGraphicsResourceGetMappedMipmappedArray) (CUmipmappedArray * pMipmappedArray, CUgraphicsResource  resource) =
	(CUresult (*) (CUmipmappedArray * pMipmappedArray, CUgraphicsResource  resource)) dlsym(cuda_handle, "cuGraphicsResourceGetMappedMipmappedArray");

CUresult (*lcuGraphicsResourceGetMappedPointer_v2) (CUdeviceptr * pDevPtr, size_t * pSize, CUgraphicsResource  resource) =
	(CUresult (*) (CUdeviceptr * pDevPtr, size_t * pSize, CUgraphicsResource  resource)) dlsym(cuda_handle, "cuGraphicsResourceGetMappedPointer_v2");

CUresult (*lcuGraphicsResourceSetMapFlags_v2) (CUgraphicsResource  resource, unsigned int  flags) =
	(CUresult (*) (CUgraphicsResource  resource, unsigned int  flags)) dlsym(cuda_handle, "cuGraphicsResourceSetMapFlags_v2");

CUresult (*lcuGraphicsMapResources) (unsigned int  count, CUgraphicsResource * resources, CUstream  hStream) =
	(CUresult (*) (unsigned int  count, CUgraphicsResource * resources, CUstream  hStream)) dlsym(cuda_handle, "cuGraphicsMapResources");

CUresult (*lcuGraphicsUnmapResources) (unsigned int  count, CUgraphicsResource * resources, CUstream  hStream) =
	(CUresult (*) (unsigned int  count, CUgraphicsResource * resources, CUstream  hStream)) dlsym(cuda_handle, "cuGraphicsUnmapResources");

CUresult (*lcuGetProcAddress_v2) (const char * symbol, void ** pfn, int  cudaVersion, cuuint64_t  flags, CUdriverProcAddressQueryResult * symbolStatus) =
	(CUresult (*) (const char * symbol, void ** pfn, int  cudaVersion, cuuint64_t  flags, CUdriverProcAddressQueryResult * symbolStatus)) dlsym(cuda_handle, "cuGetProcAddress_v2");

CUresult (*lcuCoredumpGetAttribute) (CUcoredumpSettings  attrib, void*  value, size_t * size) =
	(CUresult (*) (CUcoredumpSettings  attrib, void*  value, size_t * size)) dlsym(cuda_handle, "cuCoredumpGetAttribute");

CUresult (*lcuCoredumpGetAttributeGlobal) (CUcoredumpSettings  attrib, void * value, size_t * size) =
	(CUresult (*) (CUcoredumpSettings  attrib, void * value, size_t * size)) dlsym(cuda_handle, "cuCoredumpGetAttributeGlobal");

CUresult (*lcuCoredumpSetAttribute) (CUcoredumpSettings  attrib, void*  value, size_t * size) =
	(CUresult (*) (CUcoredumpSettings  attrib, void*  value, size_t * size)) dlsym(cuda_handle, "cuCoredumpSetAttribute");

CUresult (*lcuCoredumpSetAttributeGlobal) (CUcoredumpSettings  attrib, void * value, size_t * size) =
	(CUresult (*) (CUcoredumpSettings  attrib, void * value, size_t * size)) dlsym(cuda_handle, "cuCoredumpSetAttributeGlobal");

CUresult (*lcuGetExportTable) (const void ** ppExportTable, const CUuuid * pExportTableId) =
	(CUresult (*) (const void ** ppExportTable, const CUuuid * pExportTableId)) dlsym(cuda_handle, "cuGetExportTable");

cudaError_t (*lcudaDeviceReset) () =
	(cudaError_t (*) ()) dlsym(cudart_handle, "cudaDeviceReset");

cudaError_t (*lcudaDeviceSynchronize) () =
	(cudaError_t (*) ()) dlsym(cudart_handle, "cudaDeviceSynchronize");

cudaError_t (*lcudaDeviceSetLimit) (enum cudaLimit  limit, size_t  value) =
	(cudaError_t (*) (enum cudaLimit  limit, size_t  value)) dlsym(cudart_handle, "cudaDeviceSetLimit");

cudaError_t (*lcudaDeviceGetLimit) (size_t * pValue, enum cudaLimit  limit) =
	(cudaError_t (*) (size_t * pValue, enum cudaLimit  limit)) dlsym(cudart_handle, "cudaDeviceGetLimit");

cudaError_t (*lcudaDeviceGetTexture1DLinearMaxWidth) (size_t * maxWidthInElements, const struct cudaChannelFormatDesc * fmtDesc, int  device) =
	(cudaError_t (*) (size_t * maxWidthInElements, const struct cudaChannelFormatDesc * fmtDesc, int  device)) dlsym(cudart_handle, "cudaDeviceGetTexture1DLinearMaxWidth");

cudaError_t (*lcudaDeviceGetCacheConfig) (enum cudaFuncCache * pCacheConfig) =
	(cudaError_t (*) (enum cudaFuncCache * pCacheConfig)) dlsym(cudart_handle, "cudaDeviceGetCacheConfig");

cudaError_t (*lcudaDeviceGetStreamPriorityRange) (int * leastPriority, int * greatestPriority) =
	(cudaError_t (*) (int * leastPriority, int * greatestPriority)) dlsym(cudart_handle, "cudaDeviceGetStreamPriorityRange");

cudaError_t (*lcudaDeviceSetCacheConfig) (enum cudaFuncCache  cacheConfig) =
	(cudaError_t (*) (enum cudaFuncCache  cacheConfig)) dlsym(cudart_handle, "cudaDeviceSetCacheConfig");

cudaError_t (*lcudaDeviceGetSharedMemConfig) (enum cudaSharedMemConfig * pConfig) =
	(cudaError_t (*) (enum cudaSharedMemConfig * pConfig)) dlsym(cudart_handle, "cudaDeviceGetSharedMemConfig");

cudaError_t (*lcudaDeviceSetSharedMemConfig) (enum cudaSharedMemConfig  config) =
	(cudaError_t (*) (enum cudaSharedMemConfig  config)) dlsym(cudart_handle, "cudaDeviceSetSharedMemConfig");

cudaError_t (*lcudaDeviceGetByPCIBusId) (int * device, const char * pciBusId) =
	(cudaError_t (*) (int * device, const char * pciBusId)) dlsym(cudart_handle, "cudaDeviceGetByPCIBusId");

cudaError_t (*lcudaDeviceGetPCIBusId) (char * pciBusId, int  len, int  device) =
	(cudaError_t (*) (char * pciBusId, int  len, int  device)) dlsym(cudart_handle, "cudaDeviceGetPCIBusId");

cudaError_t (*lcudaIpcGetEventHandle) (cudaIpcEventHandle_t * handle, cudaEvent_t  event) =
	(cudaError_t (*) (cudaIpcEventHandle_t * handle, cudaEvent_t  event)) dlsym(cudart_handle, "cudaIpcGetEventHandle");

cudaError_t (*lcudaIpcOpenEventHandle) (cudaEvent_t * event, cudaIpcEventHandle_t  handle) =
	(cudaError_t (*) (cudaEvent_t * event, cudaIpcEventHandle_t  handle)) dlsym(cudart_handle, "cudaIpcOpenEventHandle");

cudaError_t (*lcudaIpcGetMemHandle) (cudaIpcMemHandle_t * handle, void * devPtr) =
	(cudaError_t (*) (cudaIpcMemHandle_t * handle, void * devPtr)) dlsym(cudart_handle, "cudaIpcGetMemHandle");

cudaError_t (*lcudaIpcOpenMemHandle) (void ** devPtr, cudaIpcMemHandle_t  handle, unsigned int  flags) =
	(cudaError_t (*) (void ** devPtr, cudaIpcMemHandle_t  handle, unsigned int  flags)) dlsym(cudart_handle, "cudaIpcOpenMemHandle");

cudaError_t (*lcudaIpcCloseMemHandle) (void * devPtr) =
	(cudaError_t (*) (void * devPtr)) dlsym(cudart_handle, "cudaIpcCloseMemHandle");

cudaError_t (*lcudaDeviceFlushGPUDirectRDMAWrites) (enum cudaFlushGPUDirectRDMAWritesTarget  target, enum cudaFlushGPUDirectRDMAWritesScope  scope) =
	(cudaError_t (*) (enum cudaFlushGPUDirectRDMAWritesTarget  target, enum cudaFlushGPUDirectRDMAWritesScope  scope)) dlsym(cudart_handle, "cudaDeviceFlushGPUDirectRDMAWrites");

cudaError_t (*lcudaThreadExit) () =
	(cudaError_t (*) ()) dlsym(cudart_handle, "cudaThreadExit");

cudaError_t (*lcudaThreadSynchronize) () =
	(cudaError_t (*) ()) dlsym(cudart_handle, "cudaThreadSynchronize");

cudaError_t (*lcudaThreadSetLimit) (enum cudaLimit  limit, size_t  value) =
	(cudaError_t (*) (enum cudaLimit  limit, size_t  value)) dlsym(cudart_handle, "cudaThreadSetLimit");

cudaError_t (*lcudaThreadGetLimit) (size_t * pValue, enum cudaLimit  limit) =
	(cudaError_t (*) (size_t * pValue, enum cudaLimit  limit)) dlsym(cudart_handle, "cudaThreadGetLimit");

cudaError_t (*lcudaThreadGetCacheConfig) (enum cudaFuncCache * pCacheConfig) =
	(cudaError_t (*) (enum cudaFuncCache * pCacheConfig)) dlsym(cudart_handle, "cudaThreadGetCacheConfig");

cudaError_t (*lcudaThreadSetCacheConfig) (enum cudaFuncCache  cacheConfig) =
	(cudaError_t (*) (enum cudaFuncCache  cacheConfig)) dlsym(cudart_handle, "cudaThreadSetCacheConfig");

cudaError_t (*lcudaGetLastError) () =
	(cudaError_t (*) ()) dlsym(cudart_handle, "cudaGetLastError");

cudaError_t (*lcudaPeekAtLastError) () =
	(cudaError_t (*) ()) dlsym(cudart_handle, "cudaPeekAtLastError");

const char* (*lcudaGetErrorName) (cudaError_t  error) =
	(const char* (*) (cudaError_t  error)) dlsym(RTLD_NEXT, "cudaGetErrorName");

const char* (*lcudaGetErrorString) (cudaError_t  error) =
	(const char* (*) (cudaError_t  error)) dlsym(RTLD_NEXT, "cudaGetErrorString");

cudaError_t (*lcudaGetDeviceCount) (int * count) =
	(cudaError_t (*) (int * count)) dlsym(cudart_handle, "cudaGetDeviceCount");

cudaError_t (*lcudaGetDeviceProperties_v2) (struct cudaDeviceProp * prop, int  device) =
	(cudaError_t (*) (struct cudaDeviceProp * prop, int  device)) dlsym(cudart_handle, "cudaGetDeviceProperties_v2");

cudaError_t (*lcudaDeviceGetAttribute) (int * value, enum cudaDeviceAttr  attr, int  device) =
	(cudaError_t (*) (int * value, enum cudaDeviceAttr  attr, int  device)) dlsym(cudart_handle, "cudaDeviceGetAttribute");

cudaError_t (*lcudaDeviceGetDefaultMemPool) (cudaMemPool_t * memPool, int  device) =
	(cudaError_t (*) (cudaMemPool_t * memPool, int  device)) dlsym(cudart_handle, "cudaDeviceGetDefaultMemPool");

cudaError_t (*lcudaDeviceSetMemPool) (int  device, cudaMemPool_t  memPool) =
	(cudaError_t (*) (int  device, cudaMemPool_t  memPool)) dlsym(cudart_handle, "cudaDeviceSetMemPool");

cudaError_t (*lcudaDeviceGetMemPool) (cudaMemPool_t * memPool, int  device) =
	(cudaError_t (*) (cudaMemPool_t * memPool, int  device)) dlsym(cudart_handle, "cudaDeviceGetMemPool");

cudaError_t (*lcudaDeviceGetNvSciSyncAttributes) (void * nvSciSyncAttrList, int  device, int  flags) =
	(cudaError_t (*) (void * nvSciSyncAttrList, int  device, int  flags)) dlsym(cudart_handle, "cudaDeviceGetNvSciSyncAttributes");

cudaError_t (*lcudaDeviceGetP2PAttribute) (int * value, enum cudaDeviceP2PAttr  attr, int  srcDevice, int  dstDevice) =
	(cudaError_t (*) (int * value, enum cudaDeviceP2PAttr  attr, int  srcDevice, int  dstDevice)) dlsym(cudart_handle, "cudaDeviceGetP2PAttribute");

cudaError_t (*lcudaChooseDevice) (int * device, const struct cudaDeviceProp * prop) =
	(cudaError_t (*) (int * device, const struct cudaDeviceProp * prop)) dlsym(cudart_handle, "cudaChooseDevice");

cudaError_t (*lcudaInitDevice) (int  device, unsigned int  deviceFlags, unsigned int  flags) =
	(cudaError_t (*) (int  device, unsigned int  deviceFlags, unsigned int  flags)) dlsym(cudart_handle, "cudaInitDevice");

cudaError_t (*lcudaSetDevice) (int  device) =
	(cudaError_t (*) (int  device)) dlsym(cudart_handle, "cudaSetDevice");

cudaError_t (*lcudaGetDevice) (int * device) =
	(cudaError_t (*) (int * device)) dlsym(cudart_handle, "cudaGetDevice");

cudaError_t (*lcudaSetValidDevices) (int * device_arr, int  len) =
	(cudaError_t (*) (int * device_arr, int  len)) dlsym(cudart_handle, "cudaSetValidDevices");

cudaError_t (*lcudaSetDeviceFlags) (unsigned int  flags) =
	(cudaError_t (*) (unsigned int  flags)) dlsym(cudart_handle, "cudaSetDeviceFlags");

cudaError_t (*lcudaGetDeviceFlags) (unsigned int * flags) =
	(cudaError_t (*) (unsigned int * flags)) dlsym(cudart_handle, "cudaGetDeviceFlags");

cudaError_t (*lcudaStreamCreate) (cudaStream_t * pStream) =
	(cudaError_t (*) (cudaStream_t * pStream)) dlsym(cudart_handle, "cudaStreamCreate");

cudaError_t (*lcudaStreamCreateWithFlags) (cudaStream_t * pStream, unsigned int  flags) =
	(cudaError_t (*) (cudaStream_t * pStream, unsigned int  flags)) dlsym(cudart_handle, "cudaStreamCreateWithFlags");

cudaError_t (*lcudaStreamCreateWithPriority) (cudaStream_t * pStream, unsigned int  flags, int  priority) =
	(cudaError_t (*) (cudaStream_t * pStream, unsigned int  flags, int  priority)) dlsym(cudart_handle, "cudaStreamCreateWithPriority");

cudaError_t (*lcudaStreamGetPriority) (cudaStream_t  hStream, int * priority) =
	(cudaError_t (*) (cudaStream_t  hStream, int * priority)) dlsym(cudart_handle, "cudaStreamGetPriority");

cudaError_t (*lcudaStreamGetFlags) (cudaStream_t  hStream, unsigned int * flags) =
	(cudaError_t (*) (cudaStream_t  hStream, unsigned int * flags)) dlsym(cudart_handle, "cudaStreamGetFlags");

cudaError_t (*lcudaStreamGetId) (cudaStream_t  hStream, unsigned long long * streamId) =
	(cudaError_t (*) (cudaStream_t  hStream, unsigned long long * streamId)) dlsym(cudart_handle, "cudaStreamGetId");

cudaError_t (*lcudaCtxResetPersistingL2Cache) () =
	(cudaError_t (*) ()) dlsym(cudart_handle, "cudaCtxResetPersistingL2Cache");

cudaError_t (*lcudaStreamCopyAttributes) (cudaStream_t  dst, cudaStream_t  src) =
	(cudaError_t (*) (cudaStream_t  dst, cudaStream_t  src)) dlsym(cudart_handle, "cudaStreamCopyAttributes");

cudaError_t (*lcudaStreamGetAttribute) (cudaStream_t  hStream, cudaLaunchAttributeID  attr, cudaLaunchAttributeValue * value_out) =
	(cudaError_t (*) (cudaStream_t  hStream, cudaLaunchAttributeID  attr, cudaLaunchAttributeValue * value_out)) dlsym(cudart_handle, "cudaStreamGetAttribute");

cudaError_t (*lcudaStreamSetAttribute) (cudaStream_t  hStream, cudaLaunchAttributeID  attr, const cudaLaunchAttributeValue * value) =
	(cudaError_t (*) (cudaStream_t  hStream, cudaLaunchAttributeID  attr, const cudaLaunchAttributeValue * value)) dlsym(cudart_handle, "cudaStreamSetAttribute");

cudaError_t (*lcudaStreamDestroy) (cudaStream_t  stream) =
	(cudaError_t (*) (cudaStream_t  stream)) dlsym(cudart_handle, "cudaStreamDestroy");

cudaError_t (*lcudaStreamWaitEvent) (cudaStream_t  stream, cudaEvent_t  event, unsigned int  flags) =
	(cudaError_t (*) (cudaStream_t  stream, cudaEvent_t  event, unsigned int  flags)) dlsym(cudart_handle, "cudaStreamWaitEvent");

cudaError_t (*lcudaStreamAddCallback) (cudaStream_t  stream, cudaStreamCallback_t  callback, void * userData, unsigned int  flags) =
	(cudaError_t (*) (cudaStream_t  stream, cudaStreamCallback_t  callback, void * userData, unsigned int  flags)) dlsym(cudart_handle, "cudaStreamAddCallback");

cudaError_t (*lcudaStreamSynchronize) (cudaStream_t  stream) =
	(cudaError_t (*) (cudaStream_t  stream)) dlsym(cudart_handle, "cudaStreamSynchronize");

cudaError_t (*lcudaStreamQuery) (cudaStream_t  stream) =
	(cudaError_t (*) (cudaStream_t  stream)) dlsym(cudart_handle, "cudaStreamQuery");

cudaError_t (*lcudaStreamAttachMemAsync) (cudaStream_t  stream, void * devPtr, size_t  length, unsigned int  flags) =
	(cudaError_t (*) (cudaStream_t  stream, void * devPtr, size_t  length, unsigned int  flags)) dlsym(cudart_handle, "cudaStreamAttachMemAsync");

cudaError_t (*lcudaStreamBeginCapture) (cudaStream_t  stream, enum cudaStreamCaptureMode  mode) =
	(cudaError_t (*) (cudaStream_t  stream, enum cudaStreamCaptureMode  mode)) dlsym(cudart_handle, "cudaStreamBeginCapture");

cudaError_t (*lcudaThreadExchangeStreamCaptureMode) (enum cudaStreamCaptureMode * mode) =
	(cudaError_t (*) (enum cudaStreamCaptureMode * mode)) dlsym(cudart_handle, "cudaThreadExchangeStreamCaptureMode");

cudaError_t (*lcudaStreamEndCapture) (cudaStream_t  stream, cudaGraph_t * pGraph) =
	(cudaError_t (*) (cudaStream_t  stream, cudaGraph_t * pGraph)) dlsym(cudart_handle, "cudaStreamEndCapture");

cudaError_t (*lcudaStreamIsCapturing) (cudaStream_t  stream, enum cudaStreamCaptureStatus * pCaptureStatus) =
	(cudaError_t (*) (cudaStream_t  stream, enum cudaStreamCaptureStatus * pCaptureStatus)) dlsym(cudart_handle, "cudaStreamIsCapturing");

cudaError_t (*lcudaStreamGetCaptureInfo_v2) (cudaStream_t  stream, enum cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, cudaGraph_t * graph_out, const cudaGraphNode_t ** dependencies_out, size_t * numDependencies_out) =
	(cudaError_t (*) (cudaStream_t  stream, enum cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, cudaGraph_t * graph_out, const cudaGraphNode_t ** dependencies_out, size_t * numDependencies_out)) dlsym(cudart_handle, "cudaStreamGetCaptureInfo_v2");

cudaError_t (*lcudaStreamUpdateCaptureDependencies) (cudaStream_t  stream, cudaGraphNode_t * dependencies, size_t  numDependencies, unsigned int  flags) =
	(cudaError_t (*) (cudaStream_t  stream, cudaGraphNode_t * dependencies, size_t  numDependencies, unsigned int  flags)) dlsym(cudart_handle, "cudaStreamUpdateCaptureDependencies");

cudaError_t (*lcudaEventCreate) (cudaEvent_t * event) =
	(cudaError_t (*) (cudaEvent_t * event)) dlsym(cudart_handle, "cudaEventCreate");

cudaError_t (*lcudaEventCreateWithFlags) (cudaEvent_t * event, unsigned int  flags) =
	(cudaError_t (*) (cudaEvent_t * event, unsigned int  flags)) dlsym(cudart_handle, "cudaEventCreateWithFlags");

cudaError_t (*lcudaEventRecord) (cudaEvent_t  event, cudaStream_t  stream) =
	(cudaError_t (*) (cudaEvent_t  event, cudaStream_t  stream)) dlsym(cudart_handle, "cudaEventRecord");

cudaError_t (*lcudaEventRecordWithFlags) (cudaEvent_t  event, cudaStream_t  stream, unsigned int  flags) =
	(cudaError_t (*) (cudaEvent_t  event, cudaStream_t  stream, unsigned int  flags)) dlsym(cudart_handle, "cudaEventRecordWithFlags");

cudaError_t (*lcudaEventQuery) (cudaEvent_t  event) =
	(cudaError_t (*) (cudaEvent_t  event)) dlsym(cudart_handle, "cudaEventQuery");

cudaError_t (*lcudaEventSynchronize) (cudaEvent_t  event) =
	(cudaError_t (*) (cudaEvent_t  event)) dlsym(cudart_handle, "cudaEventSynchronize");

cudaError_t (*lcudaEventDestroy) (cudaEvent_t  event) =
	(cudaError_t (*) (cudaEvent_t  event)) dlsym(cudart_handle, "cudaEventDestroy");

cudaError_t (*lcudaEventElapsedTime) (float * ms, cudaEvent_t  start, cudaEvent_t  end) =
	(cudaError_t (*) (float * ms, cudaEvent_t  start, cudaEvent_t  end)) dlsym(cudart_handle, "cudaEventElapsedTime");

cudaError_t (*lcudaImportExternalMemory) (cudaExternalMemory_t * extMem_out, const struct cudaExternalMemoryHandleDesc * memHandleDesc) =
	(cudaError_t (*) (cudaExternalMemory_t * extMem_out, const struct cudaExternalMemoryHandleDesc * memHandleDesc)) dlsym(cudart_handle, "cudaImportExternalMemory");

cudaError_t (*lcudaExternalMemoryGetMappedBuffer) (void ** devPtr, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryBufferDesc * bufferDesc) =
	(cudaError_t (*) (void ** devPtr, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryBufferDesc * bufferDesc)) dlsym(cudart_handle, "cudaExternalMemoryGetMappedBuffer");

cudaError_t (*lcudaExternalMemoryGetMappedMipmappedArray) (cudaMipmappedArray_t * mipmap, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryMipmappedArrayDesc * mipmapDesc) =
	(cudaError_t (*) (cudaMipmappedArray_t * mipmap, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryMipmappedArrayDesc * mipmapDesc)) dlsym(cudart_handle, "cudaExternalMemoryGetMappedMipmappedArray");

cudaError_t (*lcudaDestroyExternalMemory) (cudaExternalMemory_t  extMem) =
	(cudaError_t (*) (cudaExternalMemory_t  extMem)) dlsym(cudart_handle, "cudaDestroyExternalMemory");

cudaError_t (*lcudaImportExternalSemaphore) (cudaExternalSemaphore_t * extSem_out, const struct cudaExternalSemaphoreHandleDesc * semHandleDesc) =
	(cudaError_t (*) (cudaExternalSemaphore_t * extSem_out, const struct cudaExternalSemaphoreHandleDesc * semHandleDesc)) dlsym(cudart_handle, "cudaImportExternalSemaphore");

cudaError_t (*lcudaSignalExternalSemaphoresAsync_v2) (const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreSignalParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream) =
	(cudaError_t (*) (const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreSignalParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream)) dlsym(cudart_handle, "cudaSignalExternalSemaphoresAsync_v2");

cudaError_t (*lcudaWaitExternalSemaphoresAsync_v2) (const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreWaitParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream) =
	(cudaError_t (*) (const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreWaitParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream)) dlsym(cudart_handle, "cudaWaitExternalSemaphoresAsync_v2");

cudaError_t (*lcudaDestroyExternalSemaphore) (cudaExternalSemaphore_t  extSem) =
	(cudaError_t (*) (cudaExternalSemaphore_t  extSem)) dlsym(cudart_handle, "cudaDestroyExternalSemaphore");

cudaError_t (*lcudaLaunchKernel) (const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream) =
	(cudaError_t (*) (const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)) dlsym(cudart_handle, "cudaLaunchKernel");

cudaError_t (*lcudaLaunchKernelExC) (const cudaLaunchConfig_t * config, const void * func, void ** args) =
	(cudaError_t (*) (const cudaLaunchConfig_t * config, const void * func, void ** args)) dlsym(cudart_handle, "cudaLaunchKernelExC");

cudaError_t (*lcudaLaunchCooperativeKernel) (const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream) =
	(cudaError_t (*) (const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)) dlsym(cudart_handle, "cudaLaunchCooperativeKernel");

cudaError_t (*lcudaLaunchCooperativeKernelMultiDevice) (struct cudaLaunchParams * launchParamsList, unsigned int  numDevices, unsigned int  flags) =
	(cudaError_t (*) (struct cudaLaunchParams * launchParamsList, unsigned int  numDevices, unsigned int  flags)) dlsym(cudart_handle, "cudaLaunchCooperativeKernelMultiDevice");

cudaError_t (*lcudaFuncSetCacheConfig) (const void * func, enum cudaFuncCache  cacheConfig) =
	(cudaError_t (*) (const void * func, enum cudaFuncCache  cacheConfig)) dlsym(cudart_handle, "cudaFuncSetCacheConfig");

cudaError_t (*lcudaFuncSetSharedMemConfig) (const void * func, enum cudaSharedMemConfig  config) =
	(cudaError_t (*) (const void * func, enum cudaSharedMemConfig  config)) dlsym(cudart_handle, "cudaFuncSetSharedMemConfig");

cudaError_t (*lcudaFuncGetAttributes) (struct cudaFuncAttributes * attr, const void * func) =
	(cudaError_t (*) (struct cudaFuncAttributes * attr, const void * func)) dlsym(cudart_handle, "cudaFuncGetAttributes");

cudaError_t (*lcudaFuncSetAttribute) (const void * func, enum cudaFuncAttribute  attr, int  value) =
	(cudaError_t (*) (const void * func, enum cudaFuncAttribute  attr, int  value)) dlsym(cudart_handle, "cudaFuncSetAttribute");

cudaError_t (*lcudaSetDoubleForDevice) (double * d) =
	(cudaError_t (*) (double * d)) dlsym(cudart_handle, "cudaSetDoubleForDevice");

cudaError_t (*lcudaSetDoubleForHost) (double * d) =
	(cudaError_t (*) (double * d)) dlsym(cudart_handle, "cudaSetDoubleForHost");

cudaError_t (*lcudaLaunchHostFunc) (cudaStream_t  stream, cudaHostFn_t  fn, void * userData) =
	(cudaError_t (*) (cudaStream_t  stream, cudaHostFn_t  fn, void * userData)) dlsym(cudart_handle, "cudaLaunchHostFunc");

cudaError_t (*lcudaOccupancyMaxActiveBlocksPerMultiprocessor) (int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize) =
	(cudaError_t (*) (int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize)) dlsym(cudart_handle, "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

cudaError_t (*lcudaOccupancyAvailableDynamicSMemPerBlock) (size_t * dynamicSmemSize, const void * func, int  numBlocks, int  blockSize) =
	(cudaError_t (*) (size_t * dynamicSmemSize, const void * func, int  numBlocks, int  blockSize)) dlsym(cudart_handle, "cudaOccupancyAvailableDynamicSMemPerBlock");

cudaError_t (*lcudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) (int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags) =
	(cudaError_t (*) (int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)) dlsym(cudart_handle, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");

cudaError_t (*lcudaOccupancyMaxPotentialClusterSize) (int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig) =
	(cudaError_t (*) (int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig)) dlsym(cudart_handle, "cudaOccupancyMaxPotentialClusterSize");

cudaError_t (*lcudaOccupancyMaxActiveClusters) (int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig) =
	(cudaError_t (*) (int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig)) dlsym(cudart_handle, "cudaOccupancyMaxActiveClusters");

cudaError_t (*lcudaMallocManaged) (void ** devPtr, size_t  size, unsigned int  flags) =
	(cudaError_t (*) (void ** devPtr, size_t  size, unsigned int  flags)) dlsym(cudart_handle, "cudaMallocManaged");

cudaError_t (*lcudaMalloc) (void ** devPtr, size_t  size) =
	(cudaError_t (*) (void ** devPtr, size_t  size)) dlsym(cudart_handle, "cudaMalloc");

cudaError_t (*lcudaMallocHost) (void ** ptr, size_t  size) =
	(cudaError_t (*) (void ** ptr, size_t  size)) dlsym(cudart_handle, "cudaMallocHost");

cudaError_t (*lcudaMallocPitch) (void ** devPtr, size_t * pitch, size_t  width, size_t  height) =
	(cudaError_t (*) (void ** devPtr, size_t * pitch, size_t  width, size_t  height)) dlsym(cudart_handle, "cudaMallocPitch");

cudaError_t (*lcudaMallocArray) (cudaArray_t * array, const struct cudaChannelFormatDesc * desc, size_t  width, size_t  height, unsigned int  flags) =
	(cudaError_t (*) (cudaArray_t * array, const struct cudaChannelFormatDesc * desc, size_t  width, size_t  height, unsigned int  flags)) dlsym(cudart_handle, "cudaMallocArray");

cudaError_t (*lcudaFree) (void * devPtr) =
	(cudaError_t (*) (void * devPtr)) dlsym(cudart_handle, "cudaFree");

cudaError_t (*lcudaFreeHost) (void * ptr) =
	(cudaError_t (*) (void * ptr)) dlsym(cudart_handle, "cudaFreeHost");

cudaError_t (*lcudaFreeArray) (cudaArray_t  array) =
	(cudaError_t (*) (cudaArray_t  array)) dlsym(cudart_handle, "cudaFreeArray");

cudaError_t (*lcudaFreeMipmappedArray) (cudaMipmappedArray_t  mipmappedArray) =
	(cudaError_t (*) (cudaMipmappedArray_t  mipmappedArray)) dlsym(cudart_handle, "cudaFreeMipmappedArray");

cudaError_t (*lcudaHostAlloc) (void ** pHost, size_t  size, unsigned int  flags) =
	(cudaError_t (*) (void ** pHost, size_t  size, unsigned int  flags)) dlsym(cudart_handle, "cudaHostAlloc");

cudaError_t (*lcudaHostRegister) (void * ptr, size_t  size, unsigned int  flags) =
	(cudaError_t (*) (void * ptr, size_t  size, unsigned int  flags)) dlsym(cudart_handle, "cudaHostRegister");

cudaError_t (*lcudaHostUnregister) (void * ptr) =
	(cudaError_t (*) (void * ptr)) dlsym(cudart_handle, "cudaHostUnregister");

cudaError_t (*lcudaHostGetDevicePointer) (void ** pDevice, void * pHost, unsigned int  flags) =
	(cudaError_t (*) (void ** pDevice, void * pHost, unsigned int  flags)) dlsym(cudart_handle, "cudaHostGetDevicePointer");

cudaError_t (*lcudaHostGetFlags) (unsigned int * pFlags, void * pHost) =
	(cudaError_t (*) (unsigned int * pFlags, void * pHost)) dlsym(cudart_handle, "cudaHostGetFlags");

cudaError_t (*lcudaMalloc3D) (struct cudaPitchedPtr*  pitchedDevPtr, struct cudaExtent  extent) =
	(cudaError_t (*) (struct cudaPitchedPtr*  pitchedDevPtr, struct cudaExtent  extent)) dlsym(cudart_handle, "cudaMalloc3D");

cudaError_t (*lcudaMalloc3DArray) (cudaArray_t * array, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  flags) =
	(cudaError_t (*) (cudaArray_t * array, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  flags)) dlsym(cudart_handle, "cudaMalloc3DArray");

cudaError_t (*lcudaMallocMipmappedArray) (cudaMipmappedArray_t * mipmappedArray, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  numLevels, unsigned int  flags) =
	(cudaError_t (*) (cudaMipmappedArray_t * mipmappedArray, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  numLevels, unsigned int  flags)) dlsym(cudart_handle, "cudaMallocMipmappedArray");

cudaError_t (*lcudaGetMipmappedArrayLevel) (cudaArray_t * levelArray, cudaMipmappedArray_const_t  mipmappedArray, unsigned int  level) =
	(cudaError_t (*) (cudaArray_t * levelArray, cudaMipmappedArray_const_t  mipmappedArray, unsigned int  level)) dlsym(cudart_handle, "cudaGetMipmappedArrayLevel");

cudaError_t (*lcudaMemcpy3D) (const struct cudaMemcpy3DParms * p) =
	(cudaError_t (*) (const struct cudaMemcpy3DParms * p)) dlsym(cudart_handle, "cudaMemcpy3D");

cudaError_t (*lcudaMemcpy3DPeer) (const struct cudaMemcpy3DPeerParms * p) =
	(cudaError_t (*) (const struct cudaMemcpy3DPeerParms * p)) dlsym(cudart_handle, "cudaMemcpy3DPeer");

cudaError_t (*lcudaMemcpy3DAsync) (const struct cudaMemcpy3DParms * p, cudaStream_t  stream) =
	(cudaError_t (*) (const struct cudaMemcpy3DParms * p, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpy3DAsync");

cudaError_t (*lcudaMemcpy3DPeerAsync) (const struct cudaMemcpy3DPeerParms * p, cudaStream_t  stream) =
	(cudaError_t (*) (const struct cudaMemcpy3DPeerParms * p, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpy3DPeerAsync");

cudaError_t (*lcudaMemGetInfo) (size_t * free, size_t * total) =
	(cudaError_t (*) (size_t * free, size_t * total)) dlsym(cudart_handle, "cudaMemGetInfo");

cudaError_t (*lcudaArrayGetInfo) (struct cudaChannelFormatDesc * desc, struct cudaExtent * extent, unsigned int * flags, cudaArray_t  array) =
	(cudaError_t (*) (struct cudaChannelFormatDesc * desc, struct cudaExtent * extent, unsigned int * flags, cudaArray_t  array)) dlsym(cudart_handle, "cudaArrayGetInfo");

cudaError_t (*lcudaArrayGetPlane) (cudaArray_t * pPlaneArray, cudaArray_t  hArray, unsigned int  planeIdx) =
	(cudaError_t (*) (cudaArray_t * pPlaneArray, cudaArray_t  hArray, unsigned int  planeIdx)) dlsym(cudart_handle, "cudaArrayGetPlane");

cudaError_t (*lcudaArrayGetMemoryRequirements) (struct cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t  array, int  device) =
	(cudaError_t (*) (struct cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t  array, int  device)) dlsym(cudart_handle, "cudaArrayGetMemoryRequirements");

cudaError_t (*lcudaMipmappedArrayGetMemoryRequirements) (struct cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t  mipmap, int  device) =
	(cudaError_t (*) (struct cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t  mipmap, int  device)) dlsym(cudart_handle, "cudaMipmappedArrayGetMemoryRequirements");

cudaError_t (*lcudaArrayGetSparseProperties) (struct cudaArraySparseProperties * sparseProperties, cudaArray_t  array) =
	(cudaError_t (*) (struct cudaArraySparseProperties * sparseProperties, cudaArray_t  array)) dlsym(cudart_handle, "cudaArrayGetSparseProperties");

cudaError_t (*lcudaMipmappedArrayGetSparseProperties) (struct cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t  mipmap) =
	(cudaError_t (*) (struct cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t  mipmap)) dlsym(cudart_handle, "cudaMipmappedArrayGetSparseProperties");

cudaError_t (*lcudaMemcpy) (void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpy");

cudaError_t (*lcudaMemcpyPeer) (void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count) =
	(cudaError_t (*) (void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count)) dlsym(cudart_handle, "cudaMemcpyPeer");

cudaError_t (*lcudaMemcpy2D) (void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpy2D");

cudaError_t (*lcudaMemcpy2DToArray) (cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpy2DToArray");

cudaError_t (*lcudaMemcpy2DFromArray) (void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpy2DFromArray");

cudaError_t (*lcudaMemcpy2DArrayToArray) (cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  width, size_t  height, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  width, size_t  height, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpy2DArrayToArray");

cudaError_t (*lcudaMemcpyToSymbol) (const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpyToSymbol");

cudaError_t (*lcudaMemcpyFromSymbol) (void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpyFromSymbol");

cudaError_t (*lcudaMemcpyAsync) (void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream) =
	(cudaError_t (*) (void * dst, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpyAsync");

cudaError_t (*lcudaMemcpyPeerAsync) (void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count, cudaStream_t  stream) =
	(cudaError_t (*) (void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpyPeerAsync");

cudaError_t (*lcudaMemcpy2DAsync) (void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream) =
	(cudaError_t (*) (void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpy2DAsync");

cudaError_t (*lcudaMemcpy2DToArrayAsync) (cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream) =
	(cudaError_t (*) (cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpy2DToArrayAsync");

cudaError_t (*lcudaMemcpy2DFromArrayAsync) (void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream) =
	(cudaError_t (*) (void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpy2DFromArrayAsync");

cudaError_t (*lcudaMemcpyToSymbolAsync) (const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream) =
	(cudaError_t (*) (const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpyToSymbolAsync");

cudaError_t (*lcudaMemcpyFromSymbolAsync) (void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream) =
	(cudaError_t (*) (void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpyFromSymbolAsync");

cudaError_t (*lcudaMemset) (void * devPtr, int  value, size_t  count) =
	(cudaError_t (*) (void * devPtr, int  value, size_t  count)) dlsym(cudart_handle, "cudaMemset");

cudaError_t (*lcudaMemset2D) (void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height) =
	(cudaError_t (*) (void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height)) dlsym(cudart_handle, "cudaMemset2D");

cudaError_t (*lcudaMemset3D) (struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent) =
	(cudaError_t (*) (struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent)) dlsym(cudart_handle, "cudaMemset3D");

cudaError_t (*lcudaMemsetAsync) (void * devPtr, int  value, size_t  count, cudaStream_t  stream) =
	(cudaError_t (*) (void * devPtr, int  value, size_t  count, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemsetAsync");

cudaError_t (*lcudaMemset2DAsync) (void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height, cudaStream_t  stream) =
	(cudaError_t (*) (void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemset2DAsync");

cudaError_t (*lcudaMemset3DAsync) (struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent, cudaStream_t  stream) =
	(cudaError_t (*) (struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemset3DAsync");

cudaError_t (*lcudaGetSymbolAddress) (void ** devPtr, const void * symbol) =
	(cudaError_t (*) (void ** devPtr, const void * symbol)) dlsym(cudart_handle, "cudaGetSymbolAddress");

cudaError_t (*lcudaGetSymbolSize) (size_t * size, const void * symbol) =
	(cudaError_t (*) (size_t * size, const void * symbol)) dlsym(cudart_handle, "cudaGetSymbolSize");

cudaError_t (*lcudaMemPrefetchAsync) (const void * devPtr, size_t  count, int  dstDevice, cudaStream_t  stream) =
	(cudaError_t (*) (const void * devPtr, size_t  count, int  dstDevice, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemPrefetchAsync");

cudaError_t (*lcudaMemPrefetchAsync_v2) (const void * devPtr, size_t  count, struct cudaMemLocation  location, unsigned int  flags, cudaStream_t  stream) =
	(cudaError_t (*) (const void * devPtr, size_t  count, struct cudaMemLocation  location, unsigned int  flags, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemPrefetchAsync_v2");

cudaError_t (*lcudaMemAdvise) (const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, int  device) =
	(cudaError_t (*) (const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, int  device)) dlsym(cudart_handle, "cudaMemAdvise");

cudaError_t (*lcudaMemAdvise_v2) (const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, struct cudaMemLocation  location) =
	(cudaError_t (*) (const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, struct cudaMemLocation  location)) dlsym(cudart_handle, "cudaMemAdvise_v2");

cudaError_t (*lcudaMemRangeGetAttribute) (void * data, size_t  dataSize, enum cudaMemRangeAttribute  attribute, const void * devPtr, size_t  count) =
	(cudaError_t (*) (void * data, size_t  dataSize, enum cudaMemRangeAttribute  attribute, const void * devPtr, size_t  count)) dlsym(cudart_handle, "cudaMemRangeGetAttribute");

cudaError_t (*lcudaMemRangeGetAttributes) (void ** data, size_t * dataSizes, enum cudaMemRangeAttribute * attributes, size_t  numAttributes, const void * devPtr, size_t  count) =
	(cudaError_t (*) (void ** data, size_t * dataSizes, enum cudaMemRangeAttribute * attributes, size_t  numAttributes, const void * devPtr, size_t  count)) dlsym(cudart_handle, "cudaMemRangeGetAttributes");

cudaError_t (*lcudaMemcpyToArray) (cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpyToArray");

cudaError_t (*lcudaMemcpyFromArray) (void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpyFromArray");

cudaError_t (*lcudaMemcpyArrayToArray) (cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  count, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  count, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaMemcpyArrayToArray");

cudaError_t (*lcudaMemcpyToArrayAsync) (cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream) =
	(cudaError_t (*) (cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpyToArrayAsync");

cudaError_t (*lcudaMemcpyFromArrayAsync) (void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream) =
	(cudaError_t (*) (void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMemcpyFromArrayAsync");

cudaError_t (*lcudaMallocAsync) (void ** devPtr, size_t  size, cudaStream_t  hStream) =
	(cudaError_t (*) (void ** devPtr, size_t  size, cudaStream_t  hStream)) dlsym(cudart_handle, "cudaMallocAsync");

cudaError_t (*lcudaFreeAsync) (void * devPtr, cudaStream_t  hStream) =
	(cudaError_t (*) (void * devPtr, cudaStream_t  hStream)) dlsym(cudart_handle, "cudaFreeAsync");

cudaError_t (*lcudaMemPoolTrimTo) (cudaMemPool_t  memPool, size_t  minBytesToKeep) =
	(cudaError_t (*) (cudaMemPool_t  memPool, size_t  minBytesToKeep)) dlsym(cudart_handle, "cudaMemPoolTrimTo");

cudaError_t (*lcudaMemPoolSetAttribute) (cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value) =
	(cudaError_t (*) (cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value)) dlsym(cudart_handle, "cudaMemPoolSetAttribute");

cudaError_t (*lcudaMemPoolGetAttribute) (cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value) =
	(cudaError_t (*) (cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value)) dlsym(cudart_handle, "cudaMemPoolGetAttribute");

cudaError_t (*lcudaMemPoolSetAccess) (cudaMemPool_t  memPool, const struct cudaMemAccessDesc * descList, size_t  count) =
	(cudaError_t (*) (cudaMemPool_t  memPool, const struct cudaMemAccessDesc * descList, size_t  count)) dlsym(cudart_handle, "cudaMemPoolSetAccess");

cudaError_t (*lcudaMemPoolGetAccess) (enum cudaMemAccessFlags * flags, cudaMemPool_t  memPool, struct cudaMemLocation * location) =
	(cudaError_t (*) (enum cudaMemAccessFlags * flags, cudaMemPool_t  memPool, struct cudaMemLocation * location)) dlsym(cudart_handle, "cudaMemPoolGetAccess");

cudaError_t (*lcudaMemPoolCreate) (cudaMemPool_t * memPool, const struct cudaMemPoolProps * poolProps) =
	(cudaError_t (*) (cudaMemPool_t * memPool, const struct cudaMemPoolProps * poolProps)) dlsym(cudart_handle, "cudaMemPoolCreate");

cudaError_t (*lcudaMemPoolDestroy) (cudaMemPool_t  memPool) =
	(cudaError_t (*) (cudaMemPool_t  memPool)) dlsym(cudart_handle, "cudaMemPoolDestroy");

cudaError_t (*lcudaMallocFromPoolAsync) (void ** ptr, size_t  size, cudaMemPool_t  memPool, cudaStream_t  stream) =
	(cudaError_t (*) (void ** ptr, size_t  size, cudaMemPool_t  memPool, cudaStream_t  stream)) dlsym(cudart_handle, "cudaMallocFromPoolAsync");

cudaError_t (*lcudaMemPoolExportToShareableHandle) (void * shareableHandle, cudaMemPool_t  memPool, enum cudaMemAllocationHandleType  handleType, unsigned int  flags) =
	(cudaError_t (*) (void * shareableHandle, cudaMemPool_t  memPool, enum cudaMemAllocationHandleType  handleType, unsigned int  flags)) dlsym(cudart_handle, "cudaMemPoolExportToShareableHandle");

cudaError_t (*lcudaMemPoolImportFromShareableHandle) (cudaMemPool_t * memPool, void * shareableHandle, enum cudaMemAllocationHandleType  handleType, unsigned int  flags) =
	(cudaError_t (*) (cudaMemPool_t * memPool, void * shareableHandle, enum cudaMemAllocationHandleType  handleType, unsigned int  flags)) dlsym(cudart_handle, "cudaMemPoolImportFromShareableHandle");

cudaError_t (*lcudaMemPoolExportPointer) (struct cudaMemPoolPtrExportData * exportData, void * ptr) =
	(cudaError_t (*) (struct cudaMemPoolPtrExportData * exportData, void * ptr)) dlsym(cudart_handle, "cudaMemPoolExportPointer");

cudaError_t (*lcudaMemPoolImportPointer) (void ** ptr, cudaMemPool_t  memPool, struct cudaMemPoolPtrExportData * exportData) =
	(cudaError_t (*) (void ** ptr, cudaMemPool_t  memPool, struct cudaMemPoolPtrExportData * exportData)) dlsym(cudart_handle, "cudaMemPoolImportPointer");

cudaError_t (*lcudaPointerGetAttributes) (struct cudaPointerAttributes * attributes, const void * ptr) =
	(cudaError_t (*) (struct cudaPointerAttributes * attributes, const void * ptr)) dlsym(cudart_handle, "cudaPointerGetAttributes");

cudaError_t (*lcudaDeviceCanAccessPeer) (int * canAccessPeer, int  device, int  peerDevice) =
	(cudaError_t (*) (int * canAccessPeer, int  device, int  peerDevice)) dlsym(cudart_handle, "cudaDeviceCanAccessPeer");

cudaError_t (*lcudaDeviceEnablePeerAccess) (int  peerDevice, unsigned int  flags) =
	(cudaError_t (*) (int  peerDevice, unsigned int  flags)) dlsym(cudart_handle, "cudaDeviceEnablePeerAccess");

cudaError_t (*lcudaDeviceDisablePeerAccess) (int  peerDevice) =
	(cudaError_t (*) (int  peerDevice)) dlsym(cudart_handle, "cudaDeviceDisablePeerAccess");

cudaError_t (*lcudaGraphicsUnregisterResource) (cudaGraphicsResource_t  resource) =
	(cudaError_t (*) (cudaGraphicsResource_t  resource)) dlsym(cudart_handle, "cudaGraphicsUnregisterResource");

cudaError_t (*lcudaGraphicsResourceSetMapFlags) (cudaGraphicsResource_t  resource, unsigned int  flags) =
	(cudaError_t (*) (cudaGraphicsResource_t  resource, unsigned int  flags)) dlsym(cudart_handle, "cudaGraphicsResourceSetMapFlags");

cudaError_t (*lcudaGraphicsMapResources) (int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream) =
	(cudaError_t (*) (int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream)) dlsym(cudart_handle, "cudaGraphicsMapResources");

cudaError_t (*lcudaGraphicsUnmapResources) (int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream) =
	(cudaError_t (*) (int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream)) dlsym(cudart_handle, "cudaGraphicsUnmapResources");

cudaError_t (*lcudaGraphicsResourceGetMappedPointer) (void ** devPtr, size_t * size, cudaGraphicsResource_t  resource) =
	(cudaError_t (*) (void ** devPtr, size_t * size, cudaGraphicsResource_t  resource)) dlsym(cudart_handle, "cudaGraphicsResourceGetMappedPointer");

cudaError_t (*lcudaGraphicsSubResourceGetMappedArray) (cudaArray_t * array, cudaGraphicsResource_t  resource, unsigned int  arrayIndex, unsigned int  mipLevel) =
	(cudaError_t (*) (cudaArray_t * array, cudaGraphicsResource_t  resource, unsigned int  arrayIndex, unsigned int  mipLevel)) dlsym(cudart_handle, "cudaGraphicsSubResourceGetMappedArray");

cudaError_t (*lcudaGraphicsResourceGetMappedMipmappedArray) (cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t  resource) =
	(cudaError_t (*) (cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t  resource)) dlsym(cudart_handle, "cudaGraphicsResourceGetMappedMipmappedArray");

cudaError_t (*lcudaGetChannelDesc) (struct cudaChannelFormatDesc * desc, cudaArray_const_t  array) =
	(cudaError_t (*) (struct cudaChannelFormatDesc * desc, cudaArray_const_t  array)) dlsym(cudart_handle, "cudaGetChannelDesc");

struct cudaChannelFormatDesc (*lcudaCreateChannelDesc) (int  x, int  y, int  z, int  w, enum cudaChannelFormatKind  f) =
	(struct cudaChannelFormatDesc (*) (int  x, int  y, int  z, int  w, enum cudaChannelFormatKind  f)) dlsym(RTLD_NEXT, "cudaCreateChannelDesc");

cudaError_t (*lcudaCreateTextureObject) (cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc) =
	(cudaError_t (*) (cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc)) dlsym(cudart_handle, "cudaCreateTextureObject");

cudaError_t (*lcudaDestroyTextureObject) (cudaTextureObject_t  texObject) =
	(cudaError_t (*) (cudaTextureObject_t  texObject)) dlsym(cudart_handle, "cudaDestroyTextureObject");

cudaError_t (*lcudaGetTextureObjectResourceDesc) (struct cudaResourceDesc * pResDesc, cudaTextureObject_t  texObject) =
	(cudaError_t (*) (struct cudaResourceDesc * pResDesc, cudaTextureObject_t  texObject)) dlsym(cudart_handle, "cudaGetTextureObjectResourceDesc");

cudaError_t (*lcudaGetTextureObjectTextureDesc) (struct cudaTextureDesc * pTexDesc, cudaTextureObject_t  texObject) =
	(cudaError_t (*) (struct cudaTextureDesc * pTexDesc, cudaTextureObject_t  texObject)) dlsym(cudart_handle, "cudaGetTextureObjectTextureDesc");

cudaError_t (*lcudaGetTextureObjectResourceViewDesc) (struct cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t  texObject) =
	(cudaError_t (*) (struct cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t  texObject)) dlsym(cudart_handle, "cudaGetTextureObjectResourceViewDesc");

cudaError_t (*lcudaCreateSurfaceObject) (cudaSurfaceObject_t * pSurfObject, const struct cudaResourceDesc * pResDesc) =
	(cudaError_t (*) (cudaSurfaceObject_t * pSurfObject, const struct cudaResourceDesc * pResDesc)) dlsym(cudart_handle, "cudaCreateSurfaceObject");

cudaError_t (*lcudaDestroySurfaceObject) (cudaSurfaceObject_t  surfObject) =
	(cudaError_t (*) (cudaSurfaceObject_t  surfObject)) dlsym(cudart_handle, "cudaDestroySurfaceObject");

cudaError_t (*lcudaGetSurfaceObjectResourceDesc) (struct cudaResourceDesc * pResDesc, cudaSurfaceObject_t  surfObject) =
	(cudaError_t (*) (struct cudaResourceDesc * pResDesc, cudaSurfaceObject_t  surfObject)) dlsym(cudart_handle, "cudaGetSurfaceObjectResourceDesc");

cudaError_t (*lcudaDriverGetVersion) (int * driverVersion) =
	(cudaError_t (*) (int * driverVersion)) dlsym(cudart_handle, "cudaDriverGetVersion");

cudaError_t (*lcudaRuntimeGetVersion) (int * runtimeVersion) =
	(cudaError_t (*) (int * runtimeVersion)) dlsym(cudart_handle, "cudaRuntimeGetVersion");

cudaError_t (*lcudaGraphCreate) (cudaGraph_t * pGraph, unsigned int  flags) =
	(cudaError_t (*) (cudaGraph_t * pGraph, unsigned int  flags)) dlsym(cudart_handle, "cudaGraphCreate");

cudaError_t (*lcudaGraphAddKernelNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaKernelNodeParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaKernelNodeParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphAddKernelNode");

cudaError_t (*lcudaGraphKernelNodeGetParams) (cudaGraphNode_t  node, struct cudaKernelNodeParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  node, struct cudaKernelNodeParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphKernelNodeGetParams");

cudaError_t (*lcudaGraphKernelNodeSetParams) (cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphKernelNodeSetParams");

cudaError_t (*lcudaGraphKernelNodeCopyAttributes) (cudaGraphNode_t  hSrc, cudaGraphNode_t  hDst) =
	(cudaError_t (*) (cudaGraphNode_t  hSrc, cudaGraphNode_t  hDst)) dlsym(cudart_handle, "cudaGraphKernelNodeCopyAttributes");

cudaError_t (*lcudaGraphKernelNodeGetAttribute) (cudaGraphNode_t  hNode, cudaLaunchAttributeID  attr, cudaLaunchAttributeValue * value_out) =
	(cudaError_t (*) (cudaGraphNode_t  hNode, cudaLaunchAttributeID  attr, cudaLaunchAttributeValue * value_out)) dlsym(cudart_handle, "cudaGraphKernelNodeGetAttribute");

cudaError_t (*lcudaGraphKernelNodeSetAttribute) (cudaGraphNode_t  hNode, cudaLaunchAttributeID  attr, const cudaLaunchAttributeValue * value) =
	(cudaError_t (*) (cudaGraphNode_t  hNode, cudaLaunchAttributeID  attr, const cudaLaunchAttributeValue * value)) dlsym(cudart_handle, "cudaGraphKernelNodeSetAttribute");

cudaError_t (*lcudaGraphAddMemcpyNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemcpy3DParms * pCopyParams) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemcpy3DParms * pCopyParams)) dlsym(cudart_handle, "cudaGraphAddMemcpyNode");

cudaError_t (*lcudaGraphAddMemcpyNodeToSymbol) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaGraphAddMemcpyNodeToSymbol");

cudaError_t (*lcudaGraphAddMemcpyNodeFromSymbol) (cudaGraphNode_t*  pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t*  pDependencies, size_t  numDependencies, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaGraphNode_t*  pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t*  pDependencies, size_t  numDependencies, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaGraphAddMemcpyNodeFromSymbol");

cudaError_t (*lcudaGraphAddMemcpyNode1D) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaGraphAddMemcpyNode1D");

cudaError_t (*lcudaGraphMemcpyNodeGetParams) (cudaGraphNode_t  node, struct cudaMemcpy3DParms * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  node, struct cudaMemcpy3DParms * pNodeParams)) dlsym(cudart_handle, "cudaGraphMemcpyNodeGetParams");

cudaError_t (*lcudaGraphMemcpyNodeSetParams) (cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams)) dlsym(cudart_handle, "cudaGraphMemcpyNodeSetParams");

cudaError_t (*lcudaGraphMemcpyNodeSetParamsToSymbol) (cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaGraphMemcpyNodeSetParamsToSymbol");

cudaError_t (*lcudaGraphMemcpyNodeSetParamsFromSymbol) (cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaGraphMemcpyNodeSetParamsFromSymbol");

cudaError_t (*lcudaGraphMemcpyNodeSetParams1D) (cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaGraphMemcpyNodeSetParams1D");

cudaError_t (*lcudaGraphAddMemsetNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemsetParams * pMemsetParams) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemsetParams * pMemsetParams)) dlsym(cudart_handle, "cudaGraphAddMemsetNode");

cudaError_t (*lcudaGraphMemsetNodeGetParams) (cudaGraphNode_t  node, struct cudaMemsetParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  node, struct cudaMemsetParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphMemsetNodeGetParams");

cudaError_t (*lcudaGraphMemsetNodeSetParams) (cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphMemsetNodeSetParams");

cudaError_t (*lcudaGraphAddHostNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaHostNodeParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaHostNodeParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphAddHostNode");

cudaError_t (*lcudaGraphHostNodeGetParams) (cudaGraphNode_t  node, struct cudaHostNodeParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  node, struct cudaHostNodeParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphHostNodeGetParams");

cudaError_t (*lcudaGraphHostNodeSetParams) (cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphHostNodeSetParams");

cudaError_t (*lcudaGraphAddChildGraphNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaGraph_t  childGraph) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaGraph_t  childGraph)) dlsym(cudart_handle, "cudaGraphAddChildGraphNode");

cudaError_t (*lcudaGraphChildGraphNodeGetGraph) (cudaGraphNode_t  node, cudaGraph_t * pGraph) =
	(cudaError_t (*) (cudaGraphNode_t  node, cudaGraph_t * pGraph)) dlsym(cudart_handle, "cudaGraphChildGraphNodeGetGraph");

cudaError_t (*lcudaGraphAddEmptyNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies)) dlsym(cudart_handle, "cudaGraphAddEmptyNode");

cudaError_t (*lcudaGraphAddEventRecordNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event)) dlsym(cudart_handle, "cudaGraphAddEventRecordNode");

cudaError_t (*lcudaGraphEventRecordNodeGetEvent) (cudaGraphNode_t  node, cudaEvent_t * event_out) =
	(cudaError_t (*) (cudaGraphNode_t  node, cudaEvent_t * event_out)) dlsym(cudart_handle, "cudaGraphEventRecordNodeGetEvent");

cudaError_t (*lcudaGraphEventRecordNodeSetEvent) (cudaGraphNode_t  node, cudaEvent_t  event) =
	(cudaError_t (*) (cudaGraphNode_t  node, cudaEvent_t  event)) dlsym(cudart_handle, "cudaGraphEventRecordNodeSetEvent");

cudaError_t (*lcudaGraphAddEventWaitNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event)) dlsym(cudart_handle, "cudaGraphAddEventWaitNode");

cudaError_t (*lcudaGraphEventWaitNodeGetEvent) (cudaGraphNode_t  node, cudaEvent_t * event_out) =
	(cudaError_t (*) (cudaGraphNode_t  node, cudaEvent_t * event_out)) dlsym(cudart_handle, "cudaGraphEventWaitNodeGetEvent");

cudaError_t (*lcudaGraphEventWaitNodeSetEvent) (cudaGraphNode_t  node, cudaEvent_t  event) =
	(cudaError_t (*) (cudaGraphNode_t  node, cudaEvent_t  event)) dlsym(cudart_handle, "cudaGraphEventWaitNodeSetEvent");

cudaError_t (*lcudaGraphAddExternalSemaphoresSignalNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphAddExternalSemaphoresSignalNode");

cudaError_t (*lcudaGraphExternalSemaphoresSignalNodeGetParams) (cudaGraphNode_t  hNode, struct cudaExternalSemaphoreSignalNodeParams * params_out) =
	(cudaError_t (*) (cudaGraphNode_t  hNode, struct cudaExternalSemaphoreSignalNodeParams * params_out)) dlsym(cudart_handle, "cudaGraphExternalSemaphoresSignalNodeGetParams");

cudaError_t (*lcudaGraphExternalSemaphoresSignalNodeSetParams) (cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphExternalSemaphoresSignalNodeSetParams");

cudaError_t (*lcudaGraphAddExternalSemaphoresWaitNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphAddExternalSemaphoresWaitNode");

cudaError_t (*lcudaGraphExternalSemaphoresWaitNodeGetParams) (cudaGraphNode_t  hNode, struct cudaExternalSemaphoreWaitNodeParams * params_out) =
	(cudaError_t (*) (cudaGraphNode_t  hNode, struct cudaExternalSemaphoreWaitNodeParams * params_out)) dlsym(cudart_handle, "cudaGraphExternalSemaphoresWaitNodeGetParams");

cudaError_t (*lcudaGraphExternalSemaphoresWaitNodeSetParams) (cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphExternalSemaphoresWaitNodeSetParams");

cudaError_t (*lcudaGraphAddMemAllocNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, struct cudaMemAllocNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, struct cudaMemAllocNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphAddMemAllocNode");

cudaError_t (*lcudaGraphMemAllocNodeGetParams) (cudaGraphNode_t  node, struct cudaMemAllocNodeParams * params_out) =
	(cudaError_t (*) (cudaGraphNode_t  node, struct cudaMemAllocNodeParams * params_out)) dlsym(cudart_handle, "cudaGraphMemAllocNodeGetParams");

cudaError_t (*lcudaGraphAddMemFreeNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void * dptr) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void * dptr)) dlsym(cudart_handle, "cudaGraphAddMemFreeNode");

cudaError_t (*lcudaGraphMemFreeNodeGetParams) (cudaGraphNode_t  node, void * dptr_out) =
	(cudaError_t (*) (cudaGraphNode_t  node, void * dptr_out)) dlsym(cudart_handle, "cudaGraphMemFreeNodeGetParams");

cudaError_t (*lcudaDeviceGraphMemTrim) (int  device) =
	(cudaError_t (*) (int  device)) dlsym(cudart_handle, "cudaDeviceGraphMemTrim");

cudaError_t (*lcudaDeviceGetGraphMemAttribute) (int  device, enum cudaGraphMemAttributeType  attr, void*  value) =
	(cudaError_t (*) (int  device, enum cudaGraphMemAttributeType  attr, void*  value)) dlsym(cudart_handle, "cudaDeviceGetGraphMemAttribute");

cudaError_t (*lcudaDeviceSetGraphMemAttribute) (int  device, enum cudaGraphMemAttributeType  attr, void*  value) =
	(cudaError_t (*) (int  device, enum cudaGraphMemAttributeType  attr, void*  value)) dlsym(cudart_handle, "cudaDeviceSetGraphMemAttribute");

cudaError_t (*lcudaGraphClone) (cudaGraph_t * pGraphClone, cudaGraph_t  originalGraph) =
	(cudaError_t (*) (cudaGraph_t * pGraphClone, cudaGraph_t  originalGraph)) dlsym(cudart_handle, "cudaGraphClone");

cudaError_t (*lcudaGraphNodeFindInClone) (cudaGraphNode_t * pNode, cudaGraphNode_t  originalNode, cudaGraph_t  clonedGraph) =
	(cudaError_t (*) (cudaGraphNode_t * pNode, cudaGraphNode_t  originalNode, cudaGraph_t  clonedGraph)) dlsym(cudart_handle, "cudaGraphNodeFindInClone");

cudaError_t (*lcudaGraphNodeGetType) (cudaGraphNode_t  node, enum cudaGraphNodeType * pType) =
	(cudaError_t (*) (cudaGraphNode_t  node, enum cudaGraphNodeType * pType)) dlsym(cudart_handle, "cudaGraphNodeGetType");

cudaError_t (*lcudaGraphGetNodes) (cudaGraph_t  graph, cudaGraphNode_t * nodes, size_t * numNodes) =
	(cudaError_t (*) (cudaGraph_t  graph, cudaGraphNode_t * nodes, size_t * numNodes)) dlsym(cudart_handle, "cudaGraphGetNodes");

cudaError_t (*lcudaGraphGetRootNodes) (cudaGraph_t  graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes) =
	(cudaError_t (*) (cudaGraph_t  graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes)) dlsym(cudart_handle, "cudaGraphGetRootNodes");

cudaError_t (*lcudaGraphGetEdges) (cudaGraph_t  graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges) =
	(cudaError_t (*) (cudaGraph_t  graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges)) dlsym(cudart_handle, "cudaGraphGetEdges");

cudaError_t (*lcudaGraphNodeGetDependencies) (cudaGraphNode_t  node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies) =
	(cudaError_t (*) (cudaGraphNode_t  node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies)) dlsym(cudart_handle, "cudaGraphNodeGetDependencies");

cudaError_t (*lcudaGraphNodeGetDependentNodes) (cudaGraphNode_t  node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes) =
	(cudaError_t (*) (cudaGraphNode_t  node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes)) dlsym(cudart_handle, "cudaGraphNodeGetDependentNodes");

cudaError_t (*lcudaGraphAddDependencies) (cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies) =
	(cudaError_t (*) (cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies)) dlsym(cudart_handle, "cudaGraphAddDependencies");

cudaError_t (*lcudaGraphRemoveDependencies) (cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies) =
	(cudaError_t (*) (cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies)) dlsym(cudart_handle, "cudaGraphRemoveDependencies");

cudaError_t (*lcudaGraphDestroyNode) (cudaGraphNode_t  node) =
	(cudaError_t (*) (cudaGraphNode_t  node)) dlsym(cudart_handle, "cudaGraphDestroyNode");

cudaError_t (*lcudaGraphInstantiate) (cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags) =
	(cudaError_t (*) (cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags)) dlsym(cudart_handle, "cudaGraphInstantiate");

cudaError_t (*lcudaGraphInstantiateWithFlags) (cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags) =
	(cudaError_t (*) (cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags)) dlsym(cudart_handle, "cudaGraphInstantiateWithFlags");

cudaError_t (*lcudaGraphInstantiateWithParams) (cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, cudaGraphInstantiateParams * instantiateParams) =
	(cudaError_t (*) (cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, cudaGraphInstantiateParams * instantiateParams)) dlsym(cudart_handle, "cudaGraphInstantiateWithParams");

cudaError_t (*lcudaGraphExecGetFlags) (cudaGraphExec_t  graphExec, unsigned long long * flags) =
	(cudaError_t (*) (cudaGraphExec_t  graphExec, unsigned long long * flags)) dlsym(cudart_handle, "cudaGraphExecGetFlags");

cudaError_t (*lcudaGraphExecKernelNodeSetParams) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphExecKernelNodeSetParams");

cudaError_t (*lcudaGraphExecMemcpyNodeSetParams) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams)) dlsym(cudart_handle, "cudaGraphExecMemcpyNodeSetParams");

cudaError_t (*lcudaGraphExecMemcpyNodeSetParamsToSymbol) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaGraphExecMemcpyNodeSetParamsToSymbol");

cudaError_t (*lcudaGraphExecMemcpyNodeSetParamsFromSymbol) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaGraphExecMemcpyNodeSetParamsFromSymbol");

cudaError_t (*lcudaGraphExecMemcpyNodeSetParams1D) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)) dlsym(cudart_handle, "cudaGraphExecMemcpyNodeSetParams1D");

cudaError_t (*lcudaGraphExecMemsetNodeSetParams) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphExecMemsetNodeSetParams");

cudaError_t (*lcudaGraphExecHostNodeSetParams) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams)) dlsym(cudart_handle, "cudaGraphExecHostNodeSetParams");

cudaError_t (*lcudaGraphExecChildGraphNodeSetParams) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, cudaGraph_t  childGraph) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, cudaGraph_t  childGraph)) dlsym(cudart_handle, "cudaGraphExecChildGraphNodeSetParams");

cudaError_t (*lcudaGraphExecEventRecordNodeSetEvent) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event)) dlsym(cudart_handle, "cudaGraphExecEventRecordNodeSetEvent");

cudaError_t (*lcudaGraphExecEventWaitNodeSetEvent) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event)) dlsym(cudart_handle, "cudaGraphExecEventWaitNodeSetEvent");

cudaError_t (*lcudaGraphExecExternalSemaphoresSignalNodeSetParams) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphExecExternalSemaphoresSignalNodeSetParams");

cudaError_t (*lcudaGraphExecExternalSemaphoresWaitNodeSetParams) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphExecExternalSemaphoresWaitNodeSetParams");

cudaError_t (*lcudaGraphNodeSetEnabled) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, unsigned int  isEnabled) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, unsigned int  isEnabled)) dlsym(cudart_handle, "cudaGraphNodeSetEnabled");

cudaError_t (*lcudaGraphNodeGetEnabled) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, unsigned int * isEnabled) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, unsigned int * isEnabled)) dlsym(cudart_handle, "cudaGraphNodeGetEnabled");

cudaError_t (*lcudaGraphExecUpdate) (cudaGraphExec_t  hGraphExec, cudaGraph_t  hGraph, cudaGraphExecUpdateResultInfo * resultInfo) =
	(cudaError_t (*) (cudaGraphExec_t  hGraphExec, cudaGraph_t  hGraph, cudaGraphExecUpdateResultInfo * resultInfo)) dlsym(cudart_handle, "cudaGraphExecUpdate");

cudaError_t (*lcudaGraphUpload) (cudaGraphExec_t  graphExec, cudaStream_t  stream) =
	(cudaError_t (*) (cudaGraphExec_t  graphExec, cudaStream_t  stream)) dlsym(cudart_handle, "cudaGraphUpload");

cudaError_t (*lcudaGraphLaunch) (cudaGraphExec_t  graphExec, cudaStream_t  stream) =
	(cudaError_t (*) (cudaGraphExec_t  graphExec, cudaStream_t  stream)) dlsym(cudart_handle, "cudaGraphLaunch");

cudaError_t (*lcudaGraphExecDestroy) (cudaGraphExec_t  graphExec) =
	(cudaError_t (*) (cudaGraphExec_t  graphExec)) dlsym(cudart_handle, "cudaGraphExecDestroy");

cudaError_t (*lcudaGraphDestroy) (cudaGraph_t  graph) =
	(cudaError_t (*) (cudaGraph_t  graph)) dlsym(cudart_handle, "cudaGraphDestroy");

cudaError_t (*lcudaGraphDebugDotPrint) (cudaGraph_t  graph, const char * path, unsigned int  flags) =
	(cudaError_t (*) (cudaGraph_t  graph, const char * path, unsigned int  flags)) dlsym(cudart_handle, "cudaGraphDebugDotPrint");

cudaError_t (*lcudaUserObjectCreate) (cudaUserObject_t * object_out, void * ptr, cudaHostFn_t  destroy, unsigned int  initialRefcount, unsigned int  flags) =
	(cudaError_t (*) (cudaUserObject_t * object_out, void * ptr, cudaHostFn_t  destroy, unsigned int  initialRefcount, unsigned int  flags)) dlsym(cudart_handle, "cudaUserObjectCreate");

cudaError_t (*lcudaUserObjectRetain) (cudaUserObject_t  object, unsigned int  count) =
	(cudaError_t (*) (cudaUserObject_t  object, unsigned int  count)) dlsym(cudart_handle, "cudaUserObjectRetain");

cudaError_t (*lcudaUserObjectRelease) (cudaUserObject_t  object, unsigned int  count) =
	(cudaError_t (*) (cudaUserObject_t  object, unsigned int  count)) dlsym(cudart_handle, "cudaUserObjectRelease");

cudaError_t (*lcudaGraphRetainUserObject) (cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count, unsigned int  flags) =
	(cudaError_t (*) (cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count, unsigned int  flags)) dlsym(cudart_handle, "cudaGraphRetainUserObject");

cudaError_t (*lcudaGraphReleaseUserObject) (cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count) =
	(cudaError_t (*) (cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count)) dlsym(cudart_handle, "cudaGraphReleaseUserObject");

cudaError_t (*lcudaGraphAddNode) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, struct cudaGraphNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, struct cudaGraphNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphAddNode");

cudaError_t (*lcudaGraphNodeSetParams) (cudaGraphNode_t  node, struct cudaGraphNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphNode_t  node, struct cudaGraphNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphNodeSetParams");

cudaError_t (*lcudaGraphExecNodeSetParams) (cudaGraphExec_t  graphExec, cudaGraphNode_t  node, struct cudaGraphNodeParams * nodeParams) =
	(cudaError_t (*) (cudaGraphExec_t  graphExec, cudaGraphNode_t  node, struct cudaGraphNodeParams * nodeParams)) dlsym(cudart_handle, "cudaGraphExecNodeSetParams");

cudaError_t (*lcudaGetDriverEntryPoint) (const char * symbol, void ** funcPtr, unsigned long long  flags, enum cudaDriverEntryPointQueryResult * driverStatus) =
	(cudaError_t (*) (const char * symbol, void ** funcPtr, unsigned long long  flags, enum cudaDriverEntryPointQueryResult * driverStatus)) dlsym(cudart_handle, "cudaGetDriverEntryPoint");

cudaError_t (*lcudaGetExportTable) (const void ** ppExportTable, const cudaUUID_t * pExportTableId) =
	(cudaError_t (*) (const void ** ppExportTable, const cudaUUID_t * pExportTableId)) dlsym(cudart_handle, "cudaGetExportTable");

cudaError_t (*lcudaGetFuncBySymbol) (cudaFunction_t*  functionPtr, const void*  symbolPtr) =
	(cudaError_t (*) (cudaFunction_t*  functionPtr, const void*  symbolPtr)) dlsym(cudart_handle, "cudaGetFuncBySymbol");

cudaError_t (*lcudaGetKernel) (cudaKernel_t * kernelPtr, const void * entryFuncAddr) =
	(cudaError_t (*) (cudaKernel_t * kernelPtr, const void * entryFuncAddr)) dlsym(cudart_handle, "cudaGetKernel");

size_t (*lcudnnGetVersion) () =
	(size_t (*) ()) dlsym(cudnn_handle, "cudnnGetVersion");

size_t (*lcudnnGetMaxDeviceVersion) () =
	(size_t (*) ()) dlsym(cudnn_handle, "cudnnGetMaxDeviceVersion");

size_t (*lcudnnGetCudartVersion) () =
	(size_t (*) ()) dlsym(cudnn_handle, "cudnnGetCudartVersion");

const char * (*lcudnnGetErrorString) (cudnnStatus_t  status) =
	(const char * (*) (cudnnStatus_t  status)) dlsym(cudnn_handle, "cudnnGetErrorString");

cudnnStatus_t (*lcudnnQueryRuntimeError) (cudnnHandle_t  handle, cudnnStatus_t * rstatus, cudnnErrQueryMode_t  mode, cudnnRuntimeTag_t * tag) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnStatus_t * rstatus, cudnnErrQueryMode_t  mode, cudnnRuntimeTag_t * tag)) dlsym(cudnn_handle, "cudnnQueryRuntimeError");

cudnnStatus_t (*lcudnnGetProperty) (libraryPropertyType  type, int * value) =
	(cudnnStatus_t (*) (libraryPropertyType  type, int * value)) dlsym(cudnn_handle, "cudnnGetProperty");

cudnnStatus_t (*lcudnnCreate) (cudnnHandle_t * handle) =
	(cudnnStatus_t (*) (cudnnHandle_t * handle)) dlsym(cudnn_handle, "cudnnCreate");

cudnnStatus_t (*lcudnnDestroy) (cudnnHandle_t  handle) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle)) dlsym(cudnn_handle, "cudnnDestroy");

cudnnStatus_t (*lcudnnSetStream) (cudnnHandle_t  handle, cudaStream_t  streamId) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudaStream_t  streamId)) dlsym(cudnn_handle, "cudnnSetStream");

cudnnStatus_t (*lcudnnGetStream) (cudnnHandle_t  handle, cudaStream_t * streamId) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudaStream_t * streamId)) dlsym(cudnn_handle, "cudnnGetStream");

cudnnStatus_t (*lcudnnCreateTensorDescriptor) (cudnnTensorDescriptor_t * tensorDesc) =
	(cudnnStatus_t (*) (cudnnTensorDescriptor_t * tensorDesc)) dlsym(cudnn_handle, "cudnnCreateTensorDescriptor");

cudnnStatus_t (*lcudnnSetTensor4dDescriptor) (cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w) =
	(cudnnStatus_t (*) (cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w)) dlsym(cudnn_handle, "cudnnSetTensor4dDescriptor");

cudnnStatus_t (*lcudnnSetTensor4dDescriptorEx) (cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w, int  nStride, int  cStride, int  hStride, int  wStride) =
	(cudnnStatus_t (*) (cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w, int  nStride, int  cStride, int  hStride, int  wStride)) dlsym(cudnn_handle, "cudnnSetTensor4dDescriptorEx");

cudnnStatus_t (*lcudnnGetTensor4dDescriptor) (const cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t * dataType, int * n, int * c, int * h, int * w, int * nStride, int * cStride, int * hStride, int * wStride) =
	(cudnnStatus_t (*) (const cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t * dataType, int * n, int * c, int * h, int * w, int * nStride, int * cStride, int * hStride, int * wStride)) dlsym(cudnn_handle, "cudnnGetTensor4dDescriptor");

cudnnStatus_t (*lcudnnSetTensorNdDescriptor) (cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const int  strideA[]) =
	(cudnnStatus_t (*) (cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const int  strideA[])) dlsym(cudnn_handle, "cudnnSetTensorNdDescriptor");

cudnnStatus_t (*lcudnnSetTensorNdDescriptorEx) (cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  nbDims, const int  dimA[]) =
	(cudnnStatus_t (*) (cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  nbDims, const int  dimA[])) dlsym(cudnn_handle, "cudnnSetTensorNdDescriptorEx");

cudnnStatus_t (*lcudnnGetTensorNdDescriptor) (const cudnnTensorDescriptor_t  tensorDesc, int  nbDimsRequested, cudnnDataType_t * dataType, int * nbDims, int  dimA[], int  strideA[]) =
	(cudnnStatus_t (*) (const cudnnTensorDescriptor_t  tensorDesc, int  nbDimsRequested, cudnnDataType_t * dataType, int * nbDims, int  dimA[], int  strideA[])) dlsym(cudnn_handle, "cudnnGetTensorNdDescriptor");

cudnnStatus_t (*lcudnnGetTensorSizeInBytes) (const cudnnTensorDescriptor_t  tensorDesc, size_t * size) =
	(cudnnStatus_t (*) (const cudnnTensorDescriptor_t  tensorDesc, size_t * size)) dlsym(cudnn_handle, "cudnnGetTensorSizeInBytes");

cudnnStatus_t (*lcudnnDestroyTensorDescriptor) (cudnnTensorDescriptor_t  tensorDesc) =
	(cudnnStatus_t (*) (cudnnTensorDescriptor_t  tensorDesc)) dlsym(cudnn_handle, "cudnnDestroyTensorDescriptor");

cudnnStatus_t (*lcudnnInitTransformDest) (const cudnnTensorTransformDescriptor_t  transformDesc, const cudnnTensorDescriptor_t  srcDesc, cudnnTensorDescriptor_t  destDesc, size_t * destSizeInBytes) =
	(cudnnStatus_t (*) (const cudnnTensorTransformDescriptor_t  transformDesc, const cudnnTensorDescriptor_t  srcDesc, cudnnTensorDescriptor_t  destDesc, size_t * destSizeInBytes)) dlsym(cudnn_handle, "cudnnInitTransformDest");

cudnnStatus_t (*lcudnnCreateTensorTransformDescriptor) (cudnnTensorTransformDescriptor_t * transformDesc) =
	(cudnnStatus_t (*) (cudnnTensorTransformDescriptor_t * transformDesc)) dlsym(cudnn_handle, "cudnnCreateTensorTransformDescriptor");

cudnnStatus_t (*lcudnnSetTensorTransformDescriptor) (cudnnTensorTransformDescriptor_t  transformDesc, const uint32_t  nbDims, const cudnnTensorFormat_t  destFormat, const int32_t  padBeforeA[], const int32_t  padAfterA[], const uint32_t  foldA[], const cudnnFoldingDirection_t  direction) =
	(cudnnStatus_t (*) (cudnnTensorTransformDescriptor_t  transformDesc, const uint32_t  nbDims, const cudnnTensorFormat_t  destFormat, const int32_t  padBeforeA[], const int32_t  padAfterA[], const uint32_t  foldA[], const cudnnFoldingDirection_t  direction)) dlsym(cudnn_handle, "cudnnSetTensorTransformDescriptor");

cudnnStatus_t (*lcudnnGetTensorTransformDescriptor) (cudnnTensorTransformDescriptor_t  transformDesc, uint32_t  nbDimsRequested, cudnnTensorFormat_t * destFormat, int32_t  padBeforeA[], int32_t  padAfterA[], uint32_t  foldA[], cudnnFoldingDirection_t * direction) =
	(cudnnStatus_t (*) (cudnnTensorTransformDescriptor_t  transformDesc, uint32_t  nbDimsRequested, cudnnTensorFormat_t * destFormat, int32_t  padBeforeA[], int32_t  padAfterA[], uint32_t  foldA[], cudnnFoldingDirection_t * direction)) dlsym(cudnn_handle, "cudnnGetTensorTransformDescriptor");

cudnnStatus_t (*lcudnnDestroyTensorTransformDescriptor) (cudnnTensorTransformDescriptor_t  transformDesc) =
	(cudnnStatus_t (*) (cudnnTensorTransformDescriptor_t  transformDesc)) dlsym(cudnn_handle, "cudnnDestroyTensorTransformDescriptor");

cudnnStatus_t (*lcudnnTransformTensor) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)) dlsym(cudnn_handle, "cudnnTransformTensor");

cudnnStatus_t (*lcudnnTransformTensorEx) (cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnTensorDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnTensorDescriptor_t  destDesc, void * destData) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnTensorDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnTensorDescriptor_t  destDesc, void * destData)) dlsym(cudnn_handle, "cudnnTransformTensorEx");

cudnnStatus_t (*lcudnnAddTensor) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)) dlsym(cudnn_handle, "cudnnAddTensor");

cudnnStatus_t (*lcudnnCreateOpTensorDescriptor) (cudnnOpTensorDescriptor_t * opTensorDesc) =
	(cudnnStatus_t (*) (cudnnOpTensorDescriptor_t * opTensorDesc)) dlsym(cudnn_handle, "cudnnCreateOpTensorDescriptor");

cudnnStatus_t (*lcudnnSetOpTensorDescriptor) (cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t  opTensorOp, cudnnDataType_t  opTensorCompType, cudnnNanPropagation_t  opTensorNanOpt) =
	(cudnnStatus_t (*) (cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t  opTensorOp, cudnnDataType_t  opTensorCompType, cudnnNanPropagation_t  opTensorNanOpt)) dlsym(cudnn_handle, "cudnnSetOpTensorDescriptor");

cudnnStatus_t (*lcudnnGetOpTensorDescriptor) (const cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t * opTensorOp, cudnnDataType_t * opTensorCompType, cudnnNanPropagation_t * opTensorNanOpt) =
	(cudnnStatus_t (*) (const cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t * opTensorOp, cudnnDataType_t * opTensorCompType, cudnnNanPropagation_t * opTensorNanOpt)) dlsym(cudnn_handle, "cudnnGetOpTensorDescriptor");

cudnnStatus_t (*lcudnnDestroyOpTensorDescriptor) (cudnnOpTensorDescriptor_t  opTensorDesc) =
	(cudnnStatus_t (*) (cudnnOpTensorDescriptor_t  opTensorDesc)) dlsym(cudnn_handle, "cudnnDestroyOpTensorDescriptor");

cudnnStatus_t (*lcudnnOpTensor) (cudnnHandle_t  handle, const cudnnOpTensorDescriptor_t  opTensorDesc, const void * alpha1, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * alpha2, const cudnnTensorDescriptor_t  bDesc, const void * B, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnOpTensorDescriptor_t  opTensorDesc, const void * alpha1, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * alpha2, const cudnnTensorDescriptor_t  bDesc, const void * B, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)) dlsym(cudnn_handle, "cudnnOpTensor");

cudnnStatus_t (*lcudnnCreateReduceTensorDescriptor) (cudnnReduceTensorDescriptor_t * reduceTensorDesc) =
	(cudnnStatus_t (*) (cudnnReduceTensorDescriptor_t * reduceTensorDesc)) dlsym(cudnn_handle, "cudnnCreateReduceTensorDescriptor");

cudnnStatus_t (*lcudnnSetReduceTensorDescriptor) (cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t  reduceTensorOp, cudnnDataType_t  reduceTensorCompType, cudnnNanPropagation_t  reduceTensorNanOpt, cudnnReduceTensorIndices_t  reduceTensorIndices, cudnnIndicesType_t  reduceTensorIndicesType) =
	(cudnnStatus_t (*) (cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t  reduceTensorOp, cudnnDataType_t  reduceTensorCompType, cudnnNanPropagation_t  reduceTensorNanOpt, cudnnReduceTensorIndices_t  reduceTensorIndices, cudnnIndicesType_t  reduceTensorIndicesType)) dlsym(cudnn_handle, "cudnnSetReduceTensorDescriptor");

cudnnStatus_t (*lcudnnGetReduceTensorDescriptor) (const cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t * reduceTensorOp, cudnnDataType_t * reduceTensorCompType, cudnnNanPropagation_t * reduceTensorNanOpt, cudnnReduceTensorIndices_t * reduceTensorIndices, cudnnIndicesType_t * reduceTensorIndicesType) =
	(cudnnStatus_t (*) (const cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t * reduceTensorOp, cudnnDataType_t * reduceTensorCompType, cudnnNanPropagation_t * reduceTensorNanOpt, cudnnReduceTensorIndices_t * reduceTensorIndices, cudnnIndicesType_t * reduceTensorIndicesType)) dlsym(cudnn_handle, "cudnnGetReduceTensorDescriptor");

cudnnStatus_t (*lcudnnDestroyReduceTensorDescriptor) (cudnnReduceTensorDescriptor_t  reduceTensorDesc) =
	(cudnnStatus_t (*) (cudnnReduceTensorDescriptor_t  reduceTensorDesc)) dlsym(cudnn_handle, "cudnnDestroyReduceTensorDescriptor");

cudnnStatus_t (*lcudnnGetReductionIndicesSize) (cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetReductionIndicesSize");

cudnnStatus_t (*lcudnnGetReductionWorkspaceSize) (cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetReductionWorkspaceSize");

cudnnStatus_t (*lcudnnReduceTensor) (cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, void * indices, size_t  indicesSizeInBytes, void * workspace, size_t  workspaceSizeInBytes, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, void * indices, size_t  indicesSizeInBytes, void * workspace, size_t  workspaceSizeInBytes, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)) dlsym(cudnn_handle, "cudnnReduceTensor");

cudnnStatus_t (*lcudnnSetTensor) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * valuePtr) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * valuePtr)) dlsym(cudnn_handle, "cudnnSetTensor");

cudnnStatus_t (*lcudnnScaleTensor) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * alpha) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * alpha)) dlsym(cudnn_handle, "cudnnScaleTensor");

cudnnStatus_t (*lcudnnCreateFilterDescriptor) (cudnnFilterDescriptor_t * filterDesc) =
	(cudnnStatus_t (*) (cudnnFilterDescriptor_t * filterDesc)) dlsym(cudnn_handle, "cudnnCreateFilterDescriptor");

cudnnStatus_t (*lcudnnSetFilter4dDescriptor) (cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  k, int  c, int  h, int  w) =
	(cudnnStatus_t (*) (cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  k, int  c, int  h, int  w)) dlsym(cudnn_handle, "cudnnSetFilter4dDescriptor");

cudnnStatus_t (*lcudnnGetFilter4dDescriptor) (const cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * k, int * c, int * h, int * w) =
	(cudnnStatus_t (*) (const cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * k, int * c, int * h, int * w)) dlsym(cudnn_handle, "cudnnGetFilter4dDescriptor");

cudnnStatus_t (*lcudnnSetFilterNdDescriptor) (cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  nbDims, const int  filterDimA[]) =
	(cudnnStatus_t (*) (cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  nbDims, const int  filterDimA[])) dlsym(cudnn_handle, "cudnnSetFilterNdDescriptor");

cudnnStatus_t (*lcudnnGetFilterNdDescriptor) (const cudnnFilterDescriptor_t  filterDesc, int  nbDimsRequested, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * nbDims, int  filterDimA[]) =
	(cudnnStatus_t (*) (const cudnnFilterDescriptor_t  filterDesc, int  nbDimsRequested, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * nbDims, int  filterDimA[])) dlsym(cudnn_handle, "cudnnGetFilterNdDescriptor");

cudnnStatus_t (*lcudnnGetFilterSizeInBytes) (const cudnnFilterDescriptor_t  filterDesc, size_t * size) =
	(cudnnStatus_t (*) (const cudnnFilterDescriptor_t  filterDesc, size_t * size)) dlsym(cudnn_handle, "cudnnGetFilterSizeInBytes");

cudnnStatus_t (*lcudnnTransformFilter) (cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnFilterDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnFilterDescriptor_t  destDesc, void * destData) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnFilterDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnFilterDescriptor_t  destDesc, void * destData)) dlsym(cudnn_handle, "cudnnTransformFilter");

cudnnStatus_t (*lcudnnDestroyFilterDescriptor) (cudnnFilterDescriptor_t  filterDesc) =
	(cudnnStatus_t (*) (cudnnFilterDescriptor_t  filterDesc)) dlsym(cudnn_handle, "cudnnDestroyFilterDescriptor");

cudnnStatus_t (*lcudnnSoftmaxForward) (cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)) dlsym(cudnn_handle, "cudnnSoftmaxForward");

cudnnStatus_t (*lcudnnCreatePoolingDescriptor) (cudnnPoolingDescriptor_t * poolingDesc) =
	(cudnnStatus_t (*) (cudnnPoolingDescriptor_t * poolingDesc)) dlsym(cudnn_handle, "cudnnCreatePoolingDescriptor");

cudnnStatus_t (*lcudnnSetPooling2dDescriptor) (cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t  mode, cudnnNanPropagation_t  maxpoolingNanOpt, int  windowHeight, int  windowWidth, int  verticalPadding, int  horizontalPadding, int  verticalStride, int  horizontalStride) =
	(cudnnStatus_t (*) (cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t  mode, cudnnNanPropagation_t  maxpoolingNanOpt, int  windowHeight, int  windowWidth, int  verticalPadding, int  horizontalPadding, int  verticalStride, int  horizontalStride)) dlsym(cudnn_handle, "cudnnSetPooling2dDescriptor");

cudnnStatus_t (*lcudnnGetPooling2dDescriptor) (const cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * windowHeight, int * windowWidth, int * verticalPadding, int * horizontalPadding, int * verticalStride, int * horizontalStride) =
	(cudnnStatus_t (*) (const cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * windowHeight, int * windowWidth, int * verticalPadding, int * horizontalPadding, int * verticalStride, int * horizontalStride)) dlsym(cudnn_handle, "cudnnGetPooling2dDescriptor");

cudnnStatus_t (*lcudnnSetPoolingNdDescriptor) (cudnnPoolingDescriptor_t  poolingDesc, const cudnnPoolingMode_t  mode, const cudnnNanPropagation_t  maxpoolingNanOpt, int  nbDims, const int  windowDimA[], const int  paddingA[], const int  strideA[]) =
	(cudnnStatus_t (*) (cudnnPoolingDescriptor_t  poolingDesc, const cudnnPoolingMode_t  mode, const cudnnNanPropagation_t  maxpoolingNanOpt, int  nbDims, const int  windowDimA[], const int  paddingA[], const int  strideA[])) dlsym(cudnn_handle, "cudnnSetPoolingNdDescriptor");

cudnnStatus_t (*lcudnnGetPoolingNdDescriptor) (const cudnnPoolingDescriptor_t  poolingDesc, int  nbDimsRequested, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * nbDims, int  windowDimA[], int  paddingA[], int  strideA[]) =
	(cudnnStatus_t (*) (const cudnnPoolingDescriptor_t  poolingDesc, int  nbDimsRequested, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * nbDims, int  windowDimA[], int  paddingA[], int  strideA[])) dlsym(cudnn_handle, "cudnnGetPoolingNdDescriptor");

cudnnStatus_t (*lcudnnGetPoolingNdForwardOutputDim) (const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int  nbDims, int  outputTensorDimA[]) =
	(cudnnStatus_t (*) (const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int  nbDims, int  outputTensorDimA[])) dlsym(cudnn_handle, "cudnnGetPoolingNdForwardOutputDim");

cudnnStatus_t (*lcudnnGetPooling2dForwardOutputDim) (const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int * n, int * c, int * h, int * w) =
	(cudnnStatus_t (*) (const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int * n, int * c, int * h, int * w)) dlsym(cudnn_handle, "cudnnGetPooling2dForwardOutputDim");

cudnnStatus_t (*lcudnnDestroyPoolingDescriptor) (cudnnPoolingDescriptor_t  poolingDesc) =
	(cudnnStatus_t (*) (cudnnPoolingDescriptor_t  poolingDesc)) dlsym(cudnn_handle, "cudnnDestroyPoolingDescriptor");

cudnnStatus_t (*lcudnnPoolingForward) (cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)) dlsym(cudnn_handle, "cudnnPoolingForward");

cudnnStatus_t (*lcudnnCreateActivationDescriptor) (cudnnActivationDescriptor_t * activationDesc) =
	(cudnnStatus_t (*) (cudnnActivationDescriptor_t * activationDesc)) dlsym(cudnn_handle, "cudnnCreateActivationDescriptor");

cudnnStatus_t (*lcudnnSetActivationDescriptor) (cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t  mode, cudnnNanPropagation_t  reluNanOpt, double  coef) =
	(cudnnStatus_t (*) (cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t  mode, cudnnNanPropagation_t  reluNanOpt, double  coef)) dlsym(cudnn_handle, "cudnnSetActivationDescriptor");

cudnnStatus_t (*lcudnnGetActivationDescriptor) (const cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t * mode, cudnnNanPropagation_t * reluNanOpt, double * coef) =
	(cudnnStatus_t (*) (const cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t * mode, cudnnNanPropagation_t * reluNanOpt, double * coef)) dlsym(cudnn_handle, "cudnnGetActivationDescriptor");

cudnnStatus_t (*lcudnnSetActivationDescriptorSwishBeta) (cudnnActivationDescriptor_t  activationDesc, double  swish_beta) =
	(cudnnStatus_t (*) (cudnnActivationDescriptor_t  activationDesc, double  swish_beta)) dlsym(cudnn_handle, "cudnnSetActivationDescriptorSwishBeta");

cudnnStatus_t (*lcudnnGetActivationDescriptorSwishBeta) (cudnnActivationDescriptor_t  activationDesc, double * swish_beta) =
	(cudnnStatus_t (*) (cudnnActivationDescriptor_t  activationDesc, double * swish_beta)) dlsym(cudnn_handle, "cudnnGetActivationDescriptorSwishBeta");

cudnnStatus_t (*lcudnnDestroyActivationDescriptor) (cudnnActivationDescriptor_t  activationDesc) =
	(cudnnStatus_t (*) (cudnnActivationDescriptor_t  activationDesc)) dlsym(cudnn_handle, "cudnnDestroyActivationDescriptor");

cudnnStatus_t (*lcudnnActivationForward) (cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)) dlsym(cudnn_handle, "cudnnActivationForward");

cudnnStatus_t (*lcudnnCreateLRNDescriptor) (cudnnLRNDescriptor_t * normDesc) =
	(cudnnStatus_t (*) (cudnnLRNDescriptor_t * normDesc)) dlsym(cudnn_handle, "cudnnCreateLRNDescriptor");

cudnnStatus_t (*lcudnnSetLRNDescriptor) (cudnnLRNDescriptor_t  normDesc, unsigned  lrnN, double  lrnAlpha, double  lrnBeta, double  lrnK) =
	(cudnnStatus_t (*) (cudnnLRNDescriptor_t  normDesc, unsigned  lrnN, double  lrnAlpha, double  lrnBeta, double  lrnK)) dlsym(cudnn_handle, "cudnnSetLRNDescriptor");

cudnnStatus_t (*lcudnnGetLRNDescriptor) (cudnnLRNDescriptor_t  normDesc, unsigned * lrnN, double * lrnAlpha, double * lrnBeta, double * lrnK) =
	(cudnnStatus_t (*) (cudnnLRNDescriptor_t  normDesc, unsigned * lrnN, double * lrnAlpha, double * lrnBeta, double * lrnK)) dlsym(cudnn_handle, "cudnnGetLRNDescriptor");

cudnnStatus_t (*lcudnnDestroyLRNDescriptor) (cudnnLRNDescriptor_t  lrnDesc) =
	(cudnnStatus_t (*) (cudnnLRNDescriptor_t  lrnDesc)) dlsym(cudnn_handle, "cudnnDestroyLRNDescriptor");

cudnnStatus_t (*lcudnnLRNCrossChannelForward) (cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)) dlsym(cudnn_handle, "cudnnLRNCrossChannelForward");

cudnnStatus_t (*lcudnnDivisiveNormalizationForward) (cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)) dlsym(cudnn_handle, "cudnnDivisiveNormalizationForward");

cudnnStatus_t (*lcudnnDeriveBNTensorDescriptor) (cudnnTensorDescriptor_t  derivedBnDesc, const cudnnTensorDescriptor_t  xDesc, cudnnBatchNormMode_t  mode) =
	(cudnnStatus_t (*) (cudnnTensorDescriptor_t  derivedBnDesc, const cudnnTensorDescriptor_t  xDesc, cudnnBatchNormMode_t  mode)) dlsym(cudnn_handle, "cudnnDeriveBNTensorDescriptor");

cudnnStatus_t (*lcudnnBatchNormalizationForwardInference) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, const void * estimatedMean, const void * estimatedVariance, double  epsilon) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, const void * estimatedMean, const void * estimatedVariance, double  epsilon)) dlsym(cudnn_handle, "cudnnBatchNormalizationForwardInference");

cudnnStatus_t (*lcudnnDeriveNormTensorDescriptor) (cudnnTensorDescriptor_t  derivedNormScaleBiasDesc, cudnnTensorDescriptor_t  derivedNormMeanVarDesc, const cudnnTensorDescriptor_t  xDesc, cudnnNormMode_t  mode, int  groupCnt) =
	(cudnnStatus_t (*) (cudnnTensorDescriptor_t  derivedNormScaleBiasDesc, cudnnTensorDescriptor_t  derivedNormMeanVarDesc, const cudnnTensorDescriptor_t  xDesc, cudnnNormMode_t  mode, int  groupCnt)) dlsym(cudnn_handle, "cudnnDeriveNormTensorDescriptor");

cudnnStatus_t (*lcudnnNormalizationForwardInference) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * estimatedMean, const void * estimatedVariance, const cudnnTensorDescriptor_t  zDesc, const void * z, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y, double  epsilon, int  groupCnt) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * estimatedMean, const void * estimatedVariance, const cudnnTensorDescriptor_t  zDesc, const void * z, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y, double  epsilon, int  groupCnt)) dlsym(cudnn_handle, "cudnnNormalizationForwardInference");

cudnnStatus_t (*lcudnnCreateSpatialTransformerDescriptor) (cudnnSpatialTransformerDescriptor_t * stDesc) =
	(cudnnStatus_t (*) (cudnnSpatialTransformerDescriptor_t * stDesc)) dlsym(cudnn_handle, "cudnnCreateSpatialTransformerDescriptor");

cudnnStatus_t (*lcudnnSetSpatialTransformerNdDescriptor) (cudnnSpatialTransformerDescriptor_t  stDesc, cudnnSamplerType_t  samplerType, cudnnDataType_t  dataType, const int  nbDims, const int  dimA[]) =
	(cudnnStatus_t (*) (cudnnSpatialTransformerDescriptor_t  stDesc, cudnnSamplerType_t  samplerType, cudnnDataType_t  dataType, const int  nbDims, const int  dimA[])) dlsym(cudnn_handle, "cudnnSetSpatialTransformerNdDescriptor");

cudnnStatus_t (*lcudnnDestroySpatialTransformerDescriptor) (cudnnSpatialTransformerDescriptor_t  stDesc) =
	(cudnnStatus_t (*) (cudnnSpatialTransformerDescriptor_t  stDesc)) dlsym(cudnn_handle, "cudnnDestroySpatialTransformerDescriptor");

cudnnStatus_t (*lcudnnSpatialTfGridGeneratorForward) (cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * theta, void * grid) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * theta, void * grid)) dlsym(cudnn_handle, "cudnnSpatialTfGridGeneratorForward");

cudnnStatus_t (*lcudnnSpatialTfSamplerForward) (cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * grid, const void * beta, cudnnTensorDescriptor_t  yDesc, void * y) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * grid, const void * beta, cudnnTensorDescriptor_t  yDesc, void * y)) dlsym(cudnn_handle, "cudnnSpatialTfSamplerForward");

cudnnStatus_t (*lcudnnCreateDropoutDescriptor) (cudnnDropoutDescriptor_t * dropoutDesc) =
	(cudnnStatus_t (*) (cudnnDropoutDescriptor_t * dropoutDesc)) dlsym(cudnn_handle, "cudnnCreateDropoutDescriptor");

cudnnStatus_t (*lcudnnDestroyDropoutDescriptor) (cudnnDropoutDescriptor_t  dropoutDesc) =
	(cudnnStatus_t (*) (cudnnDropoutDescriptor_t  dropoutDesc)) dlsym(cudnn_handle, "cudnnDestroyDropoutDescriptor");

cudnnStatus_t (*lcudnnDropoutGetStatesSize) (cudnnHandle_t  handle, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnDropoutGetStatesSize");

cudnnStatus_t (*lcudnnDropoutGetReserveSpaceSize) (cudnnTensorDescriptor_t  xdesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnTensorDescriptor_t  xdesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnDropoutGetReserveSpaceSize");

cudnnStatus_t (*lcudnnSetDropoutDescriptor) (cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed) =
	(cudnnStatus_t (*) (cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed)) dlsym(cudnn_handle, "cudnnSetDropoutDescriptor");

cudnnStatus_t (*lcudnnRestoreDropoutDescriptor) (cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed) =
	(cudnnStatus_t (*) (cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed)) dlsym(cudnn_handle, "cudnnRestoreDropoutDescriptor");

cudnnStatus_t (*lcudnnGetDropoutDescriptor) (cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float * dropout, void ** states, unsigned long long * seed) =
	(cudnnStatus_t (*) (cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float * dropout, void ** states, unsigned long long * seed)) dlsym(cudnn_handle, "cudnnGetDropoutDescriptor");

cudnnStatus_t (*lcudnnDropoutForward) (cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  xdesc, const void * x, const cudnnTensorDescriptor_t  ydesc, void * y, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  xdesc, const void * x, const cudnnTensorDescriptor_t  ydesc, void * y, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnDropoutForward");

cudnnStatus_t (*lcudnnCreateAlgorithmDescriptor) (cudnnAlgorithmDescriptor_t * algoDesc) =
	(cudnnStatus_t (*) (cudnnAlgorithmDescriptor_t * algoDesc)) dlsym(cudnn_handle, "cudnnCreateAlgorithmDescriptor");

cudnnStatus_t (*lcudnnSetAlgorithmDescriptor) (cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t  algorithm) =
	(cudnnStatus_t (*) (cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t  algorithm)) dlsym(cudnn_handle, "cudnnSetAlgorithmDescriptor");

cudnnStatus_t (*lcudnnGetAlgorithmDescriptor) (const cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t * algorithm) =
	(cudnnStatus_t (*) (const cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t * algorithm)) dlsym(cudnn_handle, "cudnnGetAlgorithmDescriptor");

cudnnStatus_t (*lcudnnCopyAlgorithmDescriptor) (const cudnnAlgorithmDescriptor_t  src, cudnnAlgorithmDescriptor_t  dest) =
	(cudnnStatus_t (*) (const cudnnAlgorithmDescriptor_t  src, cudnnAlgorithmDescriptor_t  dest)) dlsym(cudnn_handle, "cudnnCopyAlgorithmDescriptor");

cudnnStatus_t (*lcudnnDestroyAlgorithmDescriptor) (cudnnAlgorithmDescriptor_t  algoDesc) =
	(cudnnStatus_t (*) (cudnnAlgorithmDescriptor_t  algoDesc)) dlsym(cudnn_handle, "cudnnDestroyAlgorithmDescriptor");

cudnnStatus_t (*lcudnnCreateAlgorithmPerformance) (cudnnAlgorithmPerformance_t * algoPerf, int  numberToCreate) =
	(cudnnStatus_t (*) (cudnnAlgorithmPerformance_t * algoPerf, int  numberToCreate)) dlsym(cudnn_handle, "cudnnCreateAlgorithmPerformance");

cudnnStatus_t (*lcudnnSetAlgorithmPerformance) (cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t  algoDesc, cudnnStatus_t  status, float  time, size_t  memory) =
	(cudnnStatus_t (*) (cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t  algoDesc, cudnnStatus_t  status, float  time, size_t  memory)) dlsym(cudnn_handle, "cudnnSetAlgorithmPerformance");

cudnnStatus_t (*lcudnnGetAlgorithmPerformance) (const cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t * algoDesc, cudnnStatus_t * status, float * time, size_t * memory) =
	(cudnnStatus_t (*) (const cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t * algoDesc, cudnnStatus_t * status, float * time, size_t * memory)) dlsym(cudnn_handle, "cudnnGetAlgorithmPerformance");

cudnnStatus_t (*lcudnnDestroyAlgorithmPerformance) (cudnnAlgorithmPerformance_t * algoPerf, int  numberToDestroy) =
	(cudnnStatus_t (*) (cudnnAlgorithmPerformance_t * algoPerf, int  numberToDestroy)) dlsym(cudnn_handle, "cudnnDestroyAlgorithmPerformance");

cudnnStatus_t (*lcudnnGetAlgorithmSpaceSize) (cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, size_t * algoSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, size_t * algoSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnGetAlgorithmSpaceSize");

cudnnStatus_t (*lcudnnSaveAlgorithm) (cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, void * algoSpace, size_t  algoSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, void * algoSpace, size_t  algoSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnSaveAlgorithm");

cudnnStatus_t (*lcudnnRestoreAlgorithm) (cudnnHandle_t  handle, void * algoSpace, size_t  algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t  algoDesc) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, void * algoSpace, size_t  algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t  algoDesc)) dlsym(cudnn_handle, "cudnnRestoreAlgorithm");

cudnnStatus_t (*lcudnnSetCallback) (unsigned  mask, void * udata, cudnnCallback_t  fptr) =
	(cudnnStatus_t (*) (unsigned  mask, void * udata, cudnnCallback_t  fptr)) dlsym(cudnn_handle, "cudnnSetCallback");

cudnnStatus_t (*lcudnnGetCallback) (unsigned * mask, void ** udata, cudnnCallback_t * fptr) =
	(cudnnStatus_t (*) (unsigned * mask, void ** udata, cudnnCallback_t * fptr)) dlsym(cudnn_handle, "cudnnGetCallback");

cudnnStatus_t (*lcudnnOpsInferVersionCheck) () =
	(cudnnStatus_t (*) ()) dlsym(cudnn_handle, "cudnnOpsInferVersionCheck");

cudnnStatus_t (*lcudnnSoftmaxBackward) (cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)) dlsym(cudnn_handle, "cudnnSoftmaxBackward");

cudnnStatus_t (*lcudnnPoolingBackward) (cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)) dlsym(cudnn_handle, "cudnnPoolingBackward");

cudnnStatus_t (*lcudnnActivationBackward) (cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)) dlsym(cudnn_handle, "cudnnActivationBackward");

cudnnStatus_t (*lcudnnLRNCrossChannelBackward) (cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)) dlsym(cudnn_handle, "cudnnLRNCrossChannelBackward");

cudnnStatus_t (*lcudnnDivisiveNormalizationBackward) (cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, const void * dy, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  dXdMeansDesc, void * dx, void * dMeans) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, const void * dy, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  dXdMeansDesc, void * dx, void * dMeans)) dlsym(cudnn_handle, "cudnnDivisiveNormalizationBackward");

cudnnStatus_t (*lcudnnGetBatchNormalizationForwardTrainingExWorkspaceSize) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");

cudnnStatus_t (*lcudnnGetBatchNormalizationBackwardExWorkspaceSize) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetBatchNormalizationBackwardExWorkspaceSize");

cudnnStatus_t (*lcudnnGetBatchNormalizationTrainingExReserveSpaceSize) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");

cudnnStatus_t (*lcudnnBatchNormalizationForwardTraining) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance)) dlsym(cudnn_handle, "cudnnBatchNormalizationForwardTraining");

cudnnStatus_t (*lcudnnBatchNormalizationForwardTrainingEx) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnBatchNormalizationForwardTrainingEx");

cudnnStatus_t (*lcudnnBatchNormalizationBackward) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScale, void * dBnScaleResult, void * dBnBiasResult, double  epsilon, const void * savedMean, const void * savedInvVariance) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScale, void * dBnScaleResult, void * dBnBiasResult, double  epsilon, const void * savedMean, const void * savedInvVariance)) dlsym(cudnn_handle, "cudnnBatchNormalizationBackward");

cudnnStatus_t (*lcudnnBatchNormalizationBackwardEx) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScaleData, const void * bnBiasData, void * dBnScaleData, void * dBnBiasData, double  epsilon, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScaleData, const void * bnBiasData, void * dBnScaleData, void * dBnBiasData, double  epsilon, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnBatchNormalizationBackwardEx");

cudnnStatus_t (*lcudnnGetNormalizationForwardTrainingWorkspaceSize) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  normScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  normScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt)) dlsym(cudnn_handle, "cudnnGetNormalizationForwardTrainingWorkspaceSize");

cudnnStatus_t (*lcudnnGetNormalizationBackwardWorkspaceSize) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt)) dlsym(cudnn_handle, "cudnnGetNormalizationBackwardWorkspaceSize");

cudnnStatus_t (*lcudnnGetNormalizationTrainingReserveSpaceSize) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, int  groupCnt) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, int  groupCnt)) dlsym(cudnn_handle, "cudnnGetNormalizationTrainingReserveSpaceSize");

cudnnStatus_t (*lcudnnNormalizationForwardTraining) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, double  exponentialAverageFactor, const cudnnTensorDescriptor_t  normMeanVarDesc, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, double  exponentialAverageFactor, const cudnnTensorDescriptor_t  normMeanVarDesc, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt)) dlsym(cudnn_handle, "cudnnNormalizationForwardTraining");

cudnnStatus_t (*lcudnnNormalizationBackward) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const void * normScaleData, const void * normBiasData, void * dNormScaleData, void * dNormBiasData, double  epsilon, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const void * normScaleData, const void * normBiasData, void * dNormScaleData, void * dNormBiasData, double  epsilon, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt)) dlsym(cudnn_handle, "cudnnNormalizationBackward");

cudnnStatus_t (*lcudnnSpatialTfGridGeneratorBackward) (cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * dgrid, void * dtheta) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * dgrid, void * dtheta)) dlsym(cudnn_handle, "cudnnSpatialTfGridGeneratorBackward");

cudnnStatus_t (*lcudnnSpatialTfSamplerBackward) (cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx, const void * alphaDgrid, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * grid, const void * betaDgrid, void * dgrid) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx, const void * alphaDgrid, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * grid, const void * betaDgrid, void * dgrid)) dlsym(cudnn_handle, "cudnnSpatialTfSamplerBackward");

cudnnStatus_t (*lcudnnDropoutBackward) (cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  dydesc, const void * dy, const cudnnTensorDescriptor_t  dxdesc, void * dx, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  dydesc, const void * dy, const cudnnTensorDescriptor_t  dxdesc, void * dx, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnDropoutBackward");

cudnnStatus_t (*lcudnnOpsTrainVersionCheck) () =
	(cudnnStatus_t (*) ()) dlsym(cudnn_handle, "cudnnOpsTrainVersionCheck");

cudnnStatus_t (*lcudnnCreateRNNDescriptor) (cudnnRNNDescriptor_t * rnnDesc) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t * rnnDesc)) dlsym(cudnn_handle, "cudnnCreateRNNDescriptor");

cudnnStatus_t (*lcudnnDestroyRNNDescriptor) (cudnnRNNDescriptor_t  rnnDesc) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc)) dlsym(cudnn_handle, "cudnnDestroyRNNDescriptor");

cudnnStatus_t (*lcudnnSetRNNDescriptor_v8) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t  algo, cudnnRNNMode_t  cellMode, cudnnRNNBiasMode_t  biasMode, cudnnDirectionMode_t  dirMode, cudnnRNNInputMode_t  inputMode, cudnnDataType_t  dataType, cudnnDataType_t  mathPrec, cudnnMathType_t  mathType, int32_t  inputSize, int32_t  hiddenSize, int32_t  projSize, int32_t  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, uint32_t  auxFlags) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t  algo, cudnnRNNMode_t  cellMode, cudnnRNNBiasMode_t  biasMode, cudnnDirectionMode_t  dirMode, cudnnRNNInputMode_t  inputMode, cudnnDataType_t  dataType, cudnnDataType_t  mathPrec, cudnnMathType_t  mathType, int32_t  inputSize, int32_t  hiddenSize, int32_t  projSize, int32_t  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, uint32_t  auxFlags)) dlsym(cudnn_handle, "cudnnSetRNNDescriptor_v8");

cudnnStatus_t (*lcudnnGetRNNDescriptor_v8) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t * algo, cudnnRNNMode_t * cellMode, cudnnRNNBiasMode_t * biasMode, cudnnDirectionMode_t * dirMode, cudnnRNNInputMode_t * inputMode, cudnnDataType_t * dataType, cudnnDataType_t * mathPrec, cudnnMathType_t * mathType, int32_t * inputSize, int32_t * hiddenSize, int32_t * projSize, int32_t * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, uint32_t * auxFlags) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t * algo, cudnnRNNMode_t * cellMode, cudnnRNNBiasMode_t * biasMode, cudnnDirectionMode_t * dirMode, cudnnRNNInputMode_t * inputMode, cudnnDataType_t * dataType, cudnnDataType_t * mathPrec, cudnnMathType_t * mathType, int32_t * inputSize, int32_t * hiddenSize, int32_t * projSize, int32_t * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, uint32_t * auxFlags)) dlsym(cudnn_handle, "cudnnGetRNNDescriptor_v8");

cudnnStatus_t (*lcudnnSetRNNDescriptor_v6) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  hiddenSize, const int  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, cudnnRNNInputMode_t  inputMode, cudnnDirectionMode_t  direction, cudnnRNNMode_t  cellMode, cudnnRNNAlgo_t  algo, cudnnDataType_t  mathPrec) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  hiddenSize, const int  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, cudnnRNNInputMode_t  inputMode, cudnnDirectionMode_t  direction, cudnnRNNMode_t  cellMode, cudnnRNNAlgo_t  algo, cudnnDataType_t  mathPrec)) dlsym(cudnn_handle, "cudnnSetRNNDescriptor_v6");

cudnnStatus_t (*lcudnnGetRNNDescriptor_v6) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int * hiddenSize, int * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, cudnnRNNInputMode_t * inputMode, cudnnDirectionMode_t * direction, cudnnRNNMode_t * cellMode, cudnnRNNAlgo_t * algo, cudnnDataType_t * mathPrec) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int * hiddenSize, int * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, cudnnRNNInputMode_t * inputMode, cudnnDirectionMode_t * direction, cudnnRNNMode_t * cellMode, cudnnRNNAlgo_t * algo, cudnnDataType_t * mathPrec)) dlsym(cudnn_handle, "cudnnGetRNNDescriptor_v6");

cudnnStatus_t (*lcudnnSetRNNMatrixMathType) (cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t  mType) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t  mType)) dlsym(cudnn_handle, "cudnnSetRNNMatrixMathType");

cudnnStatus_t (*lcudnnGetRNNMatrixMathType) (cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t * mType) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t * mType)) dlsym(cudnn_handle, "cudnnGetRNNMatrixMathType");

cudnnStatus_t (*lcudnnSetRNNBiasMode) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t  biasMode) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t  biasMode)) dlsym(cudnn_handle, "cudnnSetRNNBiasMode");

cudnnStatus_t (*lcudnnGetRNNBiasMode) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t * biasMode) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t * biasMode)) dlsym(cudnn_handle, "cudnnGetRNNBiasMode");

cudnnStatus_t (*lcudnnRNNSetClip_v8) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip)) dlsym(cudnn_handle, "cudnnRNNSetClip_v8");

cudnnStatus_t (*lcudnnRNNGetClip_v8) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip)) dlsym(cudnn_handle, "cudnnRNNGetClip_v8");

cudnnStatus_t (*lcudnnRNNSetClip) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip)) dlsym(cudnn_handle, "cudnnRNNSetClip");

cudnnStatus_t (*lcudnnRNNGetClip) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip)) dlsym(cudnn_handle, "cudnnRNNGetClip");

cudnnStatus_t (*lcudnnSetRNNProjectionLayers) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  recProjSize, const int  outProjSize) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  recProjSize, const int  outProjSize)) dlsym(cudnn_handle, "cudnnSetRNNProjectionLayers");

cudnnStatus_t (*lcudnnGetRNNProjectionLayers) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * recProjSize, int * outProjSize) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * recProjSize, int * outProjSize)) dlsym(cudnn_handle, "cudnnGetRNNProjectionLayers");

cudnnStatus_t (*lcudnnCreatePersistentRNNPlan) (cudnnRNNDescriptor_t  rnnDesc, const int  minibatch, const cudnnDataType_t  dataType, cudnnPersistentRNNPlan_t * plan) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, const int  minibatch, const cudnnDataType_t  dataType, cudnnPersistentRNNPlan_t * plan)) dlsym(cudnn_handle, "cudnnCreatePersistentRNNPlan");

cudnnStatus_t (*lcudnnDestroyPersistentRNNPlan) (cudnnPersistentRNNPlan_t  plan) =
	(cudnnStatus_t (*) (cudnnPersistentRNNPlan_t  plan)) dlsym(cudnn_handle, "cudnnDestroyPersistentRNNPlan");

cudnnStatus_t (*lcudnnSetPersistentRNNPlan) (cudnnRNNDescriptor_t  rnnDesc, cudnnPersistentRNNPlan_t  plan) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, cudnnPersistentRNNPlan_t  plan)) dlsym(cudnn_handle, "cudnnSetPersistentRNNPlan");

cudnnStatus_t (*lcudnnBuildRNNDynamic) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int  miniBatch) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int  miniBatch)) dlsym(cudnn_handle, "cudnnBuildRNNDynamic");

cudnnStatus_t (*lcudnnGetRNNWorkspaceSize) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetRNNWorkspaceSize");

cudnnStatus_t (*lcudnnGetRNNTrainingReserveSize) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetRNNTrainingReserveSize");

cudnnStatus_t (*lcudnnGetRNNTempSpaceSizes) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fwdMode, cudnnRNNDataDescriptor_t  xDesc, size_t * workSpaceSize, size_t * reserveSpaceSize) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fwdMode, cudnnRNNDataDescriptor_t  xDesc, size_t * workSpaceSize, size_t * reserveSpaceSize)) dlsym(cudnn_handle, "cudnnGetRNNTempSpaceSizes");

cudnnStatus_t (*lcudnnGetRNNParamsSize) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, cudnnDataType_t  dataType) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, cudnnDataType_t  dataType)) dlsym(cudnn_handle, "cudnnGetRNNParamsSize");

cudnnStatus_t (*lcudnnGetRNNWeightSpaceSize) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, size_t * weightSpaceSize) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, size_t * weightSpaceSize)) dlsym(cudnn_handle, "cudnnGetRNNWeightSpaceSize");

cudnnStatus_t (*lcudnnGetRNNLinLayerMatrixParams) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerMatDesc, void ** linLayerMat) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerMatDesc, void ** linLayerMat)) dlsym(cudnn_handle, "cudnnGetRNNLinLayerMatrixParams");

cudnnStatus_t (*lcudnnGetRNNLinLayerBiasParams) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerBiasDesc, void ** linLayerBias) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerBiasDesc, void ** linLayerBias)) dlsym(cudnn_handle, "cudnnGetRNNLinLayerBiasParams");

cudnnStatus_t (*lcudnnGetRNNWeightParams) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int32_t  pseudoLayer, size_t  weightSpaceSize, const void * weightSpace, int32_t  linLayerID, cudnnTensorDescriptor_t  mDesc, void ** mAddr, cudnnTensorDescriptor_t  bDesc, void ** bAddr) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int32_t  pseudoLayer, size_t  weightSpaceSize, const void * weightSpace, int32_t  linLayerID, cudnnTensorDescriptor_t  mDesc, void ** mAddr, cudnnTensorDescriptor_t  bDesc, void ** bAddr)) dlsym(cudnn_handle, "cudnnGetRNNWeightParams");

cudnnStatus_t (*lcudnnRNNForwardInference) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnRNNForwardInference");

cudnnStatus_t (*lcudnnSetRNNPaddingMode) (cudnnRNNDescriptor_t  rnnDesc, unsigned  paddingMode) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, unsigned  paddingMode)) dlsym(cudnn_handle, "cudnnSetRNNPaddingMode");

cudnnStatus_t (*lcudnnGetRNNPaddingMode) (cudnnRNNDescriptor_t  rnnDesc, unsigned * paddingMode) =
	(cudnnStatus_t (*) (cudnnRNNDescriptor_t  rnnDesc, unsigned * paddingMode)) dlsym(cudnn_handle, "cudnnGetRNNPaddingMode");

cudnnStatus_t (*lcudnnCreateRNNDataDescriptor) (cudnnRNNDataDescriptor_t * rnnDataDesc) =
	(cudnnStatus_t (*) (cudnnRNNDataDescriptor_t * rnnDataDesc)) dlsym(cudnn_handle, "cudnnCreateRNNDataDescriptor");

cudnnStatus_t (*lcudnnDestroyRNNDataDescriptor) (cudnnRNNDataDescriptor_t  rnnDataDesc) =
	(cudnnStatus_t (*) (cudnnRNNDataDescriptor_t  rnnDataDesc)) dlsym(cudnn_handle, "cudnnDestroyRNNDataDescriptor");

cudnnStatus_t (*lcudnnSetRNNDataDescriptor) (cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t  dataType, cudnnRNNDataLayout_t  layout, int  maxSeqLength, int  batchSize, int  vectorSize, const int  seqLengthArray[], void * paddingFill) =
	(cudnnStatus_t (*) (cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t  dataType, cudnnRNNDataLayout_t  layout, int  maxSeqLength, int  batchSize, int  vectorSize, const int  seqLengthArray[], void * paddingFill)) dlsym(cudnn_handle, "cudnnSetRNNDataDescriptor");

cudnnStatus_t (*lcudnnGetRNNDataDescriptor) (cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t * dataType, cudnnRNNDataLayout_t * layout, int * maxSeqLength, int * batchSize, int * vectorSize, int  arrayLengthRequested, int  seqLengthArray[], void * paddingFill) =
	(cudnnStatus_t (*) (cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t * dataType, cudnnRNNDataLayout_t * layout, int * maxSeqLength, int * batchSize, int * vectorSize, int  arrayLengthRequested, int  seqLengthArray[], void * paddingFill)) dlsym(cudnn_handle, "cudnnGetRNNDataDescriptor");

cudnnStatus_t (*lcudnnRNNForwardInferenceEx) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnRNNForwardInferenceEx");

cudnnStatus_t (*lcudnnRNNForward) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fwdMode, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnRNNDataDescriptor_t  yDesc, void * y, cudnnTensorDescriptor_t  hDesc, const void * hx, void * hy, cudnnTensorDescriptor_t  cDesc, const void * cx, void * cy, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fwdMode, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnRNNDataDescriptor_t  yDesc, void * y, cudnnTensorDescriptor_t  hDesc, const void * hx, void * hy, cudnnTensorDescriptor_t  cDesc, const void * cx, void * cy, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)) dlsym(cudnn_handle, "cudnnRNNForward");

cudnnStatus_t (*lcudnnSetRNNAlgorithmDescriptor) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnAlgorithmDescriptor_t  algoDesc) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnAlgorithmDescriptor_t  algoDesc)) dlsym(cudnn_handle, "cudnnSetRNNAlgorithmDescriptor");

cudnnStatus_t (*lcudnnGetRNNForwardInferenceAlgorithmMaxCount) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)) dlsym(cudnn_handle, "cudnnGetRNNForwardInferenceAlgorithmMaxCount");

cudnnStatus_t (*lcudnnFindRNNForwardInferenceAlgorithmEx) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnFindRNNForwardInferenceAlgorithmEx");

cudnnStatus_t (*lcudnnCreateSeqDataDescriptor) (cudnnSeqDataDescriptor_t * seqDataDesc) =
	(cudnnStatus_t (*) (cudnnSeqDataDescriptor_t * seqDataDesc)) dlsym(cudnn_handle, "cudnnCreateSeqDataDescriptor");

cudnnStatus_t (*lcudnnDestroySeqDataDescriptor) (cudnnSeqDataDescriptor_t  seqDataDesc) =
	(cudnnStatus_t (*) (cudnnSeqDataDescriptor_t  seqDataDesc)) dlsym(cudnn_handle, "cudnnDestroySeqDataDescriptor");

cudnnStatus_t (*lcudnnSetSeqDataDescriptor) (cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const cudnnSeqDataAxis_t  axes[], size_t  seqLengthArraySize, const int  seqLengthArray[], void * paddingFill) =
	(cudnnStatus_t (*) (cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const cudnnSeqDataAxis_t  axes[], size_t  seqLengthArraySize, const int  seqLengthArray[], void * paddingFill)) dlsym(cudnn_handle, "cudnnSetSeqDataDescriptor");

cudnnStatus_t (*lcudnnGetSeqDataDescriptor) (const cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t * dataType, int * nbDims, int  nbDimsRequested, int  dimA[], cudnnSeqDataAxis_t  axes[], size_t * seqLengthArraySize, size_t  seqLengthSizeRequested, int  seqLengthArray[], void * paddingFill) =
	(cudnnStatus_t (*) (const cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t * dataType, int * nbDims, int  nbDimsRequested, int  dimA[], cudnnSeqDataAxis_t  axes[], size_t * seqLengthArraySize, size_t  seqLengthSizeRequested, int  seqLengthArray[], void * paddingFill)) dlsym(cudnn_handle, "cudnnGetSeqDataDescriptor");

cudnnStatus_t (*lcudnnCreateAttnDescriptor) (cudnnAttnDescriptor_t * attnDesc) =
	(cudnnStatus_t (*) (cudnnAttnDescriptor_t * attnDesc)) dlsym(cudnn_handle, "cudnnCreateAttnDescriptor");

cudnnStatus_t (*lcudnnDestroyAttnDescriptor) (cudnnAttnDescriptor_t  attnDesc) =
	(cudnnStatus_t (*) (cudnnAttnDescriptor_t  attnDesc)) dlsym(cudnn_handle, "cudnnDestroyAttnDescriptor");

cudnnStatus_t (*lcudnnSetAttnDescriptor) (cudnnAttnDescriptor_t  attnDesc, unsigned  attnMode, int  nHeads, double  smScaler, cudnnDataType_t  dataType, cudnnDataType_t  computePrec, cudnnMathType_t  mathType, cudnnDropoutDescriptor_t  attnDropoutDesc, cudnnDropoutDescriptor_t  postDropoutDesc, int  qSize, int  kSize, int  vSize, int  qProjSize, int  kProjSize, int  vProjSize, int  oProjSize, int  qoMaxSeqLength, int  kvMaxSeqLength, int  maxBatchSize, int  maxBeamSize) =
	(cudnnStatus_t (*) (cudnnAttnDescriptor_t  attnDesc, unsigned  attnMode, int  nHeads, double  smScaler, cudnnDataType_t  dataType, cudnnDataType_t  computePrec, cudnnMathType_t  mathType, cudnnDropoutDescriptor_t  attnDropoutDesc, cudnnDropoutDescriptor_t  postDropoutDesc, int  qSize, int  kSize, int  vSize, int  qProjSize, int  kProjSize, int  vProjSize, int  oProjSize, int  qoMaxSeqLength, int  kvMaxSeqLength, int  maxBatchSize, int  maxBeamSize)) dlsym(cudnn_handle, "cudnnSetAttnDescriptor");

cudnnStatus_t (*lcudnnGetAttnDescriptor) (cudnnAttnDescriptor_t  attnDesc, unsigned * attnMode, int * nHeads, double * smScaler, cudnnDataType_t * dataType, cudnnDataType_t * computePrec, cudnnMathType_t * mathType, cudnnDropoutDescriptor_t * attnDropoutDesc, cudnnDropoutDescriptor_t * postDropoutDesc, int * qSize, int * kSize, int * vSize, int * qProjSize, int * kProjSize, int * vProjSize, int * oProjSize, int * qoMaxSeqLength, int * kvMaxSeqLength, int * maxBatchSize, int * maxBeamSize) =
	(cudnnStatus_t (*) (cudnnAttnDescriptor_t  attnDesc, unsigned * attnMode, int * nHeads, double * smScaler, cudnnDataType_t * dataType, cudnnDataType_t * computePrec, cudnnMathType_t * mathType, cudnnDropoutDescriptor_t * attnDropoutDesc, cudnnDropoutDescriptor_t * postDropoutDesc, int * qSize, int * kSize, int * vSize, int * qProjSize, int * kProjSize, int * vProjSize, int * oProjSize, int * qoMaxSeqLength, int * kvMaxSeqLength, int * maxBatchSize, int * maxBeamSize)) dlsym(cudnn_handle, "cudnnGetAttnDescriptor");

cudnnStatus_t (*lcudnnGetMultiHeadAttnBuffers) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, size_t * weightSizeInBytes, size_t * workSpaceSizeInBytes, size_t * reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, size_t * weightSizeInBytes, size_t * workSpaceSizeInBytes, size_t * reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnGetMultiHeadAttnBuffers");

cudnnStatus_t (*lcudnnGetMultiHeadAttnWeights) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnMultiHeadAttnWeightKind_t  wKind, size_t  weightSizeInBytes, const void * weights, cudnnTensorDescriptor_t  wDesc, void ** wAddr) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnMultiHeadAttnWeightKind_t  wKind, size_t  weightSizeInBytes, const void * weights, cudnnTensorDescriptor_t  wDesc, void ** wAddr)) dlsym(cudnn_handle, "cudnnGetMultiHeadAttnWeights");

cudnnStatus_t (*lcudnnMultiHeadAttnForward) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, int  currIdx, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsQO[], const int  devSeqLengthsKV[], const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const void * residuals, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  oDesc, void * out, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, int  currIdx, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsQO[], const int  devSeqLengthsKV[], const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const void * residuals, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  oDesc, void * out, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)) dlsym(cudnn_handle, "cudnnMultiHeadAttnForward");

cudnnStatus_t (*lcudnnAdvInferVersionCheck) () =
	(cudnnStatus_t (*) ()) dlsym(cudnn_handle, "cudnnAdvInferVersionCheck");

cudnnStatus_t (*lcudnnRNNForwardTraining) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnRNNForwardTraining");

cudnnStatus_t (*lcudnnRNNBackwardData) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnRNNBackwardData");

cudnnStatus_t (*lcudnnRNNBackwardData_v8) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  yDesc, const void * y, const void * dy, cudnnRNNDataDescriptor_t  xDesc, void * dx, cudnnTensorDescriptor_t  hDesc, const void * hx, const void * dhy, void * dhx, cudnnTensorDescriptor_t  cDesc, const void * cx, const void * dcy, void * dcx, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  yDesc, const void * y, const void * dy, cudnnRNNDataDescriptor_t  xDesc, void * dx, cudnnTensorDescriptor_t  hDesc, const void * hx, const void * dhy, void * dhx, cudnnTensorDescriptor_t  cDesc, const void * cx, const void * dcy, void * dcx, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)) dlsym(cudnn_handle, "cudnnRNNBackwardData_v8");

cudnnStatus_t (*lcudnnRNNBackwardWeights) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnRNNBackwardWeights");

cudnnStatus_t (*lcudnnRNNBackwardWeights_v8) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnWgradMode_t  addGrad, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnTensorDescriptor_t  hDesc, const void * hx, cudnnRNNDataDescriptor_t  yDesc, const void * y, size_t  weightSpaceSize, void * dweightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnWgradMode_t  addGrad, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnTensorDescriptor_t  hDesc, const void * hx, cudnnRNNDataDescriptor_t  yDesc, const void * y, size_t  weightSpaceSize, void * dweightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)) dlsym(cudnn_handle, "cudnnRNNBackwardWeights_v8");

cudnnStatus_t (*lcudnnRNNForwardTrainingEx) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnRNNForwardTrainingEx");

cudnnStatus_t (*lcudnnRNNBackwardDataEx) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  yDesc, const void * y, const cudnnRNNDataDescriptor_t  dyDesc, const void * dy, const cudnnRNNDataDescriptor_t  dcDesc, const void * dcAttn, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnRNNDataDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const cudnnRNNDataDescriptor_t  dkDesc, void * dkeys, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  yDesc, const void * y, const cudnnRNNDataDescriptor_t  dyDesc, const void * dy, const cudnnRNNDataDescriptor_t  dcDesc, const void * dcAttn, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnRNNDataDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const cudnnRNNDataDescriptor_t  dkDesc, void * dkeys, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnRNNBackwardDataEx");

cudnnStatus_t (*lcudnnRNNBackwardWeightsEx) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnRNNDataDescriptor_t  yDesc, const void * y, void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnRNNDataDescriptor_t  yDesc, const void * y, void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnRNNBackwardWeightsEx");

cudnnStatus_t (*lcudnnGetRNNForwardTrainingAlgorithmMaxCount) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)) dlsym(cudnn_handle, "cudnnGetRNNForwardTrainingAlgorithmMaxCount");

cudnnStatus_t (*lcudnnFindRNNForwardTrainingAlgorithmEx) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnFindRNNForwardTrainingAlgorithmEx");

cudnnStatus_t (*lcudnnGetRNNBackwardDataAlgorithmMaxCount) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)) dlsym(cudnn_handle, "cudnnGetRNNBackwardDataAlgorithmMaxCount");

cudnnStatus_t (*lcudnnFindRNNBackwardDataAlgorithmEx) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnFindRNNBackwardDataAlgorithmEx");

cudnnStatus_t (*lcudnnGetRNNBackwardWeightsAlgorithmMaxCount) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)) dlsym(cudnn_handle, "cudnnGetRNNBackwardWeightsAlgorithmMaxCount");

cudnnStatus_t (*lcudnnFindRNNBackwardWeightsAlgorithmEx) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, const void * workspace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, const void * workspace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnFindRNNBackwardWeightsAlgorithmEx");

cudnnStatus_t (*lcudnnMultiHeadAttnBackwardData) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsDQDO[], const int  devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t  doDesc, const void * dout, const cudnnSeqDataDescriptor_t  dqDesc, void * dqueries, const void * queries, const cudnnSeqDataDescriptor_t  dkDesc, void * dkeys, const void * keys, const cudnnSeqDataDescriptor_t  dvDesc, void * dvalues, const void * values, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsDQDO[], const int  devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t  doDesc, const void * dout, const cudnnSeqDataDescriptor_t  dqDesc, void * dqueries, const void * queries, const cudnnSeqDataDescriptor_t  dkDesc, void * dkeys, const void * keys, const cudnnSeqDataDescriptor_t  dvDesc, void * dvalues, const void * values, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)) dlsym(cudnn_handle, "cudnnMultiHeadAttnBackwardData");

cudnnStatus_t (*lcudnnMultiHeadAttnBackwardWeights) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnWgradMode_t  addGrad, const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  doDesc, const void * dout, size_t  weightSizeInBytes, const void * weights, void * dweights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnWgradMode_t  addGrad, const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  doDesc, const void * dout, size_t  weightSizeInBytes, const void * weights, void * dweights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)) dlsym(cudnn_handle, "cudnnMultiHeadAttnBackwardWeights");

cudnnStatus_t (*lcudnnCreateCTCLossDescriptor) (cudnnCTCLossDescriptor_t * ctcLossDesc) =
	(cudnnStatus_t (*) (cudnnCTCLossDescriptor_t * ctcLossDesc)) dlsym(cudnn_handle, "cudnnCreateCTCLossDescriptor");

cudnnStatus_t (*lcudnnSetCTCLossDescriptor) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType) =
	(cudnnStatus_t (*) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType)) dlsym(cudnn_handle, "cudnnSetCTCLossDescriptor");

cudnnStatus_t (*lcudnnSetCTCLossDescriptorEx) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode) =
	(cudnnStatus_t (*) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode)) dlsym(cudnn_handle, "cudnnSetCTCLossDescriptorEx");

cudnnStatus_t (*lcudnnSetCTCLossDescriptor_v8) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode, int  maxLabelLength) =
	(cudnnStatus_t (*) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode, int  maxLabelLength)) dlsym(cudnn_handle, "cudnnSetCTCLossDescriptor_v8");

cudnnStatus_t (*lcudnnGetCTCLossDescriptor) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType) =
	(cudnnStatus_t (*) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType)) dlsym(cudnn_handle, "cudnnGetCTCLossDescriptor");

cudnnStatus_t (*lcudnnGetCTCLossDescriptorEx) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode) =
	(cudnnStatus_t (*) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode)) dlsym(cudnn_handle, "cudnnGetCTCLossDescriptorEx");

cudnnStatus_t (*lcudnnGetCTCLossDescriptor_v8) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode, int * maxLabelLength) =
	(cudnnStatus_t (*) (cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode, int * maxLabelLength)) dlsym(cudnn_handle, "cudnnGetCTCLossDescriptor_v8");

cudnnStatus_t (*lcudnnDestroyCTCLossDescriptor) (cudnnCTCLossDescriptor_t  ctcLossDesc) =
	(cudnnStatus_t (*) (cudnnCTCLossDescriptor_t  ctcLossDesc)) dlsym(cudnn_handle, "cudnnDestroyCTCLossDescriptor");

cudnnStatus_t (*lcudnnCTCLoss) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  hostLabels[], const int  hostLabelLengths[], const int  hostInputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, void * workspace, size_t  workSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  hostLabels[], const int  hostLabelLengths[], const int  hostInputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, void * workspace, size_t  workSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnCTCLoss");

cudnnStatus_t (*lcudnnCTCLoss_v8) (cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  labels[], const int  labelLengths[], const int  inputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, size_t  workSpaceSizeInBytes, void * workspace) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  labels[], const int  labelLengths[], const int  inputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, size_t  workSpaceSizeInBytes, void * workspace)) dlsym(cudnn_handle, "cudnnCTCLoss_v8");

cudnnStatus_t (*lcudnnGetCTCLossWorkspaceSize) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, const int * labels, const int * labelLengths, const int * inputLengths, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, const int * labels, const int * labelLengths, const int * inputLengths, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetCTCLossWorkspaceSize");

cudnnStatus_t (*lcudnnGetCTCLossWorkspaceSize_v8) (cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetCTCLossWorkspaceSize_v8");

cudnnStatus_t (*lcudnnAdvTrainVersionCheck) () =
	(cudnnStatus_t (*) ()) dlsym(cudnn_handle, "cudnnAdvTrainVersionCheck");

cudnnStatus_t (*lcudnnCreateConvolutionDescriptor) (cudnnConvolutionDescriptor_t * convDesc) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t * convDesc)) dlsym(cudnn_handle, "cudnnCreateConvolutionDescriptor");

cudnnStatus_t (*lcudnnDestroyConvolutionDescriptor) (cudnnConvolutionDescriptor_t  convDesc) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t  convDesc)) dlsym(cudnn_handle, "cudnnDestroyConvolutionDescriptor");

cudnnStatus_t (*lcudnnSetConvolutionMathType) (cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t  mathType) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t  mathType)) dlsym(cudnn_handle, "cudnnSetConvolutionMathType");

cudnnStatus_t (*lcudnnGetConvolutionMathType) (cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t * mathType) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t * mathType)) dlsym(cudnn_handle, "cudnnGetConvolutionMathType");

cudnnStatus_t (*lcudnnSetConvolutionGroupCount) (cudnnConvolutionDescriptor_t  convDesc, int  groupCount) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t  convDesc, int  groupCount)) dlsym(cudnn_handle, "cudnnSetConvolutionGroupCount");

cudnnStatus_t (*lcudnnGetConvolutionGroupCount) (cudnnConvolutionDescriptor_t  convDesc, int * groupCount) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t  convDesc, int * groupCount)) dlsym(cudnn_handle, "cudnnGetConvolutionGroupCount");

cudnnStatus_t (*lcudnnSetConvolutionReorderType) (cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t  reorderType) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t  reorderType)) dlsym(cudnn_handle, "cudnnSetConvolutionReorderType");

cudnnStatus_t (*lcudnnGetConvolutionReorderType) (cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t * reorderType) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t * reorderType)) dlsym(cudnn_handle, "cudnnGetConvolutionReorderType");

cudnnStatus_t (*lcudnnSetConvolution2dDescriptor) (cudnnConvolutionDescriptor_t  convDesc, int  pad_h, int  pad_w, int  u, int  v, int  dilation_h, int  dilation_w, cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t  convDesc, int  pad_h, int  pad_w, int  u, int  v, int  dilation_h, int  dilation_w, cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType)) dlsym(cudnn_handle, "cudnnSetConvolution2dDescriptor");

cudnnStatus_t (*lcudnnGetConvolution2dDescriptor) (const cudnnConvolutionDescriptor_t  convDesc, int * pad_h, int * pad_w, int * u, int * v, int * dilation_h, int * dilation_w, cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType) =
	(cudnnStatus_t (*) (const cudnnConvolutionDescriptor_t  convDesc, int * pad_h, int * pad_w, int * u, int * v, int * dilation_h, int * dilation_w, cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType)) dlsym(cudnn_handle, "cudnnGetConvolution2dDescriptor");

cudnnStatus_t (*lcudnnSetConvolutionNdDescriptor) (cudnnConvolutionDescriptor_t  convDesc, int  arrayLength, const int  padA[], const int  filterStrideA[], const int  dilationA[], cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType) =
	(cudnnStatus_t (*) (cudnnConvolutionDescriptor_t  convDesc, int  arrayLength, const int  padA[], const int  filterStrideA[], const int  dilationA[], cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType)) dlsym(cudnn_handle, "cudnnSetConvolutionNdDescriptor");

cudnnStatus_t (*lcudnnGetConvolutionNdDescriptor) (const cudnnConvolutionDescriptor_t  convDesc, int  arrayLengthRequested, int * arrayLength, int  padA[], int  strideA[], int  dilationA[], cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType) =
	(cudnnStatus_t (*) (const cudnnConvolutionDescriptor_t  convDesc, int  arrayLengthRequested, int * arrayLength, int  padA[], int  strideA[], int  dilationA[], cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType)) dlsym(cudnn_handle, "cudnnGetConvolutionNdDescriptor");

cudnnStatus_t (*lcudnnGetConvolution2dForwardOutputDim) (const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int * n, int * c, int * h, int * w) =
	(cudnnStatus_t (*) (const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int * n, int * c, int * h, int * w)) dlsym(cudnn_handle, "cudnnGetConvolution2dForwardOutputDim");

cudnnStatus_t (*lcudnnGetConvolutionNdForwardOutputDim) (const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int  nbDims, int  tensorOuputDimA[]) =
	(cudnnStatus_t (*) (const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int  nbDims, int  tensorOuputDimA[])) dlsym(cudnn_handle, "cudnnGetConvolutionNdForwardOutputDim");

cudnnStatus_t (*lcudnnGetConvolutionForwardAlgorithmMaxCount) (cudnnHandle_t  handle, int * count) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, int * count)) dlsym(cudnn_handle, "cudnnGetConvolutionForwardAlgorithmMaxCount");

cudnnStatus_t (*lcudnnGetConvolutionForwardAlgorithm_v7) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnFilterDescriptor_t  filterDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  destDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnFilterDescriptor_t  filterDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  destDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults)) dlsym(cudnn_handle, "cudnnGetConvolutionForwardAlgorithm_v7");

cudnnStatus_t (*lcudnnFindConvolutionForwardAlgorithm) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults)) dlsym(cudnn_handle, "cudnnFindConvolutionForwardAlgorithm");

cudnnStatus_t (*lcudnnFindConvolutionForwardAlgorithmEx) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, void * y, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, void * y, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnFindConvolutionForwardAlgorithmEx");

cudnnStatus_t (*lcudnnIm2Col) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, void * colBuffer) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, void * colBuffer)) dlsym(cudnn_handle, "cudnnIm2Col");

cudnnStatus_t (*lcudnnReorderFilterAndBias) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, cudnnReorderType_t  reorderType, const void * filterData, void * reorderedFilterData, int  reorderBias, const void * biasData, void * reorderedBiasData) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, cudnnReorderType_t  reorderType, const void * filterData, void * reorderedFilterData, int  reorderBias, const void * biasData, void * reorderedBiasData)) dlsym(cudnn_handle, "cudnnReorderFilterAndBias");

cudnnStatus_t (*lcudnnGetConvolutionForwardWorkspaceSize) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, cudnnConvolutionFwdAlgo_t  algo, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, cudnnConvolutionFwdAlgo_t  algo, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetConvolutionForwardWorkspaceSize");

cudnnStatus_t (*lcudnnConvolutionForward) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)) dlsym(cudnn_handle, "cudnnConvolutionForward");

cudnnStatus_t (*lcudnnConvolutionBiasActivationForward) (cudnnHandle_t  handle, const void * alpha1, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * alpha2, const cudnnTensorDescriptor_t  zDesc, const void * z, const cudnnTensorDescriptor_t  biasDesc, const void * bias, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const void * alpha1, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * alpha2, const cudnnTensorDescriptor_t  zDesc, const void * z, const cudnnTensorDescriptor_t  biasDesc, const void * bias, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y)) dlsym(cudnn_handle, "cudnnConvolutionBiasActivationForward");

cudnnStatus_t (*lcudnnGetConvolutionBackwardDataAlgorithmMaxCount) (cudnnHandle_t  handle, int * count) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, int * count)) dlsym(cudnn_handle, "cudnnGetConvolutionBackwardDataAlgorithmMaxCount");

cudnnStatus_t (*lcudnnFindConvolutionBackwardDataAlgorithm) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults)) dlsym(cudnn_handle, "cudnnFindConvolutionBackwardDataAlgorithm");

cudnnStatus_t (*lcudnnFindConvolutionBackwardDataAlgorithmEx) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, void * dx, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, void * dx, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnFindConvolutionBackwardDataAlgorithmEx");

cudnnStatus_t (*lcudnnGetConvolutionBackwardDataAlgorithm_v7) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults)) dlsym(cudnn_handle, "cudnnGetConvolutionBackwardDataAlgorithm_v7");

cudnnStatus_t (*lcudnnGetConvolutionBackwardDataWorkspaceSize) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, cudnnConvolutionBwdDataAlgo_t  algo, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, cudnnConvolutionBwdDataAlgo_t  algo, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetConvolutionBackwardDataWorkspaceSize");

cudnnStatus_t (*lcudnnConvolutionBackwardData) (cudnnHandle_t  handle, const void * alpha, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdDataAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const void * alpha, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdDataAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)) dlsym(cudnn_handle, "cudnnConvolutionBackwardData");

cudnnStatus_t (*lcudnnGetFoldedConvBackwardDataDescriptors) (const cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const cudnnTensorFormat_t  transformFormat, cudnnFilterDescriptor_t  foldedFilterDesc, cudnnTensorDescriptor_t  paddedDiffDesc, cudnnConvolutionDescriptor_t  foldedConvDesc, cudnnTensorDescriptor_t  foldedGradDesc, cudnnTensorTransformDescriptor_t  filterFoldTransDesc, cudnnTensorTransformDescriptor_t  diffPadTransDesc, cudnnTensorTransformDescriptor_t  gradFoldTransDesc, cudnnTensorTransformDescriptor_t  gradUnfoldTransDesc) =
	(cudnnStatus_t (*) (const cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const cudnnTensorFormat_t  transformFormat, cudnnFilterDescriptor_t  foldedFilterDesc, cudnnTensorDescriptor_t  paddedDiffDesc, cudnnConvolutionDescriptor_t  foldedConvDesc, cudnnTensorDescriptor_t  foldedGradDesc, cudnnTensorTransformDescriptor_t  filterFoldTransDesc, cudnnTensorTransformDescriptor_t  diffPadTransDesc, cudnnTensorTransformDescriptor_t  gradFoldTransDesc, cudnnTensorTransformDescriptor_t  gradUnfoldTransDesc)) dlsym(cudnn_handle, "cudnnGetFoldedConvBackwardDataDescriptors");

cudnnStatus_t (*lcudnnCnnInferVersionCheck) () =
	(cudnnStatus_t (*) ()) dlsym(cudnn_handle, "cudnnCnnInferVersionCheck");

cudnnStatus_t (*lcudnnGetConvolutionBackwardFilterAlgorithmMaxCount) (cudnnHandle_t  handle, int * count) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, int * count)) dlsym(cudnn_handle, "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");

cudnnStatus_t (*lcudnnFindConvolutionBackwardFilterAlgorithm) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults)) dlsym(cudnn_handle, "cudnnFindConvolutionBackwardFilterAlgorithm");

cudnnStatus_t (*lcudnnFindConvolutionBackwardFilterAlgorithmEx) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * y, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, void * dw, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * y, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, void * dw, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)) dlsym(cudnn_handle, "cudnnFindConvolutionBackwardFilterAlgorithmEx");

cudnnStatus_t (*lcudnnGetConvolutionBackwardFilterAlgorithm_v7) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults)) dlsym(cudnn_handle, "cudnnGetConvolutionBackwardFilterAlgorithm_v7");

cudnnStatus_t (*lcudnnGetConvolutionBackwardFilterWorkspaceSize) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, cudnnConvolutionBwdFilterAlgo_t  algo, size_t * sizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, cudnnConvolutionBwdFilterAlgo_t  algo, size_t * sizeInBytes)) dlsym(cudnn_handle, "cudnnGetConvolutionBackwardFilterWorkspaceSize");

cudnnStatus_t (*lcudnnConvolutionBackwardFilter) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdFilterAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnFilterDescriptor_t  dwDesc, void * dw) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdFilterAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnFilterDescriptor_t  dwDesc, void * dw)) dlsym(cudnn_handle, "cudnnConvolutionBackwardFilter");

cudnnStatus_t (*lcudnnConvolutionBackwardBias) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dbDesc, void * db) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dbDesc, void * db)) dlsym(cudnn_handle, "cudnnConvolutionBackwardBias");

cudnnStatus_t (*lcudnnCreateFusedOpsConstParamPack) (cudnnFusedOpsConstParamPack_t * constPack, cudnnFusedOps_t  ops) =
	(cudnnStatus_t (*) (cudnnFusedOpsConstParamPack_t * constPack, cudnnFusedOps_t  ops)) dlsym(cudnn_handle, "cudnnCreateFusedOpsConstParamPack");

cudnnStatus_t (*lcudnnDestroyFusedOpsConstParamPack) (cudnnFusedOpsConstParamPack_t  constPack) =
	(cudnnStatus_t (*) (cudnnFusedOpsConstParamPack_t  constPack)) dlsym(cudnn_handle, "cudnnDestroyFusedOpsConstParamPack");

cudnnStatus_t (*lcudnnSetFusedOpsConstParamPackAttribute) (cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, const void * param) =
	(cudnnStatus_t (*) (cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, const void * param)) dlsym(cudnn_handle, "cudnnSetFusedOpsConstParamPackAttribute");

cudnnStatus_t (*lcudnnGetFusedOpsConstParamPackAttribute) (const cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, void * param, int * isNULL) =
	(cudnnStatus_t (*) (const cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, void * param, int * isNULL)) dlsym(cudnn_handle, "cudnnGetFusedOpsConstParamPackAttribute");

cudnnStatus_t (*lcudnnCreateFusedOpsVariantParamPack) (cudnnFusedOpsVariantParamPack_t * varPack, cudnnFusedOps_t  ops) =
	(cudnnStatus_t (*) (cudnnFusedOpsVariantParamPack_t * varPack, cudnnFusedOps_t  ops)) dlsym(cudnn_handle, "cudnnCreateFusedOpsVariantParamPack");

cudnnStatus_t (*lcudnnDestroyFusedOpsVariantParamPack) (cudnnFusedOpsVariantParamPack_t  varPack) =
	(cudnnStatus_t (*) (cudnnFusedOpsVariantParamPack_t  varPack)) dlsym(cudnn_handle, "cudnnDestroyFusedOpsVariantParamPack");

cudnnStatus_t (*lcudnnSetFusedOpsVariantParamPackAttribute) (cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr) =
	(cudnnStatus_t (*) (cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr)) dlsym(cudnn_handle, "cudnnSetFusedOpsVariantParamPackAttribute");

cudnnStatus_t (*lcudnnGetFusedOpsVariantParamPackAttribute) (const cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr) =
	(cudnnStatus_t (*) (const cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr)) dlsym(cudnn_handle, "cudnnGetFusedOpsVariantParamPackAttribute");

cudnnStatus_t (*lcudnnCreateFusedOpsPlan) (cudnnFusedOpsPlan_t * plan, cudnnFusedOps_t  ops) =
	(cudnnStatus_t (*) (cudnnFusedOpsPlan_t * plan, cudnnFusedOps_t  ops)) dlsym(cudnn_handle, "cudnnCreateFusedOpsPlan");

cudnnStatus_t (*lcudnnDestroyFusedOpsPlan) (cudnnFusedOpsPlan_t  plan) =
	(cudnnStatus_t (*) (cudnnFusedOpsPlan_t  plan)) dlsym(cudnn_handle, "cudnnDestroyFusedOpsPlan");

cudnnStatus_t (*lcudnnMakeFusedOpsPlan) (cudnnHandle_t  handle, cudnnFusedOpsPlan_t  plan, const cudnnFusedOpsConstParamPack_t  constPack, size_t * workspaceSizeInBytes) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnFusedOpsPlan_t  plan, const cudnnFusedOpsConstParamPack_t  constPack, size_t * workspaceSizeInBytes)) dlsym(cudnn_handle, "cudnnMakeFusedOpsPlan");

cudnnStatus_t (*lcudnnFusedOpsExecute) (cudnnHandle_t  handle, const cudnnFusedOpsPlan_t  plan, cudnnFusedOpsVariantParamPack_t  varPack) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, const cudnnFusedOpsPlan_t  plan, cudnnFusedOpsVariantParamPack_t  varPack)) dlsym(cudnn_handle, "cudnnFusedOpsExecute");

cudnnStatus_t (*lcudnnCnnTrainVersionCheck) () =
	(cudnnStatus_t (*) ()) dlsym(cudnn_handle, "cudnnCnnTrainVersionCheck");

cudnnStatus_t (*lcudnnBackendCreateDescriptor) (cudnnBackendDescriptorType_t  descriptorType, cudnnBackendDescriptor_t * descriptor) =
	(cudnnStatus_t (*) (cudnnBackendDescriptorType_t  descriptorType, cudnnBackendDescriptor_t * descriptor)) dlsym(cudnn_handle, "cudnnBackendCreateDescriptor");

cudnnStatus_t (*lcudnnBackendDestroyDescriptor) (cudnnBackendDescriptor_t  descriptor) =
	(cudnnStatus_t (*) (cudnnBackendDescriptor_t  descriptor)) dlsym(cudnn_handle, "cudnnBackendDestroyDescriptor");

cudnnStatus_t (*lcudnnBackendInitialize) (cudnnBackendDescriptor_t  descriptor) =
	(cudnnStatus_t (*) (cudnnBackendDescriptor_t  descriptor)) dlsym(cudnn_handle, "cudnnBackendInitialize");

cudnnStatus_t (*lcudnnBackendFinalize) (cudnnBackendDescriptor_t  descriptor) =
	(cudnnStatus_t (*) (cudnnBackendDescriptor_t  descriptor)) dlsym(cudnn_handle, "cudnnBackendFinalize");

cudnnStatus_t (*lcudnnBackendSetAttribute) (cudnnBackendDescriptor_t  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  elementCount, const void * arrayOfElements) =
	(cudnnStatus_t (*) (cudnnBackendDescriptor_t  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  elementCount, const void * arrayOfElements)) dlsym(cudnn_handle, "cudnnBackendSetAttribute");

cudnnStatus_t (*lcudnnBackendGetAttribute) (cudnnBackendDescriptor_t const  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  requestedElementCount, int64_t * elementCount, void * arrayOfElements) =
	(cudnnStatus_t (*) (cudnnBackendDescriptor_t const  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  requestedElementCount, int64_t * elementCount, void * arrayOfElements)) dlsym(cudnn_handle, "cudnnBackendGetAttribute");

cudnnStatus_t (*lcudnnBackendExecute) (cudnnHandle_t  handle, cudnnBackendDescriptor_t  executionPlan, cudnnBackendDescriptor_t  variantPack) =
	(cudnnStatus_t (*) (cudnnHandle_t  handle, cudnnBackendDescriptor_t  executionPlan, cudnnBackendDescriptor_t  variantPack)) dlsym(cudnn_handle, "cudnnBackendExecute");

cublasStatus_t (*lcublasCreate_v2) (cublasHandle_t*  handle) =
	(cublasStatus_t (*) (cublasHandle_t*  handle)) dlsym(cublas_handle, "cublasCreate_v2");

cublasStatus_t (*lcublasDestroy_v2) (cublasHandle_t  handle) =
	(cublasStatus_t (*) (cublasHandle_t  handle)) dlsym(cublas_handle, "cublasDestroy_v2");

cublasStatus_t (*lcublasGetVersion_v2) (cublasHandle_t  handle, int*  version) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int*  version)) dlsym(cublas_handle, "cublasGetVersion_v2");

cublasStatus_t (*lcublasGetProperty) (libraryPropertyType  type, int*  value) =
	(cublasStatus_t (*) (libraryPropertyType  type, int*  value)) dlsym(cublas_handle, "cublasGetProperty");

size_t (*lcublasGetCudartVersion) () =
	(size_t (*) ()) dlsym(cublas_handle, "cublasGetCudartVersion");

cublasStatus_t (*lcublasSetWorkspace_v2) (cublasHandle_t  handle, void*  workspace, size_t  workspaceSizeInBytes) =
	(cublasStatus_t (*) (cublasHandle_t  handle, void*  workspace, size_t  workspaceSizeInBytes)) dlsym(cublas_handle, "cublasSetWorkspace_v2");

cublasStatus_t (*lcublasSetStream_v2) (cublasHandle_t  handle, cudaStream_t  streamId) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cudaStream_t  streamId)) dlsym(cublas_handle, "cublasSetStream_v2");

cublasStatus_t (*lcublasGetStream_v2) (cublasHandle_t  handle, cudaStream_t*  streamId) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cudaStream_t*  streamId)) dlsym(cublas_handle, "cublasGetStream_v2");

cublasStatus_t (*lcublasGetPointerMode_v2) (cublasHandle_t  handle, cublasPointerMode_t*  mode) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasPointerMode_t*  mode)) dlsym(cublas_handle, "cublasGetPointerMode_v2");

cublasStatus_t (*lcublasSetPointerMode_v2) (cublasHandle_t  handle, cublasPointerMode_t  mode) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasPointerMode_t  mode)) dlsym(cublas_handle, "cublasSetPointerMode_v2");

cublasStatus_t (*lcublasGetAtomicsMode) (cublasHandle_t  handle, cublasAtomicsMode_t*  mode) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasAtomicsMode_t*  mode)) dlsym(cublas_handle, "cublasGetAtomicsMode");

cublasStatus_t (*lcublasSetAtomicsMode) (cublasHandle_t  handle, cublasAtomicsMode_t  mode) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasAtomicsMode_t  mode)) dlsym(cublas_handle, "cublasSetAtomicsMode");

cublasStatus_t (*lcublasGetMathMode) (cublasHandle_t  handle, cublasMath_t*  mode) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasMath_t*  mode)) dlsym(cublas_handle, "cublasGetMathMode");

cublasStatus_t (*lcublasSetMathMode) (cublasHandle_t  handle, cublasMath_t  mode) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasMath_t  mode)) dlsym(cublas_handle, "cublasSetMathMode");

cublasStatus_t (*lcublasGetSmCountTarget) (cublasHandle_t  handle, int*  smCountTarget) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int*  smCountTarget)) dlsym(cublas_handle, "cublasGetSmCountTarget");

cublasStatus_t (*lcublasSetSmCountTarget) (cublasHandle_t  handle, int  smCountTarget) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  smCountTarget)) dlsym(cublas_handle, "cublasSetSmCountTarget");

const char* (*lcublasGetStatusName) (cublasStatus_t  status) =
	(const char* (*) (cublasStatus_t  status)) dlsym(cublas_handle, "cublasGetStatusName");

const char* (*lcublasGetStatusString) (cublasStatus_t  status) =
	(const char* (*) (cublasStatus_t  status)) dlsym(cublas_handle, "cublasGetStatusString");

cublasStatus_t (*lcublasLoggerConfigure) (int  logIsOn, int  logToStdOut, int  logToStdErr, const char*  logFileName) =
	(cublasStatus_t (*) (int  logIsOn, int  logToStdOut, int  logToStdErr, const char*  logFileName)) dlsym(cublas_handle, "cublasLoggerConfigure");

cublasStatus_t (*lcublasSetLoggerCallback) (cublasLogCallback  userCallback) =
	(cublasStatus_t (*) (cublasLogCallback  userCallback)) dlsym(cublas_handle, "cublasSetLoggerCallback");

cublasStatus_t (*lcublasGetLoggerCallback) (cublasLogCallback*  userCallback) =
	(cublasStatus_t (*) (cublasLogCallback*  userCallback)) dlsym(cublas_handle, "cublasGetLoggerCallback");

cublasStatus_t (*lcublasSetVector) (int  n, int  elemSize, const void*  x, int  incx, void*  devicePtr, int  incy) =
	(cublasStatus_t (*) (int  n, int  elemSize, const void*  x, int  incx, void*  devicePtr, int  incy)) dlsym(cublas_handle, "cublasSetVector");

cublasStatus_t (*lcublasSetVector_64) (int64_t  n, int64_t  elemSize, const void*  x, int64_t  incx, void*  devicePtr, int64_t  incy) =
	(cublasStatus_t (*) (int64_t  n, int64_t  elemSize, const void*  x, int64_t  incx, void*  devicePtr, int64_t  incy)) dlsym(cublas_handle, "cublasSetVector_64");

cublasStatus_t (*lcublasGetVector) (int  n, int  elemSize, const void*  x, int  incx, void*  y, int  incy) =
	(cublasStatus_t (*) (int  n, int  elemSize, const void*  x, int  incx, void*  y, int  incy)) dlsym(cublas_handle, "cublasGetVector");

cublasStatus_t (*lcublasGetVector_64) (int64_t  n, int64_t  elemSize, const void*  x, int64_t  incx, void*  y, int64_t  incy) =
	(cublasStatus_t (*) (int64_t  n, int64_t  elemSize, const void*  x, int64_t  incx, void*  y, int64_t  incy)) dlsym(cublas_handle, "cublasGetVector_64");

cublasStatus_t (*lcublasSetMatrix) (int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb) =
	(cublasStatus_t (*) (int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)) dlsym(cublas_handle, "cublasSetMatrix");

cublasStatus_t (*lcublasSetMatrix_64) (int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb) =
	(cublasStatus_t (*) (int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb)) dlsym(cublas_handle, "cublasSetMatrix_64");

cublasStatus_t (*lcublasGetMatrix) (int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb) =
	(cublasStatus_t (*) (int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)) dlsym(cublas_handle, "cublasGetMatrix");

cublasStatus_t (*lcublasGetMatrix_64) (int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb) =
	(cublasStatus_t (*) (int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb)) dlsym(cublas_handle, "cublasGetMatrix_64");

cublasStatus_t (*lcublasSetVectorAsync) (int  n, int  elemSize, const void*  hostPtr, int  incx, void*  devicePtr, int  incy, cudaStream_t  stream) =
	(cublasStatus_t (*) (int  n, int  elemSize, const void*  hostPtr, int  incx, void*  devicePtr, int  incy, cudaStream_t  stream)) dlsym(cublas_handle, "cublasSetVectorAsync");

cublasStatus_t (*lcublasSetVectorAsync_64) (int64_t  n, int64_t  elemSize, const void*  hostPtr, int64_t  incx, void*  devicePtr, int64_t  incy, cudaStream_t  stream) =
	(cublasStatus_t (*) (int64_t  n, int64_t  elemSize, const void*  hostPtr, int64_t  incx, void*  devicePtr, int64_t  incy, cudaStream_t  stream)) dlsym(cublas_handle, "cublasSetVectorAsync_64");

cublasStatus_t (*lcublasGetVectorAsync) (int  n, int  elemSize, const void*  devicePtr, int  incx, void*  hostPtr, int  incy, cudaStream_t  stream) =
	(cublasStatus_t (*) (int  n, int  elemSize, const void*  devicePtr, int  incx, void*  hostPtr, int  incy, cudaStream_t  stream)) dlsym(cublas_handle, "cublasGetVectorAsync");

cublasStatus_t (*lcublasGetVectorAsync_64) (int64_t  n, int64_t  elemSize, const void*  devicePtr, int64_t  incx, void*  hostPtr, int64_t  incy, cudaStream_t  stream) =
	(cublasStatus_t (*) (int64_t  n, int64_t  elemSize, const void*  devicePtr, int64_t  incx, void*  hostPtr, int64_t  incy, cudaStream_t  stream)) dlsym(cublas_handle, "cublasGetVectorAsync_64");

cublasStatus_t (*lcublasSetMatrixAsync) (int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream) =
	(cublasStatus_t (*) (int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)) dlsym(cublas_handle, "cublasSetMatrixAsync");

cublasStatus_t (*lcublasSetMatrixAsync_64) (int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb, cudaStream_t  stream) =
	(cublasStatus_t (*) (int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb, cudaStream_t  stream)) dlsym(cublas_handle, "cublasSetMatrixAsync_64");

cublasStatus_t (*lcublasGetMatrixAsync) (int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream) =
	(cublasStatus_t (*) (int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)) dlsym(cublas_handle, "cublasGetMatrixAsync");

cublasStatus_t (*lcublasGetMatrixAsync_64) (int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb, cudaStream_t  stream) =
	(cublasStatus_t (*) (int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb, cudaStream_t  stream)) dlsym(cublas_handle, "cublasGetMatrixAsync_64");

void (*lcublasXerbla) (const char*  srName, int  info) =
	(void (*) (const char*  srName, int  info)) dlsym(cublas_handle, "cublasXerbla");

cublasStatus_t (*lcublasNrm2Ex) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executionType) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executionType)) dlsym(cublas_handle, "cublasNrm2Ex");

cublasStatus_t (*lcublasNrm2Ex_64) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  result, cudaDataType  resultType, cudaDataType  executionType) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  result, cudaDataType  resultType, cudaDataType  executionType)) dlsym(cublas_handle, "cublasNrm2Ex_64");

cublasStatus_t (*lcublasSnrm2_v2) (cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)) dlsym(cublas_handle, "cublasSnrm2_v2");

cublasStatus_t (*lcublasSnrm2_v2_64) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  result)) dlsym(cublas_handle, "cublasSnrm2_v2_64");

cublasStatus_t (*lcublasDnrm2_v2) (cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)) dlsym(cublas_handle, "cublasDnrm2_v2");

cublasStatus_t (*lcublasDnrm2_v2_64) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  result)) dlsym(cublas_handle, "cublasDnrm2_v2_64");

cublasStatus_t (*lcublasScnrm2_v2) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)) dlsym(cublas_handle, "cublasScnrm2_v2");

cublasStatus_t (*lcublasScnrm2_v2_64) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, float*  result)) dlsym(cublas_handle, "cublasScnrm2_v2_64");

cublasStatus_t (*lcublasDznrm2_v2) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)) dlsym(cublas_handle, "cublasDznrm2_v2");

cublasStatus_t (*lcublasDznrm2_v2_64) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, double*  result)) dlsym(cublas_handle, "cublasDznrm2_v2_64");

cublasStatus_t (*lcublasDotEx) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)) dlsym(cublas_handle, "cublasDotEx");

cublasStatus_t (*lcublasDotEx_64) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, const void*  y, cudaDataType  yType, int64_t  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, const void*  y, cudaDataType  yType, int64_t  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)) dlsym(cublas_handle, "cublasDotEx_64");

cublasStatus_t (*lcublasDotcEx) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)) dlsym(cublas_handle, "cublasDotcEx");

cublasStatus_t (*lcublasDotcEx_64) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, const void*  y, cudaDataType  yType, int64_t  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, const void*  y, cudaDataType  yType, int64_t  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)) dlsym(cublas_handle, "cublasDotcEx_64");

cublasStatus_t (*lcublasSdot_v2) (cublasHandle_t  handle, int  n, const float*  x, int  incx, const float*  y, int  incy, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float*  x, int  incx, const float*  y, int  incy, float*  result)) dlsym(cublas_handle, "cublasSdot_v2");

cublasStatus_t (*lcublasSdot_v2_64) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  result)) dlsym(cublas_handle, "cublasSdot_v2_64");

cublasStatus_t (*lcublasDdot_v2) (cublasHandle_t  handle, int  n, const double*  x, int  incx, const double*  y, int  incy, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double*  x, int  incx, const double*  y, int  incy, double*  result)) dlsym(cublas_handle, "cublasDdot_v2");

cublasStatus_t (*lcublasDdot_v2_64) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  result)) dlsym(cublas_handle, "cublasDdot_v2_64");

cublasStatus_t (*lcublasCdotu_v2) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)) dlsym(cublas_handle, "cublasCdotu_v2");

cublasStatus_t (*lcublasCdotu_v2_64) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  result)) dlsym(cublas_handle, "cublasCdotu_v2_64");

cublasStatus_t (*lcublasCdotc_v2) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)) dlsym(cublas_handle, "cublasCdotc_v2");

cublasStatus_t (*lcublasCdotc_v2_64) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  result)) dlsym(cublas_handle, "cublasCdotc_v2_64");

cublasStatus_t (*lcublasZdotu_v2) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)) dlsym(cublas_handle, "cublasZdotu_v2");

cublasStatus_t (*lcublasZdotu_v2_64) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  result)) dlsym(cublas_handle, "cublasZdotu_v2_64");

cublasStatus_t (*lcublasZdotc_v2) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)) dlsym(cublas_handle, "cublasZdotc_v2");

cublasStatus_t (*lcublasZdotc_v2_64) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  result)) dlsym(cublas_handle, "cublasZdotc_v2_64");

cublasStatus_t (*lcublasScalEx) (cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int  incx, cudaDataType  executionType) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int  incx, cudaDataType  executionType)) dlsym(cublas_handle, "cublasScalEx");

cublasStatus_t (*lcublasScalEx_64) (cublasHandle_t  handle, int64_t  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int64_t  incx, cudaDataType  executionType) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int64_t  incx, cudaDataType  executionType)) dlsym(cublas_handle, "cublasScalEx_64");

cublasStatus_t (*lcublasSscal_v2) (cublasHandle_t  handle, int  n, const float*  alpha, float*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float*  alpha, float*  x, int  incx)) dlsym(cublas_handle, "cublasSscal_v2");

cublasStatus_t (*lcublasSscal_v2_64) (cublasHandle_t  handle, int64_t  n, const float*  alpha, float*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const float*  alpha, float*  x, int64_t  incx)) dlsym(cublas_handle, "cublasSscal_v2_64");

cublasStatus_t (*lcublasDscal_v2) (cublasHandle_t  handle, int  n, const double*  alpha, double*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double*  alpha, double*  x, int  incx)) dlsym(cublas_handle, "cublasDscal_v2");

cublasStatus_t (*lcublasDscal_v2_64) (cublasHandle_t  handle, int64_t  n, const double*  alpha, double*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const double*  alpha, double*  x, int64_t  incx)) dlsym(cublas_handle, "cublasDscal_v2_64");

cublasStatus_t (*lcublasCscal_v2) (cublasHandle_t  handle, int  n, const cuComplex*  alpha, cuComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex*  alpha, cuComplex*  x, int  incx)) dlsym(cublas_handle, "cublasCscal_v2");

cublasStatus_t (*lcublasCscal_v2_64) (cublasHandle_t  handle, int64_t  n, const cuComplex*  alpha, cuComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuComplex*  alpha, cuComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasCscal_v2_64");

cublasStatus_t (*lcublasCsscal_v2) (cublasHandle_t  handle, int  n, const float*  alpha, cuComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float*  alpha, cuComplex*  x, int  incx)) dlsym(cublas_handle, "cublasCsscal_v2");

cublasStatus_t (*lcublasCsscal_v2_64) (cublasHandle_t  handle, int64_t  n, const float*  alpha, cuComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const float*  alpha, cuComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasCsscal_v2_64");

cublasStatus_t (*lcublasZscal_v2) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int  incx)) dlsym(cublas_handle, "cublasZscal_v2");

cublasStatus_t (*lcublasZscal_v2_64) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasZscal_v2_64");

cublasStatus_t (*lcublasZdscal_v2) (cublasHandle_t  handle, int  n, const double*  alpha, cuDoubleComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double*  alpha, cuDoubleComplex*  x, int  incx)) dlsym(cublas_handle, "cublasZdscal_v2");

cublasStatus_t (*lcublasZdscal_v2_64) (cublasHandle_t  handle, int64_t  n, const double*  alpha, cuDoubleComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const double*  alpha, cuDoubleComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasZdscal_v2_64");

cublasStatus_t (*lcublasAxpyEx) (cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasAxpyEx");

cublasStatus_t (*lcublasAxpyEx_64) (cublasHandle_t  handle, int64_t  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasAxpyEx_64");

cublasStatus_t (*lcublasSaxpy_v2) (cublasHandle_t  handle, int  n, const float*  alpha, const float*  x, int  incx, float*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float*  alpha, const float*  x, int  incx, float*  y, int  incy)) dlsym(cublas_handle, "cublasSaxpy_v2");

cublasStatus_t (*lcublasSaxpy_v2_64) (cublasHandle_t  handle, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  y, int64_t  incy)) dlsym(cublas_handle, "cublasSaxpy_v2_64");

cublasStatus_t (*lcublasDaxpy_v2) (cublasHandle_t  handle, int  n, const double*  alpha, const double*  x, int  incx, double*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double*  alpha, const double*  x, int  incx, double*  y, int  incy)) dlsym(cublas_handle, "cublasDaxpy_v2");

cublasStatus_t (*lcublasDaxpy_v2_64) (cublasHandle_t  handle, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  y, int64_t  incy)) dlsym(cublas_handle, "cublasDaxpy_v2_64");

cublasStatus_t (*lcublasCaxpy_v2) (cublasHandle_t  handle, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)) dlsym(cublas_handle, "cublasCaxpy_v2");

cublasStatus_t (*lcublasCaxpy_v2_64) (cublasHandle_t  handle, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasCaxpy_v2_64");

cublasStatus_t (*lcublasZaxpy_v2) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)) dlsym(cublas_handle, "cublasZaxpy_v2");

cublasStatus_t (*lcublasZaxpy_v2_64) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasZaxpy_v2_64");

cublasStatus_t (*lcublasCopyEx) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)) dlsym(cublas_handle, "cublasCopyEx");

cublasStatus_t (*lcublasCopyEx_64) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy)) dlsym(cublas_handle, "cublasCopyEx_64");

cublasStatus_t (*lcublasScopy_v2) (cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  y, int  incy)) dlsym(cublas_handle, "cublasScopy_v2");

cublasStatus_t (*lcublasScopy_v2_64) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  y, int64_t  incy)) dlsym(cublas_handle, "cublasScopy_v2_64");

cublasStatus_t (*lcublasDcopy_v2) (cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  y, int  incy)) dlsym(cublas_handle, "cublasDcopy_v2");

cublasStatus_t (*lcublasDcopy_v2_64) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  y, int64_t  incy)) dlsym(cublas_handle, "cublasDcopy_v2_64");

cublasStatus_t (*lcublasCcopy_v2) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, cuComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)) dlsym(cublas_handle, "cublasCcopy_v2");

cublasStatus_t (*lcublasCcopy_v2_64) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasCcopy_v2_64");

cublasStatus_t (*lcublasZcopy_v2) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)) dlsym(cublas_handle, "cublasZcopy_v2");

cublasStatus_t (*lcublasZcopy_v2_64) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasZcopy_v2_64");

cublasStatus_t (*lcublasSswap_v2) (cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy)) dlsym(cublas_handle, "cublasSswap_v2");

cublasStatus_t (*lcublasSswap_v2_64) (cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy)) dlsym(cublas_handle, "cublasSswap_v2_64");

cublasStatus_t (*lcublasDswap_v2) (cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy)) dlsym(cublas_handle, "cublasDswap_v2");

cublasStatus_t (*lcublasDswap_v2_64) (cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy)) dlsym(cublas_handle, "cublasDswap_v2_64");

cublasStatus_t (*lcublasCswap_v2) (cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy)) dlsym(cublas_handle, "cublasCswap_v2");

cublasStatus_t (*lcublasCswap_v2_64) (cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasCswap_v2_64");

cublasStatus_t (*lcublasZswap_v2) (cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)) dlsym(cublas_handle, "cublasZswap_v2");

cublasStatus_t (*lcublasZswap_v2_64) (cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasZswap_v2_64");

cublasStatus_t (*lcublasSwapEx) (cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)) dlsym(cublas_handle, "cublasSwapEx");

cublasStatus_t (*lcublasSwapEx_64) (cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy)) dlsym(cublas_handle, "cublasSwapEx_64");

cublasStatus_t (*lcublasIsamax_v2) (cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)) dlsym(cublas_handle, "cublasIsamax_v2");

cublasStatus_t (*lcublasIsamax_v2_64) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIsamax_v2_64");

cublasStatus_t (*lcublasIdamax_v2) (cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)) dlsym(cublas_handle, "cublasIdamax_v2");

cublasStatus_t (*lcublasIdamax_v2_64) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIdamax_v2_64");

cublasStatus_t (*lcublasIcamax_v2) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)) dlsym(cublas_handle, "cublasIcamax_v2");

cublasStatus_t (*lcublasIcamax_v2_64) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIcamax_v2_64");

cublasStatus_t (*lcublasIzamax_v2) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)) dlsym(cublas_handle, "cublasIzamax_v2");

cublasStatus_t (*lcublasIzamax_v2_64) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIzamax_v2_64");

cublasStatus_t (*lcublasIamaxEx) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)) dlsym(cublas_handle, "cublasIamaxEx");

cublasStatus_t (*lcublasIamaxEx_64) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIamaxEx_64");

cublasStatus_t (*lcublasIsamin_v2) (cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)) dlsym(cublas_handle, "cublasIsamin_v2");

cublasStatus_t (*lcublasIsamin_v2_64) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIsamin_v2_64");

cublasStatus_t (*lcublasIdamin_v2) (cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)) dlsym(cublas_handle, "cublasIdamin_v2");

cublasStatus_t (*lcublasIdamin_v2_64) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIdamin_v2_64");

cublasStatus_t (*lcublasIcamin_v2) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)) dlsym(cublas_handle, "cublasIcamin_v2");

cublasStatus_t (*lcublasIcamin_v2_64) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIcamin_v2_64");

cublasStatus_t (*lcublasIzamin_v2) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)) dlsym(cublas_handle, "cublasIzamin_v2");

cublasStatus_t (*lcublasIzamin_v2_64) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIzamin_v2_64");

cublasStatus_t (*lcublasIaminEx) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)) dlsym(cublas_handle, "cublasIaminEx");

cublasStatus_t (*lcublasIaminEx_64) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, int64_t*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, int64_t*  result)) dlsym(cublas_handle, "cublasIaminEx_64");

cublasStatus_t (*lcublasAsumEx) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasAsumEx");

cublasStatus_t (*lcublasAsumEx_64) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasAsumEx_64");

cublasStatus_t (*lcublasSasum_v2) (cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)) dlsym(cublas_handle, "cublasSasum_v2");

cublasStatus_t (*lcublasSasum_v2_64) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  result)) dlsym(cublas_handle, "cublasSasum_v2_64");

cublasStatus_t (*lcublasDasum_v2) (cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)) dlsym(cublas_handle, "cublasDasum_v2");

cublasStatus_t (*lcublasDasum_v2_64) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  result)) dlsym(cublas_handle, "cublasDasum_v2_64");

cublasStatus_t (*lcublasScasum_v2) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)) dlsym(cublas_handle, "cublasScasum_v2");

cublasStatus_t (*lcublasScasum_v2_64) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, float*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, float*  result)) dlsym(cublas_handle, "cublasScasum_v2_64");

cublasStatus_t (*lcublasDzasum_v2) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)) dlsym(cublas_handle, "cublasDzasum_v2");

cublasStatus_t (*lcublasDzasum_v2_64) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, double*  result) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, double*  result)) dlsym(cublas_handle, "cublasDzasum_v2_64");

cublasStatus_t (*lcublasSrot_v2) (cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  c, const float*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  c, const float*  s)) dlsym(cublas_handle, "cublasSrot_v2");

cublasStatus_t (*lcublasSrot_v2_64) (cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy, const float*  c, const float*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy, const float*  c, const float*  s)) dlsym(cublas_handle, "cublasSrot_v2_64");

cublasStatus_t (*lcublasDrot_v2) (cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  c, const double*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  c, const double*  s)) dlsym(cublas_handle, "cublasDrot_v2");

cublasStatus_t (*lcublasDrot_v2_64) (cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy, const double*  c, const double*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy, const double*  c, const double*  s)) dlsym(cublas_handle, "cublasDrot_v2_64");

cublasStatus_t (*lcublasCrot_v2) (cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const cuComplex*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const cuComplex*  s)) dlsym(cublas_handle, "cublasCrot_v2");

cublasStatus_t (*lcublasCrot_v2_64) (cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy, const float*  c, const cuComplex*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy, const float*  c, const cuComplex*  s)) dlsym(cublas_handle, "cublasCrot_v2_64");

cublasStatus_t (*lcublasCsrot_v2) (cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const float*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const float*  s)) dlsym(cublas_handle, "cublasCsrot_v2");

cublasStatus_t (*lcublasCsrot_v2_64) (cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy, const float*  c, const float*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy, const float*  c, const float*  s)) dlsym(cublas_handle, "cublasCsrot_v2_64");

cublasStatus_t (*lcublasZrot_v2) (cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const cuDoubleComplex*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const cuDoubleComplex*  s)) dlsym(cublas_handle, "cublasZrot_v2");

cublasStatus_t (*lcublasZrot_v2_64) (cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy, const double*  c, const cuDoubleComplex*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy, const double*  c, const cuDoubleComplex*  s)) dlsym(cublas_handle, "cublasZrot_v2_64");

cublasStatus_t (*lcublasZdrot_v2) (cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const double*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const double*  s)) dlsym(cublas_handle, "cublasZdrot_v2");

cublasStatus_t (*lcublasZdrot_v2_64) (cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy, const double*  c, const double*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy, const double*  c, const double*  s)) dlsym(cublas_handle, "cublasZdrot_v2_64");

cublasStatus_t (*lcublasRotEx) (cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasRotEx");

cublasStatus_t (*lcublasRotEx_64) (cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasRotEx_64");

cublasStatus_t (*lcublasSrotg_v2) (cublasHandle_t  handle, float*  a, float*  b, float*  c, float*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, float*  a, float*  b, float*  c, float*  s)) dlsym(cublas_handle, "cublasSrotg_v2");

cublasStatus_t (*lcublasDrotg_v2) (cublasHandle_t  handle, double*  a, double*  b, double*  c, double*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, double*  a, double*  b, double*  c, double*  s)) dlsym(cublas_handle, "cublasDrotg_v2");

cublasStatus_t (*lcublasCrotg_v2) (cublasHandle_t  handle, cuComplex*  a, cuComplex*  b, float*  c, cuComplex*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cuComplex*  a, cuComplex*  b, float*  c, cuComplex*  s)) dlsym(cublas_handle, "cublasCrotg_v2");

cublasStatus_t (*lcublasZrotg_v2) (cublasHandle_t  handle, cuDoubleComplex*  a, cuDoubleComplex*  b, double*  c, cuDoubleComplex*  s) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cuDoubleComplex*  a, cuDoubleComplex*  b, double*  c, cuDoubleComplex*  s)) dlsym(cublas_handle, "cublasZrotg_v2");

cublasStatus_t (*lcublasRotgEx) (cublasHandle_t  handle, void*  a, void*  b, cudaDataType  abType, void*  c, void*  s, cudaDataType  csType, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, void*  a, void*  b, cudaDataType  abType, void*  c, void*  s, cudaDataType  csType, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasRotgEx");

cublasStatus_t (*lcublasSrotm_v2) (cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  param) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  param)) dlsym(cublas_handle, "cublasSrotm_v2");

cublasStatus_t (*lcublasSrotm_v2_64) (cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy, const float*  param) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy, const float*  param)) dlsym(cublas_handle, "cublasSrotm_v2_64");

cublasStatus_t (*lcublasDrotm_v2) (cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  param) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  param)) dlsym(cublas_handle, "cublasDrotm_v2");

cublasStatus_t (*lcublasDrotm_v2_64) (cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy, const double*  param) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy, const double*  param)) dlsym(cublas_handle, "cublasDrotm_v2_64");

cublasStatus_t (*lcublasRotmEx) (cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasRotmEx");

cublasStatus_t (*lcublasRotmEx_64) (cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasRotmEx_64");

cublasStatus_t (*lcublasSrotmg_v2) (cublasHandle_t  handle, float*  d1, float*  d2, float*  x1, const float*  y1, float*  param) =
	(cublasStatus_t (*) (cublasHandle_t  handle, float*  d1, float*  d2, float*  x1, const float*  y1, float*  param)) dlsym(cublas_handle, "cublasSrotmg_v2");

cublasStatus_t (*lcublasDrotmg_v2) (cublasHandle_t  handle, double*  d1, double*  d2, double*  x1, const double*  y1, double*  param) =
	(cublasStatus_t (*) (cublasHandle_t  handle, double*  d1, double*  d2, double*  x1, const double*  y1, double*  param)) dlsym(cublas_handle, "cublasDrotmg_v2");

cublasStatus_t (*lcublasRotmgEx) (cublasHandle_t  handle, void*  d1, cudaDataType  d1Type, void*  d2, cudaDataType  d2Type, void*  x1, cudaDataType  x1Type, const void*  y1, cudaDataType  y1Type, void*  param, cudaDataType  paramType, cudaDataType  executiontype) =
	(cublasStatus_t (*) (cublasHandle_t  handle, void*  d1, cudaDataType  d1Type, void*  d2, cudaDataType  d2Type, void*  x1, cudaDataType  x1Type, const void*  y1, cudaDataType  y1Type, void*  param, cudaDataType  paramType, cudaDataType  executiontype)) dlsym(cublas_handle, "cublasRotmgEx");

cublasStatus_t (*lcublasSgemv_v2) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)) dlsym(cublas_handle, "cublasSgemv_v2");

cublasStatus_t (*lcublasSgemv_v2_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)) dlsym(cublas_handle, "cublasSgemv_v2_64");

cublasStatus_t (*lcublasDgemv_v2) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)) dlsym(cublas_handle, "cublasDgemv_v2");

cublasStatus_t (*lcublasDgemv_v2_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)) dlsym(cublas_handle, "cublasDgemv_v2_64");

cublasStatus_t (*lcublasCgemv_v2) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)) dlsym(cublas_handle, "cublasCgemv_v2");

cublasStatus_t (*lcublasCgemv_v2_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasCgemv_v2_64");

cublasStatus_t (*lcublasZgemv_v2) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)) dlsym(cublas_handle, "cublasZgemv_v2");

cublasStatus_t (*lcublasZgemv_v2_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasZgemv_v2_64");

cublasStatus_t (*lcublasSgbmv_v2) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)) dlsym(cublas_handle, "cublasSgbmv_v2");

cublasStatus_t (*lcublasSgbmv_v2_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)) dlsym(cublas_handle, "cublasSgbmv_v2_64");

cublasStatus_t (*lcublasDgbmv_v2) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)) dlsym(cublas_handle, "cublasDgbmv_v2");

cublasStatus_t (*lcublasDgbmv_v2_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)) dlsym(cublas_handle, "cublasDgbmv_v2_64");

cublasStatus_t (*lcublasCgbmv_v2) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)) dlsym(cublas_handle, "cublasCgbmv_v2");

cublasStatus_t (*lcublasCgbmv_v2_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasCgbmv_v2_64");

cublasStatus_t (*lcublasZgbmv_v2) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)) dlsym(cublas_handle, "cublasZgbmv_v2");

cublasStatus_t (*lcublasZgbmv_v2_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasZgbmv_v2_64");

cublasStatus_t (*lcublasStrmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)) dlsym(cublas_handle, "cublasStrmv_v2");

cublasStatus_t (*lcublasStrmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  A, int64_t  lda, float*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  A, int64_t  lda, float*  x, int64_t  incx)) dlsym(cublas_handle, "cublasStrmv_v2_64");

cublasStatus_t (*lcublasDtrmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)) dlsym(cublas_handle, "cublasDtrmv_v2");

cublasStatus_t (*lcublasDtrmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  A, int64_t  lda, double*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  A, int64_t  lda, double*  x, int64_t  incx)) dlsym(cublas_handle, "cublasDtrmv_v2_64");

cublasStatus_t (*lcublasCtrmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)) dlsym(cublas_handle, "cublasCtrmv_v2");

cublasStatus_t (*lcublasCtrmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasCtrmv_v2_64");

cublasStatus_t (*lcublasZtrmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)) dlsym(cublas_handle, "cublasZtrmv_v2");

cublasStatus_t (*lcublasZtrmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasZtrmv_v2_64");

cublasStatus_t (*lcublasStbmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)) dlsym(cublas_handle, "cublasStbmv_v2");

cublasStatus_t (*lcublasStbmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const float*  A, int64_t  lda, float*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const float*  A, int64_t  lda, float*  x, int64_t  incx)) dlsym(cublas_handle, "cublasStbmv_v2_64");

cublasStatus_t (*lcublasDtbmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)) dlsym(cublas_handle, "cublasDtbmv_v2");

cublasStatus_t (*lcublasDtbmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const double*  A, int64_t  lda, double*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const double*  A, int64_t  lda, double*  x, int64_t  incx)) dlsym(cublas_handle, "cublasDtbmv_v2_64");

cublasStatus_t (*lcublasCtbmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)) dlsym(cublas_handle, "cublasCtbmv_v2");

cublasStatus_t (*lcublasCtbmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasCtbmv_v2_64");

cublasStatus_t (*lcublasZtbmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)) dlsym(cublas_handle, "cublasZtbmv_v2");

cublasStatus_t (*lcublasZtbmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasZtbmv_v2_64");

cublasStatus_t (*lcublasStpmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)) dlsym(cublas_handle, "cublasStpmv_v2");

cublasStatus_t (*lcublasStpmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  AP, float*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  AP, float*  x, int64_t  incx)) dlsym(cublas_handle, "cublasStpmv_v2_64");

cublasStatus_t (*lcublasDtpmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)) dlsym(cublas_handle, "cublasDtpmv_v2");

cublasStatus_t (*lcublasDtpmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  AP, double*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  AP, double*  x, int64_t  incx)) dlsym(cublas_handle, "cublasDtpmv_v2_64");

cublasStatus_t (*lcublasCtpmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)) dlsym(cublas_handle, "cublasCtpmv_v2");

cublasStatus_t (*lcublasCtpmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  AP, cuComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  AP, cuComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasCtpmv_v2_64");

cublasStatus_t (*lcublasZtpmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)) dlsym(cublas_handle, "cublasZtpmv_v2");

cublasStatus_t (*lcublasZtpmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasZtpmv_v2_64");

cublasStatus_t (*lcublasStrsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)) dlsym(cublas_handle, "cublasStrsv_v2");

cublasStatus_t (*lcublasStrsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  A, int64_t  lda, float*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  A, int64_t  lda, float*  x, int64_t  incx)) dlsym(cublas_handle, "cublasStrsv_v2_64");

cublasStatus_t (*lcublasDtrsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)) dlsym(cublas_handle, "cublasDtrsv_v2");

cublasStatus_t (*lcublasDtrsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  A, int64_t  lda, double*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  A, int64_t  lda, double*  x, int64_t  incx)) dlsym(cublas_handle, "cublasDtrsv_v2_64");

cublasStatus_t (*lcublasCtrsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)) dlsym(cublas_handle, "cublasCtrsv_v2");

cublasStatus_t (*lcublasCtrsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasCtrsv_v2_64");

cublasStatus_t (*lcublasZtrsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)) dlsym(cublas_handle, "cublasZtrsv_v2");

cublasStatus_t (*lcublasZtrsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasZtrsv_v2_64");

cublasStatus_t (*lcublasStpsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)) dlsym(cublas_handle, "cublasStpsv_v2");

cublasStatus_t (*lcublasStpsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  AP, float*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  AP, float*  x, int64_t  incx)) dlsym(cublas_handle, "cublasStpsv_v2_64");

cublasStatus_t (*lcublasDtpsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)) dlsym(cublas_handle, "cublasDtpsv_v2");

cublasStatus_t (*lcublasDtpsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  AP, double*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  AP, double*  x, int64_t  incx)) dlsym(cublas_handle, "cublasDtpsv_v2_64");

cublasStatus_t (*lcublasCtpsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)) dlsym(cublas_handle, "cublasCtpsv_v2");

cublasStatus_t (*lcublasCtpsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  AP, cuComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  AP, cuComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasCtpsv_v2_64");

cublasStatus_t (*lcublasZtpsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)) dlsym(cublas_handle, "cublasZtpsv_v2");

cublasStatus_t (*lcublasZtpsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasZtpsv_v2_64");

cublasStatus_t (*lcublasStbsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)) dlsym(cublas_handle, "cublasStbsv_v2");

cublasStatus_t (*lcublasStbsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const float*  A, int64_t  lda, float*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const float*  A, int64_t  lda, float*  x, int64_t  incx)) dlsym(cublas_handle, "cublasStbsv_v2_64");

cublasStatus_t (*lcublasDtbsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)) dlsym(cublas_handle, "cublasDtbsv_v2");

cublasStatus_t (*lcublasDtbsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const double*  A, int64_t  lda, double*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const double*  A, int64_t  lda, double*  x, int64_t  incx)) dlsym(cublas_handle, "cublasDtbsv_v2_64");

cublasStatus_t (*lcublasCtbsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)) dlsym(cublas_handle, "cublasCtbsv_v2");

cublasStatus_t (*lcublasCtbsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasCtbsv_v2_64");

cublasStatus_t (*lcublasZtbsv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)) dlsym(cublas_handle, "cublasZtbsv_v2");

cublasStatus_t (*lcublasZtbsv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)) dlsym(cublas_handle, "cublasZtbsv_v2_64");

cublasStatus_t (*lcublasSsymv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)) dlsym(cublas_handle, "cublasSsymv_v2");

cublasStatus_t (*lcublasSsymv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)) dlsym(cublas_handle, "cublasSsymv_v2_64");

cublasStatus_t (*lcublasDsymv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)) dlsym(cublas_handle, "cublasDsymv_v2");

cublasStatus_t (*lcublasDsymv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)) dlsym(cublas_handle, "cublasDsymv_v2_64");

cublasStatus_t (*lcublasCsymv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)) dlsym(cublas_handle, "cublasCsymv_v2");

cublasStatus_t (*lcublasCsymv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasCsymv_v2_64");

cublasStatus_t (*lcublasZsymv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)) dlsym(cublas_handle, "cublasZsymv_v2");

cublasStatus_t (*lcublasZsymv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasZsymv_v2_64");

cublasStatus_t (*lcublasChemv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)) dlsym(cublas_handle, "cublasChemv_v2");

cublasStatus_t (*lcublasChemv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasChemv_v2_64");

cublasStatus_t (*lcublasZhemv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)) dlsym(cublas_handle, "cublasZhemv_v2");

cublasStatus_t (*lcublasZhemv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasZhemv_v2_64");

cublasStatus_t (*lcublasSsbmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)) dlsym(cublas_handle, "cublasSsbmv_v2");

cublasStatus_t (*lcublasSsbmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)) dlsym(cublas_handle, "cublasSsbmv_v2_64");

cublasStatus_t (*lcublasDsbmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)) dlsym(cublas_handle, "cublasDsbmv_v2");

cublasStatus_t (*lcublasDsbmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)) dlsym(cublas_handle, "cublasDsbmv_v2_64");

cublasStatus_t (*lcublasChbmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)) dlsym(cublas_handle, "cublasChbmv_v2");

cublasStatus_t (*lcublasChbmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasChbmv_v2_64");

cublasStatus_t (*lcublasZhbmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)) dlsym(cublas_handle, "cublasZhbmv_v2");

cublasStatus_t (*lcublasZhbmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasZhbmv_v2_64");

cublasStatus_t (*lcublasSspmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  AP, const float*  x, int  incx, const float*  beta, float*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  AP, const float*  x, int  incx, const float*  beta, float*  y, int  incy)) dlsym(cublas_handle, "cublasSspmv_v2");

cublasStatus_t (*lcublasSspmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  AP, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  AP, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)) dlsym(cublas_handle, "cublasSspmv_v2_64");

cublasStatus_t (*lcublasDspmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  AP, const double*  x, int  incx, const double*  beta, double*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  AP, const double*  x, int  incx, const double*  beta, double*  y, int  incy)) dlsym(cublas_handle, "cublasDspmv_v2");

cublasStatus_t (*lcublasDspmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  AP, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  AP, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)) dlsym(cublas_handle, "cublasDspmv_v2_64");

cublasStatus_t (*lcublasChpmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)) dlsym(cublas_handle, "cublasChpmv_v2");

cublasStatus_t (*lcublasChpmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasChpmv_v2_64");

cublasStatus_t (*lcublasZhpmv_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)) dlsym(cublas_handle, "cublasZhpmv_v2");

cublasStatus_t (*lcublasZhpmv_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)) dlsym(cublas_handle, "cublasZhpmv_v2_64");

cublasStatus_t (*lcublasSger_v2) (cublasHandle_t  handle, int  m, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)) dlsym(cublas_handle, "cublasSger_v2");

cublasStatus_t (*lcublasSger_v2_64) (cublasHandle_t  handle, int64_t  m, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  m, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  A, int64_t  lda)) dlsym(cublas_handle, "cublasSger_v2_64");

cublasStatus_t (*lcublasDger_v2) (cublasHandle_t  handle, int  m, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)) dlsym(cublas_handle, "cublasDger_v2");

cublasStatus_t (*lcublasDger_v2_64) (cublasHandle_t  handle, int64_t  m, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  m, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  A, int64_t  lda)) dlsym(cublas_handle, "cublasDger_v2_64");

cublasStatus_t (*lcublasCgeru_v2) (cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)) dlsym(cublas_handle, "cublasCgeru_v2");

cublasStatus_t (*lcublasCgeru_v2_64) (cublasHandle_t  handle, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasCgeru_v2_64");

cublasStatus_t (*lcublasCgerc_v2) (cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)) dlsym(cublas_handle, "cublasCgerc_v2");

cublasStatus_t (*lcublasCgerc_v2_64) (cublasHandle_t  handle, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasCgerc_v2_64");

cublasStatus_t (*lcublasZgeru_v2) (cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)) dlsym(cublas_handle, "cublasZgeru_v2");

cublasStatus_t (*lcublasZgeru_v2_64) (cublasHandle_t  handle, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasZgeru_v2_64");

cublasStatus_t (*lcublasZgerc_v2) (cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)) dlsym(cublas_handle, "cublasZgerc_v2");

cublasStatus_t (*lcublasZgerc_v2_64) (cublasHandle_t  handle, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasZgerc_v2_64");

cublasStatus_t (*lcublasSsyr_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  A, int  lda)) dlsym(cublas_handle, "cublasSsyr_v2");

cublasStatus_t (*lcublasSsyr_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  A, int64_t  lda)) dlsym(cublas_handle, "cublasSsyr_v2_64");

cublasStatus_t (*lcublasDsyr_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  A, int  lda)) dlsym(cublas_handle, "cublasDsyr_v2");

cublasStatus_t (*lcublasDsyr_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  A, int64_t  lda)) dlsym(cublas_handle, "cublasDsyr_v2_64");

cublasStatus_t (*lcublasCsyr_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)) dlsym(cublas_handle, "cublasCsyr_v2");

cublasStatus_t (*lcublasCsyr_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasCsyr_v2_64");

cublasStatus_t (*lcublasZsyr_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)) dlsym(cublas_handle, "cublasZsyr_v2");

cublasStatus_t (*lcublasZsyr_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasZsyr_v2_64");

cublasStatus_t (*lcublasCher_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)) dlsym(cublas_handle, "cublasCher_v2");

cublasStatus_t (*lcublasCher_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasCher_v2_64");

cublasStatus_t (*lcublasZher_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)) dlsym(cublas_handle, "cublasZher_v2");

cublasStatus_t (*lcublasZher_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasZher_v2_64");

cublasStatus_t (*lcublasSspr_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  AP)) dlsym(cublas_handle, "cublasSspr_v2");

cublasStatus_t (*lcublasSspr_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  AP)) dlsym(cublas_handle, "cublasSspr_v2_64");

cublasStatus_t (*lcublasDspr_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  AP)) dlsym(cublas_handle, "cublasDspr_v2");

cublasStatus_t (*lcublasDspr_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  AP)) dlsym(cublas_handle, "cublasDspr_v2_64");

cublasStatus_t (*lcublasChpr_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  AP)) dlsym(cublas_handle, "cublasChpr_v2");

cublasStatus_t (*lcublasChpr_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  AP)) dlsym(cublas_handle, "cublasChpr_v2_64");

cublasStatus_t (*lcublasZhpr_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  AP)) dlsym(cublas_handle, "cublasZhpr_v2");

cublasStatus_t (*lcublasZhpr_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  AP)) dlsym(cublas_handle, "cublasZhpr_v2_64");

cublasStatus_t (*lcublasSsyr2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)) dlsym(cublas_handle, "cublasSsyr2_v2");

cublasStatus_t (*lcublasSsyr2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  A, int64_t  lda)) dlsym(cublas_handle, "cublasSsyr2_v2_64");

cublasStatus_t (*lcublasDsyr2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)) dlsym(cublas_handle, "cublasDsyr2_v2");

cublasStatus_t (*lcublasDsyr2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  A, int64_t  lda)) dlsym(cublas_handle, "cublasDsyr2_v2_64");

cublasStatus_t (*lcublasCsyr2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)) dlsym(cublas_handle, "cublasCsyr2_v2");

cublasStatus_t (*lcublasCsyr2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasCsyr2_v2_64");

cublasStatus_t (*lcublasZsyr2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)) dlsym(cublas_handle, "cublasZsyr2_v2");

cublasStatus_t (*lcublasZsyr2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasZsyr2_v2_64");

cublasStatus_t (*lcublasCher2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)) dlsym(cublas_handle, "cublasCher2_v2");

cublasStatus_t (*lcublasCher2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasCher2_v2_64");

cublasStatus_t (*lcublasZher2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)) dlsym(cublas_handle, "cublasZher2_v2");

cublasStatus_t (*lcublasZher2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)) dlsym(cublas_handle, "cublasZher2_v2_64");

cublasStatus_t (*lcublasSspr2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  AP)) dlsym(cublas_handle, "cublasSspr2_v2");

cublasStatus_t (*lcublasSspr2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  AP)) dlsym(cublas_handle, "cublasSspr2_v2_64");

cublasStatus_t (*lcublasDspr2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  AP)) dlsym(cublas_handle, "cublasDspr2_v2");

cublasStatus_t (*lcublasDspr2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  AP)) dlsym(cublas_handle, "cublasDspr2_v2_64");

cublasStatus_t (*lcublasChpr2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  AP)) dlsym(cublas_handle, "cublasChpr2_v2");

cublasStatus_t (*lcublasChpr2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  AP)) dlsym(cublas_handle, "cublasChpr2_v2_64");

cublasStatus_t (*lcublasZhpr2_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  AP)) dlsym(cublas_handle, "cublasZhpr2_v2");

cublasStatus_t (*lcublasZhpr2_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  AP)) dlsym(cublas_handle, "cublasZhpr2_v2_64");

cublasStatus_t (*lcublasSgemvBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount)) dlsym(cublas_handle, "cublasSgemvBatched");

cublasStatus_t (*lcublasSgemvBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float* const  Aarray[], int64_t  lda, const float* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float* const  Aarray[], int64_t  lda, const float* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount)) dlsym(cublas_handle, "cublasSgemvBatched_64");

cublasStatus_t (*lcublasDgemvBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  xarray[], int  incx, const double*  beta, double* const  yarray[], int  incy, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  xarray[], int  incx, const double*  beta, double* const  yarray[], int  incy, int  batchCount)) dlsym(cublas_handle, "cublasDgemvBatched");

cublasStatus_t (*lcublasDgemvBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double* const  Aarray[], int64_t  lda, const double* const  xarray[], int64_t  incx, const double*  beta, double* const  yarray[], int64_t  incy, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double* const  Aarray[], int64_t  lda, const double* const  xarray[], int64_t  incx, const double*  beta, double* const  yarray[], int64_t  incy, int64_t  batchCount)) dlsym(cublas_handle, "cublasDgemvBatched_64");

cublasStatus_t (*lcublasCgemvBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  xarray[], int  incx, const cuComplex*  beta, cuComplex* const  yarray[], int  incy, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  xarray[], int  incx, const cuComplex*  beta, cuComplex* const  yarray[], int  incy, int  batchCount)) dlsym(cublas_handle, "cublasCgemvBatched");

cublasStatus_t (*lcublasCgemvBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  xarray[], int64_t  incx, const cuComplex*  beta, cuComplex* const  yarray[], int64_t  incy, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  xarray[], int64_t  incx, const cuComplex*  beta, cuComplex* const  yarray[], int64_t  incy, int64_t  batchCount)) dlsym(cublas_handle, "cublasCgemvBatched_64");

cublasStatus_t (*lcublasZgemvBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  xarray[], int  incx, const cuDoubleComplex*  beta, cuDoubleComplex* const  yarray[], int  incy, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  xarray[], int  incx, const cuDoubleComplex*  beta, cuDoubleComplex* const  yarray[], int  incy, int  batchCount)) dlsym(cublas_handle, "cublasZgemvBatched");

cublasStatus_t (*lcublasZgemvBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int64_t  lda, const cuDoubleComplex* const  xarray[], int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex* const  yarray[], int64_t  incy, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int64_t  lda, const cuDoubleComplex* const  xarray[], int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex* const  yarray[], int64_t  incy, int64_t  batchCount)) dlsym(cublas_handle, "cublasZgemvBatched_64");

cublasStatus_t (*lcublasHSHgemvBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half* const  Aarray[], int  lda, const __half* const  xarray[], int  incx, const float*  beta, __half* const  yarray[], int  incy, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half* const  Aarray[], int  lda, const __half* const  xarray[], int  incx, const float*  beta, __half* const  yarray[], int  incy, int  batchCount)) dlsym(cublas_handle, "cublasHSHgemvBatched");

cublasStatus_t (*lcublasHSHgemvBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  xarray[], int64_t  incx, const float*  beta, __half* const  yarray[], int64_t  incy, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  xarray[], int64_t  incx, const float*  beta, __half* const  yarray[], int64_t  incy, int64_t  batchCount)) dlsym(cublas_handle, "cublasHSHgemvBatched_64");

cublasStatus_t (*lcublasHSSgemvBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half* const  Aarray[], int  lda, const __half* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half* const  Aarray[], int  lda, const __half* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount)) dlsym(cublas_handle, "cublasHSSgemvBatched");

cublasStatus_t (*lcublasHSSgemvBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount)) dlsym(cublas_handle, "cublasHSSgemvBatched_64");

cublasStatus_t (*lcublasTSTgemvBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int  lda, const __nv_bfloat16* const  xarray[], int  incx, const float*  beta, __nv_bfloat16* const  yarray[], int  incy, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int  lda, const __nv_bfloat16* const  xarray[], int  incx, const float*  beta, __nv_bfloat16* const  yarray[], int  incy, int  batchCount)) dlsym(cublas_handle, "cublasTSTgemvBatched");

cublasStatus_t (*lcublasTSTgemvBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int64_t  lda, const __nv_bfloat16* const  xarray[], int64_t  incx, const float*  beta, __nv_bfloat16* const  yarray[], int64_t  incy, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int64_t  lda, const __nv_bfloat16* const  xarray[], int64_t  incx, const float*  beta, __nv_bfloat16* const  yarray[], int64_t  incy, int64_t  batchCount)) dlsym(cublas_handle, "cublasTSTgemvBatched_64");

cublasStatus_t (*lcublasTSSgemvBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int  lda, const __nv_bfloat16* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int  lda, const __nv_bfloat16* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount)) dlsym(cublas_handle, "cublasTSSgemvBatched");

cublasStatus_t (*lcublasTSSgemvBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int64_t  lda, const __nv_bfloat16* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int64_t  lda, const __nv_bfloat16* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount)) dlsym(cublas_handle, "cublasTSSgemvBatched_64");

cublasStatus_t (*lcublasSgemvStridedBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount)) dlsym(cublas_handle, "cublasSgemvStridedBatched");

cublasStatus_t (*lcublasSgemvStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, long long int  strideA, const float*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, long long int  strideA, const float*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)) dlsym(cublas_handle, "cublasSgemvStridedBatched_64");

cublasStatus_t (*lcublasDgemvStridedBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  x, int  incx, long long int  stridex, const double*  beta, double*  y, int  incy, long long int  stridey, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  x, int  incx, long long int  stridex, const double*  beta, double*  y, int  incy, long long int  stridey, int  batchCount)) dlsym(cublas_handle, "cublasDgemvStridedBatched");

cublasStatus_t (*lcublasDgemvStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, long long int  strideA, const double*  x, int64_t  incx, long long int  stridex, const double*  beta, double*  y, int64_t  incy, long long int  stridey, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, long long int  strideA, const double*  x, int64_t  incx, long long int  stridex, const double*  beta, double*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)) dlsym(cublas_handle, "cublasDgemvStridedBatched_64");

cublasStatus_t (*lcublasCgemvStridedBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  x, int  incx, long long int  stridex, const cuComplex*  beta, cuComplex*  y, int  incy, long long int  stridey, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  x, int  incx, long long int  stridex, const cuComplex*  beta, cuComplex*  y, int  incy, long long int  stridey, int  batchCount)) dlsym(cublas_handle, "cublasCgemvStridedBatched");

cublasStatus_t (*lcublasCgemvStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  x, int64_t  incx, long long int  stridex, const cuComplex*  beta, cuComplex*  y, int64_t  incy, long long int  stridey, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  x, int64_t  incx, long long int  stridex, const cuComplex*  beta, cuComplex*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)) dlsym(cublas_handle, "cublasCgemvStridedBatched_64");

cublasStatus_t (*lcublasZgemvStridedBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  x, int  incx, long long int  stridex, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy, long long int  stridey, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  x, int  incx, long long int  stridex, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy, long long int  stridey, int  batchCount)) dlsym(cublas_handle, "cublasZgemvStridedBatched");

cublasStatus_t (*lcublasZgemvStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, long long int  strideA, const cuDoubleComplex*  x, int64_t  incx, long long int  stridex, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy, long long int  stridey, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, long long int  strideA, const cuDoubleComplex*  x, int64_t  incx, long long int  stridex, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)) dlsym(cublas_handle, "cublasZgemvStridedBatched_64");

cublasStatus_t (*lcublasHSHgemvStridedBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  x, int  incx, long long int  stridex, const float*  beta, __half*  y, int  incy, long long int  stridey, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  x, int  incx, long long int  stridex, const float*  beta, __half*  y, int  incy, long long int  stridey, int  batchCount)) dlsym(cublas_handle, "cublasHSHgemvStridedBatched");

cublasStatus_t (*lcublasHSHgemvStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  x, int64_t  incx, long long int  stridex, const float*  beta, __half*  y, int64_t  incy, long long int  stridey, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  x, int64_t  incx, long long int  stridex, const float*  beta, __half*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)) dlsym(cublas_handle, "cublasHSHgemvStridedBatched_64");

cublasStatus_t (*lcublasHSSgemvStridedBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount)) dlsym(cublas_handle, "cublasHSSgemvStridedBatched");

cublasStatus_t (*lcublasHSSgemvStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)) dlsym(cublas_handle, "cublasHSSgemvStridedBatched_64");

cublasStatus_t (*lcublasTSTgemvStridedBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16*  A, int  lda, long long int  strideA, const __nv_bfloat16*  x, int  incx, long long int  stridex, const float*  beta, __nv_bfloat16*  y, int  incy, long long int  stridey, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16*  A, int  lda, long long int  strideA, const __nv_bfloat16*  x, int  incx, long long int  stridex, const float*  beta, __nv_bfloat16*  y, int  incy, long long int  stridey, int  batchCount)) dlsym(cublas_handle, "cublasTSTgemvStridedBatched");

cublasStatus_t (*lcublasTSTgemvStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16*  A, int64_t  lda, long long int  strideA, const __nv_bfloat16*  x, int64_t  incx, long long int  stridex, const float*  beta, __nv_bfloat16*  y, int64_t  incy, long long int  stridey, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16*  A, int64_t  lda, long long int  strideA, const __nv_bfloat16*  x, int64_t  incx, long long int  stridex, const float*  beta, __nv_bfloat16*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)) dlsym(cublas_handle, "cublasTSTgemvStridedBatched_64");

cublasStatus_t (*lcublasTSSgemvStridedBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16*  A, int  lda, long long int  strideA, const __nv_bfloat16*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16*  A, int  lda, long long int  strideA, const __nv_bfloat16*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount)) dlsym(cublas_handle, "cublasTSSgemvStridedBatched");

cublasStatus_t (*lcublasTSSgemvStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16*  A, int64_t  lda, long long int  strideA, const __nv_bfloat16*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16*  A, int64_t  lda, long long int  strideA, const __nv_bfloat16*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)) dlsym(cublas_handle, "cublasTSSgemvStridedBatched_64");

cublasStatus_t (*lcublasSgemm_v2) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)) dlsym(cublas_handle, "cublasSgemm_v2");

cublasStatus_t (*lcublasSgemm_v2_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasSgemm_v2_64");

cublasStatus_t (*lcublasDgemm_v2) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)) dlsym(cublas_handle, "cublasDgemm_v2");

cublasStatus_t (*lcublasDgemm_v2_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasDgemm_v2_64");

cublasStatus_t (*lcublasCgemm_v2) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCgemm_v2");

cublasStatus_t (*lcublasCgemm_v2_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCgemm_v2_64");

cublasStatus_t (*lcublasCgemm3m) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCgemm3m");

cublasStatus_t (*lcublasCgemm3m_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCgemm3m_64");

cublasStatus_t (*lcublasCgemm3mEx) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)) dlsym(cublas_handle, "cublasCgemm3mEx");

cublasStatus_t (*lcublasCgemm3mEx_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)) dlsym(cublas_handle, "cublasCgemm3mEx_64");

cublasStatus_t (*lcublasZgemm_v2) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZgemm_v2");

cublasStatus_t (*lcublasZgemm_v2_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZgemm_v2_64");

cublasStatus_t (*lcublasZgemm3m) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZgemm3m");

cublasStatus_t (*lcublasZgemm3m_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZgemm3m_64");

cublasStatus_t (*lcublasHgemm) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, const __half*  B, int  ldb, const __half*  beta, __half*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, const __half*  B, int  ldb, const __half*  beta, __half*  C, int  ldc)) dlsym(cublas_handle, "cublasHgemm");

cublasStatus_t (*lcublasHgemm_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half*  A, int64_t  lda, const __half*  B, int64_t  ldb, const __half*  beta, __half*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half*  A, int64_t  lda, const __half*  B, int64_t  ldb, const __half*  beta, __half*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasHgemm_64");

cublasStatus_t (*lcublasSgemmEx) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)) dlsym(cublas_handle, "cublasSgemmEx");

cublasStatus_t (*lcublasSgemmEx_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)) dlsym(cublas_handle, "cublasSgemmEx_64");

cublasStatus_t (*lcublasGemmEx) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)) dlsym(cublas_handle, "cublasGemmEx");

cublasStatus_t (*lcublasGemmEx_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)) dlsym(cublas_handle, "cublasGemmEx_64");

cublasStatus_t (*lcublasCgemmEx) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)) dlsym(cublas_handle, "cublasCgemmEx");

cublasStatus_t (*lcublasCgemmEx_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)) dlsym(cublas_handle, "cublasCgemmEx_64");

cublasStatus_t (*lcublasSsyrk_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  beta, float*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  beta, float*  C, int  ldc)) dlsym(cublas_handle, "cublasSsyrk_v2");

cublasStatus_t (*lcublasSsyrk_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  beta, float*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  beta, float*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasSsyrk_v2_64");

cublasStatus_t (*lcublasDsyrk_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  beta, double*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  beta, double*  C, int  ldc)) dlsym(cublas_handle, "cublasDsyrk_v2");

cublasStatus_t (*lcublasDsyrk_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  beta, double*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  beta, double*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasDsyrk_v2_64");

cublasStatus_t (*lcublasCsyrk_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCsyrk_v2");

cublasStatus_t (*lcublasCsyrk_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCsyrk_v2_64");

cublasStatus_t (*lcublasZsyrk_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZsyrk_v2");

cublasStatus_t (*lcublasZsyrk_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZsyrk_v2_64");

cublasStatus_t (*lcublasCsyrkEx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)) dlsym(cublas_handle, "cublasCsyrkEx");

cublasStatus_t (*lcublasCsyrkEx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)) dlsym(cublas_handle, "cublasCsyrkEx_64");

cublasStatus_t (*lcublasCsyrk3mEx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)) dlsym(cublas_handle, "cublasCsyrk3mEx");

cublasStatus_t (*lcublasCsyrk3mEx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)) dlsym(cublas_handle, "cublasCsyrk3mEx_64");

cublasStatus_t (*lcublasCherk_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const cuComplex*  A, int  lda, const float*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const cuComplex*  A, int  lda, const float*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCherk_v2");

cublasStatus_t (*lcublasCherk_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const cuComplex*  A, int64_t  lda, const float*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const cuComplex*  A, int64_t  lda, const float*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCherk_v2_64");

cublasStatus_t (*lcublasZherk_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const cuDoubleComplex*  A, int  lda, const double*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const cuDoubleComplex*  A, int  lda, const double*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZherk_v2");

cublasStatus_t (*lcublasZherk_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const cuDoubleComplex*  A, int64_t  lda, const double*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const cuDoubleComplex*  A, int64_t  lda, const double*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZherk_v2_64");

cublasStatus_t (*lcublasCherkEx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)) dlsym(cublas_handle, "cublasCherkEx");

cublasStatus_t (*lcublasCherkEx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)) dlsym(cublas_handle, "cublasCherkEx_64");

cublasStatus_t (*lcublasCherk3mEx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)) dlsym(cublas_handle, "cublasCherk3mEx");

cublasStatus_t (*lcublasCherk3mEx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)) dlsym(cublas_handle, "cublasCherk3mEx_64");

cublasStatus_t (*lcublasSsyr2k_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)) dlsym(cublas_handle, "cublasSsyr2k_v2");

cublasStatus_t (*lcublasSsyr2k_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasSsyr2k_v2_64");

cublasStatus_t (*lcublasDsyr2k_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)) dlsym(cublas_handle, "cublasDsyr2k_v2");

cublasStatus_t (*lcublasDsyr2k_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasDsyr2k_v2_64");

cublasStatus_t (*lcublasCsyr2k_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCsyr2k_v2");

cublasStatus_t (*lcublasCsyr2k_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCsyr2k_v2_64");

cublasStatus_t (*lcublasZsyr2k_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZsyr2k_v2");

cublasStatus_t (*lcublasZsyr2k_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZsyr2k_v2_64");

cublasStatus_t (*lcublasCher2k_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCher2k_v2");

cublasStatus_t (*lcublasCher2k_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const float*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const float*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCher2k_v2_64");

cublasStatus_t (*lcublasZher2k_v2) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZher2k_v2");

cublasStatus_t (*lcublasZher2k_v2_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const double*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const double*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZher2k_v2_64");

cublasStatus_t (*lcublasSsyrkx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)) dlsym(cublas_handle, "cublasSsyrkx");

cublasStatus_t (*lcublasSsyrkx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasSsyrkx_64");

cublasStatus_t (*lcublasDsyrkx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)) dlsym(cublas_handle, "cublasDsyrkx");

cublasStatus_t (*lcublasDsyrkx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasDsyrkx_64");

cublasStatus_t (*lcublasCsyrkx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCsyrkx");

cublasStatus_t (*lcublasCsyrkx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCsyrkx_64");

cublasStatus_t (*lcublasZsyrkx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZsyrkx");

cublasStatus_t (*lcublasZsyrkx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZsyrkx_64");

cublasStatus_t (*lcublasCherkx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCherkx");

cublasStatus_t (*lcublasCherkx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const float*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const float*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCherkx_64");

cublasStatus_t (*lcublasZherkx) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZherkx");

cublasStatus_t (*lcublasZherkx_64) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const double*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const double*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZherkx_64");

cublasStatus_t (*lcublasSsymm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)) dlsym(cublas_handle, "cublasSsymm_v2");

cublasStatus_t (*lcublasSsymm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasSsymm_v2_64");

cublasStatus_t (*lcublasDsymm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)) dlsym(cublas_handle, "cublasDsymm_v2");

cublasStatus_t (*lcublasDsymm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasDsymm_v2_64");

cublasStatus_t (*lcublasCsymm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCsymm_v2");

cublasStatus_t (*lcublasCsymm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCsymm_v2_64");

cublasStatus_t (*lcublasZsymm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZsymm_v2");

cublasStatus_t (*lcublasZsymm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZsymm_v2_64");

cublasStatus_t (*lcublasChemm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasChemm_v2");

cublasStatus_t (*lcublasChemm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasChemm_v2_64");

cublasStatus_t (*lcublasZhemm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZhemm_v2");

cublasStatus_t (*lcublasZhemm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZhemm_v2_64");

cublasStatus_t (*lcublasStrsm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, float*  B, int  ldb) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, float*  B, int  ldb)) dlsym(cublas_handle, "cublasStrsm_v2");

cublasStatus_t (*lcublasStrsm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, float*  B, int64_t  ldb) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, float*  B, int64_t  ldb)) dlsym(cublas_handle, "cublasStrsm_v2_64");

cublasStatus_t (*lcublasDtrsm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, double*  B, int  ldb) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, double*  B, int  ldb)) dlsym(cublas_handle, "cublasDtrsm_v2");

cublasStatus_t (*lcublasDtrsm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, double*  B, int64_t  ldb) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, double*  B, int64_t  ldb)) dlsym(cublas_handle, "cublasDtrsm_v2_64");

cublasStatus_t (*lcublasCtrsm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, cuComplex*  B, int  ldb) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, cuComplex*  B, int  ldb)) dlsym(cublas_handle, "cublasCtrsm_v2");

cublasStatus_t (*lcublasCtrsm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, cuComplex*  B, int64_t  ldb) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, cuComplex*  B, int64_t  ldb)) dlsym(cublas_handle, "cublasCtrsm_v2_64");

cublasStatus_t (*lcublasZtrsm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  B, int  ldb) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  B, int  ldb)) dlsym(cublas_handle, "cublasZtrsm_v2");

cublasStatus_t (*lcublasZtrsm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  B, int64_t  ldb) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  B, int64_t  ldb)) dlsym(cublas_handle, "cublasZtrsm_v2_64");

cublasStatus_t (*lcublasStrmm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, float*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, float*  C, int  ldc)) dlsym(cublas_handle, "cublasStrmm_v2");

cublasStatus_t (*lcublasStrmm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, float*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, float*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasStrmm_v2_64");

cublasStatus_t (*lcublasDtrmm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, double*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, double*  C, int  ldc)) dlsym(cublas_handle, "cublasDtrmm_v2");

cublasStatus_t (*lcublasDtrmm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, double*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, double*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasDtrmm_v2_64");

cublasStatus_t (*lcublasCtrmm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCtrmm_v2");

cublasStatus_t (*lcublasCtrmm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCtrmm_v2_64");

cublasStatus_t (*lcublasZtrmm_v2) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZtrmm_v2");

cublasStatus_t (*lcublasZtrmm_v2_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZtrmm_v2_64");

cublasStatus_t (*lcublasHgemmBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half* const  Aarray[], int  lda, const __half* const  Barray[], int  ldb, const __half*  beta, __half* const  Carray[], int  ldc, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half* const  Aarray[], int  lda, const __half* const  Barray[], int  ldb, const __half*  beta, __half* const  Carray[], int  ldc, int  batchCount)) dlsym(cublas_handle, "cublasHgemmBatched");

cublasStatus_t (*lcublasHgemmBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  Barray[], int64_t  ldb, const __half*  beta, __half* const  Carray[], int64_t  ldc, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  Barray[], int64_t  ldb, const __half*  beta, __half* const  Carray[], int64_t  ldc, int64_t  batchCount)) dlsym(cublas_handle, "cublasHgemmBatched_64");

cublasStatus_t (*lcublasSgemmBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  Barray[], int  ldb, const float*  beta, float* const  Carray[], int  ldc, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  Barray[], int  ldb, const float*  beta, float* const  Carray[], int  ldc, int  batchCount)) dlsym(cublas_handle, "cublasSgemmBatched");

cublasStatus_t (*lcublasSgemmBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float* const  Aarray[], int64_t  lda, const float* const  Barray[], int64_t  ldb, const float*  beta, float* const  Carray[], int64_t  ldc, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float* const  Aarray[], int64_t  lda, const float* const  Barray[], int64_t  ldb, const float*  beta, float* const  Carray[], int64_t  ldc, int64_t  batchCount)) dlsym(cublas_handle, "cublasSgemmBatched_64");

cublasStatus_t (*lcublasDgemmBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  Barray[], int  ldb, const double*  beta, double* const  Carray[], int  ldc, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  Barray[], int  ldb, const double*  beta, double* const  Carray[], int  ldc, int  batchCount)) dlsym(cublas_handle, "cublasDgemmBatched");

cublasStatus_t (*lcublasDgemmBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double* const  Aarray[], int64_t  lda, const double* const  Barray[], int64_t  ldb, const double*  beta, double* const  Carray[], int64_t  ldc, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double* const  Aarray[], int64_t  lda, const double* const  Barray[], int64_t  ldb, const double*  beta, double* const  Carray[], int64_t  ldc, int64_t  batchCount)) dlsym(cublas_handle, "cublasDgemmBatched_64");

cublasStatus_t (*lcublasCgemmBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)) dlsym(cublas_handle, "cublasCgemmBatched");

cublasStatus_t (*lcublasCgemmBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  Barray[], int64_t  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int64_t  ldc, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  Barray[], int64_t  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int64_t  ldc, int64_t  batchCount)) dlsym(cublas_handle, "cublasCgemmBatched_64");

cublasStatus_t (*lcublasCgemm3mBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)) dlsym(cublas_handle, "cublasCgemm3mBatched");

cublasStatus_t (*lcublasCgemm3mBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  Barray[], int64_t  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int64_t  ldc, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  Barray[], int64_t  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int64_t  ldc, int64_t  batchCount)) dlsym(cublas_handle, "cublasCgemm3mBatched_64");

cublasStatus_t (*lcublasZgemmBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  Barray[], int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int  ldc, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  Barray[], int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int  ldc, int  batchCount)) dlsym(cublas_handle, "cublasZgemmBatched");

cublasStatus_t (*lcublasZgemmBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int64_t  lda, const cuDoubleComplex* const  Barray[], int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int64_t  ldc, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int64_t  lda, const cuDoubleComplex* const  Barray[], int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int64_t  ldc, int64_t  batchCount)) dlsym(cublas_handle, "cublasZgemmBatched_64");

cublasStatus_t (*lcublasHgemmStridedBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  B, int  ldb, long long int  strideB, const __half*  beta, __half*  C, int  ldc, long long int  strideC, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  B, int  ldb, long long int  strideB, const __half*  beta, __half*  C, int  ldc, long long int  strideC, int  batchCount)) dlsym(cublas_handle, "cublasHgemmStridedBatched");

cublasStatus_t (*lcublasHgemmStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  B, int64_t  ldb, long long int  strideB, const __half*  beta, __half*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  B, int64_t  ldb, long long int  strideB, const __half*  beta, __half*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)) dlsym(cublas_handle, "cublasHgemmStridedBatched_64");

cublasStatus_t (*lcublasSgemmStridedBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  B, int  ldb, long long int  strideB, const float*  beta, float*  C, int  ldc, long long int  strideC, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  B, int  ldb, long long int  strideB, const float*  beta, float*  C, int  ldc, long long int  strideC, int  batchCount)) dlsym(cublas_handle, "cublasSgemmStridedBatched");

cublasStatus_t (*lcublasSgemmStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, long long int  strideA, const float*  B, int64_t  ldb, long long int  strideB, const float*  beta, float*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, long long int  strideA, const float*  B, int64_t  ldb, long long int  strideB, const float*  beta, float*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)) dlsym(cublas_handle, "cublasSgemmStridedBatched_64");

cublasStatus_t (*lcublasDgemmStridedBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  B, int  ldb, long long int  strideB, const double*  beta, double*  C, int  ldc, long long int  strideC, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  B, int  ldb, long long int  strideB, const double*  beta, double*  C, int  ldc, long long int  strideC, int  batchCount)) dlsym(cublas_handle, "cublasDgemmStridedBatched");

cublasStatus_t (*lcublasDgemmStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, long long int  strideA, const double*  B, int64_t  ldb, long long int  strideB, const double*  beta, double*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, long long int  strideA, const double*  B, int64_t  ldb, long long int  strideB, const double*  beta, double*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)) dlsym(cublas_handle, "cublasDgemmStridedBatched_64");

cublasStatus_t (*lcublasCgemmStridedBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)) dlsym(cublas_handle, "cublasCgemmStridedBatched");

cublasStatus_t (*lcublasCgemmStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  B, int64_t  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  B, int64_t  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)) dlsym(cublas_handle, "cublasCgemmStridedBatched_64");

cublasStatus_t (*lcublasCgemm3mStridedBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)) dlsym(cublas_handle, "cublasCgemm3mStridedBatched");

cublasStatus_t (*lcublasCgemm3mStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  B, int64_t  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  B, int64_t  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)) dlsym(cublas_handle, "cublasCgemm3mStridedBatched_64");

cublasStatus_t (*lcublasZgemmStridedBatched) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  B, int  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc, long long int  strideC, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  B, int  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc, long long int  strideC, int  batchCount)) dlsym(cublas_handle, "cublasZgemmStridedBatched");

cublasStatus_t (*lcublasZgemmStridedBatched_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, long long int  strideA, const cuDoubleComplex*  B, int64_t  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, long long int  strideA, const cuDoubleComplex*  B, int64_t  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)) dlsym(cublas_handle, "cublasZgemmStridedBatched_64");

cublasStatus_t (*lcublasGemmBatchedEx) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int  lda, const void* const  Barray[], cudaDataType  Btype, int  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int  ldc, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int  lda, const void* const  Barray[], cudaDataType  Btype, int  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int  ldc, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)) dlsym(cublas_handle, "cublasGemmBatchedEx");

cublasStatus_t (*lcublasGemmBatchedEx_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int64_t  lda, const void* const  Barray[], cudaDataType  Btype, int64_t  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int64_t  ldc, int64_t  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int64_t  lda, const void* const  Barray[], cudaDataType  Btype, int64_t  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int64_t  ldc, int64_t  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)) dlsym(cublas_handle, "cublasGemmBatchedEx_64");

cublasStatus_t (*lcublasGemmStridedBatchedEx) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, long long int  strideC, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, long long int  strideC, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)) dlsym(cublas_handle, "cublasGemmStridedBatchedEx");

cublasStatus_t (*lcublasGemmStridedBatchedEx_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int64_t  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc, long long int  strideC, int64_t  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int64_t  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc, long long int  strideC, int64_t  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)) dlsym(cublas_handle, "cublasGemmStridedBatchedEx_64");

cublasStatus_t (*lcublasSgeam) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  beta, const float*  B, int  ldb, float*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  beta, const float*  B, int  ldb, float*  C, int  ldc)) dlsym(cublas_handle, "cublasSgeam");

cublasStatus_t (*lcublasSgeam_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  beta, const float*  B, int64_t  ldb, float*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  beta, const float*  B, int64_t  ldb, float*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasSgeam_64");

cublasStatus_t (*lcublasDgeam) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  beta, const double*  B, int  ldb, double*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  beta, const double*  B, int  ldb, double*  C, int  ldc)) dlsym(cublas_handle, "cublasDgeam");

cublasStatus_t (*lcublasDgeam_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  beta, const double*  B, int64_t  ldb, double*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  beta, const double*  B, int64_t  ldb, double*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasDgeam_64");

cublasStatus_t (*lcublasCgeam) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCgeam");

cublasStatus_t (*lcublasCgeam_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  beta, const cuComplex*  B, int64_t  ldb, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  beta, const cuComplex*  B, int64_t  ldb, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCgeam_64");

cublasStatus_t (*lcublasZgeam) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZgeam");

cublasStatus_t (*lcublasZgeam_64) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int64_t  ldb, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int64_t  ldb, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZgeam_64");

cublasStatus_t (*lcublasStrsmBatched) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float* const  A[], int  lda, float* const  B[], int  ldb, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float* const  A[], int  lda, float* const  B[], int  ldb, int  batchCount)) dlsym(cublas_handle, "cublasStrsmBatched");

cublasStatus_t (*lcublasStrsmBatched_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float* const  A[], int64_t  lda, float* const  B[], int64_t  ldb, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float* const  A[], int64_t  lda, float* const  B[], int64_t  ldb, int64_t  batchCount)) dlsym(cublas_handle, "cublasStrsmBatched_64");

cublasStatus_t (*lcublasDtrsmBatched) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double* const  A[], int  lda, double* const  B[], int  ldb, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double* const  A[], int  lda, double* const  B[], int  ldb, int  batchCount)) dlsym(cublas_handle, "cublasDtrsmBatched");

cublasStatus_t (*lcublasDtrsmBatched_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double* const  A[], int64_t  lda, double* const  B[], int64_t  ldb, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double* const  A[], int64_t  lda, double* const  B[], int64_t  ldb, int64_t  batchCount)) dlsym(cublas_handle, "cublasDtrsmBatched_64");

cublasStatus_t (*lcublasCtrsmBatched) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  A[], int  lda, cuComplex* const  B[], int  ldb, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  A[], int  lda, cuComplex* const  B[], int  ldb, int  batchCount)) dlsym(cublas_handle, "cublasCtrsmBatched");

cublasStatus_t (*lcublasCtrsmBatched_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex* const  A[], int64_t  lda, cuComplex* const  B[], int64_t  ldb, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex* const  A[], int64_t  lda, cuComplex* const  B[], int64_t  ldb, int64_t  batchCount)) dlsym(cublas_handle, "cublasCtrsmBatched_64");

cublasStatus_t (*lcublasZtrsmBatched) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  B[], int  ldb, int  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  B[], int  ldb, int  batchCount)) dlsym(cublas_handle, "cublasZtrsmBatched");

cublasStatus_t (*lcublasZtrsmBatched_64) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int64_t  lda, cuDoubleComplex* const  B[], int64_t  ldb, int64_t  batchCount) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int64_t  lda, cuDoubleComplex* const  B[], int64_t  ldb, int64_t  batchCount)) dlsym(cublas_handle, "cublasZtrsmBatched_64");

cublasStatus_t (*lcublasSdgmm) (cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const float*  A, int  lda, const float*  x, int  incx, float*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const float*  A, int  lda, const float*  x, int  incx, float*  C, int  ldc)) dlsym(cublas_handle, "cublasSdgmm");

cublasStatus_t (*lcublasSdgmm_64) (cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const float*  A, int64_t  lda, const float*  x, int64_t  incx, float*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const float*  A, int64_t  lda, const float*  x, int64_t  incx, float*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasSdgmm_64");

cublasStatus_t (*lcublasDdgmm) (cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const double*  A, int  lda, const double*  x, int  incx, double*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const double*  A, int  lda, const double*  x, int  incx, double*  C, int  ldc)) dlsym(cublas_handle, "cublasDdgmm");

cublasStatus_t (*lcublasDdgmm_64) (cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const double*  A, int64_t  lda, const double*  x, int64_t  incx, double*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const double*  A, int64_t  lda, const double*  x, int64_t  incx, double*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasDdgmm_64");

cublasStatus_t (*lcublasCdgmm) (cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, cuComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, cuComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasCdgmm");

cublasStatus_t (*lcublasCdgmm_64) (cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, cuComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, cuComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasCdgmm_64");

cublasStatus_t (*lcublasZdgmm) (cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  C, int  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  C, int  ldc)) dlsym(cublas_handle, "cublasZdgmm");

cublasStatus_t (*lcublasZdgmm_64) (cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  C, int64_t  ldc) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  C, int64_t  ldc)) dlsym(cublas_handle, "cublasZdgmm_64");

cublasStatus_t (*lcublasSmatinvBatched) (cublasHandle_t  handle, int  n, const float* const  A[], int  lda, float* const  Ainv[], int  lda_inv, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float* const  A[], int  lda, float* const  Ainv[], int  lda_inv, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasSmatinvBatched");

cublasStatus_t (*lcublasDmatinvBatched) (cublasHandle_t  handle, int  n, const double* const  A[], int  lda, double* const  Ainv[], int  lda_inv, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double* const  A[], int  lda, double* const  Ainv[], int  lda_inv, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasDmatinvBatched");

cublasStatus_t (*lcublasCmatinvBatched) (cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, cuComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, cuComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasCmatinvBatched");

cublasStatus_t (*lcublasZmatinvBatched) (cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasZmatinvBatched");

cublasStatus_t (*lcublasSgeqrfBatched) (cublasHandle_t  handle, int  m, int  n, float* const  Aarray[], int  lda, float* const  TauArray[], int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, float* const  Aarray[], int  lda, float* const  TauArray[], int*  info, int  batchSize)) dlsym(cublas_handle, "cublasSgeqrfBatched");

cublasStatus_t (*lcublasDgeqrfBatched) (cublasHandle_t  handle, int  m, int  n, double* const  Aarray[], int  lda, double* const  TauArray[], int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, double* const  Aarray[], int  lda, double* const  TauArray[], int*  info, int  batchSize)) dlsym(cublas_handle, "cublasDgeqrfBatched");

cublasStatus_t (*lcublasCgeqrfBatched) (cublasHandle_t  handle, int  m, int  n, cuComplex* const  Aarray[], int  lda, cuComplex* const  TauArray[], int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, cuComplex* const  Aarray[], int  lda, cuComplex* const  TauArray[], int*  info, int  batchSize)) dlsym(cublas_handle, "cublasCgeqrfBatched");

cublasStatus_t (*lcublasZgeqrfBatched) (cublasHandle_t  handle, int  m, int  n, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  TauArray[], int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  m, int  n, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  TauArray[], int*  info, int  batchSize)) dlsym(cublas_handle, "cublasZgeqrfBatched");

cublasStatus_t (*lcublasSgelsBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, float* const  Aarray[], int  lda, float* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, float* const  Aarray[], int  lda, float* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)) dlsym(cublas_handle, "cublasSgelsBatched");

cublasStatus_t (*lcublasDgelsBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, double* const  Aarray[], int  lda, double* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, double* const  Aarray[], int  lda, double* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)) dlsym(cublas_handle, "cublasDgelsBatched");

cublasStatus_t (*lcublasCgelsBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuComplex* const  Aarray[], int  lda, cuComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuComplex* const  Aarray[], int  lda, cuComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)) dlsym(cublas_handle, "cublasCgelsBatched");

cublasStatus_t (*lcublasZgelsBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)) dlsym(cublas_handle, "cublasZgelsBatched");

cublasStatus_t (*lcublasStpttr) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  AP, float*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  AP, float*  A, int  lda)) dlsym(cublas_handle, "cublasStpttr");

cublasStatus_t (*lcublasDtpttr) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  AP, double*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  AP, double*  A, int  lda)) dlsym(cublas_handle, "cublasDtpttr");

cublasStatus_t (*lcublasCtpttr) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  AP, cuComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  AP, cuComplex*  A, int  lda)) dlsym(cublas_handle, "cublasCtpttr");

cublasStatus_t (*lcublasZtpttr) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  A, int  lda) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  A, int  lda)) dlsym(cublas_handle, "cublasZtpttr");

cublasStatus_t (*lcublasStrttp) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  A, int  lda, float*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  A, int  lda, float*  AP)) dlsym(cublas_handle, "cublasStrttp");

cublasStatus_t (*lcublasDtrttp) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  A, int  lda, double*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  A, int  lda, double*  AP)) dlsym(cublas_handle, "cublasDtrttp");

cublasStatus_t (*lcublasCtrttp) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  A, int  lda, cuComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  A, int  lda, cuComplex*  AP)) dlsym(cublas_handle, "cublasCtrttp");

cublasStatus_t (*lcublasZtrttp) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  AP) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  AP)) dlsym(cublas_handle, "cublasZtrttp");

cublasStatus_t (*lcublasSgetrfBatched) (cublasHandle_t  handle, int  n, float* const  A[], int  lda, int*  P, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, float* const  A[], int  lda, int*  P, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasSgetrfBatched");

cublasStatus_t (*lcublasDgetrfBatched) (cublasHandle_t  handle, int  n, double* const  A[], int  lda, int*  P, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, double* const  A[], int  lda, int*  P, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasDgetrfBatched");

cublasStatus_t (*lcublasCgetrfBatched) (cublasHandle_t  handle, int  n, cuComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, cuComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasCgetrfBatched");

cublasStatus_t (*lcublasZgetrfBatched) (cublasHandle_t  handle, int  n, cuDoubleComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, cuDoubleComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasZgetrfBatched");

cublasStatus_t (*lcublasSgetriBatched) (cublasHandle_t  handle, int  n, const float* const  A[], int  lda, const int*  P, float* const  C[], int  ldc, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const float* const  A[], int  lda, const int*  P, float* const  C[], int  ldc, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasSgetriBatched");

cublasStatus_t (*lcublasDgetriBatched) (cublasHandle_t  handle, int  n, const double* const  A[], int  lda, const int*  P, double* const  C[], int  ldc, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const double* const  A[], int  lda, const int*  P, double* const  C[], int  ldc, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasDgetriBatched");

cublasStatus_t (*lcublasCgetriBatched) (cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, const int*  P, cuComplex* const  C[], int  ldc, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, const int*  P, cuComplex* const  C[], int  ldc, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasCgetriBatched");

cublasStatus_t (*lcublasZgetriBatched) (cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, const int*  P, cuDoubleComplex* const  C[], int  ldc, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, const int*  P, cuDoubleComplex* const  C[], int  ldc, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasZgetriBatched");

cublasStatus_t (*lcublasSgetrsBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const float* const  Aarray[], int  lda, const int*  devIpiv, float* const  Barray[], int  ldb, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const float* const  Aarray[], int  lda, const int*  devIpiv, float* const  Barray[], int  ldb, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasSgetrsBatched");

cublasStatus_t (*lcublasDgetrsBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const double* const  Aarray[], int  lda, const int*  devIpiv, double* const  Barray[], int  ldb, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const double* const  Aarray[], int  lda, const int*  devIpiv, double* const  Barray[], int  ldb, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasDgetrsBatched");

cublasStatus_t (*lcublasCgetrsBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuComplex* const  Aarray[], int  lda, const int*  devIpiv, cuComplex* const  Barray[], int  ldb, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuComplex* const  Aarray[], int  lda, const int*  devIpiv, cuComplex* const  Barray[], int  ldb, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasCgetrsBatched");

cublasStatus_t (*lcublasZgetrsBatched) (cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuDoubleComplex* const  Aarray[], int  lda, const int*  devIpiv, cuDoubleComplex* const  Barray[], int  ldb, int*  info, int  batchSize) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuDoubleComplex* const  Aarray[], int  lda, const int*  devIpiv, cuDoubleComplex* const  Barray[], int  ldb, int*  info, int  batchSize)) dlsym(cublas_handle, "cublasZgetrsBatched");

cublasStatus_t (*lcublasUint8gemmBias) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, cublasOperation_t  transc, int  m, int  n, int  k, const unsigned char*  A, int  A_bias, int  lda, const unsigned char*  B, int  B_bias, int  ldb, unsigned char*  C, int  C_bias, int  ldc, int  C_mult, int  C_shift) =
	(cublasStatus_t (*) (cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, cublasOperation_t  transc, int  m, int  n, int  k, const unsigned char*  A, int  A_bias, int  lda, const unsigned char*  B, int  B_bias, int  ldb, unsigned char*  C, int  C_bias, int  ldc, int  C_mult, int  C_shift)) dlsym(cublas_handle, "cublasUint8gemmBias");

cudaError_t (*lcudaProfilerStart) () =
	(cudaError_t (*) ()) dlsym(cudart_handle, "cudaProfilerStart");

cudaError_t (*lcudaProfilerStop) () =
	(cudaError_t (*) ()) dlsym(cudart_handle, "cudaProfilerStop");

CUresult (*lcuProfilerInitialize) (const char * configFile, const char * outputFile, CUoutput_mode  outputMode) =
	(CUresult (*) (const char * configFile, const char * outputFile, CUoutput_mode  outputMode)) dlsym(cuda_handle, "cuProfilerInitialize");

CUresult (*lcuProfilerStart) () =
	(CUresult (*) ()) dlsym(cuda_handle, "cuProfilerStart");

CUresult (*lcuProfilerStop) () =
	(CUresult (*) ()) dlsym(cuda_handle, "cuProfilerStop");

const char * (*lnvrtcGetErrorString) (nvrtcResult  result) =
	(const char * (*) (nvrtcResult  result)) dlsym(RTLD_NEXT, "nvrtcGetErrorString");

nvrtcResult (*lnvrtcVersion) (int * major, int * minor) =
	(nvrtcResult (*) (int * major, int * minor)) dlsym(RTLD_NEXT, "nvrtcVersion");

nvrtcResult (*lnvrtcGetNumSupportedArchs) (int*  numArchs) =
	(nvrtcResult (*) (int*  numArchs)) dlsym(RTLD_NEXT, "nvrtcGetNumSupportedArchs");

nvrtcResult (*lnvrtcGetSupportedArchs) (int*  supportedArchs) =
	(nvrtcResult (*) (int*  supportedArchs)) dlsym(RTLD_NEXT, "nvrtcGetSupportedArchs");

nvrtcResult (*lnvrtcCreateProgram) (nvrtcProgram * prog, const char * src, const char * name, int  numHeaders, const char * const * headers, const char * const * includeNames) =
	(nvrtcResult (*) (nvrtcProgram * prog, const char * src, const char * name, int  numHeaders, const char * const * headers, const char * const * includeNames)) dlsym(RTLD_NEXT, "nvrtcCreateProgram");

nvrtcResult (*lnvrtcDestroyProgram) (nvrtcProgram * prog) =
	(nvrtcResult (*) (nvrtcProgram * prog)) dlsym(RTLD_NEXT, "nvrtcDestroyProgram");

nvrtcResult (*lnvrtcCompileProgram) (nvrtcProgram  prog, int  numOptions, const char * const * options) =
	(nvrtcResult (*) (nvrtcProgram  prog, int  numOptions, const char * const * options)) dlsym(RTLD_NEXT, "nvrtcCompileProgram");

nvrtcResult (*lnvrtcGetPTXSize) (nvrtcProgram  prog, size_t * ptxSizeRet) =
	(nvrtcResult (*) (nvrtcProgram  prog, size_t * ptxSizeRet)) dlsym(RTLD_NEXT, "nvrtcGetPTXSize");

nvrtcResult (*lnvrtcGetPTX) (nvrtcProgram  prog, char * ptx) =
	(nvrtcResult (*) (nvrtcProgram  prog, char * ptx)) dlsym(RTLD_NEXT, "nvrtcGetPTX");

nvrtcResult (*lnvrtcGetCUBINSize) (nvrtcProgram  prog, size_t * cubinSizeRet) =
	(nvrtcResult (*) (nvrtcProgram  prog, size_t * cubinSizeRet)) dlsym(RTLD_NEXT, "nvrtcGetCUBINSize");

nvrtcResult (*lnvrtcGetCUBIN) (nvrtcProgram  prog, char * cubin) =
	(nvrtcResult (*) (nvrtcProgram  prog, char * cubin)) dlsym(RTLD_NEXT, "nvrtcGetCUBIN");

nvrtcResult (*lnvrtcGetLTOIRSize) (nvrtcProgram  prog, size_t * LTOIRSizeRet) =
	(nvrtcResult (*) (nvrtcProgram  prog, size_t * LTOIRSizeRet)) dlsym(RTLD_NEXT, "nvrtcGetLTOIRSize");

nvrtcResult (*lnvrtcGetLTOIR) (nvrtcProgram  prog, char * LTOIR) =
	(nvrtcResult (*) (nvrtcProgram  prog, char * LTOIR)) dlsym(RTLD_NEXT, "nvrtcGetLTOIR");

nvrtcResult (*lnvrtcGetOptiXIRSize) (nvrtcProgram  prog, size_t * optixirSizeRet) =
	(nvrtcResult (*) (nvrtcProgram  prog, size_t * optixirSizeRet)) dlsym(RTLD_NEXT, "nvrtcGetOptiXIRSize");

nvrtcResult (*lnvrtcGetOptiXIR) (nvrtcProgram  prog, char * optixir) =
	(nvrtcResult (*) (nvrtcProgram  prog, char * optixir)) dlsym(RTLD_NEXT, "nvrtcGetOptiXIR");

nvrtcResult (*lnvrtcGetProgramLogSize) (nvrtcProgram  prog, size_t * logSizeRet) =
	(nvrtcResult (*) (nvrtcProgram  prog, size_t * logSizeRet)) dlsym(RTLD_NEXT, "nvrtcGetProgramLogSize");

nvrtcResult (*lnvrtcGetProgramLog) (nvrtcProgram  prog, char * log) =
	(nvrtcResult (*) (nvrtcProgram  prog, char * log)) dlsym(RTLD_NEXT, "nvrtcGetProgramLog");

nvrtcResult (*lnvrtcAddNameExpression) (nvrtcProgram  prog, const char * const  name_expression) =
	(nvrtcResult (*) (nvrtcProgram  prog, const char * const  name_expression)) dlsym(RTLD_NEXT, "nvrtcAddNameExpression");

nvrtcResult (*lnvrtcGetLoweredName) (nvrtcProgram  prog, const char *const  name_expression, const char**  lowered_name) =
	(nvrtcResult (*) (nvrtcProgram  prog, const char *const  name_expression, const char**  lowered_name)) dlsym(RTLD_NEXT, "nvrtcGetLoweredName");

cublasStatus_t (*lcublasLtCreate) (cublasLtHandle_t*  lightHandle) =
	(cublasStatus_t (*) (cublasLtHandle_t*  lightHandle)) dlsym(cublasLt_handle, "cublasLtCreate");

cublasStatus_t (*lcublasLtDestroy) (cublasLtHandle_t  lightHandle) =
	(cublasStatus_t (*) (cublasLtHandle_t  lightHandle)) dlsym(cublasLt_handle, "cublasLtDestroy");

const char* (*lcublasLtGetStatusName) (cublasStatus_t  status) =
	(const char* (*) (cublasStatus_t  status)) dlsym(cublasLt_handle, "cublasLtGetStatusName");

const char* (*lcublasLtGetStatusString) (cublasStatus_t  status) =
	(const char* (*) (cublasStatus_t  status)) dlsym(cublasLt_handle, "cublasLtGetStatusString");

size_t (*lcublasLtGetVersion) () =
	(size_t (*) ()) dlsym(cublasLt_handle, "cublasLtGetVersion");

size_t (*lcublasLtGetCudartVersion) () =
	(size_t (*) ()) dlsym(cublasLt_handle, "cublasLtGetCudartVersion");

cublasStatus_t (*lcublasLtGetProperty) (libraryPropertyType  type, int*  value) =
	(cublasStatus_t (*) (libraryPropertyType  type, int*  value)) dlsym(cublasLt_handle, "cublasLtGetProperty");

cublasStatus_t (*lcublasLtHeuristicsCacheGetCapacity) (size_t*  capacity) =
	(cublasStatus_t (*) (size_t*  capacity)) dlsym(cublasLt_handle, "cublasLtHeuristicsCacheGetCapacity");

cublasStatus_t (*lcublasLtHeuristicsCacheSetCapacity) (size_t  capacity) =
	(cublasStatus_t (*) (size_t  capacity)) dlsym(cublasLt_handle, "cublasLtHeuristicsCacheSetCapacity");

unsigned (*lcublasLtDisableCpuInstructionsSetMask) (unsigned  mask) =
	(unsigned (*) (unsigned  mask)) dlsym(cublasLt_handle, "cublasLtDisableCpuInstructionsSetMask");

cublasStatus_t (*lcublasLtMatmul) (cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  computeDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  B, cublasLtMatrixLayout_t  Bdesc, const void*  beta, const void*  C, cublasLtMatrixLayout_t  Cdesc, void*  D, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, void*  workspace, size_t  workspaceSizeInBytes, cudaStream_t  stream) =
	(cublasStatus_t (*) (cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  computeDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  B, cublasLtMatrixLayout_t  Bdesc, const void*  beta, const void*  C, cublasLtMatrixLayout_t  Cdesc, void*  D, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, void*  workspace, size_t  workspaceSizeInBytes, cudaStream_t  stream)) dlsym(cublasLt_handle, "cublasLtMatmul");

cublasStatus_t (*lcublasLtMatrixTransform) (cublasLtHandle_t  lightHandle, cublasLtMatrixTransformDesc_t  transformDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  beta, const void*  B, cublasLtMatrixLayout_t  Bdesc, void*  C, cublasLtMatrixLayout_t  Cdesc, cudaStream_t  stream) =
	(cublasStatus_t (*) (cublasLtHandle_t  lightHandle, cublasLtMatrixTransformDesc_t  transformDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  beta, const void*  B, cublasLtMatrixLayout_t  Bdesc, void*  C, cublasLtMatrixLayout_t  Cdesc, cudaStream_t  stream)) dlsym(cublasLt_handle, "cublasLtMatrixTransform");

cublasStatus_t (*lcublasLtMatrixLayoutInit_internal) (cublasLtMatrixLayout_t  matLayout, size_t  size, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld) =
	(cublasStatus_t (*) (cublasLtMatrixLayout_t  matLayout, size_t  size, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld)) dlsym(cublasLt_handle, "cublasLtMatrixLayoutInit_internal");

cublasStatus_t (*lcublasLtMatrixLayoutCreate) (cublasLtMatrixLayout_t*  matLayout, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld) =
	(cublasStatus_t (*) (cublasLtMatrixLayout_t*  matLayout, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld)) dlsym(cublasLt_handle, "cublasLtMatrixLayoutCreate");

cublasStatus_t (*lcublasLtMatrixLayoutDestroy) (cublasLtMatrixLayout_t  matLayout) =
	(cublasStatus_t (*) (cublasLtMatrixLayout_t  matLayout)) dlsym(cublasLt_handle, "cublasLtMatrixLayoutDestroy");

cublasStatus_t (*lcublasLtMatrixLayoutSetAttribute) (cublasLtMatrixLayout_t  matLayout, cublasLtMatrixLayoutAttribute_t  attr, const void*  buf, size_t  sizeInBytes) =
	(cublasStatus_t (*) (cublasLtMatrixLayout_t  matLayout, cublasLtMatrixLayoutAttribute_t  attr, const void*  buf, size_t  sizeInBytes)) dlsym(cublasLt_handle, "cublasLtMatrixLayoutSetAttribute");

cublasStatus_t (*lcublasLtMatrixLayoutGetAttribute) (cublasLtMatrixLayout_t  matLayout, cublasLtMatrixLayoutAttribute_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten) =
	(cublasStatus_t (*) (cublasLtMatrixLayout_t  matLayout, cublasLtMatrixLayoutAttribute_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)) dlsym(cublasLt_handle, "cublasLtMatrixLayoutGetAttribute");

cublasStatus_t (*lcublasLtMatmulDescInit_internal) (cublasLtMatmulDesc_t  matmulDesc, size_t  size, cublasComputeType_t  computeType, cudaDataType_t  scaleType) =
	(cublasStatus_t (*) (cublasLtMatmulDesc_t  matmulDesc, size_t  size, cublasComputeType_t  computeType, cudaDataType_t  scaleType)) dlsym(cublasLt_handle, "cublasLtMatmulDescInit_internal");

cublasStatus_t (*lcublasLtMatmulDescCreate) (cublasLtMatmulDesc_t*  matmulDesc, cublasComputeType_t  computeType, cudaDataType_t  scaleType) =
	(cublasStatus_t (*) (cublasLtMatmulDesc_t*  matmulDesc, cublasComputeType_t  computeType, cudaDataType_t  scaleType)) dlsym(cublasLt_handle, "cublasLtMatmulDescCreate");

cublasStatus_t (*lcublasLtMatmulDescDestroy) (cublasLtMatmulDesc_t  matmulDesc) =
	(cublasStatus_t (*) (cublasLtMatmulDesc_t  matmulDesc)) dlsym(cublasLt_handle, "cublasLtMatmulDescDestroy");

cublasStatus_t (*lcublasLtMatmulDescSetAttribute) (cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes) =
	(cublasStatus_t (*) (cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes)) dlsym(cublasLt_handle, "cublasLtMatmulDescSetAttribute");

cublasStatus_t (*lcublasLtMatmulDescGetAttribute) (cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten) =
	(cublasStatus_t (*) (cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)) dlsym(cublasLt_handle, "cublasLtMatmulDescGetAttribute");

cublasStatus_t (*lcublasLtMatrixTransformDescInit_internal) (cublasLtMatrixTransformDesc_t  transformDesc, size_t  size, cudaDataType  scaleType) =
	(cublasStatus_t (*) (cublasLtMatrixTransformDesc_t  transformDesc, size_t  size, cudaDataType  scaleType)) dlsym(cublasLt_handle, "cublasLtMatrixTransformDescInit_internal");

cublasStatus_t (*lcublasLtMatrixTransformDescCreate) (cublasLtMatrixTransformDesc_t*  transformDesc, cudaDataType  scaleType) =
	(cublasStatus_t (*) (cublasLtMatrixTransformDesc_t*  transformDesc, cudaDataType  scaleType)) dlsym(cublasLt_handle, "cublasLtMatrixTransformDescCreate");

cublasStatus_t (*lcublasLtMatrixTransformDescDestroy) (cublasLtMatrixTransformDesc_t  transformDesc) =
	(cublasStatus_t (*) (cublasLtMatrixTransformDesc_t  transformDesc)) dlsym(cublasLt_handle, "cublasLtMatrixTransformDescDestroy");

cublasStatus_t (*lcublasLtMatrixTransformDescSetAttribute) (cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes) =
	(cublasStatus_t (*) (cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes)) dlsym(cublasLt_handle, "cublasLtMatrixTransformDescSetAttribute");

cublasStatus_t (*lcublasLtMatrixTransformDescGetAttribute) (cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten) =
	(cublasStatus_t (*) (cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)) dlsym(cublasLt_handle, "cublasLtMatrixTransformDescGetAttribute");

cublasStatus_t (*lcublasLtMatmulPreferenceInit_internal) (cublasLtMatmulPreference_t  pref, size_t  size) =
	(cublasStatus_t (*) (cublasLtMatmulPreference_t  pref, size_t  size)) dlsym(cublasLt_handle, "cublasLtMatmulPreferenceInit_internal");

cublasStatus_t (*lcublasLtMatmulPreferenceCreate) (cublasLtMatmulPreference_t*  pref) =
	(cublasStatus_t (*) (cublasLtMatmulPreference_t*  pref)) dlsym(cublasLt_handle, "cublasLtMatmulPreferenceCreate");

cublasStatus_t (*lcublasLtMatmulPreferenceDestroy) (cublasLtMatmulPreference_t  pref) =
	(cublasStatus_t (*) (cublasLtMatmulPreference_t  pref)) dlsym(cublasLt_handle, "cublasLtMatmulPreferenceDestroy");

cublasStatus_t (*lcublasLtMatmulPreferenceSetAttribute) (cublasLtMatmulPreference_t  pref, cublasLtMatmulPreferenceAttributes_t  attr, const void*  buf, size_t  sizeInBytes) =
	(cublasStatus_t (*) (cublasLtMatmulPreference_t  pref, cublasLtMatmulPreferenceAttributes_t  attr, const void*  buf, size_t  sizeInBytes)) dlsym(cublasLt_handle, "cublasLtMatmulPreferenceSetAttribute");

cublasStatus_t (*lcublasLtMatmulPreferenceGetAttribute) (cublasLtMatmulPreference_t  pref, cublasLtMatmulPreferenceAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten) =
	(cublasStatus_t (*) (cublasLtMatmulPreference_t  pref, cublasLtMatmulPreferenceAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)) dlsym(cublasLt_handle, "cublasLtMatmulPreferenceGetAttribute");

cublasStatus_t (*lcublasLtMatmulAlgoGetHeuristic) (cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, cublasLtMatmulPreference_t  preference, int  requestedAlgoCount, cublasLtMatmulHeuristicResult_t  heuristicResultsArray[], int*  returnAlgoCount) =
	(cublasStatus_t (*) (cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, cublasLtMatmulPreference_t  preference, int  requestedAlgoCount, cublasLtMatmulHeuristicResult_t  heuristicResultsArray[], int*  returnAlgoCount)) dlsym(cublasLt_handle, "cublasLtMatmulAlgoGetHeuristic");

cublasStatus_t (*lcublasLtMatmulAlgoGetIds) (cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  requestedAlgoCount, int  algoIdsArray[], int*  returnAlgoCount) =
	(cublasStatus_t (*) (cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  requestedAlgoCount, int  algoIdsArray[], int*  returnAlgoCount)) dlsym(cublasLt_handle, "cublasLtMatmulAlgoGetIds");

cublasStatus_t (*lcublasLtMatmulAlgoInit) (cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  algoId, cublasLtMatmulAlgo_t*  algo) =
	(cublasStatus_t (*) (cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  algoId, cublasLtMatmulAlgo_t*  algo)) dlsym(cublasLt_handle, "cublasLtMatmulAlgoInit");

cublasStatus_t (*lcublasLtMatmulAlgoCheck) (cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulHeuristicResult_t*  result) =
	(cublasStatus_t (*) (cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulHeuristicResult_t*  result)) dlsym(cublasLt_handle, "cublasLtMatmulAlgoCheck");

cublasStatus_t (*lcublasLtMatmulAlgoCapGetAttribute) (const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoCapAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten) =
	(cublasStatus_t (*) (const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoCapAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)) dlsym(cublasLt_handle, "cublasLtMatmulAlgoCapGetAttribute");

cublasStatus_t (*lcublasLtMatmulAlgoConfigSetAttribute) (cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, const void*  buf, size_t  sizeInBytes) =
	(cublasStatus_t (*) (cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, const void*  buf, size_t  sizeInBytes)) dlsym(cublasLt_handle, "cublasLtMatmulAlgoConfigSetAttribute");

cublasStatus_t (*lcublasLtMatmulAlgoConfigGetAttribute) (const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten) =
	(cublasStatus_t (*) (const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)) dlsym(cublasLt_handle, "cublasLtMatmulAlgoConfigGetAttribute");

cublasStatus_t (*lcublasLtLoggerSetCallback) (cublasLtLoggerCallback_t  callback) =
	(cublasStatus_t (*) (cublasLtLoggerCallback_t  callback)) dlsym(cublasLt_handle, "cublasLtLoggerSetCallback");

cublasStatus_t (*lcublasLtLoggerSetFile) (FILE*  file) =
	(cublasStatus_t (*) (FILE*  file)) dlsym(cublasLt_handle, "cublasLtLoggerSetFile");

cublasStatus_t (*lcublasLtLoggerOpenFile) (const char*  logFile) =
	(cublasStatus_t (*) (const char*  logFile)) dlsym(cublasLt_handle, "cublasLtLoggerOpenFile");

cublasStatus_t (*lcublasLtLoggerSetLevel) (int  level) =
	(cublasStatus_t (*) (int  level)) dlsym(cublasLt_handle, "cublasLtLoggerSetLevel");

cublasStatus_t (*lcublasLtLoggerSetMask) (int  mask) =
	(cublasStatus_t (*) (int  mask)) dlsym(cublasLt_handle, "cublasLtLoggerSetMask");

cublasStatus_t (*lcublasLtLoggerForceDisable) () =
	(cublasStatus_t (*) ()) dlsym(cublasLt_handle, "cublasLtLoggerForceDisable");



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
    
