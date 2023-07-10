
#include <cstring>
                
#include "spdlog/spdlog.h"

#include <tally/transform.h>
#include <tally/util.h>
#include <tally/msg_struct.h>
#include <tally/generated/cuda_api.h>
#include <tally/generated/msg_struct.h>
#include <tally/generated/server.h>
        
void TallyServer::register_api_handler() {
	cuda_api_handler_map[CUDA_API_ENUM::CUGETERRORSTRING] = std::bind(&TallyServer::handle_cuGetErrorString, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGETERRORNAME] = std::bind(&TallyServer::handle_cuGetErrorName, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUINIT] = std::bind(&TallyServer::handle_cuInit, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDRIVERGETVERSION] = std::bind(&TallyServer::handle_cuDriverGetVersion, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGET] = std::bind(&TallyServer::handle_cuDeviceGet, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETCOUNT] = std::bind(&TallyServer::handle_cuDeviceGetCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETNAME] = std::bind(&TallyServer::handle_cuDeviceGetName, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETUUID] = std::bind(&TallyServer::handle_cuDeviceGetUuid, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETUUID_V2] = std::bind(&TallyServer::handle_cuDeviceGetUuid_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETLUID] = std::bind(&TallyServer::handle_cuDeviceGetLuid, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICETOTALMEM_V2] = std::bind(&TallyServer::handle_cuDeviceTotalMem_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETATTRIBUTE] = std::bind(&TallyServer::handle_cuDeviceGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETNVSCISYNCATTRIBUTES] = std::bind(&TallyServer::handle_cuDeviceGetNvSciSyncAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICESETMEMPOOL] = std::bind(&TallyServer::handle_cuDeviceSetMemPool, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETMEMPOOL] = std::bind(&TallyServer::handle_cuDeviceGetMemPool, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETDEFAULTMEMPOOL] = std::bind(&TallyServer::handle_cuDeviceGetDefaultMemPool, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUFLUSHGPUDIRECTRDMAWRITES] = std::bind(&TallyServer::handle_cuFlushGPUDirectRDMAWrites, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETPROPERTIES] = std::bind(&TallyServer::handle_cuDeviceGetProperties, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICECOMPUTECAPABILITY] = std::bind(&TallyServer::handle_cuDeviceComputeCapability, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXRETAIN] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxRetain, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXRELEASE_V2] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxRelease_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXSETFLAGS_V2] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxSetFlags_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXGETSTATE] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxGetState, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEPRIMARYCTXRESET_V2] = std::bind(&TallyServer::handle_cuDevicePrimaryCtxReset_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETEXECAFFINITYSUPPORT] = std::bind(&TallyServer::handle_cuDeviceGetExecAffinitySupport, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXCREATE_V2] = std::bind(&TallyServer::handle_cuCtxCreate_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXCREATE_V3] = std::bind(&TallyServer::handle_cuCtxCreate_v3, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXDESTROY_V2] = std::bind(&TallyServer::handle_cuCtxDestroy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXPUSHCURRENT_V2] = std::bind(&TallyServer::handle_cuCtxPushCurrent_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXPOPCURRENT_V2] = std::bind(&TallyServer::handle_cuCtxPopCurrent_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSETCURRENT] = std::bind(&TallyServer::handle_cuCtxSetCurrent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETCURRENT] = std::bind(&TallyServer::handle_cuCtxGetCurrent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETDEVICE] = std::bind(&TallyServer::handle_cuCtxGetDevice, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETFLAGS] = std::bind(&TallyServer::handle_cuCtxGetFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSYNCHRONIZE] = std::bind(&TallyServer::handle_cuCtxSynchronize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSETLIMIT] = std::bind(&TallyServer::handle_cuCtxSetLimit, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETLIMIT] = std::bind(&TallyServer::handle_cuCtxGetLimit, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETCACHECONFIG] = std::bind(&TallyServer::handle_cuCtxGetCacheConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSETCACHECONFIG] = std::bind(&TallyServer::handle_cuCtxSetCacheConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cuCtxGetSharedMemConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXSETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cuCtxSetSharedMemConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETAPIVERSION] = std::bind(&TallyServer::handle_cuCtxGetApiVersion, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETSTREAMPRIORITYRANGE] = std::bind(&TallyServer::handle_cuCtxGetStreamPriorityRange, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXRESETPERSISTINGL2CACHE] = std::bind(&TallyServer::handle_cuCtxResetPersistingL2Cache, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXGETEXECAFFINITY] = std::bind(&TallyServer::handle_cuCtxGetExecAffinity, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXATTACH] = std::bind(&TallyServer::handle_cuCtxAttach, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXDETACH] = std::bind(&TallyServer::handle_cuCtxDetach, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULELOAD] = std::bind(&TallyServer::handle_cuModuleLoad, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULELOADDATA] = std::bind(&TallyServer::handle_cuModuleLoadData, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULELOADDATAEX] = std::bind(&TallyServer::handle_cuModuleLoadDataEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULELOADFATBINARY] = std::bind(&TallyServer::handle_cuModuleLoadFatBinary, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEUNLOAD] = std::bind(&TallyServer::handle_cuModuleUnload, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEGETFUNCTION] = std::bind(&TallyServer::handle_cuModuleGetFunction, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEGETGLOBAL_V2] = std::bind(&TallyServer::handle_cuModuleGetGlobal_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEGETTEXREF] = std::bind(&TallyServer::handle_cuModuleGetTexRef, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMODULEGETSURFREF] = std::bind(&TallyServer::handle_cuModuleGetSurfRef, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKCREATE_V2] = std::bind(&TallyServer::handle_cuLinkCreate_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKADDDATA_V2] = std::bind(&TallyServer::handle_cuLinkAddData_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKADDFILE_V2] = std::bind(&TallyServer::handle_cuLinkAddFile_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKCOMPLETE] = std::bind(&TallyServer::handle_cuLinkComplete, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULINKDESTROY] = std::bind(&TallyServer::handle_cuLinkDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETINFO_V2] = std::bind(&TallyServer::handle_cuMemGetInfo_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOC_V2] = std::bind(&TallyServer::handle_cuMemAlloc_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCPITCH_V2] = std::bind(&TallyServer::handle_cuMemAllocPitch_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMFREE_V2] = std::bind(&TallyServer::handle_cuMemFree_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETADDRESSRANGE_V2] = std::bind(&TallyServer::handle_cuMemGetAddressRange_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCHOST_V2] = std::bind(&TallyServer::handle_cuMemAllocHost_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMFREEHOST] = std::bind(&TallyServer::handle_cuMemFreeHost, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTALLOC] = std::bind(&TallyServer::handle_cuMemHostAlloc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTGETDEVICEPOINTER_V2] = std::bind(&TallyServer::handle_cuMemHostGetDevicePointer_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTGETFLAGS] = std::bind(&TallyServer::handle_cuMemHostGetFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCMANAGED] = std::bind(&TallyServer::handle_cuMemAllocManaged, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETBYPCIBUSID] = std::bind(&TallyServer::handle_cuDeviceGetByPCIBusId, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETPCIBUSID] = std::bind(&TallyServer::handle_cuDeviceGetPCIBusId, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCGETEVENTHANDLE] = std::bind(&TallyServer::handle_cuIpcGetEventHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCOPENEVENTHANDLE] = std::bind(&TallyServer::handle_cuIpcOpenEventHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCGETMEMHANDLE] = std::bind(&TallyServer::handle_cuIpcGetMemHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCOPENMEMHANDLE_V2] = std::bind(&TallyServer::handle_cuIpcOpenMemHandle_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUIPCCLOSEMEMHANDLE] = std::bind(&TallyServer::handle_cuIpcCloseMemHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTREGISTER_V2] = std::bind(&TallyServer::handle_cuMemHostRegister_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMHOSTUNREGISTER] = std::bind(&TallyServer::handle_cuMemHostUnregister, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY] = std::bind(&TallyServer::handle_cuMemcpy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYPEER] = std::bind(&TallyServer::handle_cuMemcpyPeer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYHTOD_V2] = std::bind(&TallyServer::handle_cuMemcpyHtoD_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTOH_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoH_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTOD_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoD_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTOA_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoA_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYATOD_V2] = std::bind(&TallyServer::handle_cuMemcpyAtoD_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYHTOA_V2] = std::bind(&TallyServer::handle_cuMemcpyHtoA_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYATOH_V2] = std::bind(&TallyServer::handle_cuMemcpyAtoH_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYATOA_V2] = std::bind(&TallyServer::handle_cuMemcpyAtoA_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY2D_V2] = std::bind(&TallyServer::handle_cuMemcpy2D_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY2DUNALIGNED_V2] = std::bind(&TallyServer::handle_cuMemcpy2DUnaligned_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY3D_V2] = std::bind(&TallyServer::handle_cuMemcpy3D_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY3DPEER] = std::bind(&TallyServer::handle_cuMemcpy3DPeer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYASYNC] = std::bind(&TallyServer::handle_cuMemcpyAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYPEERASYNC] = std::bind(&TallyServer::handle_cuMemcpyPeerAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYHTODASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyHtoDAsync_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTOHASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoHAsync_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYDTODASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyDtoDAsync_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYHTOAASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyHtoAAsync_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPYATOHASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpyAtoHAsync_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY2DASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpy2DAsync_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY3DASYNC_V2] = std::bind(&TallyServer::handle_cuMemcpy3DAsync_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCPY3DPEERASYNC] = std::bind(&TallyServer::handle_cuMemcpy3DPeerAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD8_V2] = std::bind(&TallyServer::handle_cuMemsetD8_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD16_V2] = std::bind(&TallyServer::handle_cuMemsetD16_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD32_V2] = std::bind(&TallyServer::handle_cuMemsetD32_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D8_V2] = std::bind(&TallyServer::handle_cuMemsetD2D8_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D16_V2] = std::bind(&TallyServer::handle_cuMemsetD2D16_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D32_V2] = std::bind(&TallyServer::handle_cuMemsetD2D32_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD8ASYNC] = std::bind(&TallyServer::handle_cuMemsetD8Async, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD16ASYNC] = std::bind(&TallyServer::handle_cuMemsetD16Async, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD32ASYNC] = std::bind(&TallyServer::handle_cuMemsetD32Async, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D8ASYNC] = std::bind(&TallyServer::handle_cuMemsetD2D8Async, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D16ASYNC] = std::bind(&TallyServer::handle_cuMemsetD2D16Async, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETD2D32ASYNC] = std::bind(&TallyServer::handle_cuMemsetD2D32Async, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYCREATE_V2] = std::bind(&TallyServer::handle_cuArrayCreate_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYGETDESCRIPTOR_V2] = std::bind(&TallyServer::handle_cuArrayGetDescriptor_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYGETSPARSEPROPERTIES] = std::bind(&TallyServer::handle_cuArrayGetSparseProperties, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMIPMAPPEDARRAYGETSPARSEPROPERTIES] = std::bind(&TallyServer::handle_cuMipmappedArrayGetSparseProperties, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYGETPLANE] = std::bind(&TallyServer::handle_cuArrayGetPlane, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAYDESTROY] = std::bind(&TallyServer::handle_cuArrayDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAY3DCREATE_V2] = std::bind(&TallyServer::handle_cuArray3DCreate_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUARRAY3DGETDESCRIPTOR_V2] = std::bind(&TallyServer::handle_cuArray3DGetDescriptor_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMIPMAPPEDARRAYCREATE] = std::bind(&TallyServer::handle_cuMipmappedArrayCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMIPMAPPEDARRAYGETLEVEL] = std::bind(&TallyServer::handle_cuMipmappedArrayGetLevel, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMIPMAPPEDARRAYDESTROY] = std::bind(&TallyServer::handle_cuMipmappedArrayDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMADDRESSRESERVE] = std::bind(&TallyServer::handle_cuMemAddressReserve, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMADDRESSFREE] = std::bind(&TallyServer::handle_cuMemAddressFree, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMCREATE] = std::bind(&TallyServer::handle_cuMemCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMRELEASE] = std::bind(&TallyServer::handle_cuMemRelease, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMMAP] = std::bind(&TallyServer::handle_cuMemMap, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMMAPARRAYASYNC] = std::bind(&TallyServer::handle_cuMemMapArrayAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMUNMAP] = std::bind(&TallyServer::handle_cuMemUnmap, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMSETACCESS] = std::bind(&TallyServer::handle_cuMemSetAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETACCESS] = std::bind(&TallyServer::handle_cuMemGetAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMEXPORTTOSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cuMemExportToShareableHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMIMPORTFROMSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cuMemImportFromShareableHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETALLOCATIONGRANULARITY] = std::bind(&TallyServer::handle_cuMemGetAllocationGranularity, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMGETALLOCATIONPROPERTIESFROMHANDLE] = std::bind(&TallyServer::handle_cuMemGetAllocationPropertiesFromHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMRETAINALLOCATIONHANDLE] = std::bind(&TallyServer::handle_cuMemRetainAllocationHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMFREEASYNC] = std::bind(&TallyServer::handle_cuMemFreeAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCASYNC] = std::bind(&TallyServer::handle_cuMemAllocAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLTRIMTO] = std::bind(&TallyServer::handle_cuMemPoolTrimTo, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLSETATTRIBUTE] = std::bind(&TallyServer::handle_cuMemPoolSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLGETATTRIBUTE] = std::bind(&TallyServer::handle_cuMemPoolGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLSETACCESS] = std::bind(&TallyServer::handle_cuMemPoolSetAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLGETACCESS] = std::bind(&TallyServer::handle_cuMemPoolGetAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLCREATE] = std::bind(&TallyServer::handle_cuMemPoolCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLDESTROY] = std::bind(&TallyServer::handle_cuMemPoolDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMALLOCFROMPOOLASYNC] = std::bind(&TallyServer::handle_cuMemAllocFromPoolAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLEXPORTTOSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cuMemPoolExportToShareableHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLIMPORTFROMSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cuMemPoolImportFromShareableHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLEXPORTPOINTER] = std::bind(&TallyServer::handle_cuMemPoolExportPointer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPOOLIMPORTPOINTER] = std::bind(&TallyServer::handle_cuMemPoolImportPointer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUPOINTERGETATTRIBUTE] = std::bind(&TallyServer::handle_cuPointerGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMPREFETCHASYNC] = std::bind(&TallyServer::handle_cuMemPrefetchAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMADVISE] = std::bind(&TallyServer::handle_cuMemAdvise, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMRANGEGETATTRIBUTE] = std::bind(&TallyServer::handle_cuMemRangeGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUMEMRANGEGETATTRIBUTES] = std::bind(&TallyServer::handle_cuMemRangeGetAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUPOINTERSETATTRIBUTE] = std::bind(&TallyServer::handle_cuPointerSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUPOINTERGETATTRIBUTES] = std::bind(&TallyServer::handle_cuPointerGetAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMCREATE] = std::bind(&TallyServer::handle_cuStreamCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMCREATEWITHPRIORITY] = std::bind(&TallyServer::handle_cuStreamCreateWithPriority, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETPRIORITY] = std::bind(&TallyServer::handle_cuStreamGetPriority, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETFLAGS] = std::bind(&TallyServer::handle_cuStreamGetFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETCTX] = std::bind(&TallyServer::handle_cuStreamGetCtx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWAITEVENT] = std::bind(&TallyServer::handle_cuStreamWaitEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMADDCALLBACK] = std::bind(&TallyServer::handle_cuStreamAddCallback, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMBEGINCAPTURE_V2] = std::bind(&TallyServer::handle_cuStreamBeginCapture_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTHREADEXCHANGESTREAMCAPTUREMODE] = std::bind(&TallyServer::handle_cuThreadExchangeStreamCaptureMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMENDCAPTURE] = std::bind(&TallyServer::handle_cuStreamEndCapture, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMISCAPTURING] = std::bind(&TallyServer::handle_cuStreamIsCapturing, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETCAPTUREINFO] = std::bind(&TallyServer::handle_cuStreamGetCaptureInfo, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETCAPTUREINFO_V2] = std::bind(&TallyServer::handle_cuStreamGetCaptureInfo_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMUPDATECAPTUREDEPENDENCIES] = std::bind(&TallyServer::handle_cuStreamUpdateCaptureDependencies, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMATTACHMEMASYNC] = std::bind(&TallyServer::handle_cuStreamAttachMemAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMQUERY] = std::bind(&TallyServer::handle_cuStreamQuery, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMSYNCHRONIZE] = std::bind(&TallyServer::handle_cuStreamSynchronize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMDESTROY_V2] = std::bind(&TallyServer::handle_cuStreamDestroy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMCOPYATTRIBUTES] = std::bind(&TallyServer::handle_cuStreamCopyAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMGETATTRIBUTE] = std::bind(&TallyServer::handle_cuStreamGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMSETATTRIBUTE] = std::bind(&TallyServer::handle_cuStreamSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTCREATE] = std::bind(&TallyServer::handle_cuEventCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTRECORD] = std::bind(&TallyServer::handle_cuEventRecord, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTRECORDWITHFLAGS] = std::bind(&TallyServer::handle_cuEventRecordWithFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTQUERY] = std::bind(&TallyServer::handle_cuEventQuery, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTSYNCHRONIZE] = std::bind(&TallyServer::handle_cuEventSynchronize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTDESTROY_V2] = std::bind(&TallyServer::handle_cuEventDestroy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUEVENTELAPSEDTIME] = std::bind(&TallyServer::handle_cuEventElapsedTime, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUIMPORTEXTERNALMEMORY] = std::bind(&TallyServer::handle_cuImportExternalMemory, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUEXTERNALMEMORYGETMAPPEDBUFFER] = std::bind(&TallyServer::handle_cuExternalMemoryGetMappedBuffer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUEXTERNALMEMORYGETMAPPEDMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cuExternalMemoryGetMappedMipmappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDESTROYEXTERNALMEMORY] = std::bind(&TallyServer::handle_cuDestroyExternalMemory, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUIMPORTEXTERNALSEMAPHORE] = std::bind(&TallyServer::handle_cuImportExternalSemaphore, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSIGNALEXTERNALSEMAPHORESASYNC] = std::bind(&TallyServer::handle_cuSignalExternalSemaphoresAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUWAITEXTERNALSEMAPHORESASYNC] = std::bind(&TallyServer::handle_cuWaitExternalSemaphoresAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDESTROYEXTERNALSEMAPHORE] = std::bind(&TallyServer::handle_cuDestroyExternalSemaphore, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWAITVALUE32] = std::bind(&TallyServer::handle_cuStreamWaitValue32, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWAITVALUE64] = std::bind(&TallyServer::handle_cuStreamWaitValue64, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWRITEVALUE32] = std::bind(&TallyServer::handle_cuStreamWriteValue32, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMWRITEVALUE64] = std::bind(&TallyServer::handle_cuStreamWriteValue64, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSTREAMBATCHMEMOP] = std::bind(&TallyServer::handle_cuStreamBatchMemOp, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCGETATTRIBUTE] = std::bind(&TallyServer::handle_cuFuncGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETATTRIBUTE] = std::bind(&TallyServer::handle_cuFuncSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETCACHECONFIG] = std::bind(&TallyServer::handle_cuFuncSetCacheConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cuFuncSetSharedMemConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCGETMODULE] = std::bind(&TallyServer::handle_cuFuncGetModule, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHKERNEL] = std::bind(&TallyServer::handle_cuLaunchKernel, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHCOOPERATIVEKERNEL] = std::bind(&TallyServer::handle_cuLaunchCooperativeKernel, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHCOOPERATIVEKERNELMULTIDEVICE] = std::bind(&TallyServer::handle_cuLaunchCooperativeKernelMultiDevice, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHHOSTFUNC] = std::bind(&TallyServer::handle_cuLaunchHostFunc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETBLOCKSHAPE] = std::bind(&TallyServer::handle_cuFuncSetBlockShape, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUFUNCSETSHAREDSIZE] = std::bind(&TallyServer::handle_cuFuncSetSharedSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETSIZE] = std::bind(&TallyServer::handle_cuParamSetSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETI] = std::bind(&TallyServer::handle_cuParamSeti, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETF] = std::bind(&TallyServer::handle_cuParamSetf, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETV] = std::bind(&TallyServer::handle_cuParamSetv, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCH] = std::bind(&TallyServer::handle_cuLaunch, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHGRID] = std::bind(&TallyServer::handle_cuLaunchGrid, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CULAUNCHGRIDASYNC] = std::bind(&TallyServer::handle_cuLaunchGridAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUPARAMSETTEXREF] = std::bind(&TallyServer::handle_cuParamSetTexRef, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHCREATE] = std::bind(&TallyServer::handle_cuGraphCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDKERNELNODE] = std::bind(&TallyServer::handle_cuGraphAddKernelNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphKernelNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphKernelNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDMEMCPYNODE] = std::bind(&TallyServer::handle_cuGraphAddMemcpyNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMCPYNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemcpyNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMCPYNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemcpyNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDMEMSETNODE] = std::bind(&TallyServer::handle_cuGraphAddMemsetNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMSETNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemsetNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMSETNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemsetNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDHOSTNODE] = std::bind(&TallyServer::handle_cuGraphAddHostNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHHOSTNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphHostNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHHOSTNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphHostNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDCHILDGRAPHNODE] = std::bind(&TallyServer::handle_cuGraphAddChildGraphNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHCHILDGRAPHNODEGETGRAPH] = std::bind(&TallyServer::handle_cuGraphChildGraphNodeGetGraph, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEMPTYNODE] = std::bind(&TallyServer::handle_cuGraphAddEmptyNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEVENTRECORDNODE] = std::bind(&TallyServer::handle_cuGraphAddEventRecordNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEVENTRECORDNODEGETEVENT] = std::bind(&TallyServer::handle_cuGraphEventRecordNodeGetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEVENTRECORDNODESETEVENT] = std::bind(&TallyServer::handle_cuGraphEventRecordNodeSetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEVENTWAITNODE] = std::bind(&TallyServer::handle_cuGraphAddEventWaitNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEVENTWAITNODEGETEVENT] = std::bind(&TallyServer::handle_cuGraphEventWaitNodeGetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEVENTWAITNODESETEVENT] = std::bind(&TallyServer::handle_cuGraphEventWaitNodeSetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEXTERNALSEMAPHORESSIGNALNODE] = std::bind(&TallyServer::handle_cuGraphAddExternalSemaphoresSignalNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXTERNALSEMAPHORESSIGNALNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphExternalSemaphoresSignalNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXTERNALSEMAPHORESSIGNALNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExternalSemaphoresSignalNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDEXTERNALSEMAPHORESWAITNODE] = std::bind(&TallyServer::handle_cuGraphAddExternalSemaphoresWaitNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXTERNALSEMAPHORESWAITNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphExternalSemaphoresWaitNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXTERNALSEMAPHORESWAITNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExternalSemaphoresWaitNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDMEMALLOCNODE] = std::bind(&TallyServer::handle_cuGraphAddMemAllocNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMALLOCNODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemAllocNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDMEMFREENODE] = std::bind(&TallyServer::handle_cuGraphAddMemFreeNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHMEMFREENODEGETPARAMS] = std::bind(&TallyServer::handle_cuGraphMemFreeNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGRAPHMEMTRIM] = std::bind(&TallyServer::handle_cuDeviceGraphMemTrim, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETGRAPHMEMATTRIBUTE] = std::bind(&TallyServer::handle_cuDeviceGetGraphMemAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICESETGRAPHMEMATTRIBUTE] = std::bind(&TallyServer::handle_cuDeviceSetGraphMemAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHCLONE] = std::bind(&TallyServer::handle_cuGraphClone, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODEFINDINCLONE] = std::bind(&TallyServer::handle_cuGraphNodeFindInClone, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODEGETTYPE] = std::bind(&TallyServer::handle_cuGraphNodeGetType, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHGETNODES] = std::bind(&TallyServer::handle_cuGraphGetNodes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHGETROOTNODES] = std::bind(&TallyServer::handle_cuGraphGetRootNodes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHGETEDGES] = std::bind(&TallyServer::handle_cuGraphGetEdges, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODEGETDEPENDENCIES] = std::bind(&TallyServer::handle_cuGraphNodeGetDependencies, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHNODEGETDEPENDENTNODES] = std::bind(&TallyServer::handle_cuGraphNodeGetDependentNodes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHADDDEPENDENCIES] = std::bind(&TallyServer::handle_cuGraphAddDependencies, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHREMOVEDEPENDENCIES] = std::bind(&TallyServer::handle_cuGraphRemoveDependencies, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHDESTROYNODE] = std::bind(&TallyServer::handle_cuGraphDestroyNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHINSTANTIATE_V2] = std::bind(&TallyServer::handle_cuGraphInstantiate_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHINSTANTIATEWITHFLAGS] = std::bind(&TallyServer::handle_cuGraphInstantiateWithFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECKERNELNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecKernelNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECMEMCPYNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecMemcpyNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECMEMSETNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecMemsetNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECHOSTNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecHostNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECCHILDGRAPHNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecChildGraphNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECEVENTRECORDNODESETEVENT] = std::bind(&TallyServer::handle_cuGraphExecEventRecordNodeSetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECEVENTWAITNODESETEVENT] = std::bind(&TallyServer::handle_cuGraphExecEventWaitNodeSetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECEXTERNALSEMAPHORESSIGNALNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecExternalSemaphoresSignalNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECEXTERNALSEMAPHORESWAITNODESETPARAMS] = std::bind(&TallyServer::handle_cuGraphExecExternalSemaphoresWaitNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHUPLOAD] = std::bind(&TallyServer::handle_cuGraphUpload, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHLAUNCH] = std::bind(&TallyServer::handle_cuGraphLaunch, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECDESTROY] = std::bind(&TallyServer::handle_cuGraphExecDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHDESTROY] = std::bind(&TallyServer::handle_cuGraphDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHEXECUPDATE] = std::bind(&TallyServer::handle_cuGraphExecUpdate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODECOPYATTRIBUTES] = std::bind(&TallyServer::handle_cuGraphKernelNodeCopyAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODEGETATTRIBUTE] = std::bind(&TallyServer::handle_cuGraphKernelNodeGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHKERNELNODESETATTRIBUTE] = std::bind(&TallyServer::handle_cuGraphKernelNodeSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHDEBUGDOTPRINT] = std::bind(&TallyServer::handle_cuGraphDebugDotPrint, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUUSEROBJECTCREATE] = std::bind(&TallyServer::handle_cuUserObjectCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUUSEROBJECTRETAIN] = std::bind(&TallyServer::handle_cuUserObjectRetain, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUUSEROBJECTRELEASE] = std::bind(&TallyServer::handle_cuUserObjectRelease, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHRETAINUSEROBJECT] = std::bind(&TallyServer::handle_cuGraphRetainUserObject, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHRELEASEUSEROBJECT] = std::bind(&TallyServer::handle_cuGraphReleaseUserObject, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR] = std::bind(&TallyServer::handle_cuOccupancyMaxActiveBlocksPerMultiprocessor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSORWITHFLAGS] = std::bind(&TallyServer::handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXPOTENTIALBLOCKSIZE] = std::bind(&TallyServer::handle_cuOccupancyMaxPotentialBlockSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYMAXPOTENTIALBLOCKSIZEWITHFLAGS] = std::bind(&TallyServer::handle_cuOccupancyMaxPotentialBlockSizeWithFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUOCCUPANCYAVAILABLEDYNAMICSMEMPERBLOCK] = std::bind(&TallyServer::handle_cuOccupancyAvailableDynamicSMemPerBlock, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETARRAY] = std::bind(&TallyServer::handle_cuTexRefSetArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cuTexRefSetMipmappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETADDRESS_V2] = std::bind(&TallyServer::handle_cuTexRefSetAddress_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETADDRESS2D_V3] = std::bind(&TallyServer::handle_cuTexRefSetAddress2D_v3, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETADDRESSMODE] = std::bind(&TallyServer::handle_cuTexRefSetAddressMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETFILTERMODE] = std::bind(&TallyServer::handle_cuTexRefSetFilterMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMIPMAPFILTERMODE] = std::bind(&TallyServer::handle_cuTexRefSetMipmapFilterMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMIPMAPLEVELBIAS] = std::bind(&TallyServer::handle_cuTexRefSetMipmapLevelBias, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMIPMAPLEVELCLAMP] = std::bind(&TallyServer::handle_cuTexRefSetMipmapLevelClamp, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETMAXANISOTROPY] = std::bind(&TallyServer::handle_cuTexRefSetMaxAnisotropy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETBORDERCOLOR] = std::bind(&TallyServer::handle_cuTexRefSetBorderColor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFSETFLAGS] = std::bind(&TallyServer::handle_cuTexRefSetFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETADDRESS_V2] = std::bind(&TallyServer::handle_cuTexRefGetAddress_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETARRAY] = std::bind(&TallyServer::handle_cuTexRefGetArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cuTexRefGetMipmappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETADDRESSMODE] = std::bind(&TallyServer::handle_cuTexRefGetAddressMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETFILTERMODE] = std::bind(&TallyServer::handle_cuTexRefGetFilterMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMIPMAPFILTERMODE] = std::bind(&TallyServer::handle_cuTexRefGetMipmapFilterMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMIPMAPLEVELBIAS] = std::bind(&TallyServer::handle_cuTexRefGetMipmapLevelBias, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMIPMAPLEVELCLAMP] = std::bind(&TallyServer::handle_cuTexRefGetMipmapLevelClamp, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETMAXANISOTROPY] = std::bind(&TallyServer::handle_cuTexRefGetMaxAnisotropy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETBORDERCOLOR] = std::bind(&TallyServer::handle_cuTexRefGetBorderColor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFGETFLAGS] = std::bind(&TallyServer::handle_cuTexRefGetFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFCREATE] = std::bind(&TallyServer::handle_cuTexRefCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXREFDESTROY] = std::bind(&TallyServer::handle_cuTexRefDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFREFSETARRAY] = std::bind(&TallyServer::handle_cuSurfRefSetArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFREFGETARRAY] = std::bind(&TallyServer::handle_cuSurfRefGetArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTCREATE] = std::bind(&TallyServer::handle_cuTexObjectCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTDESTROY] = std::bind(&TallyServer::handle_cuTexObjectDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTGETRESOURCEDESC] = std::bind(&TallyServer::handle_cuTexObjectGetResourceDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTGETTEXTUREDESC] = std::bind(&TallyServer::handle_cuTexObjectGetTextureDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUTEXOBJECTGETRESOURCEVIEWDESC] = std::bind(&TallyServer::handle_cuTexObjectGetResourceViewDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFOBJECTCREATE] = std::bind(&TallyServer::handle_cuSurfObjectCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFOBJECTDESTROY] = std::bind(&TallyServer::handle_cuSurfObjectDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUSURFOBJECTGETRESOURCEDESC] = std::bind(&TallyServer::handle_cuSurfObjectGetResourceDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICECANACCESSPEER] = std::bind(&TallyServer::handle_cuDeviceCanAccessPeer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXENABLEPEERACCESS] = std::bind(&TallyServer::handle_cuCtxEnablePeerAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUCTXDISABLEPEERACCESS] = std::bind(&TallyServer::handle_cuCtxDisablePeerAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDEVICEGETP2PATTRIBUTE] = std::bind(&TallyServer::handle_cuDeviceGetP2PAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSUNREGISTERRESOURCE] = std::bind(&TallyServer::handle_cuGraphicsUnregisterResource, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSSUBRESOURCEGETMAPPEDARRAY] = std::bind(&TallyServer::handle_cuGraphicsSubResourceGetMappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSRESOURCEGETMAPPEDMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cuGraphicsResourceGetMappedMipmappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSRESOURCEGETMAPPEDPOINTER_V2] = std::bind(&TallyServer::handle_cuGraphicsResourceGetMappedPointer_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSRESOURCESETMAPFLAGS_V2] = std::bind(&TallyServer::handle_cuGraphicsResourceSetMapFlags_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSMAPRESOURCES] = std::bind(&TallyServer::handle_cuGraphicsMapResources, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGRAPHICSUNMAPRESOURCES] = std::bind(&TallyServer::handle_cuGraphicsUnmapResources, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGETPROCADDRESS] = std::bind(&TallyServer::handle_cuGetProcAddress, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUGETEXPORTTABLE] = std::bind(&TallyServer::handle_cuGetExportTable, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICERESET] = std::bind(&TallyServer::handle_cudaDeviceReset, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESYNCHRONIZE] = std::bind(&TallyServer::handle_cudaDeviceSynchronize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETLIMIT] = std::bind(&TallyServer::handle_cudaDeviceSetLimit, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETLIMIT] = std::bind(&TallyServer::handle_cudaDeviceGetLimit, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETTEXTURE1DLINEARMAXWIDTH] = std::bind(&TallyServer::handle_cudaDeviceGetTexture1DLinearMaxWidth, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETCACHECONFIG] = std::bind(&TallyServer::handle_cudaDeviceGetCacheConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETSTREAMPRIORITYRANGE] = std::bind(&TallyServer::handle_cudaDeviceGetStreamPriorityRange, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETCACHECONFIG] = std::bind(&TallyServer::handle_cudaDeviceSetCacheConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cudaDeviceGetSharedMemConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cudaDeviceSetSharedMemConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETBYPCIBUSID] = std::bind(&TallyServer::handle_cudaDeviceGetByPCIBusId, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETPCIBUSID] = std::bind(&TallyServer::handle_cudaDeviceGetPCIBusId, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCGETEVENTHANDLE] = std::bind(&TallyServer::handle_cudaIpcGetEventHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCOPENEVENTHANDLE] = std::bind(&TallyServer::handle_cudaIpcOpenEventHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCGETMEMHANDLE] = std::bind(&TallyServer::handle_cudaIpcGetMemHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCOPENMEMHANDLE] = std::bind(&TallyServer::handle_cudaIpcOpenMemHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIPCCLOSEMEMHANDLE] = std::bind(&TallyServer::handle_cudaIpcCloseMemHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEFLUSHGPUDIRECTRDMAWRITES] = std::bind(&TallyServer::handle_cudaDeviceFlushGPUDirectRDMAWrites, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADEXIT] = std::bind(&TallyServer::handle_cudaThreadExit, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADSYNCHRONIZE] = std::bind(&TallyServer::handle_cudaThreadSynchronize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADSETLIMIT] = std::bind(&TallyServer::handle_cudaThreadSetLimit, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADGETLIMIT] = std::bind(&TallyServer::handle_cudaThreadGetLimit, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADGETCACHECONFIG] = std::bind(&TallyServer::handle_cudaThreadGetCacheConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADSETCACHECONFIG] = std::bind(&TallyServer::handle_cudaThreadSetCacheConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETLASTERROR] = std::bind(&TallyServer::handle_cudaGetLastError, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAPEEKATLASTERROR] = std::bind(&TallyServer::handle_cudaPeekAtLastError, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETERRORNAME] = std::bind(&TallyServer::handle_cudaGetErrorName, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETERRORSTRING] = std::bind(&TallyServer::handle_cudaGetErrorString, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDEVICECOUNT] = std::bind(&TallyServer::handle_cudaGetDeviceCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDEVICEPROPERTIES] = std::bind(&TallyServer::handle_cudaGetDeviceProperties, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaDeviceGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETDEFAULTMEMPOOL] = std::bind(&TallyServer::handle_cudaDeviceGetDefaultMemPool, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETMEMPOOL] = std::bind(&TallyServer::handle_cudaDeviceSetMemPool, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETMEMPOOL] = std::bind(&TallyServer::handle_cudaDeviceGetMemPool, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETNVSCISYNCATTRIBUTES] = std::bind(&TallyServer::handle_cudaDeviceGetNvSciSyncAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETP2PATTRIBUTE] = std::bind(&TallyServer::handle_cudaDeviceGetP2PAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACHOOSEDEVICE] = std::bind(&TallyServer::handle_cudaChooseDevice, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETDEVICE] = std::bind(&TallyServer::handle_cudaSetDevice, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDEVICE] = std::bind(&TallyServer::handle_cudaGetDevice, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETVALIDDEVICES] = std::bind(&TallyServer::handle_cudaSetValidDevices, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETDEVICEFLAGS] = std::bind(&TallyServer::handle_cudaSetDeviceFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDEVICEFLAGS] = std::bind(&TallyServer::handle_cudaGetDeviceFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMCREATE] = std::bind(&TallyServer::handle_cudaStreamCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMCREATEWITHFLAGS] = std::bind(&TallyServer::handle_cudaStreamCreateWithFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMCREATEWITHPRIORITY] = std::bind(&TallyServer::handle_cudaStreamCreateWithPriority, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETPRIORITY] = std::bind(&TallyServer::handle_cudaStreamGetPriority, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETFLAGS] = std::bind(&TallyServer::handle_cudaStreamGetFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACTXRESETPERSISTINGL2CACHE] = std::bind(&TallyServer::handle_cudaCtxResetPersistingL2Cache, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMCOPYATTRIBUTES] = std::bind(&TallyServer::handle_cudaStreamCopyAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaStreamGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMSETATTRIBUTE] = std::bind(&TallyServer::handle_cudaStreamSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMDESTROY] = std::bind(&TallyServer::handle_cudaStreamDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMWAITEVENT] = std::bind(&TallyServer::handle_cudaStreamWaitEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMADDCALLBACK] = std::bind(&TallyServer::handle_cudaStreamAddCallback, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMSYNCHRONIZE] = std::bind(&TallyServer::handle_cudaStreamSynchronize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMQUERY] = std::bind(&TallyServer::handle_cudaStreamQuery, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMATTACHMEMASYNC] = std::bind(&TallyServer::handle_cudaStreamAttachMemAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMBEGINCAPTURE] = std::bind(&TallyServer::handle_cudaStreamBeginCapture, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDATHREADEXCHANGESTREAMCAPTUREMODE] = std::bind(&TallyServer::handle_cudaThreadExchangeStreamCaptureMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMENDCAPTURE] = std::bind(&TallyServer::handle_cudaStreamEndCapture, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMISCAPTURING] = std::bind(&TallyServer::handle_cudaStreamIsCapturing, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETCAPTUREINFO] = std::bind(&TallyServer::handle_cudaStreamGetCaptureInfo, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMGETCAPTUREINFO_V2] = std::bind(&TallyServer::handle_cudaStreamGetCaptureInfo_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASTREAMUPDATECAPTUREDEPENDENCIES] = std::bind(&TallyServer::handle_cudaStreamUpdateCaptureDependencies, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTCREATE] = std::bind(&TallyServer::handle_cudaEventCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTCREATEWITHFLAGS] = std::bind(&TallyServer::handle_cudaEventCreateWithFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTRECORD] = std::bind(&TallyServer::handle_cudaEventRecord, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTRECORDWITHFLAGS] = std::bind(&TallyServer::handle_cudaEventRecordWithFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTQUERY] = std::bind(&TallyServer::handle_cudaEventQuery, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTSYNCHRONIZE] = std::bind(&TallyServer::handle_cudaEventSynchronize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTDESTROY] = std::bind(&TallyServer::handle_cudaEventDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEVENTELAPSEDTIME] = std::bind(&TallyServer::handle_cudaEventElapsedTime, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIMPORTEXTERNALMEMORY] = std::bind(&TallyServer::handle_cudaImportExternalMemory, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEXTERNALMEMORYGETMAPPEDBUFFER] = std::bind(&TallyServer::handle_cudaExternalMemoryGetMappedBuffer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAEXTERNALMEMORYGETMAPPEDMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaExternalMemoryGetMappedMipmappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADESTROYEXTERNALMEMORY] = std::bind(&TallyServer::handle_cudaDestroyExternalMemory, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAIMPORTEXTERNALSEMAPHORE] = std::bind(&TallyServer::handle_cudaImportExternalSemaphore, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASIGNALEXTERNALSEMAPHORESASYNC_V2] = std::bind(&TallyServer::handle_cudaSignalExternalSemaphoresAsync_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAWAITEXTERNALSEMAPHORESASYNC_V2] = std::bind(&TallyServer::handle_cudaWaitExternalSemaphoresAsync_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADESTROYEXTERNALSEMAPHORE] = std::bind(&TallyServer::handle_cudaDestroyExternalSemaphore, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHKERNEL] = std::bind(&TallyServer::handle_cudaLaunchKernel, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHCOOPERATIVEKERNEL] = std::bind(&TallyServer::handle_cudaLaunchCooperativeKernel, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHCOOPERATIVEKERNELMULTIDEVICE] = std::bind(&TallyServer::handle_cudaLaunchCooperativeKernelMultiDevice, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFUNCSETCACHECONFIG] = std::bind(&TallyServer::handle_cudaFuncSetCacheConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFUNCSETSHAREDMEMCONFIG] = std::bind(&TallyServer::handle_cudaFuncSetSharedMemConfig, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFUNCGETATTRIBUTES] = std::bind(&TallyServer::handle_cudaFuncGetAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFUNCSETATTRIBUTE] = std::bind(&TallyServer::handle_cudaFuncSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETDOUBLEFORDEVICE] = std::bind(&TallyServer::handle_cudaSetDoubleForDevice, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDASETDOUBLEFORHOST] = std::bind(&TallyServer::handle_cudaSetDoubleForHost, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHHOSTFUNC] = std::bind(&TallyServer::handle_cudaLaunchHostFunc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR] = std::bind(&TallyServer::handle_cudaOccupancyMaxActiveBlocksPerMultiprocessor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAOCCUPANCYAVAILABLEDYNAMICSMEMPERBLOCK] = std::bind(&TallyServer::handle_cudaOccupancyAvailableDynamicSMemPerBlock, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSORWITHFLAGS] = std::bind(&TallyServer::handle_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCMANAGED] = std::bind(&TallyServer::handle_cudaMallocManaged, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOC] = std::bind(&TallyServer::handle_cudaMalloc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCHOST] = std::bind(&TallyServer::handle_cudaMallocHost, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCPITCH] = std::bind(&TallyServer::handle_cudaMallocPitch, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCARRAY] = std::bind(&TallyServer::handle_cudaMallocArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREE] = std::bind(&TallyServer::handle_cudaFree, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREEHOST] = std::bind(&TallyServer::handle_cudaFreeHost, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREEARRAY] = std::bind(&TallyServer::handle_cudaFreeArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREEMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaFreeMipmappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTALLOC] = std::bind(&TallyServer::handle_cudaHostAlloc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTREGISTER] = std::bind(&TallyServer::handle_cudaHostRegister, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTUNREGISTER] = std::bind(&TallyServer::handle_cudaHostUnregister, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTGETDEVICEPOINTER] = std::bind(&TallyServer::handle_cudaHostGetDevicePointer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAHOSTGETFLAGS] = std::bind(&TallyServer::handle_cudaHostGetFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOC3D] = std::bind(&TallyServer::handle_cudaMalloc3D, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOC3DARRAY] = std::bind(&TallyServer::handle_cudaMalloc3DArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaMallocMipmappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETMIPMAPPEDARRAYLEVEL] = std::bind(&TallyServer::handle_cudaGetMipmappedArrayLevel, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY3D] = std::bind(&TallyServer::handle_cudaMemcpy3D, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY3DPEER] = std::bind(&TallyServer::handle_cudaMemcpy3DPeer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY3DASYNC] = std::bind(&TallyServer::handle_cudaMemcpy3DAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY3DPEERASYNC] = std::bind(&TallyServer::handle_cudaMemcpy3DPeerAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMGETINFO] = std::bind(&TallyServer::handle_cudaMemGetInfo, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAARRAYGETINFO] = std::bind(&TallyServer::handle_cudaArrayGetInfo, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAARRAYGETPLANE] = std::bind(&TallyServer::handle_cudaArrayGetPlane, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAARRAYGETSPARSEPROPERTIES] = std::bind(&TallyServer::handle_cudaArrayGetSparseProperties, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMIPMAPPEDARRAYGETSPARSEPROPERTIES] = std::bind(&TallyServer::handle_cudaMipmappedArrayGetSparseProperties, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY] = std::bind(&TallyServer::handle_cudaMemcpy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYPEER] = std::bind(&TallyServer::handle_cudaMemcpyPeer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2D] = std::bind(&TallyServer::handle_cudaMemcpy2D, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DTOARRAY] = std::bind(&TallyServer::handle_cudaMemcpy2DToArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DFROMARRAY] = std::bind(&TallyServer::handle_cudaMemcpy2DFromArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DARRAYTOARRAY] = std::bind(&TallyServer::handle_cudaMemcpy2DArrayToArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYTOSYMBOL] = std::bind(&TallyServer::handle_cudaMemcpyToSymbol, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYFROMSYMBOL] = std::bind(&TallyServer::handle_cudaMemcpyFromSymbol, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYASYNC] = std::bind(&TallyServer::handle_cudaMemcpyAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYPEERASYNC] = std::bind(&TallyServer::handle_cudaMemcpyPeerAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DASYNC] = std::bind(&TallyServer::handle_cudaMemcpy2DAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DTOARRAYASYNC] = std::bind(&TallyServer::handle_cudaMemcpy2DToArrayAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY2DFROMARRAYASYNC] = std::bind(&TallyServer::handle_cudaMemcpy2DFromArrayAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYTOSYMBOLASYNC] = std::bind(&TallyServer::handle_cudaMemcpyToSymbolAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYFROMSYMBOLASYNC] = std::bind(&TallyServer::handle_cudaMemcpyFromSymbolAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET] = std::bind(&TallyServer::handle_cudaMemset, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET2D] = std::bind(&TallyServer::handle_cudaMemset2D, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET3D] = std::bind(&TallyServer::handle_cudaMemset3D, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSETASYNC] = std::bind(&TallyServer::handle_cudaMemsetAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET2DASYNC] = std::bind(&TallyServer::handle_cudaMemset2DAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMSET3DASYNC] = std::bind(&TallyServer::handle_cudaMemset3DAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETSYMBOLADDRESS] = std::bind(&TallyServer::handle_cudaGetSymbolAddress, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETSYMBOLSIZE] = std::bind(&TallyServer::handle_cudaGetSymbolSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPREFETCHASYNC] = std::bind(&TallyServer::handle_cudaMemPrefetchAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMADVISE] = std::bind(&TallyServer::handle_cudaMemAdvise, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMRANGEGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaMemRangeGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMRANGEGETATTRIBUTES] = std::bind(&TallyServer::handle_cudaMemRangeGetAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYTOARRAY] = std::bind(&TallyServer::handle_cudaMemcpyToArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYFROMARRAY] = std::bind(&TallyServer::handle_cudaMemcpyFromArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYARRAYTOARRAY] = std::bind(&TallyServer::handle_cudaMemcpyArrayToArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYTOARRAYASYNC] = std::bind(&TallyServer::handle_cudaMemcpyToArrayAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPYFROMARRAYASYNC] = std::bind(&TallyServer::handle_cudaMemcpyFromArrayAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCASYNC] = std::bind(&TallyServer::handle_cudaMallocAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAFREEASYNC] = std::bind(&TallyServer::handle_cudaFreeAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLTRIMTO] = std::bind(&TallyServer::handle_cudaMemPoolTrimTo, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLSETATTRIBUTE] = std::bind(&TallyServer::handle_cudaMemPoolSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaMemPoolGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLSETACCESS] = std::bind(&TallyServer::handle_cudaMemPoolSetAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLGETACCESS] = std::bind(&TallyServer::handle_cudaMemPoolGetAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLCREATE] = std::bind(&TallyServer::handle_cudaMemPoolCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLDESTROY] = std::bind(&TallyServer::handle_cudaMemPoolDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOCFROMPOOLASYNC] = std::bind(&TallyServer::handle_cudaMallocFromPoolAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLEXPORTTOSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cudaMemPoolExportToShareableHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLIMPORTFROMSHAREABLEHANDLE] = std::bind(&TallyServer::handle_cudaMemPoolImportFromShareableHandle, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLEXPORTPOINTER] = std::bind(&TallyServer::handle_cudaMemPoolExportPointer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMPOOLIMPORTPOINTER] = std::bind(&TallyServer::handle_cudaMemPoolImportPointer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAPOINTERGETATTRIBUTES] = std::bind(&TallyServer::handle_cudaPointerGetAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICECANACCESSPEER] = std::bind(&TallyServer::handle_cudaDeviceCanAccessPeer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEENABLEPEERACCESS] = std::bind(&TallyServer::handle_cudaDeviceEnablePeerAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEDISABLEPEERACCESS] = std::bind(&TallyServer::handle_cudaDeviceDisablePeerAccess, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSUNREGISTERRESOURCE] = std::bind(&TallyServer::handle_cudaGraphicsUnregisterResource, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSRESOURCESETMAPFLAGS] = std::bind(&TallyServer::handle_cudaGraphicsResourceSetMapFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSMAPRESOURCES] = std::bind(&TallyServer::handle_cudaGraphicsMapResources, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSUNMAPRESOURCES] = std::bind(&TallyServer::handle_cudaGraphicsUnmapResources, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSRESOURCEGETMAPPEDPOINTER] = std::bind(&TallyServer::handle_cudaGraphicsResourceGetMappedPointer, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSSUBRESOURCEGETMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaGraphicsSubResourceGetMappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHICSRESOURCEGETMAPPEDMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaGraphicsResourceGetMappedMipmappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDABINDTEXTURE] = std::bind(&TallyServer::handle_cudaBindTexture, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDABINDTEXTURE2D] = std::bind(&TallyServer::handle_cudaBindTexture2D, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDABINDTEXTURETOARRAY] = std::bind(&TallyServer::handle_cudaBindTextureToArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDABINDTEXTURETOMIPMAPPEDARRAY] = std::bind(&TallyServer::handle_cudaBindTextureToMipmappedArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAUNBINDTEXTURE] = std::bind(&TallyServer::handle_cudaUnbindTexture, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETTEXTUREALIGNMENTOFFSET] = std::bind(&TallyServer::handle_cudaGetTextureAlignmentOffset, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETTEXTUREREFERENCE] = std::bind(&TallyServer::handle_cudaGetTextureReference, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDABINDSURFACETOARRAY] = std::bind(&TallyServer::handle_cudaBindSurfaceToArray, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETSURFACEREFERENCE] = std::bind(&TallyServer::handle_cudaGetSurfaceReference, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETCHANNELDESC] = std::bind(&TallyServer::handle_cudaGetChannelDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACREATECHANNELDESC] = std::bind(&TallyServer::handle_cudaCreateChannelDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACREATETEXTUREOBJECT] = std::bind(&TallyServer::handle_cudaCreateTextureObject, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADESTROYTEXTUREOBJECT] = std::bind(&TallyServer::handle_cudaDestroyTextureObject, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETTEXTUREOBJECTRESOURCEDESC] = std::bind(&TallyServer::handle_cudaGetTextureObjectResourceDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETTEXTUREOBJECTTEXTUREDESC] = std::bind(&TallyServer::handle_cudaGetTextureObjectTextureDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETTEXTUREOBJECTRESOURCEVIEWDESC] = std::bind(&TallyServer::handle_cudaGetTextureObjectResourceViewDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDACREATESURFACEOBJECT] = std::bind(&TallyServer::handle_cudaCreateSurfaceObject, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADESTROYSURFACEOBJECT] = std::bind(&TallyServer::handle_cudaDestroySurfaceObject, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETSURFACEOBJECTRESOURCEDESC] = std::bind(&TallyServer::handle_cudaGetSurfaceObjectResourceDesc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADRIVERGETVERSION] = std::bind(&TallyServer::handle_cudaDriverGetVersion, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDARUNTIMEGETVERSION] = std::bind(&TallyServer::handle_cudaRuntimeGetVersion, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHCREATE] = std::bind(&TallyServer::handle_cudaGraphCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDKERNELNODE] = std::bind(&TallyServer::handle_cudaGraphAddKernelNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphKernelNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphKernelNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODECOPYATTRIBUTES] = std::bind(&TallyServer::handle_cudaGraphKernelNodeCopyAttributes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODEGETATTRIBUTE] = std::bind(&TallyServer::handle_cudaGraphKernelNodeGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHKERNELNODESETATTRIBUTE] = std::bind(&TallyServer::handle_cudaGraphKernelNodeSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMCPYNODE] = std::bind(&TallyServer::handle_cudaGraphAddMemcpyNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMCPYNODETOSYMBOL] = std::bind(&TallyServer::handle_cudaGraphAddMemcpyNodeToSymbol, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMCPYNODEFROMSYMBOL] = std::bind(&TallyServer::handle_cudaGraphAddMemcpyNodeFromSymbol, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMCPYNODE1D] = std::bind(&TallyServer::handle_cudaGraphAddMemcpyNode1D, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODESETPARAMSTOSYMBOL] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeSetParamsToSymbol, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODESETPARAMSFROMSYMBOL] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeSetParamsFromSymbol, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMCPYNODESETPARAMS1D] = std::bind(&TallyServer::handle_cudaGraphMemcpyNodeSetParams1D, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMSETNODE] = std::bind(&TallyServer::handle_cudaGraphAddMemsetNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMSETNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemsetNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMSETNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemsetNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDHOSTNODE] = std::bind(&TallyServer::handle_cudaGraphAddHostNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHHOSTNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphHostNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHHOSTNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphHostNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDCHILDGRAPHNODE] = std::bind(&TallyServer::handle_cudaGraphAddChildGraphNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHCHILDGRAPHNODEGETGRAPH] = std::bind(&TallyServer::handle_cudaGraphChildGraphNodeGetGraph, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEMPTYNODE] = std::bind(&TallyServer::handle_cudaGraphAddEmptyNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEVENTRECORDNODE] = std::bind(&TallyServer::handle_cudaGraphAddEventRecordNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEVENTRECORDNODEGETEVENT] = std::bind(&TallyServer::handle_cudaGraphEventRecordNodeGetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEVENTRECORDNODESETEVENT] = std::bind(&TallyServer::handle_cudaGraphEventRecordNodeSetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEVENTWAITNODE] = std::bind(&TallyServer::handle_cudaGraphAddEventWaitNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEVENTWAITNODEGETEVENT] = std::bind(&TallyServer::handle_cudaGraphEventWaitNodeGetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEVENTWAITNODESETEVENT] = std::bind(&TallyServer::handle_cudaGraphEventWaitNodeSetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEXTERNALSEMAPHORESSIGNALNODE] = std::bind(&TallyServer::handle_cudaGraphAddExternalSemaphoresSignalNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXTERNALSEMAPHORESSIGNALNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExternalSemaphoresSignalNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXTERNALSEMAPHORESSIGNALNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExternalSemaphoresSignalNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDEXTERNALSEMAPHORESWAITNODE] = std::bind(&TallyServer::handle_cudaGraphAddExternalSemaphoresWaitNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXTERNALSEMAPHORESWAITNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExternalSemaphoresWaitNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXTERNALSEMAPHORESWAITNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExternalSemaphoresWaitNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMALLOCNODE] = std::bind(&TallyServer::handle_cudaGraphAddMemAllocNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMALLOCNODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemAllocNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDMEMFREENODE] = std::bind(&TallyServer::handle_cudaGraphAddMemFreeNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHMEMFREENODEGETPARAMS] = std::bind(&TallyServer::handle_cudaGraphMemFreeNodeGetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGRAPHMEMTRIM] = std::bind(&TallyServer::handle_cudaDeviceGraphMemTrim, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICEGETGRAPHMEMATTRIBUTE] = std::bind(&TallyServer::handle_cudaDeviceGetGraphMemAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDADEVICESETGRAPHMEMATTRIBUTE] = std::bind(&TallyServer::handle_cudaDeviceSetGraphMemAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHCLONE] = std::bind(&TallyServer::handle_cudaGraphClone, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODEFINDINCLONE] = std::bind(&TallyServer::handle_cudaGraphNodeFindInClone, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODEGETTYPE] = std::bind(&TallyServer::handle_cudaGraphNodeGetType, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHGETNODES] = std::bind(&TallyServer::handle_cudaGraphGetNodes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHGETROOTNODES] = std::bind(&TallyServer::handle_cudaGraphGetRootNodes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHGETEDGES] = std::bind(&TallyServer::handle_cudaGraphGetEdges, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODEGETDEPENDENCIES] = std::bind(&TallyServer::handle_cudaGraphNodeGetDependencies, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHNODEGETDEPENDENTNODES] = std::bind(&TallyServer::handle_cudaGraphNodeGetDependentNodes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHADDDEPENDENCIES] = std::bind(&TallyServer::handle_cudaGraphAddDependencies, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHREMOVEDEPENDENCIES] = std::bind(&TallyServer::handle_cudaGraphRemoveDependencies, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHDESTROYNODE] = std::bind(&TallyServer::handle_cudaGraphDestroyNode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHINSTANTIATE] = std::bind(&TallyServer::handle_cudaGraphInstantiate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHINSTANTIATEWITHFLAGS] = std::bind(&TallyServer::handle_cudaGraphInstantiateWithFlags, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECKERNELNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecKernelNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMCPYNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecMemcpyNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMCPYNODESETPARAMSTOSYMBOL] = std::bind(&TallyServer::handle_cudaGraphExecMemcpyNodeSetParamsToSymbol, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMCPYNODESETPARAMSFROMSYMBOL] = std::bind(&TallyServer::handle_cudaGraphExecMemcpyNodeSetParamsFromSymbol, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMCPYNODESETPARAMS1D] = std::bind(&TallyServer::handle_cudaGraphExecMemcpyNodeSetParams1D, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECMEMSETNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecMemsetNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECHOSTNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecHostNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECCHILDGRAPHNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecChildGraphNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECEVENTRECORDNODESETEVENT] = std::bind(&TallyServer::handle_cudaGraphExecEventRecordNodeSetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECEVENTWAITNODESETEVENT] = std::bind(&TallyServer::handle_cudaGraphExecEventWaitNodeSetEvent, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECEXTERNALSEMAPHORESSIGNALNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecExternalSemaphoresSignalNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECEXTERNALSEMAPHORESWAITNODESETPARAMS] = std::bind(&TallyServer::handle_cudaGraphExecExternalSemaphoresWaitNodeSetParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECUPDATE] = std::bind(&TallyServer::handle_cudaGraphExecUpdate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHUPLOAD] = std::bind(&TallyServer::handle_cudaGraphUpload, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHLAUNCH] = std::bind(&TallyServer::handle_cudaGraphLaunch, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHEXECDESTROY] = std::bind(&TallyServer::handle_cudaGraphExecDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHDESTROY] = std::bind(&TallyServer::handle_cudaGraphDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHDEBUGDOTPRINT] = std::bind(&TallyServer::handle_cudaGraphDebugDotPrint, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAUSEROBJECTCREATE] = std::bind(&TallyServer::handle_cudaUserObjectCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAUSEROBJECTRETAIN] = std::bind(&TallyServer::handle_cudaUserObjectRetain, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAUSEROBJECTRELEASE] = std::bind(&TallyServer::handle_cudaUserObjectRelease, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHRETAINUSEROBJECT] = std::bind(&TallyServer::handle_cudaGraphRetainUserObject, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGRAPHRELEASEUSEROBJECT] = std::bind(&TallyServer::handle_cudaGraphReleaseUserObject, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETDRIVERENTRYPOINT] = std::bind(&TallyServer::handle_cudaGetDriverEntryPoint, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETEXPORTTABLE] = std::bind(&TallyServer::handle_cudaGetExportTable, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAGETFUNCBYSYMBOL] = std::bind(&TallyServer::handle_cudaGetFuncBySymbol, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETVERSION] = std::bind(&TallyServer::handle_cudnnGetVersion, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETMAXDEVICEVERSION] = std::bind(&TallyServer::handle_cudnnGetMaxDeviceVersion, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCUDARTVERSION] = std::bind(&TallyServer::handle_cudnnGetCudartVersion, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETERRORSTRING] = std::bind(&TallyServer::handle_cudnnGetErrorString, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNQUERYRUNTIMEERROR] = std::bind(&TallyServer::handle_cudnnQueryRuntimeError, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPROPERTY] = std::bind(&TallyServer::handle_cudnnGetProperty, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATE] = std::bind(&TallyServer::handle_cudnnCreate, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROY] = std::bind(&TallyServer::handle_cudnnDestroy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETSTREAM] = std::bind(&TallyServer::handle_cudnnSetStream, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETSTREAM] = std::bind(&TallyServer::handle_cudnnGetStream, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETTENSOR4DDESCRIPTOREX] = std::bind(&TallyServer::handle_cudnnSetTensor4dDescriptorEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETTENSOR4DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetTensor4dDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETTENSORNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetTensorNdDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETTENSORNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetTensorNdDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETTENSORSIZEINBYTES] = std::bind(&TallyServer::handle_cudnnGetTensorSizeInBytes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNADDTENSOR] = std::bind(&TallyServer::handle_cudnnAddTensor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEOPTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateOpTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETOPTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetOpTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETOPTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetOpTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYOPTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyOpTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNOPTENSOR] = std::bind(&TallyServer::handle_cudnnOpTensor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEREDUCETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateReduceTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETREDUCETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetReduceTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETREDUCETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetReduceTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYREDUCETENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyReduceTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETREDUCTIONINDICESSIZE] = std::bind(&TallyServer::handle_cudnnGetReductionIndicesSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETREDUCTIONWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetReductionWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNREDUCETENSOR] = std::bind(&TallyServer::handle_cudnnReduceTensor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETTENSOR] = std::bind(&TallyServer::handle_cudnnSetTensor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSCALETENSOR] = std::bind(&TallyServer::handle_cudnnScaleTensor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEFILTERDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateFilterDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETFILTERSIZEINBYTES] = std::bind(&TallyServer::handle_cudnnGetFilterSizeInBytes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYFILTERDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyFilterDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSOFTMAXFORWARD] = std::bind(&TallyServer::handle_cudnnSoftmaxForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEPOOLINGDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreatePoolingDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETPOOLING2DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetPooling2dDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPOOLING2DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetPooling2dDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETPOOLINGNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetPoolingNdDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPOOLINGNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetPoolingNdDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPOOLINGNDFORWARDOUTPUTDIM] = std::bind(&TallyServer::handle_cudnnGetPoolingNdForwardOutputDim, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETPOOLING2DFORWARDOUTPUTDIM] = std::bind(&TallyServer::handle_cudnnGetPooling2dForwardOutputDim, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYPOOLINGDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyPoolingDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNPOOLINGFORWARD] = std::bind(&TallyServer::handle_cudnnPoolingForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEACTIVATIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateActivationDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETACTIVATIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetActivationDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETACTIVATIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetActivationDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETACTIVATIONDESCRIPTORSWISHBETA] = std::bind(&TallyServer::handle_cudnnSetActivationDescriptorSwishBeta, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETACTIVATIONDESCRIPTORSWISHBETA] = std::bind(&TallyServer::handle_cudnnGetActivationDescriptorSwishBeta, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYACTIVATIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyActivationDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNACTIVATIONFORWARD] = std::bind(&TallyServer::handle_cudnnActivationForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATELRNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateLRNDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETLRNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetLRNDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETLRNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetLRNDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYLRNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyLRNDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNLRNCROSSCHANNELFORWARD] = std::bind(&TallyServer::handle_cudnnLRNCrossChannelForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDIVISIVENORMALIZATIONFORWARD] = std::bind(&TallyServer::handle_cudnnDivisiveNormalizationForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDERIVEBNTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDeriveBNTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONFORWARDINFERENCE] = std::bind(&TallyServer::handle_cudnnBatchNormalizationForwardInference, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDERIVENORMTENSORDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDeriveNormTensorDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNNORMALIZATIONFORWARDINFERENCE] = std::bind(&TallyServer::handle_cudnnNormalizationForwardInference, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateDropoutDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyDropoutDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDROPOUTGETSTATESSIZE] = std::bind(&TallyServer::handle_cudnnDropoutGetStatesSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDROPOUTGETRESERVESPACESIZE] = std::bind(&TallyServer::handle_cudnnDropoutGetReserveSpaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetDropoutDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRESTOREDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnRestoreDropoutDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETDROPOUTDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetDropoutDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDROPOUTFORWARD] = std::bind(&TallyServer::handle_cudnnDropoutForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateAlgorithmDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetAlgorithmDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetAlgorithmDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCOPYALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCopyAlgorithmDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyAlgorithmDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETALGORITHMSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetAlgorithmSpaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSAVEALGORITHM] = std::bind(&TallyServer::handle_cudnnSaveAlgorithm, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRESTOREALGORITHM] = std::bind(&TallyServer::handle_cudnnRestoreAlgorithm, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCALLBACK] = std::bind(&TallyServer::handle_cudnnSetCallback, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCALLBACK] = std::bind(&TallyServer::handle_cudnnGetCallback, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNOPSINFERVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnOpsInferVersionCheck, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNPOOLINGBACKWARD] = std::bind(&TallyServer::handle_cudnnPoolingBackward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNACTIVATIONBACKWARD] = std::bind(&TallyServer::handle_cudnnActivationBackward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNLRNCROSSCHANNELBACKWARD] = std::bind(&TallyServer::handle_cudnnLRNCrossChannelBackward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDIVISIVENORMALIZATIONBACKWARD] = std::bind(&TallyServer::handle_cudnnDivisiveNormalizationBackward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETBATCHNORMALIZATIONFORWARDTRAININGEXWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETBATCHNORMALIZATIONBACKWARDEXWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetBatchNormalizationBackwardExWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETBATCHNORMALIZATIONTRAININGEXRESERVESPACESIZE] = std::bind(&TallyServer::handle_cudnnGetBatchNormalizationTrainingExReserveSpaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONFORWARDTRAINING] = std::bind(&TallyServer::handle_cudnnBatchNormalizationForwardTraining, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONFORWARDTRAININGEX] = std::bind(&TallyServer::handle_cudnnBatchNormalizationForwardTrainingEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONBACKWARD] = std::bind(&TallyServer::handle_cudnnBatchNormalizationBackward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBATCHNORMALIZATIONBACKWARDEX] = std::bind(&TallyServer::handle_cudnnBatchNormalizationBackwardEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETNORMALIZATIONFORWARDTRAININGWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetNormalizationForwardTrainingWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETNORMALIZATIONBACKWARDWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetNormalizationBackwardWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETNORMALIZATIONTRAININGRESERVESPACESIZE] = std::bind(&TallyServer::handle_cudnnGetNormalizationTrainingReserveSpaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNNORMALIZATIONFORWARDTRAINING] = std::bind(&TallyServer::handle_cudnnNormalizationForwardTraining, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNNORMALIZATIONBACKWARD] = std::bind(&TallyServer::handle_cudnnNormalizationBackward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDROPOUTBACKWARD] = std::bind(&TallyServer::handle_cudnnDropoutBackward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNOPSTRAINVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnOpsTrainVersionCheck, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATERNNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateRNNDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYRNNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyRNNDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNDESCRIPTOR_V8] = std::bind(&TallyServer::handle_cudnnSetRNNDescriptor_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNDESCRIPTOR_V8] = std::bind(&TallyServer::handle_cudnnGetRNNDescriptor_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNDESCRIPTOR_V6] = std::bind(&TallyServer::handle_cudnnSetRNNDescriptor_v6, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNDESCRIPTOR_V6] = std::bind(&TallyServer::handle_cudnnGetRNNDescriptor_v6, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNMATRIXMATHTYPE] = std::bind(&TallyServer::handle_cudnnSetRNNMatrixMathType, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNMATRIXMATHTYPE] = std::bind(&TallyServer::handle_cudnnGetRNNMatrixMathType, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNBIASMODE] = std::bind(&TallyServer::handle_cudnnSetRNNBiasMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNBIASMODE] = std::bind(&TallyServer::handle_cudnnGetRNNBiasMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNSETCLIP_V8] = std::bind(&TallyServer::handle_cudnnRNNSetClip_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNGETCLIP_V8] = std::bind(&TallyServer::handle_cudnnRNNGetClip_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNSETCLIP] = std::bind(&TallyServer::handle_cudnnRNNSetClip, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNGETCLIP] = std::bind(&TallyServer::handle_cudnnRNNGetClip, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNPROJECTIONLAYERS] = std::bind(&TallyServer::handle_cudnnSetRNNProjectionLayers, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNPROJECTIONLAYERS] = std::bind(&TallyServer::handle_cudnnGetRNNProjectionLayers, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEPERSISTENTRNNPLAN] = std::bind(&TallyServer::handle_cudnnCreatePersistentRNNPlan, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYPERSISTENTRNNPLAN] = std::bind(&TallyServer::handle_cudnnDestroyPersistentRNNPlan, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETPERSISTENTRNNPLAN] = std::bind(&TallyServer::handle_cudnnSetPersistentRNNPlan, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBUILDRNNDYNAMIC] = std::bind(&TallyServer::handle_cudnnBuildRNNDynamic, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetRNNWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNTRAININGRESERVESIZE] = std::bind(&TallyServer::handle_cudnnGetRNNTrainingReserveSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNTEMPSPACESIZES] = std::bind(&TallyServer::handle_cudnnGetRNNTempSpaceSizes, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNPARAMSSIZE] = std::bind(&TallyServer::handle_cudnnGetRNNParamsSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNWEIGHTSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetRNNWeightSpaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNLINLAYERMATRIXPARAMS] = std::bind(&TallyServer::handle_cudnnGetRNNLinLayerMatrixParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNLINLAYERBIASPARAMS] = std::bind(&TallyServer::handle_cudnnGetRNNLinLayerBiasParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNWEIGHTPARAMS] = std::bind(&TallyServer::handle_cudnnGetRNNWeightParams, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARDINFERENCE] = std::bind(&TallyServer::handle_cudnnRNNForwardInference, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNPADDINGMODE] = std::bind(&TallyServer::handle_cudnnSetRNNPaddingMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNPADDINGMODE] = std::bind(&TallyServer::handle_cudnnGetRNNPaddingMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATERNNDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateRNNDataDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYRNNDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyRNNDataDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetRNNDataDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetRNNDataDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARDINFERENCEEX] = std::bind(&TallyServer::handle_cudnnRNNForwardInferenceEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARD] = std::bind(&TallyServer::handle_cudnnRNNForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETRNNALGORITHMDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetRNNAlgorithmDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNFORWARDINFERENCEALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetRNNForwardInferenceAlgorithmMaxCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATESEQDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateSeqDataDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYSEQDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroySeqDataDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETSEQDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetSeqDataDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETSEQDATADESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetSeqDataDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEATTNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateAttnDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYATTNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyAttnDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETATTNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetAttnDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETATTNDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetAttnDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETMULTIHEADATTNBUFFERS] = std::bind(&TallyServer::handle_cudnnGetMultiHeadAttnBuffers, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETMULTIHEADATTNWEIGHTS] = std::bind(&TallyServer::handle_cudnnGetMultiHeadAttnWeights, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNMULTIHEADATTNFORWARD] = std::bind(&TallyServer::handle_cudnnMultiHeadAttnForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNADVINFERVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnAdvInferVersionCheck, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARDTRAINING] = std::bind(&TallyServer::handle_cudnnRNNForwardTraining, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDDATA] = std::bind(&TallyServer::handle_cudnnRNNBackwardData, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDDATA_V8] = std::bind(&TallyServer::handle_cudnnRNNBackwardData_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTS] = std::bind(&TallyServer::handle_cudnnRNNBackwardWeights, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTS_V8] = std::bind(&TallyServer::handle_cudnnRNNBackwardWeights_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNFORWARDTRAININGEX] = std::bind(&TallyServer::handle_cudnnRNNForwardTrainingEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDDATAEX] = std::bind(&TallyServer::handle_cudnnRNNBackwardDataEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTSEX] = std::bind(&TallyServer::handle_cudnnRNNBackwardWeightsEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNFORWARDTRAININGALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetRNNForwardTrainingAlgorithmMaxCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNBACKWARDDATAALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetRNNBackwardDataAlgorithmMaxCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETRNNBACKWARDWEIGHTSALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetRNNBackwardWeightsAlgorithmMaxCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDDATA] = std::bind(&TallyServer::handle_cudnnMultiHeadAttnBackwardData, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNMULTIHEADATTNBACKWARDWEIGHTS] = std::bind(&TallyServer::handle_cudnnMultiHeadAttnBackwardWeights, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATECTCLOSSDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateCTCLossDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCTCLOSSDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetCTCLossDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCTCLOSSDESCRIPTOREX] = std::bind(&TallyServer::handle_cudnnSetCTCLossDescriptorEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCTCLOSSDESCRIPTOR_V8] = std::bind(&TallyServer::handle_cudnnSetCTCLossDescriptor_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetCTCLossDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSDESCRIPTOREX] = std::bind(&TallyServer::handle_cudnnGetCTCLossDescriptorEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSDESCRIPTOR_V8] = std::bind(&TallyServer::handle_cudnnGetCTCLossDescriptor_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYCTCLOSSDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyCTCLossDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCTCLOSS] = std::bind(&TallyServer::handle_cudnnCTCLoss, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCTCLOSS_V8] = std::bind(&TallyServer::handle_cudnnCTCLoss_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetCTCLossWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCTCLOSSWORKSPACESIZE_V8] = std::bind(&TallyServer::handle_cudnnGetCTCLossWorkspaceSize_v8, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNADVTRAINVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnAdvTrainVersionCheck, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATECONVOLUTIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnCreateConvolutionDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYCONVOLUTIONDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnDestroyConvolutionDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTIONMATHTYPE] = std::bind(&TallyServer::handle_cudnnSetConvolutionMathType, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONMATHTYPE] = std::bind(&TallyServer::handle_cudnnGetConvolutionMathType, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTIONGROUPCOUNT] = std::bind(&TallyServer::handle_cudnnSetConvolutionGroupCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONGROUPCOUNT] = std::bind(&TallyServer::handle_cudnnGetConvolutionGroupCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTIONREORDERTYPE] = std::bind(&TallyServer::handle_cudnnSetConvolutionReorderType, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONREORDERTYPE] = std::bind(&TallyServer::handle_cudnnGetConvolutionReorderType, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTION2DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetConvolution2dDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTION2DDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetConvolution2dDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETCONVOLUTIONNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnSetConvolutionNdDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONNDDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnGetConvolutionNdDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTION2DFORWARDOUTPUTDIM] = std::bind(&TallyServer::handle_cudnnGetConvolution2dForwardOutputDim, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONNDFORWARDOUTPUTDIM] = std::bind(&TallyServer::handle_cudnnGetConvolutionNdForwardOutputDim, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetConvolutionForwardAlgorithmMaxCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNIM2COL] = std::bind(&TallyServer::handle_cudnnIm2Col, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNREORDERFILTERANDBIAS] = std::bind(&TallyServer::handle_cudnnReorderFilterAndBias, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetConvolutionForwardWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONFORWARD] = std::bind(&TallyServer::handle_cudnnConvolutionForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONBIASACTIVATIONFORWARD] = std::bind(&TallyServer::handle_cudnnConvolutionBiasActivationForward, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDDATAALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardDataAlgorithmMaxCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDDATAWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardDataWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONBACKWARDDATA] = std::bind(&TallyServer::handle_cudnnConvolutionBackwardData, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCNNINFERVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnCnnInferVersionCheck, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDFILTERALGORITHMMAXCOUNT] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDFILTERWORKSPACESIZE] = std::bind(&TallyServer::handle_cudnnGetConvolutionBackwardFilterWorkspaceSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONBACKWARDFILTER] = std::bind(&TallyServer::handle_cudnnConvolutionBackwardFilter, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCONVOLUTIONBACKWARDBIAS] = std::bind(&TallyServer::handle_cudnnConvolutionBackwardBias, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEFUSEDOPSCONSTPARAMPACK] = std::bind(&TallyServer::handle_cudnnCreateFusedOpsConstParamPack, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYFUSEDOPSCONSTPARAMPACK] = std::bind(&TallyServer::handle_cudnnDestroyFusedOpsConstParamPack, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETFUSEDOPSCONSTPARAMPACKATTRIBUTE] = std::bind(&TallyServer::handle_cudnnSetFusedOpsConstParamPackAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETFUSEDOPSCONSTPARAMPACKATTRIBUTE] = std::bind(&TallyServer::handle_cudnnGetFusedOpsConstParamPackAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEFUSEDOPSVARIANTPARAMPACK] = std::bind(&TallyServer::handle_cudnnCreateFusedOpsVariantParamPack, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYFUSEDOPSVARIANTPARAMPACK] = std::bind(&TallyServer::handle_cudnnDestroyFusedOpsVariantParamPack, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNSETFUSEDOPSVARIANTPARAMPACKATTRIBUTE] = std::bind(&TallyServer::handle_cudnnSetFusedOpsVariantParamPackAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNGETFUSEDOPSVARIANTPARAMPACKATTRIBUTE] = std::bind(&TallyServer::handle_cudnnGetFusedOpsVariantParamPackAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCREATEFUSEDOPSPLAN] = std::bind(&TallyServer::handle_cudnnCreateFusedOpsPlan, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNDESTROYFUSEDOPSPLAN] = std::bind(&TallyServer::handle_cudnnDestroyFusedOpsPlan, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNMAKEFUSEDOPSPLAN] = std::bind(&TallyServer::handle_cudnnMakeFusedOpsPlan, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNFUSEDOPSEXECUTE] = std::bind(&TallyServer::handle_cudnnFusedOpsExecute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNCNNTRAINVERSIONCHECK] = std::bind(&TallyServer::handle_cudnnCnnTrainVersionCheck, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDCREATEDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnBackendCreateDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDDESTROYDESCRIPTOR] = std::bind(&TallyServer::handle_cudnnBackendDestroyDescriptor, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDINITIALIZE] = std::bind(&TallyServer::handle_cudnnBackendInitialize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDFINALIZE] = std::bind(&TallyServer::handle_cudnnBackendFinalize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDSETATTRIBUTE] = std::bind(&TallyServer::handle_cudnnBackendSetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDGETATTRIBUTE] = std::bind(&TallyServer::handle_cudnnBackendGetAttribute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDNNBACKENDEXECUTE] = std::bind(&TallyServer::handle_cudnnBackendExecute, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCREATE_V2] = std::bind(&TallyServer::handle_cublasCreate_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDESTROY_V2] = std::bind(&TallyServer::handle_cublasDestroy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETVERSION_V2] = std::bind(&TallyServer::handle_cublasGetVersion_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETPROPERTY] = std::bind(&TallyServer::handle_cublasGetProperty, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETCUDARTVERSION] = std::bind(&TallyServer::handle_cublasGetCudartVersion, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETWORKSPACE_V2] = std::bind(&TallyServer::handle_cublasSetWorkspace_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETSTREAM_V2] = std::bind(&TallyServer::handle_cublasSetStream_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETSTREAM_V2] = std::bind(&TallyServer::handle_cublasGetStream_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETPOINTERMODE_V2] = std::bind(&TallyServer::handle_cublasGetPointerMode_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETPOINTERMODE_V2] = std::bind(&TallyServer::handle_cublasSetPointerMode_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETATOMICSMODE] = std::bind(&TallyServer::handle_cublasGetAtomicsMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETATOMICSMODE] = std::bind(&TallyServer::handle_cublasSetAtomicsMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETMATHMODE] = std::bind(&TallyServer::handle_cublasGetMathMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETMATHMODE] = std::bind(&TallyServer::handle_cublasSetMathMode, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETSMCOUNTTARGET] = std::bind(&TallyServer::handle_cublasGetSmCountTarget, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETSMCOUNTTARGET] = std::bind(&TallyServer::handle_cublasSetSmCountTarget, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETSTATUSNAME] = std::bind(&TallyServer::handle_cublasGetStatusName, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETSTATUSSTRING] = std::bind(&TallyServer::handle_cublasGetStatusString, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASLOGGERCONFIGURE] = std::bind(&TallyServer::handle_cublasLoggerConfigure, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETLOGGERCALLBACK] = std::bind(&TallyServer::handle_cublasSetLoggerCallback, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETLOGGERCALLBACK] = std::bind(&TallyServer::handle_cublasGetLoggerCallback, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETVECTOR] = std::bind(&TallyServer::handle_cublasSetVector, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETVECTOR] = std::bind(&TallyServer::handle_cublasGetVector, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETMATRIX] = std::bind(&TallyServer::handle_cublasSetMatrix, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETMATRIX] = std::bind(&TallyServer::handle_cublasGetMatrix, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETVECTORASYNC] = std::bind(&TallyServer::handle_cublasSetVectorAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETVECTORASYNC] = std::bind(&TallyServer::handle_cublasGetVectorAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSETMATRIXASYNC] = std::bind(&TallyServer::handle_cublasSetMatrixAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGETMATRIXASYNC] = std::bind(&TallyServer::handle_cublasGetMatrixAsync, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASXERBLA] = std::bind(&TallyServer::handle_cublasXerbla, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASNRM2EX] = std::bind(&TallyServer::handle_cublasNrm2Ex, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSNRM2_V2] = std::bind(&TallyServer::handle_cublasSnrm2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDNRM2_V2] = std::bind(&TallyServer::handle_cublasDnrm2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCNRM2_V2] = std::bind(&TallyServer::handle_cublasScnrm2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDZNRM2_V2] = std::bind(&TallyServer::handle_cublasDznrm2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDOTEX] = std::bind(&TallyServer::handle_cublasDotEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDOTCEX] = std::bind(&TallyServer::handle_cublasDotcEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSDOT_V2] = std::bind(&TallyServer::handle_cublasSdot_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDDOT_V2] = std::bind(&TallyServer::handle_cublasDdot_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCDOTU_V2] = std::bind(&TallyServer::handle_cublasCdotu_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCDOTC_V2] = std::bind(&TallyServer::handle_cublasCdotc_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDOTU_V2] = std::bind(&TallyServer::handle_cublasZdotu_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDOTC_V2] = std::bind(&TallyServer::handle_cublasZdotc_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCALEX] = std::bind(&TallyServer::handle_cublasScalEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSCAL_V2] = std::bind(&TallyServer::handle_cublasSscal_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSCAL_V2] = std::bind(&TallyServer::handle_cublasDscal_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSCAL_V2] = std::bind(&TallyServer::handle_cublasCscal_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSSCAL_V2] = std::bind(&TallyServer::handle_cublasCsscal_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSCAL_V2] = std::bind(&TallyServer::handle_cublasZscal_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDSCAL_V2] = std::bind(&TallyServer::handle_cublasZdscal_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASAXPYEX] = std::bind(&TallyServer::handle_cublasAxpyEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSAXPY_V2] = std::bind(&TallyServer::handle_cublasSaxpy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDAXPY_V2] = std::bind(&TallyServer::handle_cublasDaxpy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCAXPY_V2] = std::bind(&TallyServer::handle_cublasCaxpy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZAXPY_V2] = std::bind(&TallyServer::handle_cublasZaxpy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCOPYEX] = std::bind(&TallyServer::handle_cublasCopyEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCOPY_V2] = std::bind(&TallyServer::handle_cublasScopy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDCOPY_V2] = std::bind(&TallyServer::handle_cublasDcopy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCCOPY_V2] = std::bind(&TallyServer::handle_cublasCcopy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZCOPY_V2] = std::bind(&TallyServer::handle_cublasZcopy_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSWAP_V2] = std::bind(&TallyServer::handle_cublasSswap_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSWAP_V2] = std::bind(&TallyServer::handle_cublasDswap_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSWAP_V2] = std::bind(&TallyServer::handle_cublasCswap_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSWAP_V2] = std::bind(&TallyServer::handle_cublasZswap_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSWAPEX] = std::bind(&TallyServer::handle_cublasSwapEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASISAMAX_V2] = std::bind(&TallyServer::handle_cublasIsamax_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIDAMAX_V2] = std::bind(&TallyServer::handle_cublasIdamax_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASICAMAX_V2] = std::bind(&TallyServer::handle_cublasIcamax_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIZAMAX_V2] = std::bind(&TallyServer::handle_cublasIzamax_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIAMAXEX] = std::bind(&TallyServer::handle_cublasIamaxEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASISAMIN_V2] = std::bind(&TallyServer::handle_cublasIsamin_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIDAMIN_V2] = std::bind(&TallyServer::handle_cublasIdamin_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASICAMIN_V2] = std::bind(&TallyServer::handle_cublasIcamin_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIZAMIN_V2] = std::bind(&TallyServer::handle_cublasIzamin_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASIAMINEX] = std::bind(&TallyServer::handle_cublasIaminEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASASUMEX] = std::bind(&TallyServer::handle_cublasAsumEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSASUM_V2] = std::bind(&TallyServer::handle_cublasSasum_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDASUM_V2] = std::bind(&TallyServer::handle_cublasDasum_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSCASUM_V2] = std::bind(&TallyServer::handle_cublasScasum_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDZASUM_V2] = std::bind(&TallyServer::handle_cublasDzasum_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROT_V2] = std::bind(&TallyServer::handle_cublasSrot_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROT_V2] = std::bind(&TallyServer::handle_cublasDrot_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCROT_V2] = std::bind(&TallyServer::handle_cublasCrot_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSROT_V2] = std::bind(&TallyServer::handle_cublasCsrot_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZROT_V2] = std::bind(&TallyServer::handle_cublasZrot_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDROT_V2] = std::bind(&TallyServer::handle_cublasZdrot_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTEX] = std::bind(&TallyServer::handle_cublasRotEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROTG_V2] = std::bind(&TallyServer::handle_cublasSrotg_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROTG_V2] = std::bind(&TallyServer::handle_cublasDrotg_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCROTG_V2] = std::bind(&TallyServer::handle_cublasCrotg_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZROTG_V2] = std::bind(&TallyServer::handle_cublasZrotg_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTGEX] = std::bind(&TallyServer::handle_cublasRotgEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROTM_V2] = std::bind(&TallyServer::handle_cublasSrotm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROTM_V2] = std::bind(&TallyServer::handle_cublasDrotm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTMEX] = std::bind(&TallyServer::handle_cublasRotmEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSROTMG_V2] = std::bind(&TallyServer::handle_cublasSrotmg_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDROTMG_V2] = std::bind(&TallyServer::handle_cublasDrotmg_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASROTMGEX] = std::bind(&TallyServer::handle_cublasRotmgEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMV_V2] = std::bind(&TallyServer::handle_cublasSgemv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMV_V2] = std::bind(&TallyServer::handle_cublasDgemv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMV_V2] = std::bind(&TallyServer::handle_cublasCgemv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMV_V2] = std::bind(&TallyServer::handle_cublasZgemv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGBMV_V2] = std::bind(&TallyServer::handle_cublasSgbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGBMV_V2] = std::bind(&TallyServer::handle_cublasDgbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGBMV_V2] = std::bind(&TallyServer::handle_cublasCgbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGBMV_V2] = std::bind(&TallyServer::handle_cublasZgbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRMV_V2] = std::bind(&TallyServer::handle_cublasStrmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRMV_V2] = std::bind(&TallyServer::handle_cublasDtrmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRMV_V2] = std::bind(&TallyServer::handle_cublasCtrmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRMV_V2] = std::bind(&TallyServer::handle_cublasZtrmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTBMV_V2] = std::bind(&TallyServer::handle_cublasStbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTBMV_V2] = std::bind(&TallyServer::handle_cublasDtbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTBMV_V2] = std::bind(&TallyServer::handle_cublasCtbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTBMV_V2] = std::bind(&TallyServer::handle_cublasZtbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTPMV_V2] = std::bind(&TallyServer::handle_cublasStpmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTPMV_V2] = std::bind(&TallyServer::handle_cublasDtpmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTPMV_V2] = std::bind(&TallyServer::handle_cublasCtpmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTPMV_V2] = std::bind(&TallyServer::handle_cublasZtpmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRSV_V2] = std::bind(&TallyServer::handle_cublasStrsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRSV_V2] = std::bind(&TallyServer::handle_cublasDtrsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRSV_V2] = std::bind(&TallyServer::handle_cublasCtrsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRSV_V2] = std::bind(&TallyServer::handle_cublasZtrsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTPSV_V2] = std::bind(&TallyServer::handle_cublasStpsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTPSV_V2] = std::bind(&TallyServer::handle_cublasDtpsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTPSV_V2] = std::bind(&TallyServer::handle_cublasCtpsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTPSV_V2] = std::bind(&TallyServer::handle_cublasZtpsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTBSV_V2] = std::bind(&TallyServer::handle_cublasStbsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTBSV_V2] = std::bind(&TallyServer::handle_cublasDtbsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTBSV_V2] = std::bind(&TallyServer::handle_cublasCtbsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTBSV_V2] = std::bind(&TallyServer::handle_cublasZtbsv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYMV_V2] = std::bind(&TallyServer::handle_cublasSsymv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYMV_V2] = std::bind(&TallyServer::handle_cublasDsymv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYMV_V2] = std::bind(&TallyServer::handle_cublasCsymv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYMV_V2] = std::bind(&TallyServer::handle_cublasZsymv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHEMV_V2] = std::bind(&TallyServer::handle_cublasChemv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHEMV_V2] = std::bind(&TallyServer::handle_cublasZhemv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSBMV_V2] = std::bind(&TallyServer::handle_cublasSsbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSBMV_V2] = std::bind(&TallyServer::handle_cublasDsbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHBMV_V2] = std::bind(&TallyServer::handle_cublasChbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHBMV_V2] = std::bind(&TallyServer::handle_cublasZhbmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSPMV_V2] = std::bind(&TallyServer::handle_cublasSspmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSPMV_V2] = std::bind(&TallyServer::handle_cublasDspmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHPMV_V2] = std::bind(&TallyServer::handle_cublasChpmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHPMV_V2] = std::bind(&TallyServer::handle_cublasZhpmv_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGER_V2] = std::bind(&TallyServer::handle_cublasSger_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGER_V2] = std::bind(&TallyServer::handle_cublasDger_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGERU_V2] = std::bind(&TallyServer::handle_cublasCgeru_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGERC_V2] = std::bind(&TallyServer::handle_cublasCgerc_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGERU_V2] = std::bind(&TallyServer::handle_cublasZgeru_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGERC_V2] = std::bind(&TallyServer::handle_cublasZgerc_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYR_V2] = std::bind(&TallyServer::handle_cublasSsyr_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYR_V2] = std::bind(&TallyServer::handle_cublasDsyr_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYR_V2] = std::bind(&TallyServer::handle_cublasCsyr_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYR_V2] = std::bind(&TallyServer::handle_cublasZsyr_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHER_V2] = std::bind(&TallyServer::handle_cublasCher_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHER_V2] = std::bind(&TallyServer::handle_cublasZher_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSPR_V2] = std::bind(&TallyServer::handle_cublasSspr_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSPR_V2] = std::bind(&TallyServer::handle_cublasDspr_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHPR_V2] = std::bind(&TallyServer::handle_cublasChpr_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHPR_V2] = std::bind(&TallyServer::handle_cublasZhpr_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYR2_V2] = std::bind(&TallyServer::handle_cublasSsyr2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYR2_V2] = std::bind(&TallyServer::handle_cublasDsyr2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYR2_V2] = std::bind(&TallyServer::handle_cublasCsyr2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYR2_V2] = std::bind(&TallyServer::handle_cublasZsyr2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHER2_V2] = std::bind(&TallyServer::handle_cublasCher2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHER2_V2] = std::bind(&TallyServer::handle_cublasZher2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSPR2_V2] = std::bind(&TallyServer::handle_cublasSspr2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSPR2_V2] = std::bind(&TallyServer::handle_cublasDspr2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHPR2_V2] = std::bind(&TallyServer::handle_cublasChpr2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHPR2_V2] = std::bind(&TallyServer::handle_cublasZhpr2_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMM_V2] = std::bind(&TallyServer::handle_cublasSgemm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMM_V2] = std::bind(&TallyServer::handle_cublasDgemm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM_V2] = std::bind(&TallyServer::handle_cublasCgemm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3M] = std::bind(&TallyServer::handle_cublasCgemm3m, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3MEX] = std::bind(&TallyServer::handle_cublasCgemm3mEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMM_V2] = std::bind(&TallyServer::handle_cublasZgemm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMM3M] = std::bind(&TallyServer::handle_cublasZgemm3m, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHGEMM] = std::bind(&TallyServer::handle_cublasHgemm, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMMEX] = std::bind(&TallyServer::handle_cublasSgemmEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGEMMEX] = std::bind(&TallyServer::handle_cublasGemmEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMMEX] = std::bind(&TallyServer::handle_cublasCgemmEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASUINT8GEMMBIAS] = std::bind(&TallyServer::handle_cublasUint8gemmBias, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYRK_V2] = std::bind(&TallyServer::handle_cublasSsyrk_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYRK_V2] = std::bind(&TallyServer::handle_cublasDsyrk_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRK_V2] = std::bind(&TallyServer::handle_cublasCsyrk_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYRK_V2] = std::bind(&TallyServer::handle_cublasZsyrk_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRKEX] = std::bind(&TallyServer::handle_cublasCsyrkEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRK3MEX] = std::bind(&TallyServer::handle_cublasCsyrk3mEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERK_V2] = std::bind(&TallyServer::handle_cublasCherk_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHERK_V2] = std::bind(&TallyServer::handle_cublasZherk_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERKEX] = std::bind(&TallyServer::handle_cublasCherkEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERK3MEX] = std::bind(&TallyServer::handle_cublasCherk3mEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYR2K_V2] = std::bind(&TallyServer::handle_cublasSsyr2k_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYR2K_V2] = std::bind(&TallyServer::handle_cublasDsyr2k_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYR2K_V2] = std::bind(&TallyServer::handle_cublasCsyr2k_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYR2K_V2] = std::bind(&TallyServer::handle_cublasZsyr2k_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHER2K_V2] = std::bind(&TallyServer::handle_cublasCher2k_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHER2K_V2] = std::bind(&TallyServer::handle_cublasZher2k_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYRKX] = std::bind(&TallyServer::handle_cublasSsyrkx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYRKX] = std::bind(&TallyServer::handle_cublasDsyrkx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYRKX] = std::bind(&TallyServer::handle_cublasCsyrkx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYRKX] = std::bind(&TallyServer::handle_cublasZsyrkx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHERKX] = std::bind(&TallyServer::handle_cublasCherkx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHERKX] = std::bind(&TallyServer::handle_cublasZherkx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSSYMM_V2] = std::bind(&TallyServer::handle_cublasSsymm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDSYMM_V2] = std::bind(&TallyServer::handle_cublasDsymm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCSYMM_V2] = std::bind(&TallyServer::handle_cublasCsymm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZSYMM_V2] = std::bind(&TallyServer::handle_cublasZsymm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCHEMM_V2] = std::bind(&TallyServer::handle_cublasChemm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZHEMM_V2] = std::bind(&TallyServer::handle_cublasZhemm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRSM_V2] = std::bind(&TallyServer::handle_cublasStrsm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRSM_V2] = std::bind(&TallyServer::handle_cublasDtrsm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRSM_V2] = std::bind(&TallyServer::handle_cublasCtrsm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRSM_V2] = std::bind(&TallyServer::handle_cublasZtrsm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRMM_V2] = std::bind(&TallyServer::handle_cublasStrmm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRMM_V2] = std::bind(&TallyServer::handle_cublasDtrmm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRMM_V2] = std::bind(&TallyServer::handle_cublasCtrmm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRMM_V2] = std::bind(&TallyServer::handle_cublasZtrmm_v2, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHGEMMBATCHED] = std::bind(&TallyServer::handle_cublasHgemmBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMMBATCHED] = std::bind(&TallyServer::handle_cublasSgemmBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMMBATCHED] = std::bind(&TallyServer::handle_cublasDgemmBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMMBATCHED] = std::bind(&TallyServer::handle_cublasCgemmBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3MBATCHED] = std::bind(&TallyServer::handle_cublasCgemm3mBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMMBATCHED] = std::bind(&TallyServer::handle_cublasZgemmBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGEMMBATCHEDEX] = std::bind(&TallyServer::handle_cublasGemmBatchedEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASGEMMSTRIDEDBATCHEDEX] = std::bind(&TallyServer::handle_cublasGemmStridedBatchedEx, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasSgemmStridedBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasDgemmStridedBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasCgemmStridedBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEMM3MSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasCgemm3mStridedBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasZgemmStridedBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASHGEMMSTRIDEDBATCHED] = std::bind(&TallyServer::handle_cublasHgemmStridedBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEAM] = std::bind(&TallyServer::handle_cublasSgeam, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEAM] = std::bind(&TallyServer::handle_cublasDgeam, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEAM] = std::bind(&TallyServer::handle_cublasCgeam, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEAM] = std::bind(&TallyServer::handle_cublasZgeam, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGETRFBATCHED] = std::bind(&TallyServer::handle_cublasSgetrfBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGETRFBATCHED] = std::bind(&TallyServer::handle_cublasDgetrfBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGETRFBATCHED] = std::bind(&TallyServer::handle_cublasCgetrfBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGETRFBATCHED] = std::bind(&TallyServer::handle_cublasZgetrfBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGETRIBATCHED] = std::bind(&TallyServer::handle_cublasSgetriBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGETRIBATCHED] = std::bind(&TallyServer::handle_cublasDgetriBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGETRIBATCHED] = std::bind(&TallyServer::handle_cublasCgetriBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGETRIBATCHED] = std::bind(&TallyServer::handle_cublasZgetriBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGETRSBATCHED] = std::bind(&TallyServer::handle_cublasSgetrsBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGETRSBATCHED] = std::bind(&TallyServer::handle_cublasDgetrsBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGETRSBATCHED] = std::bind(&TallyServer::handle_cublasCgetrsBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGETRSBATCHED] = std::bind(&TallyServer::handle_cublasZgetrsBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRSMBATCHED] = std::bind(&TallyServer::handle_cublasStrsmBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRSMBATCHED] = std::bind(&TallyServer::handle_cublasDtrsmBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRSMBATCHED] = std::bind(&TallyServer::handle_cublasCtrsmBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRSMBATCHED] = std::bind(&TallyServer::handle_cublasZtrsmBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSMATINVBATCHED] = std::bind(&TallyServer::handle_cublasSmatinvBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDMATINVBATCHED] = std::bind(&TallyServer::handle_cublasDmatinvBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCMATINVBATCHED] = std::bind(&TallyServer::handle_cublasCmatinvBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZMATINVBATCHED] = std::bind(&TallyServer::handle_cublasZmatinvBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGEQRFBATCHED] = std::bind(&TallyServer::handle_cublasSgeqrfBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGEQRFBATCHED] = std::bind(&TallyServer::handle_cublasDgeqrfBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGEQRFBATCHED] = std::bind(&TallyServer::handle_cublasCgeqrfBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGEQRFBATCHED] = std::bind(&TallyServer::handle_cublasZgeqrfBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSGELSBATCHED] = std::bind(&TallyServer::handle_cublasSgelsBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDGELSBATCHED] = std::bind(&TallyServer::handle_cublasDgelsBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCGELSBATCHED] = std::bind(&TallyServer::handle_cublasCgelsBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZGELSBATCHED] = std::bind(&TallyServer::handle_cublasZgelsBatched, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSDGMM] = std::bind(&TallyServer::handle_cublasSdgmm, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDDGMM] = std::bind(&TallyServer::handle_cublasDdgmm, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCDGMM] = std::bind(&TallyServer::handle_cublasCdgmm, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZDGMM] = std::bind(&TallyServer::handle_cublasZdgmm, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTPTTR] = std::bind(&TallyServer::handle_cublasStpttr, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTPTTR] = std::bind(&TallyServer::handle_cublasDtpttr, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTPTTR] = std::bind(&TallyServer::handle_cublasCtpttr, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTPTTR] = std::bind(&TallyServer::handle_cublasZtpttr, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASSTRTTP] = std::bind(&TallyServer::handle_cublasStrttp, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASDTRTTP] = std::bind(&TallyServer::handle_cublasDtrttp, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASCTRTTP] = std::bind(&TallyServer::handle_cublasCtrttp, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUBLASZTRTTP] = std::bind(&TallyServer::handle_cublasZtrttp, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAPROFILERINITIALIZE] = std::bind(&TallyServer::handle_cudaProfilerInitialize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAPROFILERSTART] = std::bind(&TallyServer::handle_cudaProfilerStart, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAPROFILERSTOP] = std::bind(&TallyServer::handle_cudaProfilerStop, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETCUBINSIZE] = std::bind(&TallyServer::handle_nvrtcGetCUBINSize, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::NVRTCGETCUBIN] = std::bind(&TallyServer::handle_nvrtcGetCUBIN, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMALLOC] = std::bind(&TallyServer::handle_cudaMalloc, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDAMEMCPY] = std::bind(&TallyServer::handle_cudaMemcpy, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::CUDALAUNCHKERNEL] = std::bind(&TallyServer::handle_cudaLaunchKernel, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::__CUDAREGISTERFUNCTION] = std::bind(&TallyServer::handle___cudaRegisterFunction, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::__CUDAREGISTERFATBINARY] = std::bind(&TallyServer::handle___cudaRegisterFatBinary, this, std::placeholders::_1);
	cuda_api_handler_map[CUDA_API_ENUM::__CUDAREGISTERFATBINARYEND] = std::bind(&TallyServer::handle___cudaRegisterFatBinaryEnd, this, std::placeholders::_1);
}

void TallyServer::handle_cuGetErrorString(void *__args)
{
	spdlog::info("Received request: cuGetErrorString");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGetErrorName(void *__args)
{
	spdlog::info("Received request: cuGetErrorName");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuInit(void *__args)
{
	spdlog::info("Received request: cuInit");

    auto args = (struct cuInitArg *) __args;
    CUresult err = cuInit(
		args->Flags

    );

    while(!send_ipc->send((void *) &err, sizeof(CUresult))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDriverGetVersion(void *__args)
{
	spdlog::info("Received request: cuDriverGetVersion");

    auto args = (struct cuDriverGetVersionArg *) __args;

    int  driverVersion;
    CUresult err = cuDriverGetVersion(&driverVersion);

    struct cuDriverGetVersionResponse res { driverVersion, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDriverGetVersionResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGet(void *__args)
{
	spdlog::info("Received request: cuDeviceGet");

    auto args = (struct cuDeviceGetArg *) __args;

    CUdevice  device;
    CUresult err = cuDeviceGet(&device, args->ordinal);

    struct cuDeviceGetResponse res { device, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDeviceGetResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetCount(void *__args)
{
	spdlog::info("Received request: cuDeviceGetCount");

    auto args = (struct cuDeviceGetCountArg *) __args;

    int  count;
    CUresult err = cuDeviceGetCount(&count);

    struct cuDeviceGetCountResponse res { count, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDeviceGetCountResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetName(void *__args)
{
	spdlog::info("Received request: cuDeviceGetName");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceGetUuid(void *__args)
{
	spdlog::info("Received request: cuDeviceGetUuid");

    auto args = (struct cuDeviceGetUuidArg *) __args;

    CUuuid  uuid;
    CUresult err = cuDeviceGetUuid(&uuid, args->dev);

    struct cuDeviceGetUuidResponse res { uuid, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDeviceGetUuidResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetUuid_v2(void *__args)
{
	spdlog::info("Received request: cuDeviceGetUuid_v2");

    auto args = (struct cuDeviceGetUuid_v2Arg *) __args;

    CUuuid  uuid;
    CUresult err = cuDeviceGetUuid_v2(&uuid, args->dev);

    struct cuDeviceGetUuid_v2Response res { uuid, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDeviceGetUuid_v2Response))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetLuid(void *__args)
{
	spdlog::info("Received request: cuDeviceGetLuid");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceTotalMem_v2(void *__args)
{
	spdlog::info("Received request: cuDeviceTotalMem_v2");

    auto args = (struct cuDeviceTotalMem_v2Arg *) __args;

    size_t  bytes;
    CUresult err = cuDeviceTotalMem_v2(&bytes, args->dev);

    struct cuDeviceTotalMem_v2Response res { bytes, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDeviceTotalMem_v2Response))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetAttribute(void *__args)
{
	spdlog::info("Received request: cuDeviceGetAttribute");

    auto args = (struct cuDeviceGetAttributeArg *) __args;

    int  pi;
    CUresult err = cuDeviceGetAttribute(&pi, args->attrib, args->dev);

    struct cuDeviceGetAttributeResponse res { pi, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDeviceGetAttributeResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetNvSciSyncAttributes(void *__args)
{
	spdlog::info("Received request: cuDeviceGetNvSciSyncAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceSetMemPool(void *__args)
{
	spdlog::info("Received request: cuDeviceSetMemPool");

    auto args = (struct cuDeviceSetMemPoolArg *) __args;
    CUresult err = cuDeviceSetMemPool(
		args->dev,
		args->pool

    );

    while(!send_ipc->send((void *) &err, sizeof(CUresult))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetMemPool(void *__args)
{
	spdlog::info("Received request: cuDeviceGetMemPool");

    auto args = (struct cuDeviceGetMemPoolArg *) __args;

    CUmemoryPool  pool;
    CUresult err = cuDeviceGetMemPool(&pool, args->dev);

    struct cuDeviceGetMemPoolResponse res { pool, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDeviceGetMemPoolResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetDefaultMemPool(void *__args)
{
	spdlog::info("Received request: cuDeviceGetDefaultMemPool");

    auto args = (struct cuDeviceGetDefaultMemPoolArg *) __args;

    CUmemoryPool  pool_out;
    CUresult err = cuDeviceGetDefaultMemPool(&pool_out, args->dev);

    struct cuDeviceGetDefaultMemPoolResponse res { pool_out, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDeviceGetDefaultMemPoolResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuFlushGPUDirectRDMAWrites(void *__args)
{
	spdlog::info("Received request: cuFlushGPUDirectRDMAWrites");

    auto args = (struct cuFlushGPUDirectRDMAWritesArg *) __args;
    CUresult err = cuFlushGPUDirectRDMAWrites(
		args->target,
		args->scope

    );

    while(!send_ipc->send((void *) &err, sizeof(CUresult))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetProperties(void *__args)
{
	spdlog::info("Received request: cuDeviceGetProperties");

    auto args = (struct cuDeviceGetPropertiesArg *) __args;

    CUdevprop  prop;
    CUresult err = cuDeviceGetProperties(&prop, args->dev);

    struct cuDeviceGetPropertiesResponse res { prop, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDeviceGetPropertiesResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceComputeCapability(void *__args)
{
	spdlog::info("Received request: cuDeviceComputeCapability");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDevicePrimaryCtxRetain(void *__args)
{
	spdlog::info("Received request: cuDevicePrimaryCtxRetain");

    auto args = (struct cuDevicePrimaryCtxRetainArg *) __args;

    CUcontext  pctx;
    CUresult err = cuDevicePrimaryCtxRetain(&pctx, args->dev);

    struct cuDevicePrimaryCtxRetainResponse res { pctx, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cuDevicePrimaryCtxRetainResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDevicePrimaryCtxRelease_v2(void *__args)
{
	spdlog::info("Received request: cuDevicePrimaryCtxRelease_v2");

    auto args = (struct cuDevicePrimaryCtxRelease_v2Arg *) __args;
    CUresult err = cuDevicePrimaryCtxRelease_v2(
		args->dev

    );

    while(!send_ipc->send((void *) &err, sizeof(CUresult))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDevicePrimaryCtxSetFlags_v2(void *__args)
{
	spdlog::info("Received request: cuDevicePrimaryCtxSetFlags_v2");

    auto args = (struct cuDevicePrimaryCtxSetFlags_v2Arg *) __args;
    CUresult err = cuDevicePrimaryCtxSetFlags_v2(
		args->dev,
		args->flags

    );

    while(!send_ipc->send((void *) &err, sizeof(CUresult))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDevicePrimaryCtxGetState(void *__args)
{
	spdlog::info("Received request: cuDevicePrimaryCtxGetState");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDevicePrimaryCtxReset_v2(void *__args)
{
	spdlog::info("Received request: cuDevicePrimaryCtxReset_v2");

    auto args = (struct cuDevicePrimaryCtxReset_v2Arg *) __args;
    CUresult err = cuDevicePrimaryCtxReset_v2(
		args->dev

    );

    while(!send_ipc->send((void *) &err, sizeof(CUresult))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cuDeviceGetExecAffinitySupport(void *__args)
{
	spdlog::info("Received request: cuDeviceGetExecAffinitySupport");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxCreate_v2(void *__args)
{
	spdlog::info("Received request: cuCtxCreate_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxCreate_v3(void *__args)
{
	spdlog::info("Received request: cuCtxCreate_v3");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxDestroy_v2(void *__args)
{
	spdlog::info("Received request: cuCtxDestroy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxPushCurrent_v2(void *__args)
{
	spdlog::info("Received request: cuCtxPushCurrent_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxPopCurrent_v2(void *__args)
{
	spdlog::info("Received request: cuCtxPopCurrent_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxSetCurrent(void *__args)
{
	spdlog::info("Received request: cuCtxSetCurrent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxGetCurrent(void *__args)
{
	spdlog::info("Received request: cuCtxGetCurrent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxGetDevice(void *__args)
{
	spdlog::info("Received request: cuCtxGetDevice");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxGetFlags(void *__args)
{
	spdlog::info("Received request: cuCtxGetFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxSynchronize(void *__args)
{
	spdlog::info("Received request: cuCtxSynchronize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxSetLimit(void *__args)
{
	spdlog::info("Received request: cuCtxSetLimit");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxGetLimit(void *__args)
{
	spdlog::info("Received request: cuCtxGetLimit");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxGetCacheConfig(void *__args)
{
	spdlog::info("Received request: cuCtxGetCacheConfig");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxSetCacheConfig(void *__args)
{
	spdlog::info("Received request: cuCtxSetCacheConfig");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxGetSharedMemConfig(void *__args)
{
	spdlog::info("Received request: cuCtxGetSharedMemConfig");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxSetSharedMemConfig(void *__args)
{
	spdlog::info("Received request: cuCtxSetSharedMemConfig");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxGetApiVersion(void *__args)
{
	spdlog::info("Received request: cuCtxGetApiVersion");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxGetStreamPriorityRange(void *__args)
{
	spdlog::info("Received request: cuCtxGetStreamPriorityRange");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxResetPersistingL2Cache(void *__args)
{
	spdlog::info("Received request: cuCtxResetPersistingL2Cache");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxGetExecAffinity(void *__args)
{
	spdlog::info("Received request: cuCtxGetExecAffinity");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxAttach(void *__args)
{
	spdlog::info("Received request: cuCtxAttach");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxDetach(void *__args)
{
	spdlog::info("Received request: cuCtxDetach");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuModuleLoad(void *__args)
{
	spdlog::info("Received request: cuModuleLoad");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuModuleLoadData(void *__args)
{
	spdlog::info("Received request: cuModuleLoadData");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuModuleLoadDataEx(void *__args)
{
	spdlog::info("Received request: cuModuleLoadDataEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuModuleLoadFatBinary(void *__args)
{
	spdlog::info("Received request: cuModuleLoadFatBinary");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuModuleUnload(void *__args)
{
	spdlog::info("Received request: cuModuleUnload");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuModuleGetFunction(void *__args)
{
	spdlog::info("Received request: cuModuleGetFunction");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuModuleGetGlobal_v2(void *__args)
{
	spdlog::info("Received request: cuModuleGetGlobal_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuModuleGetTexRef(void *__args)
{
	spdlog::info("Received request: cuModuleGetTexRef");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuModuleGetSurfRef(void *__args)
{
	spdlog::info("Received request: cuModuleGetSurfRef");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLinkCreate_v2(void *__args)
{
	spdlog::info("Received request: cuLinkCreate_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLinkAddData_v2(void *__args)
{
	spdlog::info("Received request: cuLinkAddData_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLinkAddFile_v2(void *__args)
{
	spdlog::info("Received request: cuLinkAddFile_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLinkComplete(void *__args)
{
	spdlog::info("Received request: cuLinkComplete");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLinkDestroy(void *__args)
{
	spdlog::info("Received request: cuLinkDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemGetInfo_v2(void *__args)
{
	spdlog::info("Received request: cuMemGetInfo_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemAlloc_v2(void *__args)
{
	spdlog::info("Received request: cuMemAlloc_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemAllocPitch_v2(void *__args)
{
	spdlog::info("Received request: cuMemAllocPitch_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemFree_v2(void *__args)
{
	spdlog::info("Received request: cuMemFree_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemGetAddressRange_v2(void *__args)
{
	spdlog::info("Received request: cuMemGetAddressRange_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemAllocHost_v2(void *__args)
{
	spdlog::info("Received request: cuMemAllocHost_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemFreeHost(void *__args)
{
	spdlog::info("Received request: cuMemFreeHost");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemHostAlloc(void *__args)
{
	spdlog::info("Received request: cuMemHostAlloc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemHostGetDevicePointer_v2(void *__args)
{
	spdlog::info("Received request: cuMemHostGetDevicePointer_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemHostGetFlags(void *__args)
{
	spdlog::info("Received request: cuMemHostGetFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemAllocManaged(void *__args)
{
	spdlog::info("Received request: cuMemAllocManaged");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceGetByPCIBusId(void *__args)
{
	spdlog::info("Received request: cuDeviceGetByPCIBusId");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceGetPCIBusId(void *__args)
{
	spdlog::info("Received request: cuDeviceGetPCIBusId");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuIpcGetEventHandle(void *__args)
{
	spdlog::info("Received request: cuIpcGetEventHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuIpcOpenEventHandle(void *__args)
{
	spdlog::info("Received request: cuIpcOpenEventHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuIpcGetMemHandle(void *__args)
{
	spdlog::info("Received request: cuIpcGetMemHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuIpcOpenMemHandle_v2(void *__args)
{
	spdlog::info("Received request: cuIpcOpenMemHandle_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuIpcCloseMemHandle(void *__args)
{
	spdlog::info("Received request: cuIpcCloseMemHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemHostRegister_v2(void *__args)
{
	spdlog::info("Received request: cuMemHostRegister_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemHostUnregister(void *__args)
{
	spdlog::info("Received request: cuMemHostUnregister");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpy(void *__args)
{
	spdlog::info("Received request: cuMemcpy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyPeer(void *__args)
{
	spdlog::info("Received request: cuMemcpyPeer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyHtoD_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyHtoD_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyDtoH_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyDtoH_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyDtoD_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyDtoD_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyDtoA_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyDtoA_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyAtoD_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyAtoD_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyHtoA_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyHtoA_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyAtoH_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyAtoH_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyAtoA_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyAtoA_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpy2D_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpy2D_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpy2DUnaligned_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpy2DUnaligned_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpy3D_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpy3D_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpy3DPeer(void *__args)
{
	spdlog::info("Received request: cuMemcpy3DPeer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyAsync(void *__args)
{
	spdlog::info("Received request: cuMemcpyAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyPeerAsync(void *__args)
{
	spdlog::info("Received request: cuMemcpyPeerAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyHtoDAsync_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyHtoDAsync_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyDtoHAsync_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyDtoHAsync_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyDtoDAsync_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyDtoDAsync_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyHtoAAsync_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyHtoAAsync_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpyAtoHAsync_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpyAtoHAsync_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpy2DAsync_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpy2DAsync_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpy3DAsync_v2(void *__args)
{
	spdlog::info("Received request: cuMemcpy3DAsync_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemcpy3DPeerAsync(void *__args)
{
	spdlog::info("Received request: cuMemcpy3DPeerAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD8_v2(void *__args)
{
	spdlog::info("Received request: cuMemsetD8_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD16_v2(void *__args)
{
	spdlog::info("Received request: cuMemsetD16_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD32_v2(void *__args)
{
	spdlog::info("Received request: cuMemsetD32_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD2D8_v2(void *__args)
{
	spdlog::info("Received request: cuMemsetD2D8_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD2D16_v2(void *__args)
{
	spdlog::info("Received request: cuMemsetD2D16_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD2D32_v2(void *__args)
{
	spdlog::info("Received request: cuMemsetD2D32_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD8Async(void *__args)
{
	spdlog::info("Received request: cuMemsetD8Async");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD16Async(void *__args)
{
	spdlog::info("Received request: cuMemsetD16Async");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD32Async(void *__args)
{
	spdlog::info("Received request: cuMemsetD32Async");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD2D8Async(void *__args)
{
	spdlog::info("Received request: cuMemsetD2D8Async");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD2D16Async(void *__args)
{
	spdlog::info("Received request: cuMemsetD2D16Async");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemsetD2D32Async(void *__args)
{
	spdlog::info("Received request: cuMemsetD2D32Async");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuArrayCreate_v2(void *__args)
{
	spdlog::info("Received request: cuArrayCreate_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuArrayGetDescriptor_v2(void *__args)
{
	spdlog::info("Received request: cuArrayGetDescriptor_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuArrayGetSparseProperties(void *__args)
{
	spdlog::info("Received request: cuArrayGetSparseProperties");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMipmappedArrayGetSparseProperties(void *__args)
{
	spdlog::info("Received request: cuMipmappedArrayGetSparseProperties");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuArrayGetPlane(void *__args)
{
	spdlog::info("Received request: cuArrayGetPlane");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuArrayDestroy(void *__args)
{
	spdlog::info("Received request: cuArrayDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuArray3DCreate_v2(void *__args)
{
	spdlog::info("Received request: cuArray3DCreate_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuArray3DGetDescriptor_v2(void *__args)
{
	spdlog::info("Received request: cuArray3DGetDescriptor_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMipmappedArrayCreate(void *__args)
{
	spdlog::info("Received request: cuMipmappedArrayCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMipmappedArrayGetLevel(void *__args)
{
	spdlog::info("Received request: cuMipmappedArrayGetLevel");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMipmappedArrayDestroy(void *__args)
{
	spdlog::info("Received request: cuMipmappedArrayDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemAddressReserve(void *__args)
{
	spdlog::info("Received request: cuMemAddressReserve");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemAddressFree(void *__args)
{
	spdlog::info("Received request: cuMemAddressFree");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemCreate(void *__args)
{
	spdlog::info("Received request: cuMemCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemRelease(void *__args)
{
	spdlog::info("Received request: cuMemRelease");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemMap(void *__args)
{
	spdlog::info("Received request: cuMemMap");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemMapArrayAsync(void *__args)
{
	spdlog::info("Received request: cuMemMapArrayAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemUnmap(void *__args)
{
	spdlog::info("Received request: cuMemUnmap");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemSetAccess(void *__args)
{
	spdlog::info("Received request: cuMemSetAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemGetAccess(void *__args)
{
	spdlog::info("Received request: cuMemGetAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemExportToShareableHandle(void *__args)
{
	spdlog::info("Received request: cuMemExportToShareableHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemImportFromShareableHandle(void *__args)
{
	spdlog::info("Received request: cuMemImportFromShareableHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemGetAllocationGranularity(void *__args)
{
	spdlog::info("Received request: cuMemGetAllocationGranularity");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemGetAllocationPropertiesFromHandle(void *__args)
{
	spdlog::info("Received request: cuMemGetAllocationPropertiesFromHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemRetainAllocationHandle(void *__args)
{
	spdlog::info("Received request: cuMemRetainAllocationHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemFreeAsync(void *__args)
{
	spdlog::info("Received request: cuMemFreeAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemAllocAsync(void *__args)
{
	spdlog::info("Received request: cuMemAllocAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolTrimTo(void *__args)
{
	spdlog::info("Received request: cuMemPoolTrimTo");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolSetAttribute(void *__args)
{
	spdlog::info("Received request: cuMemPoolSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolGetAttribute(void *__args)
{
	spdlog::info("Received request: cuMemPoolGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolSetAccess(void *__args)
{
	spdlog::info("Received request: cuMemPoolSetAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolGetAccess(void *__args)
{
	spdlog::info("Received request: cuMemPoolGetAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolCreate(void *__args)
{
	spdlog::info("Received request: cuMemPoolCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolDestroy(void *__args)
{
	spdlog::info("Received request: cuMemPoolDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemAllocFromPoolAsync(void *__args)
{
	spdlog::info("Received request: cuMemAllocFromPoolAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolExportToShareableHandle(void *__args)
{
	spdlog::info("Received request: cuMemPoolExportToShareableHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolImportFromShareableHandle(void *__args)
{
	spdlog::info("Received request: cuMemPoolImportFromShareableHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolExportPointer(void *__args)
{
	spdlog::info("Received request: cuMemPoolExportPointer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPoolImportPointer(void *__args)
{
	spdlog::info("Received request: cuMemPoolImportPointer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuPointerGetAttribute(void *__args)
{
	spdlog::info("Received request: cuPointerGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemPrefetchAsync(void *__args)
{
	spdlog::info("Received request: cuMemPrefetchAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemAdvise(void *__args)
{
	spdlog::info("Received request: cuMemAdvise");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemRangeGetAttribute(void *__args)
{
	spdlog::info("Received request: cuMemRangeGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuMemRangeGetAttributes(void *__args)
{
	spdlog::info("Received request: cuMemRangeGetAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuPointerSetAttribute(void *__args)
{
	spdlog::info("Received request: cuPointerSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuPointerGetAttributes(void *__args)
{
	spdlog::info("Received request: cuPointerGetAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamCreate(void *__args)
{
	spdlog::info("Received request: cuStreamCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamCreateWithPriority(void *__args)
{
	spdlog::info("Received request: cuStreamCreateWithPriority");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamGetPriority(void *__args)
{
	spdlog::info("Received request: cuStreamGetPriority");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamGetFlags(void *__args)
{
	spdlog::info("Received request: cuStreamGetFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamGetCtx(void *__args)
{
	spdlog::info("Received request: cuStreamGetCtx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamWaitEvent(void *__args)
{
	spdlog::info("Received request: cuStreamWaitEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamAddCallback(void *__args)
{
	spdlog::info("Received request: cuStreamAddCallback");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamBeginCapture_v2(void *__args)
{
	spdlog::info("Received request: cuStreamBeginCapture_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuThreadExchangeStreamCaptureMode(void *__args)
{
	spdlog::info("Received request: cuThreadExchangeStreamCaptureMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamEndCapture(void *__args)
{
	spdlog::info("Received request: cuStreamEndCapture");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamIsCapturing(void *__args)
{
	spdlog::info("Received request: cuStreamIsCapturing");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamGetCaptureInfo(void *__args)
{
	spdlog::info("Received request: cuStreamGetCaptureInfo");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamGetCaptureInfo_v2(void *__args)
{
	spdlog::info("Received request: cuStreamGetCaptureInfo_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamUpdateCaptureDependencies(void *__args)
{
	spdlog::info("Received request: cuStreamUpdateCaptureDependencies");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamAttachMemAsync(void *__args)
{
	spdlog::info("Received request: cuStreamAttachMemAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamQuery(void *__args)
{
	spdlog::info("Received request: cuStreamQuery");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamSynchronize(void *__args)
{
	spdlog::info("Received request: cuStreamSynchronize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamDestroy_v2(void *__args)
{
	spdlog::info("Received request: cuStreamDestroy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamCopyAttributes(void *__args)
{
	spdlog::info("Received request: cuStreamCopyAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamGetAttribute(void *__args)
{
	spdlog::info("Received request: cuStreamGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamSetAttribute(void *__args)
{
	spdlog::info("Received request: cuStreamSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuEventCreate(void *__args)
{
	spdlog::info("Received request: cuEventCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuEventRecord(void *__args)
{
	spdlog::info("Received request: cuEventRecord");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuEventRecordWithFlags(void *__args)
{
	spdlog::info("Received request: cuEventRecordWithFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuEventQuery(void *__args)
{
	spdlog::info("Received request: cuEventQuery");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuEventSynchronize(void *__args)
{
	spdlog::info("Received request: cuEventSynchronize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuEventDestroy_v2(void *__args)
{
	spdlog::info("Received request: cuEventDestroy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuEventElapsedTime(void *__args)
{
	spdlog::info("Received request: cuEventElapsedTime");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuImportExternalMemory(void *__args)
{
	spdlog::info("Received request: cuImportExternalMemory");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuExternalMemoryGetMappedBuffer(void *__args)
{
	spdlog::info("Received request: cuExternalMemoryGetMappedBuffer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuExternalMemoryGetMappedMipmappedArray(void *__args)
{
	spdlog::info("Received request: cuExternalMemoryGetMappedMipmappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDestroyExternalMemory(void *__args)
{
	spdlog::info("Received request: cuDestroyExternalMemory");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuImportExternalSemaphore(void *__args)
{
	spdlog::info("Received request: cuImportExternalSemaphore");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuSignalExternalSemaphoresAsync(void *__args)
{
	spdlog::info("Received request: cuSignalExternalSemaphoresAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuWaitExternalSemaphoresAsync(void *__args)
{
	spdlog::info("Received request: cuWaitExternalSemaphoresAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDestroyExternalSemaphore(void *__args)
{
	spdlog::info("Received request: cuDestroyExternalSemaphore");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamWaitValue32(void *__args)
{
	spdlog::info("Received request: cuStreamWaitValue32");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamWaitValue64(void *__args)
{
	spdlog::info("Received request: cuStreamWaitValue64");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamWriteValue32(void *__args)
{
	spdlog::info("Received request: cuStreamWriteValue32");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamWriteValue64(void *__args)
{
	spdlog::info("Received request: cuStreamWriteValue64");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuStreamBatchMemOp(void *__args)
{
	spdlog::info("Received request: cuStreamBatchMemOp");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuFuncGetAttribute(void *__args)
{
	spdlog::info("Received request: cuFuncGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuFuncSetAttribute(void *__args)
{
	spdlog::info("Received request: cuFuncSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuFuncSetCacheConfig(void *__args)
{
	spdlog::info("Received request: cuFuncSetCacheConfig");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuFuncSetSharedMemConfig(void *__args)
{
	spdlog::info("Received request: cuFuncSetSharedMemConfig");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuFuncGetModule(void *__args)
{
	spdlog::info("Received request: cuFuncGetModule");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLaunchKernel(void *__args)
{
	spdlog::info("Received request: cuLaunchKernel");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLaunchCooperativeKernel(void *__args)
{
	spdlog::info("Received request: cuLaunchCooperativeKernel");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLaunchCooperativeKernelMultiDevice(void *__args)
{
	spdlog::info("Received request: cuLaunchCooperativeKernelMultiDevice");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLaunchHostFunc(void *__args)
{
	spdlog::info("Received request: cuLaunchHostFunc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuFuncSetBlockShape(void *__args)
{
	spdlog::info("Received request: cuFuncSetBlockShape");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuFuncSetSharedSize(void *__args)
{
	spdlog::info("Received request: cuFuncSetSharedSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuParamSetSize(void *__args)
{
	spdlog::info("Received request: cuParamSetSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuParamSeti(void *__args)
{
	spdlog::info("Received request: cuParamSeti");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuParamSetf(void *__args)
{
	spdlog::info("Received request: cuParamSetf");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuParamSetv(void *__args)
{
	spdlog::info("Received request: cuParamSetv");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLaunch(void *__args)
{
	spdlog::info("Received request: cuLaunch");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLaunchGrid(void *__args)
{
	spdlog::info("Received request: cuLaunchGrid");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuLaunchGridAsync(void *__args)
{
	spdlog::info("Received request: cuLaunchGridAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuParamSetTexRef(void *__args)
{
	spdlog::info("Received request: cuParamSetTexRef");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphCreate(void *__args)
{
	spdlog::info("Received request: cuGraphCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddKernelNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddKernelNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphKernelNodeGetParams(void *__args)
{
	spdlog::info("Received request: cuGraphKernelNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphKernelNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphKernelNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddMemcpyNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddMemcpyNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphMemcpyNodeGetParams(void *__args)
{
	spdlog::info("Received request: cuGraphMemcpyNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphMemcpyNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphMemcpyNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddMemsetNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddMemsetNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphMemsetNodeGetParams(void *__args)
{
	spdlog::info("Received request: cuGraphMemsetNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphMemsetNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphMemsetNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddHostNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddHostNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphHostNodeGetParams(void *__args)
{
	spdlog::info("Received request: cuGraphHostNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphHostNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphHostNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddChildGraphNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddChildGraphNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphChildGraphNodeGetGraph(void *__args)
{
	spdlog::info("Received request: cuGraphChildGraphNodeGetGraph");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddEmptyNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddEmptyNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddEventRecordNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddEventRecordNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphEventRecordNodeGetEvent(void *__args)
{
	spdlog::info("Received request: cuGraphEventRecordNodeGetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphEventRecordNodeSetEvent(void *__args)
{
	spdlog::info("Received request: cuGraphEventRecordNodeSetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddEventWaitNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddEventWaitNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphEventWaitNodeGetEvent(void *__args)
{
	spdlog::info("Received request: cuGraphEventWaitNodeGetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphEventWaitNodeSetEvent(void *__args)
{
	spdlog::info("Received request: cuGraphEventWaitNodeSetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddExternalSemaphoresSignalNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddExternalSemaphoresSignalNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExternalSemaphoresSignalNodeGetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExternalSemaphoresSignalNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExternalSemaphoresSignalNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExternalSemaphoresSignalNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddExternalSemaphoresWaitNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddExternalSemaphoresWaitNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExternalSemaphoresWaitNodeGetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExternalSemaphoresWaitNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExternalSemaphoresWaitNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExternalSemaphoresWaitNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddMemAllocNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddMemAllocNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphMemAllocNodeGetParams(void *__args)
{
	spdlog::info("Received request: cuGraphMemAllocNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddMemFreeNode(void *__args)
{
	spdlog::info("Received request: cuGraphAddMemFreeNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphMemFreeNodeGetParams(void *__args)
{
	spdlog::info("Received request: cuGraphMemFreeNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceGraphMemTrim(void *__args)
{
	spdlog::info("Received request: cuDeviceGraphMemTrim");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceGetGraphMemAttribute(void *__args)
{
	spdlog::info("Received request: cuDeviceGetGraphMemAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceSetGraphMemAttribute(void *__args)
{
	spdlog::info("Received request: cuDeviceSetGraphMemAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphClone(void *__args)
{
	spdlog::info("Received request: cuGraphClone");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphNodeFindInClone(void *__args)
{
	spdlog::info("Received request: cuGraphNodeFindInClone");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphNodeGetType(void *__args)
{
	spdlog::info("Received request: cuGraphNodeGetType");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphGetNodes(void *__args)
{
	spdlog::info("Received request: cuGraphGetNodes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphGetRootNodes(void *__args)
{
	spdlog::info("Received request: cuGraphGetRootNodes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphGetEdges(void *__args)
{
	spdlog::info("Received request: cuGraphGetEdges");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphNodeGetDependencies(void *__args)
{
	spdlog::info("Received request: cuGraphNodeGetDependencies");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphNodeGetDependentNodes(void *__args)
{
	spdlog::info("Received request: cuGraphNodeGetDependentNodes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphAddDependencies(void *__args)
{
	spdlog::info("Received request: cuGraphAddDependencies");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphRemoveDependencies(void *__args)
{
	spdlog::info("Received request: cuGraphRemoveDependencies");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphDestroyNode(void *__args)
{
	spdlog::info("Received request: cuGraphDestroyNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphInstantiate_v2(void *__args)
{
	spdlog::info("Received request: cuGraphInstantiate_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphInstantiateWithFlags(void *__args)
{
	spdlog::info("Received request: cuGraphInstantiateWithFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecKernelNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExecKernelNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecMemcpyNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExecMemcpyNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecMemsetNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExecMemsetNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecHostNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExecHostNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecChildGraphNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExecChildGraphNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecEventRecordNodeSetEvent(void *__args)
{
	spdlog::info("Received request: cuGraphExecEventRecordNodeSetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecEventWaitNodeSetEvent(void *__args)
{
	spdlog::info("Received request: cuGraphExecEventWaitNodeSetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecExternalSemaphoresSignalNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExecExternalSemaphoresSignalNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecExternalSemaphoresWaitNodeSetParams(void *__args)
{
	spdlog::info("Received request: cuGraphExecExternalSemaphoresWaitNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphUpload(void *__args)
{
	spdlog::info("Received request: cuGraphUpload");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphLaunch(void *__args)
{
	spdlog::info("Received request: cuGraphLaunch");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecDestroy(void *__args)
{
	spdlog::info("Received request: cuGraphExecDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphDestroy(void *__args)
{
	spdlog::info("Received request: cuGraphDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphExecUpdate(void *__args)
{
	spdlog::info("Received request: cuGraphExecUpdate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphKernelNodeCopyAttributes(void *__args)
{
	spdlog::info("Received request: cuGraphKernelNodeCopyAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphKernelNodeGetAttribute(void *__args)
{
	spdlog::info("Received request: cuGraphKernelNodeGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphKernelNodeSetAttribute(void *__args)
{
	spdlog::info("Received request: cuGraphKernelNodeSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphDebugDotPrint(void *__args)
{
	spdlog::info("Received request: cuGraphDebugDotPrint");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuUserObjectCreate(void *__args)
{
	spdlog::info("Received request: cuUserObjectCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuUserObjectRetain(void *__args)
{
	spdlog::info("Received request: cuUserObjectRetain");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuUserObjectRelease(void *__args)
{
	spdlog::info("Received request: cuUserObjectRelease");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphRetainUserObject(void *__args)
{
	spdlog::info("Received request: cuGraphRetainUserObject");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphReleaseUserObject(void *__args)
{
	spdlog::info("Received request: cuGraphReleaseUserObject");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuOccupancyMaxActiveBlocksPerMultiprocessor(void *__args)
{
	spdlog::info("Received request: cuOccupancyMaxActiveBlocksPerMultiprocessor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *__args)
{
	spdlog::info("Received request: cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuOccupancyMaxPotentialBlockSize(void *__args)
{
	spdlog::info("Received request: cuOccupancyMaxPotentialBlockSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuOccupancyMaxPotentialBlockSizeWithFlags(void *__args)
{
	spdlog::info("Received request: cuOccupancyMaxPotentialBlockSizeWithFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuOccupancyAvailableDynamicSMemPerBlock(void *__args)
{
	spdlog::info("Received request: cuOccupancyAvailableDynamicSMemPerBlock");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetArray(void *__args)
{
	spdlog::info("Received request: cuTexRefSetArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetMipmappedArray(void *__args)
{
	spdlog::info("Received request: cuTexRefSetMipmappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetAddress_v2(void *__args)
{
	spdlog::info("Received request: cuTexRefSetAddress_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetAddress2D_v3(void *__args)
{
	spdlog::info("Received request: cuTexRefSetAddress2D_v3");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetAddressMode(void *__args)
{
	spdlog::info("Received request: cuTexRefSetAddressMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetFilterMode(void *__args)
{
	spdlog::info("Received request: cuTexRefSetFilterMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetMipmapFilterMode(void *__args)
{
	spdlog::info("Received request: cuTexRefSetMipmapFilterMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetMipmapLevelBias(void *__args)
{
	spdlog::info("Received request: cuTexRefSetMipmapLevelBias");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetMipmapLevelClamp(void *__args)
{
	spdlog::info("Received request: cuTexRefSetMipmapLevelClamp");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetMaxAnisotropy(void *__args)
{
	spdlog::info("Received request: cuTexRefSetMaxAnisotropy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetBorderColor(void *__args)
{
	spdlog::info("Received request: cuTexRefSetBorderColor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefSetFlags(void *__args)
{
	spdlog::info("Received request: cuTexRefSetFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetAddress_v2(void *__args)
{
	spdlog::info("Received request: cuTexRefGetAddress_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetArray(void *__args)
{
	spdlog::info("Received request: cuTexRefGetArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetMipmappedArray(void *__args)
{
	spdlog::info("Received request: cuTexRefGetMipmappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetAddressMode(void *__args)
{
	spdlog::info("Received request: cuTexRefGetAddressMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetFilterMode(void *__args)
{
	spdlog::info("Received request: cuTexRefGetFilterMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetMipmapFilterMode(void *__args)
{
	spdlog::info("Received request: cuTexRefGetMipmapFilterMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetMipmapLevelBias(void *__args)
{
	spdlog::info("Received request: cuTexRefGetMipmapLevelBias");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetMipmapLevelClamp(void *__args)
{
	spdlog::info("Received request: cuTexRefGetMipmapLevelClamp");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetMaxAnisotropy(void *__args)
{
	spdlog::info("Received request: cuTexRefGetMaxAnisotropy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetBorderColor(void *__args)
{
	spdlog::info("Received request: cuTexRefGetBorderColor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefGetFlags(void *__args)
{
	spdlog::info("Received request: cuTexRefGetFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefCreate(void *__args)
{
	spdlog::info("Received request: cuTexRefCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexRefDestroy(void *__args)
{
	spdlog::info("Received request: cuTexRefDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuSurfRefSetArray(void *__args)
{
	spdlog::info("Received request: cuSurfRefSetArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuSurfRefGetArray(void *__args)
{
	spdlog::info("Received request: cuSurfRefGetArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexObjectCreate(void *__args)
{
	spdlog::info("Received request: cuTexObjectCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexObjectDestroy(void *__args)
{
	spdlog::info("Received request: cuTexObjectDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexObjectGetResourceDesc(void *__args)
{
	spdlog::info("Received request: cuTexObjectGetResourceDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexObjectGetTextureDesc(void *__args)
{
	spdlog::info("Received request: cuTexObjectGetTextureDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuTexObjectGetResourceViewDesc(void *__args)
{
	spdlog::info("Received request: cuTexObjectGetResourceViewDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuSurfObjectCreate(void *__args)
{
	spdlog::info("Received request: cuSurfObjectCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuSurfObjectDestroy(void *__args)
{
	spdlog::info("Received request: cuSurfObjectDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuSurfObjectGetResourceDesc(void *__args)
{
	spdlog::info("Received request: cuSurfObjectGetResourceDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceCanAccessPeer(void *__args)
{
	spdlog::info("Received request: cuDeviceCanAccessPeer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxEnablePeerAccess(void *__args)
{
	spdlog::info("Received request: cuCtxEnablePeerAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuCtxDisablePeerAccess(void *__args)
{
	spdlog::info("Received request: cuCtxDisablePeerAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuDeviceGetP2PAttribute(void *__args)
{
	spdlog::info("Received request: cuDeviceGetP2PAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphicsUnregisterResource(void *__args)
{
	spdlog::info("Received request: cuGraphicsUnregisterResource");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphicsSubResourceGetMappedArray(void *__args)
{
	spdlog::info("Received request: cuGraphicsSubResourceGetMappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphicsResourceGetMappedMipmappedArray(void *__args)
{
	spdlog::info("Received request: cuGraphicsResourceGetMappedMipmappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphicsResourceGetMappedPointer_v2(void *__args)
{
	spdlog::info("Received request: cuGraphicsResourceGetMappedPointer_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphicsResourceSetMapFlags_v2(void *__args)
{
	spdlog::info("Received request: cuGraphicsResourceSetMapFlags_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphicsMapResources(void *__args)
{
	spdlog::info("Received request: cuGraphicsMapResources");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGraphicsUnmapResources(void *__args)
{
	spdlog::info("Received request: cuGraphicsUnmapResources");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGetProcAddress(void *__args)
{
	spdlog::info("Received request: cuGetProcAddress");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cuGetExportTable(void *__args)
{
	spdlog::info("Received request: cuGetExportTable");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceReset(void *__args)
{
	spdlog::info("Received request: cudaDeviceReset");

    auto args = (struct cudaDeviceResetArg *) __args;
    cudaError_t err = cudaDeviceReset(

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceSynchronize(void *__args)
{
	spdlog::info("Received request: cudaDeviceSynchronize");

    auto args = (struct cudaDeviceSynchronizeArg *) __args;
    cudaError_t err = cudaDeviceSynchronize(

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceSetLimit(void *__args)
{
	spdlog::info("Received request: cudaDeviceSetLimit");

    auto args = (struct cudaDeviceSetLimitArg *) __args;
    cudaError_t err = cudaDeviceSetLimit(
		args->limit,
		args->value

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceGetLimit(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetLimit");

    auto args = (struct cudaDeviceGetLimitArg *) __args;

    size_t  pValue;
    cudaError_t err = cudaDeviceGetLimit(&pValue, args->limit);

    struct cudaDeviceGetLimitResponse res { pValue, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaDeviceGetLimitResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceGetTexture1DLinearMaxWidth(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetTexture1DLinearMaxWidth");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceGetCacheConfig(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetCacheConfig");

    auto args = (struct cudaDeviceGetCacheConfigArg *) __args;

    enum cudaFuncCache  pCacheConfig;
    cudaError_t err = cudaDeviceGetCacheConfig(&pCacheConfig);

    struct cudaDeviceGetCacheConfigResponse res { pCacheConfig, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaDeviceGetCacheConfigResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceGetStreamPriorityRange(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetStreamPriorityRange");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceSetCacheConfig(void *__args)
{
	spdlog::info("Received request: cudaDeviceSetCacheConfig");

    auto args = (struct cudaDeviceSetCacheConfigArg *) __args;
    cudaError_t err = cudaDeviceSetCacheConfig(
		args->cacheConfig

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceGetSharedMemConfig(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetSharedMemConfig");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceSetSharedMemConfig(void *__args)
{
	spdlog::info("Received request: cudaDeviceSetSharedMemConfig");

    auto args = (struct cudaDeviceSetSharedMemConfigArg *) __args;
    cudaError_t err = cudaDeviceSetSharedMemConfig(
		args->config

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceGetByPCIBusId(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetByPCIBusId");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceGetPCIBusId(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetPCIBusId");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaIpcGetEventHandle(void *__args)
{
	spdlog::info("Received request: cudaIpcGetEventHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaIpcOpenEventHandle(void *__args)
{
	spdlog::info("Received request: cudaIpcOpenEventHandle");

    auto args = (struct cudaIpcOpenEventHandleArg *) __args;

    cudaEvent_t  event;
    cudaError_t err = cudaIpcOpenEventHandle(&event, args->handle);

    struct cudaIpcOpenEventHandleResponse res { event, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaIpcOpenEventHandleResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaIpcGetMemHandle(void *__args)
{
	spdlog::info("Received request: cudaIpcGetMemHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaIpcOpenMemHandle(void *__args)
{
	spdlog::info("Received request: cudaIpcOpenMemHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaIpcCloseMemHandle(void *__args)
{
	spdlog::info("Received request: cudaIpcCloseMemHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceFlushGPUDirectRDMAWrites(void *__args)
{
	spdlog::info("Received request: cudaDeviceFlushGPUDirectRDMAWrites");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaThreadExit(void *__args)
{
	spdlog::info("Received request: cudaThreadExit");

    auto args = (struct cudaThreadExitArg *) __args;
    cudaError_t err = cudaThreadExit(

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaThreadSynchronize(void *__args)
{
	spdlog::info("Received request: cudaThreadSynchronize");

    auto args = (struct cudaThreadSynchronizeArg *) __args;
    cudaError_t err = cudaThreadSynchronize(

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaThreadSetLimit(void *__args)
{
	spdlog::info("Received request: cudaThreadSetLimit");

    auto args = (struct cudaThreadSetLimitArg *) __args;
    cudaError_t err = cudaThreadSetLimit(
		args->limit,
		args->value

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaThreadGetLimit(void *__args)
{
	spdlog::info("Received request: cudaThreadGetLimit");

    auto args = (struct cudaThreadGetLimitArg *) __args;

    size_t  pValue;
    cudaError_t err = cudaThreadGetLimit(&pValue, args->limit);

    struct cudaThreadGetLimitResponse res { pValue, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaThreadGetLimitResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaThreadGetCacheConfig(void *__args)
{
	spdlog::info("Received request: cudaThreadGetCacheConfig");

    auto args = (struct cudaThreadGetCacheConfigArg *) __args;

    enum cudaFuncCache  pCacheConfig;
    cudaError_t err = cudaThreadGetCacheConfig(&pCacheConfig);

    struct cudaThreadGetCacheConfigResponse res { pCacheConfig, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaThreadGetCacheConfigResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaThreadSetCacheConfig(void *__args)
{
	spdlog::info("Received request: cudaThreadSetCacheConfig");

    auto args = (struct cudaThreadSetCacheConfigArg *) __args;
    cudaError_t err = cudaThreadSetCacheConfig(
		args->cacheConfig

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaGetLastError(void *__args)
{
	spdlog::info("Received request: cudaGetLastError");

    auto args = (struct cudaGetLastErrorArg *) __args;
    cudaError_t err = cudaGetLastError(

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaPeekAtLastError(void *__args)
{
	spdlog::info("Received request: cudaPeekAtLastError");

    auto args = (struct cudaPeekAtLastErrorArg *) __args;
    cudaError_t err = cudaPeekAtLastError(

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaGetErrorName(void *__args)
{
	spdlog::info("Received request: cudaGetErrorName");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetErrorString(void *__args)
{
	spdlog::info("Received request: cudaGetErrorString");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetDeviceCount(void *__args)
{
	spdlog::info("Received request: cudaGetDeviceCount");

    auto args = (struct cudaGetDeviceCountArg *) __args;

    int  count;
    cudaError_t err = cudaGetDeviceCount(&count);

    struct cudaGetDeviceCountResponse res { count, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaGetDeviceCountResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaGetDeviceProperties(void *__args)
{
	spdlog::info("Received request: cudaGetDeviceProperties");

    auto args = (struct cudaGetDevicePropertiesArg *) __args;

    struct cudaDeviceProp  prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, args->device);

    struct cudaGetDevicePropertiesResponse res { prop, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaGetDevicePropertiesResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceGetAttribute(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetAttribute");

    auto args = (struct cudaDeviceGetAttributeArg *) __args;

    int  value;
    cudaError_t err = cudaDeviceGetAttribute(&value, args->attr, args->device);

    struct cudaDeviceGetAttributeResponse res { value, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaDeviceGetAttributeResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceGetDefaultMemPool(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetDefaultMemPool");

    auto args = (struct cudaDeviceGetDefaultMemPoolArg *) __args;

    cudaMemPool_t  memPool;
    cudaError_t err = cudaDeviceGetDefaultMemPool(&memPool, args->device);

    struct cudaDeviceGetDefaultMemPoolResponse res { memPool, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaDeviceGetDefaultMemPoolResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceSetMemPool(void *__args)
{
	spdlog::info("Received request: cudaDeviceSetMemPool");

    auto args = (struct cudaDeviceSetMemPoolArg *) __args;
    cudaError_t err = cudaDeviceSetMemPool(
		args->device,
		args->memPool

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceGetMemPool(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetMemPool");

    auto args = (struct cudaDeviceGetMemPoolArg *) __args;

    cudaMemPool_t  memPool;
    cudaError_t err = cudaDeviceGetMemPool(&memPool, args->device);

    struct cudaDeviceGetMemPoolResponse res { memPool, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaDeviceGetMemPoolResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaDeviceGetNvSciSyncAttributes(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetNvSciSyncAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceGetP2PAttribute(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetP2PAttribute");

    auto args = (struct cudaDeviceGetP2PAttributeArg *) __args;

    int  value;
    cudaError_t err = cudaDeviceGetP2PAttribute(&value, args->attr, args->srcDevice, args->dstDevice);

    struct cudaDeviceGetP2PAttributeResponse res { value, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaDeviceGetP2PAttributeResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaChooseDevice(void *__args)
{
	spdlog::info("Received request: cudaChooseDevice");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaSetDevice(void *__args)
{
	spdlog::info("Received request: cudaSetDevice");

    auto args = (struct cudaSetDeviceArg *) __args;
    cudaError_t err = cudaSetDevice(
		args->device

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaGetDevice(void *__args)
{
	spdlog::info("Received request: cudaGetDevice");

    auto args = (struct cudaGetDeviceArg *) __args;

    int  device;
    cudaError_t err = cudaGetDevice(&device);

    struct cudaGetDeviceResponse res { device, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaGetDeviceResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaSetValidDevices(void *__args)
{
	spdlog::info("Received request: cudaSetValidDevices");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaSetDeviceFlags(void *__args)
{
	spdlog::info("Received request: cudaSetDeviceFlags");

    auto args = (struct cudaSetDeviceFlagsArg *) __args;
    cudaError_t err = cudaSetDeviceFlags(
		args->flags

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaGetDeviceFlags(void *__args)
{
	spdlog::info("Received request: cudaGetDeviceFlags");

    auto args = (struct cudaGetDeviceFlagsArg *) __args;

    unsigned int  flags;
    cudaError_t err = cudaGetDeviceFlags(&flags);

    struct cudaGetDeviceFlagsResponse res { flags, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaGetDeviceFlagsResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamCreate(void *__args)
{
	spdlog::info("Received request: cudaStreamCreate");

    auto args = (struct cudaStreamCreateArg *) __args;

    cudaStream_t  pStream;
    cudaError_t err = cudaStreamCreate(&pStream);

    struct cudaStreamCreateResponse res { pStream, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaStreamCreateResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamCreateWithFlags(void *__args)
{
	spdlog::info("Received request: cudaStreamCreateWithFlags");

    auto args = (struct cudaStreamCreateWithFlagsArg *) __args;

    cudaStream_t  pStream;
    cudaError_t err = cudaStreamCreateWithFlags(&pStream, args->flags);

    struct cudaStreamCreateWithFlagsResponse res { pStream, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaStreamCreateWithFlagsResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamCreateWithPriority(void *__args)
{
	spdlog::info("Received request: cudaStreamCreateWithPriority");

    auto args = (struct cudaStreamCreateWithPriorityArg *) __args;

    cudaStream_t  pStream;
    cudaError_t err = cudaStreamCreateWithPriority(&pStream, args->flags, args->priority);

    struct cudaStreamCreateWithPriorityResponse res { pStream, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaStreamCreateWithPriorityResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamGetPriority(void *__args)
{
	spdlog::info("Received request: cudaStreamGetPriority");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaStreamGetFlags(void *__args)
{
	spdlog::info("Received request: cudaStreamGetFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaCtxResetPersistingL2Cache(void *__args)
{
	spdlog::info("Received request: cudaCtxResetPersistingL2Cache");

    auto args = (struct cudaCtxResetPersistingL2CacheArg *) __args;
    cudaError_t err = cudaCtxResetPersistingL2Cache(

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamCopyAttributes(void *__args)
{
	spdlog::info("Received request: cudaStreamCopyAttributes");

    auto args = (struct cudaStreamCopyAttributesArg *) __args;
    cudaError_t err = cudaStreamCopyAttributes(
		args->dst,
		args->src

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamGetAttribute(void *__args)
{
	spdlog::info("Received request: cudaStreamGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaStreamSetAttribute(void *__args)
{
	spdlog::info("Received request: cudaStreamSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaStreamDestroy(void *__args)
{
	spdlog::info("Received request: cudaStreamDestroy");

    auto args = (struct cudaStreamDestroyArg *) __args;
    cudaError_t err = cudaStreamDestroy(
		args->stream

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamWaitEvent(void *__args)
{
	spdlog::info("Received request: cudaStreamWaitEvent");

    auto args = (struct cudaStreamWaitEventArg *) __args;
    cudaError_t err = cudaStreamWaitEvent(
		args->stream,
		args->event,
		args->flags

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamAddCallback(void *__args)
{
	spdlog::info("Received request: cudaStreamAddCallback");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaStreamSynchronize(void *__args)
{
	spdlog::info("Received request: cudaStreamSynchronize");

    auto args = (struct cudaStreamSynchronizeArg *) __args;
    cudaError_t err = cudaStreamSynchronize(
		args->stream

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamQuery(void *__args)
{
	spdlog::info("Received request: cudaStreamQuery");

    auto args = (struct cudaStreamQueryArg *) __args;
    cudaError_t err = cudaStreamQuery(
		args->stream

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamAttachMemAsync(void *__args)
{
	spdlog::info("Received request: cudaStreamAttachMemAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaStreamBeginCapture(void *__args)
{
	spdlog::info("Received request: cudaStreamBeginCapture");

    auto args = (struct cudaStreamBeginCaptureArg *) __args;
    cudaError_t err = cudaStreamBeginCapture(
		args->stream,
		args->mode

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaThreadExchangeStreamCaptureMode(void *__args)
{
	spdlog::info("Received request: cudaThreadExchangeStreamCaptureMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaStreamEndCapture(void *__args)
{
	spdlog::info("Received request: cudaStreamEndCapture");

    auto args = (struct cudaStreamEndCaptureArg *) __args;
    cudaError_t err = cudaStreamEndCapture(
		args->stream,
		args->pGraph

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaStreamIsCapturing(void *__args)
{
	spdlog::info("Received request: cudaStreamIsCapturing");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaStreamGetCaptureInfo(void *__args)
{
	spdlog::info("Received request: cudaStreamGetCaptureInfo");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaStreamGetCaptureInfo_v2(void *__args)
{
	spdlog::info("Received request: cudaStreamGetCaptureInfo_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaStreamUpdateCaptureDependencies(void *__args)
{
	spdlog::info("Received request: cudaStreamUpdateCaptureDependencies");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaEventCreate(void *__args)
{
	spdlog::info("Received request: cudaEventCreate");

    auto args = (struct cudaEventCreateArg *) __args;

    cudaEvent_t  event;
    cudaError_t err = cudaEventCreate(&event);

    struct cudaEventCreateResponse res { event, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaEventCreateResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaEventCreateWithFlags(void *__args)
{
	spdlog::info("Received request: cudaEventCreateWithFlags");

    auto args = (struct cudaEventCreateWithFlagsArg *) __args;

    cudaEvent_t  event;
    cudaError_t err = cudaEventCreateWithFlags(&event, args->flags);

    struct cudaEventCreateWithFlagsResponse res { event, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaEventCreateWithFlagsResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaEventRecord(void *__args)
{
	spdlog::info("Received request: cudaEventRecord");

    auto args = (struct cudaEventRecordArg *) __args;
    cudaError_t err = cudaEventRecord(
		args->event,
		args->stream

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaEventRecordWithFlags(void *__args)
{
	spdlog::info("Received request: cudaEventRecordWithFlags");

    auto args = (struct cudaEventRecordWithFlagsArg *) __args;
    cudaError_t err = cudaEventRecordWithFlags(
		args->event,
		args->stream,
		args->flags

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaEventQuery(void *__args)
{
	spdlog::info("Received request: cudaEventQuery");

    auto args = (struct cudaEventQueryArg *) __args;
    cudaError_t err = cudaEventQuery(
		args->event

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaEventSynchronize(void *__args)
{
	spdlog::info("Received request: cudaEventSynchronize");

    auto args = (struct cudaEventSynchronizeArg *) __args;
    cudaError_t err = cudaEventSynchronize(
		args->event

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaEventDestroy(void *__args)
{
	spdlog::info("Received request: cudaEventDestroy");

    auto args = (struct cudaEventDestroyArg *) __args;
    cudaError_t err = cudaEventDestroy(
		args->event

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaEventElapsedTime(void *__args)
{
	spdlog::info("Received request: cudaEventElapsedTime");

    auto args = (struct cudaEventElapsedTimeArg *) __args;

    float  ms;
    cudaError_t err = cudaEventElapsedTime(&ms, args->start, args->end);

    struct cudaEventElapsedTimeResponse res { ms, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaEventElapsedTimeResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaImportExternalMemory(void *__args)
{
	spdlog::info("Received request: cudaImportExternalMemory");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaExternalMemoryGetMappedBuffer(void *__args)
{
	spdlog::info("Received request: cudaExternalMemoryGetMappedBuffer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaExternalMemoryGetMappedMipmappedArray(void *__args)
{
	spdlog::info("Received request: cudaExternalMemoryGetMappedMipmappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDestroyExternalMemory(void *__args)
{
	spdlog::info("Received request: cudaDestroyExternalMemory");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaImportExternalSemaphore(void *__args)
{
	spdlog::info("Received request: cudaImportExternalSemaphore");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaSignalExternalSemaphoresAsync_v2(void *__args)
{
	spdlog::info("Received request: cudaSignalExternalSemaphoresAsync_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaWaitExternalSemaphoresAsync_v2(void *__args)
{
	spdlog::info("Received request: cudaWaitExternalSemaphoresAsync_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDestroyExternalSemaphore(void *__args)
{
	spdlog::info("Received request: cudaDestroyExternalSemaphore");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaLaunchCooperativeKernel(void *__args)
{
	spdlog::info("Received request: cudaLaunchCooperativeKernel");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaLaunchCooperativeKernelMultiDevice(void *__args)
{
	spdlog::info("Received request: cudaLaunchCooperativeKernelMultiDevice");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaFuncSetCacheConfig(void *__args)
{
	spdlog::info("Received request: cudaFuncSetCacheConfig");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaFuncSetSharedMemConfig(void *__args)
{
	spdlog::info("Received request: cudaFuncSetSharedMemConfig");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaFuncGetAttributes(void *__args)
{
	spdlog::info("Received request: cudaFuncGetAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaFuncSetAttribute(void *__args)
{
	spdlog::info("Received request: cudaFuncSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaSetDoubleForDevice(void *__args)
{
	spdlog::info("Received request: cudaSetDoubleForDevice");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaSetDoubleForHost(void *__args)
{
	spdlog::info("Received request: cudaSetDoubleForHost");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaLaunchHostFunc(void *__args)
{
	spdlog::info("Received request: cudaLaunchHostFunc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaOccupancyMaxActiveBlocksPerMultiprocessor(void *__args)
{
	spdlog::info("Received request: cudaOccupancyMaxActiveBlocksPerMultiprocessor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaOccupancyAvailableDynamicSMemPerBlock(void *__args)
{
	spdlog::info("Received request: cudaOccupancyAvailableDynamicSMemPerBlock");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *__args)
{
	spdlog::info("Received request: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMallocManaged(void *__args)
{
	spdlog::info("Received request: cudaMallocManaged");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMallocHost(void *__args)
{
	spdlog::info("Received request: cudaMallocHost");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMallocPitch(void *__args)
{
	spdlog::info("Received request: cudaMallocPitch");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMallocArray(void *__args)
{
	spdlog::info("Received request: cudaMallocArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaFree(void *__args)
{
	spdlog::info("Received request: cudaFree");

    auto args = (struct cudaFreeArg *) __args;
    cudaError_t err = cudaFree(
		args->devPtr

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaFreeHost(void *__args)
{
	spdlog::info("Received request: cudaFreeHost");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaFreeArray(void *__args)
{
	spdlog::info("Received request: cudaFreeArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaFreeMipmappedArray(void *__args)
{
	spdlog::info("Received request: cudaFreeMipmappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaHostAlloc(void *__args)
{
	spdlog::info("Received request: cudaHostAlloc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaHostRegister(void *__args)
{
	spdlog::info("Received request: cudaHostRegister");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaHostUnregister(void *__args)
{
	spdlog::info("Received request: cudaHostUnregister");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaHostGetDevicePointer(void *__args)
{
	spdlog::info("Received request: cudaHostGetDevicePointer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaHostGetFlags(void *__args)
{
	spdlog::info("Received request: cudaHostGetFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMalloc3D(void *__args)
{
	spdlog::info("Received request: cudaMalloc3D");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMalloc3DArray(void *__args)
{
	spdlog::info("Received request: cudaMalloc3DArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMallocMipmappedArray(void *__args)
{
	spdlog::info("Received request: cudaMallocMipmappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetMipmappedArrayLevel(void *__args)
{
	spdlog::info("Received request: cudaGetMipmappedArrayLevel");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy3D(void *__args)
{
	spdlog::info("Received request: cudaMemcpy3D");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy3DPeer(void *__args)
{
	spdlog::info("Received request: cudaMemcpy3DPeer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy3DAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpy3DAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy3DPeerAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpy3DPeerAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemGetInfo(void *__args)
{
	spdlog::info("Received request: cudaMemGetInfo");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaArrayGetInfo(void *__args)
{
	spdlog::info("Received request: cudaArrayGetInfo");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaArrayGetPlane(void *__args)
{
	spdlog::info("Received request: cudaArrayGetPlane");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaArrayGetSparseProperties(void *__args)
{
	spdlog::info("Received request: cudaArrayGetSparseProperties");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMipmappedArrayGetSparseProperties(void *__args)
{
	spdlog::info("Received request: cudaMipmappedArrayGetSparseProperties");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyPeer(void *__args)
{
	spdlog::info("Received request: cudaMemcpyPeer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy2D(void *__args)
{
	spdlog::info("Received request: cudaMemcpy2D");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy2DToArray(void *__args)
{
	spdlog::info("Received request: cudaMemcpy2DToArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy2DFromArray(void *__args)
{
	spdlog::info("Received request: cudaMemcpy2DFromArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy2DArrayToArray(void *__args)
{
	spdlog::info("Received request: cudaMemcpy2DArrayToArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyToSymbol(void *__args)
{
	spdlog::info("Received request: cudaMemcpyToSymbol");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyFromSymbol(void *__args)
{
	spdlog::info("Received request: cudaMemcpyFromSymbol");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpyAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyPeerAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpyPeerAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy2DAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpy2DAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy2DToArrayAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpy2DToArrayAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpy2DFromArrayAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpy2DFromArrayAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyToSymbolAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpyToSymbolAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyFromSymbolAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpyFromSymbolAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemset(void *__args)
{
	spdlog::info("Received request: cudaMemset");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemset2D(void *__args)
{
	spdlog::info("Received request: cudaMemset2D");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemset3D(void *__args)
{
	spdlog::info("Received request: cudaMemset3D");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemsetAsync(void *__args)
{
	spdlog::info("Received request: cudaMemsetAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemset2DAsync(void *__args)
{
	spdlog::info("Received request: cudaMemset2DAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemset3DAsync(void *__args)
{
	spdlog::info("Received request: cudaMemset3DAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetSymbolAddress(void *__args)
{
	spdlog::info("Received request: cudaGetSymbolAddress");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetSymbolSize(void *__args)
{
	spdlog::info("Received request: cudaGetSymbolSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPrefetchAsync(void *__args)
{
	spdlog::info("Received request: cudaMemPrefetchAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemAdvise(void *__args)
{
	spdlog::info("Received request: cudaMemAdvise");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemRangeGetAttribute(void *__args)
{
	spdlog::info("Received request: cudaMemRangeGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemRangeGetAttributes(void *__args)
{
	spdlog::info("Received request: cudaMemRangeGetAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyToArray(void *__args)
{
	spdlog::info("Received request: cudaMemcpyToArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyFromArray(void *__args)
{
	spdlog::info("Received request: cudaMemcpyFromArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyArrayToArray(void *__args)
{
	spdlog::info("Received request: cudaMemcpyArrayToArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyToArrayAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpyToArrayAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemcpyFromArrayAsync(void *__args)
{
	spdlog::info("Received request: cudaMemcpyFromArrayAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMallocAsync(void *__args)
{
	spdlog::info("Received request: cudaMallocAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaFreeAsync(void *__args)
{
	spdlog::info("Received request: cudaFreeAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolTrimTo(void *__args)
{
	spdlog::info("Received request: cudaMemPoolTrimTo");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolSetAttribute(void *__args)
{
	spdlog::info("Received request: cudaMemPoolSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolGetAttribute(void *__args)
{
	spdlog::info("Received request: cudaMemPoolGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolSetAccess(void *__args)
{
	spdlog::info("Received request: cudaMemPoolSetAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolGetAccess(void *__args)
{
	spdlog::info("Received request: cudaMemPoolGetAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolCreate(void *__args)
{
	spdlog::info("Received request: cudaMemPoolCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolDestroy(void *__args)
{
	spdlog::info("Received request: cudaMemPoolDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMallocFromPoolAsync(void *__args)
{
	spdlog::info("Received request: cudaMallocFromPoolAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolExportToShareableHandle(void *__args)
{
	spdlog::info("Received request: cudaMemPoolExportToShareableHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolImportFromShareableHandle(void *__args)
{
	spdlog::info("Received request: cudaMemPoolImportFromShareableHandle");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolExportPointer(void *__args)
{
	spdlog::info("Received request: cudaMemPoolExportPointer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaMemPoolImportPointer(void *__args)
{
	spdlog::info("Received request: cudaMemPoolImportPointer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaPointerGetAttributes(void *__args)
{
	spdlog::info("Received request: cudaPointerGetAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceCanAccessPeer(void *__args)
{
	spdlog::info("Received request: cudaDeviceCanAccessPeer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceEnablePeerAccess(void *__args)
{
	spdlog::info("Received request: cudaDeviceEnablePeerAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceDisablePeerAccess(void *__args)
{
	spdlog::info("Received request: cudaDeviceDisablePeerAccess");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphicsUnregisterResource(void *__args)
{
	spdlog::info("Received request: cudaGraphicsUnregisterResource");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphicsResourceSetMapFlags(void *__args)
{
	spdlog::info("Received request: cudaGraphicsResourceSetMapFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphicsMapResources(void *__args)
{
	spdlog::info("Received request: cudaGraphicsMapResources");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphicsUnmapResources(void *__args)
{
	spdlog::info("Received request: cudaGraphicsUnmapResources");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphicsResourceGetMappedPointer(void *__args)
{
	spdlog::info("Received request: cudaGraphicsResourceGetMappedPointer");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphicsSubResourceGetMappedArray(void *__args)
{
	spdlog::info("Received request: cudaGraphicsSubResourceGetMappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphicsResourceGetMappedMipmappedArray(void *__args)
{
	spdlog::info("Received request: cudaGraphicsResourceGetMappedMipmappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaBindTexture(void *__args)
{
	spdlog::info("Received request: cudaBindTexture");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaBindTexture2D(void *__args)
{
	spdlog::info("Received request: cudaBindTexture2D");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaBindTextureToArray(void *__args)
{
	spdlog::info("Received request: cudaBindTextureToArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaBindTextureToMipmappedArray(void *__args)
{
	spdlog::info("Received request: cudaBindTextureToMipmappedArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaUnbindTexture(void *__args)
{
	spdlog::info("Received request: cudaUnbindTexture");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetTextureAlignmentOffset(void *__args)
{
	spdlog::info("Received request: cudaGetTextureAlignmentOffset");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetTextureReference(void *__args)
{
	spdlog::info("Received request: cudaGetTextureReference");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaBindSurfaceToArray(void *__args)
{
	spdlog::info("Received request: cudaBindSurfaceToArray");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetSurfaceReference(void *__args)
{
	spdlog::info("Received request: cudaGetSurfaceReference");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetChannelDesc(void *__args)
{
	spdlog::info("Received request: cudaGetChannelDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaCreateChannelDesc(void *__args)
{
	spdlog::info("Received request: cudaCreateChannelDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaCreateTextureObject(void *__args)
{
	spdlog::info("Received request: cudaCreateTextureObject");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDestroyTextureObject(void *__args)
{
	spdlog::info("Received request: cudaDestroyTextureObject");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetTextureObjectResourceDesc(void *__args)
{
	spdlog::info("Received request: cudaGetTextureObjectResourceDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetTextureObjectTextureDesc(void *__args)
{
	spdlog::info("Received request: cudaGetTextureObjectTextureDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetTextureObjectResourceViewDesc(void *__args)
{
	spdlog::info("Received request: cudaGetTextureObjectResourceViewDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaCreateSurfaceObject(void *__args)
{
	spdlog::info("Received request: cudaCreateSurfaceObject");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDestroySurfaceObject(void *__args)
{
	spdlog::info("Received request: cudaDestroySurfaceObject");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetSurfaceObjectResourceDesc(void *__args)
{
	spdlog::info("Received request: cudaGetSurfaceObjectResourceDesc");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDriverGetVersion(void *__args)
{
	spdlog::info("Received request: cudaDriverGetVersion");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaRuntimeGetVersion(void *__args)
{
	spdlog::info("Received request: cudaRuntimeGetVersion");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphCreate(void *__args)
{
	spdlog::info("Received request: cudaGraphCreate");

    auto args = (struct cudaGraphCreateArg *) __args;

    cudaGraph_t  pGraph;
    cudaError_t err = cudaGraphCreate(&pGraph, args->flags);

    struct cudaGraphCreateResponse res { pGraph, err };
    while(!send_ipc->send((void *) &res, sizeof(struct cudaGraphCreateResponse))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaGraphAddKernelNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddKernelNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphKernelNodeGetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphKernelNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphKernelNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphKernelNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphKernelNodeCopyAttributes(void *__args)
{
	spdlog::info("Received request: cudaGraphKernelNodeCopyAttributes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphKernelNodeGetAttribute(void *__args)
{
	spdlog::info("Received request: cudaGraphKernelNodeGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphKernelNodeSetAttribute(void *__args)
{
	spdlog::info("Received request: cudaGraphKernelNodeSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddMemcpyNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddMemcpyNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddMemcpyNodeToSymbol(void *__args)
{
	spdlog::info("Received request: cudaGraphAddMemcpyNodeToSymbol");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddMemcpyNodeFromSymbol(void *__args)
{
	spdlog::info("Received request: cudaGraphAddMemcpyNodeFromSymbol");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddMemcpyNode1D(void *__args)
{
	spdlog::info("Received request: cudaGraphAddMemcpyNode1D");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphMemcpyNodeGetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphMemcpyNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphMemcpyNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphMemcpyNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphMemcpyNodeSetParamsToSymbol(void *__args)
{
	spdlog::info("Received request: cudaGraphMemcpyNodeSetParamsToSymbol");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphMemcpyNodeSetParamsFromSymbol(void *__args)
{
	spdlog::info("Received request: cudaGraphMemcpyNodeSetParamsFromSymbol");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphMemcpyNodeSetParams1D(void *__args)
{
	spdlog::info("Received request: cudaGraphMemcpyNodeSetParams1D");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddMemsetNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddMemsetNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphMemsetNodeGetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphMemsetNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphMemsetNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphMemsetNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddHostNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddHostNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphHostNodeGetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphHostNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphHostNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphHostNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddChildGraphNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddChildGraphNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphChildGraphNodeGetGraph(void *__args)
{
	spdlog::info("Received request: cudaGraphChildGraphNodeGetGraph");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddEmptyNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddEmptyNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddEventRecordNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddEventRecordNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphEventRecordNodeGetEvent(void *__args)
{
	spdlog::info("Received request: cudaGraphEventRecordNodeGetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphEventRecordNodeSetEvent(void *__args)
{
	spdlog::info("Received request: cudaGraphEventRecordNodeSetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddEventWaitNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddEventWaitNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphEventWaitNodeGetEvent(void *__args)
{
	spdlog::info("Received request: cudaGraphEventWaitNodeGetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphEventWaitNodeSetEvent(void *__args)
{
	spdlog::info("Received request: cudaGraphEventWaitNodeSetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddExternalSemaphoresSignalNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddExternalSemaphoresSignalNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExternalSemaphoresSignalNodeGetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExternalSemaphoresSignalNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExternalSemaphoresSignalNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExternalSemaphoresSignalNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddExternalSemaphoresWaitNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddExternalSemaphoresWaitNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExternalSemaphoresWaitNodeGetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExternalSemaphoresWaitNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExternalSemaphoresWaitNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExternalSemaphoresWaitNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddMemAllocNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddMemAllocNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphMemAllocNodeGetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphMemAllocNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddMemFreeNode(void *__args)
{
	spdlog::info("Received request: cudaGraphAddMemFreeNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphMemFreeNodeGetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphMemFreeNodeGetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceGraphMemTrim(void *__args)
{
	spdlog::info("Received request: cudaDeviceGraphMemTrim");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceGetGraphMemAttribute(void *__args)
{
	spdlog::info("Received request: cudaDeviceGetGraphMemAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaDeviceSetGraphMemAttribute(void *__args)
{
	spdlog::info("Received request: cudaDeviceSetGraphMemAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphClone(void *__args)
{
	spdlog::info("Received request: cudaGraphClone");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphNodeFindInClone(void *__args)
{
	spdlog::info("Received request: cudaGraphNodeFindInClone");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphNodeGetType(void *__args)
{
	spdlog::info("Received request: cudaGraphNodeGetType");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphGetNodes(void *__args)
{
	spdlog::info("Received request: cudaGraphGetNodes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphGetRootNodes(void *__args)
{
	spdlog::info("Received request: cudaGraphGetRootNodes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphGetEdges(void *__args)
{
	spdlog::info("Received request: cudaGraphGetEdges");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphNodeGetDependencies(void *__args)
{
	spdlog::info("Received request: cudaGraphNodeGetDependencies");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphNodeGetDependentNodes(void *__args)
{
	spdlog::info("Received request: cudaGraphNodeGetDependentNodes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphAddDependencies(void *__args)
{
	spdlog::info("Received request: cudaGraphAddDependencies");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphRemoveDependencies(void *__args)
{
	spdlog::info("Received request: cudaGraphRemoveDependencies");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphDestroyNode(void *__args)
{
	spdlog::info("Received request: cudaGraphDestroyNode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphInstantiate(void *__args)
{
	spdlog::info("Received request: cudaGraphInstantiate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphInstantiateWithFlags(void *__args)
{
	spdlog::info("Received request: cudaGraphInstantiateWithFlags");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecKernelNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExecKernelNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecMemcpyNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExecMemcpyNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecMemcpyNodeSetParamsToSymbol(void *__args)
{
	spdlog::info("Received request: cudaGraphExecMemcpyNodeSetParamsToSymbol");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecMemcpyNodeSetParamsFromSymbol(void *__args)
{
	spdlog::info("Received request: cudaGraphExecMemcpyNodeSetParamsFromSymbol");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecMemcpyNodeSetParams1D(void *__args)
{
	spdlog::info("Received request: cudaGraphExecMemcpyNodeSetParams1D");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecMemsetNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExecMemsetNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecHostNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExecHostNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecChildGraphNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExecChildGraphNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecEventRecordNodeSetEvent(void *__args)
{
	spdlog::info("Received request: cudaGraphExecEventRecordNodeSetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecEventWaitNodeSetEvent(void *__args)
{
	spdlog::info("Received request: cudaGraphExecEventWaitNodeSetEvent");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecExternalSemaphoresSignalNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExecExternalSemaphoresSignalNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecExternalSemaphoresWaitNodeSetParams(void *__args)
{
	spdlog::info("Received request: cudaGraphExecExternalSemaphoresWaitNodeSetParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecUpdate(void *__args)
{
	spdlog::info("Received request: cudaGraphExecUpdate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphUpload(void *__args)
{
	spdlog::info("Received request: cudaGraphUpload");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphLaunch(void *__args)
{
	spdlog::info("Received request: cudaGraphLaunch");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphExecDestroy(void *__args)
{
	spdlog::info("Received request: cudaGraphExecDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphDestroy(void *__args)
{
	spdlog::info("Received request: cudaGraphDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphDebugDotPrint(void *__args)
{
	spdlog::info("Received request: cudaGraphDebugDotPrint");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaUserObjectCreate(void *__args)
{
	spdlog::info("Received request: cudaUserObjectCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaUserObjectRetain(void *__args)
{
	spdlog::info("Received request: cudaUserObjectRetain");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaUserObjectRelease(void *__args)
{
	spdlog::info("Received request: cudaUserObjectRelease");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphRetainUserObject(void *__args)
{
	spdlog::info("Received request: cudaGraphRetainUserObject");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGraphReleaseUserObject(void *__args)
{
	spdlog::info("Received request: cudaGraphReleaseUserObject");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetDriverEntryPoint(void *__args)
{
	spdlog::info("Received request: cudaGetDriverEntryPoint");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetExportTable(void *__args)
{
	spdlog::info("Received request: cudaGetExportTable");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaGetFuncBySymbol(void *__args)
{
	spdlog::info("Received request: cudaGetFuncBySymbol");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetVersion(void *__args)
{
	spdlog::info("Received request: cudnnGetVersion");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetMaxDeviceVersion(void *__args)
{
	spdlog::info("Received request: cudnnGetMaxDeviceVersion");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetCudartVersion(void *__args)
{
	spdlog::info("Received request: cudnnGetCudartVersion");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetErrorString(void *__args)
{
	spdlog::info("Received request: cudnnGetErrorString");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnQueryRuntimeError(void *__args)
{
	spdlog::info("Received request: cudnnQueryRuntimeError");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetProperty(void *__args)
{
	spdlog::info("Received request: cudnnGetProperty");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreate(void *__args)
{
	spdlog::info("Received request: cudnnCreate");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroy(void *__args)
{
	spdlog::info("Received request: cudnnDestroy");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetStream(void *__args)
{
	spdlog::info("Received request: cudnnSetStream");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetStream(void *__args)
{
	spdlog::info("Received request: cudnnGetStream");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetTensor4dDescriptorEx(void *__args)
{
	spdlog::info("Received request: cudnnSetTensor4dDescriptorEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetTensor4dDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetTensor4dDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetTensorNdDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetTensorNdDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetTensorNdDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetTensorNdDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetTensorSizeInBytes(void *__args)
{
	spdlog::info("Received request: cudnnGetTensorSizeInBytes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnAddTensor(void *__args)
{
	spdlog::info("Received request: cudnnAddTensor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateOpTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateOpTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetOpTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetOpTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetOpTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetOpTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyOpTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyOpTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnOpTensor(void *__args)
{
	spdlog::info("Received request: cudnnOpTensor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateReduceTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateReduceTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetReduceTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetReduceTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetReduceTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetReduceTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyReduceTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyReduceTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetReductionIndicesSize(void *__args)
{
	spdlog::info("Received request: cudnnGetReductionIndicesSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetReductionWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetReductionWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnReduceTensor(void *__args)
{
	spdlog::info("Received request: cudnnReduceTensor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetTensor(void *__args)
{
	spdlog::info("Received request: cudnnSetTensor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnScaleTensor(void *__args)
{
	spdlog::info("Received request: cudnnScaleTensor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateFilterDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateFilterDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetFilterSizeInBytes(void *__args)
{
	spdlog::info("Received request: cudnnGetFilterSizeInBytes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyFilterDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyFilterDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSoftmaxForward(void *__args)
{
	spdlog::info("Received request: cudnnSoftmaxForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreatePoolingDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreatePoolingDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetPooling2dDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetPooling2dDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetPooling2dDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetPooling2dDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetPoolingNdDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetPoolingNdDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetPoolingNdDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetPoolingNdDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetPoolingNdForwardOutputDim(void *__args)
{
	spdlog::info("Received request: cudnnGetPoolingNdForwardOutputDim");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetPooling2dForwardOutputDim(void *__args)
{
	spdlog::info("Received request: cudnnGetPooling2dForwardOutputDim");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyPoolingDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyPoolingDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnPoolingForward(void *__args)
{
	spdlog::info("Received request: cudnnPoolingForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateActivationDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateActivationDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetActivationDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetActivationDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetActivationDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetActivationDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetActivationDescriptorSwishBeta(void *__args)
{
	spdlog::info("Received request: cudnnSetActivationDescriptorSwishBeta");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetActivationDescriptorSwishBeta(void *__args)
{
	spdlog::info("Received request: cudnnGetActivationDescriptorSwishBeta");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyActivationDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyActivationDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnActivationForward(void *__args)
{
	spdlog::info("Received request: cudnnActivationForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateLRNDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateLRNDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetLRNDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetLRNDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetLRNDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetLRNDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyLRNDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyLRNDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnLRNCrossChannelForward(void *__args)
{
	spdlog::info("Received request: cudnnLRNCrossChannelForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDivisiveNormalizationForward(void *__args)
{
	spdlog::info("Received request: cudnnDivisiveNormalizationForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDeriveBNTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDeriveBNTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBatchNormalizationForwardInference(void *__args)
{
	spdlog::info("Received request: cudnnBatchNormalizationForwardInference");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDeriveNormTensorDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDeriveNormTensorDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnNormalizationForwardInference(void *__args)
{
	spdlog::info("Received request: cudnnNormalizationForwardInference");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateDropoutDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateDropoutDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyDropoutDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyDropoutDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDropoutGetStatesSize(void *__args)
{
	spdlog::info("Received request: cudnnDropoutGetStatesSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDropoutGetReserveSpaceSize(void *__args)
{
	spdlog::info("Received request: cudnnDropoutGetReserveSpaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetDropoutDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetDropoutDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRestoreDropoutDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnRestoreDropoutDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetDropoutDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetDropoutDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDropoutForward(void *__args)
{
	spdlog::info("Received request: cudnnDropoutForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateAlgorithmDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateAlgorithmDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetAlgorithmDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetAlgorithmDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetAlgorithmDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetAlgorithmDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCopyAlgorithmDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCopyAlgorithmDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyAlgorithmDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyAlgorithmDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetAlgorithmSpaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetAlgorithmSpaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSaveAlgorithm(void *__args)
{
	spdlog::info("Received request: cudnnSaveAlgorithm");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRestoreAlgorithm(void *__args)
{
	spdlog::info("Received request: cudnnRestoreAlgorithm");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetCallback(void *__args)
{
	spdlog::info("Received request: cudnnSetCallback");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetCallback(void *__args)
{
	spdlog::info("Received request: cudnnGetCallback");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnOpsInferVersionCheck(void *__args)
{
	spdlog::info("Received request: cudnnOpsInferVersionCheck");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnPoolingBackward(void *__args)
{
	spdlog::info("Received request: cudnnPoolingBackward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnActivationBackward(void *__args)
{
	spdlog::info("Received request: cudnnActivationBackward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnLRNCrossChannelBackward(void *__args)
{
	spdlog::info("Received request: cudnnLRNCrossChannelBackward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDivisiveNormalizationBackward(void *__args)
{
	spdlog::info("Received request: cudnnDivisiveNormalizationBackward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetBatchNormalizationBackwardExWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetBatchNormalizationBackwardExWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetBatchNormalizationTrainingExReserveSpaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBatchNormalizationForwardTraining(void *__args)
{
	spdlog::info("Received request: cudnnBatchNormalizationForwardTraining");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBatchNormalizationForwardTrainingEx(void *__args)
{
	spdlog::info("Received request: cudnnBatchNormalizationForwardTrainingEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBatchNormalizationBackward(void *__args)
{
	spdlog::info("Received request: cudnnBatchNormalizationBackward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBatchNormalizationBackwardEx(void *__args)
{
	spdlog::info("Received request: cudnnBatchNormalizationBackwardEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetNormalizationForwardTrainingWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetNormalizationForwardTrainingWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetNormalizationBackwardWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetNormalizationBackwardWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetNormalizationTrainingReserveSpaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetNormalizationTrainingReserveSpaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnNormalizationForwardTraining(void *__args)
{
	spdlog::info("Received request: cudnnNormalizationForwardTraining");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnNormalizationBackward(void *__args)
{
	spdlog::info("Received request: cudnnNormalizationBackward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDropoutBackward(void *__args)
{
	spdlog::info("Received request: cudnnDropoutBackward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnOpsTrainVersionCheck(void *__args)
{
	spdlog::info("Received request: cudnnOpsTrainVersionCheck");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateRNNDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateRNNDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyRNNDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyRNNDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetRNNDescriptor_v8(void *__args)
{
	spdlog::info("Received request: cudnnSetRNNDescriptor_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNDescriptor_v8(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNDescriptor_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetRNNDescriptor_v6(void *__args)
{
	spdlog::info("Received request: cudnnSetRNNDescriptor_v6");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNDescriptor_v6(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNDescriptor_v6");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetRNNMatrixMathType(void *__args)
{
	spdlog::info("Received request: cudnnSetRNNMatrixMathType");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNMatrixMathType(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNMatrixMathType");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetRNNBiasMode(void *__args)
{
	spdlog::info("Received request: cudnnSetRNNBiasMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNBiasMode(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNBiasMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNSetClip_v8(void *__args)
{
	spdlog::info("Received request: cudnnRNNSetClip_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNGetClip_v8(void *__args)
{
	spdlog::info("Received request: cudnnRNNGetClip_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNSetClip(void *__args)
{
	spdlog::info("Received request: cudnnRNNSetClip");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNGetClip(void *__args)
{
	spdlog::info("Received request: cudnnRNNGetClip");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetRNNProjectionLayers(void *__args)
{
	spdlog::info("Received request: cudnnSetRNNProjectionLayers");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNProjectionLayers(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNProjectionLayers");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreatePersistentRNNPlan(void *__args)
{
	spdlog::info("Received request: cudnnCreatePersistentRNNPlan");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyPersistentRNNPlan(void *__args)
{
	spdlog::info("Received request: cudnnDestroyPersistentRNNPlan");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetPersistentRNNPlan(void *__args)
{
	spdlog::info("Received request: cudnnSetPersistentRNNPlan");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBuildRNNDynamic(void *__args)
{
	spdlog::info("Received request: cudnnBuildRNNDynamic");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNTrainingReserveSize(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNTrainingReserveSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNTempSpaceSizes(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNTempSpaceSizes");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNParamsSize(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNParamsSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNWeightSpaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNWeightSpaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNLinLayerMatrixParams(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNLinLayerMatrixParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNLinLayerBiasParams(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNLinLayerBiasParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNWeightParams(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNWeightParams");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNForwardInference(void *__args)
{
	spdlog::info("Received request: cudnnRNNForwardInference");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetRNNPaddingMode(void *__args)
{
	spdlog::info("Received request: cudnnSetRNNPaddingMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNPaddingMode(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNPaddingMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateRNNDataDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateRNNDataDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyRNNDataDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyRNNDataDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetRNNDataDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetRNNDataDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNDataDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNDataDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNForwardInferenceEx(void *__args)
{
	spdlog::info("Received request: cudnnRNNForwardInferenceEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNForward(void *__args)
{
	spdlog::info("Received request: cudnnRNNForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetRNNAlgorithmDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetRNNAlgorithmDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNForwardInferenceAlgorithmMaxCount(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNForwardInferenceAlgorithmMaxCount");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateSeqDataDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateSeqDataDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroySeqDataDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroySeqDataDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetSeqDataDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetSeqDataDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetSeqDataDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetSeqDataDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateAttnDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateAttnDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyAttnDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyAttnDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetAttnDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetAttnDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetAttnDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetAttnDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetMultiHeadAttnBuffers(void *__args)
{
	spdlog::info("Received request: cudnnGetMultiHeadAttnBuffers");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetMultiHeadAttnWeights(void *__args)
{
	spdlog::info("Received request: cudnnGetMultiHeadAttnWeights");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnMultiHeadAttnForward(void *__args)
{
	spdlog::info("Received request: cudnnMultiHeadAttnForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnAdvInferVersionCheck(void *__args)
{
	spdlog::info("Received request: cudnnAdvInferVersionCheck");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNForwardTraining(void *__args)
{
	spdlog::info("Received request: cudnnRNNForwardTraining");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNBackwardData(void *__args)
{
	spdlog::info("Received request: cudnnRNNBackwardData");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNBackwardData_v8(void *__args)
{
	spdlog::info("Received request: cudnnRNNBackwardData_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNBackwardWeights(void *__args)
{
	spdlog::info("Received request: cudnnRNNBackwardWeights");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNBackwardWeights_v8(void *__args)
{
	spdlog::info("Received request: cudnnRNNBackwardWeights_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNForwardTrainingEx(void *__args)
{
	spdlog::info("Received request: cudnnRNNForwardTrainingEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNBackwardDataEx(void *__args)
{
	spdlog::info("Received request: cudnnRNNBackwardDataEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnRNNBackwardWeightsEx(void *__args)
{
	spdlog::info("Received request: cudnnRNNBackwardWeightsEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNForwardTrainingAlgorithmMaxCount(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNForwardTrainingAlgorithmMaxCount");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNBackwardDataAlgorithmMaxCount(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNBackwardDataAlgorithmMaxCount");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetRNNBackwardWeightsAlgorithmMaxCount(void *__args)
{
	spdlog::info("Received request: cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnMultiHeadAttnBackwardData(void *__args)
{
	spdlog::info("Received request: cudnnMultiHeadAttnBackwardData");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnMultiHeadAttnBackwardWeights(void *__args)
{
	spdlog::info("Received request: cudnnMultiHeadAttnBackwardWeights");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateCTCLossDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateCTCLossDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetCTCLossDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetCTCLossDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetCTCLossDescriptorEx(void *__args)
{
	spdlog::info("Received request: cudnnSetCTCLossDescriptorEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetCTCLossDescriptor_v8(void *__args)
{
	spdlog::info("Received request: cudnnSetCTCLossDescriptor_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetCTCLossDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetCTCLossDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetCTCLossDescriptorEx(void *__args)
{
	spdlog::info("Received request: cudnnGetCTCLossDescriptorEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetCTCLossDescriptor_v8(void *__args)
{
	spdlog::info("Received request: cudnnGetCTCLossDescriptor_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyCTCLossDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyCTCLossDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCTCLoss(void *__args)
{
	spdlog::info("Received request: cudnnCTCLoss");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCTCLoss_v8(void *__args)
{
	spdlog::info("Received request: cudnnCTCLoss_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetCTCLossWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetCTCLossWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetCTCLossWorkspaceSize_v8(void *__args)
{
	spdlog::info("Received request: cudnnGetCTCLossWorkspaceSize_v8");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnAdvTrainVersionCheck(void *__args)
{
	spdlog::info("Received request: cudnnAdvTrainVersionCheck");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateConvolutionDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnCreateConvolutionDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyConvolutionDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnDestroyConvolutionDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetConvolutionMathType(void *__args)
{
	spdlog::info("Received request: cudnnSetConvolutionMathType");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionMathType(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionMathType");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetConvolutionGroupCount(void *__args)
{
	spdlog::info("Received request: cudnnSetConvolutionGroupCount");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionGroupCount(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionGroupCount");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetConvolutionReorderType(void *__args)
{
	spdlog::info("Received request: cudnnSetConvolutionReorderType");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionReorderType(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionReorderType");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetConvolution2dDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetConvolution2dDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolution2dDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolution2dDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetConvolutionNdDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnSetConvolutionNdDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionNdDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionNdDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolution2dForwardOutputDim(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolution2dForwardOutputDim");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionNdForwardOutputDim(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionNdForwardOutputDim");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionForwardAlgorithmMaxCount(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionForwardAlgorithmMaxCount");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnIm2Col(void *__args)
{
	spdlog::info("Received request: cudnnIm2Col");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnReorderFilterAndBias(void *__args)
{
	spdlog::info("Received request: cudnnReorderFilterAndBias");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionForwardWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionForwardWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnConvolutionForward(void *__args)
{
	spdlog::info("Received request: cudnnConvolutionForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnConvolutionBiasActivationForward(void *__args)
{
	spdlog::info("Received request: cudnnConvolutionBiasActivationForward");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionBackwardDataAlgorithmMaxCount(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionBackwardDataWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionBackwardDataWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnConvolutionBackwardData(void *__args)
{
	spdlog::info("Received request: cudnnConvolutionBackwardData");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCnnInferVersionCheck(void *__args)
{
	spdlog::info("Received request: cudnnCnnInferVersionCheck");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetConvolutionBackwardFilterWorkspaceSize(void *__args)
{
	spdlog::info("Received request: cudnnGetConvolutionBackwardFilterWorkspaceSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnConvolutionBackwardFilter(void *__args)
{
	spdlog::info("Received request: cudnnConvolutionBackwardFilter");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnConvolutionBackwardBias(void *__args)
{
	spdlog::info("Received request: cudnnConvolutionBackwardBias");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateFusedOpsConstParamPack(void *__args)
{
	spdlog::info("Received request: cudnnCreateFusedOpsConstParamPack");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyFusedOpsConstParamPack(void *__args)
{
	spdlog::info("Received request: cudnnDestroyFusedOpsConstParamPack");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetFusedOpsConstParamPackAttribute(void *__args)
{
	spdlog::info("Received request: cudnnSetFusedOpsConstParamPackAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetFusedOpsConstParamPackAttribute(void *__args)
{
	spdlog::info("Received request: cudnnGetFusedOpsConstParamPackAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateFusedOpsVariantParamPack(void *__args)
{
	spdlog::info("Received request: cudnnCreateFusedOpsVariantParamPack");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyFusedOpsVariantParamPack(void *__args)
{
	spdlog::info("Received request: cudnnDestroyFusedOpsVariantParamPack");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnSetFusedOpsVariantParamPackAttribute(void *__args)
{
	spdlog::info("Received request: cudnnSetFusedOpsVariantParamPackAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnGetFusedOpsVariantParamPackAttribute(void *__args)
{
	spdlog::info("Received request: cudnnGetFusedOpsVariantParamPackAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCreateFusedOpsPlan(void *__args)
{
	spdlog::info("Received request: cudnnCreateFusedOpsPlan");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnDestroyFusedOpsPlan(void *__args)
{
	spdlog::info("Received request: cudnnDestroyFusedOpsPlan");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnMakeFusedOpsPlan(void *__args)
{
	spdlog::info("Received request: cudnnMakeFusedOpsPlan");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnFusedOpsExecute(void *__args)
{
	spdlog::info("Received request: cudnnFusedOpsExecute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnCnnTrainVersionCheck(void *__args)
{
	spdlog::info("Received request: cudnnCnnTrainVersionCheck");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBackendCreateDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnBackendCreateDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBackendDestroyDescriptor(void *__args)
{
	spdlog::info("Received request: cudnnBackendDestroyDescriptor");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBackendInitialize(void *__args)
{
	spdlog::info("Received request: cudnnBackendInitialize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBackendFinalize(void *__args)
{
	spdlog::info("Received request: cudnnBackendFinalize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBackendSetAttribute(void *__args)
{
	spdlog::info("Received request: cudnnBackendSetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBackendGetAttribute(void *__args)
{
	spdlog::info("Received request: cudnnBackendGetAttribute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudnnBackendExecute(void *__args)
{
	spdlog::info("Received request: cudnnBackendExecute");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCreate_v2(void *__args)
{
	spdlog::info("Received request: cublasCreate_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDestroy_v2(void *__args)
{
	spdlog::info("Received request: cublasDestroy_v2");

    auto args = (struct cublasDestroy_v2Arg *) __args;
    cublasStatus_t err = cublasDestroy_v2(
		args->handle

    );

    while(!send_ipc->send((void *) &err, sizeof(cublasStatus_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cublasGetVersion_v2(void *__args)
{
	spdlog::info("Received request: cublasGetVersion_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetProperty(void *__args)
{
	spdlog::info("Received request: cublasGetProperty");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetCudartVersion(void *__args)
{
	spdlog::info("Received request: cublasGetCudartVersion");

    auto args = (struct cublasGetCudartVersionArg *) __args;
    size_t err = cublasGetCudartVersion(

    );

    while(!send_ipc->send((void *) &err, sizeof(size_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cublasSetWorkspace_v2(void *__args)
{
	spdlog::info("Received request: cublasSetWorkspace_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetStream_v2(void *__args)
{
	spdlog::info("Received request: cublasSetStream_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetStream_v2(void *__args)
{
	spdlog::info("Received request: cublasGetStream_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetPointerMode_v2(void *__args)
{
	spdlog::info("Received request: cublasGetPointerMode_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetPointerMode_v2(void *__args)
{
	spdlog::info("Received request: cublasSetPointerMode_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetAtomicsMode(void *__args)
{
	spdlog::info("Received request: cublasGetAtomicsMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetAtomicsMode(void *__args)
{
	spdlog::info("Received request: cublasSetAtomicsMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetMathMode(void *__args)
{
	spdlog::info("Received request: cublasGetMathMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetMathMode(void *__args)
{
	spdlog::info("Received request: cublasSetMathMode");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetSmCountTarget(void *__args)
{
	spdlog::info("Received request: cublasGetSmCountTarget");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetSmCountTarget(void *__args)
{
	spdlog::info("Received request: cublasSetSmCountTarget");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetStatusName(void *__args)
{
	spdlog::info("Received request: cublasGetStatusName");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetStatusString(void *__args)
{
	spdlog::info("Received request: cublasGetStatusString");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasLoggerConfigure(void *__args)
{
	spdlog::info("Received request: cublasLoggerConfigure");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetLoggerCallback(void *__args)
{
	spdlog::info("Received request: cublasSetLoggerCallback");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetLoggerCallback(void *__args)
{
	spdlog::info("Received request: cublasGetLoggerCallback");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetVector(void *__args)
{
	spdlog::info("Received request: cublasSetVector");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetVector(void *__args)
{
	spdlog::info("Received request: cublasGetVector");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetMatrix(void *__args)
{
	spdlog::info("Received request: cublasSetMatrix");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetMatrix(void *__args)
{
	spdlog::info("Received request: cublasGetMatrix");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetVectorAsync(void *__args)
{
	spdlog::info("Received request: cublasSetVectorAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetVectorAsync(void *__args)
{
	spdlog::info("Received request: cublasGetVectorAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSetMatrixAsync(void *__args)
{
	spdlog::info("Received request: cublasSetMatrixAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGetMatrixAsync(void *__args)
{
	spdlog::info("Received request: cublasGetMatrixAsync");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasXerbla(void *__args)
{
	spdlog::info("Received request: cublasXerbla");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasNrm2Ex(void *__args)
{
	spdlog::info("Received request: cublasNrm2Ex");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSnrm2_v2(void *__args)
{
	spdlog::info("Received request: cublasSnrm2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDnrm2_v2(void *__args)
{
	spdlog::info("Received request: cublasDnrm2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasScnrm2_v2(void *__args)
{
	spdlog::info("Received request: cublasScnrm2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDznrm2_v2(void *__args)
{
	spdlog::info("Received request: cublasDznrm2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDotEx(void *__args)
{
	spdlog::info("Received request: cublasDotEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDotcEx(void *__args)
{
	spdlog::info("Received request: cublasDotcEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSdot_v2(void *__args)
{
	spdlog::info("Received request: cublasSdot_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDdot_v2(void *__args)
{
	spdlog::info("Received request: cublasDdot_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCdotu_v2(void *__args)
{
	spdlog::info("Received request: cublasCdotu_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCdotc_v2(void *__args)
{
	spdlog::info("Received request: cublasCdotc_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZdotu_v2(void *__args)
{
	spdlog::info("Received request: cublasZdotu_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZdotc_v2(void *__args)
{
	spdlog::info("Received request: cublasZdotc_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasScalEx(void *__args)
{
	spdlog::info("Received request: cublasScalEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSscal_v2(void *__args)
{
	spdlog::info("Received request: cublasSscal_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDscal_v2(void *__args)
{
	spdlog::info("Received request: cublasDscal_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCscal_v2(void *__args)
{
	spdlog::info("Received request: cublasCscal_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsscal_v2(void *__args)
{
	spdlog::info("Received request: cublasCsscal_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZscal_v2(void *__args)
{
	spdlog::info("Received request: cublasZscal_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZdscal_v2(void *__args)
{
	spdlog::info("Received request: cublasZdscal_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasAxpyEx(void *__args)
{
	spdlog::info("Received request: cublasAxpyEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSaxpy_v2(void *__args)
{
	spdlog::info("Received request: cublasSaxpy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDaxpy_v2(void *__args)
{
	spdlog::info("Received request: cublasDaxpy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCaxpy_v2(void *__args)
{
	spdlog::info("Received request: cublasCaxpy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZaxpy_v2(void *__args)
{
	spdlog::info("Received request: cublasZaxpy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCopyEx(void *__args)
{
	spdlog::info("Received request: cublasCopyEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasScopy_v2(void *__args)
{
	spdlog::info("Received request: cublasScopy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDcopy_v2(void *__args)
{
	spdlog::info("Received request: cublasDcopy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCcopy_v2(void *__args)
{
	spdlog::info("Received request: cublasCcopy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZcopy_v2(void *__args)
{
	spdlog::info("Received request: cublasZcopy_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSswap_v2(void *__args)
{
	spdlog::info("Received request: cublasSswap_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDswap_v2(void *__args)
{
	spdlog::info("Received request: cublasDswap_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCswap_v2(void *__args)
{
	spdlog::info("Received request: cublasCswap_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZswap_v2(void *__args)
{
	spdlog::info("Received request: cublasZswap_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSwapEx(void *__args)
{
	spdlog::info("Received request: cublasSwapEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIsamax_v2(void *__args)
{
	spdlog::info("Received request: cublasIsamax_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIdamax_v2(void *__args)
{
	spdlog::info("Received request: cublasIdamax_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIcamax_v2(void *__args)
{
	spdlog::info("Received request: cublasIcamax_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIzamax_v2(void *__args)
{
	spdlog::info("Received request: cublasIzamax_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIamaxEx(void *__args)
{
	spdlog::info("Received request: cublasIamaxEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIsamin_v2(void *__args)
{
	spdlog::info("Received request: cublasIsamin_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIdamin_v2(void *__args)
{
	spdlog::info("Received request: cublasIdamin_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIcamin_v2(void *__args)
{
	spdlog::info("Received request: cublasIcamin_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIzamin_v2(void *__args)
{
	spdlog::info("Received request: cublasIzamin_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasIaminEx(void *__args)
{
	spdlog::info("Received request: cublasIaminEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasAsumEx(void *__args)
{
	spdlog::info("Received request: cublasAsumEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSasum_v2(void *__args)
{
	spdlog::info("Received request: cublasSasum_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDasum_v2(void *__args)
{
	spdlog::info("Received request: cublasDasum_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasScasum_v2(void *__args)
{
	spdlog::info("Received request: cublasScasum_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDzasum_v2(void *__args)
{
	spdlog::info("Received request: cublasDzasum_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSrot_v2(void *__args)
{
	spdlog::info("Received request: cublasSrot_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDrot_v2(void *__args)
{
	spdlog::info("Received request: cublasDrot_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCrot_v2(void *__args)
{
	spdlog::info("Received request: cublasCrot_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsrot_v2(void *__args)
{
	spdlog::info("Received request: cublasCsrot_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZrot_v2(void *__args)
{
	spdlog::info("Received request: cublasZrot_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZdrot_v2(void *__args)
{
	spdlog::info("Received request: cublasZdrot_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasRotEx(void *__args)
{
	spdlog::info("Received request: cublasRotEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSrotg_v2(void *__args)
{
	spdlog::info("Received request: cublasSrotg_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDrotg_v2(void *__args)
{
	spdlog::info("Received request: cublasDrotg_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCrotg_v2(void *__args)
{
	spdlog::info("Received request: cublasCrotg_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZrotg_v2(void *__args)
{
	spdlog::info("Received request: cublasZrotg_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasRotgEx(void *__args)
{
	spdlog::info("Received request: cublasRotgEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSrotm_v2(void *__args)
{
	spdlog::info("Received request: cublasSrotm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDrotm_v2(void *__args)
{
	spdlog::info("Received request: cublasDrotm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasRotmEx(void *__args)
{
	spdlog::info("Received request: cublasRotmEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSrotmg_v2(void *__args)
{
	spdlog::info("Received request: cublasSrotmg_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDrotmg_v2(void *__args)
{
	spdlog::info("Received request: cublasDrotmg_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasRotmgEx(void *__args)
{
	spdlog::info("Received request: cublasRotmgEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgemv_v2(void *__args)
{
	spdlog::info("Received request: cublasSgemv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgemv_v2(void *__args)
{
	spdlog::info("Received request: cublasDgemv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgemv_v2(void *__args)
{
	spdlog::info("Received request: cublasCgemv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgemv_v2(void *__args)
{
	spdlog::info("Received request: cublasZgemv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasSgbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasDgbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasCgbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasZgbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStrmv_v2(void *__args)
{
	spdlog::info("Received request: cublasStrmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtrmv_v2(void *__args)
{
	spdlog::info("Received request: cublasDtrmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtrmv_v2(void *__args)
{
	spdlog::info("Received request: cublasCtrmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtrmv_v2(void *__args)
{
	spdlog::info("Received request: cublasZtrmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasStbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasDtbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasCtbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasZtbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStpmv_v2(void *__args)
{
	spdlog::info("Received request: cublasStpmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtpmv_v2(void *__args)
{
	spdlog::info("Received request: cublasDtpmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtpmv_v2(void *__args)
{
	spdlog::info("Received request: cublasCtpmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtpmv_v2(void *__args)
{
	spdlog::info("Received request: cublasZtpmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStrsv_v2(void *__args)
{
	spdlog::info("Received request: cublasStrsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtrsv_v2(void *__args)
{
	spdlog::info("Received request: cublasDtrsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtrsv_v2(void *__args)
{
	spdlog::info("Received request: cublasCtrsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtrsv_v2(void *__args)
{
	spdlog::info("Received request: cublasZtrsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStpsv_v2(void *__args)
{
	spdlog::info("Received request: cublasStpsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtpsv_v2(void *__args)
{
	spdlog::info("Received request: cublasDtpsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtpsv_v2(void *__args)
{
	spdlog::info("Received request: cublasCtpsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtpsv_v2(void *__args)
{
	spdlog::info("Received request: cublasZtpsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStbsv_v2(void *__args)
{
	spdlog::info("Received request: cublasStbsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtbsv_v2(void *__args)
{
	spdlog::info("Received request: cublasDtbsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtbsv_v2(void *__args)
{
	spdlog::info("Received request: cublasCtbsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtbsv_v2(void *__args)
{
	spdlog::info("Received request: cublasZtbsv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSsymv_v2(void *__args)
{
	spdlog::info("Received request: cublasSsymv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDsymv_v2(void *__args)
{
	spdlog::info("Received request: cublasDsymv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsymv_v2(void *__args)
{
	spdlog::info("Received request: cublasCsymv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZsymv_v2(void *__args)
{
	spdlog::info("Received request: cublasZsymv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasChemv_v2(void *__args)
{
	spdlog::info("Received request: cublasChemv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZhemv_v2(void *__args)
{
	spdlog::info("Received request: cublasZhemv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSsbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasSsbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDsbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasDsbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasChbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasChbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZhbmv_v2(void *__args)
{
	spdlog::info("Received request: cublasZhbmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSspmv_v2(void *__args)
{
	spdlog::info("Received request: cublasSspmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDspmv_v2(void *__args)
{
	spdlog::info("Received request: cublasDspmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasChpmv_v2(void *__args)
{
	spdlog::info("Received request: cublasChpmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZhpmv_v2(void *__args)
{
	spdlog::info("Received request: cublasZhpmv_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSger_v2(void *__args)
{
	spdlog::info("Received request: cublasSger_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDger_v2(void *__args)
{
	spdlog::info("Received request: cublasDger_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgeru_v2(void *__args)
{
	spdlog::info("Received request: cublasCgeru_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgerc_v2(void *__args)
{
	spdlog::info("Received request: cublasCgerc_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgeru_v2(void *__args)
{
	spdlog::info("Received request: cublasZgeru_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgerc_v2(void *__args)
{
	spdlog::info("Received request: cublasZgerc_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSsyr_v2(void *__args)
{
	spdlog::info("Received request: cublasSsyr_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDsyr_v2(void *__args)
{
	spdlog::info("Received request: cublasDsyr_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsyr_v2(void *__args)
{
	spdlog::info("Received request: cublasCsyr_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZsyr_v2(void *__args)
{
	spdlog::info("Received request: cublasZsyr_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCher_v2(void *__args)
{
	spdlog::info("Received request: cublasCher_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZher_v2(void *__args)
{
	spdlog::info("Received request: cublasZher_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSspr_v2(void *__args)
{
	spdlog::info("Received request: cublasSspr_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDspr_v2(void *__args)
{
	spdlog::info("Received request: cublasDspr_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasChpr_v2(void *__args)
{
	spdlog::info("Received request: cublasChpr_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZhpr_v2(void *__args)
{
	spdlog::info("Received request: cublasZhpr_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSsyr2_v2(void *__args)
{
	spdlog::info("Received request: cublasSsyr2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDsyr2_v2(void *__args)
{
	spdlog::info("Received request: cublasDsyr2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsyr2_v2(void *__args)
{
	spdlog::info("Received request: cublasCsyr2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZsyr2_v2(void *__args)
{
	spdlog::info("Received request: cublasZsyr2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCher2_v2(void *__args)
{
	spdlog::info("Received request: cublasCher2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZher2_v2(void *__args)
{
	spdlog::info("Received request: cublasZher2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSspr2_v2(void *__args)
{
	spdlog::info("Received request: cublasSspr2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDspr2_v2(void *__args)
{
	spdlog::info("Received request: cublasDspr2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasChpr2_v2(void *__args)
{
	spdlog::info("Received request: cublasChpr2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZhpr2_v2(void *__args)
{
	spdlog::info("Received request: cublasZhpr2_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgemm_v2(void *__args)
{
	spdlog::info("Received request: cublasSgemm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgemm_v2(void *__args)
{
	spdlog::info("Received request: cublasDgemm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgemm_v2(void *__args)
{
	spdlog::info("Received request: cublasCgemm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgemm3m(void *__args)
{
	spdlog::info("Received request: cublasCgemm3m");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgemm3mEx(void *__args)
{
	spdlog::info("Received request: cublasCgemm3mEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgemm_v2(void *__args)
{
	spdlog::info("Received request: cublasZgemm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgemm3m(void *__args)
{
	spdlog::info("Received request: cublasZgemm3m");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasHgemm(void *__args)
{
	spdlog::info("Received request: cublasHgemm");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgemmEx(void *__args)
{
	spdlog::info("Received request: cublasSgemmEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGemmEx(void *__args)
{
	spdlog::info("Received request: cublasGemmEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgemmEx(void *__args)
{
	spdlog::info("Received request: cublasCgemmEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasUint8gemmBias(void *__args)
{
	spdlog::info("Received request: cublasUint8gemmBias");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSsyrk_v2(void *__args)
{
	spdlog::info("Received request: cublasSsyrk_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDsyrk_v2(void *__args)
{
	spdlog::info("Received request: cublasDsyrk_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsyrk_v2(void *__args)
{
	spdlog::info("Received request: cublasCsyrk_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZsyrk_v2(void *__args)
{
	spdlog::info("Received request: cublasZsyrk_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsyrkEx(void *__args)
{
	spdlog::info("Received request: cublasCsyrkEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsyrk3mEx(void *__args)
{
	spdlog::info("Received request: cublasCsyrk3mEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCherk_v2(void *__args)
{
	spdlog::info("Received request: cublasCherk_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZherk_v2(void *__args)
{
	spdlog::info("Received request: cublasZherk_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCherkEx(void *__args)
{
	spdlog::info("Received request: cublasCherkEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCherk3mEx(void *__args)
{
	spdlog::info("Received request: cublasCherk3mEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSsyr2k_v2(void *__args)
{
	spdlog::info("Received request: cublasSsyr2k_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDsyr2k_v2(void *__args)
{
	spdlog::info("Received request: cublasDsyr2k_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsyr2k_v2(void *__args)
{
	spdlog::info("Received request: cublasCsyr2k_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZsyr2k_v2(void *__args)
{
	spdlog::info("Received request: cublasZsyr2k_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCher2k_v2(void *__args)
{
	spdlog::info("Received request: cublasCher2k_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZher2k_v2(void *__args)
{
	spdlog::info("Received request: cublasZher2k_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSsyrkx(void *__args)
{
	spdlog::info("Received request: cublasSsyrkx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDsyrkx(void *__args)
{
	spdlog::info("Received request: cublasDsyrkx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsyrkx(void *__args)
{
	spdlog::info("Received request: cublasCsyrkx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZsyrkx(void *__args)
{
	spdlog::info("Received request: cublasZsyrkx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCherkx(void *__args)
{
	spdlog::info("Received request: cublasCherkx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZherkx(void *__args)
{
	spdlog::info("Received request: cublasZherkx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSsymm_v2(void *__args)
{
	spdlog::info("Received request: cublasSsymm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDsymm_v2(void *__args)
{
	spdlog::info("Received request: cublasDsymm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCsymm_v2(void *__args)
{
	spdlog::info("Received request: cublasCsymm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZsymm_v2(void *__args)
{
	spdlog::info("Received request: cublasZsymm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasChemm_v2(void *__args)
{
	spdlog::info("Received request: cublasChemm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZhemm_v2(void *__args)
{
	spdlog::info("Received request: cublasZhemm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStrsm_v2(void *__args)
{
	spdlog::info("Received request: cublasStrsm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtrsm_v2(void *__args)
{
	spdlog::info("Received request: cublasDtrsm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtrsm_v2(void *__args)
{
	spdlog::info("Received request: cublasCtrsm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtrsm_v2(void *__args)
{
	spdlog::info("Received request: cublasZtrsm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStrmm_v2(void *__args)
{
	spdlog::info("Received request: cublasStrmm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtrmm_v2(void *__args)
{
	spdlog::info("Received request: cublasDtrmm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtrmm_v2(void *__args)
{
	spdlog::info("Received request: cublasCtrmm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtrmm_v2(void *__args)
{
	spdlog::info("Received request: cublasZtrmm_v2");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasHgemmBatched(void *__args)
{
	spdlog::info("Received request: cublasHgemmBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgemmBatched(void *__args)
{
	spdlog::info("Received request: cublasSgemmBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgemmBatched(void *__args)
{
	spdlog::info("Received request: cublasDgemmBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgemmBatched(void *__args)
{
	spdlog::info("Received request: cublasCgemmBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgemm3mBatched(void *__args)
{
	spdlog::info("Received request: cublasCgemm3mBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgemmBatched(void *__args)
{
	spdlog::info("Received request: cublasZgemmBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGemmBatchedEx(void *__args)
{
	spdlog::info("Received request: cublasGemmBatchedEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasGemmStridedBatchedEx(void *__args)
{
	spdlog::info("Received request: cublasGemmStridedBatchedEx");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgemmStridedBatched(void *__args)
{
	spdlog::info("Received request: cublasSgemmStridedBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgemmStridedBatched(void *__args)
{
	spdlog::info("Received request: cublasDgemmStridedBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgemmStridedBatched(void *__args)
{
	spdlog::info("Received request: cublasCgemmStridedBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgemm3mStridedBatched(void *__args)
{
	spdlog::info("Received request: cublasCgemm3mStridedBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgemmStridedBatched(void *__args)
{
	spdlog::info("Received request: cublasZgemmStridedBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasHgemmStridedBatched(void *__args)
{
	spdlog::info("Received request: cublasHgemmStridedBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgeam(void *__args)
{
	spdlog::info("Received request: cublasSgeam");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgeam(void *__args)
{
	spdlog::info("Received request: cublasDgeam");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgeam(void *__args)
{
	spdlog::info("Received request: cublasCgeam");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgeam(void *__args)
{
	spdlog::info("Received request: cublasZgeam");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgetrfBatched(void *__args)
{
	spdlog::info("Received request: cublasSgetrfBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgetrfBatched(void *__args)
{
	spdlog::info("Received request: cublasDgetrfBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgetrfBatched(void *__args)
{
	spdlog::info("Received request: cublasCgetrfBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgetrfBatched(void *__args)
{
	spdlog::info("Received request: cublasZgetrfBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgetriBatched(void *__args)
{
	spdlog::info("Received request: cublasSgetriBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgetriBatched(void *__args)
{
	spdlog::info("Received request: cublasDgetriBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgetriBatched(void *__args)
{
	spdlog::info("Received request: cublasCgetriBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgetriBatched(void *__args)
{
	spdlog::info("Received request: cublasZgetriBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgetrsBatched(void *__args)
{
	spdlog::info("Received request: cublasSgetrsBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgetrsBatched(void *__args)
{
	spdlog::info("Received request: cublasDgetrsBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgetrsBatched(void *__args)
{
	spdlog::info("Received request: cublasCgetrsBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgetrsBatched(void *__args)
{
	spdlog::info("Received request: cublasZgetrsBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStrsmBatched(void *__args)
{
	spdlog::info("Received request: cublasStrsmBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtrsmBatched(void *__args)
{
	spdlog::info("Received request: cublasDtrsmBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtrsmBatched(void *__args)
{
	spdlog::info("Received request: cublasCtrsmBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtrsmBatched(void *__args)
{
	spdlog::info("Received request: cublasZtrsmBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSmatinvBatched(void *__args)
{
	spdlog::info("Received request: cublasSmatinvBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDmatinvBatched(void *__args)
{
	spdlog::info("Received request: cublasDmatinvBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCmatinvBatched(void *__args)
{
	spdlog::info("Received request: cublasCmatinvBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZmatinvBatched(void *__args)
{
	spdlog::info("Received request: cublasZmatinvBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgeqrfBatched(void *__args)
{
	spdlog::info("Received request: cublasSgeqrfBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgeqrfBatched(void *__args)
{
	spdlog::info("Received request: cublasDgeqrfBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgeqrfBatched(void *__args)
{
	spdlog::info("Received request: cublasCgeqrfBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgeqrfBatched(void *__args)
{
	spdlog::info("Received request: cublasZgeqrfBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSgelsBatched(void *__args)
{
	spdlog::info("Received request: cublasSgelsBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDgelsBatched(void *__args)
{
	spdlog::info("Received request: cublasDgelsBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCgelsBatched(void *__args)
{
	spdlog::info("Received request: cublasCgelsBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZgelsBatched(void *__args)
{
	spdlog::info("Received request: cublasZgelsBatched");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasSdgmm(void *__args)
{
	spdlog::info("Received request: cublasSdgmm");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDdgmm(void *__args)
{
	spdlog::info("Received request: cublasDdgmm");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCdgmm(void *__args)
{
	spdlog::info("Received request: cublasCdgmm");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZdgmm(void *__args)
{
	spdlog::info("Received request: cublasZdgmm");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStpttr(void *__args)
{
	spdlog::info("Received request: cublasStpttr");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtpttr(void *__args)
{
	spdlog::info("Received request: cublasDtpttr");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtpttr(void *__args)
{
	spdlog::info("Received request: cublasCtpttr");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtpttr(void *__args)
{
	spdlog::info("Received request: cublasZtpttr");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasStrttp(void *__args)
{
	spdlog::info("Received request: cublasStrttp");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasDtrttp(void *__args)
{
	spdlog::info("Received request: cublasDtrttp");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasCtrttp(void *__args)
{
	spdlog::info("Received request: cublasCtrttp");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cublasZtrttp(void *__args)
{
	spdlog::info("Received request: cublasZtrttp");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaProfilerInitialize(void *__args)
{
	spdlog::info("Received request: cudaProfilerInitialize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_cudaProfilerStart(void *__args)
{
	spdlog::info("Received request: cudaProfilerStart");

    auto args = (struct cudaProfilerStartArg *) __args;
    cudaError_t err = cudaProfilerStart(

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_cudaProfilerStop(void *__args)
{
	spdlog::info("Received request: cudaProfilerStop");

    auto args = (struct cudaProfilerStopArg *) __args;
    cudaError_t err = cudaProfilerStop(

    );

    while(!send_ipc->send((void *) &err, sizeof(cudaError_t))) {
        send_ipc->wait_for_recv(1);
    }
}

void TallyServer::handle_nvrtcGetCUBINSize(void *__args)
{
	spdlog::info("Received request: nvrtcGetCUBINSize");
	throw std::runtime_error("Unimplemented.");
}

void TallyServer::handle_nvrtcGetCUBIN(void *__args)
{
	spdlog::info("Received request: nvrtcGetCUBIN");
	throw std::runtime_error("Unimplemented.");
}
