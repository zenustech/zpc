/// taichi
// Driver
PER_CUDA_FUNCTION(init, cuInit, unsigned int)
PER_CUDA_FUNCTION(getDriverVersion, cuDriverGetVersion, int *)

// Device Management
PER_CUDA_FUNCTION(getDevice, cuDeviceGet, int *, int)
PER_CUDA_FUNCTION(getDeviceCount, cuDeviceGetCount, int *)
PER_CUDA_FUNCTION(getDeviceName, cuDeviceGetName, char *, int, int)
PER_CUDA_FUNCTION(getDeviceAttribute, cuDeviceGetAttribute, int *, unsigned int, int)
PER_CUDA_FUNCTION(getDeviceMemPool, cuDeviceGetMemPool, void **, int)
PER_CUDA_FUNCTION(setDeviceMemPool, cuDeviceSetMemPool, int, void *)

// Peer Memory Access
PER_CUDA_FUNCTION(canAccessPeer, cuDeviceCanAccessPeer, int *, int, int)
PER_CUDA_FUNCTION(getDeviceP2PAttribute, cuDeviceGetP2PAttribute, int *, unsigned int, int, int)
PER_CUDA_FUNCTION(enablePeerAccess, cuCtxEnablePeerAccess, void *, unsigned int)
PER_CUDA_FUNCTION(disablePeerAccess, cuCtxDisablePeerAccess, void *)
PER_CUDA_FUNCTION(memcpyPeerAsync, cuMemcpyPeerAsync, void *, void *, void *, void *, std::size_t,
                  void *)

// Context Management
PER_CUDA_FUNCTION(createContext, cuCtxCreate, void **, unsigned int, int)
PER_CUDA_FUNCTION(destroyContext, cuCtxDestroy, void *)
PER_CUDA_FUNCTION(setContext, cuCtxSetCurrent, void *)
PER_CUDA_FUNCTION(getContext, cuCtxGetCurrent, void **)
PER_CUDA_FUNCTION(getContextDevice, cuCtxGetDevice, int *)
PER_CUDA_FUNCTION(pushContext, cuCtxPushCurrent, void *)
PER_CUDA_FUNCTION(popContext, cuCtxPopCurrent, void **)
PER_CUDA_FUNCTION(syncContext, cuCtxSynchronize)

// Primary Context Management
PER_CUDA_FUNCTION(retainDevicePrimaryCtx, cuDevicePrimaryCtxRetain, void **, int)
PER_CUDA_FUNCTION(resetDevicePrimaryCtx, cuDevicePrimaryCtxReset, int)
PER_CUDA_FUNCTION(releaseDevicePrimaryCtx, cuDevicePrimaryCtxRelease, int)
PER_CUDA_FUNCTION(getDevicePrimaryCtxState, cuDevicePrimaryCtxGetState, int, unsigned int *, int *)
PER_CUDA_FUNCTION(setDevicePrimaryCtxFlags, cuDevicePrimaryCtxSetFlags, int, unsigned int)

// Stream management
PER_CUDA_FUNCTION(createStream, cuStreamCreate, void **, unsigned int)
PER_CUDA_FUNCTION(destroyStream, cuStreamDestroy, void *)
PER_CUDA_FUNCTION(syncStream, cuStreamSynchronize, void *)
PER_CUDA_FUNCTION(streamWaitEvent, cuStreamWaitEvent, void *, void *, unsigned int)
PER_CUDA_FUNCTION(queryStream, cuStreamQuery, void *)
PER_CUDA_FUNCTION(getStreamContext, cuStreamGetCtx, void *, void **)
PER_CUDA_FUNCTION(getStreamAttribute, cuStreamGetAttribute, void *, unsigned int, void *)
PER_CUDA_FUNCTION(setStreamAttribute, cuStreamSetAttribute, void *, unsigned int, const void *)

// Event management
PER_CUDA_FUNCTION(createEvent, cuEventCreate, void **, unsigned int)
PER_CUDA_FUNCTION(destroyEvent, cuEventDestroy, void *)
PER_CUDA_FUNCTION(recordEvent, cuEventRecord, void *, void *)
PER_CUDA_FUNCTION(eventElapsedTime, cuEventElapsedTime, float *, void *, void *)
PER_CUDA_FUNCTION(syncEvent, cuEventSynchronize, void *)

// memory operation
PER_CUDA_FUNCTION(memcpyHtoD, cuMemcpyHtoD, void *, void *, size_t)
PER_CUDA_FUNCTION(memcpyDtoH, cuMemcpyDtoH, void *, void *, size_t)
PER_CUDA_FUNCTION(memcpyDtoD, cuMemcpyDtoD, void *, void *, size_t)

PER_CUDA_FUNCTION(memcpyHtoDAsync, cuMemcpyHtoDAsync, void *, void *, size_t, void *)
PER_CUDA_FUNCTION(memcpyDtoHAsync, cuMemcpyDtoHAsync, void *, void *, size_t, void *)
PER_CUDA_FUNCTION(memcpyDtoDAsync, cuMemcpyDtoDAsync, void *, void *, size_t, void *)

PER_CUDA_FUNCTION(mallocAsync, cuMemAllocAsync, void **, size_t, void *)
PER_CUDA_FUNCTION(memAdvise, cuMemAdvise, void *, size_t, unsigned int, int)
PER_CUDA_FUNCTION(memcpy, cuMemcpy, void *, void *, size_t)
PER_CUDA_FUNCTION(memset, cuMemsetD8, void *, uint8_t, size_t)

// Virtual Memory
// physical memory handle
PER_CUDA_FUNCTION(vcreate, cuMemCreate, void *, size_t, const void *, unsigned long long)
// reserve a virtual address range
PER_CUDA_FUNCTION(vmalloc, cuMemAddressReserve, void **, size_t, size_t, void *, unsigned long long)
PER_CUDA_FUNCTION(vfree, cuMemAddressFree, void *, size_t)
PER_CUDA_FUNCTION(vrelease, cuMemRelease, unsigned long long)
PER_CUDA_FUNCTION(retainMemAllocHandle, cuMemRetainAllocationHandle, unsigned long long *, void *)
PER_CUDA_FUNCTION(getAllocationProperties, cuMemGetAllocationPropertiesFromHandle, void *,
                  unsigned long long)
PER_CUDA_FUNCTION(getAllocationGranularity, cuMemGetAllocationGranularity, size_t *, const void *,
                  unsigned int)
PER_CUDA_FUNCTION(getMemAccess, cuMemGetAccess, unsigned long long *, const void *, void *)
// set memory access rights for each device to the allocation
PER_CUDA_FUNCTION(setMemAccess, cuMemSetAccess, void *, size_t, const void *, size_t)
// maps physical memory handle to a virtual address range
PER_CUDA_FUNCTION(mmap, cuMemMap, void *, size_t, size_t, unsigned long long, unsigned long long)
PER_CUDA_FUNCTION(munmap, cuMemUnmap, void *, size_t)
PER_CUDA_FUNCTION(mmapAsync, cuMemMapArrayAsync, void *, unsigned int, void *)
// Conventional Memory
PER_CUDA_FUNCTION(umalloc, cuMemAllocManaged, void **, size_t, unsigned int)
PER_CUDA_FUNCTION(malloc, cuMemAlloc, void **, size_t)
PER_CUDA_FUNCTION(free, cuMemFree, void *)
PER_CUDA_FUNCTION(freeAsync, cuMemFreeAsync, void *, void *)
PER_CUDA_FUNCTION(getMemInfo, cuMemGetInfo, size_t *, size_t *)

// Module and kernels
PER_CUDA_FUNCTION(getFuncAttrib, cuFuncGetAttribute, int *, unsigned int, void *)
PER_CUDA_FUNCTION(getModuleFunc, cuModuleGetFunction, void **, void *, const char *)
PER_CUDA_FUNCTION(loadModuleData, cuModuleLoadData, void **, const void *)
PER_CUDA_FUNCTION(unloadModuleData, cuModuleUnload, void *)
PER_CUDA_FUNCTION(linkCreate, cuLinkCreate, unsigned int, unsigned int *, void **, void **)
PER_CUDA_FUNCTION(linkDestroy, cuLinkDestroy, void *)
PER_CUDA_FUNCTION(linkAddData, cuLinkAddData, void *, unsigned int, void *, size_t, const char *,
                  unsigned int, unsigned int *, void **)
// state, void**cubinOut, size_t* sizeOut
PER_CUDA_FUNCTION(linkComplete, cuLinkComplete, void *, void **, size_t *)
PER_CUDA_FUNCTION(loadModuleDataEx, cuModuleLoadDataEx, void **, const void *, unsigned int, void *,
                  void **)
PER_CUDA_FUNCTION(launchCuKernel, cuLaunchKernel, void *, unsigned int, unsigned int, unsigned int,
                  unsigned int, unsigned int, unsigned int, unsigned int, void *, void **, void **)
PER_CUDA_FUNCTION(launchCuCooperativeKernel, cuLaunchCooperativeKernel, void *, unsigned int,
                  unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                  unsigned int, void *, void **)
PER_CUDA_FUNCTION(launchHostFunc, cuLaunchHostFunc, void *, void *, void *)