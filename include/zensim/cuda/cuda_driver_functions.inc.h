
// Driver
PER_CUDA_FUNCTION(init, cuInit, int);
PER_CUDA_FUNCTION(getDriverVersion, cuDriverGetVersion, int *);

// Device management
PER_CUDA_FUNCTION(getDeviceCount, cuDeviceGetCount, int *);
PER_CUDA_FUNCTION(getDevice, cuDeviceGet, void **, int);
PER_CUDA_FUNCTION(getDeviceName, cuDeviceGetName, char *, int, void *);
PER_CUDA_FUNCTION(getDeviceAttribute, cuDeviceGetAttribute, int *, uint32_t, void *);

// Peer access
PER_CUDA_FUNCTION(canAccessPeer, cuDeviceCanAccessPeer, int *, void *, void *);
PER_CUDA_FUNCTION(enablePeerAccess, cuCtxEnablePeerAccess, void *, uint32_t);
PER_CUDA_FUNCTION(memcpyPeerAsync, cuMemcpyPeerAsync, void *, void *, void *, void *, std::size_t,
                  void *);

// Context management
PER_CUDA_FUNCTION(createContext, cuCtxCreate_v2, void **, int, void *);
PER_CUDA_FUNCTION(destroyContext, cuCtxDestroy_v2, void *);
PER_CUDA_FUNCTION(setContext, cuCtxSetCurrent, void *);
PER_CUDA_FUNCTION(getContext, cuCtxGetCurrent, void **);
PER_CUDA_FUNCTION(syncContext, cuCtxSynchronize);
PER_CUDA_FUNCTION(retainPrimaryCtx, cuDevicePrimaryCtxRetain, void **, void *);

// Stream management
// PER_CUDA_FUNCTION(createStream, cuStreamCreate, void **, uint32_t);
// PER_CUDA_FUNCTION(destroyStream, cuStreamDestroy, void *)
// PER_CUDA_FUNCTION(syncStream, cuStreamSynchronize, void *);
// PER_CUDA_FUNCTION(streamWaitEvent, cuStreamWaitEvent, void *, void *,
// uint32_t);

// Event management
// PER_CUDA_FUNCTION(createEvent, cuEventCreate, void **, uint32_t)
// PER_CUDA_FUNCTION(destroyEvent, cuEventDestroy, void *)
// PER_CUDA_FUNCTION(recordEvent, cuEventRecord, void *, void *)
// PER_CUDA_FUNCTION(eventElapsedTime, cuEventElapsedTime, float *, void *,
//                  void *);
// PER_CUDA_FUNCTION(syncEvent, cuEventSynchronize, void *);

// Memory management
PER_CUDA_FUNCTION(memcpyHtoD, cuMemcpyHtoD_v2, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpyDtoH, cuMemcpyDtoH_v2, void *, void *, std::size_t);
PER_CUDA_FUNCTION(memcpyDtoD, cuMemcpyDtoD_v2, void *, void *, std::size_t);

PER_CUDA_FUNCTION(memcpyHtoDAsync, cuMemcpyHtoDAsync_v2, void *, void *, std::size_t, void *);
PER_CUDA_FUNCTION(memcpyDtoHAsync, cuMemcpyDtoHAsync_v2, void *, void *, std::size_t, void *);
PER_CUDA_FUNCTION(memcpyDtoDAsync, cuMemcpyDtoDAsync_v2, void *, void *, std::size_t, void *);

PER_CUDA_FUNCTION(malloc, cuMemAlloc_v2, void **, std::size_t);
PER_CUDA_FUNCTION(memset, cuMemsetD8_v2, void *, uint8_t, std::size_t);
PER_CUDA_FUNCTION(free, cuMemFree_v2, void *);
PER_CUDA_FUNCTION(memInfo, cuMemGetInfo_v2, std::size_t *, std::size_t *);

// Module and kernels
PER_CUDA_FUNCTION(getModuleFunc, cuModuleGetFunction, void **, void *, const char *);
PER_CUDA_FUNCTION(getFuncAttrib, cuFuncGetAttribute, int *, uint32_t, void *);
PER_CUDA_FUNCTION(loadModuleDataEx, cuModuleLoadDataEx, void **, const char *, uint32_t, uint32_t *,
                  void **)
PER_CUDA_FUNCTION(launchKernel, cuLaunchKernel, void *, uint32_t, uint32_t, uint32_t, uint32_t,
                  uint32_t, uint32_t, uint32_t, void *, void **, void **);
PER_CUDA_FUNCTION(launchCooperativeKernel, cuLaunchCooperativeKernel, void *, uint32_t, uint32_t,
                  uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void *, void **);
