
// Stream management
PER_CUDA_FUNCTION(createStream, cudaStreamCreate, void **);
PER_CUDA_FUNCTION(destroyStream, cudaStreamDestroy, void *)
PER_CUDA_FUNCTION(syncStream, cudaStreamSynchronize, void *);
PER_CUDA_FUNCTION(streamWaitEvent, cudaStreamWaitEvent, void *, void *,
                  uint32_t);

PER_CUDA_FUNCTION(createEvent, cudaEventCreateWithFlags, void **, uint32_t)
PER_CUDA_FUNCTION(destroyEvent, cudaEventDestroy, void *)
PER_CUDA_FUNCTION(recordEvent, cudaEventRecord, void *, void *)
PER_CUDA_FUNCTION(eventElapsedTime, cudaEventElapsedTime, float *, void *,
                  void *);
PER_CUDA_FUNCTION(syncEvent, cudaEventSynchronize, void *);

// kernels
PER_CUDA_FUNCTION(launch, cudaLaunchKernel, void *, dim3, dim3, void **,
                  uint32_t, void *);
PER_CUDA_FUNCTION(launchMD, cudaLaunchCooperativeKernelMultiDevice, void *,
                  uint32_t, uint32_t);
