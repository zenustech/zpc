#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/Cuda.h"

extern "C" {

zs::CudaExecutionPolicy *policy__device() { return new zs::CudaExecutionPolicy; }
void del_policy__device(zs::CudaExecutionPolicy *v) { delete v; }

void launch__device(zs::CudaExecutionPolicy *ppol, void *kernel, zs::size_t dim,
                                  void **args) {
  using namespace zs;
  CudaExecutionPolicy &pol = *ppol;

  const int blockDim = 128;
  const int gridDim = (dim + blockDim - 1) / blockDim;

  auto &context = pol.context();
  Cuda::ContextGuard guard(context.getContext());

  CUresult ec = cuLaunchKernel((CUfunction)kernel, gridDim, 1, 1, blockDim, 1, 1, 0,
                               (CUstream)pol.getStream(), args, 0);

  if (pol.shouldSync()) {
    context.syncStreamSpare(pol.getStreamid(), source_location::current());
  }

  if (ec != CUDA_SUCCESS) {
    const char *errString = nullptr;
    if (cuGetErrorString) {
      cuGetErrorString(ec, &errString);
      checkCuApiError((u32)ec, source_location::current(), "[cuLaunchKernel]", errString);
    } else
      checkCuApiError((u32)ec, source_location::current(), "[cuLaunchKernel]");
  }
}
}