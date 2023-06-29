#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#define Zensim_EXPORT
#include "zensim/ZensimExport.hpp"
#include "zensim/cuda/Cuda.h"

extern "C" {

ZENSIM_EXPORT zs::CudaExecutionPolicy *policy__device() { return new zs::CudaExecutionPolicy; }
ZENSIM_EXPORT void del_policy__device(zs::CudaExecutionPolicy *v) { delete v; }

ZENSIM_EXPORT void launch__device(zs::CudaExecutionPolicy *ppol, void *kernel, zs::size_t dim,
                                  void **args) {
  using namespace zs;
  CudaExecutionPolicy &pol = *ppol;

  const int blockDim = 128;
  const int gridDim = (dim + blockDim - 1) / blockDim;

  auto &context = pol.context();
  Cuda::ContextGuard guard(context.getContext());

  CUresult ec = cuLaunchKernel((CUfunction)kernel, gridDim, 1, 1, blockDim, 1, 1, 0,
                               (CUstream)pol.getStream(), args, 0);

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