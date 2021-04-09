#include "AuxiliaryLibs.cuh"

namespace zs {

  CuBlas::CuBlas() {
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&handle);
    checkCudaErrors(cublasStatus);
  }
  CuBlas::~CuBlas() {
    cublasStatus_t cublasStatus;
    cublasStatus = cublasDestroy(handle);
    checkCudaErrors(cublasStatus);
  }
  CuSparse::CuSparse() {
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);
    checkCudaErrors(cusparseStatus);
    cusparseSetStream(handle, (cudaStream_t)Cuda::ref_cuda_context(0).streamCompute());
  }
  CuSparse::~CuSparse() {
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseDestroy(handle);
    checkCudaErrors(cusparseStatus);
  }

  CuSolverSp::CuSolverSp() {
    cusolverStatus_t cusolverStatus;
    cusolverStatus = cusolverSpCreate(&handle);
    checkCudaErrors(cusolverStatus);
    cusolverSpSetStream(handle, (cudaStream_t)Cuda::ref_cuda_context(0).streamCompute());
  }
  CuSolverSp::~CuSolverSp() {
    cusolverStatus_t cusolverStatus;
    cusolverStatus = cusolverSpDestroy(handle);
    checkCudaErrors(cusolverStatus);
  }
  CuSolverDn::CuSolverDn() {
    cusolverStatus_t cusolverStatus;
    cusolverStatus = cusolverDnCreate(&handle);
    checkCudaErrors(cusolverStatus);
    cusolverDnSetStream(handle, (cudaStream_t)Cuda::ref_cuda_context(0).streamCompute());
  }
  CuSolverDn::~CuSolverDn() {
    cusolverStatus_t cusolverStatus;
    cusolverStatus = cusolverDnDestroy(handle);
    checkCudaErrors(cusolverStatus);
  }

}  // namespace zs