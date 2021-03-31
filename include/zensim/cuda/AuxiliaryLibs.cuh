#pragma once

#include <zensim/Singleton.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolver_common.h>
#include <cusparse_v2.h>

#include "Cuda.h"
#include "HostUtils.hpp"

namespace zs {

  struct CuBlas : Singleton<CuBlas> {
    CuBlas();
    ~CuBlas();
    cublasHandle_t handle{0};
  };
  struct CuSparse : Singleton<CuSparse> {
    CuSparse();
    ~CuSparse();
    template <typename F, typename... Args>
    static std::enable_if_t<is_same_v<std::invoke_result_t<F, Args...>, cusparseStatus_t>> call(
        F&& f, Args&&... args) noexcept {
      cusparseStatus_t res = f(instance().handle, FWD(args)...);
      Cuda::ref_cuda_context(0).syncStream<Cuda::StreamIndex::Compute>();
      checkCudaErrors(res);
    }
    cusparseHandle_t handle{0};
  };
  struct CuSolverSp : Singleton<CuSolverSp> {
    CuSolverSp();
    ~CuSolverSp();
    template <typename F, typename... Args>
    static std::enable_if_t<is_same_v<std::invoke_result_t<F, Args...>, cusolverStatus_t>> call(
        F&& f, Args&&... args) noexcept {
      cusolverStatus_t res = f(instance().handle, FWD(args)...);
      Cuda::ref_cuda_context(0).syncStream<Cuda::StreamIndex::Compute>();
      checkCudaErrors(res);
    }
    cusolverSpHandle_t handle{0};
  };
  struct CuSolverDn : Singleton<CuSolverDn> {
    CuSolverDn();
    ~CuSolverDn();

    template <typename F, typename... Args>
    static std::enable_if_t<is_same_v<std::invoke_result_t<F, Args...>, cusolverStatus_t>> call(
        F&& f, Args&&... args) noexcept {
      cusolverStatus_t res = f(instance().handle, FWD(args)...);
      Cuda::ref_cuda_context(0).syncStream<Cuda::StreamIndex::Compute>();
      checkCudaErrors(res);
    }
    cusolverDnHandle_t handle{0};
  };

}  // namespace zs