#pragma once
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse_v2.h>

#include "ExecutionPolicy.cuh"
#include "zensim/Reflection.h"
#include "zensim/ZpcFunction.hpp"

namespace zs {

  enum CudaLibraryComponentFlagBit : u64 {
    culib_none = 0,
    culib_cusparse = 1,
    culib_cublas = 0x2,
    culib_cusolversp = 0x4,
    culib_cusolverdn = 0x8
  };

  namespace detail {
    template <typename T> void check_culib_error(T result) {
      if (static_cast<int>(result) != 0) {
        fmt::print("culib execution of {} error: code [{}]\n", demangle<T>(),
                   static_cast<int>(result));
      }
    }
    template <CudaLibraryComponentFlagBit flagbit> struct CudaLibStatusType {
      using type = void;
    };
    template <> struct CudaLibStatusType<culib_cusparse> {
      using type = cusparseStatus_t;
    };
    template <> struct CudaLibStatusType<culib_cublas> {
      using type = cublasStatus_t;
    };
    template <> struct CudaLibStatusType<culib_cusolversp> {
      using type = cusolverStatus_t;
    };
    template <> struct CudaLibStatusType<culib_cusolverdn> {
      using type = cusolverStatus_t;
    };
  }  // namespace detail
  template <CudaLibraryComponentFlagBit flagbit> using cudaLibStatus_t =
      typename detail::CudaLibStatusType<flagbit>::type;

  template <CudaLibraryComponentFlagBit flagbit> struct CudaLibHandle {};

  template <> struct CudaLibHandle<culib_cusparse> {
    cusparseHandle_t handle{nullptr};
    CudaLibHandle(CudaExecutionPolicy& cupol) {
      detail::check_culib_error(cusparseCreate(&handle));
      detail::check_culib_error(cusparseSetStream(
          handle, (cudaStream_t)Cuda::context(cupol.getProcid()).streamSpare(cupol.getStreamid())));
    }
    ~CudaLibHandle() { detail::check_culib_error(cusparseDestroy(handle)); }
  };
  template <> struct CudaLibHandle<culib_cublas> {
    cublasHandle_t handle{nullptr};
    CudaLibHandle(CudaExecutionPolicy& cupol) {
      detail::check_culib_error(cublasCreate(&handle));
      detail::check_culib_error(cublasSetStream(
          handle, (cudaStream_t)Cuda::context(cupol.getProcid()).streamSpare(cupol.getStreamid())));
    }
    ~CudaLibHandle() { detail::check_culib_error(cublasDestroy(handle)); }
  };
  template <> struct CudaLibHandle<culib_cusolversp> {
    cusolverSpHandle_t handle{nullptr};
    CudaLibHandle(CudaExecutionPolicy& cupol) {
      detail::check_culib_error(cusolverSpCreate(&handle));
      detail::check_culib_error(cusolverSpSetStream(
          handle, (cudaStream_t)Cuda::context(cupol.getProcid()).streamSpare(cupol.getStreamid())));
    }
    ~CudaLibHandle() { detail::check_culib_error(cusolverSpDestroy(handle)); }
  };

  template <CudaLibraryComponentFlagBit flagbit> struct CudaLibComponentExecutionPolicy
      : CudaLibHandle<flagbit>,
        virtual std::reference_wrapper<CudaExecutionPolicy> {
    template <typename Fn, typename... Args>
    enable_if_type<is_same_v<cudaLibStatus_t<flagbit>, typename function_traits<Fn>::return_t>>
    call(Fn&& fn, Args&&... args) const {
      using fts = function_traits<Fn>;
      if constexpr (sizeof...(Args) == fts::arity)
        detail::check_culib_error(fn(FWD(args)...));
      else
        detail::check_culib_error(fn(CudaLibHandle<flagbit>::handle, FWD(args)...));
    }

    CudaLibComponentExecutionPolicy(CudaExecutionPolicy& cupol)
        : CudaLibHandle<flagbit>{cupol}, std::reference_wrapper<CudaExecutionPolicy>{cupol} {}
  };

  template <CudaLibraryComponentFlagBit... flagbits> struct CudaLibExecutionPolicy
      : virtual std::reference_wrapper<CudaExecutionPolicy>,
        ExecutionPolicyInterface<CudaLibExecutionPolicy<flagbits...>>,
        CudaLibComponentExecutionPolicy<flagbits>... {
    template <CudaLibraryComponentFlagBit flagbit, typename Fn, typename... Args>
    constexpr enable_if_type<
        is_same_v<cudaLibStatus_t<flagbit>, typename function_traits<Fn>::return_t>>
    call(Fn&& fn, Args&&... args) const {
      CudaLibComponentExecutionPolicy<flagbit>::call(FWD(fn), FWD(args)...);
    }
    template <CudaLibraryComponentFlagBit flagbit> decltype(auto) getHandle() const {
      return static_cast<const CudaLibComponentExecutionPolicy<flagbit>&>(*this).handle;
    }
    decltype(auto) getStream() const {
      return Cuda::context(this->get().getProcid()).streamSpare(this->get().getStreamid());
    }

    CudaLibExecutionPolicy(CudaExecutionPolicy& cupol)
        : std::reference_wrapper<CudaExecutionPolicy>{cupol},
          CudaLibComponentExecutionPolicy<flagbits>{cupol}... {}
  };

}  // namespace zs