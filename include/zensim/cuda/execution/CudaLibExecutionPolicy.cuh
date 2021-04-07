#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusparse_v2.h>

#include "ExecutionPolicy.cuh"

namespace zs {

  enum CudaLibraryComponentFlagBit : u64 {
    culib_none = 0,
    culib_cusparse = 1,
    culib_cublas = 0x2,
    culib_cusolversp = 0x4,
    culib_cusolverdn = 0x8
  };

  template <CudaLibraryComponentFlagBit flagbit> struct CudaLibHandle {};

  template <> struct CudaLibHandle<culib_cusparse> {
    cusparseHandle_t handle{nullptr};
    CudaLibHandle(CudaExecutionPolicy& cupol) {
      cusparseCreate(&handle);
      cusparseSetStream(handle, (cudaStream_t)Cuda::ref_cuda_context(cupol.getProcid())
                                    .stream_spare(cupol.getStreamid()));
    }
    ~CudaLibHandle() { cusparseDestroy(handle); }
  };
  template <> struct CudaLibHandle<culib_cublas> {
    cublasHandle_t handle{nullptr};
    CudaLibHandle(CudaExecutionPolicy& cupol) {
      cublasCreate(&handle);
      cublasSetStream(handle, (cudaStream_t)Cuda::ref_cuda_context(cupol.getProcid())
                                  .stream_spare(cupol.getStreamid()));
    }
    ~CudaLibHandle() { cublasDestroy(handle); }
  };
  template <> struct CudaLibHandle<culib_cusolversp> {
    cusolverSpHandle_t handle{nullptr};
    CudaLibHandle(CudaExecutionPolicy& cupol) {
      cusolverSpCreate(&handle);
      cusolverSpSetStream(handle, (cudaStream_t)Cuda::ref_cuda_context(cupol.getProcid())
                                      .stream_spare(cupol.getStreamid()));
    }
    ~CudaLibHandle() { cusolverSpDestroy(handle); }
  };

  template <CudaLibraryComponentFlagBit... flagbits> struct CudaLibExecutionPolicy
      : ExecutionPolicyInterface<CudaLibExecutionPolicy<flagbits...>>,
        CudaLibHandle<flagbits>... {
    template <CudaLibraryComponentFlagBit flagbit> constexpr auto& handle() noexcept {
      return CudaLibHandle<flagbit>::handle;
    }

    CudaLibExecutionPolicy(CudaExecutionPolicy& cupol)
        : _cuPol{cupol}, CudaLibHandle<flagbits>{cupol}... {}

    CudaExecutionPolicy& _cuPol;
  };

}  // namespace zs