#pragma once
/// be cautious to include this header
/// to enable cuda compiler, include cuda header before this one

/// use these functions within other templated function (1) or in a source file (2)
/// (1)
/// REMEMBER! Make Sure Their Specializations Done In the Correct Compiler Context!
/// which is given a certain execution policy tag, necessary headers are to be included
/// (2)
/// inside a certain source file

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/bit/Bits.h"

namespace zs {

#if !ZS_ENABLE_CUDA || !defined(__CUDACC__)
#  define __device__ inline
#endif

  // __threadfence
  template <typename ExecTag, enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> = 0>
  __device__ void thread_fence(ExecTag) {
#if defined(__CUDACC__)
    __threadfence();
#else
    throw std::runtime_error(
        fmt::format("thread_fence(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
#endif
  }

#if 0
  template <typename ExecTag, enable_if_t<!is_same_v<ExecTag, cuda_exec_tag>> = 0>
  void thread_fence(ExecTag) noexcept {}
#endif

  // __activemask
  template <typename ExecTag, enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> = 0>
  __device__ unsigned active_mask(ExecTag) {
#if defined(__CUDACC__)
    return __activemask();
#else
    throw std::runtime_error(
        fmt::format("active_mask(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
#endif
  }

#if 0
  template <typename ExecTag, enable_if_t<!is_same_v<ExecTag, cuda_exec_tag>> = 0>
  unsigned active_mask(ExecTag) noexcept {
    return ~0u;
  }
#endif

  // __ballot_sync
  template <typename ExecTag, enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> = 0>
  __device__ unsigned ballot_sync(ExecTag, unsigned mask, int predicate) {
#if defined(__CUDACC__)
    return __ballot_sync(mask, predicate);
#else
    throw std::runtime_error(
        fmt::format("ballot_sync(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
#endif
  }

#if 0
  template <typename ExecTag, enable_if_t<!is_same_v<ExecTag, cuda_exec_tag>> = 0>
  unsigned ballot_sync(ExecTag, unsigned mask, int predicate) noexcept {
    return ~0u;
  }
#endif

}  // namespace zs