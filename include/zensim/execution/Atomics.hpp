#pragma once
#include <atomic>

#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename ExecTag, typename T>
  constexpr T atomic_add(ExecTag, T* const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
#if ZS_ENABLE_CUDA
      return atomicAdd(dest, val);
#endif
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest += val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
      return __atomic_add_fetch(dest, val, __ATOMIC_SEQ_CST);
#else
      /// introduced in c++20
      std::atomic_ref<T> target{const_cast<T&>(*dest)};
      return target.fetch_add(val);
#endif
    }
    throw std::runtime_error(
        fmt::format("atomic_add(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return (T)0;
  }

}  // namespace zs