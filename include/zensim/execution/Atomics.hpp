#pragma once
#include <atomic>

#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename ExecTag, typename T>
  constexpr T atomic_add(ExecTag, T* dest, const T val) noexcept {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
#if ZS_ENABLE_CUDA
      return atomicAdd((std::make_unsigned_t<T>*)dest, (std::make_unsigned_t<T>)val);
#endif
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      dest += val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 0
      /// introduced in c++20
      std::atomic_ref<T> target{const_cast<T&>(*dest)};
      return target.fetch_add(val);
#else
#endif
    }
    throw std::runtime_error(
        fmt::format("atomic_add(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return (T)0;
  }

}  // namespace zs