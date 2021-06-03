#pragma once
#include <atomic>

#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename ExecTag, typename T>
  constexpr T atomic_add(ExecTag, T* const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      return atomicAdd(dest, val);
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest += val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
      return __atomic_fetch_add(dest, val, __ATOMIC_SEQ_CST);
#else
      /// introduced in c++20
      std::atomic_ref<T> target{const_cast<T&>(*dest)};
      return target.fetch_add(val, std::memory_order_seq_cst);
#endif
    }
    throw std::runtime_error(
        fmt::format("atomic_add(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return (T)0;
  }

  template <typename ExecTag, typename T>
  constexpr T atomic_exch(ExecTag, T* const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>)
      return atomicExch(dest, val);
    else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest = val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
      return __atomic_exchange_n(dest, val, __ATOMIC_SEQ_CST);
#else
      std::atomic_ref<T> target{const_cast<T&>(*dest)};
      return target.exchange(val, std::memory_order_seq_cst);
#endif
    }
    throw std::runtime_error(
        fmt::format("atomic_exch(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return (T)0;
  }

  template <typename ExecTag, typename T>
  constexpr bool atomic_cas(ExecTag, T* const dest, T expected, T desired) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>)
      return (atomicCAS(dest, expected, desired) == expected);
    else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      if (*dest == expected) {
        *dest = desired;
        return true;
      }
      return false;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
      return __atomic_compare_exchange_n(dest, &expected, desired, false, __ATOMIC_SEQ_CST,
                                         __ATOMIC_SEQ_CST);
#else
      std::atomic_ref<T> target{const_cast<T&>(*dest)};
      return target.compare_exchange_strong(expected, desired, std::memory_order_seq_cst);
#endif
    }
    throw std::runtime_error(
        fmt::format("atomic_cas(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return (T)0;
  }

  template <typename ExecTag, typename T>
  constexpr T atomic_or(ExecTag, T* const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      return atomicOr(dest, val);
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest |= val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
      return __atomic_fetch_or(dest, val, __ATOMIC_SEQ_CST);
#else
      std::atomic_ref<T> target{const_cast<T&>(*dest)};
      return target.fetch_or(val, std::memory_order_seq_cst);
#endif
    }
    throw std::runtime_error(
        fmt::format("atomic_or(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return (T)0;
  }

  template <typename ExecTag, typename T>
  constexpr T atomic_and(ExecTag, T* const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      return atomicAnd(dest, val);
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest &= val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
      return __atomic_fetch_and(dest, val, __ATOMIC_SEQ_CST);
#else
      std::atomic_ref<T> target{const_cast<T&>(*dest)};
      return target.fetch_and(val, std::memory_order_seq_cst);
#endif
    }
    throw std::runtime_error(
        fmt::format("atomic_and(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return (T)0;
  }

  template <typename ExecTag, typename T>
  constexpr T atomic_xor(ExecTag, T* const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      return atomicXor(dest, val);
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest ^= val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
      return __atomic_fetch_xor(dest, val, __ATOMIC_SEQ_CST);
#else
      std::atomic_ref<T> target{const_cast<T&>(*dest)};
      return target.fetch_xor(val, std::memory_order_seq_cst);
#endif
    }
    throw std::runtime_error(
        fmt::format("atomic_xor(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return (T)0;
  }

}  // namespace zs