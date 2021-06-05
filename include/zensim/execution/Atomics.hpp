#pragma once
#include <atomic>

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/bit/Bits.h"

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
  constexpr void atomic_max(ExecTag execTag, T* const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      if constexpr (std::is_integral_v<T>) {
        atomicMax(dest, val);
        return;
      } else if constexpr (is_same_v<T, float>) {
        if (*dest >= val) return;
        int* const address_as_i = (int*)dest;
        int old = *address_as_i, assumed{};
        do {
          assumed = old;
          if (int_as_float(assumed) >= val) break;
          old = atomicCAS(address_as_i, assumed, float_as_int(val));
        } while (assumed != old);
        return;
      } else if constexpr (is_same_v<T, double>) {
        if (*dest >= val) return;
        long long* const address_as_i = (long long*)dest;
        long long old = *address_as_i, assumed{};
        do {
          assumed = old;
          if (longlong_as_double(assumed) >= val) break;
          old = atomicCAS(address_as_i, assumed, double_as_longlong(val));
        } while (assumed != old);
        return;
      }
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      if (old < val) *dest = val;
      return;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
      for (T old = *dest; old < val
                          && !__atomic_compare_exchange_n(dest, &old, val, true, __ATOMIC_SEQ_CST,
                                                          __ATOMIC_SEQ_CST);)
        ;
      return;
    }
    throw std::runtime_error(
        fmt::format("atomic_add(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return;
  }

  template <typename ExecTag, typename T>
  constexpr void atomic_min(ExecTag execTag, T* const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      if constexpr (std::is_integral_v<T>) {
        atomicMin(dest, val);
        return;
      } else if constexpr (is_same_v<T, float>) {
        if (*dest <= val) return;
        int* const address_as_i = (int*)dest;
        int old = *address_as_i, assumed{};
        do {
          assumed = old;
          if (int_as_float(assumed) <= val) break;
          old = atomicCAS(address_as_i, assumed, float_as_int(val));
        } while (assumed != old);
        return;
      } else if constexpr (is_same_v<T, double>) {
        if (*dest <= val) return;
        long long* const address_as_i = (long long*)dest;
        long long old = *address_as_i, assumed{};
        do {
          assumed = old;
          if (longlong_as_double(assumed) <= val) break;
          old = atomicCAS(address_as_i, assumed, double_as_longlong(val));
        } while (assumed != old);
        return;
      }
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      if (old > val) *dest = val;
      return;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
      for (T old = *dest; old > val
                          && !__atomic_compare_exchange_n(dest, &old, val, true, __ATOMIC_SEQ_CST,
                                                          __ATOMIC_SEQ_CST);)
        ;
      return;
    }
    throw std::runtime_error(
        fmt::format("atomic_add(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return;
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