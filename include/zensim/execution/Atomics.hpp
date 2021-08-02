#pragma once
#include <atomic>

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/bit/Bits.h"
#if defined(_WIN32)
#  include <winnt.h>
#endif

namespace zs {

  /// reference: raja/include/RAJA/policy/atomic_builtin.hpp: BuiltinAtomicCAS

  template <typename ExecTag, typename T> ZS_FUNCTION T atomic_add(ExecTag, T *dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      return atomicAdd(dest, val);
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest += val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1

#  if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (std::is_integral_v<T>) {
        if constexpr (sizeof(T) == sizeof(char))
          return InterlockedExchangeAdd8(const_cast<char volatile *>((char *)dest), (char)val);
        else if constexpr (sizeof(T) == sizeof(short))
          return InterlockedExchangeAdd16(const_cast<short volatile *>((short *)dest), (short)val);
        else if constexpr (sizeof(T) == sizeof(long))
          return InterlockedAdd(const_cast<long volatile *>((long *)dest), (long)val);
        else if constexpr (sizeof(T) == sizeof(__int64))
          return InterlockedAdd64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
      }
#  else
      return __atomic_fetch_add(dest, val, __ATOMIC_SEQ_CST);
#  endif

#else
      /// introduced in c++20
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.fetch_add(val, std::memory_order_seq_cst);
#endif
    }
    return (T)0;
  }

  // https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
  template <typename ExecTag, typename T> ZS_FUNCTION T atomic_inc(ExecTag, T *dest) {
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
    if constexpr (
        is_same_v<ExecTag,
                  cuda_exec_tag> && std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)) {
      unsigned int active = __activemask();
      int leader = __ffs(active) - 1;
      int change = __popc(active);
      // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid
      unsigned int rank = __popc(active & ((1 << (threadIdx.x & 31)) - 1));
      T warp_res;
      if (rank == 0) warp_res = atomicAdd(dest, (T)change);
      warp_res = __shfl_sync(active, warp_res, leader);
      return warp_res + rank;
    }
#endif
    return atomic_add(ExecTag{}, dest, (T)1);
  }

  ///
  /// exch, cas
  ///
  template <typename ExecTag, typename T> ZS_FUNCTION T atomic_exch(ExecTag, T *dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>)
      return atomicExch(dest, val);
    else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest = val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
#  if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (sizeof(T) == sizeof(char))
        return InterlockedExchange8(const_cast<char volatile *>((char *)dest), (char)val);
      else if constexpr (sizeof(T) == sizeof(short))
        return InterlockedExchange16(const_cast<short volatile *>((short *)dest), (short)val);
      else if constexpr (sizeof(T) == sizeof(long))
        return InterlockedExchange(const_cast<long volatile *>((long *)dest), (long)val);
      else if constexpr (sizeof(T) == sizeof(__int64))
        return InterlockedExchange64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
#  else
      return __atomic_exchange_n(dest, val, __ATOMIC_SEQ_CST);
#  endif
#else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.exchange(val, std::memory_order_seq_cst);
#endif
    }
    return (T)0;
  }

  template <typename ExecTag, typename T>
  ZS_FUNCTION T atomic_cas(ExecTag, T *dest, T expected, T desired) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      if constexpr (is_same_v<T, float> && sizeof(int) == sizeof(T))
        return reinterpret_bits<float>(atomicCAS((unsigned int *)dest,
                                                 reinterpret_bits<unsigned int>(expected),
                                                 reinterpret_bits<unsigned int>(desired)));
      else if constexpr (is_same_v<T, double> && sizeof(unsigned long long int) == sizeof(T))
        return reinterpret_bits<double>(atomicCAS(
            (unsigned long long int *)dest, reinterpret_bits<unsigned long long int>(expected),
            reinterpret_bits<unsigned long long int>(desired)));
      else
        return atomicCAS(dest, expected, desired);
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      if (old == expected) *dest = desired;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1

#  if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (sizeof(T) == sizeof(char)) {  // 8-bit
        return reinterpret_bits<T>(_InterlockedCompareExchange8(
            (char *)dest, reinterpret_bits<char>(desired), reinterpret_bits<char>(expected)));
      } else if constexpr (sizeof(T) == sizeof(short)) {  // 16-bit
        return reinterpret_bits<T>(_InterlockedCompareExchange16(
            (short *)dest, reinterpret_bits<short>(desired), reinterpret_bits<short>(expected)));
      } else if constexpr (sizeof(T) == sizeof(long)) {  // 32-bit
        return reinterpret_bits<T>(_InterlockedCompareExchange(
            (long *)dest, reinterpret_bits<long>(desired), reinterpret_bits<long>(expected)));
      } else if constexpr (sizeof(T) == sizeof(__int64)) {
        return reinterpret_bits<T>(
            InterlockedCompareExchange64((__int64 *)dest, reinterpret_bits<__int64>(desired),
                                         reinterpret_bits<__int64>(expected)));
      }
#  else

      if constexpr (is_same_v<T, float> && sizeof(int) == sizeof(T)) {
        const T old = expected;
        int expected_ = reinterpret_bits<int>(expected);
        if (__atomic_compare_exchange_n((int *)dest, &expected_, reinterpret_bits<int>(desired),
                                        false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
          return old;
        return reinterpret_bits<float>(expected_);
      } else if constexpr (is_same_v<T, double> && sizeof(long long) == sizeof(T)) {
        const T old = expected;
        long long expected_ = reinterpret_bits<long long>(expected);
        if (__atomic_compare_exchange_n((long long *)dest, &expected_,
                                        reinterpret_bits<long long>(desired), false,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
          return old;
        return reinterpret_bits<double>(expected_);
      } else {
        const T old = expected;
        if (__atomic_compare_exchange_n(dest, &expected, desired, false, __ATOMIC_SEQ_CST,
                                        __ATOMIC_SEQ_CST))
          return old;
        return expected;
      }
#  endif

#else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.compare_exchange_strong(expected, desired, std::memory_order_seq_cst);
#endif
    }
    return (T)0;
  }

  ///
  /// min/ max operations
  ///
  // https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Device.cu
  // https://herbsutter.com/2012/08/31/reader-qa-how-to-write-a-cas-loop-using-stdatomics/
  template <typename ExecTag, typename T>
  ZS_FUNCTION void atomic_max(ExecTag execTag, T *const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      if constexpr (std::is_integral_v<T>) {
        atomicMax(dest, val);
        return;
      } else {
        T old = *dest;
        for (T assumed = old;
             assumed < val && (old = atomic_cas(execTag, dest, assumed, val)) != assumed;
             assumed = old)
          ;
        return;
      }
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      if (old < val) *dest = val;
      return;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
      T old = *dest;
      for (T assumed = old;
           assumed < val && (old = atomic_cas(execTag, dest, assumed, val)) != assumed;
           assumed = old)
        ;
      return;
    }
    // throw std::runtime_error(
    //    fmt::format("atomic_add(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return;
  }

  template <typename ExecTag, typename T>
  ZS_FUNCTION void atomic_min(ExecTag execTag, T *const dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      if constexpr (std::is_integral_v<T>) {
        atomicMin(dest, val);
        return;
      } else {
        T old = *dest;
        for (T assumed = old;
             assumed > val && (old = atomic_cas(execTag, dest, assumed, val)) != assumed;
             assumed = old)
          ;
        return;
      }
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      if (old > val) *dest = val;
      return;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
      T old = *dest;
      for (T assumed = old;
           assumed > val && (old = atomic_cas(execTag, dest, assumed, val)) != assumed;
           assumed = old)
        ;
      return;
    }
    return;
  }

  ///
  /// bit-wise operations
  ///
  template <typename ExecTag, typename T> ZS_FUNCTION T atomic_or(ExecTag, T *dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      return atomicOr(dest, val);
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest |= val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1

#  if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (sizeof(T) == sizeof(char))
        return InterlockedOr8(const_cast<char volatile *>((char *)dest), (char)val);
      else if constexpr (sizeof(T) == sizeof(short))
        return InterlockedOr16(const_cast<short volatile *>((short *)dest), (short)val);
      else if constexpr (sizeof(T) == sizeof(long))
        return InterlockedOr(const_cast<long volatile *>((long *)dest), (long)val);
      else if constexpr (sizeof(T) == sizeof(__int64))
        return InterlockedOr64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
#  else
      return __atomic_fetch_or(dest, val, __ATOMIC_SEQ_CST);
#  endif

#else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.fetch_or(val, std::memory_order_seq_cst);
#endif
    }
    return (T)0;
  }

  template <typename ExecTag, typename T> ZS_FUNCTION T atomic_and(ExecTag, T *dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      return atomicAnd(dest, val);
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest &= val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
#  if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (sizeof(T) == sizeof(char))
        return InterlockedAnd8(const_cast<char volatile *>((char *)dest), (char)val);
      else if constexpr (sizeof(T) == sizeof(short))
        return InterlockedAnd16(const_cast<short volatile *>((short *)dest), (short)val);
      else if constexpr (sizeof(T) == sizeof(long))
        return InterlockedAnd(const_cast<long volatile *>((long *)dest), (long)val);
      else if constexpr (sizeof(T) == sizeof(__int64))
        return InterlockedAnd64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
#  else
      return __atomic_fetch_and(dest, val, __ATOMIC_SEQ_CST);
#  endif
#else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.fetch_and(val, std::memory_order_seq_cst);
#endif
    }
    // throw std::runtime_error(
    //    fmt::format("atomic_and(tag {}, ...) not viable\n", get_execution_space_tag(ExecTag{})));
    return (T)0;
  }

  template <typename ExecTag, typename T> ZS_FUNCTION T atomic_xor(ExecTag, T *dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      return atomicXor(dest, val);
    } else if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest ^= val;
      return old;
    } else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if 1
#  if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (sizeof(T) == sizeof(char))
        return InterlockedXor8(const_cast<char volatile *>((char *)dest), (char)val);
      else if constexpr (sizeof(T) == sizeof(short))
        return InterlockedXor16(const_cast<short volatile *>((short *)dest), (short)val);
      else if constexpr (sizeof(T) == sizeof(long))
        return InterlockedXor(const_cast<long volatile *>((long *)dest), (long)val);
      else if constexpr (sizeof(T) == sizeof(__int64))
        return InterlockedXor64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
#  else
      return __atomic_fetch_xor(dest, val, __ATOMIC_SEQ_CST);
#  endif
#else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.fetch_xor(val, std::memory_order_seq_cst);
#endif
    }
    return (T)0;
  }

}  // namespace zs