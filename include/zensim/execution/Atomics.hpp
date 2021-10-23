#pragma once
#include <atomic>

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/bit/Bits.h"
#if defined(_WIN32)
#  include <windows.h>
// # include <winnt.h>
#endif

namespace zs {

  /// reference: raja/include/RAJA/policy/atomic_builtin.hpp: BuiltinAtomicCAS

#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, T> atomic_add(
      ExecTag, T *dest, const T val) {
    if constexpr (is_same_v<T, double>) {
      unsigned long long int *address_as_ull = (unsigned long long int *)dest;
      unsigned long long int old = *address_as_ull, assumed;

      do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
      } while (assumed != old);
      return __longlong_as_double(old);
    } else
      return atomicAdd(dest, val);
  }
#endif

  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>(), T>
  atomic_add_impl(ExecTag, T *dest, const T val) {
    static_assert(is_same_v<ExecTag, omp_exec_tag>);
#if defined(_OPENMP) && ZS_ENABLE_OPENMP
    if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
      using TT = conditional_t<sizeof(T) == 2, u16, conditional_t<sizeof(T) == 4, u32, u64>>;
      static_assert(sizeof(T) == sizeof(TT));
      TT oldVal{reinterpret_bits<TT>(*dest)};
      TT newVal{reinterpret_bits<TT>(reinterpret_bits<T>(oldVal) + val)}, readVal{};
      while ((readVal = atomic_cas(ExecTag{}, (TT *)dest, oldVal, newVal)) != oldVal) {
        oldVal = readVal;
        newVal = reinterpret_bits<TT>(reinterpret_bits<T>(readVal) + val);
      }
      return reinterpret_bits<T>(oldVal);
    }
#endif
  }

  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>(), T>
  atomic_add(ExecTag, T *dest, const T val) {
    if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest += val;
      return old;
    }
#if defined(_OPENMP) && ZS_ENABLE_OPENMP
    else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#  if 1
#    if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (std::is_integral_v<T>) {
        if constexpr (sizeof(T) == sizeof(char))
          return InterlockedExchangeAdd8(const_cast<char volatile *>((char *)dest), (char)val);
        else if constexpr (sizeof(T) == sizeof(short))
          return InterlockedExchangeAdd16(const_cast<short volatile *>((short *)dest), (short)val);
        else if constexpr (sizeof(T) == sizeof(long))
          return InterlockedAdd(const_cast<long volatile *>((long *)dest), (long)val);
        else if constexpr (sizeof(T) == sizeof(__int64))
          return InterlockedAdd64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
      } else
        return atomic_add_impl(ExecTag{}, dest, val);
#    else
      if constexpr (std::is_integral_v<T>)
        return __atomic_fetch_add(dest, val, __ATOMIC_SEQ_CST);
      else
        return atomic_add_impl(ExecTag{}, dest, val);
#    endif

#  else
      /// introduced in c++20
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.fetch_add(val, std::memory_order_seq_cst);
#  endif
    }
#endif
    return (T)0;
  }

  // https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, T> atomic_inc(
      ExecTag, T *dest) {
    if constexpr (std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)) {
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
    return atomic_add(ExecTag{}, dest, (T)1);
  }
#endif
  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>(), T>
  atomic_inc(ExecTag, T *dest) {
    return atomic_add(ExecTag{}, dest, (T)1);
  }

  ///
  /// exch, cas
  ///
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, T> atomic_exch(
      ExecTag, T *dest, const T val) {
    return atomicExch(dest, val);
  }
#endif
  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>(), T>
  atomic_exch(ExecTag, T *dest, const T val) {
    if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest = val;
      return old;
    }
#if defined(_OPENMP) && ZS_ENABLE_OPENMP
    else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#  if 1
#    if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (sizeof(T) == sizeof(char))
        return InterlockedExchange8(const_cast<char volatile *>((char *)dest), (char)val);
      else if constexpr (sizeof(T) == sizeof(short))
        return InterlockedExchange16(const_cast<short volatile *>((short *)dest), (short)val);
      else if constexpr (sizeof(T) == sizeof(long))
        return InterlockedExchange(const_cast<long volatile *>((long *)dest), (long)val);
      else if constexpr (sizeof(T) == sizeof(__int64))
        return InterlockedExchange64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
#    else
      return __atomic_exchange_n(dest, val, __ATOMIC_SEQ_CST);
#    endif
#  else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.exchange(val, std::memory_order_seq_cst);
#  endif
    }
#endif
    throw std::runtime_error("not implemented");
    return (T)0;
  }

#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, T> atomic_cas(
      ExecTag, T *dest, T expected, T desired) {
    if constexpr (is_same_v<T, float> && sizeof(int) == sizeof(T))
      return reinterpret_bits<float>(atomicCAS((unsigned int *)dest,
                                               reinterpret_bits<unsigned int>(expected),
                                               reinterpret_bits<unsigned int>(desired)));
    else if constexpr (is_same_v<T, double> && sizeof(unsigned long long int) == sizeof(T))
      return reinterpret_bits<double>(atomicCAS((unsigned long long int *)dest,
                                                reinterpret_bits<unsigned long long int>(expected),
                                                reinterpret_bits<unsigned long long int>(desired)));
    else
      return atomicCAS(dest, expected, desired);
  }
#endif
  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>(), T>
  atomic_cas(ExecTag, T *dest, T expected, T desired) {
    if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      if (old == expected) *dest = desired;
      return old;
    }
#if defined(_OPENMP) && ZS_ENABLE_OPENMP
    else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#  if 1

#    if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
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
#    else

      if constexpr (is_same_v<T, float> && sizeof(int) == sizeof(T)) {
        int expected_ = reinterpret_bits<int>(expected);
        __atomic_compare_exchange_n(const_cast<int volatile *>((int *)dest), &expected_,
                                    reinterpret_bits<int>(desired), false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_RELAXED);
        return reinterpret_bits<T>(expected_);
      } else if constexpr (is_same_v<T, double> && sizeof(long long) == sizeof(T)) {
        long long expected_ = reinterpret_bits<long long>(expected);
        __atomic_compare_exchange_n(const_cast<long long volatile *>((long long *)dest), &expected_,
                                    reinterpret_bits<long long>(desired), false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_RELAXED);
        return reinterpret_bits<T>(expected_);
      } else {
        __atomic_compare_exchange_n(const_cast<T volatile *>(dest), &expected, desired, false,
                                    __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
        return expected;
      }
#    endif

#  else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.compare_exchange_strong(expected, desired, std::memory_order_seq_cst);
#  endif
    }
#endif
    throw std::runtime_error("not implemented");
    return (T)0;
  }

  ///
  /// min/ max operations
  ///
  // https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Device.cu
  // https://herbsutter.com/2012/08/31/reader-qa-how-to-write-a-cas-loop-using-stdatomics/
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> atomic_max(
      ExecTag execTag, T *const dest, const T val) {
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
  }
#endif
  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>()>
  atomic_max(ExecTag execTag, T *const dest, const T val) {
    if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      if (old < val) *dest = val;
      return;
    }
#if defined(_OPENMP) && ZS_ENABLE_OPENMP
    else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
      T old = *dest;
      for (T assumed = old;
           assumed < val && (old = atomic_cas(execTag, dest, assumed, val)) != assumed;
           assumed = old)
        ;
      return;
    }
#endif
    throw std::runtime_error(
        fmt::format("atomic_max(tag {}, ...) not viable\n", get_execution_tag_name(ExecTag{})));
    return;
  }

#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>> atomic_min(
      ExecTag execTag, T *const dest, const T val) {
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
  }
#endif
  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>()>
  atomic_min(ExecTag execTag, T *const dest, const T val) {
    if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      if (old > val) *dest = val;
      return;
    }
#if defined(_OPENMP) && ZS_ENABLE_OPENMP
    else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
      T old = *dest;
      for (T assumed = old;
           assumed > val && (old = atomic_cas(execTag, dest, assumed, val)) != assumed;
           assumed = old)
        ;
      return;
    }
#endif
    throw std::runtime_error(
        fmt::format("atomic_min(tag {}, ...) not viable\n", get_execution_tag_name(ExecTag{})));
    return;
  }

  ///
  /// bit-wise operations
  ///
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, T> atomic_or(
      ExecTag, T *dest, const T val) {
    return atomicOr(dest, val);
  }
#endif
  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>(), T>
  atomic_or(ExecTag, T *dest, const T val) {
    if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest |= val;
      return old;
    }
#if defined(_OPENMP) && ZS_ENABLE_OPENMP
    else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#  if 1

#    if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (sizeof(T) == sizeof(char))
        return InterlockedOr8(const_cast<char volatile *>((char *)dest), (char)val);
      else if constexpr (sizeof(T) == sizeof(short))
        return InterlockedOr16(const_cast<short volatile *>((short *)dest), (short)val);
      else if constexpr (sizeof(T) == sizeof(long))
        return InterlockedOr(const_cast<long volatile *>((long *)dest), (long)val);
      else if constexpr (sizeof(T) == sizeof(__int64))
        return InterlockedOr64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
#    else
      return __atomic_fetch_or(dest, val, __ATOMIC_SEQ_CST);
#    endif

#  else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.fetch_or(val, std::memory_order_seq_cst);
#  endif
    }
#endif
    throw std::runtime_error(
        fmt::format("atomic_or(tag {}, ...) not viable\n", get_execution_tag_name(ExecTag{})));
    return (T)0;
  }

#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, T> atomic_and(
      ExecTag, T *dest, const T val) {
    if constexpr (ZS_ENABLE_CUDA && is_same_v<ExecTag, cuda_exec_tag>) {
      return atomicAnd(dest, val);
    }
  }
#endif
  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>(), T>
  atomic_and(ExecTag, T *dest, const T val) {
    if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest &= val;
      return old;
    }
#if defined(_OPENMP) && ZS_ENABLE_OPENMP
    else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#  if 1
#    if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (sizeof(T) == sizeof(char))
        return InterlockedAnd8(const_cast<char volatile *>((char *)dest), (char)val);
      else if constexpr (sizeof(T) == sizeof(short))
        return InterlockedAnd16(const_cast<short volatile *>((short *)dest), (short)val);
      else if constexpr (sizeof(T) == sizeof(long))
        return InterlockedAnd(const_cast<long volatile *>((long *)dest), (long)val);
      else if constexpr (sizeof(T) == sizeof(__int64))
        return InterlockedAnd64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
#    else
      return __atomic_fetch_and(dest, val, __ATOMIC_SEQ_CST);
#    endif
#  else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.fetch_and(val, std::memory_order_seq_cst);
#  endif
    }
#endif
    throw std::runtime_error(
        fmt::format("atomic_and(tag {}, ...) not viable\n", get_execution_tag_name(ExecTag{})));
    return (T)0;
  }

#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, T> atomic_xor(
      ExecTag, T *dest, const T val) {
    return atomicXor(dest, val);
  }
#endif
  template <typename ExecTag, typename T>
  inline std::enable_if_t<!is_same_v<ExecTag, cuda_exec_tag> && is_execution_tag<ExecTag>(), T>
  atomic_xor(ExecTag, T *dest, const T val) {
    if constexpr (is_same_v<ExecTag, host_exec_tag>) {
      const T old = *dest;
      *dest ^= val;
      return old;
    }
#if defined(_OPENMP) && ZS_ENABLE_OPENMP
    else if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#  if 1
#    if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
      if constexpr (sizeof(T) == sizeof(char))
        return InterlockedXor8(const_cast<char volatile *>((char *)dest), (char)val);
      else if constexpr (sizeof(T) == sizeof(short))
        return InterlockedXor16(const_cast<short volatile *>((short *)dest), (short)val);
      else if constexpr (sizeof(T) == sizeof(long))
        return InterlockedXor(const_cast<long volatile *>((long *)dest), (long)val);
      else if constexpr (sizeof(T) == sizeof(__int64))
        return InterlockedXor64(const_cast<__int64 volatile *>((__int64 *)dest), (__int64)val);
#    else
      return __atomic_fetch_xor(dest, val, __ATOMIC_SEQ_CST);
#    endif
#  else
      std::atomic_ref<T> target{const_cast<T &>(*dest)};
      return target.fetch_xor(val, std::memory_order_seq_cst);
#  endif
    }
#endif
    throw std::runtime_error(
        fmt::format("atomic_xor(tag {}, ...) not viable\n", get_execution_tag_name(ExecTag{})));
    return (T)0;
  }

}  // namespace zs