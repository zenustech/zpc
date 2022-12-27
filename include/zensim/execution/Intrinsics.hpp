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
#if defined(_WIN32)
#  include <intrin.h>
#  include <stdlib.h>
// #  include <windows.h>
// #  include <synchapi.h>

#elif defined(__linux__)
#  include <immintrin.h>
#  include <linux/futex.h>
#  include <sys/syscall.h> /* Definition of SYS_* constants */
#  include <unistd.h>
#endif

namespace zs {

  // __threadfence
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag>
  __forceinline__ __host__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>>
  thread_fence(ExecTag) {
#  ifdef __CUDA_ARCH__
    __threadfence();
#  else
    static_assert(!is_same_v<ExecTag, cuda_exec_tag>,
                  "error in compiling cuda implementation of [thread_fence]!");
#  endif
  }
#endif

#if ZS_ENABLE_OPENMP
  inline void thread_fence(omp_exec_tag) noexcept {
    /// a thread is guaranteed to see a consistent view of memory with respect to the variables in “
    /// list ”
#  pragma omp flush
  }
#endif

  inline void thread_fence(host_exec_tag) noexcept {}

  // __syncthreads
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag>
  __forceinline__ __host__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>>
  sync_threads(ExecTag) {
#  ifdef __CUDA_ARCH__
    __syncthreads();
#  else
    static_assert(!is_same_v<ExecTag, cuda_exec_tag>,
                  "error in compiling cuda implementation of [sync_threads]!");
#  endif
  }
#endif

#if ZS_ENABLE_OPENMP
  inline void sync_threads(omp_exec_tag) noexcept {
#  pragma omp barrier
  }
#endif

  inline void sync_threads(host_exec_tag) noexcept {}

  // pause
  template <typename ExecTag = host_exec_tag,
            enable_if_t<is_same_v<ExecTag, omp_exec_tag> || is_same_v<ExecTag, host_exec_tag>> = 0>
  inline void pause_cpu(ExecTag = {}) {
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    YieldProcessor();
#elif defined(__clang__) || defined(__GNUC__)
    _mm_pause();
#endif
  }

  // __activemask
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag>
  __forceinline__ __host__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, unsigned>
  active_mask(ExecTag) {
#  ifdef __CUDA_ARCH__
    return __activemask();
#  else
    static_assert(!is_same_v<ExecTag, cuda_exec_tag>,
                  "error in compiling cuda implementation of [active_mask]!");
    return 0;
#  endif
  }
#endif

  // __ballot_sync
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag>
  __forceinline__ __host__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, unsigned>
  ballot_sync(ExecTag, unsigned mask, int predicate) {
#  ifdef __CUDA_ARCH__
    return __ballot_sync(mask, predicate);
#  else
    static_assert(!is_same_v<ExecTag, cuda_exec_tag>,
                  "error in compiling cuda implementation of [ballot_sync]!");
    return 0;
#  endif
  }
#endif

  // ref: https://graphics.stanford.edu/~seander/bithacks.html

  /// count leading zeros
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, int>
  count_lz(ExecTag, T x) {
#  ifdef __CUDA_ARCH__
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(int) == nbytes)
      return __clz((int)x);
    else if constexpr (sizeof(long long int) == nbytes)
      return __clzll((long long int)x);
    else {
      static_assert(sizeof(long long int) != nbytes && sizeof(int) != nbytes,
                    "count_lz(tag CUDA, [?] bytes) not viable\n");
    }
    return -1;
#  else
    static_assert(!is_same_v<ExecTag, cuda_exec_tag>,
                  "error in compiling cuda implementation of [count_lz]!");
    return -1;
#  endif
  }
#endif

  template <typename ExecTag, typename T,
            enable_if_t<is_same_v<ExecTag, omp_exec_tag> || is_same_v<ExecTag, host_exec_tag>> = 0>
  inline int count_lz(ExecTag, T x) {
    constexpr auto nbytes = sizeof(T);
    if (x == (T)0) return nbytes * 8;
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    if constexpr (sizeof(unsigned short) == nbytes)
      return __lzcnt16((unsigned short)x);
    else if constexpr (sizeof(unsigned int) == nbytes)
      return __lzcnt((unsigned int)x);
    else if constexpr (sizeof(unsigned __int64) == nbytes)
      return __lzcnt64((unsigned __int64)x);
#elif defined(__clang__) || defined(__GNUC__)
    if constexpr (sizeof(unsigned int) == nbytes)
      return __builtin_clz((unsigned int)x);
    else if constexpr (sizeof(unsigned long) == nbytes)
      return __builtin_clzl((unsigned long)x);
    else if constexpr (sizeof(unsigned long long) == nbytes)
      return __builtin_clzll((unsigned long long)x);
#endif
    throw std::runtime_error(fmt::format("count_lz(tag {}, {} bytes) not viable\n",
                                         get_execution_tag_name(ExecTag{}), sizeof(T)));
  }

  /// reverse bits
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ std::enable_if_t<is_same_v<ExecTag, cuda_exec_tag>, T>
  reverse_bits(ExecTag, T x) {
#  ifdef __CUDA_ARCH__
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(unsigned int) == nbytes)
      return __brev((unsigned int)x);
    else if constexpr (sizeof(unsigned long long int) == nbytes)
      return __brevll((unsigned long long int)x);
    else
      static_assert(sizeof(unsigned long long int) != nbytes && sizeof(unsigned int) != nbytes,
                    "reverse_bits(tag [?], [?] bytes) not viable\n");
    return x;
#  else
    static_assert(!is_same_v<ExecTag, cuda_exec_tag>,
                  "error in compiling cuda implementation of [reverse_bits]!");
    return x;
#  endif
  }
#endif

  template <typename ExecTag, typename T,
            enable_if_t<is_same_v<ExecTag, omp_exec_tag> || is_same_v<ExecTag, host_exec_tag>> = 0>
  inline T reverse_bits(ExecTag, T x) {
    constexpr auto nbytes = sizeof(T);
    if (x == (T)0) return 0;
    using Val = std::make_unsigned_t<T>;
    Val tmp{}, ret{0};
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    if constexpr (sizeof(unsigned short) == nbytes)
      tmp = (Val)_byteswap_ushort((unsigned short)x);
    else if constexpr (sizeof(unsigned long) == nbytes)
      tmp = (Val)_byteswap_ulong((unsigned long)x);
    else if constexpr (sizeof(unsigned __int64) == nbytes)
      tmp = (Val)_byteswap_uint64((unsigned __int64)x);
#elif defined(__clang__) || defined(__GNUC__)
    if constexpr (sizeof(unsigned short) == nbytes)
      tmp = (Val)__builtin_bswap16((unsigned short)x);
    else if constexpr (sizeof(unsigned int) == nbytes)
      tmp = (Val)__builtin_bswap32((unsigned int)x);
    else if constexpr (sizeof(unsigned long long) == nbytes)
      tmp = (Val)__builtin_bswap64((unsigned long long)x);
#endif
    else
      throw std::runtime_error(fmt::format("reverse_bits(tag {}, {} bytes) not viable\n",
                                           get_execution_tag_name(ExecTag{}), sizeof(T)));
    // reverse within each byte
    for (int bitoffset = 0; tmp; bitoffset += 8) {
      unsigned char b = tmp & 0xff;
      b = ((u64)b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
      ret |= ((Val)b << bitoffset);
      tmp >>= 8;
    }
    return (T)ret;
  }

}  // namespace zs