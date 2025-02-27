#pragma once
/// be cautious to include this header
/// to enable cuda compiler, include cuda header before this one

/// use these functions within other templated function (1) or in a source file (2)
/// (1)
/// REMEMBER! Make Sure Their Specializations Done In the Correct Compiler Context!
/// which is given a certain execution policy tag, necessary headers are to be included
/// (2)
/// inside a certain source file

// #include "zensim/execution/ExecutionPolicy.hpp"
#include <thread>

#include "zensim/math/bit/Bits.h"
#include "zensim/zpc_tpls/fmt/format.h"
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

#elif defined(__APPLE__)
#  include <arm_neon.h>
#  include <unistd.h>
#endif

namespace zs {

/// @brief synchronization funcs
// __threadfence
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>> thread_fence(
      ExecTag) {
    __threadfence();
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>> thread_fence(
      ExecTag) {
    __threadfence();
  }
#endif

#if ZS_ENABLE_OPENMP
  inline void thread_fence(omp_exec_tag) noexcept {
#  pragma omp flush
  }
#endif

  inline void thread_fence(seq_exec_tag) noexcept {}

  // __syncthreads
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>> sync_threads(
      ExecTag) {
    __syncthreads();
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>> sync_threads(
      ExecTag) {
    __syncthreads();
  }
#endif

#if ZS_ENABLE_OPENMP
  inline void sync_threads(omp_exec_tag) noexcept {
#  pragma omp barrier
  }
#endif

  inline void sync_threads(seq_exec_tag) noexcept {}

  // pause
  template <typename ExecTag = seq_exec_tag, enable_if_t<is_host_execution_tag<ExecTag>()> = 0>
  inline void pause_cpu(ExecTag = {}) {
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    YieldProcessor();
#elif defined(__clang__) || defined(__GNUC__)
#  ifdef ZS_PLATFORM_OSX
    pause();
#  else
    // _mm_pause();
    std::this_thread::yield();
#  endif
#else
    static_assert(always_false<ExecTag>, "cannot determinate appropriate pause() intrinsics");
#endif
  }

/// @brief warp shuffle funcs
// __shfl_sync
#if defined(__CUDACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, T>
  shfl_sync(ExecTag, unsigned mask, T var, int srcLane, int width = 32) {
    return __shfl_sync(mask, var, srcLane, width);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, T>
  shfl_sync(ExecTag, unsigned mask, T var, int srcLane, int width = 32) {
    return __shfl_sync(mask, var, srcLane, width);
  }
#endif

// __shfl_up_sync
#if defined(__CUDACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, T>
  shfl_up_sync(ExecTag, unsigned mask, T var, unsigned int delta, int width = 32) {
    return __shfl_up_sync(mask, var, delta, width);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, T>
  shfl_up_sync(ExecTag, unsigned mask, T var, unsigned int delta, int width = 32) {
    return __shfl_up_sync(mask, var, delta, width);
  }
#endif

// __shfl_down_sync
#if defined(__CUDACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, T>
  shfl_down_sync(ExecTag, unsigned mask, T var, unsigned int delta, int width = 32) {
    return __shfl_down_sync(mask, var, delta, width);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, T>
  shfl_down_sync(ExecTag, unsigned mask, T var, unsigned int delta, int width = 32) {
    return __shfl_down_sync(mask, var, delta, width);
  }
#endif

// __shfl_xor_sync
#if defined(__CUDACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, T>
  shfl_xor_sync(ExecTag, unsigned mask, T var, int laneMask, int width = 32) {
    return __shfl_xor_sync(mask, var, laneMask, width);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __host__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, T>
  shfl_xor_sync(ExecTag, unsigned mask, T var, int laneMask, int width = 32) {
    return __shfl_xor_sync(mask, var, laneMask, width);
  }
#endif

/// @brief warp vote funcs
// __activemask
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, unsigned>
  active_mask(ExecTag) {
    return __activemask();
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, unsigned>
  active_mask(ExecTag) {
    return __activemask();
  }
#endif

// __ballot_sync
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, unsigned>
  ballot_sync(ExecTag, unsigned mask, int predicate) {
    return __ballot_sync(mask, predicate);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, unsigned>
  ballot_sync(ExecTag, unsigned mask, int predicate) {
    return __ballot_sync(mask, predicate);
  }
#endif

// __all_sync
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, int> all_sync(
      ExecTag, unsigned mask, int predicate) {
    return __all_sync(mask, predicate);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, int> all_sync(
      ExecTag, unsigned mask, int predicate) {
    return __all_sync(mask, predicate);
  }
#endif

// __any_sync
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, int> any_sync(
      ExecTag, unsigned mask, int predicate) {
    return __any_sync(mask, predicate);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, int> any_sync(
      ExecTag, unsigned mask, int predicate) {
    return __any_sync(mask, predicate);
  }
#endif

/// @brief math intrinsics
// ffs
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, int> ffs(ExecTag,
                                                                                        int x) {
    return __ffs(x);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, int> ffs(ExecTag,
                                                                                        int x) {
    return __ffs(x);
  }
#endif

// ffsll
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, int> ffsll(
      ExecTag, long long int x) {
    return __ffsll(x);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, int> ffsll(
      ExecTag, long long int x) {
    return __ffsll(x);
  }
#endif

  // popc
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, int> popc(
      ExecTag, unsigned int x) {
    return __popc(x);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, int> popc(
      ExecTag, unsigned int x) {
    return __popc(x);
  }
#endif

  // popcll
#if defined(__CUDACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, int> popcll(
      ExecTag, unsigned long long int x) {
    return __popcll(x);
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, int> popcll(
      ExecTag, unsigned long long int x) {
    return __popcll(x);
  }
#endif

// ref: https://graphics.stanford.edu/~seander/bithacks.html
// count leading zeros
#if defined(__CUDACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, int> count_lz(
      ExecTag, T x) {
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(int) == nbytes)
      return __clz((int)x);
    else if constexpr (sizeof(long long int) == nbytes)
      return __clzll((long long int)x);
    else {
      static_assert(always_false<T>, "count_lz(tag CUDA, [?] bytes) not viable\n");
    }
    return -1;
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, int> count_lz(
      ExecTag, T x) {
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(int) == nbytes)
      return __clz((int)x);
    else if constexpr (sizeof(long long int) == nbytes)
      return __clzll((long long int)x);
    else {
      static_assert(always_false<T>, "count_lz(tag MUSA, [?] bytes) not viable\n");
    }
    return -1;
  }
#endif

  template <typename ExecTag, typename T, enable_if_t<is_host_execution_tag<ExecTag>()> = 0>
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
    else
      static_assert(always_false<T>, "unsupported type for count_lz.");
  }

  /// reverse bits
#if defined(__CUDACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, cuda_exec_tag>, T> reverse_bits(
      ExecTag, T x) {
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(unsigned int) == nbytes)
      return __brev((unsigned int)x);
    else if constexpr (sizeof(unsigned long long int) == nbytes)
      return __brevll((unsigned long long int)x);
    else
      static_assert(always_false<T>, "reverse_bits(tag [?], [?] bytes) not viable\n");
    return x;
  }
#endif
#if defined(__MUSACC__)
  template <typename ExecTag, typename T>
  __forceinline__ __device__ enable_if_type<is_same_v<ExecTag, musa_exec_tag>, T> reverse_bits(
      ExecTag, T x) {
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(unsigned int) == nbytes)
      return __brev((unsigned int)x);
    else if constexpr (sizeof(unsigned long long int) == nbytes)
      return __brevll((unsigned long long int)x);
    else
      static_assert(always_false<T>, "reverse_bits(tag [?], [?] bytes) not viable\n");
    return x;
  }
#endif

  template <typename ExecTag, typename T, enable_if_t<is_host_execution_tag<ExecTag>()> = 0>
  inline T reverse_bits(ExecTag, T x) {
    constexpr auto nbytes = sizeof(T);
    if (x == (T)0) return 0;
    using Val = zs::make_unsigned_t<T>;
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
    else {
      // static_assert(always_false<T>, "unsupported type for reverse_bits.");
      // reverse within each byte
      for (int bitoffset = 0; tmp; bitoffset += 8) {
        unsigned char b = tmp & 0xff;
        b = ((u64)b * 0x0202020202ULL & 0x010884422010ULL) % 1023;
        ret |= ((Val)b << bitoffset);
        tmp >>= 8;
      }
      return (T)ret;
    }
  }

#if defined(__CUDACC__)
  template <typename T, execspace_e space = deduce_execution_space()>
  __forceinline__ __host__ __device__ enable_if_type<space == execspace_e::cuda, int> count_ones(
      T x, wrapv<space> = {}) {
    /// @note signed integers being sign-extended should be avoided
    static_assert(is_integral_v<remove_cvref_t<T>>, "T should be an integral type");
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(unsigned int) >= nbytes) {
      return __popc((make_unsigned_t<remove_cvref_t<T>>)x);
    } else if constexpr (sizeof(unsigned long long int) >= nbytes) {
      return __popcll((make_unsigned_t<remove_cvref_t<T>>)x);
    } else {
      static_assert(always_false<T>, "error in compiling cuda implementation of [count_ones]!");
    }
    return -1;
  }

  template <typename T, execspace_e space = deduce_execution_space()>
  __forceinline__ __host__ __device__ enable_if_type<(space == execspace_e::cuda), int>
  count_tailing_zeros(T x, wrapv<space> = {}) {
    static_assert(is_integral_v<remove_cvref_t<T>>, "T should be an integral type");
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(int) == nbytes) {
      return __clz(__brev(x));
    } else if constexpr (sizeof(long long int) == nbytes) {
      return __clzll(__brevll(x));
    } else {
      static_assert(always_false<T>,
                    "error in compiling cuda implementation of [count_tailing_zeros]!");
    }
    return -1;
  }
#endif
#if defined(__MUSACC__)
  template <typename T, execspace_e space = deduce_execution_space()>
  __forceinline__ __host__ __device__ enable_if_type<space == execspace_e::musa, int> count_ones(
      T x, wrapv<space> = {}) {
    /// @note signed integers being sign-extended should be avoided
    static_assert(is_integral_v<remove_cvref_t<T>>, "T should be an integral type");
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(unsigned int) >= nbytes) {
      return __popc((make_unsigned_t<remove_cvref_t<T>>)x);
    } else if constexpr (sizeof(unsigned long long int) >= nbytes) {
      return __popcll((make_unsigned_t<remove_cvref_t<T>>)x);
    } else {
      static_assert(always_false<T>, "error in compiling musa implementation of [count_ones]!");
    }
    return -1;
  }

  template <typename T, execspace_e space = deduce_execution_space()>
  __forceinline__ __host__ __device__ enable_if_type<(space == execspace_e::musa), int>
  count_tailing_zeros(T x, wrapv<space> = {}) {
    static_assert(is_integral_v<remove_cvref_t<T>>, "T should be an integral type");
    constexpr auto nbytes = sizeof(T);
    if constexpr (sizeof(int) == nbytes) {
      return __clz(__brev(x));
    } else if constexpr (sizeof(long long int) == nbytes) {
      return __clzll(__brevll(x));
    } else {
      static_assert(always_false<T>,
                    "error in compiling musa implementation of [count_tailing_zeros]!");
    }
    return -1;
  }
#endif

  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_host_execution<space>()> = 0>
  inline int count_ones(T x, wrapv<space> = {}) {
    /// @note signed integers being sign-extended should be avoided
    constexpr auto nbytes = sizeof(T);
    int ret{};
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    unsigned long index{};
    if constexpr (sizeof(unsigned short) == nbytes)
      ret = (int)__popcnt16((unsigned short)(make_unsigned_t<remove_cvref_t<T>>)x);
    else if constexpr (sizeof(unsigned int) == nbytes)
      ret = (int)__popcnt((unsigned int)(make_unsigned_t<remove_cvref_t<T>>)x);
    else if constexpr (sizeof(unsigned __int64) == nbytes)
      ret = (int)__popcnt64((unsigned __int64)(make_unsigned_t<remove_cvref_t<T>>)x);
#elif defined(__clang__) || defined(__GNUC__)
    if constexpr (sizeof(unsigned int) == nbytes)
      ret = __builtin_popcount((unsigned int)(make_unsigned_t<remove_cvref_t<T>>)x);
    else if constexpr (sizeof(unsigned long long) == nbytes)
      ret = __builtin_popcountll((unsigned long long)(make_unsigned_t<remove_cvref_t<T>>)x);
#else
    // fall back to software implementation
    if constexpr (true) {
      // ref: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable
      static const unsigned char BitsSetTable256[256] = {
#  define ZS_B2(n) n, n + 1, n + 1, n + 2
#  define ZS_B4(n) ZS_B2(n), ZS_B2(n + 1), ZS_B2(n + 1), ZS_B2(n + 2)
#  define ZS_B6(n) ZS_B4(n), ZS_B4(n + 1), ZS_B4(n + 1), ZS_B4(n + 2)
          B6(0), B6(1), B6(1), B6(2)};
      unsigned char *p = (unsigned char *)&x;
      ret = 0;
      for (int n = sizeof(x); n--;) ret += BitsSetTable256[*(p++)];
      ret = c;
    }
#endif
    else
      static_assert(always_false<T>,
                    "unsupported type for host implementation of count_tailing_zeros.");
    return ret;
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_host_execution<space>()> = 0>
  inline int count_tailing_zeros(T x, wrapv<space> = {}) {
    constexpr auto nbytes = sizeof(T);
    if (x == (T)0) return sizeof(remove_cvref_t<T>) * 8;
    int ret{};
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    unsigned long index{};
    if constexpr (sizeof(unsigned long) == nbytes) {
      _BitScanForward(&index, (unsigned long)x);
      ret = (int)index;
    } else if constexpr (sizeof(unsigned __int64) == nbytes) {
      _BitScanForward64(&index, (unsigned __int64)x);
      ret = (int)index;
    }
#elif defined(__clang__) || defined(__GNUC__)
    if constexpr (sizeof(unsigned int) == nbytes)
      ret = __builtin_ctz((unsigned int)x);
    else if constexpr (sizeof(unsigned long) == nbytes)
      ret = __builtin_ctzl((unsigned long)x);
    else if constexpr (sizeof(unsigned long long) == nbytes)
      ret = __builtin_ctzll((unsigned long long)x);
#else
    // fall back to software implementation
    if constexpr (sizeof(u8) == nbytes) {
      static const u8 DeBruijn[8] = {0, 1, 6, 2, 7, 5, 4, 3};
      u8 v = x;
      ret = DeBruijn[u8((v & -v) * 0x1DU) >> 5];
    } else if constexpr (sizeof(u32) >= nbytes) {
      static const u8 DeBruijn[32] = {0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
                                      31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9};
      u32 v = x;
      ret = DeBruijn[(u32)((v & -v) * 0x077CB531U) >> 27];
    } else if constexpr (sizeof(u64) == nbytes) {
      static const u8 DeBruijn[64] = {
          0,  1,  2,  53, 3,  7,  54, 27, 4,  38, 41, 8,  34, 55, 48, 28, 62, 5,  39, 46, 44, 42,
          22, 9,  24, 35, 59, 56, 49, 18, 29, 11, 63, 52, 6,  26, 37, 40, 33, 47, 61, 45, 43, 21,
          23, 58, 17, 10, 51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
      };
      u64 v = x;
      ret = DeBruijn[(u64)((v & -v) * u64(0x022FDD63CC95386D)) >> 58];
    }
#endif
    else
      static_assert(always_false<T>,
                    "unsupported type for host implementation of count_tailing_zeros.");
    return (int)ret;
  }

}  // namespace zs