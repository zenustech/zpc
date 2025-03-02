#pragma once
#include "zensim/Platform.hpp"
#include "zensim/ZpcMeta.hpp"
#include "zensim/types/Property.h"

#if defined(__CUDACC__)
#elif defined(__MUSACC__)
#elif defined(__HIPCC__)
#elif defined(SYCL_LANGUAGE_VERSION)

#else
extern "C" {

#  if defined(ZS_PLATFORM_LINUX)
/// @note refer to <stdlib.h>
void *malloc(zs::size_t __size) noexcept;
/// @note refer to <string.h>
void *memcpy(void *__dest, const void *__src, zs::size_t __n) noexcept;
int printf(const char *, ...);

#  elif defined(ZS_PLATFORM_OSX)
void *malloc(zs::size_t __size);
void *memcpy(void *__dest, const void *__src, zs::size_t __n);
int printf(const char *, ...);

#  elif defined(ZS_PLATFORM_WINDOWS)
void *memcpy(void *__dest, const void *__src, zs::size_t __n);

#    ifdef ZPC_JIT_MODE
ZPC_ACRTIMP void *malloc(zs::size_t __size);
struct _iobuf;
struct __crt_locale_pointers;
using FILE = _iobuf;
using _locale_t = __crt_locale_pointers *;
ZPC_ACRTIMP FILE *__acrt_iob_func(unsigned _Ix);
ZPC_ACRTIMP int __stdio_common_vfprintf(unsigned __int64 _Options, FILE *_Stream,
                                        char const *_Format, _locale_t _Locale, char *_ArgList);
__declspec(noinline) __inline unsigned __int64 *__zs_local_stdio_printf_options(void) {
  static unsigned __int64 _OptionsStorage;
  return &_OptionsStorage;
}
inline int printf(char const *const fmtstr, ...) {
  int _Result;
  char *_ArgList;
  (void)(__va_start(&_ArgList, fmtstr));  // __crt_va_start(_ArgList, fmtstr);
  _Result = ::__stdio_common_vfprintf(
      *__zs_local_stdio_printf_options(), ::__acrt_iob_func(1), fmtstr, 0,
      _ArgList);                 // _Result = _vfprintf_l(__acrt_iob_func(1), fmtstr, 0, _ArgList);
  (void)(_ArgList = (char *)0);  // __crt_va_end(_ArgList);
  return _Result;
}
#    endif

#  endif
}
#endif

namespace zs {

  /// @note for __CUDA_ARCH__ specific usage, refer to
  /// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#preprocessor-symbols
  ///

  template <typename DstT, typename SrcT, execspace_e space = deduce_execution_space()>
  constexpr DstT reinterpret_bits(SrcT &&val, wrapv<space> = {}) noexcept {
    using Src = remove_cvref_t<SrcT>;
    using Dst = DstT;
    static_assert(is_same_v<remove_cvref_t<DstT>, Dst>, "DstT should not be decorated!");
    static_assert(sizeof(Src) == sizeof(Dst),
                  "Source Type and Destination Type must be of the same size.");
    static_assert(is_trivially_copyable_v<Src> && is_trivially_copyable_v<Dst>,
                  "Both types should be trivially copyable.");

    if constexpr (is_same_v<Src, Dst>) return FWD(val);

    if constexpr (space == execspace_e::cuda) {
#if __CUDA_ARCH__
      if constexpr (is_same_v<Dst, float> && is_same_v<Src, int>) {
        return __int_as_float(FWD(val));
      } else if constexpr (is_same_v<Dst, int> && is_same_v<Src, float>) {
        return __float_as_int(FWD(val));
      } else if constexpr (is_same_v<Dst, double> && is_same_v<Src, long long>) {
        return __longlong_as_double(FWD(val));
      } else if constexpr (is_same_v<Dst, long long> && is_same_v<Src, double>) {
        return __double_as_longlong(FWD(val));
      }
#endif
    } else if constexpr (space == execspace_e::musa) {
#if __MUSA_ARCH__
      if constexpr (is_same_v<Dst, float> && is_same_v<Src, int>) {
        return __int_as_float(FWD(val));
      } else if constexpr (is_same_v<Dst, int> && is_same_v<Src, float>) {
        return __float_as_int(FWD(val));
      } else if constexpr (is_same_v<Dst, double> && is_same_v<Src, long long>) {
        return __longlong_as_double(FWD(val));
      } else if constexpr (is_same_v<Dst, long long> && is_same_v<Src, double>) {
        return __double_as_longlong(FWD(val));
      }
#endif
    } else if constexpr (space == execspace_e::rocm) {
#if __HIP_ARCH__
      if constexpr (is_same_v<Dst, float> && is_same_v<Src, int>) {
        return __int_as_float(FWD(val));
      } else if constexpr (is_same_v<Dst, int> && is_same_v<Src, float>) {
        return __float_as_int(FWD(val));
      } else if constexpr (is_same_v<Dst, double> && is_same_v<Src, long long>) {
        return __longlong_as_double(FWD(val));
      } else if constexpr (is_same_v<Dst, long long> && is_same_v<Src, double>) {
        return __double_as_longlong(FWD(val));
      }
#endif
    }
    // safe measure
    Dst dst{};
    memcpy(&dst, const_cast<const Src *>(&val), sizeof(Dst));
    return dst;
  }

}  // namespace zs