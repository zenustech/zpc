#pragma once
#include "zensim/ZpcMeta.hpp"
#include "zensim/types/Property.h"

#if defined(__CUDACC__)

#else
extern "C" {
/// @note refer to <stdlib.h>
extern void *malloc(size_t __size) noexcept;
/// @note refer to <string.h>
extern void *memcpy(void *__dest, const void *__src, size_t __n) noexcept;
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
#if defined(__CUDA_ARCH__)
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