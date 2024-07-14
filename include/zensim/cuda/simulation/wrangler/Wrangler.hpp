#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"

namespace zs {

  struct AccessorAoSoA {
    using size_type = size_t;
    AccessorAoSoA() = default;
    /// aos
    constexpr AccessorAoSoA(wrapv<layout_e::aos>, void* ptr, unsigned short bytes,
                            unsigned short chnCnt, unsigned short aux) noexcept
        : base{ptr},
          numTileBits{(unsigned short)0},
          tileMask{(unsigned short)0},
          chnCnt{chnCnt},
          numUnitSizeBits{bit_count(bytes)},
          aux{aux} {}
    constexpr AccessorAoSoA(wrapv<layout_e::aos>, void* ptr, unsigned short bytes,
                            unsigned short chnCnt, unsigned short chnNo,
                            unsigned short aux) noexcept
        : base{(char*)ptr + chnNo * bytes},
          numTileBits{(unsigned short)0},
          tileMask{(unsigned short)0},
          chnCnt{chnCnt},
          numUnitSizeBits{bit_count(bytes)},
          aux{aux} {}
/// aosoa
#if 0
    constexpr AccessorAoSoA(wrapv<layout_e::aosoa>, void* ptr, unsigned short bytes,
                            unsigned short tileSize, unsigned short chnCnt,
                            unsigned short aux) noexcept
        : base{ptr},
          numTileBits{bit_count(tileSize)},
          tileMask{tileSize - 1},
          chnCnt{chnCnt},
          numUnitSizeBits{bit_count(bytes)},
          aux{aux} {
      if (tileSize & (tileSize - 1))
        throw std::runtime_error("does not support non power-of-two tile size");
    }
#endif
    constexpr AccessorAoSoA(wrapv<layout_e::aosoa>, void* ptr, unsigned short bytes,
                            unsigned short tileSize, unsigned short chnCnt, unsigned short chnNo,
                            unsigned short aux) noexcept
        : base{(char*)ptr + chnNo * tileSize * bytes},
          numTileBits{bit_count(tileSize)},
          tileMask{(unsigned short)((int)tileSize - 1)},
          chnCnt{chnCnt},
          numUnitSizeBits{bit_count(bytes)},
          aux{aux} {
      // (tileSize & (tileSize - (unsigned short)1))
    }

/// access
#if 0
    constexpr std::intptr_t operator()(unsigned short chnNo, size_t i) const noexcept {
      return (std::intptr_t)((char*)base
                             + (((((i >> numTileBits) * chnCnt + chnNo) << numTileBits)
                                 | (i & tileMask))
                                << numUnitSizeBits));
    }
#endif
    constexpr std::intptr_t operator()(size_t i) const noexcept {
      return (std::intptr_t)(
          (char*)base
          + (((((i >> numTileBits) * chnCnt) << numTileBits) | (i & tileMask)) << numUnitSizeBits));
    }

    void* base;
    unsigned short numTileBits, tileMask, chnCnt, numUnitSizeBits, aux;
  };

}  // namespace zs

namespace zs::cudri {

  /// jitify: preprocess() -> compile() -> link() -> load()
  ZPC_EXTENSION_API std::vector<std::string> load_all_ptx_files_at(const std::string& localPath
                                                                   = "resource");
  ZPC_EXTENSION_API std::string compile_cuda_source_to_ptx(
      std::string_view code, std::string_view name = "unnamed",
      std::vector<std::string_view> additionalOptions = {});
  ZPC_EXTENSION_API void precompile_wranglers(std::string_view progname, std::string_view source);

}  // namespace zs::cudri