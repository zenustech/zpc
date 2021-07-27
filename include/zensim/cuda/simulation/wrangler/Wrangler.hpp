#pragma once
#include <string>
#include <string_view>
#include <vector>

#include "zensim/meta/Meta.h"

namespace zs {

  enum layout_e : int { aos = 0, soa, aosoa };
  static constexpr auto aos_v = wrapv<layout_e::aos>{};
  static constexpr auto soa_v = wrapv<layout_e::soa>{};
  static constexpr auto aosoa_v = wrapv<layout_e::aosoa>{};

  struct AccessorAoSoA {
    AccessorAoSoA() = default;
    /// aos
    constexpr AccessorAoSoA(wrapv<layout_e::aos>, void* ptr, unsigned short bytes,
                            unsigned short chnCnt) noexcept
        : base{ptr}, tileStride{1}, chnCnt{chnCnt}, unitBytes{bytes} {}
    constexpr AccessorAoSoA(wrapv<layout_e::aos>, void* ptr, unsigned short bytes,
                            unsigned short chnCnt, unsigned short chnNo) noexcept
        : base{(char*)ptr + chnNo * bytes}, tileStride{1}, chnCnt{chnCnt}, unitBytes{bytes} {}
    /// soa
    constexpr AccessorAoSoA(wrapv<layout_e::soa>, void* ptr, unsigned short bytes,
                            std::size_t elementCnt) noexcept
        : base{ptr}, tileStride{elementCnt}, chnCnt{1}, unitBytes{bytes} {}
    constexpr AccessorAoSoA(wrapv<layout_e::soa>, void* ptr, unsigned short bytes,
                            std::size_t elementCnt, unsigned short chnNo) noexcept
        : base{(char*)ptr + (chnNo * elementCnt) * bytes},
          tileStride{elementCnt},
          chnCnt{1},
          unitBytes{bytes} {}
    /// aosoa
    constexpr AccessorAoSoA(wrapv<layout_e::aosoa>, void* ptr, unsigned short bytes,
                            std::size_t tileStride, unsigned short chnCnt) noexcept
        : base{ptr}, tileStride{tileStride}, chnCnt{1}, unitBytes{bytes} {}
    constexpr AccessorAoSoA(wrapv<layout_e::aosoa>, void* ptr, unsigned short bytes,
                            std::size_t tileStride, unsigned short chnCnt,
                            unsigned short chnNo) noexcept
        : base{(char*)ptr + (chnNo * tileStride) * bytes},
          tileStride{tileStride},
          chnCnt{1},
          unitBytes{bytes} {}

    /// access
    constexpr std::intptr_t operator()(unsigned short chnNo, std::size_t i) const {
      return (std::intptr_t)(
          (char*)base
          + ((i / tileStride) * tileStride * chnCnt + chnNo * tileStride + i % tileStride)
                * unitBytes);
    }
    constexpr std::intptr_t operator()(std::size_t i) const {
      return (std::intptr_t)(
          (char*)base + ((i / tileStride) * tileStride * chnCnt + i % tileStride) * unitBytes);
    }

    void* base;
    std::size_t tileStride;
    unsigned short chnCnt, unitBytes;
  };

}  // namespace zs

namespace zs::cudri {

  /// jitify: preprocess() -> compile() -> link() -> load()
  std::vector<std::string> load_all_ptx_files_at(const std::string& dirpath = ZS_PTX_INCLUDE_DIR);
  std::string compile_cuda_source_to_ptx(std::string_view code, std::string_view name = "unnamed",
                                         std::vector<std::string_view> additionalOptions = {});
  void precompile_wranglers(std::string_view progname, std::string_view source);

}  // namespace zs::cudri