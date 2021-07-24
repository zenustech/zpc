#pragma once
#include "zensim/TypeAlias.hpp"

namespace zs {

  template <typename T, size_t... Ns> struct NDArray;
  template <typename T, size_t N> struct NDArray<T, N> {
    static constexpr size_t extent = N;

    constexpr T &operator[](size_t i) { return data[i]; }
    constexpr const T &operator[](size_t i) const { return data[i]; }
    constexpr T &operator()(size_t i) { return data[i]; }
    constexpr const T &operator()(size_t i) const { return data[i]; }
    constexpr ptrdiff_t address(size_t i) const noexcept { return data + i; }
    T data[N];
  };
  template <typename T, size_t N, size_t... Ns> struct NDArray<T, N, Ns...> {
    static constexpr size_t extent = N * ... * Ns;
    constexpr decltype(auto) operator[](size_t i) { return data[i]; }
    constexpr decltype(auto) operator[](size_t i) const { return data[i]; }
    constexpr decltype(auto) operator()(size_t i) { return data[i / (Ns * ...)][i % (Ns * ...)]; }
    constexpr decltype(auto) operator()(size_t i) const {
      return data[i / (Ns * ...)][i % (Ns * ...)];
    }
    constexpr ptrdiff_t address(size_t i) const noexcept {
      return data[i / (Ns * ...)].address(i % (Ns * ...));
    }
    NDArray<T, Ns...> data[N];
  };

  struct StringTag {
    using size_type = unsigned char;
    static constexpr auto nbytes = 4 * sizeof(void *);  ///< 4 * 8 - 1 = 31 bytes (chars)

    constexpr StringTag() noexcept : buf{} {
      for (auto &c : buf) c = '\0';
    }
    constexpr StringTag(const char tmp[]) : buf{} {
      size_type i = 0;
      for (; i + 1u < nbytes && tmp[i]; ++i) buf[i] = tmp[i];
      buf[i] = '\0';
    }
    constexpr StringTag(const StringTag &) noexcept = default;
    constexpr StringTag &operator=(const StringTag &) noexcept = default;
    constexpr StringTag(StringTag &&) noexcept = default;
    constexpr StringTag &operator=(StringTag &&) noexcept = default;

    constexpr bool operator==(const StringTag &str) const noexcept {
      for (size_type i = 0; i < nbytes && buf[i] && str.buf[i]; ++i)
        if (buf[i] != str.buf[i]) return false;
      return true;
    }
    constexpr bool operator==(const char str[]) const noexcept {
      for (size_type i = 0; i < nbytes && buf[i] && str[i]; ++i)
        if (buf[i] != str[i]) return false;
      return true;
    }

    constexpr const char *asChars() const noexcept { return buf; }
    constexpr operator const char *() const noexcept { return buf; }

    char buf[nbytes];
  };

  template <typename T, size_t S = 0> struct Accessor {
    enum layout_e { aos = 0, soa, aosoa };

    template <typename DstT, typename SrcT> constexpr decltype(auto) get(SrcT &&val) {
      static_assert(sizeof(val) == sizeof(DstT),
                    "Source Type and Destination Type must be of the same size");
      static_assert(alignof(val) == alignof(DstT),
                    "Source Type and Destination Type must be of the same alignment");
      return reinterpret_cast<DstT &>(val);
    }
    template <typename DstT, typename SrcT> constexpr decltype(auto) get(SrcT &&val) const {
      static_assert(sizeof(val) == sizeof(DstT),
                    "Source Type and Destination Type must be of the same size");
      static_assert(alignof(val) == alignof(DstT),
                    "Source Type and Destination Type must be of the same alignment");
      return reinterpret_cast<DstT const &>(val);
    }
  };
  template <typename Dst = T> constexpr Dst &get(int ch, size_t i) {
    switch (layout) {
      case aos:
        return *(base + i * stride + ch);
      case soa:
        return *(base + ch * stride + i);
    }
    /// aosoa
    // S: length of a tile
    // stride: sizeof(T) * S * numChannels. the total bytes of a tile
    return *(base + (i / S) * stride + ch * S + (i % S));
  }
  template <typename Dst = T> constexpr Dst get(int ch, size_t i) const {
    switch (layout) {
      case aos:
        return *(base + i * stride + ch);
      case soa:
        return *(base + ch * stride + i);
    }
    /// aosoa
    // S: length of a tile
    // stride: sizeof(T) * S * numChannels. the total bytes of a tile
    return *(base + (i / S) * stride + ch * S + (i % S));
  }

  T *base;
  size_t stride;  // this is for soa and aos
  const layout_e layout;
};

struct ChannelDescrs {
  struct ChannelDescr {
    StringTag tag;
    int cnt;     // channel dim
    int offset;  // channel offset
  };
  constexpr ChannelDescr *find(const char str[]) {
    for (int i = 0; i != n; ++i)
      if (descrs[i].tag == str) return descrs + i;
    return nullptr;
  }
  template <size_t... Ns, typename T, size_t S>
  constexpr NDArray<T, Ns...> get(const Accessor &accessor, int ch, size_t i) const {
    constexpr size_t extent = (Ns * ...);
    NDArray<T, Ns...> ret{};
    for (size_t d = 0; d != extent; ++d) ret(d) = accessor.get(ch + d, i);
  }
  template <size_t... Ns, typename T, size_t S>
  constexpr NDArray<T, Ns...> get(const Accessor &accessor, const StringTag &tag, size_t i) const {
    constexpr size_t extent = (Ns * ...);
    ChannelDescr *descr = find(tag);
    if (descr != nullptr && descr->cnt == extent) {
      const auto ch = descr->offset;
      return get < Ns...(accessor, ch, i);
    }
    return NDArray<T, Ns...>{};
  }

  ChannelDescr *descrs;
  int n;
};

}  // namespace zs