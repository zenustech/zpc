#pragma once
#include "zensim/Platform.hpp"
#include "zensim/types/SmallVector.hpp"
namespace zs {

  /// compile-time type inspection
  template <class T> class that_type;
  template <class T> void name_that_type(T &param) {
    // forgot where I picked up this trick...
    that_type<T> tType;
    that_type<decltype(param)> paramType;
  }

  namespace detail {
#if defined(_MSC_VER)
    template <typename T> constexpr auto get_type_raw_str_helper() noexcept { return __FUNCSIG__; }
    constexpr zs::size_t get_type_len_helper(const char *p = nullptr) noexcept {
      if (p == nullptr) return 0;
      size_t i = 0;
      for (; p[i]; ++i)
        ;
      return i;
    }
#endif

    template <typename T> constexpr auto get_type_str_helper() noexcept {
#if defined(_MSC_VER)
      constexpr auto p = get_type_raw_str_helper<T>();
      constexpr auto len = get_type_len_helper(p);
      return zs::BasicSmallString<char, len + 1>{p};
#else
      return zs::BasicSmallString{__PRETTY_FUNCTION__};
#endif
    }
    template <typename T> constexpr auto get_var_type_str_helper(T &&) noexcept {
      return get_type_str_helper<T>();
    }

    struct range_pair {
      size_t l{}, r{};
    };
    constexpr range_pair locate_char_in_str_helper(const char *str, const char lc,
                                                   const char rc) noexcept {
      const char *p = str;
      if (p[0] == '\0') return range_pair{0, 0};
      size_t l{0};
      for (; *p; ++p, ++l)
        if (*p == lc) break;
      size_t r{l + 1}, cnt{1};
      for (++p; *p; ++p, ++r) {
        if (*p == lc)
          cnt++;
        else if (*p == rc)
          cnt--;
        if (cnt == 0) break;
      }
      /// [l, r]
      return range_pair{l, r};
    }

  }  // namespace detail

  template <typename T> constexpr auto get_type() noexcept {
#if defined(ZS_COMPILER_MSVC) && !defined(ZS_COMPILER_INTEL_LLVM)
#  if 0
    constexpr auto typestr = __FUNCSIG__;
    // static_assert(always_false<T>, __FUNCSIG__);
    // using StrT = RM_CVREF_T(typestr[0]);
    // using CharT = typename StrT::char_type;
    using CharT = RM_CVREF_T(typestr[0]);
#  else
    constexpr auto typestr = zs::detail::get_type_str_helper<T>();
    using StrT = RM_CVREF_T(typestr);
    using CharT = typename StrT::char_type;
#  endif
    constexpr auto pair = detail::locate_char_in_str_helper(typestr, '<', '>');
    constexpr size_t head{pair.l + 1};
    constexpr size_t ed{pair.r};
#else
    const auto typestr = zs::detail::get_type_str_helper<T>();
    using StrT = RM_CVREF_T(typestr);
    using CharT = typename StrT::char_type;
    constexpr auto typelength = StrT::nbytes;

#  if defined(__clang__)
    constexpr size_t head = 44;
    constexpr size_t ed = typelength - 2;
#  elif defined(__GNUC__)
    constexpr size_t head = 59;
    constexpr size_t ed = typelength - 2;
#  elif defined(__CUDACC__)
    constexpr size_t head = 58;
    constexpr size_t ed = typelength - 2;
#  else
    static_assert(always_false<T>, "unknown compiler for handling compile-time type reflection");
#  endif
#endif
    constexpr size_t length = ed - head;

    BasicSmallString<CharT, length + 1> ret{};
    for (size_t i = 0; i != length; ++i) ret[i] = typestr[i + head];
    ret[length] = '\0';
    return ret;
  }
  template <typename T> constexpr auto get_var_type(T &&) noexcept { return get_type<T>(); }

#if 0
  template <typename CharT, size_t N>
  auto convert_char_array_to_string(const BasicSmallString<CharT, N> &str) noexcept {
    return std::basic_string<CharT>{begin(str), end(str)};
  }
#endif
  template <typename T> constexpr auto get_var_type_str(T &&v) noexcept {
    // return convert_char_array_to_string(get_var_type(FWD(v)));
    return get_var_type(FWD(v));
  }
  template <typename T> constexpr auto get_type_str() noexcept {
    // return convert_char_array_to_string(get_type<T>());
    return get_type<T>();
  }

}  // namespace zs
