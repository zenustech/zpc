#pragma once

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
    template <typename T> constexpr auto get_type_str_helper() noexcept {
#if defined(_MSC_VER)
      return zs::BasicSmallString<char, sizeof(__FUNCSIG__) + 1>{__FUNCSIG__};
#else
      return zs::BasicSmallString<char, sizeof(__PRETTY_FUNCTION__) + 1>{__PRETTY_FUNCTION__};
#endif
    }
    template <typename T> constexpr auto get_var_type_str_helper(T &&) noexcept {
      return get_type_str_helper<T>();
    }

#if 0
    struct range_pair {
      size_t l{}, r{};
    };
    template <typename CharT, size_t N>
    constexpr range_pair locate_char_in_str_helper(const BasicSmallString<CharT, N> &str, const char lc,
                                                   const char rc) noexcept {
      const char *p = str.buf;
      if (p[0]== '\0') return range_pair{0, 0};
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
#endif

  }  // namespace detail

  template <typename T> constexpr auto get_type() noexcept {
    const auto typestr = zs::detail::get_type_str_helper<T>();
    using StrT = RM_CVREF_T(typestr);
    using CharT = typename StrT::char_type;
    constexpr auto typelength = StrT::nbytes;

#if defined(_MSC_VER)
    constexpr size_t head = 45;
    constexpr size_t ed = typelength - 18;
#elif defined(__clang__)
    constexpr size_t head = 44;
    constexpr size_t ed = typelength - 3;
#elif defined(__GNUC__)
    constexpr size_t head = 58;
    constexpr size_t ed = typelength - 3;
#elif defined(__CUDACC__)
    constexpr size_t head = 58;
    constexpr size_t ed = typelength - 3;
#else
    static_assert(always_false<T>, "unknown compiler for handling compile-time type reflection");
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
  template <typename T> auto get_var_type_str(T &&v) noexcept {
    // return convert_char_array_to_string(get_var_type(FWD(v)));
    return get_var_type(FWD(v));
  }
  template <typename T> auto get_type_str() noexcept {
    // return convert_char_array_to_string(get_type<T>());
    return get_type<T>();
  }

}  // namespace zs
