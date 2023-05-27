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
    template <typename T> constexpr auto get_var_type_str_helper(T &&) noexcept {
#if defined(_MSC_VER)
      return __FUNCSIG__;
#else
      return __PRETTY_FUNCTION__;
#endif
    }

    template <typename T> constexpr auto get_type_str_helper() noexcept {
#if defined(_MSC_VER)
      return __FUNCSIG__;
#else
      return __PRETTY_FUNCTION__;
#endif
    }

    constexpr size_t get_type_len_helper(const char *p = nullptr) noexcept {
      if (p == nullptr) return (size_t)0;
      size_t i = 0;
      for (; p[i]; ++i)
        ;
      return i;
    }

    struct range_pair {
      size_t l{}, r{};
    };
    constexpr range_pair locate_char_in_str_helper(const char *p, const char lc,
                                                   const char rc) noexcept {
      if (p == nullptr) return range_pair{0, 0};
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

    template <zs::size_t head = 0, size_t length = 0, typename T>
    constexpr auto get_var_type_substr(T &&) noexcept {
      constexpr auto typestr = get_type_str_helper<T>();
      using CharT = remove_const_t<remove_pointer_t<decltype(typestr)>>;
      constexpr auto typelength = get_type_len_helper(typestr);
      static_assert(typelength > head, "sub-string should not exceed the whole string!");
      constexpr auto substrLength
          = (length == 0 ? typelength - head
                         : (length < (typelength - head) ? length : (typelength - head)));
      SmallStringImpl<CharT, substrLength + 1> ret{};
      for (size_t i = 0; i != substrLength; ++i) ret[i] = typestr[i + head];
      ret[substrLength] = '\0';
      return ret;
    }
  }  // namespace detail

  template <typename T> constexpr auto get_type() noexcept {
    constexpr auto typestr = detail::get_type_str_helper<T>();
    using CharT = remove_const_t<remove_pointer_t<decltype(typestr)>>;
    // constexpr auto typelength = detail::get_type_len_helper(typestr);

#if defined(_MSC_VER)
    constexpr auto pair = detail::locate_char_in_str_helper(typestr, '<', '>');
    constexpr size_t head{pair.l + 1};
    constexpr size_t length{pair.r - head};
#elif defined(__clang__)
    constexpr auto pair = detail::locate_char_in_str_helper(typestr, '[', ']');
    constexpr size_t head{pair.l + 5};
    constexpr size_t length{pair.r - head};
#elif defined(__GNUC__)
    constexpr auto pair = detail::locate_char_in_str_helper(typestr, '[', ']');
    constexpr size_t head{pair.l + 10};
    constexpr size_t length{pair.r - head};
#endif

    SmallStringImpl<CharT, length + 1> ret{};
    for (size_t i = 0; i != length; ++i) ret[i] = typestr[i + head];
    ret[length] = '\0';
    return ret;
  }
  template <typename T> constexpr auto get_var_type(T &&) noexcept { return get_type<T>(); }

  template <typename CharT, size_t N>
  auto convert_char_array_to_string(const SmallStringImpl<CharT, N> &str) noexcept {
    return std::basic_string<CharT>{begin(str), end(str)};
  }
  template <typename T> auto get_var_type_str(T &&v) noexcept {
    // return convert_char_array_to_string(get_var_type(FWD(v)));
    return get_var_type(FWD(v));
  }
  template <typename T> auto get_type_str() noexcept {
    // return convert_char_array_to_string(get_type<T>());
    return get_type<T>();
  }

}  // namespace zs
