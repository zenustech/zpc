#pragma once
#include <string>
#include <string_view>

#include "zensim/meta/Meta.h"

namespace zs {

  // null-terminated string
  struct SmallString {
    using char_type = char;
    static_assert(std::is_trivial_v<char_type> && std::is_standard_layout_v<char_type>,
                  "char type is not trivial and in standard-layout.");
    using size_type = std::size_t;
    static constexpr auto nbytes = 4 * sizeof(void *);  ///< 4 * 8 - 1 = 31 bytes (chars)

    constexpr SmallString() noexcept : buf{} {
      for (auto &c : buf) c = '\0';
    }
    constexpr SmallString(const char tmp[]) : buf{} {
      size_type i = 0;
      for (; i + (size_type)1 != nbytes && tmp[i]; ++i) buf[i] = tmp[i];
      buf[i] = '\0';
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (i == nbytes - 1 && tmp[i]) {
        printf("the str [%s]\' size exceeds smallstring maximum length [%d]!\n", tmp, (int)nbytes);
      }
#endif
    }
    SmallString(const std::string &str) noexcept {
      size_type n = str.size() < nbytes ? str.size() : nbytes - 1;
      buf[n] = '\0';
      for (size_type i = 0; i != n; ++i) buf[i] = str[i];
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (str.size() >= nbytes) {
        printf("the str [%s]\' size exceeds smallstring maximum length [%d]!\n", str.c_str(),
               (int)nbytes);
      }
#endif
    }
    constexpr SmallString(const SmallString &) noexcept = default;
    constexpr SmallString &operator=(const SmallString &) noexcept = default;
    constexpr SmallString(SmallString &&) noexcept = default;
    constexpr SmallString &operator=(SmallString &&) noexcept = default;

    constexpr bool operator==(const char str[]) const noexcept {
      size_type i = 0, sz = size();
      for (; i != sz && str[i]; ++i)
        if (buf[i] != str[i]) return false;
      if (!(buf[i] || str[i])) return true;
      return false;
    }
#if 0
    constexpr bool operator==(const std::string_view str) const noexcept {
      size_type i = 0;
      for (; buf[i] && i != str.size(); ++i)
        if (buf[i] != str[i]) return false;
      if (!(buf[i] || str[i])) return true;
      return false;
    }
#endif

    std::string asString() const { return std::string{buf}; }
    constexpr const char *asChars() const noexcept { return buf; }
    constexpr operator const char *() const noexcept { return buf; }
    constexpr size_type size() const noexcept {
      size_type i = 0;
      for (; buf[i]; ++i)
        ;
      return i;
    }
    friend constexpr SmallString operator+(const SmallString &a, const SmallString &b) noexcept {
      SmallString ret{};
      size_type i = 0;
      for (; i + (size_type)1 != nbytes && a.buf[i]; ++i) ret.buf[i] = a.buf[i];
      for (size_type j = 0; i + (size_type)1 != nbytes && b.buf[j]; ++i, ++j) ret.buf[i] = a.buf[j];
      ret.buf[i] = '\0';
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (a.size() + b.size() >= nbytes) {
        printf("concatenating str [%s] and str [%s] exceeds smallstring maximum length [%d]!\n",
               a.asChars(), b.asChars(), (int)nbytes);
      }
#endif
      return ret;
    }

    char_type buf[nbytes];
  };

  /// property tag
  struct PropertyTag {
    SmallString name;
    int numChannels;
  };

}  // namespace zs