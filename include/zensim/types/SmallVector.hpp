#pragma once
#include <string>

#include "zensim/meta/Meta.h"

namespace zs {

  struct SmallString {
    using size_type = unsigned char;
    static constexpr auto nbytes = 4 * sizeof(void *);  ///< 4 * 8 - 1 = 31 bytes (chars)

    constexpr SmallString() noexcept : buf{} {
      for (auto &c : buf) c = '\0';
    }
    constexpr SmallString(const char tmp[]) : buf{} {
      size_type i = 0;
      for (; i + 1u != nbytes && tmp[i]; ++i) buf[i] = tmp[i];
      buf[i] = '\0';
    }
    SmallString(const std::string &str) {
      size_type n = str.size() < nbytes ? str.size() : nbytes - 1;
      buf[n] = '\0';
      for (; (--n) != 0;) buf[n] = str[n];
    }
    constexpr SmallString(const SmallString &) noexcept = default;
    constexpr SmallString &operator=(const SmallString &) noexcept = default;
    constexpr SmallString(SmallString &&) noexcept = default;
    constexpr SmallString &operator=(SmallString &&) noexcept = default;

    constexpr bool operator==(const SmallString &str) const noexcept {
      size_type i = 0;
      for (; i != nbytes && buf[i] && str.buf[i]; ++i)
        if (buf[i] != str.buf[i]) return false;
      if (!(buf[i] || str.buf[i])) return true;
      return false;
    }
    constexpr bool operator==(const char str[]) const noexcept {
      size_type i = 0;
      for (; i != nbytes && buf[i] && str[i]; ++i)
        if (buf[i] != str[i]) return false;
      if (!(buf[i] || str[i])) return true;
      return false;
    }

    std::string asString() const { return std::string{buf}; }
    constexpr const char *asChars() const noexcept { return buf; }
    constexpr operator const char *() const noexcept { return buf; }
    constexpr size_type size() const noexcept {
      size_type i{0};
      for (; i != nbytes && buf[i]; ++i)
        ;
      return i;
    }

    char buf[nbytes];
  };

}  // namespace zs