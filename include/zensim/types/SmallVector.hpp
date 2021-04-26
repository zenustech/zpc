#pragma once
#include <string>

#include "zensim/meta/Meta.h"

namespace zs {

  struct SmallString {
    using size_type = char;
    static constexpr auto nbytes = 4 * sizeof(void *);  ///< 4 * 8 - 1 = 31 bytes (chars)

    constexpr SmallString() noexcept : buf{} {
      for (auto &c : buf) c = '\0';
    }
    constexpr SmallString(const char tmp[]) : buf{} {
      size_type i = 0;
      for (; i + 1 < nbytes && tmp[i]; ++i) buf[i] = tmp[i];
      buf[i] = '\0';
    }
    SmallString(const std::string &str) {
      size_type n = str.size() < nbytes ? str.size() : nbytes - 1;
      buf[n] = '\0';
      for (--n; n >= 0; --n) buf[n] = str[n];
    }
    constexpr SmallString(const SmallString &) noexcept = default;
    constexpr SmallString &operator=(const SmallString &) noexcept = default;
    constexpr SmallString(SmallString &&) noexcept = default;
    constexpr SmallString &operator=(SmallString &&) noexcept = default;

    std::string asString() const { return std::string{buf}; }
    constexpr const char *asChars() const noexcept { return buf; }
    constexpr operator const char *() const noexcept { return buf; }

    char buf[nbytes];
  };

}  // namespace zs