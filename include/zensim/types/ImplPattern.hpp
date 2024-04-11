#pragma once
#include <memory>
#include <vector>

#include "zensim/ZpcImplPattern.hpp"

namespace zs {

  template <typename T> using Shared = std::shared_ptr<T>;
  template <typename T> using Weak = std::weak_ptr<T>;
  template <typename T, typename... Args> constexpr Shared<T> make_shared(Args &&...args) {
    return std::make_shared<T>(zs::forward<Args>(args)...);
  }
  template <typename T> using Unique = std::unique_ptr<T>;
  template <typename T, typename... Args> constexpr Unique<T> make_unique(Args &&...args) {
    return std::make_unique<T>(zs::forward<Args>(args)...);
  }

}  // namespace zs