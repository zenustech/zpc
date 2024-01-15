#pragma once
#include <memory>
#include <vector>

#include "zensim/ZpcImplPattern.hpp"

namespace zs {

  template <typename T> using Shared = std::shared_ptr<T>;
  template <typename T, typename... Args> constexpr Shared<T> make_shared(Args &&...args) {
    return std::make_shared<T>(zs::forward<Args>(args)...);
  }
  template <typename T> using Unique = std::unique_ptr<T>;
  template <typename T, typename... Args> constexpr Unique<T> make_unique(Args &&...args) {
    return std::make_unique<T>(zs::forward<Args>(args)...);
  }

  struct HierarchyConcept : virtual ObjectConcept {
    virtual ~HierarchyConcept() = default;

    HierarchyConcept *parent() const {  // get parent widget, may return null for the root widget
      return _parent;
    }
    HierarchyConcept *&parent() {  // get parent widget, may return null for the root widget
      return _parent;
    }

  protected:
    HierarchyConcept *_parent{nullptr};
  };

}  // namespace zs