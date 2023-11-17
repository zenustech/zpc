#pragma once
#include <memory>
#include <vector>

#include "zensim/ZpcImplPattern.hpp"

namespace zs {

  struct HierarchyConcept : virtual ObjectConcept {
    virtual ~HierarchyConcept() = default;

    template <typename T, enable_if_t<std::is_base_of_v<ObjectConcept, T>> = 0>
    T *addChild(std::unique_ptr<T> ch) {
      if constexpr (std::is_base_of_v<HierarchyConcept, T>) ch->_parent = this;
      auto ret = ch.get();
      _children.push_back(std::move(ch));
      return ret;
    }
    HierarchyConcept *parent() const {  // get parent widget, may return null for the root widget
      return _parent;
    }
    std::vector<std::unique_ptr<ObjectConcept>> &children() { return _children; }
    const std::vector<std::unique_ptr<ObjectConcept>> &children() const { return _children; }

  protected:
    std::vector<std::unique_ptr<ObjectConcept>> _children{};
    HierarchyConcept *_parent{nullptr};
  };

}  // namespace zs