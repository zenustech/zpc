#pragma once
#include "zensim/types/ImplPattern.hpp"

namespace zs {

  struct WidgetConcept : virtual ObjectConcept {
    virtual ~WidgetConcept() = default;

    virtual void paint() = 0;
  };

}  // namespace zs