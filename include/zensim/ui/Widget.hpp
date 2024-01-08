#pragma once
#include "zensim/types/ImplPattern.hpp"
#include "zensim/ui/Widget.hpp"

namespace zs {

  struct WidgetComponentConcept : virtual ObjectConcept {
    virtual ~WidgetComponentConcept() = default;

    virtual void paint() = 0;
  };
  struct WidgetConcept : virtual HierarchyConcept, WidgetComponentConcept {
    virtual ~WidgetConcept() = default;

    virtual void placeAt(u32 layoutNodeId) {}
  };

}  // namespace zs