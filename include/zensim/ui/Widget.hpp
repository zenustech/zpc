#pragma once
#include "zensim/ZpcImplPattern.hpp"
#include "zensim/ui/Widget.hpp"

namespace zs {

  struct WidgetComponentConcept : virtual ObjectConcept {
    virtual ~WidgetComponentConcept() = default;

    virtual void paint() = 0;
  };
  struct EmptyWidget : WidgetComponentConcept {
    ~EmptyWidget() override = default;
    void paint() override {}
  };
  struct WidgetConcept : virtual HierarchyConcept, WidgetComponentConcept {
    virtual ~WidgetConcept() = default;

    virtual void placeAt(u32 layoutNodeId) {}
  };

}  // namespace zs