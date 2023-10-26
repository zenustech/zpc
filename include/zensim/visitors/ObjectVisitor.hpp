#include "zensim/ZpcImplPattern.hpp"
#include <cstdio>

namespace zs {

  struct ObjectVisitor {
    virtual void visit(VisitableObjectConcept &);
    /// TBD: add interfaces for other visitees inheriting from VisitableObjectConcept
  };

  template <typename Derived, typename Parent = VisitableObjectConcept>
  struct InheritVisitableObject : virtual Parent {
    void accept(zs::ObjectVisitor &visitor) override {
      visitor.visit(*static_cast<Derived *>(this));
    }
  };
  template <typename Derived, typename... Parents> struct InheritVisitableObjects
      : virtual Parents... {
    void accept(zs::ObjectVisitor &visitor) override {
      visitor.visit(*static_cast<Derived *>(this));
    }
  };

}  // namespace zs