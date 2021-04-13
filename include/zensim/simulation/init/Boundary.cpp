#include "Boundary.hpp"

namespace zs {

  BuilderForBoundaryHeightField BuilderForBoundary::heightfield() {
    return BuilderForBoundaryHeightField{this->target()};
  }
  BuilderForBoundaryLevelset BuilderForBoundary::levelset() {
    return BuilderForBoundaryLevelset{this->target()};
  }
  BuilderForBoundaryLevelsetSequence BuilderForBoundary::sequence() {
    return BuilderForBoundaryLevelsetSequence{this->target()};
  }

}  // namespace zs