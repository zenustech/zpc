#pragma once
#include <string>

#include "zensim/container/DenseGrid.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Tuple.h"

namespace zs {

  tuple<DenseGrid<float, int, 3>, vec<float, 3>, vec<float, 3>> readPhiFromVdbFile(
      const std::string &fn, float dx);

  tuple<DenseGrid<float, int, 3>, DenseGrid<vec<float, 3>, int, 3>, vec<float, 3>, vec<float, 3>>
  readPhiVelFromVdbFile(const std::string &fn, float dx);

}  // namespace zs
