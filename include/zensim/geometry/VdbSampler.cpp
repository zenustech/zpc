#include "VdbSampler.h"

#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/math/Mat.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Tuple.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/util/Util.h>

#include <algorithm>
#include <numeric>

#include "PoissonDisk.hpp"
#include "zensim/Logger.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/Property.h"

namespace zs {

  std::vector<std::array<float, 3>> sample_from_floatgrid(const OpenVDBStruct &grid, float dx,
                                                          float ppc) {
    auto spls = convert_floatgrid_to_sparse_levelset(grid);
    auto lsv = proxy<execspace_e::host>(spls);

    PoissonDisk<float, 3> pd{};
    std::tie(pd.minCorner, pd.maxCorner) = lsv.getBoundingBox();
    pd.setDistanceByPpc(dx, ppc);

    auto sample = [&lsv](const auto &x) -> float { return lsv.getSignedDistance(x); };
    return pd.sample([&](const auto &x) { return sample(x) <= 0.f; });
  }

  std::vector<std::array<float, 3>> sample_from_vdb_file(const std::string &filename, float dx,
                                                         float ppc) {
    return sample_from_floatgrid(load_floatgrid_from_vdb_file(filename), dx, ppc);
  }

  std::vector<std::array<float, 3>> sample_from_obj_file(const std::string &filename, float dx,
                                                         float ppc) {
    return sample_from_floatgrid(load_floatgrid_from_mesh_file(filename, dx), dx, ppc);
  }

}  // namespace zs