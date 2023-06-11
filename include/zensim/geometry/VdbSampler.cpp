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
    auto spls = convert_floatgrid_to_sparse_grid(grid);
    auto lsv = proxy<execspace_e::host>(spls);

    PoissonDisk<float, 3> pd{};
    zs::tie(pd.minCorner, pd.maxCorner) = lsv.getBoundingBox();
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

  std::vector<std::array<float, 3>> sample_from_levelset(const OpenVDBStruct &ls, float dx,
                                                         float ppc) {
    const auto &gridPtr = ls.as<openvdb::FloatGrid::Ptr>();
    vec<float, 3> minCorner{}, maxCorner{};
    {
      openvdb::CoordBBox box = gridPtr->evalActiveVoxelBoundingBox();
      auto corner = box.min();
      auto length = box.max() - box.min();
      auto world_min = gridPtr->indexToWorld(box.min());
      auto world_max = gridPtr->indexToWorld(box.max());
      for (size_t d = 0; d != 3; d++) {
        minCorner(d) = world_min[d];
        maxCorner(d) = world_max[d];
      }
      for (auto &&[dx, dy, dz] : ndrange<3>(2)) {
        auto coord
            = corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0, dz ? length[2] : 0};
        auto pos = gridPtr->indexToWorld(coord);
        for (int d = 0; d != 3; d++) {
          minCorner(d) = pos[d] < minCorner(d) ? pos[d] : minCorner(d);
          maxCorner(d) = pos[d] > maxCorner(d) ? pos[d] : maxCorner(d);
        }
      }
    }
    PoissonDisk<float, 3> pd{};
    pd.minCorner = minCorner;
    pd.maxCorner = maxCorner;
    pd.setDistanceByPpc(dx, ppc);
    auto sample = [gridPtr](const auto &x) -> float {
      return openvdb::tools::BoxSampler::sample(
          gridPtr->tree(), gridPtr->worldToIndex(openvdb::Vec3R{x[0], x[1], x[2]}));
    };
    // auto sample = [&lsv](const auto &x) -> float { return lsv.getSignedDistance(x); };
    return pd.sample([&](const auto &x) { return sample(x) <= 0.f; });
  }

}  // namespace zs