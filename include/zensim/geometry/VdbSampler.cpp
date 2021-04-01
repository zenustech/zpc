#include "VdbSampler.h"

#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/math/Mat.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Tuple.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/util/Util.h>

#include <algorithm>
#include <numeric>

#include "LevelSet.h"
#include "PoissonDisk.hpp"
#include "zensim/Logger.hpp"
#include "zensim/memory/Allocator.h"

namespace zs {

  void initialize_openvdb() { openvdb::initialize(); }

  openvdb::FloatGrid::Ptr readFileToLevelset(const std::string &fn) {
    openvdb::io::File file(fn);
    file.open();
    openvdb::GridPtrVecPtr my_grids = file.getGrids();
    file.close();
    int count = 0;
    typename openvdb::FloatGrid::Ptr grid;
    for (openvdb::GridPtrVec::iterator iter = my_grids->begin(); iter != my_grids->end(); ++iter) {
      openvdb::GridBase::Ptr it = *iter;
      if ((*iter)->isType<openvdb::FloatGrid>()) {
        grid = openvdb::gridPtrCast<openvdb::FloatGrid>(*iter);
        count++;
        /// display meta data
        for (openvdb::MetaMap::MetaIterator it = grid->beginMeta(); it != grid->endMeta(); ++it) {
          const std::string &name = it->first;
          openvdb::Metadata::Ptr value = it->second;
          std::string valueAsString = value->str();
          std::cout << name << " = " << valueAsString << std::endl;
        }
      }
    }
    ZS_WARN_IF(count != 1, "Vdb file to load should only contain one levelset.");
    return grid;
  }

  std::vector<std::array<float, 3>> sample_from_levelset(openvdb::FloatGrid::Ptr vdbls, float dx,
                                                         float ppc) {
    using TV = vec<float, 3>;
    using STDV3 = std::array<float, 3>;
    TV minCorner, maxCorner;
    openvdb::CoordBBox box = vdbls->evalActiveVoxelBoundingBox();
    auto world_min = vdbls->indexToWorld(box.min());
    auto world_max = vdbls->indexToWorld(box.max());
    for (size_t d = 0; d < 3; d++) {
      minCorner(d) = world_min[d];
      maxCorner(d) = world_max[d];
    }

    PoissonDisk<float, 3> pd{};
    pd.minCorner = minCorner;
    pd.maxCorner = maxCorner;
    pd.setDistanceByPpc(dx, ppc);

    auto sample = [&vdbls](const TV &X_input) -> float {
      TV X = X_input;
      openvdb::tools::GridSampler<typename openvdb::FloatGrid::TreeType, openvdb::tools::BoxSampler>
          interpolator(vdbls->constTree(), vdbls->transform());
      openvdb::math::Vec3<float> P(X(0), X(1), X(2));
      float phi = interpolator.wsSample(P);  // ws denotes world space
      return (float)phi;
    };

    std::vector<STDV3> samples = pd.sample([&](const TV &x) { return sample(x) <= 0.f; });
    return samples;
  }

  std::vector<std::array<float, 3>> sample_from_vdb_file(const std::string &filename, float dx,
                                                         float ppc) {
    return sample_from_levelset(readFileToLevelset(filename), dx, ppc);
  }

  openvdb::FloatGrid::Ptr readMeshToLevelset(const std::string &filename, float h) {
    using Vec3ui = openvdb::math::Vec3ui;
    std::vector<openvdb::Vec3f> vertList;
    std::vector<Vec3ui> faceList;
    std::ifstream infile(filename);
    if (!infile) {
      std::cerr << "Failed to open. Terminating.\n";
      exit(-1);
    }

    int ignored_lines = 0;
    std::string line;

    while (!infile.eof()) {
      std::getline(infile, line);
      auto ed = line.find_first_of(" ");
      if (line.substr(0, ed) == std::string("v")) {
        std::stringstream data(line);
        char c;
        openvdb::Vec3f point;
        data >> c >> point[0] >> point[1] >> point[2];
        vertList.push_back(point);
      } else if (line.substr(0, ed) == std::string("f")) {
        std::stringstream data(line);
        char c;
        int v0, v1, v2;
        data >> c >> v0 >> v1 >> v2;
        faceList.push_back(Vec3ui(v0 - 1, v1 - 1, v2 - 1));
      } else {
        ++ignored_lines;
      }
    }
    infile.close();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    points.resize(vertList.size());
    triangles.resize(faceList.size());
    tbb::parallel_for(0, (int)vertList.size(), 1, [&](int p) {
      points[p] = openvdb::Vec3s(vertList[p][0], vertList[p][1], vertList[p][2]);
    });
    tbb::parallel_for(0, (int)faceList.size(), 1, [&](int p) {
      triangles[p] = openvdb::Vec3I(faceList[p][0], faceList[p][1], faceList[p][2]);
    });
    openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
        *openvdb::math::Transform::createLinearTransform(h), points, triangles, 3.0);
    return grid;
  }

  std::vector<std::array<float, 3>> sample_from_obj_file(const std::string &filename, float dx,
                                                         float ppc) {
    return sample_from_levelset(readMeshToLevelset(filename, dx), dx, ppc);
  }

}  // namespace zs