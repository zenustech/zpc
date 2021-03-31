#include "VdbLevelSet.h"

#include <zensim/Logger.hpp>
//#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <zensim/execution/Concurrency.h>

namespace zs {

  // https://www.openvdb.org/documentation/doxygen/codeExamples.html#sGettingMetadata
  tuple<DenseGrid<float, int, 3>, vec<float, 3>, vec<float, 3>> readPhiFromVdbFile(
      const std::string &fn, float dx) {
    constexpr int dim = 3;
    using TV = vec<float, dim>;
    using IV = vec<int, dim>;
    using GridT = typename openvdb::FloatGrid;
    using TreeT = typename GridT::TreeType;
    using VelGridT = typename openvdb::FloatGrid;
    using VelTreeT = typename VelGridT::TreeType;

    openvdb::io::File file(fn);
    file.open();
    openvdb::GridPtrVecPtr my_grids = file.getGrids();
    file.close();
    int count = 0;
    typename GridT::Ptr grid;
    for (openvdb::GridPtrVec::iterator iter = my_grids->begin(); iter != my_grids->end(); ++iter) {
      openvdb::GridBase::Ptr it = *iter;
      if ((*iter)->isType<GridT>()) {
        grid = openvdb::gridPtrCast<GridT>(*iter);
        count++;
        /// display meta data
        for (openvdb::MetaMap::MetaIterator it = grid->beginMeta(); it != grid->endMeta(); ++it) {
          const std::string &name = it->first;
          openvdb::Metadata::Ptr value = it->second;
          std::string valueAsString = value->str();
          std::cout << name << " = " << valueAsString << std::endl;
        }
      } else if ((*iter)->isType<VelGridT>()) {
        auto velgrid = openvdb::gridPtrCast<VelGridT>((*iter));
        for (openvdb::MetaMap::MetaIterator it = velgrid->beginMeta(); it != velgrid->endMeta();
             ++it) {
          const std::string &name = it->first;
          openvdb::Metadata::Ptr value = it->second;
          std::string valueAsString = value->str();
          std::cout << name << " = " << valueAsString << std::endl;
        }
      }
    }
    ZS_WARN_IF(count != 1, "Vdb file to load should only contain one levelset.");

    /// bounding box
    TV bmin, bmax;
    openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
    auto world_min = grid->indexToWorld(box.min());
    auto world_max = grid->indexToWorld(box.max());

    for (size_t d = 0; d < dim; d++) {
      bmin(d) = world_min[d];
      bmax(d) = world_max[d];
    }

    vec<int, 3> extents = ((bmax - bmin) / dx).cast<int>() + 1;

    /// phi
    auto sample = [&grid](const TV &X_input) -> float {
      TV X = X_input;
      openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
          grid->constTree(), grid->transform());
      openvdb::math::Vec3<float> P(X(0), X(1), X(2));
      float phi = interpolator.wsSample(P);  // ws denotes world space
      return (float)phi;
    };
    printf(
        "Vdb file domain [%f, %f, %f] - [%f, %f, %f]; resolution {%d, %d, "
        "%d}\n",
        bmin(0), bmin(1), bmin(2), bmax(0), bmax(1), bmax(2), extents(0), extents(1), extents(2));
    // DenseGrid<float, int, 3> phi(extents, std::numeric_limits<float>::min());
    DenseGrid<float, int, 3> phi(extents, 2 * dx);
#pragma omp parallel for
    for (int x = 0; x < extents(0); ++x)
      for (int y = 0; y < extents(1); ++y)
        for (int z = 0; z < extents(2); ++z) {
          IV X = vec<int, 3>{x, y, z};
          TV x = X * dx + bmin;
          phi(X) = sample(x);
        }
    return zs::make_tuple(phi, bmin, bmax);
  }

  tuple<DenseGrid<float, int, 3>, DenseGrid<vec<float, 3>, int, 3>, vec<float, 3>, vec<float, 3>>
  readPhiVelFromVdbFile(const std::string &fn, float dx) {
    constexpr int dim = 3;
    using TV = vec<float, dim>;
    using IV = vec<int, dim>;
    using PhiGridT = typename openvdb::FloatGrid;
    using PhiTreeT = typename PhiGridT::TreeType;
    // using VelGridT = typename openvdb::Grid<
    //    typename openvdb::tree::Tree4<openvdb::Vec3f, 5, 4, 3>::Type>;
    using VelGridT = typename openvdb::Vec3fGrid;
    using VelTreeT = typename VelGridT::TreeType;

    openvdb::io::File file(fn);
    file.open();
    openvdb::GridPtrVecPtr my_grids = file.getGrids();
    file.close();
    typename PhiGridT::Ptr phigrid;
    typename VelGridT::Ptr velgrid;
    for (openvdb::GridPtrVec::iterator iter = my_grids->begin(); iter != my_grids->end(); ++iter) {
      if ((*iter)->isType<PhiGridT>()) {
        if (openvdb::gridPtrCast<PhiGridT>(*iter)->metaValue<std::string>("name") == "surface") {
          phigrid = openvdb::gridPtrCast<PhiGridT>(*iter);
          for (openvdb::MetaMap::MetaIterator it = phigrid->beginMeta(); it != phigrid->endMeta();
               ++it) {
            const std::string &name = it->first;
            openvdb::Metadata::Ptr value = it->second;
            std::string valueAsString = value->str();
            std::cout << name << " = " << valueAsString << std::endl;
          }
        }
      } else if ((*iter)->isType<VelGridT>()) {
        if (openvdb::gridPtrCast<VelGridT>(*iter)->metaValue<std::string>("name") == "vel") {
          velgrid = openvdb::gridPtrCast<VelGridT>(*iter);
          for (openvdb::MetaMap::MetaIterator it = velgrid->beginMeta(); it != velgrid->endMeta();
               ++it) {
            const std::string &name = it->first;
            openvdb::Metadata::Ptr value = it->second;
            std::string valueAsString = value->str();
            std::cout << name << " = " << valueAsString << std::endl;
          }
        }
      }
    }

    /// bounding box
    TV bmin, bmax;
    openvdb::CoordBBox box = phigrid->evalActiveVoxelBoundingBox();
    auto world_min = phigrid->indexToWorld(box.min());
    auto world_max = phigrid->indexToWorld(box.max());

    for (size_t d = 0; d < dim; d++) {
      bmin(d) = world_min[d];
      bmax(d) = world_max[d];
    }

    vec<int, 3> extents = ((bmax - bmin) / dx).cast<int>() + 1;

    /// phi
    auto sample = [&phigrid, &velgrid](const TV &X_input) -> std::tuple<float, openvdb::Vec3f> {
      TV X = TV::zeros();
      for (int d = 0; d < dim; d++) X(d) = X_input(d);
      openvdb::tools::GridSampler<PhiTreeT, openvdb::tools::BoxSampler> phi_interpolator(
          phigrid->constTree(), phigrid->transform());
      openvdb::tools::GridSampler<VelTreeT, openvdb::tools::BoxSampler> vel_interpolator(
          velgrid->constTree(), velgrid->transform());
      openvdb::math::Vec3<float> P(X(0), X(1), X(2));
      float phi = phi_interpolator.wsSample(P);  // ws denotes world space
      auto vel = vel_interpolator.wsSample(P);   // ws denotes world space
      return std::make_tuple((float)phi, vel);
    };
    printf(
        "Vdb file domain [%f, %f, %f] - [%f, %f, %f]; resolution {%d, %d, "
        "%d}\n",
        bmin(0), bmin(1), bmin(2), bmax(0), bmax(1), bmax(2), extents(0), extents(1), extents(2));
    // DenseGrid<float, int, 3> phi(extents, std::numeric_limits<float>::min());
    DenseGrid<float, int, 3> phi(extents, 2 * dx);
    DenseGrid<TV, int, 3> vel(extents, TV::zeros());
#pragma omp parallel for
    for (int x = 0; x < extents(0); ++x)
      for (int y = 0; y < extents(1); ++y)
        for (int z = 0; z < extents(2); ++z) {
          IV X = vec<int, 3>{x, y, z};
          TV x = X * dx + bmin;
          auto [p, v] = sample(x);
          phi(X) = p;
          vel(X) = TV{v(0), v(1), v(2)};
        }
    return zs::make_tuple(phi, vel, bmin, bmax);
  }

}  // namespace zs