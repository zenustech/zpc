#include <openvdb/Grid.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "VdbLevelSet.h"
#include "zensim/Logger.hpp"
#include "zensim/execution/Concurrency.h"

namespace zs {

  SparseLevelSet<3> convert_floatgrid_to_sparse_levelset(const OpenVDBStruct &grid) {
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using SDFPtr = typename GridType::Ptr;
    const SDFPtr &gridPtr = grid.as<SDFPtr>();
    using SpLs = SparseLevelSet<3, grid_e::collocated>;
    using IV = typename SpLs::table_t::key_t;
    using TV = vec<typename SpLs::value_type, 3>;

    gridPtr->tree().voxelizeActiveTiles();
    SpLs ret{};
    const auto leafCount = gridPtr->tree().leafCount();
    ret._sideLength = SpLs::side_length;
    ret._space = SpLs::grid_t::block_space();
    ret._dx = gridPtr->transform().voxelSize()[0];
    ret._backgroundValue = gridPtr->background();
    ret._table = typename SpLs::table_t{leafCount, memsrc_e::host, -1};
    ret._grid = typename SpLs::grid_t{{{"sdf", 1}}, ret._dx, leafCount, memsrc_e::host, -1};
    {
      openvdb::CoordBBox box = gridPtr->evalActiveVoxelBoundingBox();
      auto corner = box.min();
      auto length = box.max() - box.min();
      auto world_min = gridPtr->indexToWorld(box.min());
      auto world_max = gridPtr->indexToWorld(box.max());
      for (size_t d = 0; d < 3; d++) {
        ret._min(d) = world_min[d];
        ret._max(d) = world_max[d];
      }
      for (auto &&[dx, dy, dz] : ndrange<3>(2)) {
        auto coord
            = corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0, dz ? length[2] : 0};
        auto pos = gridPtr->indexToWorld(coord);
        for (int d = 0; d < 3; d++) {
          ret._min(d) = pos[d] < ret._min(d) ? pos[d] : ret._min(d);
          ret._max(d) = pos[d] > ret._max(d) ? pos[d] : ret._max(d);
        }
      }
    }
    openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();

    gridPtr->transform().print();
    fmt::print("leaf count: {}. background value: {}. dx: {}. box: [{}, {}, {} ~ {}, {}, {}]\n",
               leafCount, ret._backgroundValue, ret._dx, ret._min[0], ret._min[1], ret._min[2],
               ret._max[0], ret._max[1], ret._max[2]);

    auto w2v = v2w.inverse();
    vec<float, 4, 4> transform;
    for (auto &&[r, c] : ndrange<2>(4)) transform(r, c) = w2v[r][c];  /// use [] for access
    ret._w2v = transform;

    auto table = proxy<execspace_e::host>(ret._table);
    auto gridview = proxy<execspace_e::host>(ret._grid);
    table.clear();
    for (TreeType::LeafCIter iter = gridPtr->tree().cbeginLeaf(); iter; ++iter) {
      const TreeType::LeafNodeType &node = *iter;
      if (node.onVoxelCount() > 0) {
        auto cell = node.beginValueOn();
        IV coord{};
        for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d) coord[d] = cell.getCoord()[d];
        auto blockid = coord;
        for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d)
          blockid[d] += (coord[d] < 0 ? -ret._sideLength + 1 : 0);
        blockid = blockid / ret._sideLength;
        auto blockno = table.insert(blockid);
        RM_CVREF_T(blockno) cellid = 0;
        for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
          auto sdf = cell.getValue();
          const auto offset = blockno * ret._space + cellid;
          gridview.voxel("sdf", offset) = sdf;
        }
      }
    }
    return ret;
  }
  SparseLevelSet<3> convert_floatgrid_to_sparse_levelset(const OpenVDBStruct &grid,
                                                         const MemoryHandle mh) {
    return convert_floatgrid_to_sparse_levelset(grid).clone(mh);
  }

  SparseLevelSet<3> convert_vec3fgrid_to_sparse_levelset(const OpenVDBStruct &grid) {
    using GridType = openvdb::Vec3fGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using GridPtr = typename GridType::Ptr;
    const GridPtr &gridPtr = grid.as<GridPtr>();
    using SpLs = SparseLevelSet<3, grid_e::collocated>;
    using IV = typename SpLs::table_t::key_t;
    using TV = vec<typename SpLs::value_type, 3>;

    gridPtr->tree().voxelizeActiveTiles();
    SpLs ret{};
    const auto leafCount = gridPtr->tree().leafCount();
    ret._sideLength = SpLs::side_length;
    ret._space = SpLs::grid_t::block_space();
    ret._dx = gridPtr->transform().voxelSize()[0];
    ret._backgroundValue = 0;
    {
      auto v = gridPtr->background();
      ret._backgroundVecValue = TV{v[0], v[1], v[2]};
    }
    ret._table = typename SpLs::table_t{leafCount, memsrc_e::host, -1};
    ret._grid = typename SpLs::grid_t{{{"vel", 3}}, ret._dx, leafCount, memsrc_e::host, -1};
    {
      openvdb::CoordBBox box = gridPtr->evalActiveVoxelBoundingBox();
      auto corner = box.min();
      auto length = box.max() - box.min();
      auto world_min = gridPtr->indexToWorld(box.min());
      auto world_max = gridPtr->indexToWorld(box.max());
      for (size_t d = 0; d < 3; d++) {
        ret._min(d) = world_min[d];
        ret._max(d) = world_max[d];
      }
      for (auto &&[dx, dy, dz] : ndrange<3>(2)) {
        auto coord
            = corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0, dz ? length[2] : 0};
        auto pos = gridPtr->indexToWorld(coord);
        for (int d = 0; d < 3; d++) {
          ret._min(d) = pos[d] < ret._min(d) ? pos[d] : ret._min(d);
          ret._max(d) = pos[d] > ret._max(d) ? pos[d] : ret._max(d);
        }
      }
    }
    openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();

    gridPtr->transform().print();
    fmt::print("leaf count: {}. background value: {}. dx: {}. box: [{}, {}, {} ~ {}, {}, {}]\n",
               leafCount, ret._backgroundValue, ret._dx, ret._min[0], ret._min[1], ret._min[2],
               ret._max[0], ret._max[1], ret._max[2]);

    auto w2v = v2w.inverse();
    vec<float, 4, 4> transform;
    for (auto &&[r, c] : ndrange<2>(4)) transform(r, c) = w2v[r][c];  /// use [] for access
    ret._w2v = transform;

    auto table = proxy<execspace_e::host>(ret._table);
    auto gridview = proxy<execspace_e::host>(ret._grid);
    table.clear();
    for (TreeType::LeafCIter iter = gridPtr->tree().cbeginLeaf(); iter; ++iter) {
      const TreeType::LeafNodeType &node = *iter;
      if (node.onVoxelCount() > 0) {
        auto cell = node.beginValueOn();
        IV coord{};
        for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d) coord[d] = cell.getCoord()[d];
        auto blockid = coord;
        for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d)
          blockid[d] += (coord[d] < 0 ? -ret._sideLength + 1 : 0);
        blockid = blockid / ret._sideLength;
        auto blockno = table.insert(blockid);
        RM_CVREF_T(blockno) cellid = 0;
        for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
          auto vel = cell.getValue();
          const auto offset = blockno * ret._space + cellid;
          gridview.set("vel", offset, TV{vel[0], vel[1], vel[2]});
          // gridview.voxel("mask", offset) = cell.isValueOn() ? 1 : 0;
        }
      }
    }
    return ret;
  }
  SparseLevelSet<3> convert_vec3fgrid_to_sparse_levelset(const OpenVDBStruct &grid,
                                                         const MemoryHandle mh) {
    return convert_vec3fgrid_to_sparse_levelset(grid).clone(mh);
  }

}  // namespace zs