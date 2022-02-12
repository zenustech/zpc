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

  ///
  /// vdb levelset -> zpc levelset
  ///
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
    ret._backgroundValue = gridPtr->background();
    ret._table = typename SpLs::table_t{leafCount, memsrc_e::host, -1};
    ret._grid = typename SpLs::grid_t{
        {{"sdf", 1}}, (float)gridPtr->transform().voxelSize()[0], leafCount, memsrc_e::host, -1};
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
               leafCount, ret._backgroundValue, ret._grid.dx, ret._min[0], ret._min[1], ret._min[2],
               ret._max[0], ret._max[1], ret._max[2]);

    vec<float, 4, 4> lsv2w;
    for (auto &&[r, c] : ndrange<2>(4)) lsv2w(r, c) = v2w[r][c];
    ret.resetTransformation(lsv2w);

    auto table = proxy<execspace_e::host>(ret._table);
    auto gridview = proxy<execspace_e::host>(ret._grid);
    table.clear();
    for (TreeType::LeafCIter iter = gridPtr->tree().cbeginLeaf(); iter; ++iter) {
      const TreeType::LeafNodeType &node = *iter;
      if (node.onVoxelCount() > 0) {
        auto cell = node.beginValueAll();
        IV coord{};
        for (int d = 0; d != SparseLevelSet<3>::table_t::dim; ++d) coord[d] = cell.getCoord()[d];
        auto blockid = coord - (coord & (ret.side_length - 1));
        auto blockno = table.insert(blockid);
        auto block = gridview.block(blockno);
        RM_CVREF_T(blockno) cellid = 0;
        for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
          block("sdf", cellid) = cell.getValue();
          // auto sdf = cell.getValue();
          // const auto offset = blockno * ret.block_size + cellid;
          // gridview.voxel("sdf", offset) = sdf;
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
    ret._backgroundValue = 0;
    {
      auto v = gridPtr->background();
      ret._backgroundVecValue = TV{v[0], v[1], v[2]};
    }
    ret._table = typename SpLs::table_t{leafCount, memsrc_e::host, -1};
    ret._grid = typename SpLs::grid_t{
        {{"vel", 3}}, (float)gridPtr->transform().voxelSize()[0], leafCount, memsrc_e::host, -1};
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
               leafCount, ret._backgroundValue, ret._grid.dx, ret._min[0], ret._min[1], ret._min[2],
               ret._max[0], ret._max[1], ret._max[2]);

    vec<float, 4, 4> lsv2w;
    for (auto &&[r, c] : ndrange<2>(4)) lsv2w(r, c) = v2w[r][c];
    ret.resetTransformation(lsv2w);

#if 0
    vec<float, 3, 3> scale;
    Rotation<float, 3> rot;
    vec<float, 3> trans;

    math::decompose_transform(lsv2w, scale, rot, trans, 0);
    fmt::print("scale: [{}, {}, {}; {}, {}, {}; {}, {}, {}]\n", scale(0, 0), scale(0, 1),
               scale(0, 2), scale(1, 0), scale(1, 1), scale(1, 2), scale(2, 0), scale(2, 1),
               scale(2, 2));
    fmt::print("rotation: [{}, {}, {}; {}, {}, {}; {}, {}, {}]\n", rot(0, 0), rot(0, 1), rot(0, 2),
               rot(1, 0), rot(1, 1), rot(1, 2), rot(2, 0), rot(2, 1), rot(2, 2));
    fmt::print("translation: [{}, {}, {}]\n", trans(0), trans(1), trans(2));
    {
      auto [ST, RT, TT] = math::decompose_transform(lsv2w, 0);
      auto V2W = ST * RT * TT;
      fmt::print(
          "recomposed v2w: [{}, {}, {}, {}; {}, {}, {}, {}; {}, {}, {}, {}; {}, {}, {}, {}]\n",
          V2W(0, 0), V2W(0, 1), V2W(0, 2), V2W(0, 3), V2W(1, 0), V2W(1, 1), V2W(1, 2), V2W(1, 3),
          V2W(2, 0), V2W(2, 1), V2W(2, 2), V2W(2, 3), V2W(3, 0), V2W(3, 1), V2W(3, 2), V2W(3, 3));
    }
    getchar();
#endif

    auto table = proxy<execspace_e::host>(ret._table);
    auto gridview = proxy<execspace_e::host>(ret._grid);
    table.clear();
    for (TreeType::LeafCIter iter = gridPtr->tree().cbeginLeaf(); iter; ++iter) {
      const TreeType::LeafNodeType &node = *iter;
      if (node.onVoxelCount() > 0) {
        auto cell = node.beginValueOn();
        IV coord{};
        for (int d = 0; d != SparseLevelSet<3>::table_t::dim; ++d) coord[d] = cell.getCoord()[d];
        auto blockid = coord;
        for (int d = 0; d != SparseLevelSet<3>::table_t::dim; ++d)
          blockid[d] -= (coord[d] & (ret.side_length - 1));
        auto blockno = table.insert(blockid);
        RM_CVREF_T(blockno) cellid = 0;
        for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
          auto vel = cell.getValue();
          const auto offset = blockno * ret.block_size + cellid;
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

  ///
  /// zpc levelset -> vdb levelset
  ///
  OpenVDBStruct convert_sparse_levelset_to_floatgrid(const SparseLevelSet<3> &splsIn) {
    auto spls = splsIn.clone(MemoryHandle{memsrc_e::host, -1});
    openvdb::FloatGrid::Ptr grid
        = openvdb::FloatGrid::create(/*background value=*/spls._backgroundValue);
    // meta
    grid->insertMeta("zpctag", openvdb::FloatMetadata(0.f));
    grid->setGridClass(openvdb::GRID_LEVEL_SET);
    grid->setName("ZpcLevelSet");
    // transform
    openvdb::Mat4R v2w{};
    auto lsv2w = spls.getIndexToWorldTransformation();
    for (auto &&[r, c] : ndrange<2>(4)) v2w[r][c] = lsv2w[r][c];
    grid->setTransform(openvdb::math::Transform::createLinearTransform(v2w));
    // tree
    auto table = proxy<execspace_e::host>(spls._table);
    auto gridview = proxy<execspace_e::host>(spls._grid);
    auto accessor = grid->getAccessor();
    using GridT = RM_CVREF_T(gridview);
    for (auto &&[blockno, blockid] :
         zip(range(spls._grid.size() / spls.block_size), spls._table._activeKeys))
      for (int cid = 0; cid != spls.block_size; ++cid) {
        const auto offset = (int)blockno * (int)spls.block_size + cid;
        const auto sdfVal = gridview.voxel("sdf", offset);
        if (sdfVal == spls._backgroundValue) continue;
        const auto coord = blockid + GridT::cellid_to_coord(cid);
        // (void)accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, 0.f);
        accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, sdfVal);
      }
    return OpenVDBStruct{grid};
  }

}  // namespace zs