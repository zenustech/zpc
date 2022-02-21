#include <openvdb/Grid.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "VdbLevelSet.h"
#include "zensim/Logger.hpp"
#include "zensim/execution/Concurrency.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zs {

  std::vector<const void *> get_floatgrid_interior_leaves(const OpenVDBStruct &grid) {
    using GridType = openvdb::FloatGrid;
    using ValueType = GridType::ValueType;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using NodeChainType = RootType::NodeChainType;
    using InternalNodeType = NodeChainType::template Get<1>;
    // using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    // using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;  // level 0 LeafNode

    const auto &gridPtr = grid.as<typename GridType::Ptr>();
    /// ref: implementation from sdfToFogVolume
    const TreeType &tree = gridPtr->tree();

    size_t numLeafNodes = 0;

    std::vector<const void *> ret;
    std::vector<const LeafType *> nodes;
    std::vector<size_t> leafnodeCount;

    ValueType cutoffDistance = limits<ValueType>::max();
    {
      // Compute the prefix sum of the leafnode count in each internal node.
      std::vector<const InternalNodeType *> internalNodes;
      tree.getNodes(internalNodes);

      leafnodeCount.push_back(0);
      for (size_t n = 0; n < internalNodes.size(); ++n) {
        leafnodeCount.push_back(leafnodeCount.back() + internalNodes[n]->leafCount());
      }

      numLeafNodes = leafnodeCount.back();

      // Steal all leafnodes (Removes them from the tree and transfers ownership.)
      nodes.reserve(numLeafNodes);

#if 0
      using NT = std::remove_reference_t<std::remove_pointer_t<RM_CVREF_T(internalNodes[0])>>;
      fmt::print("nc: {}\n", zs::get_type_str<typename NT::ChildNodeType>());
      fmt::print("candidate: {}\n", zs::get_type_str<typename decltype(nodes)::value_type>());
      getchar();
#endif
#if 1
      auto stealNodes
          = [&nodes, value = tree.background(), state = false](auto &self, auto nodePtr) {
              using NodeT = std::remove_reference_t<std::remove_pointer_t<RM_CVREF_T(nodePtr)>>;
              for (auto iter = nodePtr->cbeginChildOn(); iter; ++iter) {
                const auto n = iter.pos();
                if constexpr (is_same_v<LeafType, typename NodeT::ChildNodeType>) {
                  auto nnode = nodePtr->getTable()[n];
                  nodes.push_back(nnode.getChild());
                  auto valueMask = nodePtr->getValueMask();
                  const_cast<std::remove_const_t<RM_CVREF_T(valueMask)> &>(valueMask).set(n, state);
                  const_cast<std::remove_const_t<RM_CVREF_T(nnode)> &>(nnode).setValue(value);
                } else
                  self(nodePtr->getTable()[n]);
              }
              if constexpr (is_same_v<LeafType, typename NodeT::ChildNodeType>) {
                auto childMask = nodePtr->getChildMask();
                const_cast<std::remove_const_t<RM_CVREF_T(childMask)> &>(childMask).setOff();
              }
            };
      auto snf = recursive_lambda(stealNodes);
#endif
      for (size_t n = 0; n < internalNodes.size(); ++n) {
        // internalNodes[n]->stealNodes(nodes, tree.background(), false);
        snf(internalNodes[n]);
      }

      // Clamp cutoffDistance to min sdf value
      ValueType minSDFValue = limits<ValueType>::max();

      {
        openvdb::tools::level_set_util_internal::FindMinTileValue<InternalNodeType> minOp(
            internalNodes.data());
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, internalNodes.size()), minOp);
        minSDFValue = std::min(minSDFValue, minOp.minValue);
      }

      if (minSDFValue > ValueType(0.0)) {
        openvdb::tools::level_set_util_internal::FindMinVoxelValue<LeafType> minOp(nodes.data());
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()), minOp);
        minSDFValue = std::min(minSDFValue, minOp.minValue);
      }

      cutoffDistance = -std::abs(cutoffDistance);
      cutoffDistance = minSDFValue > cutoffDistance ? minSDFValue : cutoffDistance;
    }

#if 0
    for (auto leafNode : nodes) {
      if (leafNode == nullptr) {
        fmt::print("indeed there is invalid leaf!\n");
        getchar();
      }
      auto values = leafNode->buffer().data();
      for (int i = 0; i != LeafType::SIZE; ++i) {
        if (values[i] < 0) fmt::print("[{}]: {}\n", i, values[i]);
      }
      // getchar();
    }
    getchar();
#endif

    ret.resize(nodes.size());
    std::transform(std::begin(nodes), std::end(nodes), std::begin(ret),
                   [](auto ptr) { return static_cast<const void *>(ptr); });
    return ret;
  }

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
#if 0
    auto tmp = get_floatgrid_interior_leaves(grid);
    std::vector<const LeafType *> leaves(tmp.size());
    std::transform(std::begin(tmp), std::end(tmp), std::begin(leaves),
                   [](auto ptr) { return (const LeafType *)ptr; });
    fmt::print("been here!\n");
    getchar();
#endif

    gridPtr->tree().voxelizeActiveTiles();
    static_assert(8 * 8 * 8 == LeafType::SIZE, "leaf node size not 8x8x8!");
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
      const LeafType &node = *iter;
      if (node.onVoxelCount() <= 0) continue;
      // for (std::size_t leafid = 0; leafid != leaves.size(); ++leafid) {
      // for (auto leafPtr : leaves) {
      // const TreeType::LeafNodeType &node = *iter;
      // if (leafPtr == nullptr) continue;
      // const LeafType &node = *leafPtr;
      auto cell = node.beginValueAll();
      IV coord{};
      for (int d = 0; d != SparseLevelSet<3>::table_t::dim; ++d) coord[d] = cell.getCoord()[d];
      auto blockid = coord - (coord & (ret.side_length - 1));
      if (table.query(blockid) >= 0) {
        printf("what is this??? block ({}, {}, {}) already taken!\n", blockid[0], blockid[1],
               blockid[2]);
        getchar();
      }
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
    /// iterate over all inactive tiles that have negative values
    // Visit all of the grid's inactive tile and voxel values and update the values
    // that correspond to the interior region.
    for (GridType::ValueOffCIter iter = gridPtr->cbeginValueOff(); iter; ++iter) {
      if (iter.getValue() < 0.0) {
        auto coord = iter.getCoord();
        auto coord_ = IV{coord.x(), coord.y(), coord.z()};
        auto loc = (coord_ & (ret.side_length - 1));
        coord_ -= loc;
        auto blockno = table.query(coord_);
        if (blockno < 0) {
          auto nbs = ret._table.size();
          ret._table.resize(seq_exec(), nbs + 1);
          ret._grid.resize(nbs + 1);
          table = proxy<execspace_e::host>(ret._table);
          gridview = proxy<execspace_e::host>(ret._grid);

          blockno = table.insert(coord_);
#if 0
          fmt::print("inserting block ({}, {}, {}) [{}] at loc ({}, {}, {})\n", coord_[0],
                     coord_[1], coord_[2], blockno, loc[0], loc[1], loc[2]);
#endif
          auto block = gridview.block(blockno);
          for (auto cellno = 0; cellno != ret.block_size; ++cellno)
            block("sdf", cellno) = -ret._backgroundValue;
        }
        auto block = gridview.block(blockno);
        block("sdf", loc) = iter.getValue();
#if 0
        fmt::print("coord [{}, {}, {}], table query: {}\n", coord_[0], coord_[1], coord_[2], );
        fmt::print("-> [{}, {}, {}] - [{}, {}, {}]; level: {}, depth: {} (ld: {})\n", st[0], st[1],
                   st[2], ed[0], ed[1], ed[2], level, iter.getDepth(), iter.getLeafDepth());
        auto k = (ed[0] - st[0]) / ret.side_length;
        for (auto offset : ndrange<3>(k)) {
          auto blockid = corner + make_vec<int>(offset);
          fmt::print("\t[{}, {}, {}]\n", blockid[0], blockid[1], blockid[2]);
        }
#endif
      }
    }
    if constexpr (false) {
      auto lsv = proxy<execspace_e::host>(ret);
#if 1
      int actualBlockCnt = 0;
      for (TreeType::LeafCIter iter = gridPtr->tree().cbeginLeaf(); iter; ++iter) {
        const TreeType::LeafNodeType &node = *iter;
        // if (node.onVoxelCount() <= 0) continue;
        actualBlockCnt++;
        int cellid = 0;
        for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
          auto p = gridPtr->transform().indexToWorld(cell.getCoord());
          TV pp{p[0], p[1], p[2]};
          auto srcSdf = lsv.getSignedDistance(pp);
          auto refSdf = cell.getValue();
          auto refSdf_ = refSdf;
#  if 0
          openvdb::tools::BoxSampler::sample(gridPtr->tree(),
                                                            gridPtr->transform().worldToIndex(p));
#  endif
          if (srcSdf < 0 || refSdf < 0)
            fmt::print("at ({}, {}, {}). stored sdf: {}, ref sdf: {} ({})\n", p[0], p[1], p[2],
                       srcSdf, refSdf, refSdf_);
          if ((pp + TV::uniform(0.005)).l2NormSqr() < 1e-6) {
            fmt::print("chk ({}, {}, {}) -> sdf [{}]; ref [{}]\n", pp[0], pp[1], pp[2], srcSdf,
                       refSdf);
            getchar();
          }
        }
      }
      fmt::print("stored block cnt: {}; actual cnt: {}\n", ret._grid.numBlocks(), actualBlockCnt);
#else
      TV test0{1, 2, 3};
      auto w0 = lsv.indexToWorld(test0);
      auto w1 = gridPtr->indexToWorld(openvdb::Vec3R{test0[0], test0[1], test0[2]});
      auto test1 = lsv.worldToIndex(w0);
      fmt::print("ipos: {}, {}, {} vs. recovered {}, {}, {}\n", test0[0], test0[1], test0[2],
                 test1[0], test1[1], test1[2]);
      fmt::print("wpos: lsv {}, {}, {} vs. vdb {}, {}, {}\n", w0[0], w0[1], w0[2], w1[0], w1[1],
                 w1[2]);
      getchar();
      for (int blockno = 0; blockno != ret._grid.numBlocks(); ++blockno) {
        for (int cellno = 0; cellno != ret._grid.block_size; ++cellno) {
          auto icoord
              = ret._table._activeKeys[blockno] + RM_CVREF_T(lsv._grid)::cellid_to_coord(cellno);
          auto ipos = icoord;
          auto wpos = lsv.indexToWorld(ipos);
          auto ipos_ = gridPtr->worldToIndex(openvdb::Vec3R{wpos[0], wpos[1], wpos[2]});

          auto srcSdf = lsv._grid("sdf", blockno, cellno);
          auto srcSdf_ = lsv.getSignedDistance(wpos);
          auto refSdf = openvdb::tools::BoxSampler::sample(gridPtr->tree(), ipos_);
          if (refSdf < 0 || srcSdf < 0)
            fmt::print("at ({}, {}, {}). stored sdf: {} ({}), ref sdf: {}\n", wpos[0], wpos[1],
                       wpos[2], srcSdf, srcSdf_, refSdf);
        }
      }
      fmt::print("box: {}, {}, {} - {}, {}, {}\n", ret._min[0], ret._min[1], ret._min[2],
                 ret._max[0], ret._max[1], ret._max[2]);
#endif
      getchar();
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
  template <typename SplsT>
  OpenVDBStruct convert_sparse_levelset_to_floatgrid(const SplsT &splsIn) {
    static_assert(is_spls_v<SplsT>, "SplsT must be a sparse levelset type");
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
    auto lsv = proxy<execspace_e::host>(spls);
    auto gridview = proxy<execspace_e::host>(spls._grid);
    auto accessor = grid->getAccessor();
    using LsT = RM_CVREF_T(lsv);
    for (auto &&[blockno, blockid] : zip(range(spls.numBlocks()), spls._table._activeKeys))
      for (int cid = 0; cid != spls.block_size; ++cid) {
        // const auto offset = (int)blockno * (int)spls.block_size + cid;
        const auto sdfVal
            = lsv.wsample("sdf", 0, lsv.indexToWorld(blockno, cid), lsv._backgroundValue);
        if (sdfVal == spls._backgroundValue) continue;
        const auto coord = blockid + LsT::cellid_to_coord(cid);
        // (void)accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, 0.f);
        accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, sdfVal);
      }
    if constexpr (SplsT::category == grid_e::staggered)
      grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
    else
      grid->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
    return OpenVDBStruct{grid};
  }

  template OpenVDBStruct convert_sparse_levelset_to_floatgrid<
      SparseLevelSet<3, grid_e::collocated>>(const SparseLevelSet<3, grid_e::collocated> &splsIn);
  template OpenVDBStruct
  convert_sparse_levelset_to_floatgrid<SparseLevelSet<3, grid_e::cellcentered>>(
      const SparseLevelSet<3, grid_e::cellcentered> &splsIn);
  template OpenVDBStruct convert_sparse_levelset_to_floatgrid<SparseLevelSet<3, grid_e::staggered>>(
      const SparseLevelSet<3, grid_e::staggered> &splsIn);

}  // namespace zs