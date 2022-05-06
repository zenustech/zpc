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
    using LeafCIterRange = openvdb::tree::IteratorRange<typename TreeType::LeafCIter>;
    using SDFPtr = typename GridType::Ptr;
    const SDFPtr &gridPtr = grid.as<SDFPtr>();
    using SpLs = SparseLevelSet<3, grid_e::collocated>;
    using IV = typename SpLs::table_t::key_t;
    using TV = vec<typename SpLs::value_type, 3>;

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
      auto mi = box.min();
      auto ma = box.max();
      for (int d = 0; d != 3; ++d) {
        ret._min[d] = mi[d];
        ret._max[d] = ma[d];
      }
    }
    openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();

    gridPtr->transform().print();

    vec<float, 4, 4> lsv2w;
    for (auto &&[r, c] : ndrange<2>(4)) lsv2w(r, c) = v2w[r][c];
    ret.resetTransformation(lsv2w);

    {
      auto [mi, ma] = proxy<execspace_e::host>(ret).getBoundingBox();
      fmt::print(
          "leaf count: {}. background value: {}. dx: {}. ibox: [{}, {}, {} ~ {}, {}, {}], wbox: "
          "[{}, {}, {} ~ {}, {}, {}]\n",
          leafCount, ret._backgroundValue, ret._grid.dx, ret._min[0], ret._min[1], ret._min[2],
          ret._max[0], ret._max[1], ret._max[2], mi[0], mi[1], mi[2], ma[0], ma[1], ma[2]);
    }

#ifdef ZS_PLATFORM_WINDOWS
    constexpr bool onwin = true;
#else
    constexpr bool onwin = false;
#endif
    if constexpr (is_backend_available(exec_omp) && !onwin) {
      auto ompExec = omp_exec();
      ret._table.reset(ompExec, true);
      // tbb::parallel_for(LeafCIterRange{gridPtr->tree().cbeginLeaf()}, lam);
      ompExec(LeafCIterRange{gridPtr->tree().cbeginLeaf()},
              [&ret, table = proxy<execspace_e::openmp>(ret._table),
               gridview = proxy<execspace_e::openmp>(ret._grid)](LeafCIterRange &range) mutable {
                const LeafType &node = *range.iterator();
                if (node.onVoxelCount() <= 0) return;
                auto cell = node.beginValueAll();
                IV coord{};
                for (int d = 0; d != SparseLevelSet<3>::table_t::dim; ++d)
                  coord[d] = cell.getCoord()[d];
                auto blockid = coord - (coord & (ret.side_length - 1));
                if (table.query(blockid) >= 0) {
                  fmt::print("what is this??? block ({}, {}, {}) already taken!\n", blockid[0],
                             blockid[1], blockid[2]);
                }
                auto blockno = table.insert(blockid);
                auto block = gridview.block(blockno);
                RM_CVREF_T(blockno) cellid = 0;
                for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid)
                  block("sdf", cellid) = cell.getValue();
              });
      /// iterate over all inactive tiles that have negative values
      // Visit all of the grid's inactive tile and voxel values and update the values
      // that correspond to the interior region.
      auto nbs = ret._table.size();
      std::atomic<i64> valueOffCount{0};
      ompExec(gridPtr->cbeginValueOff(), [&valueOffCount](GridType::ValueOffCIter &iter) {
        if (iter.getValue() < 0) valueOffCount.fetch_add(1, std::memory_order_relaxed);
      });
      fmt::print("{} more off-value voxels to be appended to {} blocks.\n", valueOffCount.load(),
                 nbs);
      // auto newNbs = nbs + valueOffCount.load() / ret.block_size;
      auto newNbs = nbs + valueOffCount.load();  // worst-case scenario
      if (newNbs != nbs) {
        ret.resize(ompExec, newNbs);
        // init additional grid blocks
        ompExec(range(newNbs - nbs), [ls = proxy<execspace_e::openmp>(ret),
                                      nbs](typename RM_CVREF_T(ret)::size_type bi) mutable {
          auto block = ls._grid.block(bi + nbs);
          using lsv_t = RM_CVREF_T(ls);
          for (typename lsv_t::cell_index_type ci = 0; ci != ls.block_size; ++ci)
            block("sdf", ci) = -ls._backgroundValue;
        });
        // register table
        ompExec(gridPtr->cbeginValueOff(),
                [ls = proxy<execspace_e::openmp>(ret)](GridType::ValueOffCIter &iter) mutable {
                  if (iter.getValue() < 0.0) {
                    auto coord = iter.getCoord();
                    auto coord_ = IV{coord.x(), coord.y(), coord.z()};
                    coord_ -= (coord_ & (ls.side_length - 1));
                    ls._table.insert(coord_);
                  }
                });
        // write inactive voxels
        ompExec(gridPtr->cbeginValueOff(),
                [ls = proxy<execspace_e::openmp>(ret)](GridType::ValueOffCIter &iter) mutable {
                  if (iter.getValue() < 0.0) {
                    auto coord = iter.getCoord();
                    auto coord_ = IV{coord.x(), coord.y(), coord.z()};
                    auto loc = (coord_ & (ls.side_length - 1));
                    coord_ -= loc;
                    auto blockno = ls._table.query(coord_);
                    ls._grid("sdf", blockno, loc) = iter.getValue();
                  }
                });
      }
    } else {  // fall back to serial execution
      auto table = proxy<execspace_e::host>(ret._table);
      auto gridview = proxy<execspace_e::host>(ret._grid);
      table.clear();
      for (TreeType::LeafCIter iter = gridPtr->tree().cbeginLeaf(); iter; ++iter) {
        const LeafType &node = *iter;
        if (node.onVoxelCount() <= 0) continue;
        auto cell = node.beginValueAll();
        IV coord{};
        for (int d = 0; d != SparseLevelSet<3>::table_t::dim; ++d) coord[d] = cell.getCoord()[d];
        auto blockid = coord - (coord & (ret.side_length - 1));
        if (table.query(blockid) >= 0) {
          fmt::print("what is this??? block ({}, {}, {}) already taken!\n", blockid[0], blockid[1],
                     blockid[2]);
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
    using LeafCIterRange = openvdb::tree::IteratorRange<typename TreeType::LeafCIter>;
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
        {{"v", 3}}, (float)gridPtr->transform().voxelSize()[0], leafCount, memsrc_e::host, -1};
    {
      openvdb::CoordBBox box = gridPtr->evalActiveVoxelBoundingBox();
      auto mi = box.min();
      auto ma = box.max();
      for (int d = 0; d != 3; ++d) {
        ret._min[d] = mi[d];
        ret._max[d] = ma[d];
      }
    }
    openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();

    gridPtr->transform().print();

    vec<float, 4, 4> lsv2w;
    for (auto &&[r, c] : ndrange<2>(4)) lsv2w(r, c) = v2w[r][c];
    ret.resetTransformation(lsv2w);

    {
      auto [mi, ma] = proxy<execspace_e::host>(ret).getBoundingBox();
      fmt::print(
          "leaf count: {}. background value: {}. dx: {}. ibox: [{}, {}, {} ~ {}, {}, {}], wbox: "
          "[{}, {}, {} ~ {}, {}, {}]\n",
          leafCount, ret._backgroundValue, ret._grid.dx, ret._min[0], ret._min[1], ret._min[2],
          ret._max[0], ret._max[1], ret._max[2], mi[0], mi[1], mi[2], ma[0], ma[1], ma[2]);
    }
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
#ifdef ZS_PLATFORM_WINDOWS
    constexpr bool onwin = true;
#else
    constexpr bool onwin = false;
#endif
    if constexpr (is_backend_available(exec_omp) && !onwin) {
      auto ompExec = omp_exec();
      ret._table.reset(ompExec, true);
      ompExec(LeafCIterRange{gridPtr->tree().cbeginLeaf()},
              [&ret, table = proxy<execspace_e::openmp>(ret._table),
               gridview = proxy<execspace_e::openmp>(ret._grid)](LeafCIterRange &range) mutable {
                const LeafType &node = *range.iterator();
                if (node.onVoxelCount() <= 0) return;
                auto cell = node.beginValueAll();
                IV coord{};
                for (int d = 0; d != SparseLevelSet<3>::table_t::dim; ++d)
                  coord[d] = cell.getCoord()[d];
                auto blockid = coord - (coord & (ret.side_length - 1));
                if (table.query(blockid) >= 0) {
                  fmt::print("what is this??? block ({}, {}, {}) already taken!\n", blockid[0],
                             blockid[1], blockid[2]);
                }
                auto blockno = table.insert(blockid);
                auto block = gridview.block(blockno);
                RM_CVREF_T(blockno) cellid = 0;
                for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
                  auto vel = cell.getValue();
                  const auto offset = blockno * ret.block_size + cellid;
                  gridview.set("v", offset, TV{vel[0], vel[1], vel[2]});
                }
              });
      /// iterate over all inactive tiles that have negative values
      // Visit all of the grid's inactive tile and voxel values and update the values
      // that correspond to the interior region.
      auto nbs = ret._table.size();
      std::atomic<i64> valueOffCount{0};
      ompExec(gridPtr->cbeginValueOff(), [&valueOffCount](GridType::ValueOffCIter &iter) {
        if (!iter.getValue().isZero()) valueOffCount.fetch_add(1, std::memory_order_relaxed);
      });
      fmt::print("{} more off-(vec-)value voxels to be appended to {} blocks.\n",
                 valueOffCount.load(), nbs);
      auto newNbs = nbs + valueOffCount.load();  // worst-case scenario
      if (newNbs != nbs) {
        ret.resize(ompExec, newNbs);
        // init additional grid blocks
        ompExec(range(newNbs - nbs), [ls = proxy<execspace_e::openmp>(ret),
                                      nbs](typename RM_CVREF_T(ret)::size_type bi) mutable {
          using lsv_t = RM_CVREF_T(ls);
          for (typename lsv_t::cell_index_type ci = 0; ci != ls.block_size; ++ci) {
            const auto offset = (bi + nbs) * ls.block_size + ci;
            ls._grid.set("v", offset, ls._backgroundVecValue);
          }
        });
        // register table
        ompExec(gridPtr->cbeginValueOff(),
                [ls = proxy<execspace_e::openmp>(ret)](GridType::ValueOffCIter &iter) mutable {
                  if (!iter.getValue().isZero()) {
                    auto coord = iter.getCoord();
                    auto coord_ = IV{coord.x(), coord.y(), coord.z()};
                    coord_ -= (coord_ & (ls.side_length - 1));
                    ls._table.insert(coord_);
                  }
                });
        // write inactive voxels
        ompExec(gridPtr->cbeginValueOff(), [ls = proxy<execspace_e::openmp>(ret)](
                                               GridType::ValueOffCIter &iter) mutable {
          using lsv_t = RM_CVREF_T(ls);
          if (!iter.getValue().isZero()) {
            auto coord = iter.getCoord();
            auto coord_ = IV{coord.x(), coord.y(), coord.z()};
            auto loc = (coord_ & (ls.side_length - 1));
            coord_ -= loc;
            auto blockno = ls._table.query(coord_);
            auto val = TV{iter.getValue()[0], iter.getValue()[1], iter.getValue()[2]};
            ls._grid.set("v", blockno * ls.block_size + lsv_t::grid_view_t::coord_to_cellid(loc),
                         val);
          }
        });
      }
    } else {
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
            gridview.set("v", offset, TV{vel[0], vel[1], vel[2]});
            // gridview.voxel("mask", offset) = cell.isValueOn() ? 1 : 0;
          }
        }
      }
    }
    return ret;
  }
  SparseLevelSet<3> convert_vec3fgrid_to_sparse_levelset(const OpenVDBStruct &grid,
                                                         const MemoryHandle mh) {
    return convert_vec3fgrid_to_sparse_levelset(grid).clone(mh);
  }

  SparseLevelSet<3, grid_e::staggered> convert_vec3fgrid_to_sparse_staggered_grid(
      const OpenVDBStruct &grid) {
    using GridType = openvdb::Vec3fGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using LeafCIterRange = openvdb::tree::IteratorRange<typename TreeType::LeafCIter>;
    using GridPtr = typename GridType::Ptr;
    const GridPtr &gridPtr = grid.as<GridPtr>();
    using SpLs = SparseLevelSet<3, grid_e::staggered>;
    using IV = typename SpLs::table_t::key_t;
    using TV = vec<typename SpLs::value_type, 3>;

    assert(gridPtr->getGridClass() == openvdb::GridClass::GRID_STAGGERED);
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
        {{"v", 3}}, (float)gridPtr->transform().voxelSize()[0], leafCount, memsrc_e::host, -1};
    {
      openvdb::CoordBBox box = gridPtr->evalActiveVoxelBoundingBox();
      auto mi = box.min();
      auto ma = box.max();
      for (int d = 0; d != 3; ++d) {
        ret._min[d] = mi[d];
        ret._max[d] = ma[d];
      }
    }
    openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();

    gridPtr->transform().print();

    vec<float, 4, 4> lsv2w;
    for (auto &&[r, c] : ndrange<2>(4)) lsv2w(r, c) = v2w[r][c];
    ret.resetTransformation(lsv2w);

    {
      auto [mi, ma] = proxy<execspace_e::host>(ret).getBoundingBox();
      fmt::print(
          "leaf count: {}. background value: {}. dx: {}. ibox: [{}, {}, {} ~ {}, {}, {}], wbox: "
          "[{}, {}, {} ~ {}, {}, {}]\n",
          leafCount, ret._backgroundValue, ret._grid.dx, ret._min[0], ret._min[1], ret._min[2],
          ret._max[0], ret._max[1], ret._max[2], mi[0], mi[1], mi[2], ma[0], ma[1], ma[2]);
    }

#ifdef ZS_PLATFORM_WINDOWS
    constexpr bool onwin = true;
#else
    constexpr bool onwin = false;
#endif
    if constexpr (is_backend_available(exec_omp) && !onwin) {
      auto ompExec = omp_exec();
      ret._table.reset(ompExec, true);
      ompExec(LeafCIterRange{gridPtr->tree().cbeginLeaf()},
              [&ret, table = proxy<execspace_e::openmp>(ret._table),
               gridview = proxy<execspace_e::openmp>(ret._grid)](LeafCIterRange &range) mutable {
                const LeafType &node = *range.iterator();
                if (node.onVoxelCount() <= 0) return;
                auto cell = node.beginValueAll();
                IV coord{};
                for (int d = 0; d != SparseLevelSet<3>::table_t::dim; ++d)
                  coord[d] = cell.getCoord()[d];
                auto blockid = coord - (coord & (ret.side_length - 1));
                if (table.query(blockid) >= 0) {
                  fmt::print("what is this??? block ({}, {}, {}) already taken!\n", blockid[0],
                             blockid[1], blockid[2]);
                }
                auto blockno = table.insert(blockid);
                auto block = gridview.block(blockno);
                RM_CVREF_T(blockno) cellid = 0;
                for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
                  auto vel = cell.getValue();
                  const auto offset = blockno * ret.block_size + cellid;
                  gridview.set("v", offset, TV{vel[0], vel[1], vel[2]});
                }
              });
      /// iterate over all inactive tiles that have negative values
      // Visit all of the grid's inactive tile and voxel values and update the values
      // that correspond to the interior region.
      auto nbs = ret._table.size();
      std::atomic<i64> valueOffCount{0};
      ompExec(gridPtr->cbeginValueOff(), [&valueOffCount](GridType::ValueOffCIter &iter) {
        if (!iter.getValue().isZero()) valueOffCount.fetch_add(1, std::memory_order_relaxed);
      });
      fmt::print("{} more off-(vec-)value voxels to be appended to {} blocks.\n",
                 valueOffCount.load(), nbs);
      auto newNbs = nbs + valueOffCount.load();  // worst-case scenario
      if (newNbs != nbs) {
        ret.resize(ompExec, newNbs);
        // init additional grid blocks
        ompExec(range(newNbs - nbs), [ls = proxy<execspace_e::openmp>(ret),
                                      nbs](typename RM_CVREF_T(ret)::size_type bi) mutable {
          using lsv_t = RM_CVREF_T(ls);
          for (typename lsv_t::cell_index_type ci = 0; ci != ls.block_size; ++ci) {
            const auto offset = (bi + nbs) * ls.block_size + ci;
            ls._grid.set("v", offset, ls._backgroundVecValue);
          }
        });
        // register table
        ompExec(gridPtr->cbeginValueOff(),
                [ls = proxy<execspace_e::openmp>(ret)](GridType::ValueOffCIter &iter) mutable {
                  if (!iter.getValue().isZero()) {
                    auto coord = iter.getCoord();
                    auto coord_ = IV{coord.x(), coord.y(), coord.z()};
                    coord_ -= (coord_ & (ls.side_length - 1));
                    ls._table.insert(coord_);
                  }
                });
        // write inactive voxels
        ompExec(gridPtr->cbeginValueOff(), [ls = proxy<execspace_e::openmp>(ret)](
                                               GridType::ValueOffCIter &iter) mutable {
          using lsv_t = RM_CVREF_T(ls);
          if (!iter.getValue().isZero()) {
            auto coord = iter.getCoord();
            auto coord_ = IV{coord.x(), coord.y(), coord.z()};
            auto loc = (coord_ & (ls.side_length - 1));
            coord_ -= loc;
            auto blockno = ls._table.query(coord_);
            auto val = TV{iter.getValue()[0], iter.getValue()[1], iter.getValue()[2]};
            ls._grid.set("v", blockno * ls.block_size + lsv_t::grid_view_t::coord_to_cellid(loc),
                         val);
          }
        });
      }
    } else {
      auto table = proxy<execspace_e::host>(ret._table);
      auto gridview = proxy<execspace_e::host>(ret._grid);
      table.clear();
      for (TreeType::LeafCIter iter = gridPtr->tree().cbeginLeaf(); iter; ++iter) {
        const TreeType::LeafNodeType &node = *iter;
        if (node.onVoxelCount() > 0) {
          auto cell = node.beginValueAll();
          IV coord{};
          for (int d = 0; d != SpLs::dim; ++d) coord[d] = cell.getCoord()[d];
          auto blockid = coord - (coord & (ret.side_length - 1));
          auto blockno = table.insert(blockid);
          RM_CVREF_T(blockno) cellid = 0;
          for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
            auto vel = cell.getValue();
            const auto offset = blockno * ret.block_size + cellid;
            gridview.set("v", offset, TV{vel[0], vel[1], vel[2]});
            // gridview.voxel("mask", offset) = cell.isValueOn() ? 1 : 0;
          }
        }
      }
    }
    return ret;
  }
  SparseLevelSet<3, grid_e::staggered> convert_vec3fgrid_to_sparse_staggered_grid(
      const OpenVDBStruct &grid, const MemoryHandle mh) {
    return convert_vec3fgrid_to_sparse_staggered_grid(grid).clone(mh);
  }

  ///
  /// zpc levelset -> vdb levelset
  ///
  template <typename SplsT> OpenVDBStruct convert_sparse_levelset_to_vdbgrid(const SplsT &splsIn) {
    static_assert(is_spls_v<SplsT>, "SplsT must be a sparse levelset type");
    const auto &spls
        = splsIn.memspace() != memsrc_e::host ? splsIn.clone({memsrc_e::host, -1}) : splsIn;
    openvdb::FloatGrid::Ptr grid
        = openvdb::FloatGrid::create(/*background value=*/spls._backgroundValue);
    // meta
    grid->insertMeta("zpctag", openvdb::FloatMetadata(0.f));
    grid->setName("ZpcLevelSet");
    // transform
    openvdb::Mat4R v2w{};
    auto lsv2w = spls.getIndexToWorldTransformation();
    for (auto &&[r, c] : ndrange<2>(4)) v2w[r][c] = lsv2w[r][c];
    grid->setTransform(openvdb::math::Transform::createLinearTransform(v2w));

    // tree
#ifdef ZS_PLATFORM_WINDOWS
    constexpr bool onwin = true;
#else
    constexpr bool onwin = false;
#endif
    if constexpr (is_backend_available(exec_omp) && !onwin) {
      auto ompExec = omp_exec();
      auto lsv = proxy<execspace_e::openmp>(spls);
      using LsT = RM_CVREF_T(lsv);
      // for (const auto &blockid : spls._table._activeKeys)
      const auto numBlocks = spls.numBlocks();
      for (typename LsT::size_type bno = 0; bno != numBlocks; ++bno) {
        const auto blockid = spls._table._activeKeys[bno];
        grid->tree().touchLeaf(openvdb::Coord{blockid[0], blockid[1], blockid[2]});
      }
      ompExec(zip(range(spls.numBlocks()), spls._table._activeKeys), [lsv, &grid, &spls](
                                                                         typename LsT::size_type
                                                                             blockno,
                                                                         const typename LsT::IV
                                                                             &blockid) mutable {
        auto nodePtr = grid->tree().probeLeaf(openvdb::Coord{blockid[0], blockid[1], blockid[2]});
        if (nodePtr == nullptr)
          throw std::runtime_error("strangely this leaf has not yet been allocated.");
#if 0
        typename LsT::cell_index_type ci = 0;
        auto &node = *nodePtr;
        auto block = lsv._grid.block(blockno);
        for (auto cell = node.beginValueAll(); cell && ci != lsv.block_size; ++cell, ++ci) {
          const auto sdfVal
              = lsv.wsample("sdf", 0, lsv.indexToWorld(blockno, ci), lsv._backgroundValue);
          // if (sdfVal == lsv._backgroundValue) continue;
          // const auto coord = blockid + LsT::grid_view_t::cellid_to_coord(ci);
          // accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, sdfVal);
          cell.setValue(block("sdf", ci));
        }
#else
        auto accessor = grid->getAccessor();
        for (typename LsT::cell_index_type cid = 0; cid != lsv.block_size; ++cid) {
          const auto sdfVal
              = lsv.wsample("sdf", 0, lsv.indexToWorld(blockno, cid), lsv._backgroundValue);
          if (sdfVal == lsv._backgroundValue) continue;
          const auto coord = blockid + LsT::grid_view_t::cellid_to_coord(cid);
          accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, sdfVal);
        }
#endif
      });
    } else {
      auto lsv = proxy<execspace_e::host>(spls);
      using LsT = RM_CVREF_T(lsv);
      auto accessor = grid->getAccessor();
      for (auto &&[blockno, blockid] : zip(range(spls.numBlocks()), spls._table._activeKeys))
        for (int cid = 0; cid != lsv.block_size; ++cid) {
          // const auto offset = (int)blockno * (int)spls.block_size + cid;
          const auto sdfVal
              = lsv.wsample("sdf", 0, lsv.indexToWorld(blockno, cid), lsv._backgroundValue);
          if (sdfVal == lsv._backgroundValue) continue;
          const auto coord = blockid + LsT::grid_view_t::cellid_to_coord(cid);
          // (void)accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, 0.f);
          accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, sdfVal);
        }
    }
    if constexpr (SplsT::category == grid_e::staggered)
      grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
    else
      grid->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
    return OpenVDBStruct{grid};
  }

  template OpenVDBStruct convert_sparse_levelset_to_vdbgrid<SparseLevelSet<3, grid_e::collocated>>(
      const SparseLevelSet<3, grid_e::collocated> &splsIn);
  template OpenVDBStruct
  convert_sparse_levelset_to_vdbgrid<SparseLevelSet<3, grid_e::cellcentered>>(
      const SparseLevelSet<3, grid_e::cellcentered> &splsIn);
  template OpenVDBStruct convert_sparse_levelset_to_vdbgrid<SparseLevelSet<3, grid_e::staggered>>(
      const SparseLevelSet<3, grid_e::staggered> &splsIn);

}  // namespace zs