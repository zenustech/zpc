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
    using IV = typename SpLs::IV;
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
      ret._table.reset(true);
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
      ret._table.reset(true);
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

  /// floatgrid -> sparse grid
  SparseGrid<3, f32, 8> convert_floatgrid_to_sparse_grid(const OpenVDBStruct &grid) {
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
    using SpgT = SparseGrid<3, f32, 8>;
    using IV = typename SpgT::integer_coord_type;
    using TV = typename SpgT::packed_value_type;

    gridPtr->tree().voxelizeActiveTiles();
    static_assert(8 * 8 * 8 == LeafType::SIZE, "leaf node size not 8x8x8!");
    SpgT ret{};
    const auto leafCount = gridPtr->tree().leafCount();
    ret._background = gridPtr->background();
    ret._table = typename SpgT::table_type{leafCount, memsrc_e::host, -1};
    ret._grid = typename SpgT::grid_storage_type{{{"sdf", 1}}, leafCount * 512, memsrc_e::host, -1};

    openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();
    gridPtr->transform().print();
    vec<float, 4, 4> lsv2w;
    for (auto &&[r, c] : ndrange<2>(4)) lsv2w(r, c) = v2w[r][c];
    ret.resetTransformation(lsv2w);

    {
      auto [mi, ma] = proxy<execspace_e::host>(ret).getBoundingBox();
      fmt::print("leaf count: {}. background value: {}. dx: {}\n", leafCount, ret._background,
                 ret.voxelSize()[0]);
    }

#ifdef ZS_PLATFORM_WINDOWS
    constexpr bool onwin = true;
#else
    constexpr bool onwin = false;
#endif
    if constexpr (is_backend_available(exec_omp) && !onwin) {
      auto ompExec = omp_exec();
      ret._table.reset(true);
      // tbb::parallel_for(LeafCIterRange{gridPtr->tree().cbeginLeaf()}, lam);
      ompExec(LeafCIterRange{gridPtr->tree().cbeginLeaf()},
              [&ret, table = proxy<execspace_e::openmp>(ret._table),
               ls = proxy<execspace_e::openmp>(ret), leafCount](LeafCIterRange &range) mutable {
                const LeafType &node = *range.iterator();
                if (node.onVoxelCount() <= 0) return;
                auto cell = node.beginValueAll();
                IV coord{};
                for (int d = 0; d != 3; ++d) coord[d] = cell.getCoord()[d];
                auto blockid = coord - (coord & (ret.side_length - 1));
                if (table.query(blockid) >= 0) {
                  fmt::print("what is this??? block ({}, {}, {}) already registered!\n", blockid[0],
                             blockid[1], blockid[2]);
                }
                auto blockno = table.insert(blockid);
                auto block = ls.block(blockno);
                int cellid = 0;
                for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
                  block("sdf", cellid) = cell.getValue();
                }
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
      auto newNbs = nbs + valueOffCount.load();  // worst-case scenario
      if (newNbs != nbs) {
        ret.resize(ompExec, newNbs);
        // init additional grid blocks
        ompExec(range(newNbs - nbs), [ls = proxy<execspace_e::openmp>(ret),
                                      nbs](typename RM_CVREF_T(ret)::size_type bi) mutable {
          auto block = ls.block(bi + nbs);
          using spg_t = RM_CVREF_T(ls);
          for (typename spg_t::integer_coord_component_type ci = 0; ci != ls.block_size; ++ci)
            block("sdf", ci) = -ls._background;
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
                    ls("sdf", coord_) = iter.getValue();
                  }
                });
      }
    } else {  // fall back to serial execution
      auto table = proxy<execspace_e::host>(ret._table);
      auto spgv = proxy<execspace_e::host>(ret);
      ret._table.reset(true);
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
        auto block = spgv.block(blockno);
        RM_CVREF_T(blockno) cellid = 0;
        for (auto cell = node.beginValueAll(); cell; ++cell, ++cellid) {
          block("sdf", cellid) = cell.getValue();
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

            blockno = table.insert(coord_);
            auto block = spgv.block(blockno);
            for (auto cellno = 0; cellno != ret.block_size; ++cellno)
              block("sdf", cellno) = -ret._background;
          }
          auto locOffset = RM_CVREF_T(spgv)::local_coord_to_offset(loc);
          auto block = spgv.block(blockno);
          block("sdf", locOffset) = iter.getValue();
        }
      }
    }
    return ret;
  }
  SparseGrid<3, f32, 8> convert_floatgrid_to_sparse_grid(const OpenVDBStruct &grid,
                                                         const MemoryHandle mh) {
    return convert_floatgrid_to_sparse_grid(grid).clone(mh);
  }

}  // namespace zs