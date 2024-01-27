#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "VdbLevelSet.h"
#include "zensim/Logger.hpp"
#include "zensim/execution/Concurrency.h"
#include "zensim/execution/ExecutionPolicy.hpp"
#if ZS_ENABLE_OPENMP
#  include "zensim/omp/execution/ExecutionPolicy.hpp"
#endif

namespace zs {

  /// floatgrid -> sparse grid
  SparseGrid<3, f32, 8> convert_floatgrid_to_sparse_grid(const OpenVDBStruct &grid,
                                                         SmallString propTag) {
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
    ret._grid =
        typename SpgT::grid_storage_type{{{propTag, 1}}, leafCount * 512, memsrc_e::host, -1};

    openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();
    // gridPtr->transform().print();
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

#if ZS_ENABLE_OPENMP
    if constexpr (!onwin) {
      auto ompExec = omp_exec();
      ret._table.reset(true);
      // tbb::parallel_for(LeafCIterRange{gridPtr->tree().cbeginLeaf()}, lam);
      ompExec(LeafCIterRange{gridPtr->tree().cbeginLeaf()},
              [&ret, table = proxy<execspace_e::openmp>(ret._table),
               ls = proxy<execspace_e::openmp>(ret), leafCount,
               propTag](LeafCIterRange &range) mutable {
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
                  block(propTag, cellid) = cell.getValue();
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
        ompExec(range(newNbs - nbs), [ls = proxy<execspace_e::openmp>(ret), nbs,
                                      propTag](typename RM_REF_T(ret)::size_type bi) mutable {
          auto block = ls.block(bi + nbs);
          using spg_t = RM_CVREF_T(ls);
          for (typename spg_t::integer_coord_component_type ci = 0; ci != ls.block_size; ++ci)
            block(propTag, ci) = -ls._background;
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
        ompExec(gridPtr->cbeginValueOff(), [ls = proxy<execspace_e::openmp>(ret),
                                            propTag](GridType::ValueOffCIter &iter) mutable {
          if (iter.getValue() < 0.0) {
            auto coord = iter.getCoord();
            auto coord_ = IV{coord.x(), coord.y(), coord.z()};
            ls(propTag, coord_) = iter.getValue();
          }
        });
      }
      return ret;
    }
#endif
    {  // fall back to serial execution
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
          block(propTag, cellid) = cell.getValue();
        }
      }
      auto seqExec = seq_exec();
      auto nbs = ret._table.size();
      i64 valueOffCount{0};
      seqExec(gridPtr->cbeginValueOff(), [&valueOffCount](GridType::ValueOffCIter &iter) {
        if (iter.getValue() < 0) valueOffCount++;
      });
      fmt::print("{} more off-value voxels to be appended to {} blocks.\n", valueOffCount, nbs);
      auto newNbs = nbs + valueOffCount;  // worst-case scenario
      if (newNbs == nbs) return ret;
      ret.resize(seqExec, newNbs);
      // init additional grid blocks
      seqExec(range(newNbs - nbs), [ls = proxy<execspace_e::host>(ret), nbs,
                                    propTag](typename RM_REF_T(ret)::size_type bi) mutable {
        auto block = ls.block(bi + nbs);
        using spg_t = RM_CVREF_T(ls);
        for (typename spg_t::integer_coord_component_type ci = 0; ci != ls.block_size; ++ci)
          block(propTag, ci) = -ls._background;
      });
      // register table
      seqExec(gridPtr->cbeginValueOff(),
              [ls = proxy<execspace_e::host>(ret)](GridType::ValueOffCIter &iter) mutable {
                if (iter.getValue() < 0.0) {
                  auto coord = iter.getCoord();
                  auto coord_ = IV{coord.x(), coord.y(), coord.z()};
                  coord_ -= (coord_ & (ls.side_length - 1));
                  ls._table.insert(coord_);
                }
              });
      // write inactive voxels
      seqExec(gridPtr->cbeginValueOff(),
              [ls = proxy<execspace_e::host>(ret), propTag](GridType::ValueOffCIter &iter) mutable {
                if (iter.getValue() < 0.0) {
                  auto coord = iter.getCoord();
                  auto coord_ = IV{coord.x(), coord.y(), coord.z()};
                  ls(propTag, coord_) = iter.getValue();
                }
              });
#if 0
      // following impl contains bugs
      for (GridType::ValueOffCIter iter = gridPtr->cbeginValueOff(); iter; ++iter) {
        if (iter.getValue() < 0.0) {
          auto coord = iter.getCoord();
          auto coord_ = IV{coord.x(), coord.y(), coord.z()};
          auto loc = (coord_ & (ret.side_length - 1));
          coord_ -= loc;
          auto blockno = table.query(coord_);
          if (blockno < 0) {
            auto nbs = ret._table.size();
            ret._table.resize(seqExec, nbs + 1);
            ret._grid.resize(nbs + 1);
            table = proxy<execspace_e::host>(ret._table);

            blockno = table.insert(coord_);
            auto block = spgv.block(blockno);
            for (auto cellno = 0; cellno != ret.block_size; ++cellno)
              block(propTag, cellno) = -ret._background;
          }
          auto locOffset = RM_REF_T(spgv)::local_coord_to_offset(loc);
          auto block = spgv.block(blockno);
          block(propTag, locOffset) = iter.getValue();
        }
      }
#endif
    }
    return ret;
  }
  SparseGrid<3, f32, 8> convert_floatgrid_to_sparse_grid(const OpenVDBStruct &grid,
                                                         const MemoryHandle mh,
                                                         SmallString propTag) {
    return convert_floatgrid_to_sparse_grid(grid, propTag).clone(mh);
  }

  /// sparse grid -> floatgrid
  OpenVDBStruct convert_sparse_grid_to_floatgrid(const SparseGrid<3, f32, 8> &splsIn,
                                                 SmallString propTag, u32 gridClass,
                                                 SmallString gridName) {
    const auto &spls
        = splsIn.memspace() != memsrc_e::host ? splsIn.clone({memsrc_e::host, -1}) : splsIn;
    openvdb::FloatGrid::Ptr grid
        = openvdb::FloatGrid::create(/*background value=*/spls._background);
    // meta
    grid->insertMeta("zpc_version", openvdb::FloatMetadata(0.f));
    grid->setName(std::string(gridName));
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
#if ZS_ENABLE_OPENMP
    if constexpr (!onwin) {
      auto ompExec = omp_exec();
      auto lsv = proxy<execspace_e::openmp>(spls);
      using LsT = RM_CVREF_T(lsv);
      const auto numBlocks = spls.numBlocks();
      for (typename LsT::size_type bno = 0; bno != numBlocks; ++bno) {
        const auto blockid = spls._table._activeKeys[bno];
        grid->tree().touchLeaf(openvdb::Coord{blockid[0], blockid[1], blockid[2]});
      }
      ompExec(zip(range(spls.numBlocks()), spls._table._activeKeys), [lsv, &grid, &spls, propTag](
                                                                         typename LsT::size_type
                                                                             blockno,
                                                                         const typename LsT::
                                                                             integer_coord_type
                                                                                 &blockid) mutable {
        auto nodePtr = grid->tree().probeLeaf(openvdb::Coord{blockid[0], blockid[1], blockid[2]});
        if (nodePtr == nullptr)
          throw std::runtime_error("strangely this leaf has not yet been allocated.");
        auto accessor = grid->getUnsafeAccessor();
        for (typename LsT::integer_coord_component_type cid = 0; cid != lsv.block_size; ++cid) {
          // const auto sdfVal = lsv.wSample(propTag, lsv.wCoord(blockno, cid));
          const auto sdfVal = lsv(propTag, blockno, cid);
          if (sdfVal == lsv._background) continue;
          const auto coord = blockid + LsT::local_offset_to_coord(cid);
          accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, sdfVal);
        }
      });
      // GRID_UNKNOWN: 0
      // GRID_LEVEL_SET: 1
      // GRID_FOG_VOLUME: 2
      // GRID_STAGGERED: 3
      if (gridClass >= 3)
        throw std::runtime_error(fmt::format("Unknown gridclass [{}]!", gridClass));
      grid->setGridClass(static_cast<openvdb::GridClass>(gridClass));
      return OpenVDBStruct{grid};
    }
#endif
    {
      auto lsv = proxy<execspace_e::host>(spls);
      using LsT = RM_CVREF_T(lsv);
      auto accessor = grid->getAccessor();
      for (auto &&[blockno, blockid] : zip(range(spls.numBlocks()), spls._table._activeKeys))
        for (int cid = 0; cid != lsv.block_size; ++cid) {
          const auto sdfVal = lsv(propTag, blockno, cid);
          if (sdfVal == lsv._background) continue;
          const auto coord = blockid + LsT::local_offset_to_coord(cid);
          accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, sdfVal);
        }
    }
    // GRID_UNKNOWN: 0
    // GRID_LEVEL_SET: 1
    // GRID_FOG_VOLUME: 2
    // GRID_STAGGERED: 3
    if (gridClass >= 3) throw std::runtime_error(fmt::format("Unknown gridclass [{}]!", gridClass));
    grid->setGridClass(static_cast<openvdb::GridClass>(gridClass));
    return OpenVDBStruct{grid};
  }

  /// vdbgrid -> sparse grid
  void assign_floatgrid_to_sparse_grid(const OpenVDBStruct &grid, SparseGrid<3, f32, 8> &spg,
                                       SmallString propTag) {
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
    using LeafType = TreeType::LeafNodeType;  // level 0 LeafNode
    using GridPtr = typename GridType::Ptr;
    const GridPtr &gridPtr = grid.as<GridPtr>();
    using SpgT = SparseGrid<3, f32, 8>;
    using IV = typename SpgT::integer_coord_type;
    using TV = typename SpgT::packed_value_type;

    auto ret = spg.clone({memsrc_e::host, -1});

    static_assert(8 * 8 * 8 == LeafType::SIZE, "leaf node size not 8x8x8!");
#if ZS_ENABLE_OPENMP
    constexpr auto space = execspace_e::openmp;
    auto pol = omp_exec();
#else
    constexpr auto space = execspace_e::host;
    auto pol = seq_exec();
#endif
    auto nbs = ret.numBlocks();
    pol(range(nbs), [&, spgv = proxy<space>(ret)](int blockno) mutable {
      // auto accessor = gridPtr->getUnsafeAccessor();
      for (int cid = 0; cid != ret.block_size; ++cid) {
        const auto wcoord = spgv.wCoord(blockno, cid);
        // auto srcVal = accessor.getValue(openvdb::Coord{coord[0], coord[1], coord[2]});
        auto srcVal = openvdb::tools::BoxSampler::sample(
            gridPtr->tree(),
            gridPtr->transform().worldToIndex(openvdb::Vec3R{wcoord[0], wcoord[1], wcoord[2]}));
        spgv(propTag, blockno, cid) = srcVal;
      }
    });

    spg._grid = ret._grid.clone(spg.memoryLocation());
  }
  void assign_float3grid_to_sparse_grid(const OpenVDBStruct &grid, SparseGrid<3, f32, 8> &spg,
                                        SmallString propTag) {
    using GridType = openvdb::Vec3fGrid;
    using TreeType = GridType::TreeType;
    using LeafType = TreeType::LeafNodeType;  // level 0 LeafNode
    using GridPtr = typename GridType::Ptr;
    const GridPtr &gridPtr = grid.as<GridPtr>();
    using SpgT = SparseGrid<3, f32, 8>;
    using IV = typename SpgT::integer_coord_type;
    using TV = typename SpgT::packed_value_type;

    auto ret = spg.clone({memsrc_e::host, -1});

    // check if GRID_STAGGERED
    static_assert(8 * 8 * 8 == LeafType::SIZE, "leaf node size not 8x8x8!");
#if ZS_ENABLE_OPENMP
    constexpr auto space = execspace_e::openmp;
    auto pol = omp_exec();
#else
    constexpr auto space = execspace_e::host;
    auto pol = seq_exec();
#endif
    auto nbs = ret.numBlocks();
    pol(range(nbs), [&, spgv = proxy<space>(ret)](int blockno) mutable {
      for (int cid = 0; cid != ret.block_size; ++cid) {
        for (int d = 0; d != 3; ++d) {
          const auto wcoord = spgv.wStaggeredCoord(blockno, cid, d);
          auto srcVal = openvdb::tools::StaggeredBoxSampler::sample(
              gridPtr->tree(),
              gridPtr->transform().worldToIndex(openvdb::Vec3R{wcoord[0], wcoord[1], wcoord[2]}));
          spgv(propTag, d, blockno, cid) = srcVal[d];
        }
      }
    });

    spg._grid = ret._grid.clone(spg.memoryLocation());
  }

  /// float3grid -> sparse grid
  SparseGrid<3, f32, 8> convert_float3grid_to_sparse_grid(const OpenVDBStruct &grid,
                                                          SmallString propTag) {
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
    using SpgT = SparseGrid<3, f32, 8>;
    using IV = typename SpgT::integer_coord_type;
    using TV = typename SpgT::packed_value_type;

    gridPtr->tree().voxelizeActiveTiles();
    static_assert(8 * 8 * 8 == LeafType::SIZE, "leaf node size not 8x8x8!");
    SpgT ret{};
    const auto leafCount = gridPtr->tree().leafCount();
    ret._background = gridPtr->background()[0];
    ret._table = typename SpgT::table_type{leafCount, memsrc_e::host, -1};
    ret._grid =
        typename SpgT::grid_storage_type{{{propTag, 3}}, leafCount * 512, memsrc_e::host, -1};

    openvdb::Mat4R v2w = gridPtr->transform().baseMap()->getAffineMap()->getMat4();
    // gridPtr->transform().print();
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
#if ZS_ENABLE_OPENMP
    if constexpr (!onwin) {
      auto ompExec = omp_exec();
      ret._table.reset(true);
      // tbb::parallel_for(LeafCIterRange{gridPtr->tree().cbeginLeaf()}, lam);
      ompExec(LeafCIterRange{gridPtr->tree().cbeginLeaf()},
              [&ret, table = proxy<execspace_e::openmp>(ret._table),
               ls = proxy<execspace_e::openmp>(ret), leafCount,
               propTag](LeafCIterRange &range) mutable {
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
                  auto vel = cell.getValue();
                  block.template tuple<3>(propTag, cellid) = TV{vel[0], vel[1], vel[2]};
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
      fmt::print("{} more off-value voxels to be appended to {} blocks.\n", valueOffCount.load(),
                 nbs);
      auto newNbs = nbs + valueOffCount.load();  // worst-case scenario
      if (newNbs != nbs) {
        ret.resize(ompExec, newNbs);
        // init additional grid blocks
        ompExec(range(newNbs - nbs), [ls = proxy<execspace_e::openmp>(ret), nbs,
                                      propTag](typename RM_REF_T(ret)::size_type bi) mutable {
          auto block = ls.block(bi + nbs);
          using spg_t = RM_CVREF_T(ls);
          for (typename spg_t::integer_coord_component_type ci = 0; ci != ls.block_size; ++ci)
            block.template tuple<3>(propTag, ci) = TV::constant(ls._background);
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
        ompExec(gridPtr->cbeginValueOff(), [ls = proxy<execspace_e::openmp>(ret),
                                            propTag](GridType::ValueOffCIter &iter) mutable {
          if (!iter.getValue().isZero()) {
            auto coord = iter.getCoord();
            auto coord_ = IV{coord.x(), coord.y(), coord.z()};
            auto vel = iter.getValue();
            for (int d = 0; d != 3; ++d) ls(propTag, d, coord_) = vel[d];
          }
        });
      }
      return ret;
    }
#endif
    {  // fall back to serial execution
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
          auto vel = cell.getValue();
          block.template tuple<3>(propTag, cellid) = TV{vel[0], vel[1], vel[2]};
        }
      }
    }
    return ret;
  }
  SparseGrid<3, f32, 8> convert_float3grid_to_sparse_grid(const OpenVDBStruct &grid,
                                                          const MemoryHandle mh,
                                                          SmallString propTag) {
    return convert_float3grid_to_sparse_grid(grid, propTag).clone(mh);
  }

  /// sparse grid -> float3grid
  OpenVDBStruct convert_sparse_grid_to_float3grid(const SparseGrid<3, f32, 8> &splsIn,
                                                  SmallString propTag, SmallString gridName) {
    using TV = openvdb::Vec3f;
    const auto &spls
        = splsIn.memspace() != memsrc_e::host ? splsIn.clone({memsrc_e::host, -1}) : splsIn;
    openvdb::Vec3fGrid::Ptr grid
        = openvdb::Vec3fGrid::create(/*background value=*/TV{spls._background});
    // meta
    grid->insertMeta("zpc_version", openvdb::FloatMetadata(0.f));
    grid->setName(std::string(gridName));
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
#if ZS_ENABLE_OPENMP
    if constexpr (!onwin) {
      auto ompExec = omp_exec();
      auto lsv = proxy<execspace_e::openmp>(spls);
      using LsT = RM_CVREF_T(lsv);
      const auto numBlocks = spls.numBlocks();
      for (typename LsT::size_type bno = 0; bno != numBlocks; ++bno) {
        const auto blockid = spls._table._activeKeys[bno];
        grid->tree().touchLeaf(openvdb::Coord{blockid[0], blockid[1], blockid[2]});
      }
      ompExec(zip(range(spls.numBlocks()), spls._table._activeKeys), [lsv, &grid, &spls, propTag](
                                                                         typename LsT::size_type
                                                                             blockno,
                                                                         const typename LsT::
                                                                             integer_coord_type
                                                                                 &blockid) mutable {
        auto nodePtr = grid->tree().probeLeaf(openvdb::Coord{blockid[0], blockid[1], blockid[2]});
        if (nodePtr == nullptr)
          throw std::runtime_error("strangely this leaf has not yet been allocated.");
        auto accessor = grid->getUnsafeAccessor();
        for (typename LsT::integer_coord_component_type cid = 0; cid != lsv.block_size; ++cid) {
          const auto val = TV{lsv(propTag, 0, blockno, cid), lsv(propTag, 1, blockno, cid),
                              lsv(propTag, 2, blockno, cid)};
          if (val.isZero()) continue;
          const auto coord = blockid + LsT::local_offset_to_coord(cid);
          accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, val);
        }
      });
      grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
      return OpenVDBStruct{grid};
    }
#endif
    {
      auto lsv = proxy<execspace_e::host>(spls);
      using LsT = RM_CVREF_T(lsv);
      auto accessor = grid->getAccessor();
      for (auto &&[blockno, blockid] : zip(range(spls.numBlocks()), spls._table._activeKeys))
        for (int cid = 0; cid != lsv.block_size; ++cid) {
          const auto val = TV{lsv(propTag, 0, blockno, cid), lsv(propTag, 1, blockno, cid),
                              lsv(propTag, 2, blockno, cid)};
          if (val.isZero()) continue;
          const auto coord = blockid + LsT::local_offset_to_coord(cid);
          accessor.setValue(openvdb::Coord{coord[0], coord[1], coord[2]}, val);
        }
    }
    grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
    return OpenVDBStruct{grid};
  }

}  // namespace zs