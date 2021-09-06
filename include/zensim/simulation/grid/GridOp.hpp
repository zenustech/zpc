#pragma once
#include "../transfer/Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Collider.h"
#include "zensim/geometry/Structure.hpp"

namespace zs {

  template <typename GridBlocksT> struct CleanGridBlocks;
  template <typename TableT, typename GridBlocksT> struct PrintGridBlocks;
  template <transfer_scheme_e, typename GridBlocksT> struct ComputeGridBlockVelocity;
  template <typename ColliderT, typename TableT, typename GridBlocksT>
  struct ApplyBoundaryConditionOnGridBlocks;
  template <grid_e category, typename GridsViewT> struct ResetGrid;
  template <grid_e category, typename GridsViewT> struct GridMomentumToVelocity;
  template <grid_e category, typename TableT, typename GridT> struct GridAngularMomentum;

  template <execspace_e space, typename V, int d, int chnbits, int dombits>
  CleanGridBlocks(wrapv<space>, GridBlocks<GridBlock<V, d, chnbits, dombits>>)
      -> CleanGridBlocks<GridBlocksView<space, GridBlocks<GridBlock<V, d, chnbits, dombits>>>>;
  template <execspace_e space, typename T, int d, auto l>
  CleanGridBlocks(wrapv<space>, Grids<T, d, l>)
      -> CleanGridBlocks<GridsView<space, Grids<T, d, l>>>;

  template <execspace_e space, typename TableT, typename GridBlocksT>
  PrintGridBlocks(wrapv<space>, TableT, GridBlocksT)
      -> PrintGridBlocks<HashTableView<space, TableT>, GridBlocksView<space, GridBlocksT>>;

  template <execspace_e space, transfer_scheme_e scheme, typename V, int d, int chnbits,
            int dombits, typename Sth>
  ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>,
                           GridBlocks<GridBlock<V, d, chnbits, dombits>>, float dt, Sth,
                           float *maxVel)
      -> ComputeGridBlockVelocity<
          scheme, GridBlocksView<space, GridBlocks<GridBlock<V, d, chnbits, dombits>>>>;
  template <execspace_e space, transfer_scheme_e scheme, typename T, int d, auto l, typename Sth>
  ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, Grids<T, d, l>, float dt, Sth,
                           float *maxVel)
      -> ComputeGridBlockVelocity<scheme, GridsView<space, Grids<T, d, l>>>;

  template <execspace_e space, typename LevelsetT, typename TableT, typename V, int d, int chnbits,
            int dombits>
  ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Collider<LevelsetT>, TableT,
                                     GridBlocks<GridBlock<V, d, chnbits, dombits>>)
      -> ApplyBoundaryConditionOnGridBlocks<
          Collider<LevelsetT>, HashTableView<space, TableT>,
          GridBlocksView<space, GridBlocks<GridBlock<V, d, chnbits, dombits>>>>;
  template <execspace_e space, typename TableT, typename V, int d, int chnbits, int dombits>
  ApplyBoundaryConditionOnGridBlocks(wrapv<space>, LevelSetBoundary<SparseLevelSet<3>>, TableT,
                                     GridBlocks<GridBlock<V, d, chnbits, dombits>>)
      -> ApplyBoundaryConditionOnGridBlocks<
          Collider<SparseLevelSetView<space, SparseLevelSet<3>>>, HashTableView<space, TableT>,
          GridBlocksView<space, GridBlocks<GridBlock<V, d, chnbits, dombits>>>>;
  template <execspace_e space, typename LevelsetT, typename TableT, typename T, int d, auto l,
            typename... Sth>
  ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Collider<LevelsetT>, TableT, Grids<T, d, l>,
                                     Sth...)
      -> ApplyBoundaryConditionOnGridBlocks<Collider<LevelsetT>, HashTableView<space, TableT>,
                                            GridsView<space, Grids<T, d, l>>>;
  template <execspace_e space, typename TableT, typename T, int d, auto l, typename... Sth>
  ApplyBoundaryConditionOnGridBlocks(wrapv<space>, LevelSetBoundary<SparseLevelSet<3>>, TableT,
                                     Grids<T, d, l>, Sth...)
      -> ApplyBoundaryConditionOnGridBlocks<Collider<SparseLevelSetView<space, SparseLevelSet<3>>>,
                                            HashTableView<space, TableT>,
                                            GridsView<space, Grids<T, d, l>>>;

  template <execspace_e space, typename T, int d, auto l, grid_e category>
  ResetGrid(wrapv<space>, Grid<T, d, l, category>)
      -> ResetGrid<category, GridsView<space, Grids<T, d, l>>>;
  template <execspace_e space, typename T, int d, auto l, grid_e category>
  GridMomentumToVelocity(wrapv<space>, Grid<T, d, l, category>, int, int, float *maxVel)
      -> GridMomentumToVelocity<category, GridsView<space, Grids<T, d, l>>>;

  template <execspace_e space, typename TableT, typename T, int d, auto l, grid_e category>
  GridAngularMomentum(wrapv<space>, TableT &table, Grid<T, d, l, category>, int, int, double *sum)
      -> GridAngularMomentum<category, HashTableView<space, TableT>,
                             GridsView<space, Grids<T, d, l>>>;
  template <execspace_e space, typename TableT, typename V, int d, int chnbits, int dombits>
  GridAngularMomentum(wrapv<space>, TableT &table, GridBlocks<GridBlock<V, d, chnbits, dombits>>,
                      int, int, double *sum)
      -> GridAngularMomentum<grid_e::collocated, HashTableView<space, TableT>,
                             GridBlocksView<space, GridBlocks<GridBlock<V, d, chnbits, dombits>>>>;

  template <execspace_e space, typename GridBlocksT>
  struct CleanGridBlocks<GridBlocksView<space, GridBlocksT>> {
    static constexpr auto exectag = wrapv<space>{};
    using gridblocks_t = GridBlocksView<space, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    explicit CleanGridBlocks(wrapv<space>, GridBlocksT &gridblocks)
        : gridblocks{proxy<space>(gridblocks)} {}

    constexpr void operator()(typename gridblocks_t::size_type blockid,
                              typename gridblock_t::size_type cellid) noexcept {
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      block(0, cellid).asFloat() = static_cast<VT>(0);
      for (int d = 0; d < gridblocks_t::dim; ++d)
        block(d + 1, cellid).asFloat() = static_cast<VT>(0);
    }

    gridblocks_t gridblocks;
  };

  template <execspace_e space, typename GridsT> struct CleanGridBlocks<GridsView<space, GridsT>> {
    using grids_t = GridsView<space, GridsT>;

    explicit CleanGridBlocks(wrapv<space>, GridsT &grids) : grids{proxy<space>(grids)} {}

    constexpr void operator()(typename GridsT::size_type blockid,
                              typename GridsT::cell_index_type cellid) noexcept {
      using value_type = typename GridsT::value_type;
      auto grid = grids.grid(collocated_v);
      auto block = grid.block(blockid);
      auto nchns = grid.numChannels();
      for (int i = 0; i != nchns; ++i) block(i, cellid) = 0;
    }

    grids_t grids;
  };

  template <execspace_e space, typename TableT, typename GridBlocksT>
  struct PrintGridBlocks<HashTableView<space, TableT>, GridBlocksView<space, GridBlocksT>> {
    using partition_t = HashTableView<space, TableT>;
    using gridblocks_t = GridBlocksView<space, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    explicit PrintGridBlocks(wrapv<space>, TableT &table, GridBlocksT &gridblocks)
        : partition{proxy<space>(table)}, gridblocks{proxy<space>(gridblocks)} {}

    constexpr void operator()(typename gridblocks_t::size_type blockid,
                              typename gridblock_t::size_type cellid) noexcept {
      auto blockkey = partition._activeKeys[blockid];
      auto checkedid = partition.query(blockkey);
      if (checkedid != blockid && cellid == 0)
        printf("%d-th block(%d, %d, %d) table index: %d\n", blockid, blockkey[0], blockkey[1],
               blockkey[2], checkedid);
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      if (blockid == 0 && cellid == 0) printf("block space: %d\n", gridblock_t::space());
      if (blockid == 0)  // || (float)block(1, cellid).asFloat() > 0)
        printf("(%d, %d): mass %e, (%e, %e, %e)\n", (int)blockid, (int)cellid,
               block(0, cellid).asFloat(), block(1, cellid).asFloat(), block(2, cellid).asFloat(),
               block(3, cellid).asFloat());
    }

    partition_t partition;
    gridblocks_t gridblocks;
  };

  template <execspace_e space, transfer_scheme_e scheme, typename GridBlocksT>
  struct ComputeGridBlockVelocity<scheme, GridBlocksView<space, GridBlocksT>> {
    static constexpr auto exectag = wrapv<space>{};
    using gridblocks_t = GridBlocksView<space, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;
    using value_t = typename gridblocks_t::value_type;
    using float_t = RM_CVREF_T(std::declval<value_t>().asFloat());
    using TV = vec<float_t, gridblocks_t::dim>;

    constexpr ComputeGridBlockVelocity() = default;
    ~ComputeGridBlockVelocity() = default;

    explicit ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, GridBlocksT &gridblocks,
                                      float dt, float gravity, float *maxVel)
        : gridblocks{proxy<space>(gridblocks)}, dt{dt}, extf{TV::zeros()}, maxVel{maxVel} {
      extf[1] = gravity;
    }
    explicit ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, GridBlocksT &gridblocks,
                                      float dt, TV extf, float *maxVel)
        : gridblocks{proxy<space>(gridblocks)}, dt{dt}, extf{extf}, maxVel{maxVel} {}

    constexpr void operator()(typename gridblocks_t::size_type blockid,
                              typename gridblock_t::size_type cellid) noexcept {
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      VT mass = block(0, cellid).asFloat();
      if (mass != (VT)0) {
        mass = (VT)1 / mass;
        vec<VT, gridblocks_t::dim> vel{};
        for (int d = 0; d < gridblocks_t::dim; ++d) {
          vel[d] = block(d + 1, cellid).asFloat() * mass;
          vel[d] += extf[d] * dt;
        }
        // vel[1] += gravity * dt;

        /// write back
        for (int d = 0; d < gridblocks_t::dim; ++d) block(d + 1, cellid).asFloat() = vel[d];

        /// cfl dt
        float ret{0.f};
        for (int d = 0; d < gridblocks_t::dim; ++d) ret += vel[d] * vel[d];
        atomic_max(exectag, maxVel, ret);
      }
    }

    gridblocks_t gridblocks;
    float dt;
    TV extf;
    float *maxVel;
  };
  template <execspace_e space, transfer_scheme_e scheme, typename GridsT>
  struct ComputeGridBlockVelocity<scheme, GridsView<space, GridsT>> {
    static constexpr auto exectag = wrapv<space>{};
    using grids_t = GridsView<space, GridsT>;
    using value_type = typename grids_t::value_type;
    using TV = vec<value_type, grids_t::dim>;

    constexpr ComputeGridBlockVelocity() = default;
    ~ComputeGridBlockVelocity() = default;

    explicit ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, GridsT &grids, float dt,
                                      float gravity, float *maxVel)
        : grids{proxy<space>(grids)}, dt{dt}, extf{TV::zeros()}, maxVel{maxVel} {
      extf[1] = gravity;
    }
    explicit ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, GridsT &grids, float dt, TV extf,
                                      float *maxVel)
        : grids{proxy<space>(grids)}, dt{dt}, extf{extf}, maxVel{maxVel} {}

    constexpr void operator()(typename grids_t::size_type blockid,
                              typename grids_t::cell_index_type cellid) noexcept {
      auto block = grids[blockid];
      value_type mass = block(0, cellid);
      if (mass != (value_type)0) {
        mass = (value_type)1 / mass;
        TV vel = block.pack<grids_t::dim>(1, cellid) * mass + extf * dt;
        /// write back
        // for (int d = 0; d != grids_t::dim; ++d) block(1 + d, cellid) = vel[d];
        block.set(1, cellid, vel);
        /// cfl dt
        value_type ret = vel.l2NormSqr();
        atomic_max(exectag, maxVel, ret);
      }
    }

    grids_t grids;
    float dt;
    TV extf;
    float *maxVel;
  };

  template <execspace_e space, typename ColliderT, typename TableT, typename GridBlocksT>
  struct ApplyBoundaryConditionOnGridBlocks<ColliderT, HashTableView<space, TableT>,
                                            GridBlocksView<space, GridBlocksT>> {
    using collider_t = ColliderT;
    using partition_t = HashTableView<space, TableT>;
    using gridblocks_t = GridBlocksView<space, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    template <typename Boundary = ColliderT,
              enable_if_t<!is_levelset_boundary<Boundary>::value> = 0>
    ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Boundary &collider, TableT &table,
                                       GridBlocksT &gridblocks)
        : collider{collider},
          partition{proxy<space>(table)},
          gridblocks{proxy<space>(gridblocks)} {}
    template <typename Boundary, enable_if_t<is_levelset_boundary<Boundary>::value> = 0>
    ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Boundary &boundary, TableT &table,
                                       GridBlocksT &gridblocks)
        : collider{Collider{proxy<space>(boundary.levelset), boundary.type/*, boundary.s,
                            boundary.dsdt, boundary.R, boundary.omega, boundary.b, boundary.dbdt*/}},
          partition{proxy<space>(table)},
          gridblocks{proxy<space>(gridblocks)} {}

    constexpr void operator()(typename gridblocks_t::size_type blockid,
                              typename gridblock_t::size_type cellid) noexcept {
      auto blockkey = partition._activeKeys[blockid];
      auto &block = gridblocks[blockid];
      using VT = typename collider_t::T;
      VT dx = static_cast<VT>(gridblocks._dx.asFloat());

      if (block(0, cellid).asFloat() > 0) {
        vec<VT, gridblocks_t::dim> vel{},
            pos = (blockkey * gridblock_t::side_length() + gridblock_t::to_coord(cellid)) * dx;
        for (int d = 0; d < gridblocks_t::dim; ++d)
          vel[d] = static_cast<VT>(block(d + 1, cellid).asFloat());

        collider.resolveCollision(pos, vel);

        for (int d = 0; d < gridblocks_t::dim; ++d) block(d + 1, cellid).asFloat() = vel[d];
      }
    }

    collider_t collider;
    partition_t partition;
    gridblocks_t gridblocks;
  };
  template <execspace_e space, typename ColliderT, typename TableT, typename GridsT>
  struct ApplyBoundaryConditionOnGridBlocks<ColliderT, HashTableView<space, TableT>,
                                            GridsView<space, GridsT>> {
    using collider_t = ColliderT;
    using partition_t = HashTableView<space, TableT>;
    using grids_t = GridsView<space, GridsT>;
    using value_type = typename grids_t::value_type;

    template <typename Boundary = ColliderT,
              enable_if_t<!is_levelset_boundary<Boundary>::value> = 0>
    ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Boundary &collider, TableT &table,
                                       GridsT &grids, float t = 0.f)
        : collider{collider},
          partition{proxy<space>(table)},
          grids{proxy<space>(grids)},
          curTime{t} {}
    template <typename Boundary, enable_if_t<is_levelset_boundary<Boundary>::value> = 0>
    ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Boundary &boundary, TableT &table,
                                       GridsT &grids, float t = 0.f)
        : collider{Collider{proxy<space>(boundary.levelset), boundary.type/*, boundary.s,
                            boundary.dsdt, boundary.R, boundary.omega, boundary.b, boundary.dbdt*/}},
          partition{proxy<space>(table)},
          grids{proxy<space>(grids)}, curTime{t} {}

    constexpr void operator()(typename grids_t::size_type blockid,
                              typename grids_t::cell_index_type cellid) noexcept {
      auto blockkey = partition._activeKeys[blockid];
      auto block = grids[blockid];

      if (block(0, cellid) > 0) {
        auto vel = block.pack<grids_t::dim>(1, cellid);
        auto pos = (blockkey * (value_type)grids_t::side_length + grids_t::cellid_to_coord(cellid))
                   * grids._dx;

#if 1
        collider.resolveCollision(pos, vel);
#else
        float vy = sinf(curTime * 300) * 2;
        if (pos[2] < 0) {
          vel[0] = vel[2] = 0.f;
          vel[1] = vy;
        }
#endif

        block.set(1, cellid, vel);
      }
    }

    collider_t collider;
    partition_t partition;
    grids_t grids;
    float curTime;
  };

  template <grid_e category, execspace_e space, typename T, int d, auto l>
  struct ResetGrid<category, GridsView<space, Grids<T, d, l>>> {
    using grid_view_t = typename GridsView<space, Grids<T, d, l>>::template Grid<category>;
    using value_type = typename grid_view_t::value_type;
    static constexpr int dim = grid_view_t::dim;

    explicit ResetGrid(wrapv<space>, Grid<T, d, l, category> &grid) : grid{proxy<space>(grid)} {}

    constexpr void operator()(typename grid_view_t::size_type blockid,
                              typename grid_view_t::cell_index_type cellid) noexcept {
      auto block = grid.block(blockid);
      auto nchns = grid.numChannels();
      for (int i = 0; i != nchns; ++i) block(i, cellid) = 0;
    }

    grid_view_t grid;
  };

  template <grid_e category, execspace_e space, typename T, int d, auto l>
  struct GridMomentumToVelocity<category, GridsView<space, Grids<T, d, l>>> {
    using grid_view_t = typename GridsView<space, Grids<T, d, l>>::template Grid<category>;
    using value_type = typename grid_view_t::value_type;
    static constexpr int dim = grid_view_t::dim;

    GridMomentumToVelocity() = default;
    explicit GridMomentumToVelocity(wrapv<space>, Grid<T, d, l, category> &grid, int mChn = 0,
                                    int mvChn = 1, float *maxVel = nullptr)
        : grid{proxy<space>(grid)}, maxVel{maxVel}, mChn{mChn}, mvChn{mvChn} {}

    constexpr void operator()(typename grid_view_t::size_type blockid,
                              typename grid_view_t::cell_index_type cellid) noexcept {
      auto block = grid[blockid];
      value_type mass = block(mChn, cellid);
      if (mass != (value_type)0) {
        mass = (value_type)1 / mass;
        auto vel = block.pack<dim>(mvChn, cellid) * mass;
        /// write back
        // for (int d = 0; d != grids_t::dim; ++d) block(1 + d, cellid) = vel[d];
        block.set(mvChn, cellid, vel);
        /// cfl dt
        value_type ret = vel.l2NormSqr();
        atomic_max(wrapv<space>{}, maxVel, ret);
      }
    }

    grid_view_t grid;
    float *maxVel;
    int mChn, mvChn;
  };

  template <grid_e category, execspace_e space, typename TableT, typename T, int d, auto l>
  struct GridAngularMomentum<category, HashTableView<space, TableT>,
                             GridsView<space, Grids<T, d, l>>> {
    using partition_t = HashTableView<space, TableT>;
    using grid_view_t = typename GridsView<space, Grids<T, d, l>>::template Grid<category>;
    using value_type = typename grid_view_t::value_type;
    static constexpr int dim = grid_view_t::dim;

    GridAngularMomentum() = default;
    explicit GridAngularMomentum(wrapv<space>, TableT &table, Grid<T, d, l, category> &grid,
                                 int mChn = 0, int mvChn = 1, double *sumAngularMomentum = nullptr)
        : partition{proxy<space>(table)},
          grid{proxy<space>(grid)},
          sumAngularMomentum{sumAngularMomentum},
          mChn{mChn},
          mvChn{mvChn} {}

    constexpr void operator()(typename grid_view_t::size_type blockid,
                              typename grid_view_t::cell_index_type cellid) noexcept {
      auto block = grid.block(blockid);
      value_type mass = block(mChn, cellid);
      if (mass != (value_type)0) {
        auto blockkey = partition._activeKeys[blockid];
        auto x = (blockkey * (value_type)grid_view_t::side_length
                  + grid_view_t::cellid_to_coord(cellid))
                 * grid.dx;
        auto mv = block.pack<dim>(mvChn, cellid);
        /// x cross mv;
        if constexpr (dim == 3) {
          auto res = x.cross(mv);

          for (int i = 0; i != dim; ++i)
            atomic_add(wrapv<space>{}, sumAngularMomentum + i, (double)res[i]);
        }
        for (int i = 0; i != dim; ++i)
          atomic_add(wrapv<space>{}, sumAngularMomentum + i + dim, (double)mv[i]);
      }
    }

    partition_t partition;
    grid_view_t grid;
    double *sumAngularMomentum;
    int mChn, mvChn;
  };
#if 1
  template <execspace_e space, typename TableT, typename GridBlocksT>
  struct GridAngularMomentum<grid_e::collocated, HashTableView<space, TableT>,
                             GridBlocksView<space, GridBlocksT>> {
    using partition_t = HashTableView<space, TableT>;
    using grid_view_t = GridBlocksView<space, GridBlocksT>;
    using gridblock_t = typename grid_view_t::block_t;
    using value_type = float;
    static constexpr int dim = grid_view_t::dim;

    GridAngularMomentum() = default;
    explicit GridAngularMomentum(wrapv<space>, TableT &table, GridBlocksT &grid, int mChn = 0,
                                 int mvChn = 1, double *sumAngularMomentum = nullptr)
        : partition{proxy<space>(table)},
          grid{proxy<space>(grid)},
          sumAngularMomentum{sumAngularMomentum},
          mChn{mChn},
          mvChn{mvChn} {}

    constexpr void operator()(typename grid_view_t::size_type blockid,
                              typename gridblock_t::size_type cellid) noexcept {
      auto &block = grid[blockid];
      value_type mass = block(mChn, cellid).asFloat();
      if (mass != (value_type)0) {
        auto blockkey = partition._activeKeys[blockid];
        auto x = (blockkey * gridblock_t::side_length() + gridblock_t::to_coord(cellid))
                 * grid._dx.asFloat();
        vec<value_type, dim> mv{};
        for (int d = 0; d != dim; ++d)
          mv[d] = static_cast<value_type>(block(d + mvChn, cellid).asFloat());
        /// x cross mv;
        auto res = x.cross(mv);

        for (int i = 0; i != dim; ++i)
          atomic_add(wrapv<space>{}, sumAngularMomentum + i, (double)res[i]);
        for (int i = 0; i != dim; ++i)
          atomic_add(wrapv<space>{}, sumAngularMomentum + i + dim, (double)mv[i]);
      }
    }

    partition_t partition;
    grid_view_t grid;
    double *sumAngularMomentum;
    int mChn, mvChn;
  };
#endif

#if 0
  template <execspace_e space, typename ColliderT, typename TableT, typename GridsT>
  struct ApplyBoundaryConditionOnGrid<ColliderT, HashTableView<space, TableT>,
                                      GridsView<space, GridsT>> {
    using collider_t = ColliderT;
    using partition_t = HashTableView<space, TableT>;
    using grids_t = GridsView<space, GridsT>;
    using value_type = typename grids_t::value_type;

    template <typename Boundary = ColliderT,
              enable_if_t<!is_levelset_boundary<Boundary>::value> = 0>
    ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Boundary &collider, TableT &table,
                                       GridsT &grids, float t = 0.f)
        : collider{collider},
          partition{proxy<space>(table)},
          grids{proxy<space>(grids)},
          curTime{t} {}
    template <typename Boundary, enable_if_t<is_levelset_boundary<Boundary>::value> = 0>
    ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Boundary &boundary, TableT &table,
                                       GridsT &grids, float t = 0.f)
        : collider{Collider{proxy<space>(boundary.levelset), boundary.type/*, boundary.s,
                            boundary.dsdt, boundary.R, boundary.omega, boundary.b, boundary.dbdt*/}},
          partition{proxy<space>(table)},
          grids{proxy<space>(grids)}, curTime{t} {}

    constexpr void operator()(typename grids_t::size_type blockid,
                              typename grids_t::cell_index_type cellid) noexcept {
      auto blockkey = partition._activeKeys[blockid];
      auto block = grids[blockid];

      if (block(0, cellid) > 0) {
        auto vel = block.pack<grids_t::dim>(1, cellid);
        auto pos = (blockkey * (value_type)grids_t::side_length + grids_t::cellid_to_coord(cellid))
                   * grids._dx;

        collider.resolveCollision(pos, vel);

        block.set(1, cellid, vel);
      }
    }

    collider_t collider;
    partition_t partition;
    grids_t grids;
    float curTime;
  };
#endif

}  // namespace zs