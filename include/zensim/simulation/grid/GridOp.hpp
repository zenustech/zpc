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

  template <execspace_e space, typename T, int d, auto l>
  CleanGridBlocks(wrapv<space>, Grids<T, d, l>)
      -> CleanGridBlocks<GridsView<space, Grids<T, d, l>>>;

  template <execspace_e space, transfer_scheme_e scheme, typename T, int d, auto l, typename Sth>
  ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, Grids<T, d, l>, float dt, Sth,
                           float *maxVel)
      -> ComputeGridBlockVelocity<scheme, GridsView<space, Grids<T, d, l>>>;

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

  template <execspace_e space, typename GridsT> struct CleanGridBlocks<GridsView<space, GridsT>> {
    using grids_t = GridsView<space, GridsT>;

    explicit CleanGridBlocks(wrapv<space>, GridsT &grids) : grids{proxy<space>(grids)} {}

    constexpr void operator()(typename GridsT::size_type blockid,
                              typename GridsT::cell_index_type cellid) noexcept {
      using value_type = typename GridsT::value_type;
      auto grid = grids.grid(collocated_c);
      auto block = grid.block(blockid);
      auto nchns = grid.numChannels();
      for (int i = 0; i != nchns; ++i) block(i, cellid) = 0;
    }

    grids_t grids;
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
    using grid_view_t = decltype(proxy<space>(declval<Grid<T, d, l, category> &>()));
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
    using grid_view_t = decltype(proxy<space>(declval<Grid<T, d, l, category> &>()));
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
    using grid_view_t = decltype(proxy<space>(declval<Grid<T, d, l, category> &>()));
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