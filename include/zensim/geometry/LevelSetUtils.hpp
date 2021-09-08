#pragma once
#include "SparseLevelSet.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename ExecPol, int dim, grid_e category>
  void flood_fill_levelset(ExecPol &&policy, SparseLevelSet<dim, category> &ls);

  template <typename GridView> struct FillGridChannels {
    using grid_view_t = GridView;
    using channel_counter_type = typename grid_view_t::channel_counter_type;
    using cell_index_type = typename grid_view_t::cell_index_type;
    using value_type = typename grid_view_t::value_type;
    using size_type = typename grid_view_t::size_type;
    static constexpr int dim = grid_view_t::dim;

    explicit FillGridChannels(GridView grid, channel_counter_type st = 0, value_type v = 0,
                              channel_counter_type cnt = 1)
        : grid{grid}, base{st}, cnt{cnt}, val{v} {}

    constexpr void operator()(size_type cellid) noexcept {
      size_type blockid = cellid / grid_view_t::block_space();
      cellid %= grid_view_t::block_space();
      auto block = grid.block(blockid);
      for (channel_counter_type i = 0; i != cnt; ++i) block(base + i, cellid) = val;
    }

    grid_view_t grid;
    value_type val;
    channel_counter_type base, cnt;
  };

  template <execspace_e space, typename SparseLevelSetT> struct MarkInteriorTag {
    using spls_t = SparseLevelSetView<space, SparseLevelSetT>;
    using grid_view_t = typename spls_t::grid_view_t;
    using value_type = typename spls_t ::value_type;
    static constexpr int dim = spls_t ::dim;

    MarkInteriorTag() = default;
    explicit MarkInteriorTag(const spls_t &&ls) : ls{ls} {}

    constexpr void operator()(typename grid_view_t::size_type blockid,
                              typename grid_view_t::cell_index_type cellid) noexcept {}

    spls_t ls;
  };

}  // namespace zs