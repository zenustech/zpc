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
    using size_type = typename spls_t::size_type;
    using value_type = typename spls_t::value_type;
    using cell_index_type = typename grid_view_t::cell_index_type;
    static constexpr int dim = spls_t::dim;
    static constexpr auto side_length = grid_view_t::side_length;

    MarkInteriorTag() = default;
    explicit MarkInteriorTag(SparseLevelSetT &ls, int *modified)
        : ls{proxy<space>(ls)}, modified{modified} {}

    constexpr void operator()(size_type cellid) noexcept {
      size_type blockid = cellid / grid_view_t::block_space();
      cellid %= grid_view_t::block_space();
      auto coord = ls._table._activeKeys[blockid] * side_length + ls._grid.cellid_to_coord(cellid);
      auto block = ls._grid.block(blockid);

      if (block("sdf", cellid) > 0) return;

      if (block("tag", cellid) == 1) {  // expand tag[1] voxels
        int count = 0;
        // iterating neighbors
        for (int i = -1; i != 3; i += 2)
          for (int j = 0; j != dim; ++j) {
            auto neighborCoord = coord;
            neighborCoord[j] += i;

            auto neighborBlockid = neighborCoord;
            for (int d = 0; d != dim; ++d)
              neighborBlockid[d] += (neighborCoord[d] < 0 ? -side_length + 1 : 0);
            neighborBlockid = neighborBlockid / side_length;
            neighborCoord -= neighborBlockid * side_length;

            auto neighborNo = ls._table.query(neighborBlockid);
            if (neighborNo < 0) {
              if ((neighborNo = ls._table.insert(neighborBlockid)) >= 0) {
                // printf("adding neighbor %d, %d, %d at %d\n", (int)neighborCoord[0],
                //       (int)neighborCoord[1], (int)neighborCoord[2], (int)neighborNo);
                auto neighborBlock = ls._grid.block(neighborNo);
                for (cell_index_type ci = 0, ed = grid_view_t::block_space(); ci != ed; ++ci)
                  neighborBlock("tag", ci) = 0;
              }
              *modified = 1;
            } else if (ls._grid.block(neighborNo)("tag", neighborCoord) >= 1)
              count++;
          }
        if (count == 2 * dim) {
          block("tag", cellid) = 2;
          *modified = 1;
        }
      }
    }

    spls_t ls;
    int *modified;
  };

  template <execspace_e space, typename SparseLevelSetT> struct ComputeTaggedSDF {
    using spls_t = SparseLevelSetView<space, SparseLevelSetT>;
    using grid_view_t = typename spls_t::grid_view_t;
    using size_type = typename spls_t::size_type;
    using value_type = typename spls_t::value_type;
    using cell_index_type = typename grid_view_t::cell_index_type;
    static constexpr int dim = spls_t::dim;
    static constexpr auto side_length = grid_view_t::side_length;

    ComputeTaggedSDF() = default;
    explicit ComputeTaggedSDF(SparseLevelSetT &ls, int *modified)
        : ls{proxy<space>(ls)}, modified{modified} {}

    constexpr void operator()(size_type cellid) noexcept {
      size_type blockid = cellid / grid_view_t::block_space();
      cellid %= grid_view_t::block_space();
      auto coord = ls._table._activeKeys[blockid] * side_length + ls._grid.cellid_to_coord(cellid);
      auto block = ls._grid.block(blockid);

      if (block("tag", cellid) == 0) {
        value_type dis[dim] = {};
        for (auto &&d : dis) d = std::numeric_limits<value_type>::max();
        value_type sign[dim] = {};
        bool flags[dim] = {};
        for (auto &&f : flags) f = false;

        for (int i = -1; i != 3; i += 2)
          for (int j = 0; j != dim; ++j) {
            auto neighborCoord = coord;
            neighborCoord[j] += i;

            auto neighborBlockid = neighborCoord;
            for (int d = 0; d != dim; ++d)
              neighborBlockid[d] += (neighborCoord[d] < 0 ? -side_length + 1 : 0);
            neighborBlockid = neighborBlockid / side_length;
            neighborCoord -= neighborBlockid * side_length;

            auto neighborNo = ls._table.query(neighborBlockid);
            if (neighborNo < 0) continue;  // neighbor not found
            auto neighborBlock = ls._grid.block(neighborNo);
            if (neighborBlock("tag", neighborCoord) == 0)
              continue;  // neighbor sdf not yet computed
            auto neighborValue = neighborBlock("sdf", neighborCoord);

            for (int t = 0; t != dim; ++t)
              if (auto v = gcem::abs(neighborValue); v < dis[t] && neighborValue != 0) {
                for (int tt = dim - 1; tt >= t + 1; --tt) {
                  dis[tt] = dis[tt - 1];
                  sign[tt] = sign[tt - 1];
                }
                dis[t] = v;
                sign[t] = neighborValue / v;
                flags[j] = true;
                break;  // next j (dimension)
              }
          }

        // not yet ready
        for (auto &&f : flags)
          if (!f) return;
        // if (count != dim) return;

        value_type d = dis[0] + ls._dx;
        if (d > dis[1]) {
          d = 0.5
              * (dis[0] + dis[1]
                 + gcem::sqrt(2 * ls._dx * ls._dx - (dis[1] - dis[0]) * (dis[1] - dis[0])));
          if constexpr (dim > 2)
            if (d > dis[2]) {
              value_type delta = dis[0] + dis[1] + dis[2];
              delta = delta * delta
                      - 3 * (dis[0] * dis[0] + dis[1] * dis[1] + dis[2] * dis[2] - ls._dx * ls._dx);
              if (delta < 0) delta = 0;
              d = 0.3333 * (dis[0] + dis[1] + dis[2] + gcem::sqrt(delta));
            }
        }
        *modified = 1;
        block("sdf", cellid) = sign[0] * d;
        block("tag", cellid) = 1;
      }
    }

    spls_t ls;
    int *modified;
  };

}  // namespace zs