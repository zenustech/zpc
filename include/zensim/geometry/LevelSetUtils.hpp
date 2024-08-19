#pragma once
#include "SparseLevelSet.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/Vec.h"

namespace zs {

  template <typename ExecPol, int dim, grid_e category>
  void flood_fill_levelset(ExecPol &&policy, SparseLevelSet<dim, category> &ls);

  template <typename GridView> struct InitFloodFillGridChannels {
    using grid_view_t = GridView;
    using channel_counter_type = typename grid_view_t::channel_counter_type;
    using cell_index_type = typename grid_view_t::cell_index_type;
    using value_type = typename grid_view_t::value_type;
    using size_type = typename grid_view_t::size_type;
    static constexpr int dim = grid_view_t::dim;

    explicit InitFloodFillGridChannels(GridView grid) : grid{grid} {}

    constexpr void operator()(size_type cellid) noexcept {
      size_type blockid = cellid / grid_view_t::block_space();
      cellid %= grid_view_t::block_space();
      auto block = grid.block(blockid);
      block("tag", cellid) = 0;
      block("tagmask", cellid) = block("mask", cellid);
    }

    grid_view_t grid;
  };

  template <execspace_e space, typename SparseLevelSetT> struct ReserveForNeighbor {
    using spls_t = SparseLevelSetView<space, SparseLevelSetT>;
    using grid_view_t = typename spls_t::grid_view_t;
    using size_type = typename spls_t::size_type;
    using value_type = typename spls_t::value_type;
    using cell_index_type = typename grid_view_t::cell_index_type;
    static constexpr int dim = spls_t::dim;
    static constexpr int side_length = grid_view_t::side_length;

    ReserveForNeighbor() = default;
    explicit ReserveForNeighbor(SparseLevelSetT &ls) : ls{proxy<space>(ls)} {}

    constexpr void operator()(size_type cellid) noexcept {
      size_type blockid = cellid / grid_view_t::block_space();
      cellid %= grid_view_t::block_space();
      auto coord
          = ls._table._activeKeys[blockid] + ls._grid.cellid_to_coord(cellid).template cast<int>();

      auto block = ls._grid.block(blockid);

      if (block("mask", cellid) == 0 || block("tagmask", cellid) == 0 || block("tag", cellid) == 1)
        return;
      if (block("sdf", cellid) >= 0) return;

      {
        // iterating neighbors
        for (int i = -1; i != 3; i += 2)
          for (int j = 0; j != dim; ++j) {
            auto neighborCoord = coord;
            neighborCoord[j] += i;

            auto neighborBlockid = neighborCoord;
            for (int d = 0; d != dim; ++d)
              neighborBlockid[d] -= (neighborCoord[d] & (side_length - 1));

            auto neighborNo = ls._table.query(neighborBlockid);
            if (neighborNo < 0) {
              if ((neighborNo = ls._table.insert(neighborBlockid)) >= 0) {
                auto neighborBlock = ls._grid.block(neighborNo);
                for (cell_index_type ci = 0, ed = grid_view_t::block_space(); ci != ed; ++ci) {
                  neighborBlock("mask", ci) = 0;
                  neighborBlock("tagmask", ci) = 0;
                }
              }
            }
          }
      }
    }

    spls_t ls;
  };

  template <execspace_e space, typename SparseLevelSetT, typename VectorT> struct MarkInteriorTag {
    using vector_t = VectorView<space, VectorT>;
    using spls_t = SparseLevelSetView<space, SparseLevelSetT>;
    using grid_view_t = typename spls_t::grid_view_t;
    using size_type = typename spls_t::size_type;
    using value_type = typename spls_t::value_type;
    using cell_index_type = typename grid_view_t::cell_index_type;
    static constexpr int dim = spls_t::dim;
    static constexpr int side_length = grid_view_t::side_length;

    MarkInteriorTag() = default;
    explicit MarkInteriorTag(SparseLevelSetT &ls, VectorT &seeds, int *cnt)
        : ls{proxy<space>(ls)}, seeds{proxy<space>(seeds)}, cnt{cnt} {}

    constexpr void operator()(size_type cellid) noexcept {
      size_type blockid = cellid / grid_view_t::block_space();
      cellid %= grid_view_t::block_space();
      auto coord
          = ls._table._activeKeys[blockid] + ls._grid.cellid_to_coord(cellid).template cast<int>();

      auto block = ls._grid.block(blockid);

      if (block("mask", cellid) == 0 || block("tagmask", cellid) == 0 || block("tag", cellid) == 1)
        return;
      if (block("sdf", cellid) >= 0) return;

      {
        // iterating neighbors
        for (int i = -1; i != 3; i += 2)
          for (int j = 0; j != dim; ++j) {
            auto neighborCoord = coord;
            neighborCoord[j] += i;

            auto neighborBlockid = neighborCoord;
            for (int d = 0; d != dim; ++d)
              neighborBlockid[d] -= (neighborCoord[d] & (side_length - 1));

            auto neighborNo = ls._table.query(neighborBlockid);
#if 0
            if (neighborNo < 0)
              printf("Ill be damned (%d, %d, %d) finds (%d, %d, %d) at %d\n", coord[0], coord[1],
                     coord[2], neighborBlockid[0], neighborBlockid[1], neighborBlockid[2],
                     (int)neighborNo);
#endif

            neighborCoord -= neighborBlockid;
            auto neighborBlock = ls._grid.block(neighborNo);
            if (neighborBlock("mask", neighborCoord) == 0) {
              neighborBlock("tagmask", neighborCoord) = 1;
              neighborBlock("tag", neighborCoord) = 0;
              seeds(atomic_add(wrapv<space>{}, cnt, 1)) = neighborNo * grid_view_t::block_space()
                                                          + ls._grid.coord_to_cellid(neighborCoord);
            }
          }
        block("tag", cellid) = 1;
      }
    }

    spls_t ls;
    vector_t seeds;
    int *cnt;
  };

  template <execspace_e space, typename SparseLevelSetT, typename VectorT> struct ComputeTaggedSDF {
    using vector_t = VectorView<space, VectorT>;
    using spls_t = SparseLevelSetView<space, SparseLevelSetT>;
    using grid_view_t = typename spls_t::grid_view_t;
    using size_type = typename spls_t::size_type;
    using value_type = typename spls_t::value_type;
    using cell_index_type = typename grid_view_t::cell_index_type;
    static constexpr int dim = spls_t::dim;
    static constexpr int side_length = grid_view_t::side_length;

    ComputeTaggedSDF() = default;
    explicit ComputeTaggedSDF(SparseLevelSetT &ls, VectorT &seeds)
        : ls{proxy<space>(ls)}, seeds{proxy<space>(seeds)} {}

    constexpr void operator()(int id) noexcept {
      size_type cellid = seeds[id];
      size_type blockid = cellid / grid_view_t::block_space();
      cellid %= grid_view_t::block_space();
      auto coord
          = ls._table._activeKeys[blockid] + ls._grid.cellid_to_coord(cellid).template cast<int>();
      auto block = ls._grid.block(blockid);

      if (block("mask", cellid) == 1) return;

      if (block("tagmask", cellid) == 1 && block("tag", cellid) == 0) {
        value_type dis[dim] = {};
        for (auto &&d : dis) d = detail::deduce_numeric_max<value_type>();
        value_type sign[dim] = {};

        for (int i = -1; i != 3; i += 2)
          for (int j = 0; j != dim; ++j) {
            auto neighborCoord = coord;
            neighborCoord[j] += i;

            auto neighborBlockid = neighborCoord;
            for (int d = 0; d != dim; ++d)
              neighborBlockid[d] -= (neighborCoord[d] & (side_length - 1));
            auto neighborNo = ls._table.query(neighborBlockid);
            if (neighborNo < 0) {
              continue;  // neighbor not found
            }
            auto neighborBlock = ls._grid.block(neighborNo);

            neighborCoord -= neighborBlockid;
            if (neighborBlock("tagmask", neighborCoord) == 0
                || neighborBlock("tag", neighborCoord) == 0)
              continue;  // neighbor sdf not yet computed
            auto neighborValue = neighborBlock("sdf", neighborCoord);

            for (int t = 0; t != dim; ++t)
              if (auto v = zs::abs(neighborValue); v < dis[t] && neighborValue != 0) {
                for (int tt = dim - 1; tt > t; --tt) {
                  dis[tt] = dis[tt - 1];
                  sign[tt] = sign[tt - 1];
                }
                dis[t] = v;
                sign[t] = neighborValue / v;
                break;  // next j (dimension)
              }
          }

        // not yet ready
        value_type d = dis[0] + ls._grid._dx;
        if constexpr (dim == 2)
          if (d > dis[1]) {
            d = 0.5
                * (dis[0] + dis[1]
                   + zs::sqrt(2 * ls._grid._dx * ls._grid._dx
                              - (dis[1] - dis[0]) * (dis[1] - dis[0])));
            if constexpr (dim == 3)
              if (d > dis[2]) {
                value_type delta = dis[0] + dis[1] + dis[2];
                delta = delta * delta
                        - 3
                              * (dis[0] * dis[0] + dis[1] * dis[1] + dis[2] * dis[2]
                                 - ls._grid._dx * ls._grid._dx);
                if (delta < 0) delta = 0;
                d = 0.3333 * (dis[0] + dis[1] + dis[2] + zs::sqrt(delta));
              }
          }

        auto prev = block("sdf", cellid);
        block("sdf", cellid) = sign[0] * d;
        block("mask", cellid) = 1;

        if constexpr (false) {
          auto bid = ls._table._activeKeys[blockid];
          auto cid = ls._grid.cellid_to_coord(cellid).template cast<int>();
          printf(
              "sdf of (%d): (%d, %d) [(%d): %d, %d; (%d) %d, %d]is being computed, "
              "sdf: %f -> %f, tag %d (%d)\n",
              (int)prev, (int)coord[0], (int)coord[1], (int)blockid, (int)bid[0], (int)bid[1],
              (int)cellid, (int)cid[0], (int)cid[1], prev, block("sdf", cellid),
              (int)block("tag", cellid), (int)block("tagmask", cellid));
        }
      }
    }

    spls_t ls;
    vector_t seeds;
  };

}  // namespace zs