#pragma once
#include "SparseLevelSet.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/Vec.h"
#include "zensim/math/curve/InterpolationKernel.hpp"
#include "zensim/tpls/magic_enum/magic_enum.hpp"

namespace zs {

  template <typename LsvT, kernel_e kt_ = kernel_e::quadratic, int drv_order = 0>
  struct LevelSetArena {
    using lsv_t = LsvT;
    using value_type = typename lsv_t::value_type;
    using index_type = typename lsv_t::index_type;

    static_assert(std::is_signed_v<index_type>, "index_type should be a signed integer.");
    static constexpr grid_e category = lsv_t::category;
    static constexpr kernel_e kt = kt_;
    static constexpr int dim = lsv_t::dim;
    static constexpr index_type width = magic_enum::enum_integer(kt);
    static constexpr int deriv_order = drv_order;
    static constexpr index_type coord_offset
        = kt == kernel_e::quadratic ? -1 : (kt == kernel_e::linear ? 0 : -2);

    using TV = typename lsv_t::TV;
    using TWM = vec<value_type, dim, width>;
    using IV = typename lsv_t::IV;

    using coord_index_type = typename lsv_t::coord_index_type;
    using channel_counter_type = typename lsv_t::channel_counter_type;
    using cell_index_type = typename lsv_t::cell_index_type;

    static_assert(deriv_order >= 0 && deriv_order <= 2,
                  "weight derivative order should be a integer within [0, 2]");

    using WeightScratchPad
        = conditional_t<deriv_order == 0, tuple<TWM>,
                        conditional_t<deriv_order == 1, tuple<TWM, TWM>, tuple<TWM, TWM, TWM>>>;

    template <typename Val, std::size_t... Is>
    static constexpr auto arena_type_impl(index_seq<Is...>) {
      return vec<Val, (Is + 1 > 0 ? width : width)...>{};
    }
    template <typename Val, int d> static constexpr auto arena_type() {
      return arena_type_impl<Val>(std::make_index_sequence<d>{});
    }
    template <typename Val> using Arena = RM_CVREF_T(arena_type<Val, dim>());

    /// constructors
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate != grid_e::staggered> = 0>
    constexpr LevelSetArena(lsv_t &lsv, const VecInterface<VecT> &x, wrapv<kt> = {},
                            wrapv<deriv_order> = {}) noexcept
        : lsPtr{&lsv}, weights{}, iLocalPos{}, iCorner{} {
      auto X = lsv.worldToIndex(x);
      for (int d = 0; d != dim; ++d) iCorner[d] = lower_trunc(X[d]) + coord_offset;
      iLocalPos = X - iCorner;
      if constexpr (kt == kernel_e::linear)
        weights = linear_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::quadratic)
        weights = quadratic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::cubic)
        weights = cubic_bspline_weights<deriv_order>(iLocalPos);
    }
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate == grid_e::staggered> = 0>
    constexpr LevelSetArena(lsv_t &lsv, const VecInterface<VecT> &x, int f, wrapv<kt> = {},
                            wrapv<deriv_order> = {}) noexcept
        : lsPtr{&lsv}, weights{}, iLocalPos{}, iCorner{} {
      auto X = lsv.worldToIndex(x);
      auto delta = TV::init([f](int d) { return d != f ? (value_type)0.5 : (value_type)0; });
      for (int d = 0; d != dim; ++d) iCorner[d] = lower_trunc(X[d] - delta[d]) + coord_offset;
      iLocalPos = X - (iCorner + delta);
      if constexpr (kt == kernel_e::linear)
        weights = linear_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::quadratic)
        weights = quadratic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::cubic)
        weights = cubic_bspline_weights<deriv_order>(iLocalPos);
    }

    /// scalar arena
    constexpr Arena<value_type> arena(typename lsv_t::channel_counter_type chn,
                                      typename lsv_t::value_type defaultVal = 0) const noexcept {
      // ensure that chn's orientation is aligned with initialization if within a staggered grid
      Arena<value_type> pad{};
      int cnt = 0;
      for (auto offset : ndrange<dim>(width)) {
        pad.val(offset) = lsPtr->value_or(
            chn, iCorner + make_vec<typename lsv_t::coord_index_type>(offset), defaultVal);
      }
      return pad;
    }
    constexpr Arena<value_type> arena(const SmallString &propName,
                                      typename lsv_t::channel_counter_type chn = 0,
                                      typename lsv_t::value_type defaultVal = 0) const noexcept {
      return arena(lsPtr->_grid.propertyOffset(propName) + chn, defaultVal);
    }

    /// minimum
    constexpr value_type minimum(typename lsv_t::channel_counter_type chn = 0) const noexcept {
      auto pad = arena(chn, limits<value_type>::max());
      if constexpr (kt == kernel_e::linear)
        return xlerp(iLocalPos, pad);
      else {
        value_type ret = limits<value_type>::max();
        for (auto offset : ndrange<dim>(width))
          if (const auto &v = pad.val(offset); v < ret) ret = v;
        return ret;
      }
    }
    constexpr value_type minimum(const SmallString &propName,
                                 typename lsv_t::channel_counter_type chn = 0) const noexcept {
      return minimum(lsPtr->_grid.propertyOffset(propName) + chn);
    }

    /// maximum
    constexpr value_type maximum(typename lsv_t::channel_counter_type chn = 0) const noexcept {
      auto pad = arena(chn, limits<value_type>::lowest());
      if constexpr (kt == kernel_e::linear)
        return xlerp(iLocalPos, pad);
      else {
        value_type ret = limits<value_type>::lowest();
        for (auto offset : ndrange<dim>(width))
          if (const auto &v = pad.val(offset); v > ret) ret = v;
        return ret;
      }
    }
    constexpr value_type maximum(const SmallString &propName,
                                 typename lsv_t::channel_counter_type chn = 0) const noexcept {
      return maximum(lsPtr->_grid.propertyOffset(propName) + chn);
    }

    /// isample
    constexpr value_type isample(typename lsv_t::channel_counter_type chn,
                                 typename lsv_t::value_type defaultVal) const noexcept {
      auto pad = arena(chn, defaultVal);
      if constexpr (kt == kernel_e::linear)
        return xlerp(iLocalPos, pad);
      else {
        value_type ret = 0;
        for (auto offset : ndrange<dim>(width)) ret += weight(offset) * pad.val(offset);
        return ret;
      }
    }
    constexpr value_type isample(const SmallString &propName,
                                 typename lsv_t::channel_counter_type chn,
                                 typename lsv_t::value_type defaultVal) const noexcept {
      auto pad = arena(propName, chn, defaultVal);
      if constexpr (kt == kernel_e::linear)
        return xlerp(iLocalPos, pad);
      else {
        value_type ret = 0;
        for (auto offset : ndrange<dim>(width)) ret += weight(offset) * pad.val(offset);
        return ret;
      }
    }

    /// weight
    template <typename... Tn>
    constexpr value_type weight(const std::tuple<Tn...> &loc) const noexcept {
      return weight_impl(loc, std::index_sequence_for<Tn...>{});
    }
    template <typename... Tn,
              enable_if_all<(!is_std_tuple<Tn>() && ... && (sizeof...(Tn) == dim))> = 0>
    constexpr auto weight(Tn &&...is) const noexcept {
      return weight(std::forward_as_tuple(FWD(is)...));
    }
    /// weight gradient
    template <std::size_t I, typename... Tn, auto ord = deriv_order>
    constexpr std::enable_if_t<(ord > 0), value_type> weightGradient(
        const std::tuple<Tn...> &loc) const noexcept {
      return weightGradient_impl<I>(loc, std::index_sequence_for<Tn...>{});
    }
    void printWeights() {
      value_type sum = 0;
      for (int d = 0; d != dim; ++d) {
        for (int w = 0; w != width; ++w) {
          sum += get<0>(weights)(d, w);
          fmt::print("weights({}, {}): [{}]\t", d, w, get<0>(weights)(d, w));
          if constexpr (deriv_order > 0) fmt::print("[{}]\t", get<1>(weights)(d, w));
          if constexpr (deriv_order > 1) fmt::print("[{}]\t", get<2>(weights)(d, w));
          fmt::print("\n");
        }
      }
      fmt::print("weight sum: {}\n", sum);
    }

    lsv_t *lsPtr{nullptr};
    WeightScratchPad weights{};
    TV iLocalPos{TV::zeros()};  // index-space local offset
    IV iCorner{IV::zeros()};    // index-space global coord

  protected:
    template <typename... Tn, std::size_t... Is,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim)> = 0>
    constexpr value_type weight_impl(const std::tuple<Tn...> &loc,
                                     index_seq<Is...>) const noexcept {
      value_type ret{1};
      ((void)(ret *= get<0>(weights)(Is, std::get<Is>(loc))), ...);
      return ret;
    }
    template <std::size_t I, typename... Tn, std::size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr value_type weightGradient_impl(const std::tuple<Tn...> &loc,
                                             index_seq<Is...>) const noexcept {
      value_type ret{1};
      ((void)(ret *= (I == Is ? get<1>(weights)(Is, std::get<Is>(loc))
                              : get<0>(weights)(Is, std::get<Is>(loc)))),
       ...);
      return ret;
    }
    template <typename... Tn, std::size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr TV weightGradients_impl(const std::tuple<Tn...> &loc,
                                      index_seq<Is...>) const noexcept {
      return TV{weightGradient_impl<Is>(loc, index_seq<Is...>{})...};
    }
  };

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
        for (auto &&d : dis) d = limits<value_type>::max();
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