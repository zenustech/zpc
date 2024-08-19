#pragma once
#include "zensim/ZpcMeta.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/py_interop/VectorView.hpp"

namespace zs {

  /// @note modification of bvh through BvhViewLite in kernel is prohibited
  /// the maintenance of bvh is only allowed through certain routines issued on host
  template <int dim_, typename T_, typename Ti_ = int> struct BvhViewLite {  // T may be const
    static constexpr int dim = dim_;
    using value_type = T_;
    using bv_t = AABBBox<dim, value_type>;
    using index_type = make_signed_t<Ti_>;
    using size_type = make_unsigned_t<Ti_>;

    constexpr BvhViewLite() noexcept = default;
    ~BvhViewLite() = default;
    constexpr BvhViewLite(size_type numNodes, const bv_t *const bvs,
                          const index_type *const parents, const index_type *const levels,
                          const index_type *const leafInds,
                          const index_type *const auxIndices) noexcept
        : _orderedBvs{bvs},
          _parents{parents},
          _levels{levels},
          _leafInds{leafInds},
          _auxIndices{auxIndices},
          _numNodes{numNodes} {}

    constexpr size_type numNodes() const noexcept { return _numNodes; }
    constexpr size_type numLeaves() const noexcept {
      return numNodes() > 2 ? (numNodes() + 1) / 2 : _numNodes;
    }

    constexpr bv_t getNodeBV(index_type node) const { return _orderedBvs[node]; }

    template <typename VecT, class F> constexpr void ray_intersect(const VecInterface<VecT> &ro,
                                                                   const VecInterface<VecT> &rd,
                                                                   F &&f) const {  // hit distance
      if (auto nl = numNodes(); nl <= 2) {
        for (index_type i = 0; i != nl; ++i) {
          if (ray_box_intersect(ro, rd, getNodeBV(i))) f(i);
        }
        return;
      }
      index_type node = 0;
      while (node != -1 && node != _numNodes) {
        index_type level = _levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (!ray_box_intersect(ro, rd, getNodeBV(node))) break;
        // leaf node check
        if (level == 0) {
          if (ray_box_intersect(ro, rd, getNodeBV(node))) f(_auxIndices[node]);
          node++;
        } else  // separate at internal nodes
          node = _auxIndices[node];
      }
    }
    /// @note dist must be updated within 'f'
    template <typename VecT, class F, bool IndexRequired = false>
    constexpr auto find_nearest(const VecInterface<VecT> &p, F &&f, typename VecT::value_type cap,
                                wrapv<IndexRequired> = {}) const {
      using T = typename VecT::value_type;
      index_type idx = -1;
      T dist = cap;
      if (auto nl = numNodes(); nl <= 2) {
        for (index_type i = 0; i != nl; ++i) {
          if (auto d = distance(p, getNodeBV(i)); d < dist) {
            f(i, dist, idx);
          }
        }
        if constexpr (IndexRequired)
          return zs::make_tuple(idx, dist);
        else
          return dist;
      }
      index_type node = 0;
      while (node != -1 && node != _numNodes) {
        index_type level = _levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (auto d = distance(p, getNodeBV(node)); d > dist) break;
        // leaf node check
        if (level == 0) {
          if (auto d = distance(p, getNodeBV(node)); d < dist) f(_auxIndices[node], dist, idx);
          node++;
        } else  // separate at internal nodes
          node = _auxIndices[node];
      }
      if constexpr (IndexRequired)
        return zs::make_tuple(idx, dist);
      else
        return dist;
    }
    template <typename VecT, class F, bool IndexRequired = false>
    constexpr auto find_nearest(const VecInterface<VecT> &p, F &&f,
                                wrapv<IndexRequired> = {}) const {
      using T = typename VecT::value_type;
      index_type idx = -1;
      T dist = detail::deduce_numeric_max<T>();
      if (auto nl = numNodes(); nl <= 2) {
        for (index_type i = 0; i != nl; ++i) {
          if (auto d = distance(p, getNodeBV(i)); d < dist) {
            f(i, dist, idx);
          }
        }
        if constexpr (IndexRequired)
          return zs::make_tuple(idx, dist);
        else
          return dist;
      }
      index_type node = 0;
      while (node != -1 && node != _numNodes) {
        index_type level = _levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (auto d = distance(p, getNodeBV(node)); d > dist) break;
        // leaf node check
        if (level == 0) {
          if (auto d = distance(p, getNodeBV(node)); d < dist) f(_auxIndices[node], dist, idx);
          node++;
        } else  // separate at internal nodes
          node = _auxIndices[node];
      }
      if constexpr (IndexRequired)
        return zs::make_tuple(idx, dist);
      else
        return dist;
    }
    template <typename VecT, bool IndexRequired = false>
    constexpr auto find_nearest_point(const VecInterface<VecT> &p,
                                      typename VecT::value_type dist2
                                      = detail::deduce_numeric_max<value_type>(),
                                      index_type idx = -1, wrapv<IndexRequired> = {}) const {
      using T = typename VecT::value_type;
      if (auto nl = numNodes(); nl <= 2) {
        for (index_type i = 0; i != nl; ++i) {
          if (auto d2 = (p - getNodeBV(i)._min).l2NormSqr(); d2 < dist2) {
            dist2 = d2;
            idx = i;
            // f(i, dist, idx);
          }
        }
        if constexpr (IndexRequired)
          return zs::make_tuple(idx, zs::sqrt(dist2));
        else
          return zs::sqrt(dist2);
      }
      index_type node = 0;
      while (node != -1 && node != _numNodes) {
        index_type level = _levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (auto d = zs::max((value_type)0, distance(p, getNodeBV(node))); d * d > dist2) break;
        // leaf node check
        if (level == 0) {
          if (auto d2 = (p - getNodeBV(node)._min).l2NormSqr(); d2 < dist2) {
            dist2 = d2;
            idx = _auxIndices[node];
            // f(_auxIndices[node], dist, idx);
          }
          node++;
        } else  // separate at internal nodes
          node = _auxIndices[node];
      }
      if constexpr (IndexRequired)
        return zs::make_tuple(idx, zs::sqrt(dist2));
      else
        return zs::sqrt(dist2);
    }
    /// @note F return_value indicates early exit
    template <typename BV, class F> constexpr void iter_neighbors(const BV &bv, F &&f) const {
      if (auto nl = numNodes(); nl <= 2) {
        for (index_type i = 0; i != nl; ++i) {
          if (overlaps(getNodeBV(i), bv)) {
            if constexpr (is_same_v<decltype(declval<F>()(declval<index_type>())), void>)
              f(i);
            else {
              if (f(i)) return;
            }
          }
        }
        return;
      }
      index_type node = 0;
      while (node != -1 && node != _numNodes) {
        index_type level = _levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (!overlaps(getNodeBV(node), bv)) break;
        // leaf node check
        if (level == 0) {
          if (overlaps(getNodeBV(node), bv)) {
            if constexpr (is_same_v<decltype(declval<F>()(declval<index_type>())), void>)
              f(_auxIndices[node]);
            else {
              if (f(_auxIndices[node])) return;
            }
          }
          node++;
        } else  // separate at internal nodes
          node = _auxIndices[node];
      }
    }

    VectorViewLite<const bv_t> _orderedBvs{};
    VectorViewLite<const index_type> _parents{}, _levels{}, _leafInds{}, _auxIndices{};
    size_type _numNodes{0};
  };

}  // namespace zs