#pragma once

#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/Intrinsics.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#if defined(__CUDACC__)
#  include <cooperative_groups.h>
#endif

namespace zs {

  struct _compute_bounding_box_naive {
    template <typename BoxT, typename ParamT>
    constexpr void operator()(BoxT &bv, const ParamT &params) noexcept {
      constexpr int dim = BoxT::dim;
      using value_type = typename BoxT::value_type;
      auto &[box, execTag] = params;
      for (int d = 0; d != dim; ++d) {
        atomic_min(execTag, &box[0]._min[d], bv._min[d] - 10 * limits<value_type>::epsilon());
        atomic_max(execTag, &box[0]._max[d], bv._max[d] + 10 * limits<value_type>::epsilon());
      }
    }
  };
  struct _compute_bounding_box_extraction {
    template <typename Ti, typename Box, typename ParamT>
    constexpr void operator()(Ti i, const Box &bv, ParamT &&params) noexcept {
      constexpr int dim = Box::dim;
      using value_type = typename Box::value_type;
      auto &[gmins, gmaxs, sz] = params;
      Ti offset{};
      for (int d = 0; d < dim; ++d) {
        offset = d * sz;
        gmins[offset + i] = bv._min[d] - 10 * limits<value_type>::epsilon();
        gmaxs[offset + i] = bv._max[d] + 10 * limits<value_type>::epsilon();
      }
    }
  };

  template <typename Policy, typename BoxRange,
            typename Box = RM_CVREF_T(*declval<BoxRange &&>().begin())>
  inline Box compute_bounding_box(Policy &&policy, BoxRange &&bvs) {
    constexpr int dim = Box::dim;
    constexpr auto space = remove_reference_t<Policy>::exec_tag::value;
    using value_type = typename Box::value_type;
    using TV = typename Box::TV;
    static_assert(sizeof(Box) == sizeof(value_type) * dim * 2, "???");

    Box gbv{};
    auto allocator = get_temporary_memory_source(policy);
    const size_t sz = range_size(bvs);

    bool shouldSync = policy.shouldSync();
    policy.sync(true);
#if 1
    Vector<Box> wholeBox{allocator, 1};
    constexpr auto defaultBox = Box{TV::constant(detail::deduce_numeric_max<value_type>()),
                                    TV::constant(detail::deduce_numeric_lowest<value_type>())};
    wholeBox.setVal(defaultBox);
    const auto &params = zs::make_tuple(proxy<space>(wholeBox), wrapv<space>{});
    policy(bvs, params, _compute_bounding_box_naive{});
    gbv = wholeBox.getVal();
#else
    Vector<value_type> ret{allocator, dim * 2};
    Vector<value_type> gmins{allocator, sz * dim}, gmaxs{allocator, sz * dim};

    const auto &params = zs::make_tuple(view<space>(gmins), view<space>(gmaxs), sz);
    policy(enumerate(bvs), params, _compute_bounding_box_extraction{});
    for (int d = 0; d != dim; ++d) {
      reduce(policy, zs::begin(gmins) + d * sz, zs::begin(gmins) + (d + 1) * sz, zs::begin(ret) + d,
             detail::deduce_numeric_max<value_type>(), getmin<value_type>{});
      reduce(policy, zs::begin(gmaxs) + d * sz, zs::begin(gmaxs) + (d + 1) * sz,
             zs::begin(ret) + dim + d, detail::deduce_numeric_lowest<value_type>(),
             getmax<value_type>{});
    }
    ret.retrieveVals((value_type *)(&gbv));
#endif
    policy.sync(shouldSync);
    return gbv;
  }

  template <int dim_ = 3, typename Index = int, typename ValueT = zs::f32,
            typename AllocatorT = zs::ZSPmrAllocator<>>
  struct LBvh {
    static constexpr int dim = dim_;
    using allocator_type = AllocatorT;
    using value_type = ValueT;
    // must be signed integer, since we are using -1 as sentinel value
    using index_type = zs::make_signed_t<Index>;
    using size_type = zs::make_unsigned_t<Index>;
    static_assert(is_floating_point_v<value_type>, "value_type should be floating point");
    static_assert(is_integral_v<index_type>, "index_type should be an integral");

    using mc_t = zs::u32;
    using Box = zs::AABBBox<dim, value_type>;
    using TV = zs::vec<value_type, dim>;
    using IV = zs::vec<index_type, dim>;
    using bvs_t = zs::Vector<Box, allocator_type>;
    using vector_t = zs::Vector<value_type, allocator_type>;
    using indices_t = zs::Vector<index_type, allocator_type>;

    constexpr decltype(auto) memoryLocation() const noexcept { return orderedBvs.memoryLocation(); }
    constexpr zs::ProcID devid() const noexcept { return orderedBvs.devid(); }
    constexpr zs::memsrc_e memspace() const noexcept { return orderedBvs.memspace(); }
    decltype(auto) get_allocator() const noexcept { return orderedBvs.get_allocator(); }
    decltype(auto) get_default_allocator(zs::memsrc_e mre, zs::ProcID devid) const {
      return orderedBvs.get_default_allocator(mre, devid);
    }

    LBvh() = default;

    LBvh clone(const allocator_type &allocator) const {
      LBvh ret{};
      ret.orderedBvs = orderedBvs.clone(allocator);
      ret.parents = parents.clone(allocator);
      ret.levels = levels.clone(allocator);
      ret.leafInds = leafInds.clone(allocator);
      ret.auxIndices = auxIndices.clone(allocator);
      return ret;
    }
    LBvh clone(const zs::MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    constexpr auto getNumLeaves() const noexcept { return leafInds.size(); }
    constexpr auto getNumNodes() const noexcept {
      auto nl = getNumLeaves();
      return nl > 2 ? nl * 2 - 1 : nl;
    }

    template <typename Policy, bool Refit = true>
    void build(Policy &&, const zs::Vector<zs::AABBBox<dim, value_type>> &primBvs,
               zs::wrapv<Refit> = {});

    template <typename Policy>
    void buildRefit(Policy &&, const zs::Vector<zs::AABBBox<dim, value_type>> &primBvs);

    template <typename Policy>
    void refit(Policy &&, const zs::Vector<zs::AABBBox<dim, value_type>> &primBvs);

    template <typename BvhView, typename BoxView> struct _GetBoxHelper {
      BvhView bvh;
      BoxView box;
      constexpr void operator()(int vi) noexcept {
        constexpr auto space = BvhView::space;
        auto bv = bvh.getNodeBV(vi);
        for (int d = 0; d != dim; ++d) {
          atomic_min(wrapv<space>{}, &box[0]._min[d], bv._min[d]);
          atomic_max(wrapv<space>{}, &box[0]._max[d], bv._max[d]);
        }
      }
    };

    template <typename Policy> Box getTotalBox(Policy &&pol) const {
      using namespace zs;
      constexpr auto space = remove_reference_t<Policy>::exec_tag::value;

      auto numLeaves = getNumLeaves();
      Vector<Box> box{orderedBvs.get_allocator(), 1};
      if (numLeaves <= 2) {
        using TV = typename Box::TV;
        box.setVal(Box{TV::constant(limits<value_type>::max()),
                       TV::constant(limits<value_type>::lowest())});
        auto bvhv = proxy<space>(*this);
        auto boxv = proxy<space>(box);
        pol(Collapse{numLeaves}, _GetBoxHelper<RM_CVREF_T(bvhv), RM_CVREF_T(boxv)>{bvhv, boxv});
        return box.getVal();
      } else {
        return orderedBvs.getVal();
      }
    }

    bvs_t orderedBvs;
    indices_t parents{}, levels{}, leafInds{}, auxIndices{};  // escape ids/ prim ids

    /// build
    struct _build_init_mc_id {
      template <typename ParamT>
      constexpr void operator()(size_type id, const ParamT &params) noexcept {
        auto &[wholeBox, primBvs, mcs, indices] = params;
        const auto bv = primBvs[id];
        auto c = bv.getBoxCenter();
        auto coord = wholeBox.getUniformCoord(c).template cast<f32>();  // vec<f32, dim>
        mcs[id] = morton_code<RM_REF_T(wholeBox)::dim>(coord);
        indices[id] = id;
      }
    };
    struct _build_init_depths {
      template <typename ParamT>
      constexpr void operator()(size_type idx, const ParamT &params) noexcept {
        auto &[indices, pInds, lDepths, numTrunk] = params;
        auto ind = indices[idx];
        pInds[idx] = ind;
        lDepths[idx] = 1;
        if (idx == 0) lDepths[numTrunk + 1] = 0;
      }
    };
    struct _build_build_topo {
      template <typename ParamT>
      constexpr void operator()(index_type idx, const ParamT &params) noexcept {
        auto &[mcs, tPars, tLcs, tRcs, tLs, tRs, lPars, lDepths, execTag, numTrunk] = params;
        index_type i = 0, j = 0;
        auto num_leaves = numTrunk + 1;
        // determine range
        if (idx == 0) {
          i = 0;
          j = (num_leaves - 1);
        } else {
          index_type left = idx, right = idx;
          index_type dir{};
          u32 minLZ{};
          u32 preCode = mcs(idx - 1);
          u32 curCode = mcs(idx);
          u32 nxtCode = mcs(idx + 1);

          if (preCode == curCode && curCode == nxtCode) {
            for (++right; right < num_leaves - 1; ++right)
              if (mcs(right) != mcs(right + 1)) break;
            j = right;
            i = left;
          } else {
            u32 lLZ = count_lz(execTag, preCode ^ curCode),
                rLZ = count_lz(execTag, nxtCode ^ curCode);
            if (lLZ > rLZ)  ///< expand to the left
              dir = -1, minLZ = rLZ;
            else  ///< expand to the right
              dir = 1, minLZ = lLZ;

            index_type step{};
            for (step = 2; right = left + step * dir,
                (right < num_leaves && right >= 0 ? count_lz(execTag, mcs(right) ^ curCode) > minLZ
                                                  : false);
                 step <<= 1)
              ;
            index_type len{};
            for (len = 0, step >>= 1; step >= 1; step >>= 1) {
              right = left + (len + step) * dir;
              if (right < num_leaves && right >= 0)
                if (count_lz(execTag, mcs(right) ^ curCode) > minLZ) len += step;
            }
            // printf("dealing node %d: %u %u %u dir:%d mlz:%d len:%d\n", idx, preCode, curCode,
            // nxtCode, dir, minLZ, len);
            if (dir == 1) {
              i = left;
              j = left + len;
            } else {
              i = left - len;
              j = left;
            }
          }
        }
        atomic_add(execTag, &lDepths[i], (index_type)1);
        tLs(idx) = i;
        tRs(idx) = j;
        // find split
        index_type gamma{};
        auto lCode = mcs(i);
        auto rCode = mcs(j);
        if (lCode == rCode)
          gamma = i;
        else {
          auto LZ = count_lz(execTag, lCode ^ rCode);
          index_type step{}, len{};
          for (step = (j - i + 1) >> 1, len = 0; true; step = (step + 1) >> 1) {
            if (i + len + step > numTrunk) continue;
            if (count_lz(execTag, mcs(i + len + step) ^ lCode) > LZ) len += step;
            if (step <= 1) break;
          }
          gamma = i + len;
        }
        tLcs(idx) = gamma;
        tRcs(idx) = gamma + 1;
        auto mi = i < j ? i : j;
        auto ma = i > j ? i : j;
        if (mi == gamma) {
          lPars(gamma) = idx;
          tLcs(idx) += numTrunk;
        } else
          tPars(gamma) = idx;
        if (ma == gamma + 1) {
          lPars(gamma + 1) = idx;
          tRcs(idx) += numTrunk;
        } else
          tPars(gamma + 1) = idx;

        if (idx == 0) tPars(0) = -1;
      }
    };
    struct _build_supp_topo {
      template <typename ParamT>
      constexpr void operator()(index_type idx, const ParamT &params) noexcept {
        auto &[levels, lOffsets, lPars, lLcas, tPars, tDst, numTrunk] = params;
        auto depth = lOffsets[idx + 1] - lOffsets[idx];
        auto dst = lOffsets[idx + 1] - 2;
        index_type node = lPars[idx], ch = idx + numTrunk;
        index_type level = 0;
        for (; --depth; node = tPars[node], --dst) {
          tDst[node] = dst;
          levels[dst] = ++level;
          ch = node;
        }
        lLcas[idx] = ch;
      }
    };
    struct _build_reorder_leaf {
      template <typename ParamT>
      constexpr void operator()(index_type idx, const ParamT &params) noexcept {
        auto &[lOffsets, lPars, auxIndices, parents, levels, pInds, lInds, tDst] = params;
        auto dst = lOffsets[idx + 1] - 1;
        // aux (primids)
        auxIndices[dst] = pInds[idx];
        // parent
        parents[dst] = tDst[lPars[idx]];
        // levels
        levels[dst] = 0;
        // leafinds
        lInds[idx] = dst;
      }
    };
    struct _build_reorder_trunk {
      template <typename ParamT>
      constexpr void operator()(index_type idx, const ParamT &params) noexcept {
        auto &[lLcas, lOffsets, auxIndices, parents, tRs, tPars, tDst, numTrunk] = params;
        auto dst = tDst[idx];
        auto r = tRs[idx];
        // aux (esc)
        if (r != numTrunk) {
          auto lca = lLcas[r + 1];
          auxIndices[dst] = lca < numTrunk ? tDst[lca] : lOffsets[r + 1];
        } else
          auxIndices[dst] = -1;
        // parent
        if (idx != 0)
          parents[dst] = tDst[tPars[idx]];
        else
          parents[dst] = -1;
        // levels for trunk are already set
      }
    };
    /// build refit
    struct _build_init_split_location {
      template <typename ParamT>
      constexpr void operator()(size_type id, mc_t &split, const ParamT &params) noexcept {
        auto &[mcs, numLeaves, execTag] = params;
        constexpr mc_t totalBits = sizeof(mc_t) * 8;
        if (id + 1 != numLeaves)
          split = totalBits - count_lz(execTag, mcs(id) ^ mcs(id + 1));
        else
          split = totalBits + 1;
      }
    };
    struct _build_build_topo_and_refit {
      template <typename ParamT>
      constexpr void operator()(index_type idx, const ParamT &params) noexcept {
        auto &[tPars, tLcs, tRcs, tLs, tRs, tBvs, lLcas, lDepths, lPars, numTrunk, primBvs, indices,
               splits, flags, execTag]
            = params;

        lLcas[idx] = idx + numTrunk;  // set to itself
        lDepths[idx] = 1;
        Index l = idx - 1, r = idx;  ///< (l, r]
        bool mark{false};

        if (l >= 0) mark = splits[l] < splits[r];  ///< true when right child, false otherwise

        int cur = mark ? l : r;
        lPars[idx] = cur;
        if (mark) {
          tRcs[cur] = idx + numTrunk;
          tRs[cur] = idx;
          // atomic_or(execTag, &marks(cur), 0x00000002u);
        } else {
          tLcs[cur] = idx + numTrunk;
          tLs[cur] = idx;
          // atomic_or(execTag, &marks(cur), 0x00000001u);
        }

        while (atomic_add(execTag, &flags[cur], 1) == 1) {
          {  // refit
            int lc = tLcs[cur], rc = tRcs[cur];
            // const auto childMask = marks(cur) & 3;
            const auto &leftBox = (lc >= numTrunk) ? primBvs[indices[lc - numTrunk]] : tBvs[lc];
            const auto &rightBox = (rc >= numTrunk) ? primBvs[indices[rc - numTrunk]] : tBvs[rc];

            for (int d = 0; d != dim; ++d) {
              tBvs[cur]._min[d] = zs::min(leftBox._min[d], rightBox._min[d]);
              tBvs[cur]._max[d] = zs::max(leftBox._max[d], rightBox._max[d]);
            }
          }
          // marks(cur) &= 0x00000007;

          l = tLs[cur] - 1;
          r = tRs[cur];
          lLcas[l + 1] = cur;
          lDepths[l + 1]++;

          thread_fence(execTag);  // this is needed

          if (l >= 0)
            mark = splits[l] < splits[r];  ///< true when right child, false otherwise
          else
            mark = false;

          if (l + 1 == 0 && r == numTrunk) {
            tPars[cur] = -1;
            // rt[0] = cur;  // assign root
            // marks(cur) &= 0xFFFFFFFB;
            break;
          }

          int par = mark ? l : r;
          tPars[cur] = par;
          if (mark) {
            tRcs[par] = cur;
            tRs[par] = r;
            // atomic_and(execTag, &marks(par), 0xFFFFFFFD), marks(cur) |= 0x00000004;
          } else {
            tLcs[par] = cur;
            tLs[par] = l + 1;
            // atomic_and(execTag, &marks(par), 0xFFFFFFFE), marks(cur) &= 0xFFFFFFFB;
          }
          cur = par;
        }
      }
    };
    struct _build_reorder_leaf_with_bv {
      template <typename ParamT>
      constexpr void operator()(index_type idx, const ParamT &params) noexcept {
        auto &[lOffsets, lPars, auxIndices, parents, levels, pInds, lInds, tDst, primBvs, bvs]
            = params;
        auto dst = lOffsets[idx + 1] - 1;
        auto pId = pInds[idx];
        // aux (primids)
        auxIndices[dst] = pId;
        // parent
        parents[dst] = tDst[lPars[idx]];
        // levels
        levels[dst] = 0;
        // leafinds
        lInds[idx] = dst;
        // bv
        bvs[dst] = primBvs[pId];
      }
    };
    struct _build_reorder_trunk_with_bv {
      template <typename ParamT>
      constexpr void operator()(index_type idx, const ParamT &params) noexcept {
        auto &[lLcas, lOffsets, auxIndices, parents, tRs, tPars, tDst, tBvs, bvs, numTrunk]
            = params;
        auto dst = tDst[idx];
        auto r = tRs[idx];
        // bv
        bvs[dst] = tBvs[idx];
        // aux (esc)
        if (r != numTrunk) {
          auto lca = lLcas[r + 1];
          auxIndices[dst] = lca < numTrunk ? tDst[lca] : lOffsets[r + 1];
        } else
          auxIndices[dst] = -1;
        // parent
        if (idx != 0)
          parents[dst] = tDst[tPars[idx]];
        else
          parents[dst] = -1;
        // levels for trunk are already set
      }
    };
    struct _refit_bottom_up {
      template <typename ParamT>
      constexpr void operator()(index_type idx, const ParamT &params) noexcept {
        auto &[primBvs, orderedBvs, auxIndices, leafInds, parents, levels, flags, execTag] = params;
        auto node = leafInds[idx];
        auto primid = auxIndices[node];
        auto bv = primBvs[primid];
        orderedBvs[node] = bv;
        node = parents[node];
        while (node != -1) {
          if (atomic_cas(execTag, &flags[node], 0, 1) == 0) break;
          auto lc = node + 1;
          auto rc = levels[lc] ? auxIndices[lc] : lc + 1;
          auto bv = orderedBvs[lc];
          auto rbv = orderedBvs[rc];
          merge(bv, rbv._min);
          merge(bv, rbv._max);
          thread_fence(execTag);  //__threadfence();
          orderedBvs[node] = bv;
          // if (node == 0)
          //     printf("diag size: %f\n", (float)(bv._max - bv._min).length());
          node = parents[node];
        }
      }
    };
  };

  template <zs::execspace_e, typename LBvhT, bool Base = false, typename = void> struct LBvhView;

  /// proxy to work within each backends
  template <zs::execspace_e space_, typename LBvhT, bool Base>
  struct LBvhView<space_, const LBvhT, Base> {
    static constexpr int dim = LBvhT::dim;
    static constexpr auto space = space_;
    using index_t = typename LBvhT::index_type;
    using bv_t = typename LBvhT::Box;
    using bvs_t = typename LBvhT::bvs_t;
    using indices_t = typename LBvhT::indices_t;

    using value_type = typename LBvhT::value_type;

    constexpr LBvhView() = default;
    ~LBvhView() = default;

    explicit constexpr LBvhView(const LBvhT &lbvh)
        : _orderedBvs{zs::view<space>(lbvh.orderedBvs, wrapv<Base>{})},
          _parents{zs::view<space>(lbvh.parents, wrapv<Base>{})},
          _levels{zs::view<space>(lbvh.levels, wrapv<Base>{})},
          _leafInds{zs::view<space>(lbvh.leafInds, wrapv<Base>{})},
          _auxIndices{zs::view<space>(lbvh.auxIndices, wrapv<Base>{})},
          _numNodes{static_cast<index_t>(lbvh.getNumNodes())} {}

    constexpr auto numNodes() const noexcept { return _numNodes; }
    constexpr auto numLeaves() const noexcept {
      return _numNodes > 2 ? (numNodes() + 1) / 2 : _numNodes;
    }

    constexpr bv_t getNodeBV(index_t node) const { return _orderedBvs[node]; }

    template <typename VecT, class F> constexpr void ray_intersect(const VecInterface<VecT> &ro,
                                                                   const VecInterface<VecT> &rd,
                                                                   F &&f) const {  // hit distance
      if (auto nl = numNodes(); nl <= 2) {
        for (index_t i = 0; i != nl; ++i) {
          if (ray_box_intersect(ro, rd, getNodeBV(i))) f(i);
        }
        return;
      }
      index_t node = 0;
      while (node != -1 && node != _numNodes) {
        index_t level = _levels[node];
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
      index_t idx = -1;
      T dist = cap;
      if (auto nl = numNodes(); nl <= 2) {
        for (index_t i = 0; i != nl; ++i) {
          if (auto d = distance(p, getNodeBV(i)); d < dist) {
            f(i, dist, idx);
          }
        }
        if constexpr (IndexRequired)
          return zs::make_tuple(idx, dist);
        else
          return dist;
      }
      index_t node = 0;
      while (node != -1 && node != _numNodes) {
        index_t level = _levels[node];
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
      index_t idx = -1;
      T dist = limits<T>::max();
      if (auto nl = numNodes(); nl <= 2) {
        for (index_t i = 0; i != nl; ++i) {
          if (auto d = distance(p, getNodeBV(i)); d < dist) {
            f(i, dist, idx);
          }
        }
        if constexpr (IndexRequired)
          return zs::make_tuple(idx, dist);
        else
          return dist;
      }
      index_t node = 0;
      while (node != -1 && node != _numNodes) {
        index_t level = _levels[node];
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
                                      typename VecT::value_type dist2 = limits<value_type>::max(),
                                      index_t idx = -1, wrapv<IndexRequired> = {}) const {
      using T = typename VecT::value_type;
      if (auto nl = numNodes(); nl <= 2) {
        for (index_t i = 0; i != nl; ++i) {
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
      index_t node = 0;
      while (node != -1 && node != _numNodes) {
        index_t level = _levels[node];
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
        for (index_t i = 0; i != nl; ++i) {
          if (overlaps(getNodeBV(i), bv)) {
            if constexpr (is_same_v<decltype(declval<F>()(declval<index_t>())), void>)
              f(i);
            else {
              if (f(i)) return;
            }
          }
        }
        return;
      }
      index_t node = 0;
      while (node != -1 && node != _numNodes) {
        index_t level = _levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (!overlaps(getNodeBV(node), bv)) break;
        // leaf node check
        if (level == 0) {
          if (overlaps(getNodeBV(node), bv)) {
            if constexpr (is_same_v<decltype(declval<F>()(declval<index_t>())), void>)
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
    /// @note self iteration
    template <class F> constexpr void self_iter_neighbors(index_t leafId, F &&f) const {
      if (auto nl = numNodes(); nl <= 2) {
        const auto bv = getNodeBV(leafId);
        for (index_t i = leafId + 1; i != nl; ++i) {
          if (overlaps(getNodeBV(i), bv)) {
            if constexpr (is_same_v<decltype(declval<F>()(declval<index_t>())), void>)
              f(i);
            else {
              if (f(i)) return;
            }
          }
        }
        return;
      }
      index_t node = _leafInds[leafId];
      const auto bv = getNodeBV(node);
      while (node != -1 && node != _numNodes) {
        index_t level = _levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (!overlaps(getNodeBV(node), bv)) break;
        // leaf node check
        if (level == 0) {
          if (overlaps(getNodeBV(node), bv)) {
            if constexpr (is_same_v<decltype(declval<F>()(declval<index_t>())), void>)
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
    /// @note iterate treelet (subtree)
    template <typename BV, class F>
    constexpr void iter_neighbors(const BV &bv, index_t node, F &&f) const {
      if (auto nl = numNodes(); nl <= 2) {
        if (overlaps(getNodeBV(node), bv)) f(node);
        return;
      }
      const auto ed = _levels[node] != 0 ? _auxIndices[node] : node + 1;
      while (node != ed && node != _numNodes) {
        index_t level = _levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (!overlaps(getNodeBV(node), bv)) break;
        // leaf node check
        if (level == 0) {
          if (overlaps(getNodeBV(node), bv)) f(_auxIndices[node]);
          node++;
        } else  // separate at internal nodes
          node = _auxIndices[node];
      }
    }
    /// @note iterate and maintain front nodes
    template <typename BV, typename Front, class F>
    ZS_FUNCTION void iter_neighbors(const BV &bv, index_t i, Front &front, F &&f) const {
      auto node = front.node(i);
      if (auto nl = numNodes(); nl <= 2) {
        if (overlaps(getNodeBV(node), bv)) f(node);
        return;
      }
      const auto primid = front.prim(i);
      bool expanded = false;
      auto record = [&]() {
        if (!expanded) {
          front.assign(i, node);
          expanded = true;
        } else {
          front.push_back(primid, node);
        }
      };
      const auto ed = _levels[node] != 0 ? _auxIndices[node] : node + 1;
      while (node != ed && node != _numNodes) {
        index_t level = _levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (!overlaps(getNodeBV(node), bv)) break;
        // leaf node check
        if (level == 0) {
          if (overlaps(getNodeBV(node), bv)) f(_auxIndices[node]);
          record();
          node++;
        } else {  // separate at internal nodes
          record();
          node = _auxIndices[node];
        }
      }
    }

    zs::VectorView<space, const bvs_t, Base> _orderedBvs;
    zs::VectorView<space, const indices_t, Base> _parents, _levels, _leafInds, _auxIndices;
    index_t _numNodes;
  };

  template <zs::execspace_e space, int dim, typename Ti, typename T, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const LBvh<dim, Ti, T, Allocator> &lbvh, wrapv<Base> = {}) {
    return LBvhView<space, const LBvh<dim, Ti, T, Allocator>, Base>{lbvh};
  }

  template <zs::execspace_e space, int dim, typename Ti, typename T, typename Allocator>
  decltype(auto) proxy(const LBvh<dim, Ti, T, Allocator> &lbvh) {
    return view<space>(lbvh, false_c);
  }

  template <typename BvhView, typename BV, class F>
  constexpr void iter_neighbors(const BvhView &bvh, const BV &bv, F &&f) {
    bvh.iter_neighbors(bv, FWD(f));
  }

  template <int dim, typename Index, typename Value, typename Allocator>
  template <typename Policy, bool Refit>
  void LBvh<dim, Index, Value, Allocator>::build(Policy &&policy,
                                                 const zs::Vector<zs::AABBBox<dim, Value>> &primBvs,
                                                 zs::wrapv<Refit>) {
    using T = value_type;
    using Ti = index_type;
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    constexpr auto execTag = wrapv<space>{};

    if (primBvs.size() == 0) return;
    const size_type numLeaves = primBvs.size();
    if (numLeaves <= 2) {  // edge cases where not enough primitives to form a tree
      orderedBvs = primBvs;
      leafInds = indices_t{primBvs.get_allocator(), numLeaves};
      for (int i = 0; i < numLeaves; ++i) leafInds.setVal(i, i);
      auxIndices = indices_t{primBvs.get_allocator(), numLeaves};
      for (int i = 0; i < numLeaves; ++i) auxIndices.setVal(i, i);
      return;
    }

    const size_type numTrunk = numLeaves - 1;
    const size_type numNodes = numLeaves * 2 - 1;

    auto allocator = get_temporary_memory_source(policy);

    Vector<index_type> trunkPars{allocator, numTrunk};
    Vector<index_type> trunkLcs{allocator, numTrunk};
    Vector<index_type> trunkRcs{allocator, numTrunk};
    Vector<index_type> trunkLs{allocator, numTrunk};
    Vector<index_type> trunkRs{allocator, numTrunk};
    Vector<index_type> trunkDst{allocator, numTrunk};
    Vector<index_type> leafPars{allocator, numLeaves};
    Vector<index_type> leafLcas{allocator, numLeaves};
    Vector<index_type> primInds{allocator, numLeaves};
    Vector<index_type> leafDepths{allocator, numLeaves + 1};
    Vector<index_type> leafOffsets{allocator, numLeaves + 1};

    orderedBvs = bvs_t{primBvs.get_allocator(), numNodes};
    auxIndices = indices_t{primBvs.get_allocator(), numNodes};
    parents = indices_t{primBvs.get_allocator(), numNodes};
    levels = indices_t{primBvs.get_allocator(), numNodes};
    leafInds = indices_t{primBvs.get_allocator(), numLeaves};

    /// views
    auto tPars = proxy<space>(trunkPars);
    auto tLcs = proxy<space>(trunkLcs);
    auto tRcs = proxy<space>(trunkRcs);
    auto tLs = proxy<space>(trunkLs);
    auto tRs = proxy<space>(trunkRs);
    auto tDst = proxy<space>(trunkDst);
    auto lPars = proxy<space>(leafPars);
    auto lLcas = proxy<space>(leafLcas);
    auto lInds = proxy<space>(leafInds);
    auto pInds = proxy<space>(primInds);
    auto lDepths = proxy<space>(leafDepths);
    auto lOffsets = proxy<space>(leafOffsets);

    // total bounding volume
    const auto defaultBox
        = Box{TV::constant(limits<value_type>::max()), TV::constant(limits<value_type>::lowest())};
    Vector<Box> wholeBox{primBvs.get_allocator(), 1};
#if 0
    wholeBox.setVal(defaultBox);
    {
      const auto &params = zs::make_tuple(proxy<space>(wholeBox), execTag);
      policy(primBvs, params, _build_compute_bounding_box{});
    }
#else
    wholeBox.setVal(compute_bounding_box(policy, range(primBvs)));
#endif

    // morton codes
    Vector<mc_t> mcs{allocator, numLeaves};
    Vector<index_type> indices{allocator, numLeaves};

    {
      const auto &params = zs::make_tuple(wholeBox.getVal(), proxy<space>(primBvs),
                                          proxy<space>(mcs), proxy<space>(indices));
      policy(range(numLeaves), params, _build_init_mc_id{});
    }

    // sort by morton codes
    Vector<mc_t> sortedMcs{allocator, numLeaves};
    Vector<index_type> sortedIndices{allocator, numLeaves};
    radix_sort_pair(policy, mcs.begin(), indices.begin(), sortedMcs.begin(), sortedIndices.begin(),
                    numLeaves);

    // build
    {
      Vector<int> trunkBuildFlags{allocator, numTrunk};
      trunkBuildFlags.reset(0);
      {
        const auto &params = zs::make_tuple(proxy<space>(sortedIndices), pInds, lDepths, numTrunk);
        policy(range(numLeaves), params, _build_init_depths{});
      }
      {
        const auto &params = zs::make_tuple(proxy<space>(sortedMcs), tPars, tLcs, tRcs, tLs, tRs,
                                            lPars, lDepths, execTag, numTrunk);
        /// ref: 2012 Tero Karras
        /// also: efficient BVH-based collision detection scheme with ordering and restructuring
        policy(range(numTrunk), params, _build_build_topo{});
      }
    }

    exclusive_scan(policy, zs::begin(leafDepths), zs::end(leafDepths), zs::begin(leafOffsets));

    lOffsets = proxy<space>(leafOffsets);

    // calc trunk order, leaf lca, levels
    {
      const auto &params
          = zs::make_tuple(proxy<space>(levels), lOffsets, lPars, lLcas, tPars, tDst, numTrunk);
      policy(range(numLeaves), params, _build_supp_topo{});
    }

    // reorder leaf
    {
      const auto &params
          = zs::make_tuple(lOffsets, lPars, proxy<space>(auxIndices), proxy<space>(parents),
                           proxy<space>(levels), pInds, lInds, tDst);
      policy(range(numLeaves), params, _build_reorder_leaf{});
    }

#if 0
    {
      fmt::print("{} leaves\n", numLeaves);
      // check trunk order & lInds
      Vector<u64> tab{primBvs.get_allocator(), numNodes}, tmp{primBvs.get_allocator(), 1};
      policy(enumerate(tab), [] ZS_LAMBDA(auto id, auto &i) { i = id + 1; });
      reduce(policy, zs::begin(tab), zs::end(tab), zs::begin(tmp), (u64)0);
      auto chkSum = tmp.getVal();
      fmt::print("{} total nodes, sum {} (ref: {})\n", numNodes, chkSum,
                 ((u64)1 + (u64)numNodes) * (u64)numNodes / 2);

      policy(enumerate(trunkDst), [tab = proxy<space>(tab)] ZS_LAMBDA(auto id, auto dst) {
        if (atomic_cas(execTag, &tab[dst], (u64)(dst + 1), (u64)0) != dst + 1)
          printf("\t%d-th trunk node (dst %d) invalid\n", (int)id, (int)dst);
      });
      policy(enumerate(leafInds), [tab = proxy<space>(tab)] ZS_LAMBDA(auto id, auto dst) {
        if (atomic_cas(execTag, &tab[dst], (u64)(dst + 1), (u64)0) != dst + 1)
          printf("\t%d-th leaf node (dst %d) invalid\n", (int)id, (int)dst);
      });
      reduce(policy, zs::begin(tab), zs::end(tab), zs::begin(tmp), (u64)0);
      chkSum = tmp.getVal();
      fmt::print("end sum {}\n", chkSum);
    }
#endif

    // reorder trunk
    {
      const auto &params = zs::make_tuple(lLcas, lOffsets, proxy<space>(auxIndices),
                                          proxy<space>(parents), tRs, tPars, tDst, numTrunk);
      policy(range(numTrunk), params, _build_reorder_trunk{});
    }

#if 0
    {
      policy(range(numTrunk), [lDepths, lOffsets, tLs, tRs, tLcs, tRcs, tPars, lPars, lLcas, tDst,
                               auxIndices = proxy<space>(auxIndices), levels = proxy<space>(levels),
                               parents = proxy<space>(parents),
                               numTrunk] ZS_LAMBDA(Ti idx) mutable {
        auto dst = tDst[idx];
        auto l = tLs[idx];
        auto r = tRs[idx];
        auto lc = tLcs[idx];
        auto rc = tRcs[idx];
        auto triggered = [&]() {
          int ids[20] = {};
          for (auto &id : ids) id = -3;
          int cnt = 0;
          int id = idx;
          int depth = 1;
          while (id < numTrunk) {
            ids[cnt++] = id;
            int lch = tLcs[id];
            id = lch;
          }
          for (auto &&i : zs::range(cnt)) {
            id = ids[i];
            int lch = tLcs[id];
            printf("tk node %d <%d (depth %d), %d> cur depth %d, level %d, lch %d\n", id, tLs[id],
                   lDepths[tLs[id]], tRs[id], depth++, levels[tDst[id]], lch);
          }
        };
        auto chkPar = [&](int ch, int par) {
          if (ch < numTrunk) {
            if (int p = tPars[ch]; p != par) {
              printf("trunk child %d <%d, %d>\' parent %d <%d, %d> not %d <%d, %d>!\n", ch, tLs[ch],
                     tRs[ch], p, tLs[p], tRs[p], par, tLs[par], tRs[par]);
              triggered();
            }
            if (parents[tDst[ch]] != tDst[par]) {
              int actPar = parents[tDst[ch]];
              int incPar = tDst[par];
              printf("ordered trunk ch %d\' parent %d not %d\n", (int)tDst[ch], actPar, incPar);
            }
          } else {
            if (int p = lPars[ch - numTrunk]; p != par) {
              printf("leaf child %d\' parent %d <%d, %d> not %d <%d, %d>!\n", ch, p, tLs[p], tRs[p],
                     par, tLs[par], tRs[par]);
              triggered();
            }
            if (parents[lOffsets[ch - numTrunk + 1] - 1] != tDst[par]) {
              int actPar = parents[lOffsets[ch - numTrunk + 1] - 1];
              int incPar = tDst[par];
              printf("ordered leaf ch %d\' parent %d not %d\n",
                     (int)(lOffsets[ch - numTrunk + 1] - 1), actPar, incPar);
            }
          }
        };
        auto getRange = [&](int ch) {
          if (ch < numTrunk)
            return zs::make_tuple(tLs[ch], tRs[ch]);
          else
            return zs::make_tuple((int)(ch - numTrunk), (int)(ch - numTrunk));
        };
        chkPar(lc, idx);
        chkPar(rc, idx);
        auto lRange = getRange(lc);
        auto rRange = getRange(rc);
        auto range = getRange(idx);
        if (idx == 0
            || !(zs::get<0>(range) == zs::get<0>(lRange) && zs::get<1>(range) == zs::get<1>(rRange)
                 && zs::get<1>(lRange) + 1 == zs::get<0>(rRange)))
          printf("<%d, %d> != <%d, %d> + <%d, %d>\n", zs::get<0>(range), zs::get<1>(range),
                 zs::get<0>(lRange), zs::get<1>(lRange), zs::get<0>(rRange), zs::get<1>(rRange));
        auto level = levels[dst];
        // lca
        Ti lcc = lc;
        for (int i = 1; i != level; ++i) {
          if (lcc >= numTrunk) printf("\t!! f**k, should not reach leaf node yet.\n");
          if (tDst[lcc] != dst + i) printf("\t!! damn, left branch mapping not consequential.\n");
          if (tLs[lcc] != zs::get<0>(range)) printf("\t!! s**t, left bound not aligned!\n");
          lcc = tLcs[lcc];
        }
        if (lcc < numTrunk) {
          printf("\t!! wtf, should just reach left node.\n");
        }
        lcc -= numTrunk;
        if (lOffsets[lcc + 1] - 1 != dst + level)
          printf("\t!! damn, left branch mapping not consequential.\n");
        if (lcc != zs::get<0>(lRange))
          printf("damn, left bottom ch %d not equals left range %d\n", (int)lcc,
                 (int)zs::get<0>(lRange));
        Ti lca = tPars[idx], node = idx;
        while (lca != -1 && tLcs[lca] == node) {
          node = lca;
          lca = tPars[node];
        }
        lca = node;
        if (lDepths[lcc] != levels[tDst[lca]] + 1) {
          printf("depth and level misalign. leaf %d, depth %d, lca level %d !\n", (int)lcc,
                 (int)lDepths[lcc], (int)(levels[tDst[lca]] + 1));
        }
        if (false) {  // l == 4960 && levels[dst] > 2
          printf("leaf %d branch, lca %d. depth %d, level %d, trunk node %d (%d ordered)\n", (int)l,
                 (int)lca, (int)lDepths[l], (int)levels[tDst[lca]], (int)idx, (int)dst);
          for (int n = lca; n < numTrunk; n = tLcs[n]) {
            auto lc_ = tLcs[n];
            printf("node %d <%d, %d>, level %d, lc %d (%d ordered)\n", n, (int)tLs[n], (int)tRs[n],
                   (int)levels[tDst[n]], (int)lc_,
                   lc_ < numTrunk ? (int)tDst[lc_] : (int)lOffsets[l + 1] - 1);
          }
        }
        if (lLcas[lcc] != lca)
          printf("lca mismatch! leaf %d lca %d (found lca %d)\n", lcc, lLcas[lcc], lca);
        // esc
        auto rDst = rc < numTrunk ? tDst[rc] : lOffsets[rc - numTrunk + 1] - 1;
        if (lc < numTrunk) {
          auto lDst = tDst[lc];
          if (auxIndices[lDst] != rDst) printf("left trunk ch escape not right!.\n");
        } else {
          auto lDst = lOffsets[lc - numTrunk + 1] - 1;
          if (lDst + 1 != rDst) printf("left leaf ch escape not right!.\n");
        }
      });
    }
#endif
    if constexpr (Refit) {
      refit(policy, primBvs);
    }
    return;
  }

  template <int dim, typename Index, typename Value, typename Allocator> template <typename Policy>
  void LBvh<dim, Index, Value, Allocator>::buildRefit(
      Policy &&policy, const zs::Vector<zs::AABBBox<dim, Value>> &primBvs) {
    using T = value_type;
    using Ti = index_type;
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    constexpr auto execTag = wrapv<space>{};

    if (primBvs.size() == 0) return;
    const size_type numLeaves = primBvs.size();
    if (numLeaves <= 2) {  // edge cases where not enough primitives to form a tree
      orderedBvs = primBvs;
      leafInds = indices_t{primBvs.get_allocator(), numLeaves};
      for (int i = 0; i < numLeaves; ++i) leafInds.setVal(i, i);
      auxIndices = indices_t{primBvs.get_allocator(), numLeaves};
      for (int i = 0; i < numLeaves; ++i) auxIndices.setVal(i, i);
      return;
    }

    const size_type numTrunk = numLeaves - 1;
    const size_type numNodes = numLeaves * 2 - 1;
    auto allocator = get_temporary_memory_source(policy);

    Vector<index_type> trunkPars{allocator, numTrunk};
    Vector<index_type> trunkLcs{allocator, numTrunk};
    Vector<index_type> trunkRcs{allocator, numTrunk};
    Vector<index_type> trunkLs{allocator, numTrunk};
    Vector<index_type> trunkRs{allocator, numTrunk};
    Vector<index_type> trunkDst{allocator, numTrunk};
    Vector<Box> trunkBvs{allocator, numTrunk};
    Vector<index_type> leafPars{allocator, numLeaves};
    Vector<index_type> leafLcas{allocator, numLeaves};
    Vector<index_type> primInds{allocator, numLeaves};
    Vector<index_type> leafDepths{allocator, numLeaves + 1};
    Vector<index_type> leafOffsets{allocator, numLeaves + 1};

    orderedBvs = bvs_t{primBvs.get_allocator(), numNodes};
    auxIndices = indices_t{primBvs.get_allocator(), numNodes};
    parents = indices_t{primBvs.get_allocator(), numNodes};
    levels = indices_t{primBvs.get_allocator(), numNodes};
    leafInds = indices_t{primBvs.get_allocator(), numLeaves};

    /// views
    auto tPars = proxy<space>(trunkPars);
    auto tLcs = proxy<space>(trunkLcs);
    auto tRcs = proxy<space>(trunkRcs);
    auto tLs = proxy<space>(trunkLs);
    auto tRs = proxy<space>(trunkRs);
    auto tDst = proxy<space>(trunkDst);
    auto tBvs = proxy<space>(trunkBvs);
    auto lPars = proxy<space>(leafPars);
    auto lLcas = proxy<space>(leafLcas);
    auto pInds = proxy<space>(primInds);
    auto lDepths = proxy<space>(leafDepths);
    auto lOffsets = proxy<space>(leafOffsets);

    auto lInds = proxy<space>(leafInds);
    auto bvs = proxy<space>(orderedBvs);

    // total bounding volume
    const auto defaultBox
        = Box{TV::constant(limits<value_type>::max()), TV::constant(limits<value_type>::lowest())};
    Vector<Box> wholeBox{primBvs.get_allocator(), 1};
    wholeBox.setVal(compute_bounding_box(policy, range(primBvs)));

    // morton codes
    Vector<mc_t> mcs{allocator, numLeaves};
    Vector<index_type> indices{allocator, numLeaves};
    {
      const auto &params = zs::make_tuple(wholeBox.getVal(), proxy<space>(primBvs),
                                          proxy<space>(mcs), proxy<space>(indices));
      policy(range(numLeaves), params, _build_init_mc_id{});
    }

    // sort by morton codes
    Vector<mc_t> sortedMcs{allocator, numLeaves};
    Vector<index_type> sortedIndices{allocator, numLeaves};
    radix_sort_pair(policy, mcs.begin(), indices.begin(), sortedMcs.begin(), sortedIndices.begin(),
                    numLeaves);

    // split metrics
    Vector<mc_t> splits{allocator, numLeaves};
    {
      const auto &params = zs::make_tuple(view<space>(sortedMcs), numLeaves, execTag);
      policy(enumerate(splits), params, _build_init_split_location{});
    }

    // build
    {
      Vector<int> trunkBuildFlags{allocator, numTrunk};
      trunkBuildFlags.reset(0);
      {
        const auto &params = zs::make_tuple(proxy<space>(sortedIndices), pInds, lDepths, numTrunk);
        policy(range(numLeaves), params, _build_init_depths{});
      }
      {
        const auto &params
            = zs::make_tuple(tPars, tLcs, tRcs, tLs, tRs, tBvs, lLcas, lDepths, lPars, numTrunk,
                             view<space>(primBvs), view<space>(sortedIndices), view<space>(splits),
                             view<space>(trunkBuildFlags), execTag);
        policy(range(numLeaves), params, _build_build_topo_and_refit{});
      }
    }

    exclusive_scan(policy, std::begin(leafDepths), std::end(leafDepths), std::begin(leafOffsets));

    lOffsets = proxy<space>(leafOffsets);

    // calc trunk order, leaf lca, levels
    {
      const auto &params
          = zs::make_tuple(proxy<space>(levels), lOffsets, lPars, lLcas, tPars, tDst, numTrunk);
      policy(range(numLeaves), params, _build_supp_topo{});
    }

    // reorder leaf
    {
      const auto &params
          = zs::make_tuple(lOffsets, lPars, proxy<space>(auxIndices), proxy<space>(parents),
                           proxy<space>(levels), pInds, lInds, tDst, view<space>(primBvs), bvs);
      policy(range(numLeaves), params, _build_reorder_leaf_with_bv{});
    }

    // reorder trunk
    {
      const auto &params
          = zs::make_tuple(lLcas, lOffsets, view<space>(auxIndices), view<space>(parents), tRs,
                           tPars, tDst, tBvs, bvs, numTrunk);
      policy(range(numTrunk), params, _build_reorder_trunk_with_bv{});
    }

    return;
  }

  template <int dim, typename Index, typename Value, typename Allocator> template <typename Policy>
  void LBvh<dim, Index, Value, Allocator>::refit(
      Policy &&policy, const zs::Vector<zs::AABBBox<dim, Value>> &primBvs) {
    using namespace zs;
    using T = value_type;
    using Ti = index_type;
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    constexpr auto execTag = wrapv<space>{};

    const size_type numLeaves = getNumLeaves();

    if (primBvs.size() != numLeaves)
      throw std::runtime_error("bvh topology changes, require rebuild!");
    if (numLeaves <= 2) {  // edge cases where not enough primitives to form a tree
      orderedBvs = primBvs;
      return;
    }
    const size_type numNodes = getNumNodes();
    // init bvs, refit flags
    auto allocator = get_temporary_memory_source(policy);
    Vector<int> refitFlags{allocator, numNodes};
    refitFlags.reset(0);
    // refit
    {
      const auto &params
          = zs::make_tuple(proxy<space>(primBvs), proxy<space>(orderedBvs),
                           proxy<space>(auxIndices), proxy<space>(leafInds), proxy<space>(parents),
                           proxy<space>(levels), proxy<space>(refitFlags), execTag);
      policy(range(numLeaves), params, _refit_bottom_up{});
    }
    return;
  }

#if ZS_ENABLE_SERIALIZATION
  template <typename S, int dim, typename Index, typename Value>
  void serialize(S &s, LBvh<dim, Index, Value, ZSPmrAllocator<>> &bvh) {
    if (!bvh.memoryLocation().onHost()) {
      bvh = bvh.clone({memsrc_e::host, -1});
    }

    serialize(s, bvh.orderedBvs);
    serialize(s, bvh.parents);
    serialize(s, bvh.levels);
    serialize(s, bvh.leafInds);
    serialize(s, bvh.auxIndices);
  }
#endif

}  // namespace zs