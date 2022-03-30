#include "zensim/container/Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"

namespace zs {

  ///
  template <execspace_e space, int dim, typename T> struct ComputeBoundingVolume {
    using BV = AABBBox<dim, T>;
    using bv_t = VectorView<space, Vector<BV>>;
    ComputeBoundingVolume() = default;
    constexpr ComputeBoundingVolume(wrapv<space>, Vector<BV>& v) noexcept : box{proxy<space>(v)} {}
    constexpr void operator()(const BV& bv) noexcept {
      for (int d = 0; d != dim; ++d) {
        atomic_min(wrapv<space>{}, &box(0)._min[d], bv._min[d]);
        atomic_max(wrapv<space>{}, &box(0)._max[d], bv._max[d]);
      }
    }

    bv_t box{};
  };

  ///
  template <typename BV> struct ComputeMortonCodes {
    ComputeMortonCodes() = default;
    constexpr ComputeMortonCodes(const BV& wholeBox) noexcept : wholeBox{wholeBox} {}

    template <typename Ti0, typename Ti1, typename CodeT>
    constexpr void operator()(Ti0 id, const BV& bv, CodeT& code, Ti1& index) {
      auto c = bv.getBoxCenter();
      auto coord = wholeBox.getUniformCoord(c);  // this is a vec<T, dim>
      static_assert(is_same_v<CodeT, decltype(morton_code<BV::dim>(coord))>,
                    "morton code assignee and producer type mismatch");
      code = morton_code<BV::dim>(coord);
      index = id;
    }

    BV wholeBox{};
  };

  ///
  template <execspace_e space, typename CodeT> struct ComputeSplitMetric {
    ComputeSplitMetric() = default;
    constexpr ComputeSplitMetric(wrapv<space>, std::size_t numLeaves, const Vector<CodeT>& mcs,
                                 u8 totalBits) noexcept
        : mcs{proxy<space>(mcs)}, numLeaves{numLeaves}, totalBits{totalBits} {}

    template <typename Ti> constexpr void operator()(Ti id, CodeT& split) {
      if (id != numLeaves - 1)
        split = totalBits - count_lz(wrapv<space>{}, mcs(id) ^ mcs(id + 1));
      else
        split = totalBits + 1;
    }

    VectorView<space, const Vector<CodeT>> mcs;
    std::size_t numLeaves;
    u8 totalBits;
  };

  ///
  struct ResetBuildStates {
    constexpr void operator()(u32& mark, int& flag) noexcept {
      mark = 0;
      flag = 0;
    }
  };

  template <execspace_e space, typename BvTiles, int dim, typename T, typename MC, typename Index>
  struct BuildRefitLBvh {
    using BV = AABBBox<dim, T>;
    using bv_t = VectorView<space, Vector<BV>>;
    using cbv_t = VectorView<space, const Vector<BV>>;
    using bvtiles_t = decltype(proxy<space>({}, std::declval<BvTiles&>()));
    using size_type = typename bvtiles_t::size_type;
    using mc_t = VectorView<space, const Vector<MC>>;
    using id_t = VectorView<space, Vector<Index>>;
    using mark_t = VectorView<space, Vector<u32>>;
    using flag_t = VectorView<space, Vector<int>>;
    BuildRefitLBvh() = default;
    constexpr BuildRefitLBvh(wrapv<space>, size_type numLeaves, const Vector<BV>& primBvs,
                             BvTiles& leafBvs, BvTiles& trunkBvs, const Vector<MC>& splits,
                             Vector<Index>& indices, Vector<Index>& leafLca,
                             Vector<Index>& leafDepths, Vector<Index>& trunkL,
                             Vector<Index>& trunkR, Vector<Index>& trunkLc, Vector<Index>& trunkRc,
                             Vector<u32>& marks, Vector<int>& flags) noexcept
        : primBvs{proxy<space>(primBvs)},
          leafBvs{proxy<space>({}, leafBvs)},
          trunkBvs{proxy<space>({}, trunkBvs)},
          splits{proxy<space>(splits)},
          indices{proxy<space>(indices)},
          leafLca{proxy<space>(leafLca)},
          leafDepths{proxy<space>(leafDepths)},
          trunkL{proxy<space>(trunkL)},
          trunkR{proxy<space>(trunkR)},
          trunkLc{proxy<space>(trunkLc)},
          trunkRc{proxy<space>(trunkRc)},
          marks{proxy<space>(marks)},
          flags{proxy<space>(flags)},
          numLeaves{numLeaves} {}

    constexpr void operator()(Index idx) noexcept {
      using TV = vec<T, dim>;
      {
        BV bv = primBvs(indices(idx));
        leafBvs.template tuple<dim>("min", idx) = bv._min;
        leafBvs.template tuple<dim>("max", idx) = bv._max;
      }

      leafLca(idx) = -1, leafDepths(idx) = 1;
      Index l = idx - 1, r = idx;  ///< (l, r]
      bool mark{false};

      if (l >= 0) mark = splits(l) < splits(r);  ///< true when right child, false otherwise

      int cur = mark ? l : r;
      if (mark)
        trunkRc(cur) = idx, trunkR(cur) = idx, atomic_or(wrapv<space>{}, &marks(cur), 0x00000002u);
      else
        trunkLc(cur) = idx, trunkL(cur) = idx, atomic_or(wrapv<space>{}, &marks(cur), 0x00000001u);

      while (atomic_add(wrapv<space>{}, &flags(cur), 1) == 1) {
        {  // refit
          int lc = trunkLc(cur), rc = trunkRc(cur);
          const auto childMask = marks(cur) & 3;
          const auto& leftBox = (childMask & 1) ? BV{leafBvs.template pack<dim>("min", lc),
                                                     leafBvs.template pack<dim>("max", lc)}
                                                : BV{trunkBvs.template pack<dim>("min", lc),
                                                     trunkBvs.template pack<dim>("max", lc)};
          const auto& rightBox = (childMask & 2) ? BV{leafBvs.template pack<dim>("min", rc),
                                                      leafBvs.template pack<dim>("max", rc)}
                                                 : BV{trunkBvs.template pack<dim>("min", rc),
                                                      trunkBvs.template pack<dim>("max", rc)};

          BV bv{};
          for (int d = 0; d != dim; ++d) {
            bv._min[d] = leftBox._min[d] < rightBox._min[d] ? leftBox._min[d] : rightBox._min[d];
            bv._max[d] = leftBox._max[d] > rightBox._max[d] ? leftBox._max[d] : rightBox._max[d];
          }
          trunkBvs.template tuple<dim>("min", cur) = bv._min;
          trunkBvs.template tuple<dim>("max", cur) = bv._max;
        }
        marks(cur) &= 0x00000007;

        l = trunkL(cur) - 1, r = trunkR(cur);
        leafLca(l + 1) = cur, leafDepths(l + 1)++;
        thread_fence(wrapv<space>{});  // this is needed

        if (l >= 0)
          mark = splits(l) < splits(r);  ///< true when right child, false otherwise
        else
          mark = false;

        if (l + 1 == 0 && r == numLeaves - 1) {
          // trunkPar(cur) = -1;
          marks(cur) &= 0xFFFFFFFB;
          break;
        }

        int par = mark ? l : r;
        // trunkPar(cur) = par;
        if (mark)
          trunkRc(par) = cur, trunkR(par) = r, atomic_and(wrapv<space>{}, &marks(par), 0xFFFFFFFD),
          marks(cur) |= 0x00000004;
        else
          trunkLc(par) = cur, trunkL(par) = l + 1,
          atomic_and(wrapv<space>{}, &marks(par), 0xFFFFFFFE), marks(cur) &= 0xFFFFFFFB;
        cur = par;
      }
    }

    cbv_t primBvs;
    bvtiles_t leafBvs, trunkBvs;
    mc_t splits;
    id_t indices, leafLca, leafDepths, trunkL, trunkR, trunkLc, trunkRc;
    mark_t marks;
    flag_t flags;
    size_type numLeaves;
  };

  ///
  template <execspace_e space, typename Index> struct ComputeTrunkOrder {
    using vector_t = Vector<Index>;
    using cid_t = VectorView<space, const vector_t>;
    using id_t = VectorView<space, vector_t>;

    ComputeTrunkOrder(wrapv<space>, const vector_t& trunklc, vector_t& trunkdst, vector_t& levels,
                      vector_t& parents)
        : trunkLc{proxy<space>(trunklc)},
          trunkDst{proxy<space>(trunkdst)},
          levels{proxy<space>(levels)},
          parents{proxy<space>(parents)} {}

    constexpr void operator()(Index node, Index level, Index offset) {
      parents(offset) = -1;
      for (; --level; node = trunkLc(node)) {
        levels(offset) = level;
        parents(offset + 1) = offset;  // setup left child's parent
        // if (offset < 20) printf("node %d level %d\n", (int)offset, (int)levels(offset));
        trunkDst(node) = offset++;
      }
    }

    cid_t trunkLc;
    id_t trunkDst, levels, parents;
  };

  ///
  template <execspace_e space, typename BV, typename IndicesT, typename BvTiles>
  struct ReorderTrunk {
    using T = typename BvTiles::value_type;
    static constexpr int dim = BV::dim;
    using Ti = typename IndicesT::value_type;
    static_assert(std::is_signed_v<Ti>, "Ti should be a signed integer");
    using bvtiles_t = decltype(proxy<space>({}, std::declval<BvTiles&>()));
    using cbvtiles_t = decltype(proxy<space>({}, std::declval<const BvTiles&>()));
    using cid_t = decltype(proxy<space>(std::declval<const IndicesT&>()));
    using id_t = decltype(proxy<space>(std::declval<IndicesT&>()));

    ReorderTrunk(wrapv<space>, wrapt<BV>, Ti numLeaves, const IndicesT& trunkDst,
                 const IndicesT& leafLca, const IndicesT& leafOffsets, const BvTiles& trunkBvs,
                 BvTiles& sortedBvs, IndicesT& escapeIndices, IndicesT& parents)
        : trunkDst{proxy<space>(trunkDst)},
          leafLca{proxy<space>(leafLca)},
          leafOffsets{proxy<space>(leafOffsets)},
          trunkBvs{proxy<space>({}, trunkBvs)},
          sortedBvs{proxy<space>({}, sortedBvs)},
          escapeIndices{proxy<space>(escapeIndices)},
          parents{proxy<space>(parents)},
          numLeaves{numLeaves} {}

    constexpr void operator()(Ti tid, Ti dst, Ti r) noexcept {
      sortedBvs.template tuple<dim>("min", dst) = trunkBvs.pack<dim>("min", tid);
      sortedBvs.template tuple<dim>("max", dst) = trunkBvs.pack<dim>("max", tid);
      const auto rb = r + 1;
      if (rb < numLeaves) {
        auto lca = leafLca(rb);  // rb must be in left-branch
        auto brother = (lca != -1 ? trunkDst(lca) : leafOffsets(rb));
        escapeIndices(dst) = brother;
        if (parents(dst) == dst - 1)   // most likely
          parents(brother) = dst - 1;  // setup right-branch brother's parent
      } else
        escapeIndices(dst) = -1;
    }

    cid_t trunkDst, leafLca, leafOffsets;
    cbvtiles_t trunkBvs;
    bvtiles_t sortedBvs;
    id_t escapeIndices, parents;
    Ti numLeaves;
  };

  template <execspace_e space, typename BV, typename IndicesT, typename BvTiles>
  struct ReorderLeafs {
    using T = typename BvTiles::value_type;
    using Ti = typename IndicesT::value_type;
    static constexpr int dim = BV::dim;
    using bvtiles_t = decltype(proxy<space>({}, std::declval<BvTiles&>()));
    using cbvtiles_t = decltype(proxy<space>({}, std::declval<const BvTiles&>()));
    using cid_t = decltype(proxy<space>(std::declval<const IndicesT&>()));
    using id_t = decltype(proxy<space>(std::declval<IndicesT&>()));

    ReorderLeafs(wrapv<space>, wrapt<BV>, Ti numLeaves, const IndicesT& sortedIndices,
                 IndicesT& primitiveIndices, IndicesT& parents, IndicesT& levels,
                 const BvTiles& leafBvs, BvTiles& sortedBvs, IndicesT& leafIndices)
        : sortedIndices{proxy<space>(sortedIndices)},
          primitiveIndices{proxy<space>(primitiveIndices)},
          parents{proxy<space>(parents)},
          levels{proxy<space>(levels)},
          leafBvs{proxy<space>({}, leafBvs)},
          sortedBvs{proxy<space>({}, sortedBvs)},
          leafIndices{proxy<space>(leafIndices)},
          numLeaves{numLeaves} {}

    constexpr void operator()(Ti idx, Ti leafOffset, Ti leafDepth) noexcept {
      const auto dst = leafOffset + leafDepth - 1;
      leafIndices(idx) = dst;
      sortedBvs.template tuple<dim>("min", dst) = leafBvs.pack<dim>("min", idx);
      sortedBvs.template tuple<dim>("max", dst) = leafBvs.pack<dim>("max", idx);
      primitiveIndices(dst) = sortedIndices(idx);
      levels(dst) = 0;
      if (leafDepth > 1) parents(dst + 1) = dst - 1;  // setup right-branch brother's parent
    }

    cid_t sortedIndices;
    id_t primitiveIndices, parents, levels;
    cbvtiles_t leafBvs;
    bvtiles_t sortedBvs;
    id_t leafIndices;
    Ti numLeaves;
  };

  ///
  /// refit
  ///
  struct ResetRefitStates {
    constexpr void operator()(int& flag) noexcept { flag = 0; }
  };

  template <execspace_e space, typename BvsT, typename IndicesT, typename BvTilesT>
  struct RefitLBvh {
    using BV = typename BvsT::value_type;
    static constexpr int dim = BV::dim;
    using Ti = typename IndicesT::value_type;
    using bvtiles_t = decltype(proxy<space>({}, std::declval<BvTilesT&>()));
    using cid_t = decltype(proxy<space>(std::declval<const IndicesT&>()));
    using cbvs_t = decltype(proxy<space>(std::declval<const BvsT&>()));
    using flags_t = decltype(proxy<space>(std::declval<Vector<int>&>()));

    RefitLBvh() = default;
    RefitLBvh(wrapv<space>, const BvsT& primBvs, const IndicesT& parents, const IndicesT& levels,
              const IndicesT& auxIndices, Vector<int>& refitFlags, BvTilesT& bvs)
        : primBvs{proxy<space>(primBvs)},
          parents{proxy<space>(parents)},
          levels{proxy<space>(levels)},
          auxIndices{proxy<space>(auxIndices)},
          flags{proxy<space>(refitFlags)},
          bvs{proxy<space>({}, bvs)} {}

    constexpr void operator()(Ti primid, Ti node) noexcept {
      {
        auto bv = primBvs(primid);
        bvs.template tuple<dim>("min", node) = bv._min;
        bvs.template tuple<dim>("max", node) = bv._max;
      }
      Ti fa = parents(node);

      while (fa != -1) {
        if (atomic_cas(wrapv<space>{}, &flags[fa], 0, 1) == 0) break;
        auto lc = fa + 1;
        auto rc = levels[lc] == 0 ? lc + 1 : auxIndices[lc];
        auto lMin = bvs.template pack<3>("min", lc);
        auto lMax = bvs.template pack<3>("max", lc);
        auto rMin = bvs.template pack<3>("min", rc);
        auto rMax = bvs.template pack<3>("max", rc);
        for (int d = 0; d != dim; ++d) {
          bvs("min", d, fa) = lMin[d] < rMin[d] ? lMin[d] : rMin[d];
          bvs("max", d, fa) = lMax[d] > rMax[d] ? lMax[d] : rMax[d];
        }
        thread_fence(wrapv<space>{});
        fa = parents[fa];
      }
    }

    cbvs_t primBvs;
    cid_t parents, levels, auxIndices;
    flags_t flags;
    bvtiles_t bvs;
  };

  template <execspace_e space, typename Collider, typename BV, typename Index, typename ResultIndex>
  struct IntersectLBvh {
    using T = typename BV::T;
    static constexpr int dim = BV::dim;
    using cbv_t = VectorView<space, const Vector<BV>>;
    using bv_t = VectorView<space, Vector<BV>>;
    using cid_t = VectorView<space, const Vector<Index>>;

    IntersectLBvh() = default;
    IntersectLBvh(wrapv<space>, const Vector<BV>& bvhBvs, const Vector<Index>& auxIndices,
                  const Vector<Index>& levels, Vector<ResultIndex>& cnt,
                  Vector<tuple<ResultIndex, ResultIndex>>& records)
        : bvhBvs{proxy<space>(bvhBvs)},
          auxIndices{proxy<space>(auxIndices)},
          levels{proxy<space>(levels)},
          cnt{proxy<space>(cnt)},
          records{proxy<space>(records)},
          numNodes{bvhBvs.size()},
          bound{records.size()} {}

    constexpr void operator()(ResultIndex cid, const Collider& collider) {
      Index node = 0;
      while (node != -1 && node != numNodes) {
        Index level = levels(node);
        // internal node traversal
        for (; level; --level, ++node)
          if (!overlaps(collider, bvhBvs(node))) break;
        // leaf node check
        if (level == 0) {
          if (overlaps(collider, bvhBvs(node))) {
            auto no = atomic_add(wrapv<space>{}, &cnt(0), (ResultIndex)1);
            /// bound check
            if (no < bound)
              records(no) = make_tuple((ResultIndex)cid, (ResultIndex)auxIndices(node));
          }
          node++;
        } else
          node = auxIndices(node);
      }
    }

    cbv_t bvhBvs;
    cid_t auxIndices, levels;
    VectorView<space, Vector<ResultIndex>> cnt;
    VectorView<space, Vector<tuple<ResultIndex, ResultIndex>>> records;
    Index numNodes, bound;
  };

}  // namespace zs