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
    ZS_FUNCTION void operator()(const BV& bv) {
      for (int d = 0; d < dim; ++d) {
        atomic_min(wrapv<space>{}, &box(0)._min[d], bv._min[d]);
        atomic_max(wrapv<space>{}, &box(0)._max[d], bv._max[d]);
      }
    }

    bv_t box{};
  };

  ///
  template <execspace_e space, typename CodeT, typename Index, typename BV>
  struct ComputeMortonCodes {
    ComputeMortonCodes() = default;
    constexpr ComputeMortonCodes(wrapv<space>, const BV& wholeBox) noexcept : wholeBox{wholeBox} {}

    constexpr void operator()(Index id, const BV& bv, CodeT& code, Index& index) {
      auto c = bv.getBoxCenter();
      auto coord = wholeBox.getUniformCoord(c);  // this is a vec<T, dim>
      code = morton_code<BV::dim>(coord);
      index = id;
#if 0
      if (id < 7 || code == 0)
        printf("%d: %e, %e, %e -> %llx\n", (int)id, coord[0], coord[1], coord[2], (long long)code);
#endif
    }

    BV wholeBox{};
  };

  ///
  template <execspace_e space, typename CodeT> struct ComputeSplitMetric {
    using mc_t = VectorView<space, const Vector<CodeT>>;
    ComputeSplitMetric() = default;
    constexpr ComputeSplitMetric(wrapv<space>, std::size_t numLeaves, const Vector<CodeT>& mcs,
                                 u8 totalBits) noexcept
        : mcs{proxy<space>(mcs)}, numLeaves{numLeaves}, totalBits{totalBits} {}

    constexpr void operator()(std::size_t id, CodeT& split) {
      if (id != numLeaves - 1)
        split = totalBits - count_lz(wrapv<space>{}, mcs(id) ^ mcs(id + 1));
      else
        split = totalBits + 1;
      // if (id < 7) printf("%d, split %lld\n", id, split);
    }

    mc_t mcs;
    std::size_t numLeaves;
    u8 totalBits;
  };

  ///
  template <execspace_e space> struct ResetBuildStates {
    using flag_t = VectorView<space, Vector<int>>;
    using mark_t = VectorView<space, Vector<u32>>;
    ResetBuildStates() = default;
    constexpr ResetBuildStates(wrapv<space>) noexcept {}

    constexpr void operator()(u32& mark, int& flag) {
      mark = 0;
      flag = 0;
    }
  };

  template <execspace_e space> ResetBuildStates(wrapv<space>) -> ResetBuildStates<space>;

  template <execspace_e space, int dim, typename T, typename MC, typename Index>
  struct BuildRefitLBvh {
    using BV = AABBBox<dim, T>;
    using bv_t = VectorView<space, Vector<BV>>;
    using cbv_t = VectorView<space, const Vector<BV>>;
    using mc_t = VectorView<space, const Vector<MC>>;
    using id_t = VectorView<space, Vector<Index>>;
    using mark_t = VectorView<space, Vector<u32>>;
    using flag_t = VectorView<space, Vector<int>>;
    BuildRefitLBvh() = default;
    constexpr BuildRefitLBvh(wrapv<space>, std::size_t numLeaves, const Vector<BV>& primBvs,
                             Vector<BV>& leafBvs, Vector<BV>& trunkBvs, const Vector<MC>& splits,
                             Vector<Index>& indices, Vector<Index>& leafLca,
                             Vector<Index>& leafDepths, Vector<Index>& trunkL,
                             Vector<Index>& trunkR, Vector<Index>& trunkLc, Vector<Index>& trunkRc,
                             Vector<u32>& marks, Vector<int>& flags) noexcept
        : primBvs{proxy<space>(primBvs)},
          leafBvs{proxy<space>(leafBvs)},
          trunkBvs{proxy<space>(trunkBvs)},
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

    ZS_FUNCTION void operator()(Index idx) {
      using TV = vec<T, dim>;
      leafBvs(idx) = primBvs(indices(idx));

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
          bv_t left{}, right{};
          switch (marks(cur) & 3) {
            case 0:
              left = trunkBvs, right = trunkBvs;
              break;
            case 1:
              left = leafBvs, right = trunkBvs;
              break;
            case 2:
              left = trunkBvs, right = leafBvs;
              break;
            case 3:
              left = leafBvs, right = leafBvs;
              break;
          }
          const BV& leftBox = left(lc);
          const BV& rightBox = right(rc);
          BV bv{/*TV::uniform(std::numeric_limits<T>().max()),
                TV::uniform(std::numeric_limits<T>().lowest())*/};
          for (int d = 0; d < dim; ++d) {
            bv._min[d] = leftBox._min[d] < rightBox._min[d] ? leftBox._min[d] : rightBox._min[d];
            bv._max[d] = leftBox._max[d] > rightBox._max[d] ? leftBox._max[d] : rightBox._max[d];
          }
          trunkBvs(cur) = bv;
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
    bv_t leafBvs, trunkBvs;
    mc_t splits;
    id_t indices, leafLca, leafDepths, trunkL, trunkR, trunkLc, trunkRc;
    mark_t marks;
    flag_t flags;
    std::size_t numLeaves;
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
  template <execspace_e space, typename BV, typename Index> struct ReorderTrunk {
    using vector_t = Vector<Index>;
    using T = typename BV::T;
    static constexpr int dim = BV::dim;
    using bv_t = VectorView<space, Vector<BV>>;
    using cid_t = VectorView<space, const vector_t>;
    using id_t = VectorView<space, vector_t>;

    ReorderTrunk(wrapv<space>, std::size_t numLeaves, const vector_t& trunkDst,
                 const vector_t& leafLca, const vector_t& leafOffsets, Vector<BV>& sortedBvs,
                 vector_t& escapeIndices, vector_t& parents)
        : trunkDst{proxy<space>(trunkDst)},
          leafLca{proxy<space>(leafLca)},
          leafOffsets{proxy<space>(leafOffsets)},
          sortedBvs{proxy<space>(sortedBvs)},
          escapeIndices{proxy<space>(escapeIndices)},
          parents{proxy<space>(parents)},
          numLeaves{static_cast<Index>(numLeaves)} {}

    constexpr void operator()(Index dst, const BV& bv, Index l, Index r) {
      sortedBvs(dst) = bv;
      const auto rb = r + 1;
      if (rb < numLeaves) {
        auto lca = leafLca(rb);  // rb must be in left-branch
        auto brother = (lca != -1 ? trunkDst(lca) : leafOffsets(rb));
        escapeIndices(dst) = brother;
        if (dst > 0 && parents(dst) == dst - 1)  // most likely
          parents(brother) = dst - 1;            // setup right-branch brother's parent
#if 0
        if (dst < 20 || escapeIndices(dst) >= numLeaves * 2 - 1)
          printf("numnodes %d | trunk %d lb on leaf %d (- %d), esc %d, lca[%c] %d, bro %d\n",
                 (int)numLeaves * 2 - 1, (int)dst, (int)l, (int)r, (int)escapeIndices(dst),
                 lca != -1 ? 'T' : 'L', (int)lca, (int)brother);
#endif
      } else
        escapeIndices(dst) = -1;
    }

    cid_t trunkDst, leafLca, leafOffsets;
    bv_t sortedBvs;
    id_t escapeIndices, parents;
    Index numLeaves;
  };

  template <execspace_e space, int dim, typename T, typename Index> struct ReorderLeafs {
    using vector_t = Vector<Index>;
    using BV = AABBBox<dim, T>;
    using cbv_t = VectorView<space, const Vector<BV>>;
    using bv_t = VectorView<space, Vector<BV>>;
    using cid_t = VectorView<space, const vector_t>;
    using id_t = VectorView<space, vector_t>;

    ReorderLeafs(wrapv<space>, std::size_t numLeaves, const vector_t& sortedIndices,
                 vector_t& primitiveIndices, vector_t& parents, vector_t& levels,
                 Vector<BV>& sortedBvs, vector_t& leafIndices)
        : sortedIndices{proxy<space>(sortedIndices)},
          primitiveIndices{proxy<space>(primitiveIndices)},
          parents{proxy<space>(parents)},
          levels{proxy<space>(levels)},
          sortedBvs{proxy<space>(sortedBvs)},
          leafIndices{proxy<space>(leafIndices)},
          numLeaves{static_cast<Index>(numLeaves)} {}

    constexpr void operator()(Index idx, const BV& bv, Index leafOffset, Index leafDepth) {
      const auto dst = leafOffset + leafDepth - 1;
      leafIndices(idx) = dst;
      sortedBvs(dst) = bv;
      primitiveIndices(dst) = sortedIndices(idx);
      levels(dst) = 0;
      if (leafDepth > 1) parents(dst + 1) = dst - 1;  // setup right-branch brother's parent
#if 0
      if (dst < 20)
        printf("%d-th leaf %d, prim index %d\n", (int)idx, (int)dst, (int)sortedIndices(idx));
#endif
    }

    cid_t sortedIndices;
    id_t primitiveIndices, parents, levels;
    bv_t sortedBvs;
    id_t leafIndices;
    Index numLeaves;
  };

  ///
  /// refit
  ///
  template <execspace_e space, int dim, typename T> struct ResetRefitStates {
    using BV = AABBBox<dim, T>;
    using TV = vec<T, dim>;
    ResetRefitStates() = default;
    constexpr ResetRefitStates(wrapv<space>) noexcept {}

    constexpr void operator()(int& flag, BV& bv) {
      flag = 0;
      bv = BV{TV::uniform(std::numeric_limits<T>().max()),
              TV::uniform(std::numeric_limits<T>().lowest())};
    }
  };

  template <execspace_e space, int dim, typename T, typename Index> struct RefitLBvh {
    using BV = AABBBox<dim, T>;
    using cbv_t = VectorView<space, const Vector<BV>>;
    using bv_t = VectorView<space, Vector<BV>>;
    using cid_t = VectorView<space, const Vector<Index>>;

    RefitLBvh() = default;
    RefitLBvh(wrapv<space>, const Vector<BV>& primBvs, const Vector<Index>& primitiveIndices,
              const Vector<Index>& parents, Vector<int>& refitFlags, Vector<BV>& bvs)
        : primBvs{proxy<space>(primBvs)},
          primitiveIndices{proxy<space>(primitiveIndices)},
          parents{proxy<space>(parents)},
          flags{proxy<space>(refitFlags)},
          bvs{proxy<space>(bvs)} {}

    ZS_FUNCTION void operator()(Index node) {
      bvs(node) = primBvs(primitiveIndices(node));
      Index fa = parents(node);

      while (fa != -1) {
        BV& bv = bvs(fa);
        const BV box = bvs(node);
        for (int d = 0; d < dim; ++d) {
          atomic_min(wrapv<space>{}, &bv._min[d], box._min[d]);
          atomic_max(wrapv<space>{}, &bv._max[d], box._max[d]);
        }
        if (atomic_add(wrapv<space>{}, &flags(fa), 1) != 1) break;
        node = fa;
        fa = parents(node);
      }
    }

    cbv_t primBvs;
    cid_t primitiveIndices, parents;
    VectorView<space, Vector<int>> flags;
    bv_t bvs;
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

    ZS_FUNCTION void operator()(ResultIndex cid, const Collider& collider) {
      Index node = 0;
      while (node != -1 && node != numNodes) {
        Index level = levels(node);
        // internal node traversal
        for (; level && overlaps(collider, bvhBvs(node)); --level, ++node)
          ;
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