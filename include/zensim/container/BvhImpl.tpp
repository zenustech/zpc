#include "zensim/container/Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"

namespace zs {

  ///
  template <execspace_e space, int dim, typename T> struct ComputeBoundingVolume {
    using BV = AABBBox<dim, T>;
    using bv_t = VectorProxy<space, Vector<BV>>;
    ComputeBoundingVolume() = default;
    constexpr ComputeBoundingVolume(wrapv<space>, Vector<BV>& v) noexcept : box{proxy<space>(v)} {}
    constexpr void operator()(const BV& bv) {
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
    }

    BV wholeBox{};
  };

  ///
  template <execspace_e space, typename CodeT> struct ComputeSplitMetric {
    using mc_t = VectorProxy<space, const Vector<CodeT>>;
    ComputeSplitMetric() = default;
    constexpr ComputeSplitMetric(wrapv<space>, std::size_t numLeaves, const Vector<CodeT>& mcs,
                                 u8 totalBits) noexcept
        : mcs{proxy<space>(mcs)}, numLeaves{numLeaves}, totalBits{totalBits} {}

    constexpr void operator()(std::size_t id, CodeT& split) {
      if (id != numLeaves - 1)
        split = totalBits - count_lz(wrapv<space>{}, mcs(id) ^ mcs(id + 1));
      else
        split = totalBits + 1;
    }

    mc_t mcs;
    std::size_t numLeaves;
    u8 totalBits;
  };

  ///
  template <execspace_e space> struct ResetBuildStates {
    using flag_t = VectorProxy<space, Vector<int>>;
    using mark_t = VectorProxy<space, Vector<u32>>;
    ResetBuildStates() = default;
    constexpr ResetBuildStates(wrapv<space>) noexcept {}

    constexpr void operator()(std::size_t trunkId, u32& mark, int& flag) {
      mark = 0;
      flag = 0;
    }
  };

  template <execspace_e space> ResetBuildStates(wrapv<space>) -> ResetBuildStates<space>;

  template <execspace_e space, int dim, typename T, typename MC, typename Index>
  struct BuildRefitLBvh {
    using BV = AABBBox<dim, T>;
    using bv_t = VectorProxy<space, Vector<BV>>;
    using cbv_t = VectorProxy<space, const Vector<BV>>;
    using mc_t = VectorProxy<space, const Vector<MC>>;
    using id_t = VectorProxy<space, Vector<Index>>;
    using mark_t = VectorProxy<space, Vector<u32>>;
    using flag_t = VectorProxy<space, Vector<int>>;
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

    constexpr void operator()(Index idx) {
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
    using cid_t = VectorProxy<space, const vector_t>;
    using id_t = VectorProxy<space, vector_t>;

    ComputeTrunkOrder(wrapv<space>, const vector_t& trunklc, vector_t& trunkdst, vector_t& depths)
        : trunkLc{proxy<space>(trunklc)},
          trunkDst{proxy<space>(trunkdst)},
          depths{proxy<space>(depths)} {}

    constexpr void operator()(Index idx, Index node, Index level, Index offset) {
      for (Index depth = 0; --level; node = trunkLc(node)) {
        trunkDst(node) = offset++;
        depths(node) = depth++;
      }
    }

    cid_t trunkLc;
    id_t trunkDst, depths;
  };

  ///
  template <execspace_e space, typename BV, typename Index> struct ReorderTrunk {
    using vector_t = Vector<Index>;
    using T = typename BV::T;
    static constexpr int dim = BV::dim;
    using bv_t = VectorProxy<space, Vector<BV>>;
    using cid_t = VectorProxy<space, const vector_t>;
    using id_t = VectorProxy<space, vector_t>;

    ReorderTrunk(wrapv<space>, std::size_t numLeaves, const vector_t& trunkDst,
                 const vector_t& depths, const vector_t& leafLca, const vector_t& leafDepths,
                 const vector_t& leafOffsets, Vector<BV>& sortedBvs, vector_t& escapeIndices,
                 vector_t& levels)
        : trunkDst{proxy<space>(trunkDst)},
          depths{proxy<space>(depths)},
          leafLca{proxy<space>(leafLca)},
          leafDepths{proxy<space>(leafDepths)},
          leafOffsets{proxy<space>(leafOffsets)},
          sortedBvs{proxy<space>(sortedBvs)},
          escapeIndices{proxy<space>(escapeIndices)},
          levels{proxy<space>(levels)},
          numLeaves{numLeaves} {}

    constexpr void operator()(Index dst, const BV& bv, Index l, Index r) {
      const auto depth = depths(dst);
      sortedBvs(dst) = bv;
      const auto rb = r + 1;
      if (rb < numLeaves) {
        auto lca = leafLca(rb);
        escapeIndices(dst) = (lca != -1 ? trunkDst(lca) : leafOffsets(rb));
      } else
        escapeIndices(dst) = -1;
      levels(dst) = leafDepths(l) - depth - 1;
    }

    cid_t trunkDst, depths, leafLca, leafDepths, leafOffsets;
    bv_t sortedBvs;
    id_t escapeIndices, levels;
    Index numLeaves;
  };

  template <execspace_e space, int dim, typename T, typename Index> struct ReorderLeafs {
    using vector_t = Vector<Index>;
    using BV = AABBBox<dim, T>;
    using cbv_t = VectorProxy<space, const Vector<BV>>;
    using bv_t = VectorProxy<space, Vector<BV>>;
    using cid_t = VectorProxy<space, const vector_t>;
    using id_t = VectorProxy<space, vector_t>;

    ReorderLeafs(wrapv<space>, std::size_t numLeaves, const vector_t& sortedIndices,
                 vector_t& primitiveIndices, vector_t& levels, vector_t& depths,
                 Vector<BV>& sortedBvs, vector_t& leafIndices)
        : sortedIndices{proxy<space>(sortedIndices)},
          primitiveIndices{primitiveIndices},
          levels{proxy<space>(levels)},
          depths{proxy<space>(depths)},
          leafIndices{proxy<space>(leafIndices)},
          sortedBvs{proxy<space>(sortedBvs)},
          numLeaves{numLeaves} {}

    constexpr void operator()(Index idx, const BV& bv, Index leafOffset, Index leafDepth) {
      const auto dst = leafOffset + leafDepth - 1;
      leafIndices(idx) = dst;
      sortedBvs(dst) = bv;
      primitiveIndices(dst) = sortedIndices(idx);
      levels(dst) = 0;
      depths(dst) = leafDepth - 1;
    }

    cid_t sortedIndices;
    id_t primitiveIndices, levels, depths;
    id_t leafIndices;
    bv_t sortedBvs;
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
    using cbv_t = VectorProxy<space, const Vector<BV>>;
    using bv_t = VectorProxy<space, Vector<BV>>;
    using cid_t = VectorProxy<space, const Vector<Index>>;

    RefitLBvh() = default;
    RefitLBvh(wrapv<space>, const Vector<BV>& primBvs, const Vector<Index>& primitiveIndices,
              const Vector<Index>& depths, Vector<int>& refitFlags, Vector<BV>& bvs)
        : primBvs{proxy<space>(primBvs)},
          primitiveIndices{proxy<space>(primitiveIndices)},
          depths{proxy<space>(depths)},
          flags{proxy<space>(refitFlags)},
          bvs{proxy<space>(bvs)} {}

    constexpr void operator()(Index node) {
      bvs(node) = primBvs(primitiveIndices(node));
      Index depth = depths(node);
      bool isLc = depth != 0;
      Index fa = isLc ? node - 1 : node - 1 - depths(node - 1);

      while (fa != -1)
        if (atomic_add(wrapv<space>{}, &flags(fa), 1) == 1) {
          BV& bv = bvs(fa);
          const BV box = bvs(node);
          const BV otherBox = bvs(isLc ? node + depth + 1 : node - depth);
          for (int d = 0; d < dim; ++d) {
            bv._min[d] = box._min[d] < otherBox._min[d] ? box._min[d] : otherBox._min[d];
            bv._max[d] = box._max[d] > otherBox._max[d] ? box._max[d] : otherBox._max[d];
          }
          // update fa
          node = fa;
          depth = depths(node);
          isLc = depth != 0 || node == 0;
          fa = isLc ? node - 1 : node - 1 - depths(node - 1);
        }
    }

    cbv_t primBvs;
    cid_t primitiveIndices, depths;
    VectorProxy<space, Vector<int>> flags;
    bv_t bvs;
  };

  template <execspace_e space, typename Collider, typename BV, typename Index>
  struct IntersectLBvh {
    using T = typename BV::T;
    static constexpr int dim = BV::dim;
    using cbv_t = VectorProxy<space, const Vector<BV>>;
    using bv_t = VectorProxy<space, Vector<BV>>;
    using cid_t = VectorProxy<space, const Vector<Index>>;

    IntersectLBvh() = default;
    IntersectLBvh(wrapv<space>, const Index bound, const Vector<BV>& bvhBvs,
                  const Vector<Index>& auxIndices, const Vector<Index>& levels, Vector<Index>& cnt,
                  Vector<tuple<Index, Index>>& records)
        : bvhBvs{proxy<space>(bvhBvs)},
          auxIndices{proxy<space>(auxIndices)},
          levels{proxy<space>(levels)},
          cnt{proxy<space>(cnt)},
          records{proxy<space>(records)},
          bound{bound} {}

    constexpr void operator()(Index cid, const Collider& collider) {
      Index node = 0;
      while (node != -1) {
        Index level = levels(node);
        // internal node traversal
        for (; level && overlaps(collider, bvhBvs(node)); --level, ++node)
          ;
        // leaf node check
        if (level == 0) {
          if (overlaps(collider, bvhBvs(node))) {
            auto no = atomic_add(wrapv<space>{}, &cnt(0), (Index)1);
            /// safe check
            if (no < bound) records(no) = make_tuple(cid, auxIndices(node));
          }
          ++node;
        } else
          node = auxIndices(node);
      }
    }

    cbv_t bvhBvs;
    cid_t auxIndices, levels;
    VectorProxy<space, Vector<Index>> cnt;
    VectorProxy<space, Vector<tuple<Index, Index>>> records;
    Index bound;
  };

}  // namespace zs