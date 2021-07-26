#pragma once

#include "BvhImpl.tpp"
#include "TileVector.hpp"
#include "Vector.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"

namespace zs {

  template <int dim_ = 3, int lane_width_ = 32, bool is_double = false> struct LBvh {
    static constexpr int dim = dim_;
    static constexpr int lane_width = lane_width_;
    using value_type = conditional_t<is_double, dat64, dat32>;
    using float_type = remove_cvref_t<decltype(std::declval<value_type>().asFloat())>;
    using integer_type = remove_cvref_t<decltype(std::declval<value_type>().asSignedInteger())>;
    // must be signed integer, since we are using -1 as sentinel value
    using index_type = conditional_t<is_double, i64, i32>;
    using Box = AABBBox<dim, float_type>;
    using TV = vec<float_type, dim>;
    using IV = vec<integer_type, dim>;
    using vector_t = Vector<value_type>;
    using indices_t = Vector<index_type>;
    using bvs_t = Vector<Box>;
    using tilevector_t = TileVector<value_type, lane_width>;

    LBvh() = default;

    constexpr auto numNodes() const noexcept { return auxIndices.size(); }
    constexpr auto numLeaves() const noexcept { return (numNodes() + 1) / 2; }

    Box wholeBox{TV::uniform(std::numeric_limits<float>().max()),
                 TV::uniform(std::numeric_limits<float>().min())};

    bvs_t sortedBvs;  // bounding volumes
    // escape index for internal nodes, primitive index for leaf nodes
    indices_t auxIndices;
    indices_t levels;   // count from bottom up (0-based) in left branch
    indices_t parents;  // parent

    indices_t leafIndices;  // leaf indices within optimized lbvh
  };

  template <execspace_e, typename LBvhT, typename = void> struct LBvhView;

  /// proxy to work within each backends
  template <execspace_e space, typename LBvhT> struct LBvhView<space, const LBvhT> {
    static constexpr int dim = LBvhT::dim;
    static constexpr auto exectag = wrapv<space>{};
    using Tn = typename LBvhT::float_type;
    using index_t = typename LBvhT::integer_type;
    using bv_t = typename LBvhT::Box;
    using bvs_t = typename LBvhT::bvs_t;
    using indices_t = typename LBvhT::indices_t;

    constexpr LBvhView() = default;
    ~LBvhView() = default;

    explicit constexpr LBvhView(const LBvhT &lbvh)
        : _sortedBvs{proxy<space>(lbvh.sortedBvs)},
          _auxIndices{proxy<space>(lbvh.auxIndices)},
          _levels{proxy<space>(lbvh.levels)},
          _parents{proxy<space>(lbvh.parents)},
          _leafIndices{proxy<space>(lbvh.leafIndices)},
          _numNodes{static_cast<index_t>(lbvh.numNodes())} {}

    constexpr auto numNodes() const noexcept { return _numNodes; }
    constexpr auto numLeaves() const noexcept { return (numNodes() + 1) / 2; }

    VectorView<space, const bvs_t> _sortedBvs;
    VectorView<space, const indices_t> _auxIndices, _levels, _parents, _leafIndices;
    index_t _numNodes;
  };

  template <execspace_e space, int dim, int lane_width, bool is_double>
  constexpr decltype(auto) proxy(const LBvh<dim, lane_width, is_double> &lbvh) {
    return LBvhView<space, const LBvh<dim, lane_width, is_double>>{lbvh};
  }

  /// build bvh
  template <execspace_e space, int lane_width, int dim, typename T>
  inline auto build_lbvh(const Vector<AABBBox<dim, T>> &primBvs) {
    using namespace zs;
    static constexpr bool is_double = is_same_v<T, double>;
    using mc_t = conditional_t<is_double, u64, u32>;
    using lbvh_t = LBvh<dim, lane_width, is_double>;
    using float_type = typename lbvh_t::float_type;
    using index_type = typename lbvh_t::index_type;
    using Box = typename lbvh_t::Box;
    using TV = vec<float_type, dim>;
    static_assert(is_same_v<float_type, typename Box::T>, "float type mismatch");

    lbvh_t lbvh{};

    const auto numLeaves = primBvs.size();
    const memsrc_e memdst{primBvs.memspace()};
    const ProcID devid{primBvs.devid()};
    const auto execTag = wrapv<space>{};

    auto execPol = par_exec(execTag).sync(true);
    if constexpr (space == execspace_e::cuda) execPol = execPol.device(devid);

    {
      /// total bounding volume
      Vector<Box> tmp{1, memsrc_e::host, -1};
      tmp[0] = Box{TV::uniform(std::numeric_limits<float_type>().max()),
                   TV::uniform(std::numeric_limits<float_type>().lowest())};
      Vector<Box> wholeBox = tmp.clone({memdst, devid});
      execPol(primBvs, ComputeBoundingVolume{execTag, wholeBox});
      lbvh.wholeBox = wholeBox.clone({memsrc_e::host, -1})[0];
    }

    if constexpr (false) {
      auto &wholeBox = lbvh.wholeBox;
      fmt::print("fin {} leaves, {}, {}, {} - {}, {}, {}\n", numLeaves, wholeBox._min[0],
                 wholeBox._min[1], wholeBox._min[2], wholeBox._max[0], wholeBox._max[1],
                 wholeBox._max[2]);
    }

    // morton codes
    Vector<mc_t> mcs{numLeaves, memdst, devid};
    Vector<index_type> indices{numLeaves, memdst, devid};
    execPol(enumerate(primBvs, mcs, indices),
            ComputeMortonCodes<space, mc_t, index_type, Box>{execTag, lbvh.wholeBox});
    // puts("done mcs compute");

    // sort by morton codes
    Vector<mc_t> sortedMcs{numLeaves, memdst, devid};
    Vector<index_type> sortedIndices{numLeaves, memdst, devid};
    radix_sort_pair(execPol, mcs.begin(), indices.begin(), sortedMcs.begin(), sortedIndices.begin(),
                    mcs.size());

    // for (int id = 0; id < 5; ++id) fmt::print("[{}]: {}\t", sortedIndices[id], sortedMcs[id]);
    // puts("done sort");

    Vector<mc_t> splits{numLeaves, memdst, devid};
    constexpr auto totalBits = sizeof(mc_t) * 8;
    execPol(enumerate(splits), ComputeSplitMetric{execTag, numLeaves, sortedMcs, totalBits});

    // puts("done calc splits");

    Vector<Box> leafBvs{numLeaves, memdst, devid}, trunkBvs{numLeaves - 1, memdst, devid};
    Vector<index_type> leafLca{numLeaves, memdst, devid};
    // Vector<index_type> leafPar{numLeaves, memdst, devid};
    Vector<index_type> leafDepths{numLeaves, memdst, devid};
    // Vector<index_type> trunkPar{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkR{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkL{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkRc{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkLc{numLeaves - 1, memdst, devid};

    /// build + refit
    Vector<u32> trunkTopoMarks{numLeaves - 1, memdst, devid};
    {
      Vector<int> trunkBuildFlags{numLeaves - 1, memdst, devid};
      execPol(zip(trunkTopoMarks, trunkBuildFlags), ResetBuildStates{execTag});

      execPol(range(numLeaves),
              BuildRefitLBvh{execTag, numLeaves, primBvs, leafBvs, trunkBvs, splits, sortedIndices,
                             leafLca, leafDepths, trunkL, trunkR, trunkLc, trunkRc, trunkTopoMarks,
                             trunkBuildFlags});
    }

    // puts("done init build");
    ///
    auto &sortedBvs = lbvh.sortedBvs;
    auto &auxIndices = lbvh.auxIndices;
    auto &levels = lbvh.levels;
    auto &parents = lbvh.parents;
    auto &leafIndices = lbvh.leafIndices;

    const auto numNodes = numLeaves + numLeaves - 1;
    sortedBvs = Vector<Box>{numNodes, memdst, devid};
    auxIndices = Vector<index_type>{numNodes, memdst, devid};
    levels = Vector<index_type>{numNodes, memdst, devid};
    parents = Vector<index_type>{numNodes, memdst, devid};

    leafIndices = Vector<index_type>{numLeaves, memdst, devid};  // for refit

    /// sort bvh
    Vector<index_type> leafOffsets{numLeaves, memdst, devid};
    exclusive_scan(execPol, leafDepths.begin(), leafDepths.end(), leafOffsets.begin());
    Vector<index_type> trunkDst{numLeaves - 1, memdst, devid};

    execPol(zip(leafLca, leafDepths, leafOffsets),
            ComputeTrunkOrder{execTag, trunkLc, trunkDst, levels, parents});
    execPol(zip(trunkDst, trunkBvs, trunkL, trunkR),
            ReorderTrunk{execTag, numLeaves, trunkDst, leafLca, leafOffsets, sortedBvs, auxIndices,
                         parents});
    execPol(enumerate(leafBvs, leafOffsets, leafDepths),
            ReorderLeafs{execTag, numLeaves, sortedIndices, auxIndices, parents, levels, sortedBvs,
                         leafIndices});
    // puts("done reorder");

    return lbvh;
  }

  /// refit bvh
  template <execspace_e space, int dim, int lane_width, bool is_double, typename T>
  inline void refit_lbvh(LBvh<dim, lane_width, is_double> &lbvh,
                         const Vector<AABBBox<dim, T>> &primBvs) {
    using namespace zs;
    using lbvh_t = LBvh<dim, lane_width, is_double>;
    using float_type = typename lbvh_t::float_type;
    using Box = typename lbvh_t::Box;
    using TV = vec<float_type, dim>;

    const auto numNodes = lbvh.numNodes();
    const auto memdst = lbvh.sortedBvs.memspace();
    const auto devid = lbvh.sortedBvs.devid();
    constexpr auto execTag = wrapv<space>{};

    auto execPol = par_exec(execTag).sync(true);
    if constexpr (space == execspace_e::cuda) execPol = execPol.device(devid);
    Vector<int> refitFlags{numNodes, memdst, devid};
    auto &leafIndices = lbvh.leafIndices;
    auto &sortedBvs = lbvh.sortedBvs;
    auto &auxIndices = lbvh.auxIndices;
    auto &parents = lbvh.parents;

    // puts("refit begin");
    // init bvs, refit flags
    execPol(zip(refitFlags, sortedBvs), ResetRefitStates<space, dim, typename Box::T>{execTag});
    // puts("done reset refit states");
    // refit
    execPol(leafIndices, RefitLBvh{execTag, primBvs, auxIndices, parents, refitFlags, sortedBvs});
    // puts("done refit");
    if constexpr (false) {
      auto wholeBox = sortedBvs[0];
      fmt::print("fin {} leaves, {}, {}, {} - {}, {}, {}\n", lbvh.numLeaves(), wholeBox._min[0],
                 wholeBox._min[1], wholeBox._min[2], wholeBox._max[0], wholeBox._max[1],
                 wholeBox._max[2]);
    }
  }

  /// collision detection traversal
  // collider (aabb, point) - bvh
  template <execspace_e space, typename Index, int dim, int lane_width, bool is_double,
            typename Collider>
  inline Vector<tuple<Index, Index>> intersect_lbvh(LBvh<dim, lane_width, is_double> &lbvh,
                                                    const Vector<Collider> &colliders) {
    using namespace zs;
    using lbvh_t = LBvh<dim, lane_width, is_double>;
    using float_type = typename lbvh_t::float_type;
    using TV = vec<float_type, dim>;

    const auto memdst = lbvh.sortedBvs.memspace();
    const auto devid = lbvh.sortedBvs.devid();

    std::size_t evaluatedSize = colliders.size() * dim * dim * 10;
    Vector<tuple<Index, Index>> records{evaluatedSize, memdst,
                                        devid};  // this is the estimated count
    Vector<Index> tmp{1, memsrc_e::host, -1};
    tmp[0] = 0;
    Vector<Index> cnt = tmp.clone({memdst, devid});

    auto execPol = par_exec(wrapv<space>{}).sync(true);
    if constexpr (space == execspace_e::cuda) execPol = execPol.device(devid);

    auto &sortedBvs = lbvh.sortedBvs;
    auto &auxIndices = lbvh.auxIndices;
    auto &levels = lbvh.levels;
    // init bvs, refit flags
    execPol(
        enumerate(colliders),
        IntersectLBvh<space, Collider, typename lbvh_t::Box, typename lbvh_t::index_type, Index>{
            wrapv<space>{}, sortedBvs, auxIndices, levels, cnt, records});
    auto n = cnt.clone({memsrc_e::host, -1});
    if (n[0] >= records.size())
      throw std::runtime_error(fmt::format(
          "not enough space reserved for collision indices, requires {} at least, currently {}\n",
          n[0], evaluatedSize));
    records.resize(n[0]);
    return records;
  }

}  // namespace zs
