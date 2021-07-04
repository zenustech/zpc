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
    using TV = vec<float_type, dim>;
    using IV = vec<integer_type, dim>;
    using vector_t = Vector<value_type>;
    using tilevector_t = TileVector<value_type, lane_width>;
    using Box = AABBBox<dim, float_type>;

    LBvh() = default;

    constexpr auto numNodes() const noexcept { return auxIndices.size(); }
    constexpr auto numLeaves() const noexcept { return (numNodes() + 1) / 2; }

    Box wholeBox{TV::uniform(std::numeric_limits<float>().max()),
                 TV::uniform(std::numeric_limits<float>().min())};

    Vector<Box> sortedBvs;  // bounding volumes
    // escape index for internal nodes, primitive index for leaf nodes
    Vector<index_type> auxIndices;
    Vector<index_type> levels;  // count from bottom up (0-based) in left branch
    Vector<index_type> depths;  // count from top down (0-based) in left branch

    Vector<index_type> leafIndices;  // leaf indices within optimized lbvh
  };

  /// build bvh
  template <execspace_e space, int lane_width, int dim, typename T>
  auto build_lbvh(const Vector<AABBBox<dim, T>> &primBvs) {
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
    constexpr auto execTag = wrapv<space>{};

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
      fmt::print("{} leaves, {}, {}, {} - {}, {}, {}\n", numLeaves, wholeBox._min[0],
                 wholeBox._min[1], wholeBox._min[2], wholeBox._max[0], wholeBox._max[1],
                 wholeBox._max[2]);
      getchar();
    }

    // morton codes
    Vector<mc_t> mcs{numLeaves, memdst, devid};
    Vector<index_type> indices{numLeaves, memdst, devid};
    execPol(enumerate(primBvs, mcs, indices),
            ComputeMortonCodes<space, mc_t, index_type, Box>{execTag, lbvh.wholeBox});

    // sort by morton codes
    Vector<mc_t> sortedMcs{numLeaves, memdst, devid};
    Vector<index_type> sortedIndices{numLeaves, memdst, devid};
    radix_sort_pair(execPol, mcs.begin(), indices.begin(), sortedMcs.begin(), sortedIndices.begin(),
                    mcs.size());

    Vector<mc_t> splits{numLeaves, memdst, devid};
    constexpr auto totalBits = sizeof(mc_t) * 8;
    execPol(enumerate(splits), ComputeSplitMetric{execTag, numLeaves, sortedMcs, totalBits});

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
      execPol(enumerate(trunkTopoMarks, trunkBuildFlags), ResetBuildStates{execTag});

      execPol(range(numLeaves),
              BuildRefitLBvh{execTag, numLeaves, primBvs, leafBvs, trunkBvs, splits, sortedIndices,
                             leafLca, leafDepths, trunkL, trunkR, trunkLc, trunkRc, trunkTopoMarks,
                             trunkBuildFlags});
      ///
      auto &sortedBvs = lbvh.sortedBvs;
      auto &auxIndices = lbvh.auxIndices;
      auto &levels = lbvh.levels;
      auto &depths = lbvh.depths;
      auto &leafIndices = lbvh.leafIndices;

      const auto numNodes = numLeaves + numLeaves - 1;
      sortedBvs = Vector<Box>{numNodes, memdst, devid};
      auxIndices = Vector<index_type>{numNodes, memdst, devid};
      levels = Vector<index_type>{numNodes, memdst, devid};
      depths = Vector<index_type>{numNodes, memdst, devid};

      leafIndices = Vector<index_type>{numLeaves, memdst, devid};  // for refit

      /// sort bvh
      Vector<index_type> leafOffsets{numLeaves, memdst, devid};
      exclusive_scan(execPol, leafDepths.begin(), leafDepths.end(), leafOffsets.begin());
      Vector<index_type> trunkDst{numLeaves - 1, memdst, devid};

      execPol(enumerate(leafLca, leafDepths, leafOffsets),
              ComputeTrunkOrder{execTag, trunkLc, trunkDst, depths});

      execPol(zip(trunkBvs, trunkL, trunkR),
              ReorderTrunk{execTag, numLeaves, trunkDst, depths, leafLca, leafDepths, leafOffsets,
                           sortedBvs, auxIndices, levels});
      execPol(enumerate(leafBvs, leafOffsets, leafDepths),
              ReorderLeafs{execTag, numLeaves, sortedIndices, auxIndices, levels, depths, sortedBvs,
                           leafIndices});

      return lbvh;
    }

    /// refit bvh
    template <execspace_e space, int dim, int lane_width, bool is_double, typename T>
    void refit_lbvh(LBvh<dim, lane_width, is_double> & lbvh,
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
      auto &depths = lbvh.depths;
      auto &auxIndices = lbvh.auxIndices;
      // init bvs, refit flags
      execPol(zip(refitFlags, sortedBvs), ResetRefitStates<space, dim, typename Box::T>{execTag});
      // refit
      execPol(leafIndices, RefitLBvh{execTag, primBvs, auxIndices, depths, refitFlags, sortedBvs});
    }

    /// collision detection traversal
    // collider (aabb, point) - bvh
    template <execspace_e space, typename Index, int dim, int lane_width, bool is_double,
              typename Collider>
    Vector<Index> intersect_lbvh(LBvh<dim, lane_width, is_double> & lbvh,
                                 const Vector<Collider> &colliders) {
      using namespace zs;
      using lbvh_t = LBvh<dim, lane_width, is_double>;
      using float_type = typename lbvh_t::float_type;
      using TV = vec<float_type, dim>;

      const auto memdst = lbvh.sortedBvs.memspace();
      const auto devid = lbvh.sortedBvs.devid();

      Vector<Index> ret{colliders.size() * dim * dim, memdst,
                        devid};  // this is the estimated count
      Vector<tuple<Index, Index>> records{colliders.size() * dim * dim, memdst,
                                          devid};  // this is the estimated count
      Vector<Index> cnt{1, memdst, devid};

      auto execPol = par_exec(wrapv<space>{}).sync(true);
      if constexpr (space == execspace_e::cuda) execPol = execPol.device(devid);

      auto &sortedBvs = lbvh.sortedBvs;
      auto &auxIndices = lbvh.auxIndices;
      auto &levels = lbvh.levels;
      // init bvs, refit flags
      execPol(enumerate(colliders), IntersectLBvh{wrapv<space>{}, records.size(), sortedBvs,
                                                  auxIndices, levels, cnt, records});
      auto n = cnt.clone({memsrc_e::host, -1});
      if (n[0] >= ret.size())
        throw std::runtime_error("not enough space reserved for collision indices");
      ret.resize(n[0]);
      return ret;
    }

  }  // namespace zs
