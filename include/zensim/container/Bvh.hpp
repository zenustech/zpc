#pragma once

#include "BvhImpl.tpp"
#include "TileVector.hpp"
#include "Vector.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"

namespace zs {

  template <int dim_ = 3, int lane_width_ = 32, typename Index = int, typename ValueT = f32,
            typename AllocatorT = ZSPmrAllocator<>>
  struct LBvh {
    static constexpr int dim = dim_;
    static constexpr int lane_width = lane_width_;
    using allocator_type = AllocatorT;
    using value_type = ValueT;
    // must be signed integer, since we are using -1 as sentinel value
    using index_type = std::make_signed_t<Index>;
    using size_type = std::make_unsigned_t<Index>;
    static_assert(std::is_floating_point_v<value_type>, "value_type should be floating point");
    static_assert(std::is_integral_v<index_type>, "index_type should be an integral");

    using mc_t = conditional_t<is_same_v<value_type, f64>, u64, u32>;
    using Box = AABBBox<dim, value_type>;
    using TV = vec<value_type, dim>;
    using IV = vec<index_type, dim>;
    using bvs_t = Vector<Box, allocator_type>;
    using vector_t = Vector<value_type, allocator_type>;
    using indices_t = Vector<index_type, allocator_type>;
    using tilevector_t = TileVector<value_type, lane_width, allocator_type>;

    constexpr decltype(auto) memoryLocation() const noexcept {
      return leafIndices.memoryLocation();
    }
    constexpr ProcID devid() const noexcept { return leafIndices.devid(); }
    constexpr memsrc_e memspace() const noexcept { return leafIndices.memspace(); }
    decltype(auto) get_allocator() const noexcept { return leafIndices.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      return leafIndices.get_default_allocator(mre, devid);
    }

    LBvh() = default;

    LBvh clone(const allocator_type &allocator) const {
      LBvh ret{};
      ret.sortedBvs = sortedBvs.clone(allocator);
      ret.auxIndices = auxIndices.clone(allocator);
      ret.levels = levels.clone(allocator);
      ret.parents = parents.clone(allocator);
      ret.leafIndices = leafIndices.clone(allocator);
      return ret;
    }
    LBvh clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    constexpr auto getNumNodes() const noexcept { return auxIndices.size(); }
    constexpr auto getNumLeaves() const noexcept { return leafIndices.size(); }

    template <typename Policy>
    void build(Policy &&, const Vector<AABBBox<dim, value_type>> &primBvs);
    template <typename Policy>
    void refit(Policy &&, const Vector<AABBBox<dim, value_type>> &primBvs);
    template <typename Policy> Box getTotalBox(Policy &&pol) const {
      constexpr auto space = std::remove_reference_t<Policy>::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
      // ZS_LAMBDA -> __device__
      static_assert(space == execspace_e::cuda, "specialized policy and compiler not match");
#else
      static_assert(space != execspace_e::cuda, "specialized policy and compiler not match");
#endif

      auto numLeaves = getNumLeaves();
      Vector<Box> box{sortedBvs.get_allocator(), 1};
      if (numLeaves <= 2) {
        using TV = typename Box::TV;
        box.setVal(
            Box{TV::uniform(limits<value_type>::max()), TV::uniform(limits<value_type>::lowest())});
        pol(Collapse{numLeaves},
            [bvh = proxy<space>(*this), box = proxy<space>(box)] ZS_LAMBDA(int vi) mutable {
              auto bv = bvh.getNodeBV(vi);
              for (int d = 0; d != dim; ++d) {
                atomic_min(wrapv<space>{}, &box[0]._min[d], bv._min[d]);
                atomic_max(wrapv<space>{}, &box[0]._max[d], bv._max[d]);
              }
            });
      } else {
        pol(Collapse{1}, [bvh = proxy<space>(*this), box = proxy<space>(box)] ZS_LAMBDA(
                             int vi) mutable { box[0] = bvh.getNodeBV(0); });
      }
      return box.getVal();
    }

    tilevector_t sortedBvs;
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
    using index_t = typename LBvhT::index_type;
    using bv_t = typename LBvhT::Box;
    using bvs_t = typename LBvhT::bvs_t;
    using indices_t = typename LBvhT::indices_t;
    using tilevector_t = typename LBvhT::tilevector_t;

    constexpr LBvhView() = default;
    ~LBvhView() = default;

    explicit constexpr LBvhView(const LBvhT &lbvh)
        : _sortedBvs{proxy<space>({}, lbvh.sortedBvs)},
          _auxIndices{proxy<space>(lbvh.auxIndices)},
          _levels{proxy<space>(lbvh.levels)},
          _parents{proxy<space>(lbvh.parents)},
          _leafIndices{proxy<space>(lbvh.leafIndices)},
          _numNodes{static_cast<index_t>(lbvh.getNumNodes())} {}

    constexpr auto numNodes() const noexcept { return _numNodes; }
    constexpr auto numLeaves() const noexcept { return (numNodes() + 1) / 2; }

    constexpr bv_t getNodeBV(index_t node) const {
      auto mi = _sortedBvs.template pack<dim>("min", node);
      auto ma = _sortedBvs.template pack<dim>("max", node);
      return bv_t{mi, ma};
    }

    // BV can be either VecInterface<VecT> or AABBBox<dim, T>
    template <typename BV, class F> constexpr void iter_neighbors(const BV &bv, F &&f) const {
      if (auto nl = numLeaves(); nl <= 2) {
        for (index_t i = 0; i != nl; ++i) {
          if (overlaps(getNodeBV(i), bv)) f(_auxIndices[i]);
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
          if (overlaps(getNodeBV(node), bv)) f(_auxIndices[node]);
          node++;
        } else  // separate at internal nodes
          node = _auxIndices[node];
      }
    }

    TileVectorView<space, const tilevector_t, false> _sortedBvs;
    VectorView<space, const indices_t> _auxIndices, _levels, _parents, _leafIndices;
    index_t _numNodes;
  };

  template <execspace_e space, int dim, int lane_width, typename Ti, typename T, typename Allocator>
  constexpr decltype(auto) proxy(const LBvh<dim, lane_width, Ti, T, Allocator> &lbvh) {
    return LBvhView<space, const LBvh<dim, lane_width, Ti, T, Allocator>>{lbvh};
  }

  template <typename BvhView, typename BV, class F>
  constexpr void iter_neighbors(const BvhView &bvh, const BV &bv, F &&f) {
    using index_t = typename BvhView::index_t;
    if (auto nl = bvh.numLeaves(); nl <= 2) {
      for (index_t i = 0; i != nl; ++i) {
        if (overlaps(bvh.getNodeBV(i), bv)) f(bvh._auxIndices[i]);
      }
      return;
    }
    index_t node = 0;
    while (node != -1 && node != bvh._numNodes) {
      index_t level = bvh._levels[node];
      // level and node are always in sync
      for (; level; --level, ++node)
        if (!overlaps(bvh.getNodeBV(node), bv)) break;
      // leaf node check
      if (level == 0) {
        if (overlaps(bvh.getNodeBV(node), bv)) f(bvh._auxIndices[node]);
        node++;
      } else  // separate at internal nodes
        node = bvh._auxIndices[node];
    }
  }

  template <typename BvTilesView, typename BvVectorView> struct AssignBV {
    using size_type = typename BvTilesView::size_type;
    static constexpr int d = remove_cvref_t<typename BvVectorView::value_type>::dim;
    constexpr AssignBV(BvTilesView t, BvVectorView v) noexcept : tiles{t}, vector{v} {}
    constexpr void operator()(size_type i) noexcept {
      tiles.template tuple<d>("min", i) = vector[i]._min;
      tiles.template tuple<d>("max", i) = vector[i]._max;
    }
    BvTilesView tiles;
    BvVectorView vector;
  };

  template <int dim, int lane_width, typename Index, typename Value, typename Allocator>
  template <typename Policy> void LBvh<dim, lane_width, Index, Value, Allocator>::build(
      Policy &&policy, const Vector<AABBBox<dim, Value>> &primBvs) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    constexpr auto execTag = wrapv<space>{};

    const size_type numLeaves = primBvs.size();
    const size_type numNodes = numLeaves * 2 - 1;

    const memsrc_e memdst{primBvs.memspace()};
    const ProcID devid{primBvs.devid()};

    sortedBvs = tilevector_t{{{"min", dim}, {"max", dim}}, numNodes, memdst, devid};
    auxIndices = indices_t{numNodes, memdst, devid};
    levels = indices_t{numNodes, memdst, devid};
    parents = indices_t{numNodes, memdst, devid};
    leafIndices = indices_t{numLeaves, memdst, devid};

    if (numLeaves <= 2) {  // edge cases where not enough primitives to form a tree
      policy(range(numLeaves), AssignBV{proxy<space>({}, sortedBvs), proxy<space>(primBvs)});
      for (size_type i = 0; i != numLeaves; ++i) {
        leafIndices.setVal(i, i);
        levels.setVal(0, i);
        auxIndices.setVal(i, i);
        parents.setVal(-1, i);
      }
      return;
    }
    // total bounding volume
    Vector<Box> wholeBox{1, memdst, devid};
    wholeBox.setVal(
        Box{TV::uniform(limits<value_type>::max()), TV::uniform(limits<value_type>::lowest())});
    policy(primBvs, ComputeBoundingVolume{execTag, wholeBox});

    // morton codes
    Vector<mc_t> mcs{numLeaves, memdst, devid};
    Vector<index_type> indices{numLeaves, memdst, devid};
    policy(enumerate(primBvs, mcs, indices), ComputeMortonCodes{wholeBox.getVal()});
    // puts("done mcs compute");

    // sort by morton codes
    Vector<mc_t> sortedMcs{numLeaves, memdst, devid};
    Vector<index_type> sortedIndices{numLeaves, memdst, devid};
    radix_sort_pair(policy, mcs.begin(), indices.begin(), sortedMcs.begin(), sortedIndices.begin(),
                    numLeaves);

    // split metrics
    Vector<mc_t> splits{numLeaves, memdst, devid};
    constexpr auto totalBits = sizeof(mc_t) * 8;
    policy(enumerate(splits), ComputeSplitMetric{execTag, numLeaves, sortedMcs, totalBits});

    // build + refit
    tilevector_t leafBvs{{{"min", dim}, {"max", dim}}, numLeaves, memdst, devid};
    tilevector_t trunkBvs{{{"min", dim}, {"max", dim}}, numLeaves - 1, memdst, devid};
    indices_t leafLca{numLeaves, memdst, devid};
    indices_t leafDepths{numLeaves + 1, memdst, devid};
    indices_t trunkR{numLeaves - 1, memdst, devid};
    indices_t trunkLc{numLeaves - 1, memdst, devid};

    /// build + refit
    Vector<u32> trunkTopoMarks{numLeaves - 1, memdst, devid};
    {
      indices_t trunkL{numLeaves - 1, memdst, devid};
      indices_t trunkRc{numLeaves - 1, memdst, devid};
      Vector<int> trunkBuildFlags{numLeaves - 1, memdst, devid};
      policy(zip(trunkTopoMarks, trunkBuildFlags), ResetBuildStates{});

      policy(range(numLeaves),
             BuildRefitLBvh{execTag, numLeaves, primBvs, leafBvs, trunkBvs, splits, sortedIndices,
                            leafLca, leafDepths, trunkL, trunkR, trunkLc, trunkRc, trunkTopoMarks,
                            trunkBuildFlags});
    }

    /// sort bvh
#if 0
    indices_t leafOffsets{numLeaves + 1, memdst, devid};
    exclusive_scan(policy, leafDepths.begin(), leafDepths.end(), leafOffsets.begin());
#else
    indices_t leafOffsets{};
    {
      const auto &depths
          = memdst == memsrc_e::host ? leafDepths : leafDepths.clone({memsrc_e::host, -1});
      indices_t offsets{numLeaves + 1, memsrc_e::host, -1};
      offsets[0] = 0;
      for (int i = 0; i != numLeaves; ++i) offsets[i + 1] = offsets[i] + depths[i];
      if (memdst != memsrc_e::host)
        leafOffsets = offsets.clone({memdst, devid});
      else
        leafOffsets = std::move(offsets);
    }
#endif
    indices_t trunkDst{numLeaves - 1, memdst, devid};

    policy(zip(leafLca, leafDepths, leafOffsets),
           ComputeTrunkOrder{execTag, trunkLc, trunkDst, levels, parents});
    policy(enumerate(trunkDst, trunkR),
           ReorderTrunk{execTag, wrapt<Box>{}, (index_type)numLeaves, trunkDst, leafLca,
                        leafOffsets, trunkBvs, sortedBvs, auxIndices, parents});
    policy(zip(range(numLeaves), leafOffsets, leafDepths),
           ReorderLeafs{execTag, wrapt<Box>{}, (index_type)numLeaves, sortedIndices, auxIndices,
                        parents, levels, leafBvs, sortedBvs, leafIndices});
    return;
  }

  template <int dim, int lane_width, typename Index, typename Value, typename Allocator>
  template <typename Policy> void LBvh<dim, lane_width, Index, Value, Allocator>::refit(
      Policy &&policy, const Vector<AABBBox<dim, Value>> &primBvs) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    constexpr auto execTag = wrapv<space>{};

    const size_type numLeaves = getNumLeaves();
    const size_type numNodes = getNumNodes();

    if (primBvs.size() != numLeaves)
      throw std::runtime_error("bvh topology changes, require rebuild!");

    const memsrc_e memdst{sortedBvs.memspace()};
    const ProcID devid{sortedBvs.devid()};

    if (numLeaves <= 2) {  // edge cases where not enough primitives to form a tree
      policy(range(numLeaves), AssignBV{proxy<space>({}, sortedBvs), proxy<space>(primBvs)});
      for (size_type i = 0; i != numLeaves; ++i) {
        leafIndices.setVal(i, i);
        levels.setVal(0, i);
        auxIndices.setVal(i, i);
        parents.setVal(-1, i);
      }
      return;
    }

    // init bvs, refit flags
    Vector<int> refitFlags{numNodes, memdst, devid};
    policy(refitFlags, ResetRefitStates{});
    // refit
    policy(leafIndices,
           RefitLBvh{execTag, primBvs, parents, levels, auxIndices, refitFlags, sortedBvs});
    return;
  }

}  // namespace zs
