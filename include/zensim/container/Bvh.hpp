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
    using tiles_t = TileVector<value_type, lane_width, allocator_type>;
    using itiles_t = TileVector<index_type, lane_width, allocator_type>;

    constexpr decltype(auto) memoryLocation() const noexcept { return leafBvs.memoryLocation(); }
    constexpr ProcID devid() const noexcept { return leafBvs.devid(); }
    constexpr memsrc_e memspace() const noexcept { return leafBvs.memspace(); }
    decltype(auto) get_allocator() const noexcept { return leafBvs.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      return leafBvs.get_default_allocator(mre, devid);
    }

    LBvh() = default;

    LBvh clone(const allocator_type &allocator) const {
      LBvh ret{};
      ret.leafBvs = leafBvs.clone(allocator);
      ret.trunkBvs = trunkBvs.clone(allocator);
      ret.leafTopo = leafTopo.clone(allocator);
      ret.trunkTopo = trunkTopo.clone(allocator);
      ret.root = root;
      return ret;
    }
    LBvh clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    constexpr auto getNumLeaves() const noexcept { return leafBvs.size(); }
    constexpr auto getNumNodes() const noexcept { return getNumLeaves() * 2 - 1; }

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
      Vector<Box> box{leafBvs.get_allocator(), 1};
      if (numLeaves <= 2) {
        using TV = typename Box::TV;
        box.setVal(
            Box{TV::uniform(limits<value_type>::max()), TV::uniform(limits<value_type>::lowest())});
        pol(Collapse{numLeaves},
            [bvh = proxy<space>(*this), box = proxy<space>(box)] ZS_LAMBDA(int vi) mutable {
              auto nt = bvh.numLeaves() - 1;
              auto bv = bvh.getNodeBV(nt + vi);
              for (int d = 0; d != dim; ++d) {
                atomic_min(wrapv<space>{}, &box[0]._min[d], bv._min[d]);
                atomic_max(wrapv<space>{}, &box[0]._max[d], bv._max[d]);
              }
            });
      } else {
        pol(Collapse{1}, [bvh = proxy<space>(*this), box = proxy<space>(box),
                          root = root] ZS_LAMBDA(int vi) mutable { box[0] = bvh.getNodeBV(root); });
      }
      return box.getVal();
    }

    tiles_t leafBvs, trunkBvs;
    itiles_t leafTopo, trunkTopo;

    // indices_t prim2leaf;  // leaf index mapping
    index_type root;
  };

  template <execspace_e, typename LBvhT, typename = void> struct LBvhView;

  /// proxy to work within each backends
  template <execspace_e space, typename LBvhT> struct LBvhView<space, const LBvhT> {
    static constexpr int dim = LBvhT::dim;
    static constexpr auto exectag = wrapv<space>{};
    using index_t = typename LBvhT::index_type;
    using bv_t = typename LBvhT::Box;
    using tiles_t = typename LBvhT::tiles_t;
    using itiles_t = typename LBvhT::itiles_t;

    constexpr LBvhView() = default;
    ~LBvhView() = default;

    explicit constexpr LBvhView(const LBvhT &lbvh)
        : _leafBvs{proxy<space>({}, lbvh.leafBvs)},
          _trunkBvs{proxy<space>({}, lbvh.trunkBvs)},
          _leafTopo{proxy<space>({}, lbvh.leafTopo)},
          _trunkTopo{proxy<space>({}, lbvh.trunkTopo)},
          _numNodes{static_cast<index_t>(lbvh.getNumNodes())},
          _root{lbvh.root} {}

    constexpr auto numNodes() const noexcept { return _numNodes; }
    constexpr auto numLeaves() const noexcept { return (numNodes() + 1) / 2; }

    constexpr bv_t getNodeBV(index_t node) const {
      auto nt = numLeaves() - 1;
      auto mi = node >= nt ? _leafBvs.template pack<dim>("min", node - nt)
                           : _trunkBvs.template pack<dim>("min", node);
      auto ma = node >= nt ? _leafBvs.template pack<dim>("max", node - nt)
                           : _trunkBvs.template pack<dim>("max", node);
      return bv_t{mi, ma};
    }

    // BV can be either VecInterface<VecT> or AABBBox<dim, T>
    template <typename BV, class F> constexpr void iter_neighbors(const BV &bv, F &&f) const {
#if 0
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
#else
      auto nt = numLeaves() - 1;
      if (nt + 1 <= 2) {
        for (index_t i = 0; i != nt + 1; ++i) {
          if (overlaps(getNodeBV(nt + i), bv)) f(i);
        }
        return;
      }
      index_t node = _root;
      while (node != -1) {
        for (; node < nt; node = _trunkTopo("lc", node))
          if (!overlaps(getNodeBV(node), bv)) break;
        // leaf node check
        if (node >= nt) {
          if (overlaps(getNodeBV(node), bv)) f(_leafTopo("inds", node - nt));
          node = _leafTopo("esc", node - nt);
        } else  // separate at internal nodes
          node = _trunkTopo("esc", node);
      }
#endif
    }

    TileVectorView<space, const tiles_t, false> _leafBvs, _trunkBvs;
    TileVectorView<space, const itiles_t, false> _leafTopo, _trunkTopo;
    index_t _numNodes, _root;
  };

  template <execspace_e space, int dim, int lane_width, typename Ti, typename T, typename Allocator>
  constexpr decltype(auto) proxy(const LBvh<dim, lane_width, Ti, T, Allocator> &lbvh) {
    return LBvhView<space, const LBvh<dim, lane_width, Ti, T, Allocator>>{lbvh};
  }

  template <typename BvhView, typename BV, class F>
  constexpr void iter_neighbors(const BvhView &bvh, const BV &bv, F &&f) {
    bvh.iter_neighbors(bv, FWD(f));
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

    if (primBvs.size() == 0) return;
    const size_type numLeaves = primBvs.size();
    const size_type numTrunk = numLeaves - 1;
    const size_type numNodes = numLeaves * 2 - 1;

    const memsrc_e memdst{primBvs.memspace()};
    const ProcID devid{primBvs.devid()};

    root = -1;
    leafBvs = tiles_t{primBvs.get_allocator(), {{"min", dim}, {"max", dim}}, numLeaves};

    if (numLeaves <= 2) {  // edge cases where not enough primitives to form a tree
      policy(range(numLeaves), AssignBV{proxy<space>({}, leafBvs), proxy<space>(primBvs)});
      return;
    }
    trunkBvs = tiles_t{primBvs.get_allocator(), {{"min", dim}, {"max", dim}}, numTrunk};
    // total bounding volume
    Vector<Box> wholeBox{1, memdst, devid};
    wholeBox.setVal(
        Box{TV::uniform(limits<value_type>::max()), TV::uniform(limits<value_type>::lowest())});
    policy(primBvs, ComputeBoundingVolume{execTag, wholeBox});

    // morton codes
    Vector<mc_t> mcs{numLeaves, memdst, devid};
    Vector<index_type> indices{numLeaves, memdst, devid};
    policy(zip(range(numLeaves), primBvs, mcs, indices), ComputeMortonCodes{wholeBox.getVal()});
    // puts("done mcs compute");

    // sort by morton codes
    Vector<mc_t> sortedMcs{numLeaves, memdst, devid};
    Vector<index_type> sortedIndices{numLeaves, memdst, devid};
    radix_sort_pair(policy, mcs.begin(), indices.begin(), sortedMcs.begin(), sortedIndices.begin(),
                    numLeaves);

    // split metrics
    Vector<mc_t> splits{primBvs.get_allocator(), numLeaves};
    constexpr auto totalBits = sizeof(mc_t) * 8;
    policy(zip(range(numLeaves), splits),
           ComputeSplitMetric{execTag, numLeaves, sortedMcs, totalBits});

    // build + refit
    leafTopo = itiles_t{primBvs.get_allocator(),
                        {{"par", 1}, {"lca", 1}, {"depth", 1}, {"esc", 1}, {"inds", 1}},
                        numLeaves};
    trunkTopo = itiles_t{primBvs.get_allocator(),
                         {{"par", 1}, {"lc", 1}, {"rc", 1}, {"l", 1}, {"r", 1}, {"esc", 1}},
                         numTrunk};

    indices_t rt{1, memdst, devid};

    Vector<u32> trunkTopoMarks{numTrunk, memdst, devid};
    trunkTopoMarks.reset(0);
    {
      Vector<int> trunkBuildFlags{numTrunk, memdst, devid};
      trunkBuildFlags.reset(0);
      // policy(zip(trunkTopoMarks, trunkBuildFlags), ResetBuildStates{});

      policy(range(numLeaves),
             BuildRefitLBvh{execTag, numLeaves, primBvs, leafBvs, trunkBvs, leafTopo, trunkTopo,
                            splits, sortedIndices, trunkBuildFlags, rt});

      root = rt.getVal();
    }
    policy(range(numLeaves),
           [leafTopo = proxy<space>({}, leafTopo), numLeaves] ZS_LAMBDA(int i) mutable {
             leafTopo("esc", i) = i + 1 != numLeaves ? leafTopo("lca", i + 1) : -1;
           });
    policy(range(numTrunk),
           [leafTopo = proxy<space>({}, leafTopo), trunkTopo = proxy<space>({}, trunkTopo),
            numLeaves] ZS_LAMBDA(int i) mutable {
             auto r = trunkTopo("r", i);
             trunkTopo("esc", i) = r + 1 != numLeaves ? leafTopo("lca", r + 1) : -1;
           });
    // auxIndices (jump table), leafIndices (leafid -> primid)
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

#if 0
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
#else
    if (numLeaves <= 2) {  // edge cases where not enough primitives to form a tree
      policy(range(numLeaves), AssignBV{proxy<space>({}, leafBvs), proxy<space>(primBvs)});
      return;
    }
    // init bvs, refit flags
    Vector<int> refitFlags{leafBvs.get_allocator(), numLeaves};
    refitFlags.reset(0);
    // refit
    policy(Collapse{numLeaves},
           [primBvs = proxy<space>(primBvs), leafBvs = proxy<space>({}, leafBvs),
            trunkBvs = proxy<space>({}, trunkBvs), leafTopo = proxy<space>({}, leafTopo),
            trunkTopo = proxy<space>({}, trunkTopo), flags = proxy<space>(refitFlags),
            numTrunk = numLeaves - 1] ZS_LAMBDA(int node) mutable {
             auto primid = leafTopo("inds", node);
             auto bv = primBvs(primid);
             leafBvs.template tuple<dim>("min", node) = bv._min;
             leafBvs.template tuple<dim>("max", node) = bv._max;
             node = leafTopo("par", node);
             while (node != -1) {
               if (atomic_cas(wrapv<space>{}, &flags[node], 0, 1) == 0) break;
               auto lc = trunkTopo("lc", node);
               auto rc = trunkTopo("rc", node);
               auto lMin = lc < numTrunk ? trunkBvs.template pack<dim>("min", lc)
                                         : leafBvs.template pack<dim>("min", lc - numTrunk);
               auto lMax = lc < numTrunk ? trunkBvs.template pack<dim>("max", lc)
                                         : leafBvs.template pack<dim>("max", lc - numTrunk);
               auto rMin = rc < numTrunk ? trunkBvs.template pack<dim>("min", rc)
                                         : leafBvs.template pack<dim>("min", rc - numTrunk);
               auto rMax = rc < numTrunk ? trunkBvs.template pack<dim>("max", rc)
                                         : leafBvs.template pack<dim>("max", rc - numTrunk);
               for (int d = 0; d != dim; ++d) {
                 trunkBvs("min", d, node) = lMin[d] < rMin[d] ? lMin[d] : rMin[d];
                 trunkBvs("max", d, node) = lMax[d] > rMax[d] ? lMax[d] : rMax[d];
               }
               thread_fence(wrapv<space>{});
               node = trunkTopo("par", node);
             }
           });
#endif
    return;
  }

}  // namespace zs
