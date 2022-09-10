#pragma once

#include "BvhImpl.tpp"
#include "TileVector.hpp"
#include "Vector.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/zpc_tpls/fmt/color.h"

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

    indices_t rt{primBvs.get_allocator(), 1};

    Vector<int> trunkBuildFlags{primBvs.get_allocator(), numTrunk};
    indices_t leafDepths{primBvs.get_allocator(), numLeaves + 1},
        leafOffsets{primBvs.get_allocator(), numLeaves + 1},
        trunkDst{primBvs.get_allocator(), numTrunk};

    trunkBuildFlags.reset(0);
    {
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

    if constexpr (true) {  // ordering
      policy(range(numLeaves),
             [leafTopo = proxy<space>({}, leafTopo), leafOffsets = proxy<space>(leafOffsets),
              leafDepths = proxy<space>(leafDepths)] ZS_LAMBDA(int i) mutable {
               leafDepths[i] = leafTopo("depth", i) - 1;
             });

      exclusive_scan(policy, std::begin(leafDepths), std::end(leafDepths), std::begin(leafOffsets));

      policy(range(numLeaves - 1),
             [leafTopo = proxy<space>({}, leafTopo), trunkTopo = proxy<space>({}, trunkTopo),
              trunkDst = proxy<space>(trunkDst), leafOffsets = proxy<space>(leafOffsets),
              numTrunk] ZS_LAMBDA(int i) mutable {
               auto trunkOffset = leafOffsets(i);
               auto node = leafTopo("lca", i);
               for (; node < numTrunk; node = trunkTopo("lc", node)) {
                 trunkDst[node] = trunkOffset++;
               }
             });
      auto orderedTrunkTopo = trunkTopo;
      auto orderedTrunkBvs = trunkBvs;

      policy(
          range(numTrunk),
          [trunkDst = proxy<space>(trunkDst), trunkTopo = proxy<space>({}, trunkTopo),
           trunkBvs = proxy<space>({}, trunkBvs),
           orderedTrunkTopo = proxy<space>({}, orderedTrunkTopo),
           orderedTrunkBvs = proxy<space>({}, orderedTrunkBvs), numTrunk] ZS_LAMBDA(int i) mutable {
            auto dst = trunkDst[i];
            auto mapped_index = [&trunkDst, numTrunk](int id) {
              // >=0 predicate ensuring -1 (esc) is preserved
              return id >= 0 && id < numTrunk ? trunkDst[id] : id;
            };
            orderedTrunkTopo("par", dst) = mapped_index(trunkTopo("par", i));
            orderedTrunkTopo("lc", dst) = mapped_index(trunkTopo("lc", i));
            orderedTrunkTopo("rc", dst) = mapped_index(trunkTopo("rc", i));
            orderedTrunkTopo("esc", dst) = mapped_index(trunkTopo("esc", i));
            orderedTrunkTopo("l", dst) = trunkTopo("l", i);
            orderedTrunkTopo("r", dst) = trunkTopo("r", i);
            orderedTrunkBvs.template tuple<dim>("min", dst) = trunkBvs.template pack<dim>("min", i);
            orderedTrunkBvs.template tuple<dim>("max", dst) = trunkBvs.template pack<dim>("max", i);
          });
      trunkTopo = std::move(orderedTrunkTopo);
      trunkBvs = std::move(orderedTrunkBvs);

      policy(range(numLeaves),
             [trunkDst = proxy<space>(trunkDst), leafTopo = proxy<space>({}, leafTopo),
              numTrunk] ZS_LAMBDA(int i) mutable {
               auto mapped_index = [&trunkDst, numTrunk](int id) {
                 return id >= 0 && id < numTrunk ? trunkDst[id] : id;
               };
               leafTopo("par", i) = trunkDst[leafTopo("par", i)];
               leafTopo("lca", i) = mapped_index(leafTopo("lca", i));
               leafTopo("esc", i) = mapped_index(leafTopo("esc", i));
             });
#if 0
      if (false) {  // dbg
        policy(range(numTrunk), [lDepths, lOffsets, tLs, tRs, tLcs, tRcs, tPars, lPars, lLcas, tDst,
                                 auxIndices = proxy<space>(auxIndices),
                                 levels = proxy<space>(levels), parents = proxy<space>(parents),
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
                printf("trunk child %d <%d, %d>\' parent %d <%d, %d> not %d <%d, %d>!\n", ch,
                       tLs[ch], tRs[ch], p, tLs[p], tRs[p], par, tLs[par], tRs[par]);
                triggered();
              }
              if (parents[tDst[ch]] != tDst[par]) {
                int actPar = parents[tDst[ch]];
                int incPar = tDst[par];
                printf("ordered trunk ch %d\' parent %d not %d\n", (int)tDst[ch], actPar, incPar);
              }
            } else {
              if (int p = lPars[ch - numTrunk]; p != par) {
                printf("leaf child %d\' parent %d <%d, %d> not %d <%d, %d>!\n", ch, p, tLs[p],
                       tRs[p], par, tLs[par], tRs[par]);
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
              || !(zs::get<0>(range) == zs::get<0>(lRange)
                   && zs::get<1>(range) == zs::get<1>(rRange)
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
            printf("leaf %d branch, lca %d. depth %d, level %d, trunk node %d (%d ordered)\n",
                   (int)l, (int)lca, (int)lDepths[l], (int)levels[tDst[lca]], (int)idx, (int)dst);
            for (int n = lca; n < numTrunk; n = tLcs[n]) {
              auto lc_ = tLcs[n];
              printf("node %d <%d, %d>, level %d, lc %d (%d ordered)\n", n, (int)tLs[n],
                     (int)tRs[n], (int)levels[tDst[n]], (int)lc_,
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
      }  // end dbg
      root = 0;
#endif
    }
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
                 trunkBvs("min", d, node) = zs::min(lMin[d], rMin[d]);
                 trunkBvs("max", d, node) = zs::max(lMax[d], rMax[d]);
               }
               thread_fence(wrapv<space>{});
               node = trunkTopo("par", node);
             }
           });
    return;
  }

}  // namespace zs