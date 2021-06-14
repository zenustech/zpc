#pragma once

#include "TileVector.hpp"
#include "Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"

namespace zs {

  template <int dim, typename T = float> using AABBBox
      = AnalyticLevelSet<analytic_geometry_e::Cuboid, T, dim>;

  template <int dim_ = 3, int lane_width_ = 32, bool is_double = false> struct LBvh {
    static constexpr int dim = dim_;
    static constexpr int lane_width = lane_width_;
    using value_type = conditional_t<is_double, dat64, dat32>;
    using float_type = remove_cvref_t<decltype(std::declval<value_type>().asFloat())>;
    using integer_type = remove_cvref_t<decltype(std::declval<value_type>().asSignedInteger())>;

    using index_type = conditional_t<is_double, i64, i32>;
    using TV = vec<float_type, dim>;
    using IV = vec<integer_type, dim>;
    using vector_t = Vector<value_type>;
    using tilevector_t = TileVector<value_type, lane_width>;
    using Box = AABBBox<dim, float_type>;

    /// preserved properties
    struct IntNodes {
      // LC, RC, PAR, RCD, MARK, RANGEX, RANGEY
      static constexpr PropertyTag properties[] = {{"indices", 1}, {"lca", 1}, {"upper", dim}};
    };
    struct ExtNodes {
      // PAR, LCA, RCL, STIDX, SEGLEN
      static constexpr PropertyTag properties[] = {{"indices", 1}, {"lower", dim}, {"upper", dim}};
    };

    LBvh() = default;

    constexpr auto numLeaves() noexcept { return (sortedBvs.size() + 1) / 2; }
    Vector<Box> bvs;
    Box wholeBox{TV::uniform(std::numeric_limits<float>().max()),
                 TV::uniform(std::numeric_limits<float>().min())};

    Vector<Box> sortedBvs;               // bounding volumes
    Vector<index_type> escapeIndices;    // 0-th bit marks leaf/ internal node
    Vector<index_type> levels;           // count from bottom up, 0-based
    Vector<index_type> originalIndices;  // map to original primitives
    Vector<index_type> leafIndices;      // leaf indices within optimized lbvh

    tilevector_t tree;
  };

  /// utilities
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

    lbvh_t lbvh{};

    const auto numLeaves = primBvs.size();
    const memsrc_e memdst{primBvs.memspace()};
    const ProcID devid{primBvs.devid()};
    auto execPol = par_exec(wrapv<space>{}).sync(true);

    Vector<Box> wholeBox{1, memdst, devid};
    execPol(range(numLeaves),
            [bvs = proxy<space>(primBvs), box = proxy<space>(wholeBox)](int id) mutable {
              const Box bv = bvs(id);
              for (int d = 0; d < 3; ++d) {
                atomic_min(wrapv<space>{}, &box(0)._min[d], bv._min[d]);
                atomic_max(wrapv<space>{}, &box(0)._max[d], bv._max[d]);
              }
            });
    lbvh.wholeBox = wholeBox.clone(MemoryHandle{memsrc_e::host, -1})[0];
    {
      auto &wholeBox = lbvh.wholeBox;
      fmt::print("{}, {}, {} - {}, {}, {}\n", wholeBox._min[0], wholeBox._min[1], wholeBox._min[2],
                 wholeBox._max[0], wholeBox._max[1], wholeBox._max[2]);
    }

    // morton codes
    Vector<mc_t> mcs{numLeaves, memdst, devid};
    Vector<index_type> indices{numLeaves, memdst, devid};
    execPol(range(numLeaves),
            [bvs = proxy<space>(primBvs), wholeBox = lbvh.wholeBox, mcs = proxy<space>(mcs),
             indices = proxy<space>(indices)](index_type id) mutable {
              auto c = bvs(id).getBoxCenter();
              auto coord = wholeBox.getUniformCoord(c);
              mcs(id) = morton_3d(coord[0], coord[1], coord[2]);
              indices(id) = id;
            });

    // sort by morton codes
    Vector<mc_t> sortedMcs{numLeaves, memdst, devid};
    Vector<index_type> sortedIndices{numLeaves, memdst, devid};
    radix_sort_pair(execPol, mcs.begin(), indices.begin(), sortedMcs.begin(), sortedIndices.begin(),
                    mcs.size());

    Vector<mc_t> splits{numLeaves, memdst, devid};
    constexpr auto totalBits = sizeof(mc_t) * 8;
    execPol(range(numLeaves), [numLeaves, splits = proxy<space>(splits),
                               sortedMcs = proxy<space>(sortedMcs)](int id) mutable {
      /// divergent level count
      splits(id) = id != numLeaves - 1
                       ? totalBits - count_lz(wrapv<space>{}, sortedMcs(id) ^ sortedMcs(id + 1))
                       : totalBits + 1;
    });

    Vector<Box> leafBvs{numLeaves, memdst, devid}, trunkBvs{numLeaves - 1, memdst, devid};
    Vector<index_type> leafLca{numLeaves, memdst, devid};
    Vector<index_type> leafPar{numLeaves, memdst, devid};
    Vector<index_type> leafDepths{numLeaves, memdst, devid};
    Vector<index_type> trunkPar{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkR{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkL{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkRc{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkLc{numLeaves - 1, memdst, devid};

    /// build + refit
    Vector<index_type> trunkTopoMarks{numLeaves - 1, memdst, devid};
    {
      Vector<index_type> trunkBuildFlags{numLeaves - 1, memdst, devid};
      execPol(range(numLeaves - 1), [marks = proxy<space>(trunkTopoMarks),
                                     flags = proxy<space>(trunkBuildFlags)](int trunkId) mutable {
        flags(trunkId) = 0;
        marks(trunkId) = 0;
      });
      ;
      execPol(range(numLeaves), [numLeaves, leafBvs = proxy<space>(leafBvs),
                                 trunkBvs = proxy<space>(trunkBvs), splits = proxy<space>(splits),
                                 leafLca = proxy<space>(leafLca), leafPar = proxy<space>(leafPar),
                                 leafDepths = proxy<space>(leafDepths),
                                 trunkPar = proxy<space>(trunkPar), trunkR = proxy<space>(trunkR),
                                 trunkL = proxy<space>(trunkL), trunkLc = proxy<space>(trunkLc),
                                 trunkRc = proxy<space>(trunkRc),
                                 trunkTopoMarks = proxy<space>(trunkTopoMarks),
                                 trunkBuildFlags = proxy<space>(trunkBuildFlags)](int idx) mutable {
        using BvsProxy = remove_cvref_t<decltype(leafBvs)>;
        leafLca(idx) = -1, leafDepths(idx) = 1;
        int l = idx - 1, r = idx;  ///< (l, r]
        bool mark{};

        if (l >= 0)
          mark = splits(l) < splits(r);  ///< true when right child, false otherwise
        else
          mark = false;

        int cur = mark ? l : r;
        leafPar(idx) = cur;
        if (mark)
          trunkRc(cur) = idx, trunkR(cur) = idx,
          atomic_or(wrapv<space>{}, &trunkTopoMarks(cur), (index_type)0x00000002);
        else
          trunkLc(cur) = idx, trunkL(cur) = idx,
          atomic_or(wrapv<space>{}, &trunkTopoMarks(cur), (index_type)0x00000001);

        while (atomic_add(wrapv<space>{}, &trunkBuildFlags(cur), (index_type)1) == 1) {
          {  // refit
            int lc = trunkLc(cur), rc = trunkRc(cur);
            Box bv{TV::uniform(std::numeric_limits<float>().max()),
                   TV::uniform(std::numeric_limits<float>().min())};
            BvsProxy left{}, right{};
            switch (trunkTopoMarks(cur) & 3) {
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
            const Box &leftBox = left(lc);
            const Box &rightBox = right(rc);
            for (int d = 0; d < 3; ++d) {
              bv._min[d] = leftBox._min[d] < rightBox._min[d] ? leftBox._min[d] : rightBox._min[d];
              bv._max[d] = leftBox._max[d] > rightBox._max[d] ? leftBox._max[d] : rightBox._max[d];
            }
            trunkBvs(cur) = bv;
          }
          trunkTopoMarks(cur) &= 0x00000007;

          l = trunkL(cur) - 1, r = trunkR(cur);
          leafLca(l + 1) = cur /*, trunkRcd(cur) = ++_lvs.rcl(r)*/, leafDepths(l + 1)++;
          if (l >= 0)
            mark = splits(l) < splits(r);  ///< true when right child, false otherwise
          else
            mark = false;

          if (l + 1 == 0 && r == numLeaves - 1) {
            trunkPar(cur) = -1;
            trunkTopoMarks(cur) &= 0xFFFFFFFB;
            break;
          }

          int par = mark ? l : r;
          trunkPar(cur) = par;
          if (mark)
            trunkRc(par) = cur, trunkR(par) = r,
            atomic_and(exec_omp, &trunkTopoMarks(par), (index_type)0xFFFFFFFD),
            trunkTopoMarks(cur) |= 0x00000004;
          else
            trunkLc(par) = cur, trunkL(par) = l + 1,
            atomic_and(exec_omp, &trunkTopoMarks(par), (index_type)0xFFFFFFFE),
            trunkTopoMarks(cur) &= 0xFFFFFFFB;
          cur = par;
        }
      });
    }
    /// sort bvh
    Vector<index_type> leafOffsets{numLeaves, memdst, devid};
    exclusive_scan(execPol, leafDepths.begin(), leafDepths.end(), leafOffsets.begin());
    Vector<index_type> trunkDst{numLeaves - 1, memdst, devid};
    execPol(range(numLeaves),
            [leafLca = proxy<space>(leafLca), leafDepths = proxy<space>(leafDepths),
             leafOffsets = proxy<space>(leafOffsets), trunkLc = proxy<space>(trunkLc),
             trunkDst = proxy<space>(trunkDst)](index_type idx) mutable {
              int node = leafLca(idx), depth = leafDepths(idx), id = leafOffsets(idx);
              for (; --depth; node = trunkLc(node)) trunkDst(node) = id++;
            });

    auto &sortedBvs = lbvh.sortedBvs;
    auto &escapeIndices = lbvh.escapeIndices;
    auto &originalIndices = lbvh.originalIndices;
    auto &levels = lbvh.levels;
    auto &leafIndices = lbvh.leafIndices;

    sortedBvs.resize(numLeaves + numLeaves - 1);
    escapeIndices.resize(numLeaves + numLeaves - 1);
    originalIndices.resize(numLeaves + numLeaves - 1);
    levels.resize(numLeaves + numLeaves - 1);
    leafIndices.resize(numLeaves);  // for refit
    execPol(range(numLeaves - 1),
            [sortedBvs = proxy<space>(sortedBvs), escapeIndices = proxy<space>(escapeIndices),
             originalIndices = proxy<space>(originalIndices), levels = proxy<space>(levels),
             leafLca = proxy<space>(leafLca), leafOffsets = proxy<space>(leafOffsets),
             leafDepths = proxy<space>(leafDepths), trunkBvs = proxy<space>(trunkBvs),
             trunkL = proxy<space>(trunkL), trunkR = proxy<space>(trunkR),
             trunkDst = proxy<space>(trunkDst)](index_type idx) mutable {
              int dst = trunkDst(idx);
              const Box &bv = trunkBvs(idx);
              sortedBvs(dst)._min = bv._min;
              sortedBvs(dst)._max = bv._max;
              const auto rb = trunkR(idx);
              const auto escIndex = leafLca(rb + 1);
              escapeIndices(dst) = escIndex == -1 ? leafOffsets(rb + 1) : trunkDst(escIndex);
              originalIndices(dst) = idx << 1;
              levels(dst) = leafDepths(trunkL(idx)) - 1;  // 0-based
            });
    // leaves
    execPol(range(numLeaves),
            [sortedBvs = proxy<space>(sortedBvs), escapeIndices = proxy<space>(escapeIndices),
             originalIndices = proxy<space>(originalIndices), levels = proxy<space>(levels),
             leafIndices = proxy<space>(leafIndices), leafBvs = proxy<space>(leafBvs),
             leafOffsets = proxy<space>(leafOffsets),
             leafDepths = proxy<space>(leafDepths)](index_type idx) mutable {
              const auto dst = leafOffsets(idx) + leafDepths(idx) - 1;
              const Box &bv = leafBvs(idx);
              leafIndices(idx) = dst;
              sortedBvs(dst)._min = bv._min;
              sortedBvs(dst)._max = bv._max;
              escapeIndices(dst) = dst + 1;
              originalIndices(dst) = (idx << 1) | 1;
              levels(dst) = 0;
            });
#if 0
    fmt::print("{} leaves in total\n", numLeaves);

    int count = 0;
    bool isInternal = true;
    int node = leafLca(0);
    execPol(range(1), [&](int) {
      while (true) {
        count++;
        if (isInternal) {
          (void)(trunkBvs[node]);
          if (trunkTopoMarks[node] & 1) {
            isInternal = false;
            node = trunkL[node];
          } else {
            node = trunkLc[node];
          }
        } else {
          (void)(leafBvs[node]);
          if (node == numLeaves - 1) break;
          auto lca = leafLca[node + 1];
          if (lca == -1)
            node++;
          else {
            node = lca;
            isInternal = true;
          }
        }
      }
    });
    fmt::print("traversed {} nodes in total\n", count);
    getchar();
#endif
    return lbvh;
  }

  template <execspace_e space, int dim, int lane_width, bool is_double, typename T>
  void refit_lbvh(LBvh<dim, lane_width, is_double> &lbvh, const Vector<AABBBox<dim, T>> &primBvs) {
    using namespace zs;
    using lbvh_t = LBvh<dim, lane_width, is_double>;
    using float_type = typename lbvh_t::float_type;
    using index_type = typename lbvh_t::index_type;
    using Box = typename lbvh_t::Box;
    using TV = vec<float_type, dim>;

    const auto numLeaves = lbvh.numLeaves();
    const auto memdst = lbvh.sortedBvs.memspace();
    const auto devid = lbvh.sortedBvs.devid();
    Vector<int> refitFlags{numLeaves - 1, memdst, devid};
    auto &leafIndices = lbvh.leafIndices;
  }

}  // namespace zs
