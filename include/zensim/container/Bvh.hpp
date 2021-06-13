#pragma once

#include "TileVector.hpp"
#include "Vector.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"

namespace zs {

  template <int dim, typename T = float> using AABBBox
      = AnalyticLevelSet<analytic_geometry_e::Cuboid, T, dim>;

  template <int dim_ = 3, int lane_width_ = 32, bool is_double = false> struct LBvh {
    static constexpr int dim = dim_;
    static constexpr int lane_width = lane_width_;
    using value_type = conditional_t<is_double, dat32, dat64>;
    using float_type = remove_cvref_t<decltype(std::declval<value_type>().asFloat())>;
    using integer_type = remove_cvref_t<decltype(std::declval<value_type>().asSignedInteger())>;

    using index_type = conditional_t<is_double, i64, i32>;
    using TV = vec<float_type, dim>;
    using IV = vec<integer_type, dim>;
    using vector_t = Vector<value_type>;
    using indices_t = Vector<integer_type>;
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

    Vector<Box> bvs;
    Box wholeBox{TV::uniform(std::numeric_limits<float>().max()),
                 TV::uniform(std::numeric_limits<float>().min())};

    Vector<Box> sortedBvs;             // bounding volumes
    Vector<index_type> escapeIndices;  // 0-th bit marks leaf/ internal node
    Vector<index_type> levels;         // count from bottom up, 0-based
    indices_t originalIndices;         // map to original primitives

    Vector<int> trunkBuildFlags;
    tilevector_t tree;

    indices_t primitiveIndices;
  };

  template <execspace_e space, int lane_width, int dim, typename T>
  auto build_lbvh(const Vector<AABBBox<dim, T>> &primBvs) {
    using namespace zs;
    static constexpr bool is_double = is_same_v<T, double>;
    using mc_t = conditional_t<is_double, u64, u32>;
    using lbvh_t = LBvh<dim, lane_width, is_double>;
    using index_type = typename lbvh_t::index_type;
    using Box = typename lbvh_t::Box;
    lbvh_t lbvh{};

    const auto numLeaves = primBvs.size();
    const memsrc_e memdst{primBvs.memspace()};
    const ProcID devid{primBvs.devid()};
    auto execPol = par_exec(wrapv<space>{}).sync(true);

    Vector<Box> wholeBox{};
    execPol(range(numLeaves), [bvs = proxy<space>(primBvs), box = proxy<space>(wholeBox)](int id) {
      const Box &box = bvs(id);
      for (int d = 0; d < 3; ++d) {
        atomic_min(exec_omp, &box(0)._min[d], box._min[d]);
        atomic_max(exec_omp, &box(0)._max[d], box._max[d]);
      }
    });
    lbvh.wholeBox = wholeBox.clone(memsrc_e::host)[0];
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
             indices = proxy<space>(indices)](index_type id) {
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
    execPol(range(numLeaves), [numLeaves, splits = proxy<space>(splits),
                               sortedMcs = proxy<space>(sortedMcs)](int id) {
      /// divergent level count
      splits(id) = id != numLeaves - 1
                       ? 32 - count_lz(wrapv<space>{}, sortedMcs[id] ^ sortedMcs[id + 1])
                       : 33;
    });

    Vector<Box> leafBvs{numLeaves, memdst, devid}, trunkBvs{numLeaves - 1, memdst, devid};
    Vector<index_type> leafLca{numLeaves, memdst, devid};
    Vector<index_type> leafPar{numLeaves, memdst, devid};
    Vector<index_type> leafDepths{numLeaves, memdst, devid};
    Vector<index_type> leafTopoMarks{numLeaves, memdst, devid};
    Vector<index_type> trunkTopoMarks{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkPar{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkR{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkL{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkRc{numLeaves - 1, memdst, devid};
    Vector<index_type> trunkLc{numLeaves - 1, memdst, devid};

    /// ...
    return lbvh;
  }

}  // namespace zs
