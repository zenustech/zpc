#pragma once

#include "TileVector.hpp"
#include "Vector.hpp"
#include "zensim/types/Polymorphism.h"
#include "zensim/types/SmallVector.hpp"

namespace zs {

#if 0
  template <int dim_ = 3, int lane_width_ = 32, bool is_double = false> struct LBvh {
    static constexpr int dim = dim_;
    static constexpr int lane_width = lane_width_;
    using value_type = conditional_t<is_double, dat32, dat64>;
    using float_type = remove_cvref_t<std::declval<value_type>().asFloat()>;
    using integer_type = remove_cvref_t<std::declval<value_type>().asSignedInteger()>;

    using index_type = i64;
    using TV = vec<float_type, dim>;
    using IV = vec<integer_type, dim>;
    using vector_t = Vector<value_type>;
    using indices_t = Vector<integer_type>;
    using tilevector_t = TileVector<value_type, lane_width>;

    /// preserved properties
    struct IntNodes {
      // LC, RC, PAR, RCD, MARK, RANGEX, RANGEY
      static constexpr PropertyTag properties[] = {{"indices", 1}, {"lca", 1}, {"upper", dim}};
    };
    struct ExtNodes {
      // PAR, LCA, RCL, STIDX, SEGLEN
      static constexpr PropertyTag properties[] = {{"indices", 1}, {"lower", dim}, {"upper", dim}};
    };

    constexpr LBvh() = default;

    constexpr LBvh() : tree{} { ; }

    tilevector_t tree;
    indices_t primitiveIndices;
  };
#endif

}  // namespace zs
