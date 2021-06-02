#pragma once

#include "TileVector.hpp"
#include "Vector.hpp"
#include "zensim/types/Polymorphism.h"

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
    using tilevector_t = TileVector<value_type, lane_width>;

    /// preserved properties
    static constexpr const char *prop_names[] = {""};
    static constexpr char prop_size[] = {};

    constexpr LBvh() = default;

    constexpr LBvh() : tree{} { ; }

    tilevector_t tree;
  };
#endif

}  // namespace zs
