#include "zensim/container/TileVector.hpp"

namespace zs {

template <typename Pol, typename T, auto L0, typename Allocator, typename TEle, auto L1>
void spray_points(Pol &&pol, const TileVector<T, L0, Allocator> &verts, const TileVector<TEle, L1, Allocator> &eles, std::size_t numSamples, TileVector<TOut> &pts) {
    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;

    using Ti = conditional_t<sizeof(TEle) == sizeof(i64), i64, conditional_t<sizeof(TEle) == sizeof(i32), zs::i32, conditional_t<sizeof(TEle) == sizeof(i16), i64, int>>>;
    static_assert(sizeof(Ti) == sizeof(TEle), "(reinterpreted) integer type should have the same size as element type.");

    ;
}

}