#pragma once
#include <array>
#include <vector>

//
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"

namespace zs {

  template <typename T, int dim, typename Tn = int, int dimE = dim + 1> struct Mesh {
    using Node = std::array<T, dim>;
    using Elem = std::array<Tn, dimE>;
    std::vector<Node> nodes;
    std::vector<Elem> elems;
  };

  ZPC_API void compute_mesh_normal(const Mesh<float, 3, int, 3> &, float,
                                   std::vector<std::array<float, 3>> &);
  ZPC_API void compute_mesh_normal(const Mesh<float, 3, u32, 3> &, float,
                                   std::vector<std::array<float, 3>> &);
  ZPC_API void compute_mesh_normal(const Mesh<float, 3, int, 3> &, float, Vector<vec<float, 3>> &);
  ZPC_API void compute_mesh_normal(const Mesh<float, 3, u32, 3> &, float, Vector<vec<float, 3>> &);

#if 0
  template <typename T, typename Ti, template <typename> class VectorT, typename ValueT>
  void compute_mesh_normal(const Mesh<T, /*dim*/ 3, Ti, /*codim*/ 3> &surfs, float scale,
                           VectorT<ValueT> &nrms);

  extern template void compute_mesh_normal<float, int, std::vector, std::array<float, 3>>(
      const Mesh<float, 3, int, 3> &, float, std::vector<std::array<float, 3>> &);
  extern template void compute_mesh_normal<float, u32, std::vector, std::array<float, 3>>(
      const Mesh<float, 3, u32, 3> &, float, std::vector<std::array<float, 3>> &);
  extern template void compute_mesh_normal<float, int, Vector, vec<float, 3>>(
      const Mesh<float, 3, int, 3> &, float, Vector<vec<float, 3>> &);
  extern template void compute_mesh_normal<float, u32, Vector, vec<float, 3>>(
      const Mesh<float, 3, u32, 3> &, float, Vector<vec<float, 3>> &);
#endif

}  // namespace zs
