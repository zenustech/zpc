#pragma once
#include <array>
#include <vector>

//
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"

namespace zs {

  template <typename T, int dim_, typename Tn = int, int dimE = dim_ + 1> struct Mesh {
    using value_type = T;
    static constexpr int dim = dim_;
    using index_type = Tn;
    static constexpr int dim_elem = dimE;

    using Node = std::array<T, dim>;
    using Elem = std::array<Tn, dimE>;
    using UV = std::array<float, 2>;
    using Norm = std::array<float, 3>;
    using Tan = std::array<float, 3>;
    using Color = std::array<float, 3>;
    using TexId = std::array<int, 2>;  // <type, element id>
    std::vector<Node> nodes;
    std::vector<Tn> vids;  // splitted-point-id to vertex-id
    std::vector<Elem> elems;
    std::vector<UV> uvs;
    std::vector<Norm> norms;
    std::vector<Tan> tans;
    std::vector<Color> colors;
    std::vector<TexId> texids;
    std::string texturePath;

    inline void clear() {
      nodes.clear();
      vids.clear();
      elems.clear();
      uvs.clear();
      norms.clear();
      tans.clear();
      colors.clear();
      texids.clear();
      texturePath.clear();
    }
  };

  ZPC_API void compute_mesh_normal(const Mesh<float, 3, int, 3> &, float,
                                   std::vector<std::array<float, 3>> &);
  ZPC_API void compute_mesh_normal(const Mesh<float, 3, u32, 3> &, float,
                                   std::vector<std::array<float, 3>> &);
  ZPC_API void compute_mesh_normal(const Mesh<float, 3, int, 3> &, float, Vector<vec<float, 3>> &);
  ZPC_API void compute_mesh_normal(const Mesh<float, 3, u32, 3> &, float, Vector<vec<float, 3>> &);
  ZPC_API void compute_mesh_normal(const std::vector<std::array<float, 3>> &nodes,
                                   size_t nodeOffset, size_t nodeNum,
                                   const std::vector<std::array<u32, 3>> &tris, size_t triOffset,
                                   size_t triNum, std::array<float, 3> *nrms);

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
