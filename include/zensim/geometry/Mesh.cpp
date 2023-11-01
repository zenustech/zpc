#include "Mesh.hpp"

#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#if ZS_ENABLE_OPENMP
#  include "zensim/omp/execution/ExecutionPolicy.hpp"
#endif

namespace zs {

  template <typename T, typename Ti, typename VectorT>
  static void compute_mesh_normal_impl(const Mesh<T, /*dim*/ 3, Ti, /*codim*/ 3> &surfs,
                                       float scale, VectorT &nrms) {
    using ValueT = typename VectorT::value_type;
    static_assert(std::is_arithmetic_v<T>, "input mesh value type should be an arithmetic type");
    static_assert(
        sizeof(T) * 3 == sizeof(ValueT) && std::alignment_of_v<ValueT> == std::alignment_of_v<T>,
        "normal value type not a match for the input mesh");

    auto &pos = surfs.nodes;
    auto &tris = surfs.elems;

    std::memset(nrms.data(), 0, sizeof(ValueT) * nrms.size());

#if ZS_ENABLE_OPENMP
    constexpr auto space = execspace_e::openmp;
    auto pol = omp_exec();
#else
    constexpr auto space = execspace_e::host;
    auto pol = seq_exec();
#endif

    pol(tris, [&pos, &nrms, execTag = wrapv<space>{}](const std::array<Ti, 3> &tri) {
      const auto &a = pos[tri[0]];
      const auto &b = pos[tri[1]];
      const auto &c = pos[tri[2]];
      zs::vec<T, 3> e0{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
      zs::vec<T, 3> e1{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
      auto n = cross(e0, e1);

      for (int j = 0; j != 3; ++j) {
        auto &n_i = nrms[tri[j]];
        for (int d = 0; d != 3; ++d) atomic_add(execTag, &n_i[d], n[d]);
      }
    });

    pol(nrms, [scale](auto &nrm) {
      auto res = zs::vec<T, 3>{nrm[0], nrm[1], nrm[2]}.normalized() * scale;
      nrm[0] = res[0];
      nrm[1] = res[1];
      nrm[2] = res[2];
    });
  };

  void compute_mesh_normal(const Mesh<float, 3, int, 3> &surfs, float scale,
                           std::vector<std::array<float, 3>> &nrms) {
    compute_mesh_normal_impl<float, int, std::vector<std::array<float, 3>>>(surfs, scale, nrms);
  }
  void compute_mesh_normal(const Mesh<float, 3, u32, 3> &surfs, float scale,
                           std::vector<std::array<float, 3>> &nrms) {
    compute_mesh_normal_impl<float, u32, std::vector<std::array<float, 3>>>(surfs, scale, nrms);
  }
  void compute_mesh_normal(const Mesh<float, 3, int, 3> &surfs, float scale,
                           Vector<vec<float, 3>> &nrms) {
    compute_mesh_normal_impl<float, int, Vector<vec<float, 3>>>(surfs, scale, nrms);
  }
  void compute_mesh_normal(const Mesh<float, 3, u32, 3> &surfs, float scale,
                           Vector<vec<float, 3>> &nrms) {
    compute_mesh_normal_impl<float, u32, Vector<vec<float, 3>>>(surfs, scale, nrms);
  }

#if 0
  template void compute_mesh_normal<float, int, std::vector, std::array<float, 3>>(
      const Mesh<float, 3, int, 3> &, float, std::vector<std::array<float, 3>> &);
  template void compute_mesh_normal<float, u32, std::vector, std::array<float, 3>>(
      const Mesh<float, 3, u32, 3> &, float, std::vector<std::array<float, 3>> &);
  template void compute_mesh_normal<float, int, Vector, vec<float, 3>>(
      const Mesh<float, 3, int, 3> &, float, Vector<vec<float, 3>> &);
  template void compute_mesh_normal<float, u32, Vector, vec<float, 3>>(
      const Mesh<float, 3, u32, 3> &, float, Vector<vec<float, 3>> &);
#endif

}  // namespace zs