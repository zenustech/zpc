#pragma once
#include "zensim/math/VecInterface.hpp"

/// ref: bow/codim-ipc
namespace zs {

  /**
   *             v1 --- v3
   *            /  \    /
   *           /    \  /
   *          v2 --- v0
   * \param branch 0: the angle is in (-pi, pi); +1: the angle is in (0, 2pi); -1: the angle is in
   * (-2pi, 0)
   */
  template <typename VecT, execspace_e space = deduce_execution_space(),
            enable_if_all<VecT::dim == 1, VecT::extent == 3,
                          std::is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto dihedral_angle(const VecInterface<VecT>& v2, const VecInterface<VecT>& v0,
                                const VecInterface<VecT>& v1, const VecInterface<VecT>& v3,
                                wrapv<space> tag = {}) {
    using T = typename VecT::value_type;
    auto n1 = (v0 - v2).cross(v1 - v2);
    auto n2 = (v1 - v3).cross(v0 - v3);
    T DA = zs::acos(
        zs::max((T)-1, zs::min((T)1, n1.dot(n2) / zs::sqrt(n1.l2NormSqr() * n2.l2NormSqr(), tag))));
    if (n2.cross(n1).dot(v0 - v1) < 0) DA = -DA;
    return DA;
  }

  /**
   *             v1 --- v3
   *            /  \    /
   *           /    \  /
   *          v2 --- v0
   */
  template <typename VecT, execspace_e space = deduce_execution_space(),
            enable_if_all<VecT::dim == 1, VecT::extent == 3,
                          std::is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto dihedral_angle_gradient(const VecInterface<VecT>& v2, const VecInterface<VecT>& v0,
                                         const VecInterface<VecT>& v1, const VecInterface<VecT>& v3,
                                         wrapv<space> tag = {}) {
    using T = typename VecT::value_type;
    using GradT =
        typename VecT::template variant_vec<T, integer_seq<typename VecT::index_type, 12>>;
    // here we map our v order to rusmas' in this function for implementation convenience
    auto e0 = v1 - v0;
    auto e1 = v2 - v0;
    auto e2 = v3 - v0;
    auto e3 = v2 - v1;
    auto e4 = v3 - v1;
    auto n1 = e0.cross(e1);
    auto n2 = e2.cross(e0);
    T n1SqNorm = n1.l2NormSqr();
    T n2SqNorm = n2.l2NormSqr();
    T e0norm = e0.norm(tag);
    auto da_dv2 = -e0norm / n1SqNorm * n1;
    auto da_dv0 = -e0.dot(e3) / (e0norm * n1SqNorm) * n1 - e0.dot(e4) / (e0norm * n2SqNorm) * n2;
    auto da_dv1 = e0.dot(e1) / (e0norm * n1SqNorm) * n1 + e0.dot(e2) / (e0norm * n2SqNorm) * n2;
    auto da_dv3 = -e0norm / n2SqNorm * n2;

    GradT grad{};
    for (int d = 0; d != 3; ++d) {
      grad(0 * 3 + d) = da_dv2[d];
      grad(1 * 3 + d) = da_dv0[d];
      grad(2 * 3 + d) = da_dv1[d];
      grad(3 * 3 + d) = da_dv3[d];
    }
    return grad;
  }

  /**
   *             v1 --- v3
   *            /  \    /
   *           /    \  /
   *          v2 --- v0
   */
  template <typename VecT, execspace_e space = deduce_execution_space(),
            enable_if_all<VecT::dim == 1, VecT::extent == 3,
                          std::is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto dihedral_angle_hessian(const VecInterface<VecT>& v2, const VecInterface<VecT>& v0,
                                        const VecInterface<VecT>& v1, const VecInterface<VecT>& v3,
                                        wrapv<space> tag = {}) {
    using T = typename VecT::value_type;
    using HessT =
        typename VecT::template variant_vec<T, integer_seq<typename VecT::index_type, 12, 12>>;
    using TV = typename VecT::template variant_vec<T, typename VecT::extents>;
    /// @note
    /// https://studios.disneyresearch.com/wp-content/uploads/2019/03/Discrete-Bending-Forces-and-Their-Jacobians-Paper.pdf
    TV e[5] = {v1 - v0, v2 - v0, v3 - v0, v2 - v1, v3 - v1};
    T norm_e[5] = {e[0].norm(tag), e[1].norm(tag), e[2].norm(tag), e[3].norm(tag), e[4].norm(tag)};
    auto n1 = e[0].cross(e[1]);
    auto n2 = e[2].cross(e[0]);
    T n1norm = n1.norm(tag);
    T n2norm = n2.norm(tag);
    auto compute_mHat = [tag](const auto& xp, const auto& xe0, const auto& xe1) {
      auto e = xe1 - xe0;
      auto mHat = xe0 + (xp - xe0).dot(e) / e.l2NormSqr() * e - xp;
      mHat /= mHat.norm(tag);
      return mHat;
    };
    auto mHat1 = compute_mHat(v1, v0, v2);
    auto mHat2 = compute_mHat(v1, v0, v3);
    auto mHat3 = compute_mHat(v0, v1, v2);
    auto mHat4 = compute_mHat(v0, v1, v3);
    auto mHat01 = compute_mHat(v2, v0, v1);
    auto mHat02 = compute_mHat(v3, v0, v1);
    T cosalpha1 = e[0].dot(e[1]) / (norm_e[0] * norm_e[1]);
    T cosalpha2 = e[0].dot(e[2]) / (norm_e[0] * norm_e[2]);
    T cosalpha3 = -e[0].dot(e[3]) / (norm_e[0] * norm_e[3]);
    T cosalpha4 = -e[0].dot(e[4]) / (norm_e[0] * norm_e[4]);
    T h1 = n1norm / norm_e[1];
    T h2 = n2norm / norm_e[2];
    T h3 = n1norm / norm_e[3];
    T h4 = n2norm / norm_e[4];
    T h01 = n1norm / norm_e[0];
    T h02 = n2norm / norm_e[0];

    auto N1_01 = dyadic_prod(n1, (mHat01 / (h01 * h01 * n1norm)));
    auto N2_02 = dyadic_prod(n2, (mHat02 / (h02 * h02 * n2norm)));
    auto N1_3 = dyadic_prod(n1, (mHat3 / (h01 * h3 * n1norm)));
    auto N1_1 = dyadic_prod(n1, (mHat1 / (h01 * h1 * n1norm)));
    auto N2_4 = dyadic_prod(n2, (mHat4 / (h02 * h4 * n2norm)));
    auto N2_2 = dyadic_prod(n2, (mHat2 / (h02 * h2 * n2norm)));
    auto M3_01_1 = dyadic_prod((cosalpha3 / (h3 * h01 * n1norm) * mHat01), n1);
    auto M1_01_1 = dyadic_prod((cosalpha1 / (h1 * h01 * n1norm) * mHat01), n1);
    auto M1_1_1 = dyadic_prod((cosalpha1 / (h1 * h1 * n1norm) * mHat1), n1);
    auto M3_3_1 = dyadic_prod((cosalpha3 / (h3 * h3 * n1norm) * mHat3), n1);
    auto M3_1_1 = dyadic_prod((cosalpha3 / (h3 * h1 * n1norm) * mHat1), n1);
    auto M1_3_1 = dyadic_prod((cosalpha1 / (h1 * h3 * n1norm) * mHat3), n1);
    auto M4_02_2 = dyadic_prod((cosalpha4 / (h4 * h02 * n2norm) * mHat02), n2);
    auto M2_02_2 = dyadic_prod((cosalpha2 / (h2 * h02 * n2norm) * mHat02), n2);
    auto M4_4_2 = dyadic_prod((cosalpha4 / (h4 * h4 * n2norm) * mHat4), n2);
    auto M2_4_2 = dyadic_prod((cosalpha2 / (h2 * h4 * n2norm) * mHat4), n2);
    auto M4_2_2 = dyadic_prod((cosalpha4 / (h4 * h2 * n2norm) * mHat2), n2);
    auto M2_2_2 = dyadic_prod((cosalpha2 / (h2 * h2 * n2norm) * mHat2), n2);
    auto B1 = dyadic_prod(n1, mHat01) / (norm_e[0] * norm_e[0] * n1norm);
    auto B2 = dyadic_prod(n2, mHat02) / (norm_e[0] * norm_e[0] * n2norm);

    auto hess = HessT::zeros();
    auto assignSubMat = [&hess](int baseI, int baseJ, const auto& m) {
      for (int i = 0; i != 3; ++i)
        for (int j = 0; j != 3; ++j) {
          hess(baseI + i, baseJ + j) = m(i, j);
        }
    };
    auto H00 = -(N1_01 + N1_01.transpose());
    assignSubMat(0, 0, H00);

    auto H10 = M3_01_1 - N1_3;
    assignSubMat(3, 0, H10);
    assignSubMat(0, 3, H10.transpose());

    auto H20 = M1_01_1 - N1_1;
    assignSubMat(6, 0, H20);
    assignSubMat(0, 6, H20.transpose());

    auto H11 = M3_3_1 + M3_3_1.transpose() - B1 + M4_4_2 + M4_4_2.transpose() - B2;
    assignSubMat(3, 3, H11);

    auto H12 = M3_1_1 + M1_3_1.transpose() + B1 + M4_2_2 + M2_4_2.transpose() + B2;
    assignSubMat(3, 6, H12);
    assignSubMat(6, 3, H12.transpose());

    auto H13 = M4_02_2 - N2_4;
    assignSubMat(3, 9, H13);
    assignSubMat(9, 3, H13.transpose());

    auto H22 = M1_1_1 + M1_1_1.transpose() - B1 + M2_2_2 + M2_2_2.transpose() - B2;
    assignSubMat(6, 6, H22);

    auto H23 = M2_02_2 - N2_2;
    assignSubMat(6, 9, H23);
    assignSubMat(9, 6, H23.transpose());

    auto H33 = -(N2_02 + N2_02.transpose());
    assignSubMat(9, 9, H33);

    return hess;
  }

}  // namespace zs