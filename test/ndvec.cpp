#include <limits>
#include <random>

#include "zensim/math/Vec.h"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/math/matrix/SVD.hpp"
#include "zensim/zpc_tpls/fmt/color.h"

int main() {
  using namespace zs;
  using mat2 = vec<double, 2, 2>;
  using vec2 = vec<double, 2>;
  using mat3 = vec<double, 3, 3>;
  using vec3 = vec<double, 3>;
  using mat4 = vec<double, 4, 4>;
  using mat12 = vec<double, 12, 12>;

  std::random_device rd;
  std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(1, 100);
  ///
  auto genMat = [&distrib, &gen](auto &mat) {
    using Mat = RM_CVREF_T(mat);
    do {
      for (int i = 0; i != Mat::extent; ++i) mat.val(i) = distrib(gen) / 50.f;
    } while (determinant(mat) == 0);
  };
  ///
  auto isMatIdentity = [](const auto &mat) {
    using Mat = remove_cvref_t<decltype(mat)>;
    static_assert(
        Mat::dim == 2 && Mat::template range_t<0>::value == Mat::template range_t<1>::value, "???");
    constexpr auto M = Mat::template range_t<0>::value;
    for (int i = 0; i != M; ++i)
      for (int j = 0; j != M; ++j) {
        if (i != j) {
          if (std::abs(mat(i, j)) > std::numeric_limits<float>::epsilon()) return false;
        } else {
          if (std::abs(mat(i, j) - 1) > std::numeric_limits<float>::epsilon()) return false;
        }
      }
    return true;
  };
  auto isMatNear = [](const auto &A, const auto &B) {
    using MatA = remove_cvref_t<decltype(A)>;
    using MatB = RM_CVREF_T(B);
    using Mat = MatA;
    static_assert(is_same_v<MatA, MatB> && Mat::dim == 2
                      && Mat::template range_t<0>::value == Mat::template range_t<1>::value,
                  "???");
    constexpr auto M = Mat::template range_t<0>::value;
    for (int i = 0; i != M; ++i)
      for (int j = 0; j != M; ++j) {
        if (std::abs(A(i, j) - B(i, j)) > std::numeric_limits<float>::epsilon()) return false;
      }
    return true;
  };
  ///
  auto printVec = [](auto &&v, std::string msg = "") {
    using Vec = remove_cvref_t<decltype(v)>;
    static_assert(Vec::dim == 1, "???");
    if (!msg.empty()) fmt::print("## msg: {}\n", msg);
    constexpr auto N = Vec::extent;
    for (int i = 0; i != N; ++i) fmt::print("{:.6f} ", v(i));
    fmt::print("\n");
  };
  ///
  auto printMat = [](auto &&mat, std::string msg = "") {
    using Mat = remove_cvref_t<decltype(mat)>;
    static_assert(
        Mat::dim == 2 && Mat::template range_t<0>::value == Mat::template range_t<1>::value, "???");
    constexpr auto M = Mat::template range_t<0>::value;

    if (!msg.empty()) fmt::print("## msg: {}\n", msg);
    if constexpr (M == 2)
      fmt::print("mat2[{}] (det: {})==\n{}, {}\n{}, {}\n", (void *)&mat, determinant(mat),
                 mat(0, 0), mat(0, 1), mat(1, 0), mat(1, 1));
    else if constexpr (M == 3)
      fmt::print("mat3[{}] (det: {})==\n{}, {}, {}\n{}, {}, {}\n{}, {}, {}\n", (void *)&mat,
                 determinant(mat), mat(0, 0), mat(0, 1), mat(0, 2), mat(1, 0), mat(1, 1), mat(1, 2),
                 mat(2, 0), mat(2, 1), mat(2, 2));
    else if constexpr (M == 4)
      fmt::print(
          "mat4[{}] (det: {})==\n{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, "
          "{}\n",
          (void *)&mat, determinant(mat), mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3), mat(1, 0),
          mat(1, 1), mat(1, 2), mat(1, 3), mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3), mat(3, 0),
          mat(3, 1), mat(3, 2), mat(3, 3));
    else {
      fmt::print("mat[{}]==\n", (void *)&mat);
      for (int i = 0; i != M; ++i) {
        for (int j = 0; j != M; ++j) fmt::print("{:.6f} ", mat(i, j));
        fmt::print("\n");
      }
    }
  };
  ///
  /// inversion
  ///
  auto m3_0 = mat3::identity();
  auto m4_0 = mat4::identity();
  fmt::print(fg(fmt::color::yellow), "mat3 inv\n");
  genMat(m3_0);
  printMat(m3_0);
  auto m3_1 = inverse(m3_0);
  printMat(m3_1);
  // fmt::print("{} * {}\n", demangle(m3_0), demangle(m3_1));
  auto m3_2 = (m3_1 * m3_0);
  printMat(m3_2);
  if (!isMatIdentity(m3_2)) throw std::runtime_error("3x3 inversion test failed");

  fmt::print(fg(fmt::color::yellow), "mat4 inv\n");
  genMat(m4_0);
  printMat(m4_0);
  auto m4_1 = inverse(m4_0);
  printMat(m4_1);
  auto m4_2 = (m4_0 * m4_1);
  printMat(m4_2);
  if (!isMatIdentity(m4_2)) throw std::runtime_error("4x4 inversion test failed");

  // 2d
  auto m2 = mat2::identity();
  fmt::print(fg(fmt::color::yellow), "mat2 QR\n");
  genMat(m2);
  printMat(m2, "qrsvd-M2");
  {
    auto [R2, S2] = math::polar_decomposition(m2);
    printMat(R2, "polar-R");
    printMat(S2, "polar-S");

    auto m2Chk = R2 * S2;
    printMat(m2Chk, "m2Chk");
    if (!isMatNear(m2, m2Chk)) throw std::runtime_error("2x2 qr test failed");
  }
  fmt::print(fg(fmt::color::yellow), "mat2 SVD\n");
  {
    auto [U, S, V] = math::qr_svd(m2);
    printMat(U, "qrsvd-U");
    fmt::print("qrsvd-S: \n[{}, {}]\n", S(0), S(1));
    printMat(V, "qrsvd-V");

    auto m2Chk = diag_mul(U, S) * V.transpose();
    printMat(m2Chk, "m2Chk");
    if (!isMatNear(m2, m2Chk)) throw std::runtime_error("2x2 qr svd test failed");
  }
  // 3d
  fmt::print(fg(fmt::color::yellow), "mat3 QR\n");
  auto m3 = mat3::identity();
  m3 = {0.6000000238418579,
        0.6000000238418579,
        0.699999988079071,
        0.699999988079071,
        0.699999988079071,
        0.4000000059604645,
        0.5,
        0.30000001192092896,
        0.5};
  // genMat(m3);
  {
    auto [R3, S3] = math::polar_decomposition(m3);
    printMat(R3, "polar-R");
    printMat(S3, "polar-S");

    auto m3Chk = R3 * S3;
    printMat(m3Chk, "m3Chk");
    if (!isMatNear(m3, m3Chk)) throw std::runtime_error("3x3 qr test failed");
  }
  {
    auto [U, S, V] = math::svd(m3);
    fmt::print("jacobi svd {}, {}, {}\n", S(0), S(1), S(2));
    auto m3Chk = diag_mul(U, S) * V.transpose();
    printMat(m3Chk, "m3Chk-jacobi_svd");
    if (!isMatNear(m3, m3Chk)) throw std::runtime_error("3x3 jacobi svd test failed");
  }

  // 3d qrsvd
  {
    auto [U, S, V] = math::qr_svd(m3);
    fmt::print("qrsvd {}, {}, {}\n", S(0), S(1), S(2));
    auto m3Chk = diag_mul(U, S) * V.transpose();
    printMat(m3Chk, "m3Chk-qrsvd");
    if (!isMatNear(m3, m3Chk)) throw std::runtime_error("3x3 qr svd test failed");
  }
  return 0;
}