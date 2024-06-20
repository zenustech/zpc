#pragma once
#include "Vec.h"
#include "zensim/math/MathUtils.h"

namespace zs {

  enum euler_angle_convention_e { roe = 0, ypr };
  constexpr auto roe_c = wrapv<euler_angle_convention_e::roe>{};
  constexpr auto ypr_c = wrapv<euler_angle_convention_e::ypr>{};

  enum angle_unit_e { radian = 0, degree };
  constexpr auto radian_c = wrapv<angle_unit_e::radian>{};
  constexpr auto degree_c = wrapv<angle_unit_e::degree>{};
  /**
     Imported from ZIRAN, wraps Eigen's Rotation2D (in 2D) and Quaternion (in 3D).
   */
  template <typename T = float, int dim_ = 3> struct Rotation : vec<T, dim_, dim_> {
    // 3d rotation can be viewed as a series of three successive rotations about coordinate axes
    using value_type = T;
    static constexpr int dim = dim_;  // do not let TM's dim fool you!
    using TV = vec<value_type, dim>;
    using TM = vec<value_type, dim, dim>;

    constexpr auto &self() noexcept { return static_cast<TM &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const TM &>(*this); }

    constexpr Rotation() noexcept : TM{TM::identity()} {}

    constexpr Rotation(const TM &m) noexcept : TM{m} {}

    constexpr Rotation &operator=(const TM &o) noexcept {
      // Rotation tmp{o};
      // std::swap(*this, tmp);
      this->self() = o.self();
      return *this;
    }

    template <auto d = dim, enable_if_t<d == 2> = 0> constexpr Rotation(value_type theta) noexcept
        : TM{TM::identity()} {
      value_type sinTheta = zs::sin(theta);
      value_type cosTheta = zs::cos(theta);
      (*this)(0, 0) = cosTheta;
      (*this)(0, 1) = -sinTheta;
      (*this)(1, 0) = sinTheta;
      (*this)(1, 1) = cosTheta;
    }
    /// axis + rotation
    template <typename VecT, auto unit = angle_unit_e::radian, auto d = dim,
              enable_if_all<d == 3, std::is_convertible_v<typename VecT::value_type, T>,
                            VecT::dim == 1, VecT::template range_t<0>::value == 3>
              = 0>
    constexpr Rotation(const VecInterface<VecT> &p_, value_type alpha, wrapv<unit> = {}) noexcept
        : TM{} {
      if constexpr (unit == angle_unit_e::degree) alpha *= ((value_type)g_pi / (value_type)180);
      auto p = p_.normalized();
      TM P{0, p(2), -p(1), -p(2), 0, p(0), p(1), -p(0), 0};
      value_type sinAlpha = zs::sin(alpha);
      value_type cosAlpha = zs::cos(alpha);
      self() = cosAlpha * TM::identity() + (1 - cosAlpha) * dyadic_prod(p, p) - sinAlpha * P;
    }
    template <auto unit = angle_unit_e::radian, auto d = dim, enable_if_t<d == 3> = 0>
    constexpr auto extractAxisRotation(wrapv<unit> = {}) const noexcept {
      const auto cosAlpha = (value_type)0.5 * (trace(self()) - 1);
      value_type alpha = zs::acos(cosAlpha);
      if (math::near_zero(cosAlpha - 1)) return zs::make_tuple(TV{0, 1, 0}, (value_type)0);

      TV p{};
      if (math::near_zero(cosAlpha + 1)) {
        p(0) = math::sqrtNewtonRaphson(((*this)(0, 0) + 1) * (value_type)0.5);
        if (math::near_zero(p(0))) {
          p(1) = math::sqrtNewtonRaphson(((*this)(1, 1) + 1) * (value_type)0.5);
          p(2) = math::sqrtNewtonRaphson(((*this)(2, 2) + 1) * (value_type)0.5);
          if ((*this)(1, 2) < (value_type)0) p(2) = -p(2);
        } else {
          p(1) = (*this)(0, 1) * (value_type)0.5 / p(0);
          p(2) = (*this)(0, 2) * (value_type)0.5 / p(0);
        }
      } else {
        const auto sinAlpha = math::sqrtNewtonRaphson((value_type)1 - cosAlpha * cosAlpha);
        p(0) = ((*this)(2, 1) - (*this)(1, 2)) * (value_type)0.5 / sinAlpha;
        p(1) = ((*this)(0, 2) - (*this)(2, 0)) * (value_type)0.5 / sinAlpha;
        p(2) = ((*this)(1, 0) - (*this)(0, 1)) * (value_type)0.5 / sinAlpha;
      }
      if constexpr (unit == angle_unit_e::radian)
        return zs::make_tuple(p, alpha);
      else if constexpr (unit == angle_unit_e::degree)
        return zs::make_tuple(p, alpha * (value_type)180 / (value_type)g_pi);
    }

    /// euler angles
    template <auto unit = angle_unit_e::radian, auto convention = euler_angle_convention_e::roe,
              auto d = dim, enable_if_t<d == 3> = 0>
    constexpr Rotation(value_type psi, value_type theta, value_type phi, wrapv<unit> = {},
                       wrapv<convention> = {}) noexcept
        : TM{} {
      if constexpr (unit == angle_unit_e::degree) {
        psi *= ((value_type)g_pi / (value_type)180);
        theta *= ((value_type)g_pi / (value_type)180);
        phi *= ((value_type)g_pi / (value_type)180);
      }
      auto sinPsi = zs::sin(psi);
      auto cosPsi = zs::cos(psi);
      auto sinTheta = zs::sin(theta);
      auto cosTheta = zs::cos(theta);
      auto sinPhi = zs::sin(phi);
      auto cosPhi = zs::cos(phi);
      if constexpr (convention == euler_angle_convention_e::roe) {
        // Roe convention (successive rotations)
        // ref: https://www.continuummechanics.org/rotationmatrix.html
        // [z] psi -> [y'] theta -> [z'] phi
        auto cosPsi_cosTheta = cosPsi * cosTheta;
        auto sinPsi_cosTheta = sinPsi * cosTheta;
        (*this)(0, 0) = cosPsi_cosTheta * cosPhi - sinPsi * sinPhi;
        (*this)(0, 1) = -cosPsi_cosTheta * sinPhi - sinPsi * cosPhi;
        (*this)(0, 2) = cosPsi * sinTheta;
        (*this)(1, 0) = sinPsi_cosTheta * cosPhi + cosPsi * sinPhi;
        (*this)(1, 1) = -sinPsi_cosTheta * sinPhi + cosPsi * cosPhi;
        (*this)(1, 2) = sinPsi * sinTheta;
        (*this)(2, 0) = -sinTheta * cosPhi;
        (*this)(2, 1) = sinTheta * sinPhi;
        (*this)(2, 2) = cosTheta;
      } else if constexpr (convention == euler_angle_convention_e::ypr) {
        // navigation (yaw, pitch, roll)
        // axis [x, y, z] = direction [north, east, down] = body [front, right, bottom]
        // ref:
        // http://personal.maths.surrey.ac.uk/T.Bridges/SLOSH/3-2-1-Eulerangles.pdf
        // [z] psi -> [y'] theta -> [x'] phi
        auto sinPhi_sinTheta = sinPhi * sinTheta;
        auto cosPhi_sinTheta = cosPhi * sinTheta;
        (*this)(0, 0) = cosTheta * cosPsi;
        (*this)(0, 1) = cosTheta * sinPsi;
        (*this)(0, 2) = -sinTheta;
        (*this)(1, 0) = sinPhi_sinTheta * cosPsi - cosPhi * sinPsi;
        (*this)(1, 1) = sinPhi_sinTheta * sinPsi + cosPhi * cosPsi;
        (*this)(1, 2) = sinPhi * cosTheta;
        (*this)(2, 0) = cosPhi_sinTheta * cosPsi + sinPhi * sinPsi;
        (*this)(2, 1) = cosPhi_sinTheta * sinPsi - sinPhi * cosPsi;
        (*this)(2, 2) = cosPhi * cosTheta;
      }
    }
    // ref: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    template <auto unit = angle_unit_e::radian, auto convention = euler_angle_convention_e::roe,
              auto d = dim, enable_if_t<d == 3> = 0>
    constexpr auto extractAngles(wrapv<unit> = {}, wrapv<convention> = {}) const noexcept {
      value_type psi{}, theta{}, phi{};
      if constexpr (convention == euler_angle_convention_e::roe) {
        const auto cosTheta = (*this)(2, 2);
        if (math::near_zero(cosTheta - 1)) {
          theta = 0;
          psi = zs::atan2((*this)(1, 0), (*this)(0, 0));
          phi = (value_type)0;
        } else if (math::near_zero(cosTheta + 1)) {
          theta = g_pi;
          psi = zs::atan2(-(*this)(1, 0), -(*this)(0, 0));
          phi = (value_type)0;
        } else {
          theta = zs::acos(cosTheta);  // another solution (-theta)
          /// theta [0, g_pi], thus (sinTheta > 0) always holds true
          psi = zs::atan2((*this)(1, 2), (*this)(0, 2));  // no need to divide sinTheta
          phi = zs::atan2((*this)(2, 1), -(*this)(2, 0));
        }
      } else if constexpr (convention == euler_angle_convention_e::ypr) {
        const auto sinTheta = -(*this)(0, 2);
        if (math::near_zero(sinTheta - 1)) {
          theta = g_half_pi;
          psi = zs::atan2((*this)(2, 1), (*this)(2, 0));
          phi = (value_type)0;
        } else if (math::near_zero(sinTheta + 1)) {
          theta = -g_half_pi;
          psi = zs::atan2(-(*this)(2, 1), -(*this)(2, 0));
          phi = (value_type)0;
        } else {
          theta = std::asin(sinTheta);  // another solution: (g_pi - theta)
          /// theta [-g_pi/2, g_pi/2], thus (cosTheta > 0) always holds true
          psi = std::atan2((*this)(0, 1), (*this)(0, 0));  // no need to divide cosTheta
          phi = std::atan2((*this)(1, 2), (*this)(2, 2));
        }
      }
      if constexpr (unit == angle_unit_e::radian)
        return zs::make_tuple(psi, theta, phi);
      else if constexpr (unit == angle_unit_e::degree)
        return zs::make_tuple(psi * (value_type)180 / (value_type)g_pi,
                              theta * (value_type)180 / (value_type)g_pi,
                              phi * (value_type)180 / (value_type)g_pi);
    }
    template <typename VecT, enable_if_all<std::is_convertible_v<typename VecT::value_type, T>,
                                           VecT::dim == 1, VecT::template range_t<0>::value == 4>
                             = 0>
    constexpr Rotation(const VecInterface<VecT> &q) noexcept : TM{} {
      if constexpr (dim == 2) {
        /// Construct a 2D counter clock wise rotation from the angle \a a in
        /// radian.
        T sinA = zs::sin(q(0)), cosA = zs::cos(q(0));
        (*this)(0, 0) = cosA;
        (*this)(0, 1) = -sinA;
        (*this)(1, 0) = sinA;
        (*this)(1, 1) = cosA;
      } else if constexpr (dim == 3) {
        /// The quaternion is required to be normalized, otherwise the result is
        /// undefined.
        self() = quaternion2matrix(q);
      }
    }
    template <typename VecTA, typename VecTB,
              enable_if_all<std::is_convertible_v<typename VecTA::value_type, value_type>,
                            std::is_convertible_v<typename VecTB::value_type, value_type>,
                            VecTA::dim == 1, VecTB::dim == 1,
                            VecTA::template range_t<0>::value == VecTB::template range_t<0>::value>
              = 0>
    constexpr Rotation(const VecInterface<VecTA> &a, const VecInterface<VecTB> &b) noexcept : TM{} {
      if constexpr (dim == 2 && VecTA::template range_t<0>::value == 2) {
        auto aa = a.normalized();
        auto bb = b.normalized();
        (*this)(0, 0) = aa(0) * bb(0) + aa(1) * bb(1);
        (*this)(0, 1) = -(aa(0) * bb(1) - bb(0) * aa(1));
        (*this)(1, 0) = aa(0) * bb(1) - bb(0) * aa(1);
        (*this)(1, 1) = aa(0) * bb(0) + aa(1) * bb(1);
      } else if constexpr (dim == 3 && VecTA::template range_t<0>::value == 3) {
        T k_cos_theta = a.dot(b);
        T k = zs::sqrt(a.l2NormSqr() * b.l2NormSqr());
        vec<T, 4> q{};
        if (k_cos_theta / k == -1) {
          // 180 degree rotation around any orthogonal vector
          q(3) = 0;
          auto c = a.orthogonal().normalized();
          q(0) = c(0);
          q(1) = c(1);
          q(2) = c(2);
        } else {
          q(3) = k_cos_theta + k;
          auto c = a.cross(b);
          q(0) = c(0);
          q(1) = c(1);
          q(2) = c(2);
          q = q.normalized();
        }
        self() = quaternion2matrix(q);
      }
    }

    template <typename VecT, int d = dim,
              enable_if_all<d == 3, std::is_convertible_v<typename VecT::value_type, T>,
                            VecT::dim == 1, VecT::template range_t<0>::value == 4>
              = 0>
    static constexpr auto quaternionMultiply(const VecInterface<VecT> &q1,
                                             const VecInterface<VecT> &q2) noexcept {
      // VecT res{};
      typename VecT::template variant_vec<typename VecT::value_type, typename VecT::extents> res{};
      const T w1 = q1.w();
      const T x1 = q1.x();
      const T y1 = q1.y();
      const T z1 = q1.z();

      const T w2 = q2.w();
      const T x2 = q2.x();
      const T y2 = q2.y();
      const T z2 = q2.z();

      res.w() = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
      res.x() = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
      res.y() = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
      res.z() = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

      return res;
    }

    template <typename VecT, int d = dim,
              enable_if_all<d == 3, std::is_convertible_v<typename VecT::value_type, T>,
                            VecT::dim == 1, VecT::template range_t<0>::value == 4>
              = 0>
    static constexpr auto quaternionConjugateMultiply(const VecInterface<VecT> &q1,
                                                      const VecInterface<VecT> &q2) noexcept {
      // VecT res{};
      typename VecT::template variant_vec<typename VecT::value_type, typename VecT::extents> res{};
      const T w1 = q1.w();
      const T x1 = -q1.x();
      const T y1 = -q1.y();
      const T z1 = -q1.z();

      const T w2 = q2.w();
      const T x2 = q2.x();
      const T y2 = q2.y();
      const T z2 = q2.z();

      res.w() = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
      res.x() = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
      res.y() = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
      res.z() = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

      return res;
    }
    // template <typename VecT,int d = dim,
    //         enable_if_all<d == 3,std::is_convertible_v<typename VecT::value_type,T>,
    //                     typename VecT::dim == 1,typename VecT::extent == 4> = 0>
    // static constexpr VecT quaternionConjugate(const VecInterface<VecT>& q) noexcept {
    //     auto res = q;
    //     res.x() = -res.x();
    //     res.y() = -res.y();
    //     res.z() = -res.z();

    //     return res;
    // }

    template <typename VecT, int d = dim,
              enable_if_all<d == 3, std::is_convertible_v<typename VecT::value_type, T>,
                            VecT::dim == 1, VecT::template range_t<0>::value == 4>
              = 0>
    static constexpr TM quaternion2matrix(const VecInterface<VecT> &q) noexcept {
      /// (0, 1, 2, 3)
      /// (x, y, z, w)
      const T tx = T(2) * q(0);
      const T ty = T(2) * q(1);
      const T tz = T(2) * q(2);
      const T twx = tx * q(3);
      const T twy = ty * q(3);
      const T twz = tz * q(3);
      const T txx = tx * q(0);
      const T txy = ty * q(0);
      const T txz = tz * q(0);
      const T tyy = ty * q(1);
      const T tyz = tz * q(1);
      const T tzz = tz * q(2);
      TM rot{};
      rot(0, 0) = T(1) - (tyy + tzz);
      rot(0, 1) = txy - twz;
      rot(0, 2) = txz + twy;
      rot(1, 0) = txy + twz;
      rot(1, 1) = T(1) - (txx + tzz);
      rot(1, 2) = tyz - twx;
      rot(2, 0) = txz - twy;
      rot(2, 1) = tyz + twx;
      rot(2, 2) = T(1) - (txx + tyy);
      return rot;
    }
  };

  template <class T, int dim> struct AngularVelocity;

  template <class T> struct AngularVelocity<T, 2> {
    using TV = vec<T, 2>;
    T omega{0};
    constexpr AngularVelocity operator+(const AngularVelocity &o) const noexcept {
      return AngularVelocity{omega + o.omega};
    }
    constexpr AngularVelocity operator*(T alpha) const noexcept {
      return AngularVelocity{omega * alpha};
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == 2> = 0>
    constexpr auto cross(const VecInterface<VecT> &x) const noexcept {
      return TV{-omega * x(1), omega * x(0)};
    }
  };

  template <class T> struct AngularVelocity<T, 3> {
    using TV = vec<T, 3>;
    TV omega{0, 0, 0};
    constexpr AngularVelocity operator+(const AngularVelocity &o) const noexcept {
      return AngularVelocity{omega + o.omega};
    }
    constexpr AngularVelocity operator*(T alpha) const noexcept {
      return AngularVelocity{omega * alpha};
    }
    friend constexpr AngularVelocity operator*(T alpha, const AngularVelocity &o) noexcept {
      return AngularVelocity{o.omega * alpha};
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == 3> = 0>
    constexpr auto cross(const VecInterface<VecT> &x) const noexcept {
      return omega.cross(x);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == 3> = 0>
    friend constexpr auto cross(const VecInterface<VecT> &x, const AngularVelocity &o) noexcept {
      return x.cross(o.omega);
    }
  };

}  // namespace zs
