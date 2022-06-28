#pragma once

#include "Predicates.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/math/Vec.h"

namespace zs {

  /// ref: libigl
  enum orientation_e : int {
    POSITIVE = 1,
    INSIDE = 1,
    NEGATIVE = -1,
    OUTSIDE = -1,
    COLLINEAR = 0,
    COPLANAR = 0,
    COCIRCULAR = 0,
    COSPHERICAL = 0,
    DEGENERATE = 0
  };

  template <typename VecT, enable_if_all<std::is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 2> = 0>
  constexpr orientation_e orient2d(const VecInterface<VecT>& pa, const VecInterface<VecT>& pb,
                                   const VecInterface<VecT>& pc) noexcept {
    double a[] = {pa[0], pa[1], pa[2]};
    double b[] = {pb[0], pb[1], pb[2]};
    double c[] = {pc[0], pc[1], pc[2]};
    const auto r = orient2d(a, b, c);
    if (r > 0)
      return orientation_e::POSITIVE;
    else if (r < 0)
      return orientation_e::NEGATIVE;
    else
      return orientation_e::COLLINEAR;
  }

  template <typename VecT, enable_if_all<std::is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3> = 0>
  constexpr orientation_e orient3d(const VecInterface<VecT>& pa, const VecInterface<VecT>& pb,
                                   const VecInterface<VecT>& pc,
                                   const VecInterface<VecT>& pd) noexcept {
    double a[] = {pa[0], pa[1], pa[2]};
    double b[] = {pb[0], pb[1], pb[2]};
    double c[] = {pc[0], pc[1], pc[2]};
    double d[] = {pd[0], pd[1], pd[2]};
    const auto r = orient3d(a, b, c, d);
    if (r > 0)
      return orientation_e::POSITIVE;
    else if (r < 0)
      return orientation_e::NEGATIVE;
    else
      return orientation_e::COPLANAR;
  }

  /// ref: ExactRootParityCCD
  // Wang Bolun, Zachary Ferguson
  // bilinear
  struct bilinear {
    using ivec3 = zs::vec<int, 3>;
    using vec3 = zs::vec<double, 3>;

    // v0, v1 are vertices of one triangle, v2, v3 are the vertices of another one.
    template <typename VecT, enable_if_all<std::is_floating_point_v<typename VecT::value_type>,
                                           VecT::dim == 1, VecT::extent == 3> = 0>
    constexpr bilinear(const VecInterface<VecT>& v0, const VecInterface<VecT>& v1,
                       const VecInterface<VecT>& v2, const VecInterface<VecT>& v3) noexcept
        : v{v0.clone(), v1.clone(), v2.clone(), v3.clone()}, phi_f{2, 2} {
      int ori = orient3d(v0, v1, v2, v3);
      is_degenerated = ori == 0 ? true : false;
      if (ori >= 0) {
        facets[0] = ivec3{1, 2, 0};
        facets[1] = ivec3{3, 0, 2};
        facets[2] = ivec3{0, 3, 1};
        facets[3] = ivec3{2, 1, 3};
      } else if (ori == -1) {
        facets[0] = ivec3{1, 0, 2};
        facets[1] = ivec3{3, 2, 0};
        facets[2] = ivec3{0, 1, 3};
        facets[3] = ivec3{2, 3, 1};
      }
    }

    bool is_degenerated;
    ivec3 facets[4];
    int phi_f[2];
    vec3 v[4];
  };

  // prism
  struct prism {
    using ivec2 = zs::vec<int, 2>;
    using vec2 = zs::vec<double, 2>;
    using vec3 = zs::vec<double, 3>;
    using bv_t = AABBBox<3, double>;

    template <typename VecT, enable_if_all<std::is_floating_point_v<typename VecT::value_type>,
                                           VecT::dim == 1, VecT::extent == 3> = 0>
    constexpr prism(const VecInterface<VecT>& vs, const VecInterface<VecT>& fs0,
                    const VecInterface<VecT>& fs1, const VecInterface<VecT>& fs2,
                    const VecInterface<VecT>& ve, const VecInterface<VecT>& fe0,
                    const VecInterface<VecT>& fe1, const VecInterface<VecT>& fe2) noexcept
        : prism_edge_id{ivec2{0, 1}, ivec2{1, 2}, ivec2{2, 0}, ivec2{3, 4}, ivec2{4, 5},
                        ivec2{5, 3}, ivec2{0, 3}, ivec2{1, 4}, ivec2{2, 5}} {
      p_vertices[0] = vs - fs0;
      p_vertices[1] = vs - fs2;
      p_vertices[2] = vs - fs1;
      p_vertices[3] = ve - fs0;
      p_vertices[4] = ve - fs2;
      p_vertices[5] = ve - fs1;
    }

    template <typename T> constexpr bool isPrismBboxCutBbox(const T& min, const T& max) const {
      bv_t bv = bv_t{get_bounding_box(p_vertices[0], p_vertices[1])};
      for (int i = 2; i != 6; ++i) merge(bv, p_vertices[i]);
      return overlaps(bv, bv_t{min, max});
    }

    // 0 means up, 1 means bottom
    constexpr bool isTriangleDegenerated(const int up_or_bottom) const noexcept {
      int pid = up_or_bottom == 0 ? 0 : 3;
      double r
          = ((p_vertices[pid] - p_vertices[pid + 1]).cross(p_vertices[pid] - p_vertices[pid + 2]))
                .norm();
      if (absolute(r) > 1e-8) return false;
      int ori{};
      vec2 p[3] = {};
      const auto to_2d = [](const vec3& p, int t) { return vec2{p[(t + 1) % 3], p[(t + 2) % 3]}; };
      for (int j = 0; j < 3; j++) {
        p[0] = to_2d(p_vertices[pid], j);
        p[1] = to_2d(p_vertices[pid + 1], j);
        p[2] = to_2d(p_vertices[pid + 2], j);

        ori = orient2d(p[0], p[1], p[2]);
        if (ori != 0) return false;
      }
      return true;
    }

    ivec2 prism_edge_id[9];
    vec3 p_vertices[6];
  };

  struct hex {
    using ivec2 = zs::vec<int, 2>;
    using vec2 = zs::vec<double, 2>;
    using vec3 = zs::vec<double, 3>;
    using bv_t = AABBBox<3, double>;

    template <typename VecT, enable_if_all<std::is_floating_point_v<typename VecT::value_type>,
                                           VecT::dim == 1, VecT::extent == 3> = 0>
    constexpr hex(const VecInterface<VecT>& a0, const VecInterface<VecT>& a1,
                  const VecInterface<VecT>& b0, const VecInterface<VecT>& b1,
                  const VecInterface<VecT>& a0b, const VecInterface<VecT>& a1b,
                  const VecInterface<VecT>& b0b, const VecInterface<VecT>& b1b) noexcept
        : hex_edge_id{ivec2{0, 1}, ivec2{1, 2}, ivec2{2, 3}, ivec2{3, 0},
                      ivec2{4, 5}, ivec2{5, 6}, ivec2{6, 7}, ivec2{7, 4},
                      ivec2{0, 4}, ivec2{1, 5}, ivec2{2, 6}, ivec2{3, 7}} {
      h_vertices[0] = a0 - b0;
      h_vertices[1] = a1 - b0;
      h_vertices[2] = a1 - b1;
      h_vertices[3] = a0 - b1;
      h_vertices[4] = a0b - b0b;
      h_vertices[5] = a1b - b0b;
      h_vertices[6] = a1b - b1b;
      h_vertices[7] = a0b - b1b;
    }
    //(1 - tar) * ((1 - tr) * a0rs_ + tr * a0re_) + tar * ((1 - tr) * a1rs_
    //+ tr
    //* a1re_) - ((1 - tbr) * ((1 - tr) * b0rs_ + tr * b0re_) + tbr * ((1 -
    // tr)
    //* b1rs_ + tr * b1re_));

    template <typename T> bool isHexBboxCutBbox(const T& min, const T& max) const {
      bv_t bv = bv_t{get_bounding_box(h_vertices[0], h_vertices[1])};
      for (int i = 2; i != 8; ++i) merge(bv, h_vertices[i]);
      return overlaps(bv, bv_t{min, max});
    }

    ivec2 hex_edge_id[12];
    vec3 h_vertices[8];
  };

}  // namespace zs