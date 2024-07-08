#pragma once

#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/geometry/Predicates.hpp"
#include "zensim/math/Rational.hpp"
#include "zensim/math/Vec.h"
#include "zensim/math/probability/Random.hpp"

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

  enum bilinear_degeneration_e : int {
    BI_DEGE_PLANE = 1,
    BI_DEGE_XOR_02 = 2,
    BI_DEGE_XOR_13 = 3,
  };

  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 2>
                           = 0>
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

  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
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
    template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                           VecT::dim == 1, VecT::extent == 3>
                             = 0>
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

    template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                           VecT::dim == 1, VecT::extent == 3>
                             = 0>
    constexpr prism(const VecInterface<VecT>& vs, const VecInterface<VecT>& fs0,
                    const VecInterface<VecT>& fs1, const VecInterface<VecT>& fs2,
                    const VecInterface<VecT>& ve, const VecInterface<VecT>& fe0,
                    const VecInterface<VecT>& fe1, const VecInterface<VecT>& fe2) noexcept
        : prism_edge_id{ivec2{0, 1}, ivec2{1, 2}, ivec2{2, 0}, ivec2{3, 4}, ivec2{4, 5},
                        ivec2{5, 3}, ivec2{0, 3}, ivec2{1, 4}, ivec2{2, 5}},
          p_vertices{vs - fs0, vs - fs2, vs - fs1, ve - fe0, ve - fe2, ve - fe1} {
#if 0
      p_vertices[0] = vs - fs0;
      p_vertices[1] = vs - fs2;
      p_vertices[2] = vs - fs1;
      p_vertices[3] = ve - fe0;
      p_vertices[4] = ve - fe2;
      p_vertices[5] = ve - fe1;
#endif
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

    template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                           VecT::dim == 1, VecT::extent == 3>
                             = 0>
    constexpr hex(const VecInterface<VecT>& a0, const VecInterface<VecT>& a1,
                  const VecInterface<VecT>& b0, const VecInterface<VecT>& b1,
                  const VecInterface<VecT>& a0b, const VecInterface<VecT>& a1b,
                  const VecInterface<VecT>& b0b, const VecInterface<VecT>& b1b) noexcept
        : hex_edge_id{ivec2{0, 1}, ivec2{1, 2}, ivec2{2, 3}, ivec2{3, 0}, ivec2{4, 5}, ivec2{5, 6},
                      ivec2{6, 7}, ivec2{7, 4}, ivec2{0, 4}, ivec2{1, 5}, ivec2{2, 6}, ivec2{3, 7}},
          h_vertices{a0 - b0,   a1 - b0,   a1 - b1,   a0 - b1,
                     a0b - b0b, a1b - b0b, a1b - b1b, a0b - b1b} {
#if 0
      h_vertices[0] = a0 - b0;
      h_vertices[1] = a1 - b0;
      h_vertices[2] = a1 - b1;
      h_vertices[3] = a0 - b1;
      h_vertices[4] = a0b - b0b;
      h_vertices[5] = a1b - b0b;
      h_vertices[6] = a1b - b1b;
      h_vertices[7] = a0b - b1b;
#endif
    }
    //(1 - tar) * ((1 - tr) * a0rs_ + tr * a0re_) + tar * ((1 - tr) * a1rs_
    //+ tr
    //* a1re_) - ((1 - tbr) * ((1 - tr) * b0rs_ + tr * b0re_) + tbr * ((1 -
    // tr)
    //* b1rs_ + tr * b1re_));

    template <typename T> constexpr bool isHexBboxCutBbox(const T& min, const T& max) const {
      bv_t bv = bv_t{get_bounding_box(h_vertices[0], h_vertices[1])};
      for (int i = 2; i != 8; ++i) merge(bv, h_vertices[i]);
      return overlaps(bv, bv_t{min, max});
    }

    ivec2 hex_edge_id[12];
    vec3 h_vertices[8];
  };

  ///
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr bool is_triangle_degenerated(const VecInterface<VecT>& t1, const VecInterface<VecT>& t2,
                                         const VecInterface<VecT>& t3) noexcept {
    using T = typename VecT::value_type;
    using vec2 = typename VecT::template variant_vec<T, integer_sequence<int, 2>>;
    const auto to_2d = [](const VecInterface<VecT>& p, int t) {
      vec2 ret{};
      ret.val(0) = p[(t + 1) % 3];
      ret.val(1) = p[(t + 2) % 3];
      return ret;
    };
    double r = ((t1 - t2).cross(t1 - t3)).norm();
    if (absolute(r) > 1e-8) return false;
    int ori{};
    vec2 p[3] = {};
    for (int j = 0; j != 3; ++j) {
      p[0] = to_2d(t1, j);
      p[1] = to_2d(t2, j);
      p[2] = to_2d(t3, j);

      ori = orient2d(p[0], p[1], p[2]);
      if (ori != 0) return false;
    }
    return true;
  }

  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr bool same_point(const VecInterface<VecT>& p1, const VecInterface<VecT>& p2) noexcept {
    if (p1[0] == p2[0] && p1[1] == p2[1] && p1[2] == p2[2]) return true;
    return false;
  }

  // 0 not intersected; 1 intersected; 2 pt on s0
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int point_on_ray(const VecInterface<VecT>& s0, const VecInterface<VecT>& e0,
                             const VecInterface<VecT>& dir0,
                             const VecInterface<VecT>& pt) noexcept {
    if (same_point(s0, pt)) return 2;
    if (!is_triangle_degenerated(s0, e0, pt)) return 0;

    // now the pt is on the line
    if (dir0[0] > 0) {
      if (pt[0] <= s0[0]) return 0;
    }
    if (dir0[0] < 0) {
      if (pt[0] >= s0[0]) return 0;
    }
    if (dir0[0] == 0) {
      if (pt[0] != s0[0]) return 0;
    }
    //
    if (dir0[1] > 0) {
      if (pt[1] <= s0[1]) return 0;
    }
    if (dir0[1] <= 0) {
      if (pt[1] >= s0[1]) return 0;
    }
    if (dir0[1] == 0) {
      if (pt[1] != s0[1]) return 0;
    }
    //
    if (dir0[2] > 0) {
      if (pt[2] <= s0[2]) return 0;
    }
    if (dir0[2] <= 0) {
      if (pt[2] >= s0[2]) return 0;
    }
    if (dir0[2] == 0) {
      if (pt[2] != s0[2]) return 0;
    }
    return 1;
  }

  // point and seg are colinear, now check if point is on the segment(can deal
  // with segment degeneration)
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr bool colinear_point_on_segment(const VecInterface<VecT>& pt,
                                           const VecInterface<VecT>& s0,
                                           const VecInterface<VecT>& s1) {
    if (zs::min(s0[0], s1[0]) <= pt[0] && pt[0] <= zs::max(s0[0], s1[0]))
      if (zs::min(s0[1], s1[1]) <= pt[1] && pt[1] <= zs::max(s0[1], s1[1]))
        if (zs::min(s0[2], s1[2]) <= pt[2] && pt[2] <= zs::max(s0[2], s1[2])) return true;
    return false;
  }
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr bool point_on_segment(const VecInterface<VecT>& pt, const VecInterface<VecT>& s0,
                                  const VecInterface<VecT>& s1) {
    if (!is_triangle_degenerated(pt, s0, s1)) return false;
    return colinear_point_on_segment(pt, s0, s1);
  }

  // 0 not intersected, 1 intersected, 2 s0 on segment
  // can deal with degenerated cases
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int ray_segment_intersection(const VecInterface<VecT>& s0, const VecInterface<VecT>& e0,
                                         const VecInterface<VecT>& dir0,
                                         const VecInterface<VecT>& s1,
                                         const VecInterface<VecT>& e1) {
    using vec3 =
        typename VecT::template variant_vec<typename VecT::value_type,
                                            integer_sequence<typename VecT::index_type, 3>>;
    if (same_point(e1, s1))  // degenerated case
      return point_on_ray(s0, e0, dir0, s1);

    /////////////////////////////////////
    if (orient3d(s0, e0, s1, e1) != 0) return 0;

    if (point_on_segment(s0, s1, e1)) return 2;
    if (!is_triangle_degenerated(s1, s0, e0)) {
      // we can get a point out of the plane
      PCG pcg{};
      auto np = vec3::ones();
      while (orient3d(np, s1, s0, e0) == 0) {  // if coplanar, random
        np[0] = pcg();
        np[1] = pcg();
        np[2] = pcg();
      }
      int o1 = orient3d(np, e1, s0, e0);
      if (o1 == 0) return point_on_ray(s0, e0, dir0, e1);
      int o2 = -1 * orient3d(np, s1, s0, e0);  // already know this is not 0
      int oo = orient3d(np, e1, s0, s1);
      if (oo == 0) {  // actually can directly return 0 because s0-s1-e0 is not
                      // degenerated
        if (point_on_ray(s0, e0, dir0, s1) > 0 || point_on_ray(s0, e0, dir0, e1) > 0) return 1;
        return 0;
      }
      if (o1 == oo && o2 == oo) return 1;
      return 0;
    } else {
      // s1 is on the line, if s1 is on ray, return 1; else return 0.(s0 is
      // not on seg)
      if (point_on_ray(s0, e0, dir0, s1) == 1) return 1;
      return 0;
    }
    return 0;
  }

  // 2 on edge, 1 interior, 0 not intersect, 3 intersect OPEN edge t2-t3
  template <typename VecT,
            enable_if_all<is_floating_point_v<typename VecT::value_type>, VecT::dim == 1,
                          VecT::extent == 3>
            = 0>  // norm follows right hand law
  constexpr int point_inter_triangle(const VecInterface<VecT>& pt, const VecInterface<VecT>& t1,
                                     const VecInterface<VecT>& t2, const VecInterface<VecT>& t3,
                                     bool dege, const bool halfopen) {
    using vec3 =
        typename VecT::template variant_vec<typename VecT::value_type,
                                            integer_sequence<typename VecT::index_type, 3>>;
    if (dege) {  // check 2 edges are enough
      if (point_on_segment(pt, t1, t2)) return 2;
      if (point_on_segment(pt, t1, t3)) return 2;
      return 0;
    } else {
      if (orient3d(pt, t1, t2, t3) != 0) return false;
      /*if (point_on_segment(pt, t1, t2))
              return 2;
      if (point_on_segment(pt, t1, t3))
              return 2;
      if (point_on_segment(pt, t2, t3))
              return 2;*///no need to do above
      PCG pcg{};
      auto np = vec3::ones();
      while (orient3d(np, t1, t2, t3) == 0) {  // if coplanar, random
        np[0] = pcg();
        np[1] = pcg();
        np[2] = pcg();
      }
      int o1 = orient3d(pt, np, t1, t2);
      int o2 = orient3d(pt, np, t2, t3);  // this edge
      int o3 = orient3d(pt, np, t3, t1);
      if (halfopen) {
        if (o2 == 0 && o1 == o3) return 3;  // on open edge t2-t3
      }

      if (o1 == o2 && o1 == o3) return 1;
      if (o1 == 0 && o2 * o3 >= 0) return 2;
      if (o2 == 0 && o1 * o3 >= 0) return 2;
      if (o3 == 0 && o2 * o1 >= 0) return 2;

      return 0;
    }
  }

  // 0 no intersection, 1 intersect, 2 point on triangle(including two edges), 3
  // point or ray shoot t2-t3 edge, -1 shoot on border
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int ray_triangle_intersection(const VecInterface<VecT>& pt,
                                          const VecInterface<VecT>& pt1,
                                          const VecInterface<VecT>& dir,
                                          const VecInterface<VecT>& t1,
                                          const VecInterface<VecT>& t2,
                                          const VecInterface<VecT>& t3, const bool halfopen) {
    if (is_triangle_degenerated(t1, t2, t3))  // triangle degeneration
    {
      // std::cout<<"tri degenerated "<<std::endl;
      int inter1 = ray_segment_intersection(pt, pt1, dir, t1, t2);
      if (inter1 == 1) return -1;
      if (inter1 == 2) return 2;

      int inter2 = ray_segment_intersection(pt, pt1, dir, t1, t3);  // check two segs is enough
      if (inter2 == 1) return -1;
      if (inter2 == 2) return 2;

      return 0;
    }

    int o1 = orient3d(pt, t1, t2, t3);
    if (o1 == 0) {  // point is on the plane
      int inter = point_inter_triangle(pt, t1, t2, t3, false, halfopen);
      if (inter == 1 || inter == 2) return 2;
      if (inter == 3) return 3;
      // if (inter == 0)
      else {                                   // pt on the plane but not intersect triangle.
        if (orient3d(pt1, t1, t2, t3) == 0) {  // if ray is on the plane
          int inter1 = ray_segment_intersection(pt, pt1, dir, t1, t2);
          if (inter1 == 1) return -1;
          if (inter1 == 2)
            return 2;  // acutally, cannot be 2 because already checked
                       // by pt_inter_tri

          int inter2 = ray_segment_intersection(pt, pt1, dir, t1, t3);
          if (inter2 == 1) return -1;
          if (inter2 == 2) return 2;
          // actually since point do not intersect triangle, check two
          // segs are enough int inter3 = ray_segment_intersection(pt, dir,
          // t2, t3);
          //// ray overlaps t2-t3, shoot another ray, ray intersect it,
          /// shoot another one
          // if (inter3 == 1) return -1;
          // if (inter3 == 2) return 2;

          return 0;
        }
      }
      return 0;
    }

    // the rest of cases are point not on plane and triangle not degenerated
    // 3 point or ray shoot t2-t3 edge, -1 shoot on border, 0 not intersected, 1
    // intersect interior
    int ori12 = orient3d(pt1, pt, t1, t2);
    int ori23 = orient3d(pt1, pt, t2, t3);
    int ori31 = orient3d(pt1, pt, t3, t1);
    // if(ori12*ori23<0||ori12*ori31<0||ori23*ori31<0) return 0;//ray triangle
    // not intersected;
    int oris = orient3d(t3, pt, t1, t2);  // if ray shoot triangle, oris should have
                                          // same sign with the three oritations
    if (oris * ori12 < 0 || oris * ori23 < 0 || oris * ori31 < 0)
      return 0;  // ray triangle not intersected;

    // bool in1=is_ray_intersect_plane(pt, dir, t1, t2, t3);
    // if (!in1) return 0;//ray triangle not intersected;
    // std::cout<<"before return 1"<<std::endl;
    if (ori12 == ori23 && ori12 == ori31) return 1;  //
    // std::cout<<"after return 1"<<std::endl;
    if (ori12 * ori23 >= 0 && ori31 == 0) return -1;
    if (ori31 * ori23 >= 0 && ori12 == 0) return -1;
    if (ori12 == ori31 && ori23 == 0) {
      if (halfopen)
        return 3;
      else
        return -1;
    }
    return 0;
  }

  constexpr zs::vec<rational, 3> cross(const zs::vec<rational, 3>& v1,
                                       const zs::vec<rational, 3>& v2) noexcept {
    zs::vec<rational, 3> ret{};
    ret[0] = v1[1] * v2[2] - v1[2] * v2[1];
    ret[1] = v1[2] * v2[0] - v1[0] * v2[2];
    ret[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return ret;
  }
  constexpr rational dot(const zs::vec<rational, 3>& v1, const zs::vec<rational, 3>& v2) noexcept {
    rational ret{v1[0] * v2[0]};
    ret += v1[1] * v2[1];
    ret += v1[2] * v2[2];
    return ret;
  }
  constexpr zs::vec<rational, 3> operator+(const zs::vec<rational, 3>& v1,
                                           const zs::vec<rational, 3>& v2) noexcept {
    zs::vec<rational, 3> ret{};
    ret[0] = v1[0] + v2[0];
    ret[1] = v1[1] + v2[1];
    ret[2] = v1[2] + v2[2];
    return ret;
  }
  constexpr zs::vec<rational, 3> operator-(const zs::vec<rational, 3>& v1,
                                           const zs::vec<rational, 3>& v2) noexcept {
    zs::vec<rational, 3> ret{};
    ret[0] = v1[0] - v2[0];
    ret[1] = v1[1] - v2[1];
    ret[2] = v1[2] - v2[2];
    return ret;
  }
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr rational func_g(const zs::vec<rational, 3>& xr, const VecT corners[4],
                            const zs::vec<int, 3> indices) {
    const int p = indices[0];
    const int q = indices[1];
    const int r = indices[2];
    zs::vec<rational, 3> pr(corners[p][0], corners[p][1], corners[p][2]),
        qr(corners[q][0], corners[q][1], corners[q][2]),
        rr(corners[r][0], corners[r][1], corners[r][2]);
    return dot(xr - pr, cross(qr - pr, rr - pr));
  }
  constexpr bool int_seg_XOR(const int a, const int b) noexcept {
    if (a == 2 || b == 2) return true;
    if (a == 0 && b == 1) return true;
    if (a == 1 && b == 0) return true;
    if (a == 3 || b == 3) return false;
    if (a == b) return false;
    return false;
  }
  constexpr int int_ray_XOR(const int a, const int b) noexcept {
    if (a == -1 || b == -1) return -1;
    if (a == 0) return b;
    if (b == 0) return a;
    if (a == b) return 0;
    if (a == 2 || b == 2) return 2;  // this is case 2-3
    printf("impossible to go here\n");
    return -1;
  }
  template <typename VecTA, typename VecT,
            enable_if_all<is_floating_point_v<typename VecT::value_type>, VecT::dim == 1,
                          VecT::extent == 3>
            = 0>
  constexpr rational phi(const VecInterface<VecTA>& xd, const VecT corners[4]) {
    zs::vec<rational, 3> x{xd[0], xd[1], xd[2]};
    const rational g012 = func_g(x, corners, {0, 1, 2});
    const rational g132 = func_g(x, corners, {1, 3, 2});
    const rational g013 = func_g(x, corners, {0, 1, 3});
    const rational g032 = func_g(x, corners, {0, 3, 2});

    const rational h12 = g012 * g032;
    const rational h03 = g132 * g013;

    const rational phi = h12 - h03;

    return phi;
  }
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr rational phi(const zs::vec<rational, 3>& x, const VecT corners[4]) {
    const rational g012 = func_g(x, corners, {0, 1, 2});
    const rational g132 = func_g(x, corners, {1, 3, 2});
    const rational g013 = func_g(x, corners, {0, 1, 3});
    const rational g032 = func_g(x, corners, {0, 3, 2});

    const rational h12 = g012 * g032;
    const rational h03 = g132 * g013;

    const rational phi = h12 - h03;

    return phi;
  }

  constexpr void get_tet_phi(bilinear& bl) {
    rational v0{}, v1{}, v2{};
    v0 = rational(bl.v[0][0]) + rational(bl.v[2][0]);
    v1 = rational(bl.v[0][1]) + rational(bl.v[2][1]);
    v2 = rational(bl.v[0][2]) + rational(bl.v[2][2]);
    zs::vec<rational, 3> p02{};
    p02[0] = v0 / rational(2);
    p02[1] = v1 / rational(2);
    p02[2] = v2 / rational(2);

    rational phi02 = phi(p02, bl.v);
    if (phi02.get_sign() > 0) {
      bl.phi_f[0] = 1;
      bl.phi_f[1] = -1;
      return;
    } else {
      bl.phi_f[0] = -1;
      bl.phi_f[1] = 1;
      return;
    }
    // std::cout << "!!can not happen, get tet phi" << std::endl;
  }
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int ray_correct_bilinear_face_pair_inter(const VecInterface<VecT>& p,
                                                     const VecInterface<VecT>& p1,
                                                     const rational& phi_p,
                                                     const VecInterface<VecT>& dir,
                                                     const bilinear& bl) {
    int r1{}, r2{};
    /*if (phi_p.get_sign() == 0)
            return 2;*/
    if (bl.phi_f[0] * phi_p.get_sign() < 0) {
      r1 = ray_triangle_intersection(  // -1,0,1,2,3
          p, p1, dir, bl.v[bl.facets[0][0]], bl.v[bl.facets[0][1]], bl.v[bl.facets[0][2]], true);
      r2 = ray_triangle_intersection(p, p1, dir, bl.v[bl.facets[1][0]], bl.v[bl.facets[1][1]],
                                     bl.v[bl.facets[1][2]], true);
      // when there is -1, -1(shoot on one of two edges); impossible to have
      // 2; when there is 1, 1 when there is 3, pt is on t2-t3
      // edge(impossible), or ray go across that edge, parity +1, return 1
      // since pt is inside of tet, then impossible on any plane of the two
      // triangles
      if (r1 == -1 || r2 == -1) return -1;
      if (r1 > 0 || r2 > 0) return 1;  // cannot be degenerated, so this can work
      return 0;
    } else {
      r1 = ray_triangle_intersection(p, p1, dir, bl.v[bl.facets[2][0]], bl.v[bl.facets[2][1]],
                                     bl.v[bl.facets[2][2]], true);
      r2 = ray_triangle_intersection(p, p1, dir, bl.v[bl.facets[3][0]], bl.v[bl.facets[3][1]],
                                     bl.v[bl.facets[3][2]], true);
      if (r1 == -1 || r2 == -1) return -1;
      if (r1 > 0 || r2 > 0) return 1;  // cannot be degenerated, so this can work
      return 0;
    }
  }

  // if end point pt is inside of tet or on the border
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int ray_shoot_correct_pair(bilinear& bl, const VecInterface<VecT>& pt,
                                       const VecInterface<VecT>& pt1,
                                       const VecInterface<VecT>& dir) {
    if (bl.phi_f[0] == 2) {  // phi never calculated, need calculated
      get_tet_phi(bl);
    }
    rational phip = phi(pt, bl.v);
    if (phip == 0) return 2;  // point on bilinear
    return ray_correct_bilinear_face_pair_inter(pt, pt1, phip, dir, bl);
  }
  // we already know the bilinear is degenerated, next check which kind
  constexpr int bilinear_degeneration(const bilinear& bl) {
    bool dege1 = is_triangle_degenerated(bl.v[0], bl.v[1], bl.v[2]);
    bool dege2 = is_triangle_degenerated(bl.v[0], bl.v[2], bl.v[3]);

    if (dege1 && dege2) {
      return BI_DEGE_PLANE;
    }
    zs::vec<double, 3> p0{}, p1{}, p2{};

    if (dege1) {
      p0 = bl.v[0];
      p1 = bl.v[2];
      p2 = bl.v[3];
    } else {
      p0 = bl.v[0];
      p1 = bl.v[1];
      p2 = bl.v[2];
    }

    PCG pcg{};
    zs::vec<double, 3> np = zs::vec<double, 3>::ones();
    int ori = orient3d(np, p0, p1, p2);
    while (ori == 0) {  // if coplanar, random
      np[0] = pcg();
      np[1] = pcg();
      np[2] = pcg();
      ori = orient3d(np, p0, p1, p2);
    }
    int ori0 = orient3d(np, bl.v[0], bl.v[1], bl.v[2]);
    int ori1 = orient3d(np, bl.v[0], bl.v[2], bl.v[3]);
    if (ori0 * ori1 <= 0) {
      return BI_DEGE_XOR_02;
    }
    ori0 = orient3d(np, bl.v[0], bl.v[1], bl.v[3]);
    ori1 = orient3d(np, bl.v[3], bl.v[1], bl.v[2]);
    if (ori0 * ori1 <= 0) {
      return BI_DEGE_XOR_13;
    }
    return BI_DEGE_PLANE;
  }
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int ray_degenerated_bilinear_parity(const bilinear& bl, const VecInterface<VecT>& pt,
                                                const VecInterface<VecT>& pt1,
                                                const VecInterface<VecT>& dir, const int dege) {
    int r1{}, r2{};
    if (dege == BI_DEGE_PLANE) {
      r1 = ray_triangle_intersection(  // -1, 0, 1, 2
          pt, pt1, dir, bl.v[0], bl.v[1], bl.v[2], true);
      // std::cout<<"inter t1, "<<r1<<"\n"<<std::endl;
      // std::cout<<"v0 "<<bl.v[0][0]<<", "<<bl.v[0][1]<<",
      // "<<bl.v[0][2]<<std::endl; std::cout<<"v1 "<<bl.v[1][0]<<",
      // "<<bl.v[1][1]<<", "<<bl.v[1][2]<<std::endl; std::cout<<"v2
      // "<<bl.v[2][0]<<", "<<bl.v[2][1]<<", "<<bl.v[2][2]<<std::endl;
      // std::cout<<"dir "<<dir[0]<<", "<<dir[1]<<", "<<dir[2]<<std::endl;
      // ray_triangle_intersection_rational
      if (r1 == 2) return 2;
      if (r1 == -1) return -1;
      if (r1 == 1) return 1;
      if (r1 == 3) {
        r1 = ray_triangle_intersection(  // -1, 0, 1, 2
            pt, pt1, dir, bl.v[0], bl.v[1], bl.v[2],
            false);             //  this should have 3, if 3, see it as 1
        if (r1 == 2) return 2;  // point on t2-t3 edge
        return 1;               // ray go through t2-t3 edge
      }
      r2 = ray_triangle_intersection(pt, pt1, dir, bl.v[0], bl.v[3], bl.v[2], false);
      // std::cout<<"inter t2, "<<r2<<std::endl;
      if (r2 == 2) return 2;
      if (r2 == -1) return -1;
      if (r2 == 1) return 1;
      return 0;
    } else {
      if (dege == BI_DEGE_XOR_02) {  // triangle 0-1-2 and 0-2-3
        r1 = ray_triangle_intersection(pt, pt1, dir, bl.v[0], bl.v[1], bl.v[2],
                                       true);  // 0: not hit, 1: hit on open triangle, 2: pt on
                                               // halfopen T, since already checked, accept it
        r2 = ray_triangle_intersection(pt, pt1, dir, bl.v[0], bl.v[3], bl.v[2], true);
        return int_ray_XOR(r1, r2);
      }

      if (dege == BI_DEGE_XOR_13) {  // triangle 0-1-3 and 3-1-2
        r1 = ray_triangle_intersection(pt, pt1, dir, bl.v[0], bl.v[1], bl.v[3],
                                       true);  // 0: not hit, 1: hit on open triangle, 2: pt on
                                               // halfopen T, since already checked, accept it
        r2 = ray_triangle_intersection(pt, pt1, dir, bl.v[2], bl.v[1], bl.v[3], true);
        return int_ray_XOR(r1, r2);
      }
    }
    printf("!! THIS CANNOT HAPPEN\n");
    return false;
  }
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int ray_bilinear_parity(bilinear& bl, const VecInterface<VecT>& pt,
                                    const VecInterface<VecT>& pt1, const VecInterface<VecT>& dir,
                                    const bool is_degenerated,
                                    const bool is_point_in_tet)  // out of tet means no touch tet
  {
    if (!is_degenerated) {
      bool check = false;
      if (!is_point_in_tet) {  // p out of tet,or on the border
        int r1{}, r2{};
        // we need to know: if shoot on any edge?(two edges return -1, one
        // edge return 1) std::cout<<"is dege "<<is_degenerated<<std::endl;

        r1 = ray_triangle_intersection(pt, pt1, dir, bl.v[bl.facets[0][0]], bl.v[bl.facets[0][1]],
                                       bl.v[bl.facets[0][2]], true);
        r2 = ray_triangle_intersection(pt, pt1, dir, bl.v[bl.facets[1][0]], bl.v[bl.facets[1][1]],
                                       bl.v[bl.facets[1][2]], true);

        // std::cout<<"r1 "<<r1<<" r2 "<<r2<<std::endl;
        // idea is: if -1
        if (r1 == -1 || r2 == -1) return -1;
        if (r1 == 3 || r2 == 3)
          check = true;  // 3-3(pt on t2-t3 or shoot t2-t3) or 2-3 (pt in
                         // one face, shoot another t2-t3)
        if (r1 == 2 && r2 == 0) check = true;
        if (r1 == 0 && r2 == 2) check = true;
        if (r1 == 1 && r2 == 1) return 0;
        if (r1 + r2 == 1) return 1;        // 1-0 case
        if (r1 == 1 || r2 == 1) return 0;  // 1-2 case
        if (r1 == 0 && r2 == 0) return 0;

        if (check == false)
          return 0;
        else {  // intersect the t2-t3 edge, or point on triangle
          if (r1 == 3 || r2 == 3) {
            if (r1 == 2 || r2 == 2) return 0;
            if (point_inter_triangle(pt, bl.v[bl.facets[0][0]], bl.v[bl.facets[0][1]],
                                     bl.v[bl.facets[0][2]], false, false)
                    > 0
                || point_inter_triangle(pt, bl.v[bl.facets[1][0]], bl.v[bl.facets[1][1]],
                                        bl.v[bl.facets[1][2]], false, false)
                       > 0) {  // point on t2-t3 edge, regard as inside
              return ray_shoot_correct_pair(bl, pt, pt1, dir);
            } else {
              if (orient3d(pt, bl.v[bl.facets[0][0]], bl.v[bl.facets[0][1]], bl.v[bl.facets[0][2]])
                      > 0
                  && orient3d(pt, bl.v[bl.facets[1][0]], bl.v[bl.facets[1][1]],
                              bl.v[bl.facets[1][2]])
                         > 0)
                return 1;
              return 0;
            }
          }
          if (r1 == 2 || r2 == 2) return ray_shoot_correct_pair(bl, pt, pt1, dir);
          printf("impossible to go here.\n");
          return 0;
        }
      } else {  // p inside open tet
        return ray_shoot_correct_pair(bl, pt, pt1, dir);
      }
    } else {  // degenerated bilinear
      int degetype = bilinear_degeneration(bl);

      return ray_degenerated_bilinear_parity(bl, pt, pt1, dir, degetype);
    }
  }

  // the facets of the tet are all oriented to outside. check if p is inside of
  // OPEN tet
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr bool is_point_inside_tet(const bilinear& bl, const VecInterface<VecT>& p) noexcept {
    for (int i = 0; i != 4; ++i) {  // facets.size()==4
      const auto &pt1 = bl.v[bl.facets[i][0]], &pt2 = bl.v[bl.facets[i][1]],
                 &pt3 = bl.v[bl.facets[i][2]];
      if (orient3d(p, pt1, pt2, pt3) >= 0) return false;
    }
    return true;  // all the orientations are -1, then point inside
  }

  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int ray_triangle_parity(const VecInterface<VecT>& pt, const VecInterface<VecT>& pt1,
                                    const VecInterface<VecT>& dir, const VecInterface<VecT>& t0,
                                    const VecInterface<VecT>& t1, const VecInterface<VecT>& t2,
                                    const bool is_triangle_degenerated) {
    if (!is_triangle_degenerated) {
      int res = ray_triangle_intersection(pt, pt1, dir, t0, t1, t2, false);

      return res;
      // 0 not hit, 1 hit on open triangle, -1 parallel or hit on edge, need
      // another shoot.
    } else {
      // if pt on it (2), return 2; if 1(including overlap) return -1
      int i1 = ray_segment_intersection(pt, pt1, dir, t0, t1);  // 2 means pt on the segment
      if (i1 == 2) return 2;
      if (i1 == 1) return -1;
      int i2 = ray_segment_intersection(pt, pt1, dir, t1, t2);
      if (i2 == 2) return 2;
      if (i2 == 1) return -1;

      return 0;
    }
  }

  // dir = pt1 - pt
  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int point_inside_prism(prism& psm, bilinear bls[3], const VecInterface<VecT>& pt,
                                   const VecInterface<VecT>& pt1, const VecInterface<VecT>& dir,
                                   bool is_pt_in_tet[3]) {
    int S = 0;
    if (dir[0] == 0 && dir[1] == 0 && dir[2] == 0) {
      printf("random direction wrong!\n");
      return -1;
    }

    for (int patch = 0; patch < 3; ++patch) {
      int is_ray_patch = ray_bilinear_parity(bls[patch], pt, pt1, dir, bls[patch].is_degenerated,
                                             is_pt_in_tet[patch]);

      if (is_ray_patch == 2) return 1;

      if (is_ray_patch == -1) return -1;

      if (is_ray_patch == 1) S++;
    }

    int res{};

    res = ray_triangle_parity(pt, pt1, dir, psm.p_vertices[0], psm.p_vertices[1], psm.p_vertices[2],
                              psm.isTriangleDegenerated(0));

    if (res == 2) return 1;  // it should be impossible
    if (res == -1) return -1;

    if (res > 0) S++;

    res = ray_triangle_parity(pt, pt1, dir, psm.p_vertices[3], psm.p_vertices[4], psm.p_vertices[5],
                              psm.isTriangleDegenerated(1));

    if (res == 2) return 1;  // it should be impossible
    if (res == -1) return -1;

    if (res > 0) S++;

    return ((S % 2) == 1) ? 1 : 0;
  }
  constexpr bool shoot_origin_ray_prism(prism& psm, bilinear bls[3]) {
    constexpr int max_trials = 8;
    using vec3 = zs::vec<double, 3>;

    // if a/2<=b<=2*a, then a-b is exact.
    bool is_pt_in_tet[3] = {};
    for (int i = 0; i < 3; i++) {
      if (bls[i].is_degenerated)
        is_pt_in_tet[i] = false;
      else
        is_pt_in_tet[i] = is_point_inside_tet(bls[i], vec3::zeros());
    }

    zs::vec<double, 3> dir(1, 0, 0);
    zs::vec<double, 3> pt2 = dir;

    int res = -1;
    int trials{};
    PCG pcg{};

    for (trials = 0; trials < max_trials; ++trials) {
      res = point_inside_prism(psm, bls, vec3::zeros(), pt2, dir, is_pt_in_tet);

      if (res >= 0) break;

      dir[0] = pcg();
      dir[1] = pcg();
      dir[2] = pcg();
      pt2 = dir;
    }

    if (trials == max_trials) {
      printf("All rays are on edges, increase trials\n");
      return false;
    }

    return res >= 1;  // >=1 means point inside of prism
  }

  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr bool vertexFaceCCD(
      const VecInterface<VecT>& vertex_start, const VecInterface<VecT>& face_vertex0_start,
      const VecInterface<VecT>& face_vertex1_start, const VecInterface<VecT>& face_vertex2_start,
      const VecInterface<VecT>& vertex_end, const VecInterface<VecT>& face_vertex0_end,
      const VecInterface<VecT>& face_vertex1_end, const VecInterface<VecT>& face_vertex2_end) {
    // Rational rtn=minimum_distance;

    // std::cout << "eps "<<rtn.to_double() << std::endl;
    prism vfprism(vertex_start, face_vertex0_start, face_vertex1_start, face_vertex2_start,
                  vertex_end, face_vertex0_end, face_vertex1_end, face_vertex2_end);

    zs::vec<double, 3> bmin(0, 0, 0), bmax(0, 0, 0);
    bool intersection = vfprism.isPrismBboxCutBbox(bmin, bmax);
    if (!intersection) {
      return false;  // if bounding box not intersected, then not intersected
    }
    bilinear bl0(vfprism.p_vertices[0], vfprism.p_vertices[1], vfprism.p_vertices[4],
                 vfprism.p_vertices[3]);
    bilinear bl1(vfprism.p_vertices[1], vfprism.p_vertices[2], vfprism.p_vertices[5],
                 vfprism.p_vertices[4]);
    bilinear bl2(vfprism.p_vertices[0], vfprism.p_vertices[2], vfprism.p_vertices[5],
                 vfprism.p_vertices[3]);
    bilinear bls[3] = {bl0, bl1, bl2};
    bool oin = shoot_origin_ray_prism(vfprism, bls);
    return oin;
  }

  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr int point_inside_hex(bilinear bls[6], const VecInterface<VecT>& pt,
                                 const VecInterface<VecT>& pt1, const VecInterface<VecT>& dir,
                                 const bool is_pt_in_tet[6]) {
    int S = 0;
    if (dir[0] == 0 && dir[1] == 0 && dir[2] == 0) {
      printf("random direction wrong\n");
      return -1;
    }
    for (int patch = 0; patch != 6; ++patch) {
      int is_ray_patch = ray_bilinear_parity(bls[patch], pt, pt1, dir, bls[patch].is_degenerated,
                                             is_pt_in_tet[patch]);
      // std::cout<<"\nis ray parity "<<is_ray_patch<<" is pt in tet
      // "<<is_pt_in_tet[patch]<<std::endl; std::cout<<"bilinear ori,
      // "<<orient_3d(bls[patch].v[0],bls[patch].v[1],bls[patch].v[2],bls[patch].v[3])<<"this
      // bilinear finished\n"<<std::endl;

      if (is_ray_patch == 2) return 1;

      if (is_ray_patch == -1) return -1;

      if (is_ray_patch == 1) S++;
    }
    return ((S % 2) == 1) ? 1 : 0;
  }
  constexpr bool shoot_origin_ray_hex(bilinear bls[6]) {
    constexpr int max_trials = 8;
    using vec3 = zs::vec<double, 3>;

    // if a/2<=b<=2*a, then a-b is exact.
    bool is_pt_in_tet[6] = {};
    for (int i = 0; i < 6; i++) {
      if (bls[i].is_degenerated)
        is_pt_in_tet[i] = false;
      else
        is_pt_in_tet[i] = is_point_inside_tet(bls[i], vec3::zeros());
    }
    vec3 dir(1, 0, 0);
    vec3 pt2 = dir;

    int res = -1;
    int trials{};
    PCG pcg{};

    for (trials = 0; trials < max_trials; ++trials) {
      res = point_inside_hex(bls, vec3::zeros(), pt2, dir, is_pt_in_tet);

      if (res >= 0) break;

      dir[0] = pcg();
      dir[1] = pcg();
      dir[2] = pcg();
      pt2 = dir;
    }

    if (trials == max_trials) {
      printf("All rays are on edges, increase trials.\n");
      // throw "All rays are on edges, increase trials";
      return false;
    }

    return res >= 1;  // >=1 means point inside of prism
  }

  template <typename VecT, enable_if_all<is_floating_point_v<typename VecT::value_type>,
                                         VecT::dim == 1, VecT::extent == 3>
                           = 0>
  constexpr bool edgeEdgeCCD(
      const VecInterface<VecT>& edge0_vertex0_start, const VecInterface<VecT>& edge0_vertex1_start,
      const VecInterface<VecT>& edge1_vertex0_start, const VecInterface<VecT>& edge1_vertex1_start,
      const VecInterface<VecT>& edge0_vertex0_end, const VecInterface<VecT>& edge0_vertex1_end,
      const VecInterface<VecT>& edge1_vertex0_end, const VecInterface<VecT>& edge1_vertex1_end) {
    hex hx(edge0_vertex0_start, edge0_vertex1_start, edge1_vertex0_start, edge1_vertex1_start,
           edge0_vertex0_end, edge0_vertex1_end, edge1_vertex0_end, edge1_vertex1_end);

    // step 1. bounding box checking
    zs::vec<double, 3> bmin(0, 0, 0), bmax(0, 0, 0);
    bool intersection = hx.isHexBboxCutBbox(bmin, bmax);

    if (!intersection) return false;  // if bounding box not intersected, then not intersected
    bilinear bl0(hx.h_vertices[0], hx.h_vertices[1], hx.h_vertices[2], hx.h_vertices[3]);
    bilinear bl1(hx.h_vertices[4], hx.h_vertices[5], hx.h_vertices[6], hx.h_vertices[7]);
    bilinear bl2(hx.h_vertices[0], hx.h_vertices[1], hx.h_vertices[5], hx.h_vertices[4]);
    bilinear bl3(hx.h_vertices[1], hx.h_vertices[2], hx.h_vertices[6], hx.h_vertices[5]);
    bilinear bl4(hx.h_vertices[2], hx.h_vertices[3], hx.h_vertices[7], hx.h_vertices[6]);
    bilinear bl5(hx.h_vertices[0], hx.h_vertices[3], hx.h_vertices[7], hx.h_vertices[4]);
    bilinear bls[6] = {bl0, bl1, bl2, bl3, bl4, bl5};
    bool oin = shoot_origin_ray_hex(bls);
    return oin;
  }

}  // namespace zs