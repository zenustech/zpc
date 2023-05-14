#pragma once
#include <type_traits>

#include "LevelSetInterface.h"
#include "SparseLevelSet.hpp"
#include "VdbLevelSet.h"
#include "zensim/container/RingBuffer.hpp"

namespace zs {

  template <typename LS> struct LevelSetWindow {
    using ls_t = LS;
    static constexpr int dim = ls_t::dim;
    using T = typename ls_t::value_type;
    using TV = typename ls_t::TV;
    static_assert(is_floating_point_v<T>, "");

    constexpr LevelSetWindow(T dt = 0, T alpha = 0) : stepDt{dt}, alpha{alpha} {}
    ~LevelSetWindow() = default;

    /// push
    template <typename LsT, enable_if_t<std::is_assignable_v<ls_t, LsT>> = 0> void push(LsT &&ls) {
      st = std::move(ed);
      ed = FWD(ls);
      alpha = 0;
    }
    /// frame dt
    void setDeltaT(T time) { stepDt = time; }
    /// advance
    void advance(T ratio) { alpha += ratio; }

    ls_t st;
    ls_t ed;
    T stepDt, alpha;
  };

  template <execspace_e, typename LevelSetWindowT, typename = void> struct LevelSetWindowView;

  template <execspace_e Space, typename LevelSetWindowT>
  struct LevelSetWindowView<Space, LevelSetWindowT>
      : LevelSetInterface<LevelSetWindowView<Space, LevelSetWindowT>> {
    using ls_t = typename LevelSetWindowT::ls_t;
    static constexpr int dim = LevelSetWindowT::dim;
    using T = typename LevelSetWindowT::T;
    using TV = typename LevelSetWindowT::TV;

    constexpr LevelSetWindowView() = default;
    ~LevelSetWindowView() = default;
    explicit constexpr LevelSetWindowView(LevelSetWindowT &ls)
        : st{ls.st.self()}, ed{ls.ed.self()}, stepDt{ls.stepDt}, alpha{ls.alpha} {}

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      TV v = (st.getMaterialVelocity(x) + ed.getMaterialVelocity(x)) / 2;
      TV x0 = x - alpha * stepDt * v, x1 = x + (1 - alpha) * stepDt * v;
      return (1 - alpha) * st.getSignedDistance(x0) + alpha * ed.getSignedDistance(x1);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getNormal(const VecInterface<VecT> &x) const noexcept {
      TV v = (st.getMaterialVelocity(x) + ed.getMaterialVelocity(x)) / 2;
      TV x0 = x - alpha * stepDt * v, x1 = x + (1 - alpha) * stepDt * v;
      return (1 - alpha) * st.getNormal(x0) + alpha * ed.getNormal(x1);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      auto v = (st.getMaterialVelocity(x) + ed.getMaterialVelocity(x)) / 2;
      auto x0 = x - alpha * stepDt * v, x1 = x + (1 - alpha) * stepDt * v;
      return (1 - alpha) * st.getMaterialVelocity(x0) + alpha * ed.getMaterialVelocity(x1);
    }
    constexpr decltype(auto) do_getBoundingBox() const noexcept { return st.getBoundingBox(); }

    SparseLevelSetView<Space, ls_t> st, ed;
    T stepDt, alpha;
  };

  template <execspace_e ExecSpace, typename LS>
  constexpr decltype(auto) proxy(LevelSetWindow<LS> &levelset) {
    return LevelSetWindowView<ExecSpace, LevelSetWindow<LS>>{levelset};
  }

}  // namespace zs
