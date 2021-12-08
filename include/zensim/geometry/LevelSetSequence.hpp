#pragma once
#include <type_traits>

#include "LevelSetInterface.h"
#include "SparseLevelSet.hpp"
#include "VdbLevelSet.h"
#include "zensim/container/RingBuffer.hpp"

namespace zs {

#if 0
  template <int WindowSize, typename DataType, int dim_, typename Tn = int> struct LevelSetSequence
      : RingBuffer<LevelSet<DataType, dim_, Tn>, WindowSize> {
    static constexpr int dim = dim_;
    using ls_t = LevelSet<DataType, dim, Tn>;
    using base_t = RingBuffer<ls_t, WindowSize>;
    using T = typename ls_t::T;
    using TV = typename ls_t::TV;

    template <typename Allocator = heap_allocator>
    constexpr void loadVdbFile(const std::string &fn, T dx,
                               Allocator &&allocator = heap_allocator{}) {
      base_t::push_back();
      ls_t &ls = base_t::back();
      ls.setDx(dx);
      ls.constructPhiVelFromVdbFile(fn, std::forward<Allocator>(allocator));
      return;
    }
    constexpr void roll() { base_t::pop(); }

    constexpr ls_t &ls(int index) noexcept { return (*this)[index]; }
    constexpr ls_t const &ls(int index) const noexcept { return (*this)[index]; }

    constexpr void setDt(T dtIn) noexcept { dt = dtIn; }
    constexpr void setAlpha(T ratio) noexcept { alpha = ratio; }
    constexpr T getSignedDistance(const TV &x) const noexcept {
      TV v = (ls(0).getMaterialVelocity(X) + ls(1).getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * dt * v, X1 = X + (1 - alpha) * dt * v;
      return (1 - alpha) * ls(0).getSignedDistance(X0) + alpha * ls(1).getSignedDistance(X1);
    }
    constexpr TV getNormal(const TV &x) const noexcept {
      TV v = (ls(0).getMaterialVelocity(X) + ls(1).getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * dt * v, X1 = X + (1 - alpha) * dt * v;
      return (1 - alpha) * ls(0).getNormal(X0) + alpha * ls(1).getNormal(X1);
    }
    constexpr TV getMaterialVelocity(const TV &x) const noexcept {
      TV v = (ls(0).getMaterialVelocity(X) + ls(1).getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * dt * v, X1 = X + (1 - alpha) * dt * v;
      return (1 - alpha) * ls(0).getMaterialVelocity(X0) + alpha * ls(1).getMaterialVelocity(X1);
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return ls(0).getBoundingBox(); }

  protected:
    T dt, alpha;  ///< [0, 1]
  };
#endif

  template <typename LS> struct LevelSetWindow {
    using ls_t = LS;
    static constexpr int dim = ls_t::dim;
    using T = typename ls_t::value_type;
    using TV = typename ls_t::TV;
    static_assert(std::is_floating_point_v<T>, "");

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
      : LevelSetInterface<LevelSetWindowView<Space, LevelSetWindowT>, typename LevelSetWindowT::T,
                          LevelSetWindowT::dim> {
    using ls_t = typename LevelSetWindowT::ls_t;
    static constexpr int dim = LevelSetWindowT::dim;
    using T = typename LevelSetWindowT::T;
    using TV = typename LevelSetWindowT::TV;

    constexpr LevelSetWindowView() = default;
    ~LevelSetWindowView() = default;
    explicit constexpr LevelSetWindowView(LevelSetWindowT &ls)
        : st{ls.st.self()}, ed{ls.ed.self()}, stepDt{ls.stepDt}, alpha{ls.alpha} {}

    constexpr T getSignedDistance(const TV &x) const noexcept {
      TV v = (st.getMaterialVelocity(x) + ed.getMaterialVelocity(x)) / 2;
      TV x0 = x - alpha * stepDt * v, x1 = x + (1 - alpha) * stepDt * v;
      return (1 - alpha) * st.getSignedDistance(x0) + alpha * ed.getSignedDistance(x1);
    }
    constexpr TV getNormal(const TV &x) const noexcept {
      TV v = (st.getMaterialVelocity(x) + ed.getMaterialVelocity(x)) / 2;
      TV x0 = x - alpha * stepDt * v, x1 = x + (1 - alpha) * stepDt * v;
      return (1 - alpha) * st.getNormal(x0) + alpha * ed.getNormal(x1);
    }
    constexpr TV getMaterialVelocity(const TV &x) const noexcept {
      TV v = (st.getMaterialVelocity(x) + ed.getMaterialVelocity(x)) / 2;
      TV x0 = x - alpha * stepDt * v, x1 = x + (1 - alpha) * stepDt * v;
      return (1 - alpha) * st.getMaterialVelocity(x0) + alpha * ed.getMaterialVelocity(x1);
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return st.getBoundingBox(); }

    SparseLevelSetView<Space, ls_t> st, ed;
    T stepDt, alpha;
  };

  template <execspace_e ExecSpace, typename LS>
  constexpr decltype(auto) proxy(LevelSetWindow<LS> &levelset) {
    return LevelSetWindowView<ExecSpace, LevelSetWindow<LS>>{levelset};
  }

}  // namespace zs
