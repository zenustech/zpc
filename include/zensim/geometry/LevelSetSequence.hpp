#pragma once
#include "LevelSet.h"
#include "zensim/container/RingBuffer.hpp"

namespace zs {

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
    constexpr T getSignedDistance(const TV &X) const noexcept {
      TV v = (ls(0).getMaterialVelocity(X) + ls(1).getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * dt * v, X1 = X + (1 - alpha) * dt * v;
      return (1 - alpha) * ls(0).getSignedDistance(X0) + alpha * ls(1).getSignedDistance(X1);
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV v = (ls(0).getMaterialVelocity(X) + ls(1).getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * dt * v, X1 = X + (1 - alpha) * dt * v;
      return (1 - alpha) * ls(0).getNormal(X0) + alpha * ls(1).getNormal(X1);
    }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept {
      TV v = (ls(0).getMaterialVelocity(X) + ls(1).getMaterialVelocity(X)) / 2;
      TV X0 = X - alpha * dt * v, X1 = X + (1 - alpha) * dt * v;
      return (1 - alpha) * ls(0).getMaterialVelocity(X0) + alpha * ls(1).getMaterialVelocity(X1);
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return ls(0).getBoundingBox(); }

  protected:
    T dt, alpha;  ///< [0, 1]
  };

}  // namespace zs
