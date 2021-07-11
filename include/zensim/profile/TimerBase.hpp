#pragma once
#include <string_view>
#include "zensim/meta/Meta.h"

namespace zs {

  template <typename Timer> struct ScopedTimer {
    ScopedTimer(std::string_view msg, Timer&& timer) : msg{msg}, timer{std::move(timer)} {
      this->timer.tick();
    };
    ~ScopedTimer() { this->timer.tock(msg); }

    Timer timer;
    std::string_view msg;
  };
  template <typename Timer> ScopedTimer(std::string_view, Timer&&)
      -> ScopedTimer<remove_cvref_t<Timer>>;

}  // namespace zs