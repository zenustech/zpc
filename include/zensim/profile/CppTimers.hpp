#pragma once

#include <string_view>

#include "TimerBase.hpp"

namespace zs {

  /// wall time clock for now
  struct CppTimer {
    void tick();
    void tock();
    float elapsed() const noexcept;
    void tock(std::string_view tag);

  private:
    double last, cur;
  };

}  // namespace zs
