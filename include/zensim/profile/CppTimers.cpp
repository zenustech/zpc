#include "CppTimers.hpp"

#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/core.h"

namespace zs {

  float CppTimer::elapsed() {
    float duration = std::chrono::duration_cast<NS>(cur - last).count() * 1e-6;
    return duration;
  }
  void CppTimer::tock(std::string tag) {
    tock();
    fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag.c_str(), elapsed());
  }

}  // namespace zs