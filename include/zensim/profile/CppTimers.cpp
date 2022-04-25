#include "CppTimers.hpp"

#include <time.h>
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/core.h"

namespace zs {

  void CppTimer::tick() {
    struct timespec t;
    std::timespec_get(&t, TIME_UTC);
    last = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
  }
  void CppTimer::tock() {
    struct timespec t;
    std::timespec_get(&t, TIME_UTC);
    cur = t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
  }
  void CppTimer::tock(std::string_view tag) {
    tock();
    fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag, elapsed());
  }

  float CppTimer::elapsed() const noexcept { return cur - last; }

}  // namespace zs