#include "CppTimers.hpp"

#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/core.h"
#ifdef _OPENMP
#  include <omp.h>

#else
#  if defined(ZS_PLATFORM_WINDOWS)
#    include <processthreadsapi.h>  // cputime: GetProcessTimes
#    include <sysinfoapi.h>         // walltime: long long int GetTickCount64()

// https://levelup.gitconnected.com/8-ways-to-measure-execution-time-in-c-c-48634458d0f9
// https://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows/17440673#17440673
static double get_cpu_time() {
  FILETIME a, b, c, d;
  if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
    //  Returns total user time.
    //  Can be tweaked to include kernel times as well.
    return (double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
  } else {
    //  Handle error
    return 0;
  }
}
#  elif defined(ZS_PLATFORM_LINUX)
#    include <time.h>
#  endif

#endif

namespace zs {

  void CppTimer::tick() {
#ifdef _OPENMP
    double t = omp_get_wtime();
    last = t * 1e3;
#else
    struct timespec t;
    std::timespec_get(&t, TIME_UTC);
    last = t.tv_sec * 1e3 + t.tv_sec * 1e-6;
#endif
  }
  void CppTimer::tock() {
#ifdef _OPENMP
    double t = omp_get_wtime();
    cur = t * 1e3;
#else
    struct timespec t;
    std::timespec_get(&t, TIME_UTC);
    cur = t.tv_sec * 1e3 + t.tv_sec * 1e-6;
#endif
  }
  void CppTimer::tock(std::string_view tag) {
    tock();
    fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag, elapsed());
  }

  float CppTimer::elapsed() const noexcept { return cur - last; }

}  // namespace zs