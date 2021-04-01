#include "CudaTimers.cuh"
#include "zensim/cuda/CudaConstants.inc"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/core.h"

namespace zs {

  CudaTimer::CudaTimer(stream_t sid) : streamId{sid} {
    Cuda::driver().createEvent(&last, CU_EVENT_DEFAULT);
    Cuda::driver().createEvent(&cur, CU_EVENT_DEFAULT);
  }
  float CudaTimer::elapsed() {
    float duration;
    Cuda::driver().syncEvent(cur);
    Cuda::driver().eventElapsedTime(&duration, last, cur);
    return duration;
  }
  void CudaTimer::tock(std::string tag) {
    tock();
    fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag.c_str(), elapsed());
  }

}  // namespace zs