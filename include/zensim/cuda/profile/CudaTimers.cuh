#pragma once

#include <string>

#include "zensim/cuda/Cuda.h"

namespace zs {

  struct CudaTimer {
    using event_t = void *;
    using stream_t = void *;
    explicit CudaTimer(stream_t sid);
    ~CudaTimer() {
      Cuda::driver().destroyEvent(last);
      Cuda::driver().destroyEvent(cur);
    }
    void tick() { Cuda::driver().recordEvent(last, streamId); }
    void tock() { Cuda::driver().recordEvent(cur, streamId); }
    float elapsed();
    void tock(std::string tag);

  private:
    stream_t streamId;
    event_t last, cur;
  };

}  // namespace zs
