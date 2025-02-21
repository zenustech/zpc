#include "MusaTimers.h"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/core.h"

namespace zs {

  MusaTimer::MusaTimer(stream_t sid) : streamId{sid} {
    musaEventCreateWithFlags((musaEvent_t *)&last, musaEventBlockingSync);
    musaEventCreateWithFlags((musaEvent_t *)&cur, musaEventBlockingSync);
  }
  MusaTimer::~MusaTimer() {
    musaEventDestroy((musaEvent_t)last);
    musaEventDestroy((musaEvent_t)cur);
  }
  float MusaTimer::elapsed() {
    float duration;
    musaEventSynchronize((musaEvent_t)cur);
    musaEventElapsedTime(&duration, (musaEvent_t)last, (musaEvent_t)cur);
    return duration;
  }
  void MusaTimer::tick() { musaEventRecord((musaEvent_t)last, (musaStream_t)streamId); }
  void MusaTimer::tock() { musaEventRecord((musaEvent_t)cur, (musaStream_t)streamId); }
  void MusaTimer::tock(std::string_view tag) {
    tock();
    fmt::print(fg(fmt::color::cyan), "{}: {} ms\n", tag, elapsed());
  }

}  // namespace zs