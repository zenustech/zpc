#pragma once

#include <string_view>

#include "zensim/profile/TimerBase.hpp"

namespace zs {

  struct MusaTimer {
    using event_t = void *;
    using stream_t = void *;
    explicit MusaTimer(stream_t sid);
    ~MusaTimer();
    void tick();
    void tock();
    float elapsed();
    void tock(std::string_view tag);

  private:
    stream_t streamId;
    event_t last, cur;
  };

}  // namespace zs
