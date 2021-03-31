#pragma once

#include <chrono>
#include <string>

namespace zs {

  struct CppTimer {
    using HRC = std::chrono::high_resolution_clock;
    using NS = std::chrono::nanoseconds;  ///< default timer unit
    using TimeStamp = HRC::time_point;

    void tick() { last = HRC::now(); }
    void tock() { cur = HRC::now(); }
    float elapsed();
    void tock(std::string tag);

  private:
    TimeStamp last, cur;
  };

}  // namespace zs
