#pragma once
#include <atomic>

#include "zensim/TypeAlias.hpp"
#include "zensim/ZpcFunctional.hpp"

namespace zs {

  namespace detail {
    enum UnparkControl { RetainContinue, RemoveContinue, RetainBreak, RemoveBreak };
    enum ParkResult { Skip, Unpark, Timeout };
  }  // namespace detail
  enum ZPC_CORE_API FutexResult {
    value_changed,  // when expected != atomic value
    awoken,         // awoken from 'wake' (success state)
    interrupted,    // interrupted by certain signal
    timedout        // not applicable to wait, applicable to wait until
  };
  enum class ZPC_CORE_API CvStatus {  // identical to std cpp
    no_timeout,
    timeout
  };

  /// @note ref: Multithreading 101 concurrency primitive from scratch
  /// shared within a single process
  /// blocking construct in the context of shared-memory synchronization
  struct Futex {
    // put the current thread to sleep if the expected value matches the value in the atomic
    // waitmask will be saved and compared to the wakemask later in the wake call
    // to check if you wanna wake up this thread or keep it sleeping
    ZPC_CORE_API static FutexResult wait(std::atomic<u32> *v, u32 expected,
                                         u32 waitMask = 0xffffffff);
    /// @note duration in milli-seconds
    ZPC_CORE_API static FutexResult wait_for(std::atomic<u32> *v, u32 expected, i64 duration = -1,
                                             u32 waitMask = 0xffffffff);
    // wake up the thread if (wakeMask & waitMask == true)
    ZPC_CORE_API static int wake(std::atomic<u32> *v, int count = detail::deduce_numeric_max<int>(),
                                 u32 wakeMask = 0xffffffff);
  };

  ZPC_CORE_API void await_change(std::atomic<u32> &v, u32 cur);
  ZPC_CORE_API void await_equal(std::atomic<u32> &v, u32 desired);

  // process-local mutex
  struct Mutex : std::atomic<u32> {
    // 0: unlocked
    // 1: locked
    // 257: locked and contended (...0001 | 00000001)
    // Mutex(u32 offset = 0) noexcept : _kMask{(u32)1 << (offset & (u32)31)} {}
    // ~Mutex() = default;
    ZPC_CORE_API void lock() noexcept;
    ZPC_CORE_API void unlock() noexcept;
    ZPC_CORE_API bool try_lock() noexcept;
    static constexpr u32 _kMask{1};
  };

  // 8 bytes alignment for rollover issue
  // https://docs.ntpsec.org/latest/rollover.html
  struct ConditionVariable {
    ZPC_CORE_API void notify_one() noexcept;
    ZPC_CORE_API void notify_all() noexcept;

    ZPC_CORE_API void wait(Mutex &lk) noexcept;
    template <typename Pred> void wait(Mutex &lk, Pred p) {
      while (!p()) wait(lk);
    }

    ZPC_CORE_API CvStatus wait_for(Mutex &lk, i64 duration) noexcept;
    template <typename Pred> bool wait_for(Mutex &lk, i64 duration, Pred p) {
      while (!p())
        if (wait_for(lk, duration) == CvStatus::timeout) return p();
      return true;
    }

    std::atomic<u32> seq{0};
  };

}  // namespace zs