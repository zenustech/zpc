#include "ConcurrencyPrimitive.hpp"

#include <atomic>
#include <ctime>
#include <thread>

#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/Intrinsics.hpp"

#if defined(_WIN32)
#  include <windows.h>
// #  include <synchapi.h>
#elif defined(__linux__)
// #  include <immintrin.h>
#  include <linux/futex.h>
#  include <sys/syscall.h> /* Definition of SYS_* constants */
#  include <unistd.h>
#elif defined(__APPLE__)
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

#define ZS_USE_NATIVE_LINUX_FUTEX 0

namespace zs {

  namespace detail {
    std::atomic<u32> g_idCache = 0;
  }
  detail::ParkingLot<u32> g_lot;

  void await_change(std::atomic<u32> &v, u32 cur) {
    while (true) {
      // spin lock
      for (int i = 0; i != 1024; ++i) {
        if (v.load(std::memory_order_relaxed) != cur)  // finally changes
          return;
        std::this_thread::yield();
      }

      Futex::wait(&v, cur);  // system call
    }
  }
  void await_equal(std::atomic<u32> &v, u32 desired) {
    u32 cur{};
    while (true) {
      // spin lock
      for (int i = 0; i != 1024; ++i) {
        cur = v.load(std::memory_order_relaxed);
        if (cur == desired)  // finally equals
          return;
        std::this_thread::yield();
      }

      Futex::wait(&v, cur, (u32)1 << (desired & (u32)0x1f));  // system call
    }
  }

  // int futex(int *uaddr, int op, int val, const struct timespec *timeout, int *uaddr2, int val3);
  // long syscall(SYS_futex, u32 *uaddr, int op, u32 val, const timespec *, u32 *uaddr2, u32 val3);
  // bool WaitOnAddress(volatile void *addr, void *compareAddress, size_t, addressSize, dword dwMs)
  FutexResult Futex::wait(std::atomic<u32> *v, u32 expected, u32 waitMask) {
    return wait_for(v, expected, (i64)-1, waitMask);
  }
  FutexResult Futex::wait_for(std::atomic<u32> *v, u32 expected, i64 duration, u32 waitMask) {
#if defined(ZS_PLATFORM_LINUX) && ZS_USE_NATIVE_LINUX_FUTEX
    struct timespec tm {};
    struct timespec *timeout = nullptr;
    if (duration > -1) {
      /// @note seconds, nanoseconds
      struct timespec offset {
        duration / 1000, (duration % 1000) * 1000000
      };
      /// @note ref: https://www.man7.org/linux/man-pages/man3/clock_gettime.3.html
      clock_gettime(CLOCK_MONOTONIC, &tm);
      // clock_add(&tm, offset);
      tm.tv_sec += offset.tv_sec;
      tm.tv_nsec += offset.tv_nsec;
      tm.tv_sec += tm.tv_nsec / 1000000000;
      tm.tv_nsec %= 1000000000;
      timeout = &tm;
    }
    int const op = FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG;
    /// @note ref: https://man7.org/linux/man-pages/man2/futex.2.html
    long rc
        = syscall(SYS_futex, reinterpret_cast<u32 *>(v), op, expected, timeout, nullptr, waitMask);
    if (rc == 0)
      return FutexResult::awoken;
    else {
      switch (rc) {
        case ETIMEDOUT:
          return FutexResult::timedout;
        case EINTR:
          return FutexResult::interrupted;
        case EWOULDBLOCK:
          return FutexResult::value_changed;
        default:
          return FutexResult::value_changed;
      }
    }

#  if 0
#  elif defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    /// windows
    u32 undesired = expected;
    bool rc
        = WaitOnAddress((void *)v, &undesired, sizeof(u32), duration == -1 ? INFINITE : duration);
    if (rc) return FutexResult::awoken;
    if (undesired != expected) return FutexResult::value_changed;
    if (GetLastError() == ERROR_TIMEOUT) return FutexResult::timedout;
    return FutexResult::interrupted;
#  endif

#else
    return emulated_futex_wait_for(v, expected, duration, waitMask);
#endif
  }
  // wake up the thread(s) if (wakeMask & waitMask == true)
  // WakeByAddressSingle/All
  int Futex::wake(std::atomic<u32> *v, int count, u32 wakeMask) {
#if defined(ZS_PLATFORM_LINUX) && ZS_USE_NATIVE_LINUX_FUTEX
    int const op = FUTEX_WAKE_BITSET | FUTEX_PRIVATE_FLAG;
    long rc = syscall(SYS_futex, reinterpret_cast<u32 *>(v), op, count, nullptr, nullptr, wakeMask);
    if (rc < 0) return 0;
    return rc;

#  if 0
#  elif defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    if (count == limits<int>::max()) {
      WakeByAddressAll((void *)v);
      return limits<int>::max();
    } else if (count > 0) {
      for (int i = 0; i < count; ++i) WakeByAddressSingle((void *)v);
      return count;
    } else
      return 0;
#  endif

#else
    return emulated_futex_wake(v, count, wakeMask);
#endif
  }

  // ref
  // https://locklessinc-com.translate.goog/articles/mutex_cv_futex/
  void Mutex::lock() noexcept {
    u32 oldState = this->load(std::memory_order_relaxed);
    if (oldState == 0
        && this->compare_exchange_weak(oldState, oldState | _kMask, std::memory_order_acquire,
                                       std::memory_order_relaxed))
      return;

    {
      u32 spinCount = 0;
      constexpr size_t spin_limit = 1024;
      u32 newState;
    mutex_lock_retry:
      if ((oldState & _kMask) != 0) {
        ++spinCount;
        if (spinCount > spin_limit) {
          newState = oldState | _kMask;
          if (newState != oldState) {
            if (!this->compare_exchange_weak(oldState, newState, std::memory_order_relaxed,
                                             std::memory_order_relaxed))
              goto mutex_lock_retry;
          }
          Futex::wait_for(this, newState, (i64)-1, _kMask);
        } else if (spinCount > spin_limit + 1) {
          // zs::pause_cpu();
          std::this_thread::yield();
        } else {
          using namespace std::chrono_literals;
          std::this_thread::sleep_for(1ms);
        }
        oldState = this->load(std::memory_order_relaxed);
        goto mutex_lock_retry;
      }

      newState = oldState | _kMask;
      if (!this->compare_exchange_weak(oldState, newState, std::memory_order_acquire,
                                       std::memory_order_relaxed))
        goto mutex_lock_retry;
    }
  }

  void Mutex::unlock() noexcept {
    u32 oldState = this->load(std::memory_order_relaxed);
    u32 newState;
    do {
      newState = oldState & ~_kMask;
    } while (!this->compare_exchange_weak(oldState, newState, std::memory_order_release,
                                          std::memory_order_relaxed));

    if (oldState & _kMask) {
      Futex::wake(this, limits<int>::max(), _kMask);
    }
  }

  bool Mutex::trylock() noexcept {
    u32 state = this->load(std::memory_order_relaxed);
    do {
      if (state) return false;
    } while (!this->compare_exchange_weak(state, state | _kMask, std::memory_order_acquire,
                                          std::memory_order_relaxed));
    return true;
  }

#if 0
  void ConditionVariable::notify_one() {
    seq.fetch_add(1, std::memory_order_acq_rel);
    Futex::wake(&seq, 1);
  }

  void ConditionVariable::notify_all() {
    if (m == nullptr) return;
    seq.fetch_add(1, std::memory_order_acq_rel);
#  if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    // cannot find win32 alternative for FUTEX_REQUEUE
    WakeByAddressAll((void *)&seq);
#  elif defined(__clang__) || defined(__GNUC__)
#    ifdef ZS_PLATFORM_OSX
    throw std::runtime_error("no futex implementation for now on macos.");
#    else
    syscall(SYS_futex, reinterpret_cast<i32 *>(&seq), FUTEX_REQUEUE | FUTEX_PRIVATE_FLAG, 1,
            limits<i32>::max(), m, 0);
#    endif
#  endif
  }

  bool ConditionVariable::wait(Mutex &mut) {
    // https://www.cnblogs.com/bbqzsl/p/6808176.html
    if (m != &mut) {
      if (m != nullptr) return false;  // invalid argument
      atomic_cas(exec_seq, (void **)&m, (void *)nullptr, (void *)&mut);
      if (m != &mut) return false;  // invalid argument
    }
    m->unlock();
    Futex::wait(&seq, seq.load(std::memory_order_consume));
    // value (257) is coupled with mutex internal representation
    while (m->exchange(257, std::memory_order_acq_rel) & 1) Futex::wait(m, 257);
    return true;
  }
#endif

}  // namespace zs