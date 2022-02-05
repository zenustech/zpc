#include "ConcurrencyPrimitive.hpp"

#include <time.h>

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
#endif

namespace zs {

  void await_change(std::atomic<i32> &v, i32 cur) {
    while (true) {
      // spin lock
      for (int i = 0; i != 1024; ++i) {
        if (v.load(std::memory_order_relaxed) != cur)  // finally changes
          return;
        pause_cpu();
      }

      Futex::wait(&v, cur);  // system call
    }
  }
  void await_equal(std::atomic<i32> &v, i32 desired) {
    u32 cur{};
    while (true) {
      // spin lock
      for (int i = 0; i != 1024; ++i) {
        cur = v.load(std::memory_order_relaxed);
        if (cur == desired)  // finally equals
          return;
        pause_cpu();
      }

      Futex::wait(&v, cur, (i32)1 << (desired & (i32)0x1f));  // system call
    }
  }

  // int futex(int *uaddr, int op, int val, const struct timespec *timeout, int *uaddr2, int val3);
  // long syscall(SYS_futex, u32 *uaddr, int op, u32 val, const timespec *, u32 *uaddr2, u32 val3);
  // bool WaitOnAddress(volatile void *addr, void *compareAddress, size_t, addressSize, dword dwMs)
  Futex::result_t Futex::wait(std::atomic<i32> *v, i32 expected, i32 waitMask) {
    return waitUntil(v, expected, (i64)-1, waitMask);
  }
  Futex::result_t Futex::waitUntil(std::atomic<i32> *v, i32 expected, i64 deadline, i32 waitMask) {
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    /// windows
    i32 undesired = expected;
    bool rc = WaitOnAddress((void *)v, &undesired, sizeof(i32),
                            deadline == (i64)-1 ? INFINITE : deadline);
    if (rc) return result_t::awoken;
    if (undesired != expected) return result_t::value_changed;
    if (GetLastError() == ERROR_TIMEOUT) return result_t::timedout;
    return result_t::interrupted;

#elif defined(__clang__) || defined(__GNUC__)
    /// linux
    struct timespec tm {};
    struct timespec *timeout = nullptr;
    if (deadline > -1) {
      // seconds, nanoseconds
      tm = timespec{deadline / 1000, (deadline % 1000) * 1000000};
      timeout = &tm;
    }
    int const op = FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG;
    long rc
        = syscall(SYS_futex, reinterpret_cast<i32 *>(v), op, expected, timeout, nullptr, waitMask);
    if (rc == 0)
      return result_t::awoken;
    else {
      switch (rc) {
        case ETIMEDOUT:
          return result_t::timedout;
        case EINTR:
          return result_t::interrupted;
        case EWOULDBLOCK:
          return result_t::value_changed;
        default:
          return result_t::value_changed;
      }
    }
#endif
  }
  // wake up the thread if (wakeMask & waitMask == true)
  // WakeByAddressSingle/All
  int Futex::wake(std::atomic<i32> *v, int count, i32 wakeMask) {
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    if (count == limits<int>::max()) {
      WakeByAddressAll((void *)v);
      return limits<int>::max();
    } else if (count > 0) {
      for (int i = 0; i != count; ++i) WakeByAddressSingle((void *)v);
      return count;
    } else
      return 0;

#elif defined(__clang__) || defined(__GNUC__)
    int const op = FUTEX_WAKE_BITSET | FUTEX_PRIVATE_FLAG;
    long rc = syscall(SYS_futex, reinterpret_cast<i32 *>(v), op, count, nullptr, nullptr, wakeMask);
    if (rc < 0) return 0;
    return rc;
#endif
  }

  /// https://locklessinc-com.translate.goog/articles/mutex_cv_futex/?_x_tr_sl=en&_x_tr_tl=en&_x_tr_hl=en-US&_x_tr_pto=op,sc
  void Mutex::lock() {
    int c;
    /* Spin and try to take lock */
    for (int i = 0; i != 128; ++i) {
      c = 0;
      if (m.compare_exchange_strong(c, 1)) return;
      pause_cpu();
    }

    /* The lock is now contended */
    if (c == 1) {
      c = m.exchange(2);
    }

    while (c) {
      /* Wait in the kernel */
      Futex::wait(&m, 2);
      c = m.exchange(2);
    }
  }

  void Mutex::unlock() {
    /* Unlock, and if not contended then exit. */
    if (m.load() == 2) {
      m.store(0);
    } else if (m.exchange(0) == 1)
      return;

    /* Spin and hope someone takes the lock */
    for (int i = 0; i != 128; ++i) {
      if (m.load()) { /* Need to set to state 2 because there may be waiters */
        i32 c = 1; 
        bool switched = m.compare_exchange_strong(c, 2);
        if (switched || c > 0) return;
      }
      pause_cpu();
    }

    /* We need to wake someone up */
    Futex::wake(&m, 1);
    return;
  }

  bool Mutex::trylock() {
    i32 c = 0; 
    if (m.compare_exchange_strong(c, 1)) return true;
	  return false;
  }

}  // namespace zs