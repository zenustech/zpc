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

  // ref
  // https://locklessinc-com.translate.goog/articles/mutex_cv_futex/
  void Mutex::lock() {
#if 0
    int c;
    /* Spin and try to take lock */
    for (int i = 0; i != 128; ++i) {
      c = 0;
      if (this->compare_exchange_strong(c, 1, std::memory_order_acq_rel)) return;
      pause_cpu();
    }

    /* The lock is now contended */
    if (c == 1) {
      c = this->exchange(2, std::memory_order_acq_rel);
    }

    while (c) {
      /* Wait in the kernel */
      Futex::wait(this, 2);
      c = this->exchange(2, std::memory_order_acq_rel);
    }
#else
    for (int i = 0; i != 128; ++i) {
      if ((this->fetch_or(1, std::memory_order_acq_rel) & 1) == 0) return;
      pause_cpu();
    }

    while ((this->exchange(257) & 1) == 1)
      Futex::wait(this, 257);
#endif
  }

  void Mutex::unlock() {
#if 0
    /* Unlock, and if not contended then exit. */
    if (this->load(std::memory_order_consume) == 2) {
      this->store(0, std::memory_order_release);
    } else if (this->exchange(0, std::memory_order_acq_rel) == 1)
      return;

    /* Spin and hope someone takes the lock */
    for (int i = 0; i != 128; ++i) {
      if (this->load(std::memory_order_consume)) { /* Need to set to state 2 because there may be waiters */
        i32 c = 1; 
        bool switched = this->compare_exchange_strong(c, 2, std::memory_order_acq_rel);
        if (switched || c > 0) return;
      }
      pause_cpu();
    }

    /* We need to wake someone up */
    Futex::wake(this, 1);
    return;
#else
    /* Locked and not contended */
    if (this->load(std::memory_order_consume) == 1) {
      i32 c = 1;
      if (this->compare_exchange_strong(c, 0)) 
        return;
    }
    /* Unlock */
    this->fetch_and(~1, std::memory_order_acq_rel);  // bit-wise not
    std::atomic_thread_fence(std::memory_order_seq_cst);  // https://stackoverflow.com/questions/19965076/gcc-memory-barrier-sync-synchronize-vs-asm-volatile-memory

    /* Spin and hope someone takes the lock */
    for (int i = 0; i != 128; ++i) {
      if ((this->load(std::memory_order_consume) & 1) == 1)
        return;
      pause_cpu();
    }

    /* We need to wake someone up */
    this->fetch_and(~256, std::memory_order_acq_rel);
    Futex::wake(this, 1);
#endif
  }

  bool Mutex::trylock() {
    i32 c = 0; 
    if (this->compare_exchange_strong(c, 1, std::memory_order_acq_rel)) return true;
	  return false; // resource busy or locked
  }

  void ConditionVariable::notify_one() {
    seq.fetch_add(1, std::memory_order_acq_rel);
    Futex::wake(&seq, 1);
  }

  void ConditionVariable::notify_all() {
    if (m == nullptr) return;
    seq.fetch_add(1, std::memory_order_acq_rel);
#if defined(_MSC_VER) || (defined(_WIN32) && defined(__INTEL_COMPILER))
    // cannot find win32 alternative for FUTEX_REQUEUE
    WakeByAddressAll((void *)&seq);
#elif defined(__clang__) || defined(__GNUC__)
    syscall(SYS_futex, reinterpret_cast<i32 *>(&seq), FUTEX_REQUEUE | FUTEX_PRIVATE_FLAG, 1, limits<i32>::max(), m, 0);
#endif
  }

  bool ConditionVariable::wait(Mutex &mut) {
    // https://www.cnblogs.com/bbqzsl/p/6808176.html
    if (m != &mut) {
      if (m != nullptr) return false; // invalid argument
      atomic_cas(exec_seq, (void **)&m, (void *)nullptr, (void *)&mut);
      if (m != &mut) return false; // invalid argument
    }
    m->unlock();
    Futex::wait(&seq, seq.load(std::memory_order_consume));
    // value (257) is coupled with mutex internal representation
    while (m->exchange(257, std::memory_order_acq_rel) & 1)
      Futex::wait(m, 257);
    return true;
  }

}  // namespace zs