#include "ConcurrencyPrimitive.hpp"

#include <atomic>
#include <condition_variable>
#include <ctime>
#include <mutex>
#include <thread>
#include <chrono>

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
    /// @note ref: meta folly
    constexpr u64 twang_mix64(u64 key) noexcept {
      key = (~key) + (key << 21);  // key *= (1 << 21) - 1; key -= 1;
      key = key ^ (key >> 24);
      key = key + (key << 3) + (key << 8);  // key *= 1 + (1 << 3) + (1 << 8)
      key = key ^ (key >> 14);
      key = key + (key << 2) + (key << 4);  // key *= 1 + (1 << 2) + (1 << 4)
      key = key ^ (key >> 28);
      key = key + (key << 31);  // key *= 1 + (1 << 31)
      return key;
    }
    constexpr u64 twang_unmix64(u64 key) noexcept {
      key *= 4611686016279904257U;
      key ^= (key >> 28) ^ (key >> 56);
      key *= 14933078535860113213U;
      key ^= (key >> 14) ^ (key >> 28) ^ (key >> 42) ^ (key >> 56);
      key *= 15244667743933553977U;
      key ^= (key >> 24) ^ (key >> 48);
      key = (key + 1) * 9223367638806167551U;
      return key;
    }
    struct WaitNodeBase {
      const u64 _key;
      const u64 _lotid;
      std::mutex _mtx;  // neither copyable nor movable
      std::condition_variable _cv;
      bool _signaled;

      WaitNodeBase(u64 key, u64 lotid)
          : _key{key}, _lotid{lotid}, _mtx{}, _cv{}, _signaled{false} {}

      std::cv_status waitFor(i64 waitMs = -1) noexcept {
        using namespace std::chrono_literals;
        std::cv_status status = std::cv_status::no_timeout;
        std::unique_lock<std::mutex> lk(_mtx);
        /// @note may be woken spuriously rather than signaling
        while (!_signaled && status != std::cv_status::timeout) {
          if (waitMs > -1)
            status = _cv.wait_for(lk, waitMs * 1ms);
          else
            _cv.wait(lk);
        }
        return status;
      }

      void wake() noexcept {
        std::lock_guard<std::mutex> lk(_mtx);
        _signaled = true;
        _cv.notify_one();
      }

      bool signaled() const noexcept { return _signaled; }
    };

    struct WaitNode final : WaitNodeBase {
      WaitNode *_next;
      const u32 _data;  // wait/wake mask type

      WaitNode(u64 key, u64 lotid, u32 data) noexcept
          : WaitNodeBase{key, lotid}, _next{nullptr}, _data(FWD(data)) {}
    };

    struct WaitQueue {
      std::mutex _mtx{};
      WaitNode *_list{nullptr};
      std::atomic<u64> _count{0};

      static WaitQueue *get_queue(u64 key) {
        /// @note must be power of two
        static constexpr size_t num_queues = 4096;
        static WaitQueue queues[num_queues];
        return &queues[key & (num_queues - 1)];
      }

      WaitQueue() = default;
      ~WaitQueue() {
        WaitNode *node = _list;
        while (node != nullptr) {
          auto cur = node;
          node = node->_next;
          delete cur;
        }
      }

      [[nodiscard]] WaitNode *insertHead(const WaitNode &newNode_) {
        WaitNode *newNode = new WaitNode{newNode_._key, newNode_._lotid, newNode_._data};
        newNode->_next = _list;
        _list = newNode;
        return _list;
      }
      void erase(WaitNode *node) {
        if (_list == node) {
          delete _list;
          _list = _list->_next;
          return;
        }
        WaitNode *cur = _list->_next, *prev = _list;
        while (cur != nullptr) {
          if (node == cur) {
            prev->_next = cur->_next;
            delete cur;
            return;
          }
          prev = cur;
          cur = cur->_next;
        }
      }
    };

    static std::atomic<u32> g_idCache = 0;

    template <typename Data> struct ParkingLot {
      static_assert(std::is_trivially_destructible_v<Data>,
                    "Data type here should be both trivially and nothrow destructible!");

      const u32 _lotid;
      ParkingLot() noexcept : _lotid{g_idCache++} {}
      ParkingLot(const ParkingLot &) = delete;

      // @note Key is generally the address of a variable
      // @note D is generally the waitmask
      template <typename Key,            // index wait queue
                typename D,              // wait mask
                typename ParkCondition,  // lambda called before parking threads
                typename PreWait         // lambda called right before putting to actual sleep
                >
      ParkResult parkFor(const Key bits, D &&data, ParkCondition &&parkCondition, PreWait &&preWait,
                         i64 timeoutMs) noexcept {
        u64 key = twang_mix64((u64)bits);
        WaitQueue *queue = WaitQueue::get_queue(key);
        WaitNode node{key, _lotid, (u32)FWD(data)};
        WaitNode *pnode = nullptr;
        {
          queue->_count.fetch_add(1, std::memory_order_seq_cst);
          std::unique_lock queueLock{queue->_mtx};
          if (!FWD(parkCondition)()) {
            // current one being put to sleep is already awoken by another thread
            queueLock.unlock();
            queue->_count.fetch_sub(1, std::memory_order_relaxed);
            return ParkResult::Skip;
          }

          pnode = queue->insertHead(node);
        }
        FWD(preWait)();

        auto status = (*pnode).waitFor(timeoutMs);
        if (status == std::cv_status::timeout) {
          std::lock_guard queueLock{queue->_mtx};
          if (!(*pnode).signaled()) {
            queue->erase(pnode);
            return ParkResult::Timeout;
          }
        }
        return ParkResult::Unpark;
      }

      template <typename Key, typename Unparker> void unpark(const Key bits, Unparker &&func) {
        u64 key = twang_mix64((u64)bits);
        WaitQueue *queue = WaitQueue::get_queue(key);

        // std::atomic_thread_fence(std::memory_order_seq_cst);
        if (queue->_count.load(std::memory_order_seq_cst) == 0) return;

        std::lock_guard queueLock(queue->_mtx);
        WaitNode *st = queue->_list;
        while (st != nullptr) {
          auto &node = *st;
          st = st->_next;
          if (node._key == key && node._lotid == _lotid) {
            UnparkControl result = FWD(func)(node._data);
            if (result == UnparkControl::RemoveBreak || result == UnparkControl::RemoveContinue) {
              queue->erase(st);
              node.wake();
            }
            if (result == UnparkControl::RemoveBreak || result == UnparkControl::RetainBreak) {
              return;
            }
          }
        }
      }
    };
  }  // namespace detail

  static detail::ParkingLot<u32> g_lot;

  static int emulated_futex_wake(void *addr, int count = detail::deduce_numeric_max<int>(),
                                 u32 wakeMask = 0xffffffff) {
    int woken = 0;
    g_lot.unpark(addr, [&count, &woken, wakeMask](u32 const &mask) {
      if ((mask & wakeMask) == 0) return detail::UnparkControl::RetainContinue;
      count--;
      woken++;
      return count > 0 ? detail::UnparkControl::RemoveContinue : detail::UnparkControl::RemoveBreak;
    });
    return woken;
  }

  static FutexResult emulated_futex_wait_for(std::atomic<u32> *addr, u32 expected,
                                             i64 duration = -1, u32 waitMask = 0xffffffff) {
    detail::ParkResult res;
    res = g_lot.parkFor(
        addr, waitMask, [&]() -> bool { return addr->load(std::memory_order_seq_cst) == expected; },
        []() {}, duration);
    switch (res) {
      case detail::ParkResult::Skip:
        return FutexResult::value_changed;
      case detail::ParkResult::Unpark:
        return FutexResult::awoken;
      case detail::ParkResult::Timeout:
        return FutexResult::timedout;
    }
    return FutexResult::interrupted;
  }

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
          // Futex::wait_for(this, newState, (i64)-1, _kMask);
          g_lot.parkFor(
              this, _kMask, [this, newState]() { return this->load() == newState; }, []() {}, -1);
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
      // Futex::wake(this, limits<int>::max(), _kMask);
      g_lot.unpark(this, [](const u32 &) { return detail::UnparkControl::RemoveBreak; });
    }
  }

  bool Mutex::try_lock() noexcept {
    u32 state = this->load(std::memory_order_relaxed);
    do {
      if (state) return false;
    } while (!this->compare_exchange_weak(state, state | _kMask, std::memory_order_acquire,
                                          std::memory_order_relaxed));
    return true;
  }

  void ConditionVariable::wait(Mutex &lk) noexcept {
    g_lot.parkFor(
        &seq, 0 /* mask data used for unpark is irrelevant here*/,
        [this]() {
          seq.store(1);
          return true;
        },
        [&lk]() { lk.unlock(); }, (i64)-1);
    lk.lock();
  }
  CvStatus ConditionVariable::wait_for(Mutex &lk, i64 duration) noexcept {
    detail::ParkResult res = g_lot.parkFor(
        &seq, 0 /* mask data used for unpark is irrelevant here*/,
        [this]() {
          seq.store(1);
          return true;
        },
        [&lk]() { lk.unlock(); }, duration);
    lk.lock();
    return res == detail::ParkResult::Timeout ? CvStatus::timeout : CvStatus::no_timeout;
#if 0 
    // https://www.cnblogs.com/bbqzsl/p/6808176.html
    if (m != &mut) {
      if (m != nullptr) return false;  // invalid argument
      atomic_cas(seq_c, (void **)&m, (void *)nullptr, (void *)&mut);
      if (m != &mut) return false;  // invalid argument
    }
    m->unlock();
    Futex::wait(&seq, seq.load(std::memory_order_consume));
    // value (257) is coupled with mutex internal representation
    while (m->exchange(257, std::memory_order_acq_rel) & 1) Futex::wait(m, 257);
    return true;
#endif
  }

  void ConditionVariable::notify_one() noexcept {
    if (!seq.load(std::memory_order_relaxed)) return;
    g_lot.unpark(&seq, [](const u32 &) { return detail::UnparkControl::RemoveBreak; });
  }

  // WakeByAddressAll((void *)&seq);
  // syscall(SYS_futex, reinterpret_cast<i32 *>(&seq), FUTEX_REQUEUE | FUTEX_PRIVATE_FLAG, 1,
  //         limits<i32>::max(), m, 0);
  void ConditionVariable::notify_all() noexcept {
    if (!seq.load(std::memory_order_relaxed)) return;
    seq.store(0, std::memory_order_release);
    g_lot.unpark(&seq, [](const u32 &) { return detail::UnparkControl::RemoveContinue; });
  }

}  // namespace zs