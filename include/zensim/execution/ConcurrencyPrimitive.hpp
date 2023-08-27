#pragma once
#include <any>
#include <atomic>
#include <list>
#include <mutex>
#include <condition_variable>

#include "zensim/TypeAlias.hpp"
#include "zensim/meta/Functional.h"

namespace zs {

#if 0
  struct ttas_lock {
    void lock() {
      for (;;) {
        if (!_lock.exchange(true, std::memory_order_acquire))
          break;
        while (_lock.load(std::memory_order_relaxed))
          pause_cpu();
      }
    }

    std::atomic<bool> _lock;
  };
#endif

  namespace detail {
    struct WaitNodeBase {
      const u64 _key;
      const u64 _lotid;
      std::mutex _mtx;
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
          if (waitMs != -1)
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

    struct WaitQueue {
      std::mutex _mtx;
      std::list<std::unique_ptr<WaitNodeBase>> _list;
      std::atomic<u64> _count;

      static WaitQueue *get_queue(u64 key) {
        static constexpr size_t num_queues = 4096;
        static WaitQueue queues[num_queues];
        return &queues[key & (num_queues - 1)];
      }

      void pushBack(std::unique_ptr<WaitNodeBase> &&node) noexcept {
        _list.push_front(std::move(node));
      }

      void erase(WaitNodeBase *node) noexcept {
        _list.remove_if([node](const std::unique_ptr<WaitNodeBase> &e) { return e.get() == node; });
      }
    };

    extern std::atomic<u32> g_idCache;

    enum UnparkControl { RetainContinue, RemoveContinue, RetainBreak, RemoveBreak };
    enum ParkResult { Skip, Unpark, Timeout };

    template <typename Data> struct ParkingLot {
      static_assert(std::is_trivially_destructible_v<Data>,
                    "Data type here should be both trivially and nothrow destructible!");

      const u32 _lotid;
      ParkingLot() noexcept : _lotid{g_idCache++} {}
      ParkingLot(const ParkingLot &) = delete;

      struct WaitNode final : WaitNodeBase {
        const Data _data;

        template <typename T> WaitNode(u64 key, u64 lotid, T &&data) noexcept
            : WaitNodeBase{key, lotid}, _data(FWD(data)) {}
      };

      // @note Key is generally the address of a variable
      // @note D is generally the waitmask
      template <typename Key,            // index wait queue
                typename D,              // wait mask
                typename ParkCondition,  // lambda called before parking threads
                typename PreWait         // lambda called right before putting to actual sleep
                >
      ParkResult parkFor(const Key bits, D &&data, ParkCondition &&parkCondition, PreWait &&preWait,
                         i64 timeoutMs) noexcept {
        u64 key = (u64)bits;
        WaitQueue *queue = WaitQueue::get_queue(key);
        auto pnode = std::make_unique<WaitNode>(key, _lotid, FWD(data));
        auto node = pnode.get();
        {
          queue->_count.fetch_add(1, std::memory_order_seq_cst);
          std::unique_lock queueLock{queue->_mtx};
          if (!FWD(parkCondition)()) {
            // current one being put to sleep is already awoken by another thread
            queueLock.unlock();
            queue->_count.fetch_sub(1, std::memory_order_relaxed);
            return ParkResult::Skip;
          }

          queue->pushBack(std::move(pnode));
        }
        FWD(preWait)();

        auto status = node->waitFor(timeoutMs);
        if (status == std::cv_status::timeout) {
          std::lock_guard queueLock{queue->_mtx};
          if (!node->signaled()) {
            queue->erase(node);
            return ParkResult::Timeout;
          }
        }
        return ParkResult::Unpark;
      }

      template <typename Key, typename Unparker> void unpark(const Key bits, Unparker &&func) {
        u64 key = (u64)bits;
        WaitQueue *queue = WaitQueue::get_queue(key);

        if (queue->_count.load(std::memory_order_seq_cst) == 0) return;

        std::lock_guard queueLock(queue->_mtx);
        for (std::unique_ptr<WaitNodeBase> &iter : queue->_list) {
          auto node = static_cast<WaitNode *>(iter.get());
          if (node->_key == key && node->_lotid == _lotid) {
            auto result = FWD(func)(node->_data);
            if (result == UnparkControl::RemoveBreak || result == UnparkControl::RemoveContinue) {
              queue->erase(node);
              node->wake();
            }
            if (result == UnparkControl::RemoveBreak || result == UnparkControl::RetainContinue) {
              return;
            }
          }
        }
      }
    };
  }  // namespace detail

  extern detail::ParkingLot<u32> g_lot;

  int emulated_futex_wake(void *addr, int count = limits<int>::max(), u32 wakeMask = 0xffffffff) {
    int woken = 0;
    g_lot.unpark(addr, [&count, &woken, &wakeMask](u32 const &mask) {
      if ((mask & wakeMask) == 0) return detail::UnparkControl::RetainContinue;
      count--;
      woken++;
      return count > 0 ? detail::UnparkControl::RemoveContinue : detail::UnparkControl::RemoveBreak;
    });
    return woken;
  }
  enum FutexResult {
    value_changed,  // when expected != atomic value
    awoken,         // awoken from 'wake' (success state)
    interrupted,    // interrupted by certain signal
    timedout        // not applicable to wait, applicable to wait until
  };

  FutexResult emulated_futex_wait_for(std::atomic<u32> *addr, u32 expected, i64 duration = -1,
                                      u32 waitMask = 0xffffffff) {
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


  /// @note ref: Multithreading 101 concurrency primitive from scratch
  /// shared within a single process
  /// blocking construct in the context of shared-memory synchronization
  struct ZPC_API Futex {
    // put the current thread to sleep if the expected value matches the value in the atomic
    // waitmask will be saved and compared to the wakemask later in the wake call
    // to check if you wanna wake up this thread or keep it sleeping
    static FutexResult wait(std::atomic<u32> *v, u32 expected, u32 waitMask = 0xffffffff);
    /// @note duration in milli-seconds
    static FutexResult wait_for(std::atomic<u32> *v, u32 expected, i64 duration = -1,
                              u32 waitMask = 0xffffffff);
    // wake up the thread if (wakeMask & waitMask == true)
    static int wake(std::atomic<u32> *v, int count = limits<int>::max(), u32 wakeMask = 0xffffffff);
  };

  ZPC_API void await_change(std::atomic<u32> &v, u32 cur);
  ZPC_API void await_equal(std::atomic<u32> &v, u32 desired);

  // process-local mutex
  struct ZPC_API Mutex : std::atomic<u32> {
    // 0: unlocked
    // 1: locked
    // 257: locked and contended (...0001 | 00000001)
    void lock();
    void unlock();
    bool trylock();
  };

  // 8 bytes alignment for rollover issue
  // https://docs.ntpsec.org/latest/rollover.html
  struct alignas(16) ConditionVariable {
    void notify_one();
    void notify_all();
    bool wait(Mutex &mut);

    Mutex *m{nullptr};        // 4 bytes, the cv belongs to this mutex
    std::atomic<u32> seq{0};  // 4 bytes, sequence lock for concurrent wakes and sleeps
  };

#if 0
  struct Mutex : std::atomic<u8> {
    void lock() noexcept {
      u8 state = this->load(std::memory_order_relaxed);
      if (likely((state & kIsLocked) == 0
                 && this->compare_exchange_weak(&state, state | kIsLocked,
                                                std::memory_order_acquire,
                                                std::memory_order_relaxed)))
        return;
      lockImpl(state);
    }
    bool try_lock() noexcept {
      u8 state = this->load(std::memory_order_relaxed);
      do {
        if (state & kIsLocked) return false;
      } while (!this->compare_exchange_weak(&state, state | kIsLocked, std::memory_order_acquire,
                                            std::memory_order_relaxed));
      return true;
    }
    void unlock() noexcept {
      u8 oldState = this->load(std::memory_order_relaxed), newState;
      do {
        newState = oldState & ~(kIsLocked | kIsParked);
      } while (!this->compare_exchange_weak(&oldState, newState, std::memory_order_acquire,
                                            std::memory_order_relaxed));
      if (oldState & kIsParked) unlockImpl();
    }

  private:
    void lockImpl(u8 oldState) noexcept {
      size_t spinCount = 0;
      static constexpr size_t spinLimit = 1000;
      static constexpr size_t yieldLimit = 1;
      u8 newState;
      u8 needPark = 0;
    retry:
      if ((oldState & kIsLocked) != 0) {
        ++spinCount;
        if (spinCount > spinLimit + yieldLimit) {
          newState = oldState | kIsParked;
          if (newState != oldState) {
            if (!this->compare_exchange_weak(&oldState, newState, std::memory_order_acquire,
                                             std::memory_order_relaxed))
              goto retry;
          }
        } else if (spinCount > spinLimit) {
          THREAD_YIELD();
        } else {
          THREAD_PAUSE();
        }
        oldState = this->load(std::memory_order_relaxed);
        goto retry;
      }
      newState = oldState | kIsLocked | needPark;
      if (!this->compare_exchange_weak(&oldState, newState, std::memory_order_acquire,
                                       std::memory_order_relaxed))
        goto retry;
    }
    void unlockImpl() noexcept { ; }
  };
#endif

#if 0

  class condition_variable {
    using steady_clock = chrono::steady_clock;
    using system_clock = chrono::system_clock;
#  ifdef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
    using __clock_t = steady_clock;
#  else
    using __clock_t = system_clock;
#  endif
    typedef __gthread_cond_t __native_type;

#  ifdef __GTHREAD_COND_INIT
    __native_type _M_cond = __GTHREAD_COND_INIT;
#  else
    __native_type _M_cond;
#  endif

  public:
    typedef __native_type* native_handle_type;

    condition_variable() noexcept;
    ~condition_variable() noexcept;

    condition_variable(const condition_variable&) = delete;
    condition_variable& operator=(const condition_variable&) = delete;

    void notify_one() noexcept;

    void notify_all() noexcept;

    void wait(unique_lock<mutex>& __lock) noexcept;

    template <typename _Predicate> void wait(unique_lock<mutex>& __lock, _Predicate __p) {
      while (!__p()) wait(__lock);
    }

#  ifdef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
    template <typename _Duration>
    cv_status wait_until(unique_lock<mutex>& __lock,
                         const chrono::time_point<steady_clock, _Duration>& __atime) {
      return __wait_until_impl(__lock, __atime);
    }
#  endif

    template <typename _Duration>
    cv_status wait_until(unique_lock<mutex>& __lock,
                         const chrono::time_point<system_clock, _Duration>& __atime) {
      return __wait_until_impl(__lock, __atime);
    }

    template <typename _Clock, typename _Duration>
    cv_status wait_until(unique_lock<mutex>& __lock,
                         const chrono::time_point<_Clock, _Duration>& __atime) {
#  if __cplusplus > 201703L
      static_assert(chrono::is_clock_v<_Clock>);
#  endif
      const typename _Clock::time_point __c_entry = _Clock::now();
      const __clock_t::time_point __s_entry = __clock_t::now();
      const auto __delta = __atime - __c_entry;
      const auto __s_atime = __s_entry + __delta;

      if (__wait_until_impl(__lock, __s_atime) == cv_status::no_timeout)
        return cv_status::no_timeout;
      // We got a timeout when measured against __clock_t but
      // we need to check against the caller-supplied clock
      // to tell whether we should return a timeout.
      if (_Clock::now() < __atime) return cv_status::no_timeout;
      return cv_status::timeout;
    }

    template <typename _Clock, typename _Duration, typename _Predicate>
    bool wait_until(unique_lock<mutex>& __lock,
                    const chrono::time_point<_Clock, _Duration>& __atime, _Predicate __p) {
      while (!__p())
        if (wait_until(__lock, __atime) == cv_status::timeout) return __p();
      return true;
    }

    template <typename _Rep, typename _Period>
    cv_status wait_for(unique_lock<mutex>& __lock, const chrono::duration<_Rep, _Period>& __rtime) {
      using __dur = typename steady_clock::duration;
      auto __reltime = chrono::duration_cast<__dur>(__rtime);
      if (__reltime < __rtime) ++__reltime;
      return wait_until(__lock, steady_clock::now() + __reltime);
    }

    template <typename _Rep, typename _Period, typename _Predicate>
    bool wait_for(unique_lock<mutex>& __lock, const chrono::duration<_Rep, _Period>& __rtime,
                  _Predicate __p) {
      using __dur = typename steady_clock::duration;
      auto __reltime = chrono::duration_cast<__dur>(__rtime);
      if (__reltime < __rtime) ++__reltime;
      return wait_until(__lock, steady_clock::now() + __reltime, std::move(__p));
    }

    native_handle_type native_handle() { return &_M_cond; }

  private:
#  ifdef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
    template <typename _Dur>
    cv_status __wait_until_impl(unique_lock<mutex>& __lock,
                                const chrono::time_point<steady_clock, _Dur>& __atime) {
      auto __s = chrono::time_point_cast<chrono::seconds>(__atime);
      auto __ns = chrono::duration_cast<chrono::nanoseconds>(__atime - __s);

      __gthread_time_t __ts = {static_cast<std::time_t>(__s.time_since_epoch().count()),
                               static_cast<long>(__ns.count())};

      pthread_cond_clockwait(&_M_cond, __lock.mutex()->native_handle(), CLOCK_MONOTONIC, &__ts);

      return (steady_clock::now() < __atime ? cv_status::no_timeout : cv_status::timeout);
    }
#  endif

    template <typename _Dur>
    cv_status __wait_until_impl(unique_lock<mutex>& __lock,
                                const chrono::time_point<system_clock, _Dur>& __atime) {
      auto __s = chrono::time_point_cast<chrono::seconds>(__atime);
      auto __ns = chrono::duration_cast<chrono::nanoseconds>(__atime - __s);

      __gthread_time_t __ts = {static_cast<std::time_t>(__s.time_since_epoch().count()),
                               static_cast<long>(__ns.count())};

      __gthread_cond_timedwait(&_M_cond, __lock.mutex()->native_handle(), &__ts);

      return (system_clock::now() < __atime ? cv_status::no_timeout : cv_status::timeout);
    }
  };
#endif

}  // namespace zs