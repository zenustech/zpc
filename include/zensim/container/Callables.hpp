#pragma once
#include <map>

#include "zensim/ZpcFunction.hpp"

namespace zs {

  namespace detail {
    struct DummyMutex {
      static constexpr void lock() noexcept { return; }
      static constexpr void unlock() noexcept { return; }
      static constexpr bool try_lock() noexcept { return true; }
    };
  }  // namespace detail

  template <class F, typename M = detail::DummyMutex> struct callbacks;

  template <class... Args, class M>

  struct callbacks<void(Args...), M> : M {
    static_assert((!is_rvalue_reference_v<Args> && ...), "callbacks do not allow rvalue arguments");
    using Id = i32;
    using F = function<void(Args...)>;

    callbacks() = default;
    ~callbacks() = default;
    callbacks(callbacks&& o) noexcept : _funcs{zs::move(o._funcs)}, _n{o._n} { o._n = 0; }
    callbacks& operator=(callbacks&& o) noexcept {
      _funcs = zs::move(o._funcs);
      _n = o._n;
      o._n = 0;
      return *this;
    }
    callbacks(const callbacks& o) noexcept : _funcs{o._funcs}, _n{o._n} {}
    callbacks& operator=(const callbacks& o) noexcept {
      _funcs = o._funcs;
      _n = o._n;
      return *this;
    }

    template <typename F, enable_if_t<is_invocable_r_v<void, F&&, Args...>> = 0> Id insert(F&& f) {
      Id ret;
      M::lock();
      ret = _n++;
      _funcs.emplace(ret, FWD(f));
      M::unlock();
      return ret;
    }
    template <typename F, enable_if_t<is_invocable_r_v<void, F&&, Args...>> = 0>
    callbacks& operator=(F&& f) {
      clear();
      insert(FWD(f));
      return *this;
    }
    template <typename F, enable_if_t<is_invocable_r_v<void, F&&, Args...>> = 0>
    void assign(Id id, F&& f) {
      M::lock();
      _funcs.insert_or_assign(id, FWD(f));
      M::unlock();
    }

    void erase(Id i) {
      M::lock();
      _funcs.erase(i);
      M::unlock();
    }
    void clear() {
      M::lock();
      _funcs.clear();
      M::unlock();
    }

    explicit operator bool() const noexcept { return _funcs.size() != 0; }
    void operator()(Args... args) const {
      M::lock();
      for (const auto& [id, f] : _funcs) f(forward<Args>(args)...);
      M::unlock();
    }

  private:
    std::map<Id, F> _funcs{};
    Id _n = 0;
  };

}  // namespace zs