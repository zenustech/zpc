#pragma once
#include "zensim/types/Object.h"
#include "zensim/types/Optional.h"

namespace zs {

  // executor or scheduler
  // should use optional since the error can be frequent
  struct Executor : Inherit<Object, Executor> {
    template <typename F> decltype(auto) operator()(F&& f) const { return f(); }
  };

  struct synchronous_scheduler {
    template <typename F> decltype(auto) operator()(F&& f) const { return f(); }
  };

  struct asynchronous_scheduler {
    template <typename F> decltype(auto) operator()(F&& f) const {
      auto fut = std::async(f);
      return fut.get();
    }
  };

  struct world_s_best_thread_pool {
    template <typename F> void operator()(F&& f) {
      std::thread{std::forward<decltype(f)>(f)}.detach();
    }
  };

}  // namespace zs