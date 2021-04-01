#pragma once

#include <omp.h>

#include <execution>
#include <numeric>

#include "Concurrency.h"
#include "zensim/TypeAlias.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Function.h"
#include "zensim/types/Iterator.h"
namespace zs {

  enum struct execspace_e : char { host = 0, openmp = 0, cuda, hip };

  struct DeviceHandle {
    NodeID nodeid{0};   ///<
    ProcID procid{-1};  ///< processor id (cpu: negative, gpu: positive)
  };

  struct ParallelTask {
    // vulkan compute: ***.spv
    // lambda functions or functors (with host or device decoration)
    // cuda module file
    // string literals
    std::string source{};
    std::function<void()> func;
  };

  /// execution policy
  template <typename Derived> struct ExecutionPolicyInterface {
    bool launch(const ParallelTask &kernel) const noexcept { return selfPtr()->do_launch(kernel); }
    bool shouldSync() const noexcept { return selfPtr()->do_shouldSync(); }
    bool shouldWait() const noexcept { return selfPtr()->do_shouldWait(); }

    Derived &sync(bool sync_) noexcept {
      _sync = sync_;
      return *selfPtr();
    }

    // constexpr DeviceHandle device() const noexcept { return handle; }

  protected:
    constexpr bool do_launch(const ParallelTask &) const noexcept { return false; }
    constexpr bool do_shouldSync() const noexcept { return _sync; }
    constexpr bool do_shouldWait() const noexcept { return _wait; }

    constexpr Derived *selfPtr() noexcept { return static_cast<Derived *>(this); }
    constexpr const Derived *selfPtr() const noexcept { return static_cast<const Derived *>(this); }

    bool _sync{true}, _wait{false};
    // DeviceHandle handle{0, -1};
  };

  struct SequentialExecutionPolicy : ExecutionPolicyInterface<SequentialExecutionPolicy> {
    template <typename Range, typename F> constexpr void operator()(Range &&range, F &&f) const {
      using fts = function_traits<F>;
      if constexpr (fts::arity == 0) {
        for (auto &&it : range) f();
      } else {
        for (auto &&it : range) {
          if constexpr (is_tuple<remove_cvref_t<decltype(it)>>::value)
            std::apply(f, it);
          else
            f(it);
        }
      }
    }

    template <std::size_t I, std::size_t... Is, typename... Iters, typename... Policies,
              typename... Ranges, typename... Bodies>
    constexpr void exec(index_seq<Is...> indices, std::tuple<Iters...> prefixIters,
                        const zs::tuple<Policies...> &policies, const zs::tuple<Ranges...> &ranges,
                        const Bodies &...bodies) const {
      using Range = zs::select_indexed_type<I, std::decay_t<Ranges>...>;
      const auto &range = zs::get<I>(ranges);
      if constexpr (I + 1 == sizeof...(Ranges)) {
        for (auto &&it : range) {
          const auto args = shuffle(indices, std::tuple_cat(prefixIters, std::make_tuple(it)));
          (std::apply(FWD(bodies), args), ...);
        }
      } else if constexpr (I + 1 < sizeof...(Ranges)) {
        auto &policy = zs::get<I + 1>(policies);
        for (auto &&it : range)
          policy.template exec<I + 1>(indices, std::tuple_cat(prefixIters, std::make_tuple(it)),
                                      policies, ranges, bodies...);
      }
    }

  protected:
    bool do_launch(const ParallelTask &) const noexcept;
    bool do_sync() const noexcept { return true; }
    friend struct ExecutionPolicyInterface<SequentialExecutionPolicy>;
  };

  /// use pragma syntax instead of attribute syntax
  struct OmpExecutionPolicy : ExecutionPolicyInterface<OmpExecutionPolicy> {
    // EventID eventid{0}; ///< event id
    template <typename Range, typename F> void operator()(Range &&range, F &&f) const {
      using fts = function_traits<F>;
#pragma omp parallel num_threads(_dop)
#pragma omp master
      for (auto &&it : range)
#pragma omp task
      {
        if constexpr (fts::arity == 0) {
          f();
        } else {
          if constexpr (is_tuple<remove_cvref_t<decltype(it)>>::value)
            std::apply(f, it);
          else
            f(it);
        }
      }
    }

    template <std::size_t I, std::size_t... Is, typename... Iters, typename... Policies,
              typename... Ranges, typename... Bodies>
    void exec(index_seq<Is...> indices, std::tuple<Iters...> prefixIters,
              const zs::tuple<Policies...> &policies, const zs::tuple<Ranges...> &ranges,
              const Bodies &...bodies) const {
      using Range = zs::select_indexed_type<I, std::decay_t<Ranges>...>;
      const auto &range = zs::get<I>(ranges);
      auto ed = range.end();
      if constexpr (I + 1 == sizeof...(Ranges)) {
#pragma omp parallel num_threads(_dop)
#pragma omp master
        for (auto &&it : range)
#pragma omp task
        {
          const auto args = shuffle(indices, std::tuple_cat(prefixIters, std::make_tuple(it)));
          (std::apply(FWD(bodies), args), ...);
        }
      } else if constexpr (I + 1 < sizeof...(Ranges)) {
        auto &policy = zs::get<I + 1>(policies);
#pragma omp parallel num_threads(_dop)
#pragma omp master
        for (auto &&it : range)
#pragma omp task
        {
          policy.template exec<I + 1>(indices, std::tuple_cat(prefixIters, std::make_tuple(it)),
                                      policies, ranges, bodies...);
        }
      }
    }

    OmpExecutionPolicy &threads(int numThreads) noexcept {
      _dop = numThreads;
      return *this;
    }

  protected:
    friend struct ExecutionPolicyInterface<OmpExecutionPolicy>;

    int _dop{1};
  };

  struct CudaExecutionPolicy;

  struct sequential_execution_tag {};
  struct omp_execution_tag {};
  struct cuda_execution_tag {};
  struct hip_execution_tag {};

  constexpr sequential_execution_tag exec_seq{};
  constexpr omp_execution_tag exec_omp{};
  constexpr cuda_execution_tag exec_cuda{};
  constexpr hip_execution_tag exec_hip{};

  constexpr SequentialExecutionPolicy par_exec(sequential_execution_tag) noexcept {
    return SequentialExecutionPolicy{};
  }
  constexpr SequentialExecutionPolicy seq_exec() noexcept { return SequentialExecutionPolicy{}; }
  inline OmpExecutionPolicy omp_exec() noexcept {
    return OmpExecutionPolicy{}.threads(std::thread::hardware_concurrency());
  }
  inline OmpExecutionPolicy par_exec(omp_execution_tag) noexcept {
    return OmpExecutionPolicy{}.threads(std::thread::hardware_concurrency());
  }

  /// ========================================================================
  /// kernel, for_each, reduce, scan, gather, sort
  /// ========================================================================
  /// this can only be called on host side
  template <std::size_t... Is, typename... Policies, typename... Ranges, typename... Bodies>
  constexpr void par_exec(zs::tuple<Policies...> policies, zs::tuple<Ranges...> ranges,
                          Bodies &&...bodies) {
    /// these backends should all be on the host side
    static_assert(sizeof...(Policies) == sizeof...(Ranges),
                  "there should be a corresponding policy for every range\n");
    static_assert(sizeof...(Is) == 0 || sizeof...(Is) == sizeof...(Ranges),
                  "loop index mapping not legal\n");
    using Indices
        = conditional_t<sizeof...(Is) == 0, std::index_sequence_for<Ranges...>, index_seq<Is...>>;
    if constexpr (sizeof...(Policies) == 0)
      return;
    else {
      auto &policy = zs::get<0>(policies);
      policy.template exec<0>(Indices{}, std::tuple<>{}, policies, ranges, bodies...);
    }
  }

  /// default policy is 'sequential'
  /// this should be able to be used within a kernel
  template <std::size_t... Is, typename... Ranges, typename... Bodies>
  constexpr void par_exec(zs::tuple<Ranges...> ranges, Bodies &&...bodies) {
    using SeqPolicies =
        typename gen_seq<sizeof...(Ranges)>::template uniform_types_t<zs::tuple,
                                                                      SequentialExecutionPolicy>;
    par_exec<Is...>(SeqPolicies{}, std::move(ranges), FWD(bodies)...);
  }

  // ===================== parallel pattern wrapper ====================
  /// for_each
  template <class ExecutionPolicy, class ForwardIt, class UnaryFunction>
  constexpr void for_each(ExecutionPolicy &&policy, ForwardIt &&first, ForwardIt &&last,
                          UnaryFunction &&f) {
    if constexpr (std::is_execution_policy_v<std::decay_t<ExecutionPolicy>>)
      std::for_each(FWD(policy), FWD(first), FWD(last), FWD(f));
    else
      policy.for_each(FWD(first), FWD(last), FWD(f));
  }
  /// scan
  template <class ExecutionPolicy, class InputIt, class OutputIt,
            class BinaryOperation = std::plus<remove_cvref_t<decltype(*std::declval<InputIt>())>>>
  constexpr void inclusive_scan(ExecutionPolicy &&policy, InputIt &&first, InputIt &&last,
                                OutputIt &&d_first, BinaryOperation &&binary_op = {}) {
    if constexpr (std::is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)
      std::inclusive_scan(FWD(policy), FWD(first), FWD(last), FWD(d_first), FWD(binary_op),
                          monoid_op<BinaryOperation>::e);
    else
      policy.inclusive_scan(FWD(first), FWD(last), FWD(d_first), FWD(binary_op));
  }
  template <class ExecutionPolicy, class InputIt, class OutputIt,
            class T = remove_cvref_t<decltype(*std::declval<InputIt>())>,
            class BinaryOperation = std::plus<T>>
  constexpr void exclusive_scan(ExecutionPolicy &&policy, InputIt &&first, InputIt &&last,
                                OutputIt &&d_first, T init = monoid_op<BinaryOperation>::e,
                                BinaryOperation &&binary_op = {}) {
    if constexpr (std::is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)
      std::exclusive_scan(FWD(policy), FWD(first), FWD(last), FWD(d_first), init, FWD(binary_op));
    else
      policy.exclusive_scan(FWD(first), FWD(last), FWD(d_first), init, FWD(binary_op));
  }
  /// reduce
  template <class ExecutionPolicy, class InputIt, class OutputIt, class T,
            class BinaryOp = std::plus<T>>
  constexpr void reduce(ExecutionPolicy &&policy, InputIt &&first, InputIt &&last,
                        OutputIt &&d_first, T init, BinaryOp &&binary_op = {}) {
    if constexpr (std::is_execution_policy_v<remove_cvref_t<ExecutionPolicy>>)
      *d_first = std::reduce(FWD(policy), FWD(first), FWD(last), init, FWD(binary_op));
    else
      policy.reduce(FWD(first), FWD(last), FWD(d_first), init, FWD(binary_op));
  }
  /// sort
  /// gather/ select (flagged, if, unique)

}  // namespace zs