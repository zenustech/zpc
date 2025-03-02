#pragma once

#include <cassert>
#include <numeric>

#include "zensim/TypeAlias.hpp"
#include "zensim/ZpcFunction.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/memory/MemoryResource.h"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"
#include "zensim/types/Polymorphism.h"
#include "zensim/types/Property.h"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/types/Tuple.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include "zensim/zpc_tpls/magic_enum/magic_enum.hpp"
namespace zs {

  using exec_tags = variant<seq_exec_tag, omp_exec_tag, cuda_exec_tag, musa_exec_tag, rocm_exec_tag,
                            sycl_exec_tag>;

  constexpr const char *execution_space_tag[] = {"HOST", "OPENMP", "CUDA", "MUSA", "ROCm", "SYCL"};
  constexpr const char *get_execution_tag_name(execspace_e execpol) {
    return execution_space_tag[magic_enum::enum_integer(execpol)];
  }

  constexpr exec_tags suggest_exec_space(const MemoryLocation &mloc) {
    switch (mloc.memspace()) {
      case memsrc_e::host:
#if ZS_ENABLE_OPENMP
        return omp_c;
#else
        return seq_c;
#endif
      case memsrc_e::device:
      case memsrc_e::um:
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
        return cuda_c;
#elif ZS_ENABLE_MUSA && defined(__MUSACC__)
        return musa_c;
#elif ZS_ENABLE_ROCM && defined(__HIPCC__)
        return rocm_c;  // __HIP_PLATFORM_AMD__
#elif ZS_ENABLE_SYCL && defined(SYCL_LANGUAGE_VERSION)
        return sycl_c;
#endif
      default:
        throw std::runtime_error(
            fmt::format("no valid execution space suggestions for the memory handle [{}, {}]\n",
                        get_memory_tag_name(mloc.memspace()), (int)mloc.devid()));
        return seq_c;
    }
  }

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
    function<void()> func;
  };

  // granularity between thread and block (cuda:thread_block/ rocm:work_group)
  // cuda: thread_block_tile<B, cg::thread_block>
  // rocm: wavefront<32/64>
  // cpu: core (SIMD)
  // sycl: sub_group
  struct SubGroup {
    ;
  };

  // if identity cannot be deduced, trigger SFINAE in case of misuse.
  template <typename BinaryOp, typename T,
            typename = decltype(monoid<remove_cvref_t<BinaryOp>>::identity())>
  constexpr auto deduce_identity() {
    return monoid<remove_cvref_t<BinaryOp>>::identity();
  }
#if 0
  template <typename BinaryOp, typename T> constexpr auto deduce_identity() {
    constexpr auto canExtractIdentity = is_valid(
        [](auto t) -> decltype((void)monoid<remove_cvref_t<decltype(t)>>::identity()) {});
    if constexpr (canExtractIdentity(wrapt<BinaryOp>{}))
      return monoid<remove_cvref_t<BinaryOp>>::identity();
    else
      return T{};
  }
#endif

#define assert_with_msg(exp, msg) assert(((void)msg, exp))

  /// execution policy
  template <typename Derived> struct ExecutionPolicyInterface {
    bool launch(const ParallelTask &kernel) const noexcept { return selfPtr()->do_launch(kernel); }
    bool shouldSync() const noexcept { return selfPtr()->do_shouldSync(); }
    bool shouldWait() const noexcept { return selfPtr()->do_shouldWait(); }
    bool shouldProfile() const noexcept { return selfPtr()->do_shouldProfile(); }

    Derived &sync(bool sync_) noexcept {
      _sync = sync_;
      return *selfPtr();
    }
    Derived &profile(bool profile_) noexcept {
      _profile = profile_;
      return *selfPtr();
    }

    // constexpr DeviceHandle device() const noexcept { return handle; }

  protected:
    constexpr bool do_launch(const ParallelTask &) const noexcept { return false; }
    constexpr bool do_shouldSync() const noexcept { return _sync; }
    constexpr bool do_shouldWait() const noexcept { return _wait; }
    constexpr bool do_shouldProfile() const noexcept { return _profile; }

    constexpr Derived *selfPtr() noexcept { return static_cast<Derived *>(this); }
    constexpr const Derived *selfPtr() const noexcept { return static_cast<const Derived *>(this); }

    bool _sync{true}, _wait{false}, _profile{false};
    // DeviceHandle handle{0, -1};
  };

  struct _zs_policy_assign_operator {
    template <typename Src, typename Dst, typename ParamT>
    constexpr void operator()(const Src &src, Dst &dst, ParamT &&) noexcept {
      dst = src;
    }
  };

  struct SequentialExecutionPolicy;
  ZPC_API extern ZSPmrAllocator<> get_temporary_memory_source(const SequentialExecutionPolicy &pol);

  struct SequentialExecutionPolicy : ExecutionPolicyInterface<SequentialExecutionPolicy> {
    using exec_tag = seq_exec_tag;
    template <typename Range, typename F>
    constexpr void operator()(Range &&range, F &&f,
                              const source_location &loc = source_location::current()) const {
      constexpr auto hasBegin = is_valid(
          [](auto t) -> decltype((void)std::begin(declval<typename decltype(t)::type>())) {});
      constexpr auto hasEnd = is_valid(
          [](auto t) -> decltype((void)std::end(declval<typename decltype(t)::type>())) {});
      CppTimer timer;
      if (shouldProfile()) timer.tick();
      if constexpr (!hasBegin(wrapt<Range>{}) || !hasEnd(wrapt<Range>{})) {
        /// for iterator-like range (e.g. openvdb)
        /// for openvdb parallel iteration...
        auto iter = FWD(range);  // otherwise fails on win
        for (; iter; ++iter) {
          if constexpr (is_invocable_v<F>)
            zs::invoke(f);
          else
            zs::invoke(f, iter);
        }
      } else {
        for (auto &&it : range) {
          if constexpr (is_invocable_v<F, decltype(it)>)
            zs::invoke(f, it);
          else if constexpr (is_std_tuple_v<remove_cvref_t<decltype(it)>>)
            std::apply(f, it);
          else if constexpr (is_tuple_v<remove_cvref_t<decltype(it)>>)
            zs::apply(f, it);
          else if constexpr (is_invocable_v<F>)
            zs::invoke(f);
          else
            static_assert(always_false<F>, "unable to handle this callable and the range.");
        }
      }
      if (shouldProfile())
        timer.tock(fmt::format("[Seq Exec | File {}, Ln {}, Col {}]", loc.file_name(), loc.line(),
                               loc.column()));
    }
    template <typename Range, typename ParamTuple, typename F,
              enable_if_t<is_tuple_v<remove_cvref_t<ParamTuple>>> = 0>
    constexpr void operator()(Range &&range, ParamTuple &&params, F &&f,
                              const source_location &loc = source_location::current()) const {
      constexpr auto hasBegin = is_valid(
          [](auto t) -> decltype((void)std::begin(declval<typename decltype(t)::type>())) {});
      constexpr auto hasEnd = is_valid(
          [](auto t) -> decltype((void)std::end(declval<typename decltype(t)::type>())) {});
      CppTimer timer;
      if (shouldProfile()) timer.tick();
      if constexpr (!hasBegin(wrapt<Range>{}) || !hasEnd(wrapt<Range>{})) {
        /// for iterator-like range (e.g. openvdb)
        /// for openvdb parallel iteration...
        auto iter = FWD(range);  // otherwise fails on win
        for (; iter; ++iter) {
          if constexpr (is_invocable_v<F, ParamTuple>) {
            zs::invoke(f, params);
          } else {
            zs::invoke(f, iter, params);
          }
        }
      } else {
        for (auto &&it : range) {
          if constexpr (is_invocable_v<F, decltype(it), ParamTuple>)
            zs::invoke(f, it, params);
          else if constexpr (is_std_tuple_v<remove_cvref_t<decltype(it)>>)
            std::apply(f, std::tuple_cat(it, std::tie(params)));
          else if constexpr (is_tuple_v<remove_cvref_t<decltype(it)>>)
            zs::apply(f, zs::tuple_cat(it, zs::tie(params)));
          else if constexpr (is_invocable_v<F, ParamTuple>)
            zs::invoke(f, params);
          else
            static_assert(always_false<F>, "unable to handle this callable and the range.");
        }
      }
      if (shouldProfile())
        timer.tock(fmt::format("[Seq Exec | File {}, Ln {}, Col {}]", loc.file_name(), loc.line(),
                               loc.column()));
    }

    template <zs::size_t I, size_t... Is, typename... Iters, typename... Policies,
              typename... Ranges, typename... Bodies>
    constexpr void exec(index_sequence<Is...> indices, zs::tuple<Iters...> prefixIters,
                        const zs::tuple<Policies...> &policies, const zs::tuple<Ranges...> &ranges,
                        const Bodies &...bodies) const {
      // using Range = zs::select_indexed_type<I, decay_t<Ranges>...>;
      const auto &range = zs::get<I>(ranges);
      if constexpr (I + 1 == sizeof...(Ranges)) {
        for (auto &&it : range) {
          const auto args = shuffle(indices, zs::tuple_cat(prefixIters, zs::make_tuple(it)));
          (zs::apply(FWD(bodies), args), ...);
        }
      } else if constexpr (I + 1 < sizeof...(Ranges)) {
        auto &policy = zs::get<I + 1>(policies);
        for (auto &&it : range)
          policy.template exec<I + 1>(indices, zs::tuple_cat(prefixIters, zs::make_tuple(it)),
                                      policies, ranges, bodies...);
      }
    }

    /// serial version of several parallel primitives
    template <class ForwardIt, class UnaryFunction>
    constexpr void for_each(ForwardIt &&first, ForwardIt &&last, UnaryFunction &&f,
                            const source_location &loc = source_location::current()) const {
      (*this)(detail::iter_range(FWD(first), FWD(last)), FWD(f));
    }

    template <class InputIt, class OutputIt,
              class BinaryOperation = plus<remove_cvref_t<decltype(*declval<InputIt>())>>>
    constexpr void inclusive_scan(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                                  BinaryOperation &&binary_op = {},
                                  const source_location &loc = source_location::current()) const {
      if (first == last) return;
      auto prev = *(d_first++) = *(first++);
      while (first != last) *(d_first++) = prev = binary_op(prev, *(first++));
    }
    template <class InputIt, class OutputIt,
              class T = remove_cvref_t<decltype(*declval<InputIt>())>,
              class BinaryOperation = plus<T>>
    constexpr void exclusive_scan(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                                  T init = deduce_identity<BinaryOperation, T>(),
                                  BinaryOperation &&binary_op = {},
                                  const source_location &loc = source_location::current()) const {
      if (first == last) return;
      *(d_first++) = init;
      while (first + 1 != last) *(d_first++) = init = binary_op(init, *(first++));
    }
    template <class InputIt, class OutputIt,
              class BinaryOp = plus<remove_cvref_t<decltype(*declval<InputIt>())>>>
    constexpr void reduce(
        InputIt &&first, InputIt &&last, OutputIt &&d_first,
        remove_cvref_t<decltype(*declval<InputIt>())> init
        = deduce_identity<BinaryOp, remove_cvref_t<decltype(*declval<InputIt>())>>(),
        BinaryOp &&binary_op = {}, const source_location &loc = source_location::current()) const {
      for (; first != last;) init = binary_op(init, *(first++));
      *d_first = init;
    }
    template <class KeyIter,
              typename CompareOpT
              = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
    void sort(KeyIter &&first, KeyIter &&last, CompareOpT &&compOp = {},
              const source_location &loc = source_location::current()) {
      std::sort(FWD(first), FWD(last), FWD(compOp));
    }
    template <class KeyIter,
              typename CompareOpT
              = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
    void merge_sort(KeyIter &&first, KeyIter &&last, CompareOpT &&compOp = {},
                    const source_location &loc = source_location::current()) {
      std::stable_sort(FWD(first), FWD(last), FWD(compOp));
    }

    template <typename KeyIter, typename ValueIter, typename DiffT, typename CompareOpT>
    static void quick_sort_impl(KeyIter &&keys, ValueIter &&vals, DiffT l, DiffT r,
                                CompareOpT &&compOp) {
      // ref: https://www.geeksforgeeks.org/quick-sort/
      DiffT pi = l;
      if (l < r) {
        const auto &pivot = keys[r];
        for (DiffT j = l; j != r; ++j) {
          if (keys[j] < pivot) {
            std::swap(keys[pi], keys[j]);
            std::swap(vals[pi], vals[j]);
            pi++;
          }
        }
        std::swap(keys[pi], keys[r]);
        std::swap(vals[pi], vals[r]);
        quick_sort_impl(keys, vals, l, pi - 1, compOp);
        quick_sort_impl(keys, vals, pi + 1, r, compOp);
      }
    }
    template <typename KeyIter, typename ValueIter, typename CompareOpT, bool Stable>
    void merge_sort_pair_impl(
        KeyIter &&keys, ValueIter &&vals,
        typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type dist,
        CompareOpT &&compOp, wrapv<Stable>, const source_location &loc) const {
      using KeyIterT = remove_cvref_t<KeyIter>;
      using ValueIterT = remove_cvref_t<ValueIter>;
      using DiffT = typename std::iterator_traits<KeyIterT>::difference_type;
      using KeyT = typename std::iterator_traits<KeyIterT>::value_type;
      using ValueT = typename std::iterator_traits<ValueIterT>::value_type;

      CppTimer timer;
      if (shouldProfile()) timer.tick();

      auto allocator = get_temporary_memory_source(*this);
      Vector<KeyT> okeys_{allocator, (size_t)dist};
      Vector<ValueT> ovals_{allocator, (size_t)dist};
      // std::vector<KeyT> okeys_(dist);
      // std::vector<ValueT> ovals_(dist);
      auto okeys = std::begin(okeys_);
      auto ovals = std::begin(ovals_);

      bool switched = false;

      {
        const DiffT l = 0;
        const DiffT r = dist;

        bool flipped = false;

        /// @note currently [unstable] adopts the [stable] routine
        if constexpr (Stable) {
          //  bottom-up fashion
          // insertion sort for segments of granularity <= 16
          for (DiffT ll = l; ll < r;) {
            auto rr = std::min(ll + 16, r);
            // [ll, rr)
            for (DiffT i = ll + 1; i != rr; ++i) {
              for (DiffT k = i; k != ll; --k) {  // insert k
                auto j = k - 1;
                if (compOp(keys[k], keys[j])) {
                  std::swap(keys[k], keys[j]);
                  std::swap(vals[k], vals[j]);
                } else
                  break;
              }
            }
            ll = rr;
          }
          for (DiffT halfStride = 16; halfStride < (r - l);) {
            DiffT stride = halfStride * 2;
            // auto bgCur = flipped ? okeys : keys;
            // auto bgCurVals = flipped ? ovals : vals;
            // auto bgNext = flipped ? keys : okeys;
            // auto bgNextVals = flipped ? vals : ovals;
            if (flipped) {
              for (DiffT ll = l; ll < r; ll += stride) {
                DiffT mid = std::min(ll + halfStride, r);
                DiffT rr = std::min(ll + stride, r);
                // [ll, mid) [mid, rr)
                DiffT left = ll, right = mid, k = ll;
                while (left < mid && right < rr) {
                  const auto &a = okeys[left];
                  const auto &b = okeys[right];
                  if (!compOp(b, a)) {
                    keys[k] = a;
                    vals[k++] = ovals[left++];
                  } else {
                    keys[k] = b;
                    vals[k++] = ovals[right++];
                  }
                }
                while (left < mid) {
                  keys[k] = okeys[left];
                  vals[k++] = ovals[left++];
                }
                while (right < rr) {
                  keys[k] = okeys[right];
                  vals[k++] = ovals[right++];
                }
              }
            } else {
              for (DiffT ll = l; ll < r; ll += stride) {
                DiffT mid = std::min(ll + halfStride, r);
                DiffT rr = std::min(ll + stride, r);
                // [ll, mid) [mid, rr)
                DiffT left = ll, right = mid, k = ll;
                while (left < mid && right < rr) {
                  const auto &a = keys[left];
                  const auto &b = keys[right];
                  if (!compOp(b, a)) {
                    okeys[k] = a;
                    ovals[k++] = vals[left++];
                  } else {
                    okeys[k] = b;
                    ovals[k++] = vals[right++];
                  }
                }
                while (left < mid) {
                  okeys[k] = keys[left];
                  ovals[k++] = vals[left++];
                }
                while (right < rr) {
                  okeys[k] = keys[right];
                  ovals[k++] = vals[right++];
                }
              }
            }
            flipped = !flipped;
            halfStride = stride;
          }
        } else {
          quick_sort_impl(keys, vals, l, r - 1, compOp);
        }

        switched = flipped;

        if (switched) {
          for (DiffT k = l; k < r; ++k) {
            keys[k] = okeys[k];
            vals[k] = ovals[k];
          }
        }
      }

      if (shouldProfile())
        timer.tock(fmt::format("[Seq merge_sort_pair | File {}, Ln {}, Col {}]", loc.file_name(),
                               loc.line(), loc.column()));
    }
    template <typename KeyIter, typename ValueIter,
              typename CompareOpT
              = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
    void sort_pair(
        KeyIter &&keys, ValueIter &&vals,
        typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type count,
        CompareOpT &&compOp = {}, const source_location &loc = source_location::current()) const {
      merge_sort_pair_impl(FWD(keys), FWD(vals), count, FWD(compOp), false_c, loc);  // unstable
    }
    template <class KeyIter, class ValueIter,
              typename CompareOpT
              = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
    void merge_sort_pair(
        KeyIter &&keys, ValueIter &&vals,
        typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type count,
        CompareOpT &&compOp = {}, const source_location &loc = source_location::current()) const {
      merge_sort_pair_impl(FWD(keys), FWD(vals), count, FWD(compOp), true_c, loc);  // stable
    }
    template <class InputIt, class OutputIt>
    void radix_sort(InputIt &&first, InputIt &&last, OutputIt &&d_first, int sbit = 0,
                    int ebit
                    = sizeof(typename std::iterator_traits<remove_reference_t<InputIt>>::value_type)
                      * 8,
                    const source_location &loc = source_location::current()) const {
      using IterT = remove_cvref_t<InputIt>;
      using DiffT = typename std::iterator_traits<IterT>::difference_type;
      using InputValueT = typename std::iterator_traits<IterT>::value_type;

      const auto dist = last - first;
      bool skip = false;
      constexpr int binBits = 8;  // by byte
      int binCount = 1 << binBits;
      int binMask = binCount - 1;

      auto allocator = get_temporary_memory_source(*this);
      Vector<DiffT> binGlobalSizes{allocator, (size_t)binCount};
      Vector<DiffT> binOffsets{allocator, (size_t)binCount};
      // std::vector<DiffT> binGlobalSizes(binCount);
      // std::vector<DiffT> binOffsets(binCount);

      Vector<InputValueT> buffers[2] = {{allocator, (size_t)dist}, {allocator, (size_t)dist}};
      // buffers[0].resize(dist);
      // buffers[1].resize(dist);
      InputValueT *cur{buffers[0].data()}, *next{buffers[1].data()};

      /// sign-related handling
      for (DiffT i = 0; i < dist; ++i) {
        if constexpr (is_signed_v<InputValueT>)
          cur[i] = *(first + i) ^ ((InputValueT)1 << (sizeof(InputValueT) * 8 - 1));
        else
          cur[i] = *(first + i);
      }

      for (int st = sbit; st < ebit; st += binBits) {
        if (st + binBits > ebit) {
          binMask >>= (st + binBits - ebit);
          binCount >>= (st + binBits - ebit);
        }

        for (DiffT i = 0; i < binCount; ++i) binGlobalSizes[i] = 0;
        for (DiffT i = 0; i < dist; ++i) binGlobalSizes[(cur[i] >> st) & binMask]++;

        binOffsets[0] = 0;
        skip = binGlobalSizes[0] == dist;
        for (int i = 1; i < binCount; ++i) {
          if (binGlobalSizes[i] == dist) {
            skip = true;
            break;
          }
          binOffsets[i] = binOffsets[i - 1] + binGlobalSizes[i - 1];
        }
        if (!skip) {
          for (int i = 0; i < binCount; i++) binGlobalSizes[i] += binOffsets[i];

          for (DiffT i = dist - 1; i >= 0; --i)
            next[--binGlobalSizes[(cur[i] >> st) & binMask]] = cur[i];
          std::swap(cur, next);
        }
      }

      /// sign-related handling
      for (DiffT i = 0; i < dist; ++i) {
        if constexpr (is_signed_v<InputValueT>)
          *(d_first + i) = cur[i] ^ ((InputValueT)1 << (sizeof(InputValueT) * 8 - 1));
        else
          *(d_first + i) = cur[i];
      }
    }
    template <class KeyIter, class ValueIter,
              typename Tn
              = typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type>
    void radix_sort_pair(
        KeyIter &&keysIn, ValueIter &&valsIn, KeyIter &&keysOut, ValueIter &&valsOut, Tn count = 0,
        int sbit = 0,
        int ebit
        = sizeof(typename std::iterator_traits<remove_reference_t<KeyIter>>::value_type) * 8,
        const source_location &loc = source_location::current()) const {
      using KeyT = typename std::iterator_traits<remove_reference_t<KeyIter>>::value_type;
      using ValueT = typename std::iterator_traits<remove_reference_t<ValueIter>>::value_type;
      using DiffT = typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type;

      const auto dist = count;
      bool skip = false;
      constexpr int binBits = 8;  // by byte
      int binCount = 1 << binBits;
      int binMask = binCount - 1;

      auto allocator = get_temporary_memory_source(*this);
      Vector<DiffT> binGlobalSizes{allocator, (size_t)binCount};
      Vector<DiffT> binOffsets{allocator, (size_t)binCount};

      Vector<KeyT> keyBuffers[2] = {{allocator, (size_t)count}, {allocator, (size_t)count}};
      Vector<ValueT> valBuffers[2] = {{allocator, (size_t)count}, {allocator, (size_t)count}};
      // keyBuffers[0].resize(count);
      // keyBuffers[1].resize(count);
      // valBuffers[0].resize(count);
      // valBuffers[1].resize(count);
      KeyT *cur{keyBuffers[0].data()}, *next{keyBuffers[1].data()};
      ValueT *curVals{valBuffers[0].data()}, *nextVals{valBuffers[1].data()};

      /// sign-related handling
      for (DiffT i = 0; i < dist; ++i) {
        if constexpr (is_signed_v<KeyT>)
          cur[i] = *(keysIn + i) ^ ((KeyT)1 << (sizeof(KeyT) * 8 - 1));
        else
          cur[i] = *(keysIn + i);
        curVals[i] = *(valsIn + i);
      }

      for (int st = sbit; st < ebit; st += binBits) {
        if (st + binBits > ebit) {
          binMask >>= (st + binBits - ebit);
          binCount >>= (st + binBits - ebit);
        }

        for (DiffT i = 0; i < binCount; ++i) binGlobalSizes[i] = 0;
        for (DiffT i = 0; i < dist; ++i) binGlobalSizes[(cur[i] >> st) & binMask]++;

        binOffsets[0] = 0;
        skip = binGlobalSizes[0] == dist;
        for (int i = 1; i < binCount; ++i) {
          if (binGlobalSizes[i] == dist) {
            skip = true;
            break;
          }
          binOffsets[i] = binOffsets[i - 1] + binGlobalSizes[i - 1];
        }
        if (!skip) {
          for (int i = 0; i < binCount; i++) binGlobalSizes[i] += binOffsets[i];

          for (DiffT i = dist - 1; i >= 0; --i) {
            // next[--binGlobalSizes[(cur[i] >> st) & binMask]] = cur[i];
            const auto loc = --binGlobalSizes[(cur[i] >> st) & binMask];
            next[loc] = cur[i];
            nextVals[loc] = curVals[i];
          }
          std::swap(cur, next);
          std::swap(curVals, nextVals);
        }
      }

      /// sign-related handling
      for (DiffT i = 0; i < dist; ++i) {
        if constexpr (is_signed_v<KeyT>)
          *(keysOut + i) = cur[i] ^ ((KeyT)1 << (sizeof(KeyT) * 8 - 1));
        else
          *(keysOut + i) = cur[i];
        *(valsOut + i) = curVals[i];
      }
    }

  protected:
    bool do_launch(const ParallelTask &) const noexcept;
    bool do_sync() const noexcept { return true; }
    friend struct ExecutionPolicyInterface<SequentialExecutionPolicy>;
  };

  struct CudaExecutionPolicy;
  struct OmpExecutionPolicy;

  constexpr SequentialExecutionPolicy par_exec(seq_exec_tag) noexcept {
    return SequentialExecutionPolicy{};
  }
  constexpr SequentialExecutionPolicy seq_exec() noexcept { return SequentialExecutionPolicy{}; }

  /// ========================================================================
  /// kernel, for_each, reduce, scan, gather, sort
  /// ========================================================================
  /// this can only be called on host side
  template <size_t... Is, typename... Policies, typename... Ranges, typename... Bodies>
  constexpr void par_exec(zs::tuple<Policies...> policies, zs::tuple<Ranges...> ranges,
                          Bodies &&...bodies) {
    /// these backends should all be on the host side
    static_assert(sizeof...(Policies) == sizeof...(Ranges),
                  "there should be a corresponding policy for every range\n");
    static_assert(sizeof...(Is) == 0 || sizeof...(Is) == sizeof...(Ranges),
                  "loop index mapping not legal\n");
    using Indices
        = conditional_t<sizeof...(Is) == 0, index_sequence_for<Ranges...>, index_sequence<Is...>>;
    if constexpr (sizeof...(Policies) == 0)
      return;
    else {
      auto &policy = zs::get<0>(policies);
      policy.template exec<0>(Indices{}, zs::tuple<>{}, policies, ranges, bodies...);
    }
  }

  /// default policy is 'sequential'
  /// this should be able to be used within a kernel
  template <size_t... Is, typename... Ranges, typename... Bodies>
  constexpr void par_exec(zs::tuple<Ranges...> ranges, Bodies &&...bodies) {
    using SeqPolicies =
        typename build_seq<sizeof...(Ranges)>::template uniform_types_t<zs::tuple,
                                                                        SequentialExecutionPolicy>;
    par_exec<Is...>(SeqPolicies{}, std::move(ranges), FWD(bodies)...);
  }

  //
  template <typename ExecPol> constexpr bool is_backend_available(ExecPol = {}) noexcept {
    return false;
  }

  template <typename ExecTag> constexpr void assert_backend_presence(ExecTag) noexcept {
    if constexpr (is_same_v<ExecTag, omp_exec_tag>) {
#if ZS_ENABLE_OPENMP && (defined(_OPENMP) || defined(ZS_COMPILER_INTEL_LLVM))
      static_assert(is_same_v<ExecTag, omp_exec_tag>, "zs openmp compiler not activated here");
#else
      static_assert(!is_same_v<ExecTag, omp_exec_tag>, "openmp compiler not activated here");
#endif
    } else if constexpr (is_same_v<ExecTag, cuda_exec_tag>) {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
      static_assert(is_same_v<ExecTag, cuda_exec_tag>, "zs openmp compiler not activated here");
#else
      static_assert(!is_same_v<ExecTag, cuda_exec_tag>, "cuda compiler not activated here");
#endif
    } else if constexpr (is_same_v<ExecTag, seq_exec_tag>) {
      // always present
    }
  }
  template <execspace_e space> constexpr void assert_backend_presence() noexcept {
    assert_backend_presence(wrapv<space>{});
  }

  // ===================== parallel pattern wrapper ====================
  /// for_each
  template <class ExecutionPolicy, class ForwardIt, class UnaryFunction>
  constexpr void for_each(ExecutionPolicy &&policy, ForwardIt &&first, ForwardIt &&last,
                          UnaryFunction &&f,
                          const source_location &loc = source_location::current()) {
    policy.for_each(FWD(first), FWD(last), FWD(f), loc);
  }
  /// transform
  template <class ExecutionPolicy, class ForwardIt, class UnaryFunction>
  constexpr void transform(ExecutionPolicy &&policy, ForwardIt &&first, ForwardIt &&last,
                           UnaryFunction &&f,
                           const source_location &loc = source_location::current()) {
    policy.for_each(FWD(first), FWD(last), FWD(f), loc);
  }
  /// scan
  template <class ExecutionPolicy, class InputIt, class OutputIt,
            class BinaryOperation
            = plus<typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>>
  constexpr void inclusive_scan(ExecutionPolicy &&policy, InputIt &&first, InputIt &&last,
                                OutputIt &&d_first, BinaryOperation &&binary_op = {},
                                const source_location &loc = source_location::current()) {
    policy.inclusive_scan(FWD(first), FWD(last), FWD(d_first), FWD(binary_op), loc);
  }
  template <class ExecutionPolicy, class InputIt, class OutputIt,
            class BinaryOperation
            = plus<typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>>
  constexpr void exclusive_scan(
      ExecutionPolicy &&policy, InputIt &&first, InputIt &&last, OutputIt &&d_first,
      typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type init
      = deduce_identity<BinaryOperation,
                        typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>(),
      BinaryOperation &&binary_op = {}, const source_location &loc = source_location::current()) {
    policy.exclusive_scan(FWD(first), FWD(last), FWD(d_first), init, FWD(binary_op), loc);
  }
  /// reduce
  template <class ExecutionPolicy, class InputIt, class OutputIt,
            class BinaryOp
            = plus<typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>>
  constexpr void reduce(
      ExecutionPolicy &&policy, InputIt &&first, InputIt &&last, OutputIt &&d_first,
      typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type init
      = deduce_identity<BinaryOp,
                        typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>(),
      BinaryOp &&binary_op = {}, const source_location &loc = source_location::current()) {
    policy.reduce(FWD(first), FWD(last), FWD(d_first), init, FWD(binary_op), loc);
  }
  /// sort
  template <typename ExecutionPolicy, class KeyIter, class ValueIter,
            typename CompareOpT
            = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
  void sort_pair(ExecutionPolicy &&policy, KeyIter &&keys, ValueIter &&vals,
                 typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type count,
                 CompareOpT &&compOp = {},
                 const source_location &loc = source_location::current()) {
    policy.sort_pair(FWD(keys), FWD(vals), count, FWD(compOp), loc);
  }
  template <typename ExecutionPolicy, class KeyIter,
            typename CompareOpT
            = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
  void sort(ExecutionPolicy &&policy, KeyIter &&first, KeyIter &&last, CompareOpT &&compOp = {},
            const source_location &loc = source_location::current()) {
    policy.sort(FWD(first), FWD(last), FWD(compOp), loc);
  }
  /// merge sort
  template <typename ExecutionPolicy, class KeyIter, class ValueIter,
            typename CompareOpT
            = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
  void merge_sort_pair(
      ExecutionPolicy &&policy, KeyIter &&keys, ValueIter &&vals,
      typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type count,
      CompareOpT &&compOp = {}, const source_location &loc = source_location::current()) {
    policy.merge_sort_pair(FWD(keys), FWD(vals), count, FWD(compOp), loc);
  }
  template <typename ExecutionPolicy, class KeyIter,
            typename CompareOpT
            = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
  void merge_sort(ExecutionPolicy &&policy, KeyIter &&first, KeyIter &&last,
                  CompareOpT &&compOp = {},
                  const source_location &loc = source_location::current()) {
    policy.merge_sort(FWD(first), FWD(last), FWD(compOp), loc);
  }
  /// sort
  template <class ExecutionPolicy, class KeyIter, class ValueIter,
            typename Tn
            = typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type>
  constexpr enable_if_type<is_ra_iter_v<remove_reference_t<KeyIter>>> radix_sort_pair(
      ExecutionPolicy &&policy, KeyIter &&keysIn, ValueIter &&valsIn, KeyIter &&keysOut,
      ValueIter &&valsOut, Tn count, int sbit = 0,
      int ebit = sizeof(typename std::iterator_traits<remove_reference_t<KeyIter>>::value_type) * 8,
      const source_location &loc = source_location::current()) {
    policy.radix_sort_pair(FWD(keysIn), FWD(valsIn), FWD(keysOut), FWD(valsOut), count, sbit, ebit,
                           loc);
  }
  template <class ExecutionPolicy, class InputIt, class OutputIt> constexpr void radix_sort(
      ExecutionPolicy &&policy, InputIt &&first, InputIt &&last, OutputIt &&d_first, int sbit = 0,
      int ebit = sizeof(typename std::iterator_traits<remove_reference_t<InputIt>>::value_type) * 8,
      const source_location &loc = source_location::current()) {
    policy.radix_sort(FWD(first), FWD(last), FWD(d_first), sbit, ebit, loc);
  }
  /// gather/ select (flagged, if, unique)

}  // namespace zs