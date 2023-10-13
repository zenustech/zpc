#pragma once

#if !ZS_ENABLE_OPENMP
#  error "ZS_ENABLE_OPENMP was not enabled, but Omp::ExecutionPolicy.hpp was included anyway."
#endif

#if ZS_ENABLE_OPENMP && !defined(_OPENMP) && !defined(__CUDACC__)
#  error "ZS_ENABLE_OPENMP defined but the compiler is not defining the _OPENMP macro as expected"
#endif

#include <omp.h>

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Function.h"
#include "zensim/types/Iterator.h"

namespace zs {

  /// use pragma syntax instead of attribute syntax
  struct OmpExecutionPolicy : ExecutionPolicyInterface<OmpExecutionPolicy> {
    using exec_tag = omp_exec_tag;
    // EventID eventid{0}; ///< event id

    template <typename Ts, typename Is, typename F>
    void operator()(Collapse<Ts, Is> dims, F &&f,
                    const source_location &loc = source_location::current()) const {
      using namespace index_literals;
      constexpr auto dim = Collapse<Ts, Is>::dim;
      CppTimer timer;
      if (shouldProfile()) timer.tick();
      if constexpr (dim == 1) {
#pragma omp parallel for if (_dop < dims.get(0_th)) num_threads(_dop)
        for (RM_CVREF_T(dims.get(0_th)) i = 0; i < dims.get(0_th); ++i) std::invoke(f, i);
      } else if constexpr (dim == 2) {
#pragma omp parallel for collapse(2) if (_dop < dims.get(0_th) * dims.get(1_th)) num_threads(_dop)
        for (RM_CVREF_T(dims.get(0_th)) i = 0; i < dims.get(0_th); ++i)
          for (RM_CVREF_T(dims.get(1_th)) j = 0; j < dims.get(1_th); ++j) std::invoke(f, i, j);
      } else if constexpr (dim == 3) {
#pragma omp parallel for collapse(3) if (_dop < dims.get(0_th) * dims.get(1_th) * dims.get(2_th)) \
    num_threads(_dop)
        for (RM_CVREF_T(dims.get(0_th)) i = 0; i < dims.get(0_th); ++i)
          for (RM_CVREF_T(dims.get(1_th)) j = 0; j < dims.get(1_th); ++j)
            for (RM_CVREF_T(dims.get(2_th)) k = 0; k < dims.get(2_th); ++k) std::invoke(f, i, j, k);
      } else {
        throw std::runtime_error(
            fmt::format("execution of {}-layers of loops not supported!", dim));
      }
      if (shouldProfile())
        timer.tock(fmt::format("[Omp Exec | File {}, Ln {}, Col {}]", loc.file_name(), loc.line(),
                               loc.column()));
    }
    template <typename Range, typename F>
    void operator()(Range &&range, F &&f,
                    const source_location &loc = source_location::current()) const {
      CppTimer timer;
      if (shouldProfile()) timer.tick();
      constexpr auto hasBegin = is_valid(
          [](auto t) -> decltype((void)std::begin(std::declval<typename decltype(t)::type>())) {});
      constexpr auto hasEnd = is_valid(
          [](auto t) -> decltype((void)std::end(std::declval<typename decltype(t)::type>())) {});
      if constexpr (!hasBegin(wrapt<Range>{}) || !hasEnd(wrapt<Range>{})) {
        /// for iterator-like range (e.g. openvdb)
        /// for openvdb parallel iteration...
        auto iter = FWD(range);  // otherwise fails on win
#pragma omp parallel num_threads(_dop)
#pragma omp master
        for (; iter; ++iter)
#pragma omp task firstprivate(iter)
        {
          if constexpr (std::is_invocable_v<F>) {
            f();
          } else {
            std::invoke(f, iter);
          }
        }
      } else {
        /// not stl conforming iterator
        using IterT = remove_cvref_t<decltype(std::begin(range))>;
        // random iterator category
        if constexpr (std::is_convertible_v<typename std::iterator_traits<IterT>::iterator_category,
                                            std::random_access_iterator_tag>) {
          using DiffT = typename std::iterator_traits<IterT>::difference_type;
          auto iter = std::begin(range);
          const DiffT dist = std::end(range) - iter;

#pragma omp parallel for if (_dop < dist) num_threads(_dop)
          for (DiffT i = 0; i < dist; ++i) {
            if constexpr (std::is_invocable_v<F>)
              f();
            else {
              auto &&it = *(iter + i);
              if constexpr (is_std_tuple_v<remove_cvref_t<decltype(it)>>)
                std::apply(f, it);
              else if constexpr (is_tuple_v<remove_cvref_t<decltype(it)>>)
                zs::apply(f, it);
              else
                std::invoke(f, it);
            }
          }
        } else {
          // forward iterator category
#pragma omp parallel num_threads(_dop)
#pragma omp master
          for (auto &&it : range)
#pragma omp task firstprivate(it)
          {
            if constexpr (std::is_invocable_v<F>) {
              f();
            } else {
              if constexpr (is_std_tuple_v<remove_cvref_t<decltype(it)>>)
                std::apply(f, it);
              else if constexpr (is_tuple_v<remove_cvref_t<decltype(it)>>)
                zs::apply(f, it);
              else
                std::invoke(f, it);
            }
          }
        }
      }
      if (shouldProfile())
        timer.tock(fmt::format("[Omp Exec | File {}, Ln {}, Col {}]", loc.file_name(), loc.line(),
                               loc.column()));
    }

    template <std::size_t I, std::size_t... Is, typename... Iters, typename... Policies,
              typename... Ranges, typename... Bodies>
    void exec(index_seq<Is...> indices, zs::tuple<Iters...> prefixIters,
              const zs::tuple<Policies...> &policies, const zs::tuple<Ranges...> &ranges,
              const Bodies &...bodies) const {
      // using Range = zs::select_indexed_type<I, std::decay_t<Ranges>...>;
      const auto &range = zs::get<I>(ranges);
      auto ed = range.end();
      if constexpr (I + 1 == sizeof...(Ranges)) {
#pragma omp parallel num_threads(_dop)
#pragma omp master
        for (auto &&it : range)
#pragma omp task firstprivate(it)
        {
          const auto args = shuffle(indices, zs::tuple_cat(prefixIters, zs::make_tuple(it)));
          (zs::apply(bodies, args), ...);
        }
      } else if constexpr (I + 1 < sizeof...(Ranges)) {
        auto &policy = zs::get<I + 1>(policies);
#pragma omp parallel num_threads(_dop)
#pragma omp master
        for (auto &&it : range)
#pragma omp task firstprivate(it)
        {
          policy.template exec<I + 1>(indices, zs::tuple_cat(prefixIters, zs::make_tuple(it)),
                                      policies, ranges, bodies...);
        }
      }
    }

    /// for_each
    template <class ForwardIt, class UnaryFunction>
    void for_each_impl(std::random_access_iterator_tag, ForwardIt &&first, ForwardIt &&last,
                       UnaryFunction &&f,
                       const source_location &loc = source_location::current()) const {
      (*this)(detail::iter_range(FWD(first), FWD(last)), FWD(f), loc);
    }
    template <class ForwardIt, class UnaryFunction>
    void for_each(ForwardIt &&first, ForwardIt &&last, UnaryFunction &&f,
                  const source_location &loc = source_location::current()) const {
      for_each_impl(typename std::iterator_traits<remove_cvref_t<ForwardIt>>::iterator_category{},
                    FWD(first), FWD(last), FWD(f), loc);
    }

    /// inclusive scan
    template <class InputIt, class OutputIt, class BinaryOperation>
    void inclusive_scan_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                             OutputIt &&d_first, BinaryOperation &&binary_op,
                             const source_location &loc) const {
      using IterT = remove_cvref_t<InputIt>;
      using DstIterT = remove_cvref_t<OutputIt>;
      using DiffT = typename std::iterator_traits<IterT>::difference_type;
      using ValueT = typename std::iterator_traits<DstIterT>::value_type;
      static_assert(
          std::is_convertible_v<DiffT, typename std::iterator_traits<DstIterT>::difference_type>,
          "diff type not compatible");
      static_assert(std::is_convertible_v<typename std::iterator_traits<IterT>::value_type, ValueT>,
                    "value type not compatible");
      CppTimer timer;
      if (shouldProfile()) timer.tick();
      const auto dist = last - first;
      std::vector<ValueT> localRes{};
      DiffT nths{};
#pragma omp parallel if (_dop < dist) num_threads(_dop) \
    shared(dist, nths, first, last, d_first, localRes, binary_op)
      {
#pragma omp single
        {
          nths = omp_get_num_threads();
          localRes.resize(nths);
        }
#pragma omp barrier
        DiffT tid = omp_get_thread_num();
        DiffT nwork = (dist + nths - 1) / nths;
        DiffT st = nwork * tid;
        DiffT ed = st + nwork;
        if (ed > dist) ed = dist;

        ValueT res{};
        if (st < ed) {
          res = *(first + st);
          *(d_first + st) = res;
          for (auto offset = st + 1; offset < ed; ++offset) {
            res = binary_op(res, *(first + offset));
            *(d_first + offset) = res;
          }
          localRes[tid] = res;
        }
#pragma omp barrier

        ValueT tmp = res;
        for (DiffT stride = 1; stride < nths; stride *= 2) {
          if (tid >= stride && st < ed) tmp = binary_op(tmp, localRes[tid - stride]);
#pragma omp barrier
          if (tid >= stride && st < ed) localRes[tid] = tmp;
#pragma omp barrier
        }

        if (tid != 0 && st < ed) {
          tmp = localRes[tid - 1];
          for (auto offset = st; offset < ed; ++offset)
            *(d_first + offset) = binary_op(*(d_first + offset), tmp);
        }
      }
      if (shouldProfile())
        timer.tock(fmt::format("[Omp Exec | File {}, Ln {}, Col {}]", loc.file_name(), loc.line(),
                               loc.column()));
    }
    template <class InputIt, class OutputIt,
              class BinaryOperation = std::plus<remove_cvref_t<decltype(*std::declval<InputIt>())>>>
    void inclusive_scan(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                        BinaryOperation &&binary_op = {},
                        const source_location &loc = source_location::current()) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      inclusive_scan_impl(
          typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{}, FWD(first),
          FWD(last), FWD(d_first), FWD(binary_op), loc);
    }

    /// exclusive scan
    template <class InputIt, class OutputIt, class T, class BinaryOperation>
    void exclusive_scan_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                             OutputIt &&d_first, T init, BinaryOperation &&binary_op,
                             const source_location &loc) const {
      using IterT = remove_cvref_t<InputIt>;
      using DstIterT = remove_cvref_t<OutputIt>;
      using DiffT = typename std::iterator_traits<IterT>::difference_type;
      using ValueT = typename std::iterator_traits<DstIterT>::value_type;
      static_assert(
          std::is_convertible_v<DiffT, typename std::iterator_traits<DstIterT>::difference_type>,
          "diff type not compatible");
      static_assert(std::is_convertible_v<typename std::iterator_traits<IterT>::value_type, ValueT>,
                    "value type not compatible");
      CppTimer timer;
      if (shouldProfile()) timer.tick();
      const auto dist = last - first;
      std::vector<ValueT> localRes{};
      DiffT nths{};
#pragma omp parallel if (_dop < dist) num_threads(_dop) \
    shared(dist, nths, first, last, d_first, localRes, binary_op)
      {
#pragma omp single
        {
          nths = omp_get_num_threads();
          localRes.resize(nths);
        }
#pragma omp barrier
        DiffT tid = omp_get_thread_num();
        DiffT nwork = (dist + nths - 1) / nths;
        DiffT st = nwork * tid;
        DiffT ed = st + nwork;
        if (ed > dist) ed = dist;

        ValueT res{};
        if (st < ed) {
          *(d_first + st) = init;
          res = *(first + st);
          for (auto offset = st + 1; offset < ed; ++offset) {
            *(d_first + offset) = res;
            res = binary_op(res, *(first + offset));
          }
          localRes[tid] = res;
        }
#pragma omp barrier

        ValueT tmp = res;
        for (DiffT stride = 1; stride < nths; stride *= 2) {
          if (tid >= stride && st < ed) tmp = binary_op(tmp, localRes[tid - stride]);
#pragma omp barrier
          if (tid >= stride && st < ed) localRes[tid] = tmp;
#pragma omp barrier
        }

        if (tid != 0 && st < ed) {
          tmp = localRes[tid - 1];
          for (auto offset = st; offset < ed; ++offset)
            *(d_first + offset) = binary_op(*(d_first + offset), tmp);
        }
      }
      if (shouldProfile())
        timer.tock(fmt::format("[Omp Exec | File {}, Ln {}, Col {}]", loc.file_name(), loc.line(),
                               loc.column()));
    }
    template <class InputIt, class OutputIt,
              class BinaryOperation
              = std::plus<typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>>
    void exclusive_scan(
        InputIt &&first, InputIt &&last, OutputIt &&d_first,
        typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type init
        = deduce_identity<BinaryOperation,
                          typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>(),
        BinaryOperation &&binary_op = {},
        const source_location &loc = source_location::current()) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      exclusive_scan_impl(
          typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{}, FWD(first),
          FWD(last), FWD(d_first), init, FWD(binary_op), loc);
    }
    /// reduce
    template <class InputIt, class OutputIt, class BinaryOperation>
    void reduce_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                     OutputIt &&d_first,
                     typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type init,
                     BinaryOperation &&binary_op, const source_location &loc) const {
      using IterT = remove_cvref_t<InputIt>;
      using DstIterT = remove_cvref_t<OutputIt>;
      using DiffT = typename std::iterator_traits<IterT>::difference_type;
      using ValueT = typename std::iterator_traits<DstIterT>::value_type;
      static_assert(
          std::is_convertible_v<DiffT, typename std::iterator_traits<DstIterT>::difference_type>,
          "diff type not compatible");
      static_assert(std::is_convertible_v<typename std::iterator_traits<IterT>::value_type, ValueT>,
                    "value type not compatible");
      CppTimer timer;
      if (shouldProfile()) timer.tick();
      const auto dist = last - first;
      auto allocator = get_temporary_memory_source(*this);
      Vector<ValueT> localRes{allocator, (size_t)0};
      DiffT nths{};
#pragma omp parallel if (_dop < dist) num_threads(_dop) shared(dist, nths, first, last, d_first)
      {
#pragma omp single
        {
          nths = omp_get_num_threads();
          localRes.resize(nths);
        }
#pragma omp barrier
        DiffT tid = omp_get_thread_num();
        DiffT nwork = (dist + nths - 1) / nths;
        DiffT st = nwork * tid;
        DiffT ed = st + nwork;
        if (ed > dist) ed = dist;

        ValueT res{init};
        if (st < ed) {
          for (auto offset = st; offset < ed; ++offset) res = binary_op(res, *(first + offset));
        }
        localRes[tid] = res;
#pragma omp barrier

        ValueT tmp = res;
        for (DiffT stride = 1; stride < nths; stride *= 2) {
          if (tid + stride < nths) tmp = binary_op(tmp, localRes[tid + stride]);
#pragma omp barrier
          if (tid + stride < nths) localRes[tid] = tmp;
#pragma omp barrier
        }

        if (tid == 0) *d_first = tmp;
      }
      if (shouldProfile())
        timer.tock(fmt::format("[Omp Reduce | File {}, Ln {}, Col {}]", loc.file_name(), loc.line(),
                               loc.column()));
    }
    template <class InputIt, class OutputIt,
              class BinaryOp
              = std::plus<typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>>
    void reduce(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type init
                = deduce_identity<
                    BinaryOp, typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>(),
                BinaryOp &&binary_op = {},
                const source_location &loc = source_location::current()) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      reduce_impl(typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{},
                  FWD(first), FWD(last), FWD(d_first), init, FWD(binary_op), loc);
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
        typename std::iterator_traits<std::remove_reference_t<KeyIter>>::difference_type dist,
        CompareOpT &&compOp, wrapv<Stable>, const source_location &loc) const {
      using KeyIterT = remove_cvref_t<KeyIter>;
      using ValueIterT = remove_cvref_t<ValueIter>;
      using DiffT = typename std::iterator_traits<KeyIterT>::difference_type;
      using KeyT = typename std::iterator_traits<KeyIterT>::value_type;
      using ValueT = typename std::iterator_traits<ValueIterT>::value_type;

      CppTimer timer;
      if (shouldProfile()) timer.tick();

      std::vector<KeyT> okeys_(dist);
      std::vector<ValueT> ovals_(dist);
      auto okeys = std::begin(okeys_);
      auto ovals = std::begin(ovals_);

      DiffT nths{}, nwork{};
      bool switched = false;
#pragma omp parallel if (_dop * 256 < dist) num_threads(_dop) \
    shared(switched, nths, nwork, keys, vals, okeys, ovals, compOp)
      {
#pragma omp single
        {
          nths = omp_get_num_threads();
          nwork = (dist + nths - 1) / nths;
        }
#pragma omp barrier
        DiffT tid = omp_get_thread_num();
        const DiffT l = nwork * tid;
        const DiffT r = (l + nwork) > dist ? dist : (l + nwork);

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

        for (DiffT halfStride = 1; halfStride < nths; halfStride *= 2) {
          DiffT stride = halfStride * 2;
#pragma omp barrier
          if (tid % stride == 0) {
            // auto bgCur = flipped ? okeys : keys;
            // auto bgCurVals = flipped ? ovals : vals;
            // auto bgNext = flipped ? keys : okeys;
            // auto bgNextVals = flipped ? vals : ovals;
            DiffT mid = std::min(nwork * (tid + halfStride), dist);
            DiffT rr = std::min(nwork * (tid + stride), dist);
            // std::inplace_merge(first + l, first + mid, first + r, compOp);
            DiffT left = l, right = mid, k = l;
            if (flipped) {
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
            } else {
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
            flipped = !flipped;
          }
        }
        if (tid == 0) switched = flipped;
#pragma omp barrier
        if (switched) {
          for (DiffT k = l; k < r; ++k) {
            keys[k] = okeys[k];
            vals[k] = ovals[k];
          }
        }
      }

      if (shouldProfile())
        timer.tock(fmt::format("[Omp merge_sort | File {}, Ln {}, Col {}]", loc.file_name(),
                               loc.line(), loc.column()));
    }
    template <typename KeyIter, typename ValueIter,
              typename CompareOpT
              = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
    void sort_pair(
        KeyIter &&keys, ValueIter &&vals,
        typename std::iterator_traits<std::remove_reference_t<KeyIter>>::difference_type count,
        CompareOpT &&compOp = {}, const source_location &loc = source_location::current()) const {
      merge_sort_pair_impl(FWD(keys), FWD(vals), count, FWD(compOp), false_c, loc);  // unstable
    }
    template <typename KeyIter, typename ValueIter,
              typename CompareOpT
              = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
    void merge_sort_pair(
        KeyIter &&keys, ValueIter &&vals,
        typename std::iterator_traits<std::remove_reference_t<KeyIter>>::difference_type count,
        CompareOpT &&compOp = {}, const source_location &loc = source_location::current()) const {
      merge_sort_pair_impl(FWD(keys), FWD(vals), count, FWD(compOp), true_c, loc);  // stable
    }
    template <class KeyIter, typename CompareOpT, bool Stable>
    void merge_sort_impl(KeyIter &&first, KeyIter &&last, CompareOpT &&compOp, wrapv<Stable>,
                         const source_location &loc) const {
      using IterT = remove_cvref_t<KeyIter>;
      using DiffT = typename std::iterator_traits<IterT>::difference_type;
      using KeyT = typename std::iterator_traits<IterT>::value_type;

      CppTimer timer;
      if (shouldProfile()) timer.tick();
      const auto dist = last - first;

      std::vector<KeyT> tmp(dist);
      auto ofirst = std::begin(tmp);

      DiffT nths{}, nwork{};
      bool switched = false;
#pragma omp parallel if (_dop * 256 < dist) num_threads(_dop) \
    shared(switched, nths, nwork, first, ofirst, compOp)
      {
#pragma omp single
        {
          nths = omp_get_num_threads();
          nwork = (dist + nths - 1) / nths;
        }
#pragma omp barrier
        DiffT tid = omp_get_thread_num();
        const DiffT l = nwork * tid;
        const DiffT r = (l + nwork) > dist ? dist : (l + nwork);

        bool flipped = false;

        if constexpr (Stable) {
          // std::stable_sort(first + l, first + r, compOp);
          // insertion sort for segments of granularity <= 16
          for (DiffT ll = l; ll < r;) {
            auto rr = std::min(ll + 16, r);
            // [ll, rr)
            for (DiffT i = ll + 1; i != rr; ++i) {
              for (DiffT k = i; k != ll; --k) {  // insert k
                auto j = k - 1;
                if (compOp(first[k], first[j]))
                  std::swap(first[k], first[j]);
                else
                  break;
              }
            }
            ll = rr;
          }
          //  bottom-up fashion
          for (DiffT halfStride = 16; halfStride < (r - l);) {
            DiffT stride = halfStride * 2;
            // auto bgCur = flipped ? ofirst : first;
            // auto bgNext = flipped ? first : ofirst;
            if (flipped) {
              for (DiffT ll = l; ll < r; ll += stride) {
                DiffT mid = std::min(ll + halfStride, r);
                DiffT rr = std::min(ll + stride, r);
                // [ll, mid) [mid, rr)
                DiffT left = ll, right = mid, k = ll;
                while (left < mid && right < rr) {
                  const auto &a = ofirst[left];
                  const auto &b = ofirst[right];
                  if (!compOp(b, a)) {
                    first[k++] = a;
                    left++;
                  } else {
                    first[k++] = b;
                    right++;
                  }
                }
                while (left < mid) first[k++] = ofirst[left++];
                while (right < rr) first[k++] = ofirst[right++];
              }
            } else {
              for (DiffT ll = l; ll < r; ll += stride) {
                DiffT mid = std::min(ll + halfStride, r);
                DiffT rr = std::min(ll + stride, r);
                // [ll, mid) [mid, rr)
                DiffT left = ll, right = mid, k = ll;
                while (left < mid && right < rr) {
                  const auto &a = first[left];
                  const auto &b = first[right];
                  if (!compOp(b, a)) {
                    ofirst[k++] = a;
                    left++;
                  } else {
                    ofirst[k++] = b;
                    right++;
                  }
                }
                while (left < mid) ofirst[k++] = first[left++];
                while (right < rr) ofirst[k++] = first[right++];
              }
            }
            flipped = !flipped;
            halfStride = stride;
          }
        } else {
          std::sort(first + l, first + r, compOp);
        }

        for (DiffT halfStride = 1; halfStride < nths; halfStride *= 2) {
          DiffT stride = halfStride * 2;
#pragma omp barrier
          if (tid % stride == 0) {
            // auto bgCur = flipped ? ofirst : first;
            // auto bgNext = flipped ? first : ofirst;
            DiffT mid = std::min(nwork * (tid + halfStride), dist);
            DiffT rr = std::min(nwork * (tid + stride), dist);
            // std::inplace_merge(first + l, first + mid, first + r, compOp);
            DiffT left = l, right = mid, k = l;
            if (flipped) {
              while (left < mid && right < rr) {
                const auto &a = ofirst[left];
                const auto &b = ofirst[right];
                if (!compOp(b, a)) {
                  first[k++] = a;
                  left++;
                } else {
                  first[k++] = b;
                  right++;
                }
              }
              while (left < mid) first[k++] = ofirst[left++];
              while (right < rr) first[k++] = ofirst[right++];
            } else {
              while (left < mid && right < rr) {
                const auto &a = first[left];
                const auto &b = first[right];
                if (!compOp(b, a)) {
                  ofirst[k++] = a;
                  left++;
                } else {
                  ofirst[k++] = b;
                  right++;
                }
              }
              while (left < mid) ofirst[k++] = first[left++];
              while (right < rr) ofirst[k++] = first[right++];
            }
            flipped = !flipped;
          }
        }
        if (tid == 0) switched = flipped;
#pragma omp barrier
        if (switched)
          for (DiffT k = l; k < r; ++k) first[k] = ofirst[k];
      }

      if (shouldProfile())
        timer.tock(fmt::format("[Omp merge_sort | File {}, Ln {}, Col {}]", loc.file_name(),
                               loc.line(), loc.column()));
    }

    template <class KeyIter,
              typename CompareOpT
              = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
    void sort(KeyIter &&first, KeyIter &&last, CompareOpT &&compOp = {},
              const source_location &loc = source_location::current()) const {
      merge_sort_impl(FWD(first), FWD(last), FWD(compOp), false_c, loc);  // unstable
    }
    template <class KeyIter,
              typename CompareOpT
              = std::less<typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type>>
    void merge_sort(KeyIter &&first, KeyIter &&last, CompareOpT &&compOp = {},
                    const source_location &loc = source_location::current()) const {
      merge_sort_impl(FWD(first), FWD(last), FWD(compOp), true_c, loc);  // stable
    }

    /// radix sort
    template <class InputIt, class OutputIt>
    void radix_sort_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                         OutputIt &&d_first, int sbit, int ebit, const source_location &loc) const {
      using IterT = remove_cvref_t<InputIt>;
      using DstIterT = remove_cvref_t<OutputIt>;
      using DiffT = typename std::iterator_traits<IterT>::difference_type;
      using InputValueT = typename std::iterator_traits<IterT>::value_type;
      using ValueT = typename std::iterator_traits<DstIterT>::value_type;
      static_assert(
          std::is_convertible_v<DiffT, typename std::iterator_traits<DstIterT>::difference_type>,
          "diff type not compatible");
      static_assert(std::is_convertible_v<InputValueT, ValueT>, "value type not compatible");
      static_assert(std::is_integral_v<ValueT>, "value type not integral");

      CppTimer timer;
      if (shouldProfile()) timer.tick();
      const auto dist = last - first;
      DiffT nths{}, nwork{};
      // const int binBits = bit_length(_dop);
      bool skip = false;
      constexpr int binBits = 8;  // by byte
      int binCount = 1 << binBits;
      int binMask = binCount - 1;
      std::vector<std::vector<DiffT>> binSizes{};
      std::vector<DiffT> binGlobalSizes(binCount);
      std::vector<DiffT> binOffsets(binCount);

      /// double buffer strategy
      std::vector<InputValueT> buffers[2];
      buffers[0].resize(dist);
      buffers[1].resize(dist);
      InputValueT *cur{buffers[0].data()}, *next{buffers[1].data()};

      /// move to local buffer first (bit hack for signed type)
#pragma omp parallel for if (_dop < dist) num_threads(_dop)
      for (DiffT i = 0; i < dist; ++i) {
        if constexpr (std::is_signed_v<InputValueT>)
          cur[i] = *(first + i) ^ ((InputValueT)1 << (sizeof(InputValueT) * 8 - 1));
        else
          cur[i] = *(first + i);
      }

      /// LSB style (outmost loop)
      for (int st = sbit; st < ebit; st += binBits) {
        if (st + binBits > ebit) {
          binMask >>= (st + binBits - ebit);
          binCount >>= (st + binBits - ebit);
        }

        /// init
#pragma omp parallel if (_dop < dist) num_threads(_dop) \
    shared(skip, nths, nwork, binSizes, binGlobalSizes, binOffsets, cur, next)
        {
#pragma omp single
          {
            nths = omp_get_num_threads();
            nwork = (dist + nths - 1) / nths;
            binSizes.resize(nths);
            skip = false;
          }
#pragma omp barrier
          /// work block partition
          DiffT tid = omp_get_thread_num();
          DiffT l = nwork * tid;
          DiffT r = l + nwork;
          if (r > dist) r = dist;
          /// init
          binSizes[tid].resize(binCount);

          /// local count
          for (DiffT i = 0; i < binCount; ++i) binSizes[tid][i] = 0;
          if (l < dist)
            for (auto i = l; i < r; ++i) binSizes[tid][(cur[i] >> st) & binMask]++;

#pragma omp barrier

#pragma omp single
          {
            /// reduce binSizes from all threads
            for (int i = 0; i < binCount; ++i) {
              binGlobalSizes[i] = 0;
              for (int j = 0; j < nths; ++j) binGlobalSizes[i] += binSizes[j][i];
              if (binGlobalSizes[i] == dist) {
                skip = true;
                break;
              }
            }

            if (!skip) {
              /// exclusive scan
              binOffsets[0] = 0;
              for (int i = 1; i < binCount; ++i)
                binOffsets[i] = binOffsets[i - 1] + binGlobalSizes[i - 1];

              /// update local offsets
              for (int i = 0; i < binCount; i++) {
                binSizes[0][i] += binOffsets[i];
                for (int j = 1; j < nths; j++) binSizes[j][i] += binSizes[j - 1][i];
              }
            }
          }

          if (!skip) {
/// distribute
#pragma omp barrier
            if (l < dist)
              for (auto i = r - 1; i >= l; --i)
                next[--binSizes[tid][(cur[i] >> st) & binMask]] = cur[i];
#pragma omp barrier
#pragma omp single
            { std::swap(cur, next); }
          }
#pragma omp barrier
        }
      }

#pragma omp parallel for if (_dop < dist) num_threads(_dop)
      for (DiffT i = 0; i < dist; ++i) {
        if constexpr (std::is_signed_v<InputValueT>)
          *(d_first + i) = cur[i] ^ ((InputValueT)1 << (sizeof(InputValueT) * 8 - 1));
        else
          *(d_first + i) = cur[i];
      }
      if (shouldProfile())
        timer.tock(fmt::format("[Omp Exec | File {}, Ln {}, Col {}]", loc.file_name(), loc.line(),
                               loc.column()));
    }
    template <class InputIt, class OutputIt> void radix_sort(
        InputIt &&first, InputIt &&last, OutputIt &&d_first, int sbit = 0,
        int ebit = sizeof(typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type) * 8,
        const source_location &loc = source_location::current()) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      static_assert(is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::pointer,
                              typename std::iterator_traits<remove_cvref_t<OutputIt>>::pointer>,
                    "Input iterator pointer different from output iterator\'s");
      radix_sort_impl(typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{},
                      FWD(first), FWD(last), FWD(d_first), sbit, ebit, loc);
    }

    template <class KeyIter, class ValueIter, typename Tn>
    void radix_sort_pair_impl(std::random_access_iterator_tag, KeyIter &&keysIn, ValueIter &&valsIn,
                              KeyIter &&keysOut, ValueIter &&valsOut, Tn count, int sbit, int ebit,
                              const source_location &loc) const {
      using KeyT = typename std::iterator_traits<KeyIter>::value_type;
      using ValueT = typename std::iterator_traits<ValueIter>::value_type;
      using DiffT = typename std::iterator_traits<KeyIter>::difference_type;
      static_assert(std::is_integral_v<KeyT>, "key type not integral");

      CppTimer timer;
      if (shouldProfile()) timer.tick();
      const auto dist = count;
      DiffT nths{}, nwork{};
      // const int binBits = bit_length(_dop);
      bool skip = false;
      constexpr int binBits = 8;  // by byte
      int binCount = 1 << binBits;
      int binMask = binCount - 1;
      std::vector<std::vector<DiffT>> binSizes{};
      std::vector<DiffT> binGlobalSizes(binCount);
      std::vector<DiffT> binOffsets(binCount);

      /// double buffer strategy
      std::vector<KeyT> keyBuffers[2];
      std::vector<ValueT> valBuffers[2];
      keyBuffers[0].resize(count);
      keyBuffers[1].resize(count);
      valBuffers[0].resize(count);
      valBuffers[1].resize(count);
      KeyT *cur{keyBuffers[0].data()}, *next{keyBuffers[1].data()};
      ValueT *curVals{valBuffers[0].data()}, *nextVals{valBuffers[1].data()};

      /// move to local buffer first (bit hack for signed type)
#pragma omp parallel for if (_dop < dist) num_threads(_dop)
      for (DiffT i = 0; i < dist; ++i) {
        if constexpr (std::is_signed_v<KeyT>)
          cur[i] = *(keysIn + i) ^ ((KeyT)1 << (sizeof(KeyT) * 8 - 1));
        else
          cur[i] = *(keysIn + i);
        curVals[i] = *(valsIn + i);
      }

      /// LSB style (outmost loop)
      for (int st = sbit; st < ebit; st += binBits) {
        if (st + binBits > ebit) {
          binMask >>= (st + binBits - ebit);
          binCount >>= (st + binBits - ebit);
        }

        /// init
#pragma omp parallel if (_dop < dist) num_threads(_dop) \
    shared(skip, nths, nwork, binSizes, binGlobalSizes, binOffsets, cur, next, curVals, nextVals)
        {
#pragma omp single
          {
            nths = omp_get_num_threads();
            nwork = (dist + nths - 1) / nths;
            binSizes.resize(nths);
            skip = false;
          }
#pragma omp barrier
          /// work block partition
          DiffT tid = omp_get_thread_num();
          DiffT l = nwork * tid;
          DiffT r = l + nwork;
          if (r > dist) r = dist;
          /// init
          binSizes[tid].resize(binCount);

          /// local count
          for (DiffT i = 0; i < binCount; ++i) binSizes[tid][i] = 0;
          if (l < dist)
            for (auto i = l; i < r; ++i) binSizes[tid][(cur[i] >> st) & binMask]++;

#pragma omp barrier

#pragma omp single
          {
            /// reduce binSizes from all threads
            for (int i = 0; i < binCount; ++i) {
              binGlobalSizes[i] = 0;
              for (int j = 0; j < nths; ++j) binGlobalSizes[i] += binSizes[j][i];
              if (binGlobalSizes[i] == dist) {
                skip = true;
                break;
              }
            }

            if (!skip) {
              /// exclusive scan
              binOffsets[0] = 0;
              for (int i = 1; i < binCount; ++i)
                binOffsets[i] = binOffsets[i - 1] + binGlobalSizes[i - 1];

              /// update local offsets
              for (int i = 0; i < binCount; i++) {
                binSizes[0][i] += binOffsets[i];
                for (int j = 1; j < nths; j++) binSizes[j][i] += binSizes[j - 1][i];
              }
            }
          }

          if (!skip) {
/// distribute
#pragma omp barrier
            if (l < dist)
              for (auto i = r - 1; i >= l; --i) {
                const auto loc = --binSizes[tid][(cur[i] >> st) & binMask];
                next[loc] = cur[i];
                nextVals[loc] = curVals[i];
              }
#pragma omp barrier
#pragma omp single
            {
              std::swap(cur, next);
              std::swap(curVals, nextVals);
            }
          }
#pragma omp barrier
        }
      }

#pragma omp parallel for if (_dop < dist) num_threads(_dop)
      for (DiffT i = 0; i < dist; ++i) {
        if constexpr (std::is_signed_v<KeyT>)
          *(keysOut + i) = cur[i] ^ ((KeyT)1 << (sizeof(KeyT) * 8 - 1));
        else
          *(keysOut + i) = cur[i];
        *(valsOut + i) = curVals[i];
      }
      if (shouldProfile())
        timer.tock(fmt::format("[Omp Exec | File {}, Ln {}, Col {}]", loc.file_name(), loc.line(),
                               loc.column()));
    }
    template <class KeyIter, class ValueIter,
              typename Tn
              = typename std::iterator_traits<std::remove_reference_t<KeyIter>>::difference_type>
    void radix_sort_pair(
        KeyIter &&keysIn, ValueIter &&valsIn, KeyIter &&keysOut, ValueIter &&valsOut, Tn count = 0,
        int sbit = 0,
        int ebit
        = sizeof(typename std::iterator_traits<std::remove_reference_t<KeyIter>>::value_type) * 8,
        const source_location &loc = source_location::current()) const {
      static_assert(
          is_same_v<
              typename std::iterator_traits<std::remove_reference_t<KeyIter>>::iterator_category,
              typename std::iterator_traits<std::remove_reference_t<ValueIter>>::iterator_category>,
          "Key Iterator and Val Iterator should be from the same category");
      radix_sort_pair_impl(
          typename std::iterator_traits<std::remove_reference_t<KeyIter>>::iterator_category{},
          FWD(keysIn), FWD(valsIn), FWD(keysOut), FWD(valsOut), count, sbit, ebit, loc);
    }

    OmpExecutionPolicy &threads(int numThreads) noexcept {
      _dop = numThreads;
      return *this;
    }

  protected:
    friend struct ExecutionPolicyInterface<OmpExecutionPolicy>;

    int _dop{1};
  };

  constexpr bool is_backend_available(OmpExecutionPolicy) noexcept { return true; }
  constexpr bool is_backend_available(omp_exec_tag) noexcept { return true; }

  uint get_hardware_concurrency() noexcept;
  inline OmpExecutionPolicy omp_exec() noexcept {
    return OmpExecutionPolicy{}.threads(get_hardware_concurrency() - 1);
  }
  inline OmpExecutionPolicy par_exec(omp_exec_tag) noexcept {
    return OmpExecutionPolicy{}.threads(get_hardware_concurrency() - 1);
  }

  inline ZPC_API ZSPmrAllocator<> get_temporary_memory_source(OmpExecutionPolicy &pol) {
    return get_memory_source(memsrc_e::host, (ProcID)-1);
  }

}  // namespace zs