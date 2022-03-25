#pragma once

#include <cooperative_groups.h>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include "zensim/cuda/Cuda.h"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/types/Tuple.h"
// #include <device_types.h>
#include <iterator>
#include <nvfunctional>
#include <type_traits>

#include "zensim/types/Function.h"
#include "zensim/types/SourceLocation.hpp"

/// extracted from compiler error message...
template <class Tag, class... CapturedVarTypePack> struct __nv_dl_wrapper_t;
template <class U, U func, unsigned int> struct __nv_dl_tag;

namespace zs {

  namespace detail {

    template <typename F, typename ArgSeq, typename = void> struct deduce_fts {
      static constexpr bool fts_available
          = is_valid([](auto t) -> decltype(zs::function_traits<typename decltype(t)::type>{},
                                            void()) {})(zs::wrapt<F>{});
      template <typename IterOrIndex> static constexpr bool deref_available
          = is_valid([](auto t) -> decltype((void)(*std::declval<typename decltype(t)::type>())) {
            })(zs::wrapt<IterOrIndex>{});
      template <typename Iter> using iter_arg_t = decltype(*std::declval<Iter>());

      template <typename Seq> struct impl;
      template <typename... Args> struct impl<std::tuple<Args...>> {
        static constexpr bool all_deref_available = (deref_available<Args> && ...);
        static constexpr auto deduce_args_t() noexcept {
          if constexpr (fts_available)
            return typename function_traits<F>::arguments_t{};
          else if constexpr (all_deref_available) {
            if constexpr (std::is_invocable_v<F, int, iter_arg_t<Args>...>)
              return std::tuple<int, iter_arg_t<Args>...>{};
            else if constexpr (std::is_invocable_v<F, void *, iter_arg_t<Args>...>)
              return std::tuple<void *, iter_arg_t<Args>...>{};
            else
              return std::tuple<iter_arg_t<Args>...>{};
          } else {
            if constexpr (std::is_invocable_v<F, int, Args...>)
              return std::tuple<int, Args...>{};
            else if constexpr (std::is_invocable_v<F, void *, Args...>)
              return std::tuple<void *, Args...>{};
            else
              return std::tuple<Args...>{};
          }
        }
        static constexpr auto deduce_return_t() noexcept {
          if constexpr (fts_available)
            return typename function_traits<F>::return_t{};
          else if constexpr (all_deref_available) {
            if constexpr (std::is_invocable_v<F, int, iter_arg_t<Args>...>)
              return std::invoke_result_t<F, int, iter_arg_t<Args>...>{};
            else if constexpr (std::is_invocable_v<F, void *, iter_arg_t<Args>...>)
              return std::invoke_result_t<F, void *, iter_arg_t<Args>...>{};
            else
              return std::invoke_result_t<F, iter_arg_t<Args>...>{};
          } else {
            if constexpr (std::is_invocable_v<F, int, Args...>)
              return std::invoke_result_t<F, int, Args...>{};
            else if constexpr (std::is_invocable_v<F, void *, Args...>)
              return std::invoke_result_t<F, void *, Args...>{};
            else
              return std::invoke_result_t<F, Args...>{};
          }
        }
        static constexpr std::size_t deduce_arity() noexcept {
          if constexpr (fts_available)
            return function_traits<F>::arity;
          else
            return std::tuple_size_v<decltype(deduce_args_t())>;
        }
      };

      using arguments_t = decltype(impl<ArgSeq>::deduce_args_t());
      using return_t = decltype(impl<ArgSeq>::deduce_return_t());
      static constexpr std::size_t arity = impl<ArgSeq>::deduce_arity();

      static_assert(is_same_v<return_t, void>,
                    "callable for execution policy should only return void");
    };

    template <bool withIndex, typename Tn, typename F, typename ZipIter, std::size_t... Is>
    __forceinline__ __device__ void range_foreach(std::bool_constant<withIndex>, Tn i, F &&f,
                                                  ZipIter &&iter, index_seq<Is...>) {
      (std::get<Is>(iter.iters).advance(i), ...);
      if constexpr (withIndex)
        f(i, *std::get<Is>(iter.iters)...);
      else {
        f(*std::get<Is>(iter.iters)...);
      }
    }
    template <bool withIndex, typename Tn, typename F, typename ZipIter, std::size_t... Is>
    __forceinline__ __device__ void range_foreach(std::bool_constant<withIndex>, char *shmem, Tn i,
                                                  F &&f, ZipIter &&iter, index_seq<Is...>) {
      (std::get<Is>(iter.iters).advance(i), ...);
      using func_traits = detail::deduce_fts<remove_cvref_t<F>, RM_CVREF_T(iter.iters)>;
      using shmem_ptr_t = std::tuple_element_t<0, typename func_traits::arguments_t>;
      if constexpr (withIndex)
        f(reinterpret_cast<shmem_ptr_t>(shmem), i, *std::get<Is>(iter.iters)...);
      else
        f(reinterpret_cast<shmem_ptr_t>(shmem), *std::get<Is>(iter.iters)...);
    }
  }  // namespace detail

  // =========================  signature  ==============================
  // loopbody signature: (scratchpadMemory, blockid, warpid, threadid)
  template <typename Tn, typename F> __global__ void thread_launch(Tn n, F f) {
    extern __shared__ char shmem[];
    Tn id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
      using func_traits = detail::deduce_fts<F, std::tuple<RM_CVREF_T(id)>>;
      if constexpr (func_traits::arity == 1)
        f(id);
      else if constexpr (func_traits::arity == 2
                         && std::is_pointer_v<
                             std::tuple_element_t<0, typename func_traits::arguments_t>>)
        f(reinterpret_cast<std::tuple_element_t<0, typename func_traits::arguments_t>>(shmem), id);
    }
  }
  template <typename F> __global__ void block_thread_launch(F f) {
    extern __shared__ char shmem[];
    using func_traits
        = detail::deduce_fts<F, std::tuple<RM_CVREF_T(blockIdx.x), RM_CVREF_T(threadIdx.x)>>;
    if constexpr (func_traits::arity == 2
                  && !std::is_pointer_v<std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(blockIdx.x, threadIdx.x);
    else if constexpr (func_traits::arity == 3
                       && std::is_pointer_v<
                           std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(reinterpret_cast<std::tuple_element_t<0, typename func_traits::arguments_t>>(shmem),
        blockIdx.x, threadIdx.x);
  }

  template <typename Tn, typename F, typename ZipIter> __global__ std::enable_if_t<
      std::is_convertible_v<
          typename std::iterator_traits<ZipIter>::iterator_category,
          std::
              random_access_iterator_tag> && is_std_tuple<typename std::iterator_traits<ZipIter>::reference>::value>
  range_launch(Tn n, F f, ZipIter iter) {
    extern __shared__ char shmem[];
    Tn id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
      using func_traits = detail::deduce_fts<F, RM_CVREF_T(iter.iters)>;
      constexpr auto numArgs = std::tuple_size_v<typename std::iterator_traits<ZipIter>::reference>;
      constexpr auto indices = std::make_index_sequence<numArgs>{};

      if constexpr (func_traits::arity == numArgs) {
        detail::range_foreach(std::false_type{}, id, f, iter, indices);
      } else if constexpr (func_traits::arity == numArgs + 1) {
        if constexpr (std::is_integral_v<
                          std::tuple_element_t<0, typename func_traits::arguments_t>>)
          detail::range_foreach(std::true_type{}, id, f, iter, indices);
        else if constexpr (std::is_pointer_v<
                               std::tuple_element_t<0, typename func_traits::arguments_t>>)
          detail::range_foreach(
              std::false_type{},
              reinterpret_cast<std::tuple_element_t<0, typename func_traits::arguments_t>>(shmem),
              id, f, iter, indices);
      } else if constexpr (func_traits::arity == numArgs + 2
                           && std::is_pointer_v<
                               std::tuple_element_t<0, typename func_traits::arguments_t>>)
        detail::range_foreach(
            std::true_type{},
            reinterpret_cast<std::tuple_element_t<0, typename func_traits::arguments_t>>(shmem), id,
            f, iter, indices);
    }
  }
  namespace cg = cooperative_groups;
  template <typename Tn, typename F> __global__ void block_tile_lane_launch(Tn tileSize, F f) {
    extern __shared__ char shmem[];
    cg::thread_block block = cg::this_thread_block();
    cg::thread_group tile = cg::tiled_partition(block, tileSize);
    using func_traits = detail::deduce_fts<
        F, std::tuple<RM_CVREF_T(blockIdx.x), RM_CVREF_T(block.thread_rank() / tileSize),
                      RM_CVREF_T(tile.thread_rank())>>;
    if constexpr (func_traits::arity == 3
                  && !std::is_pointer_v<std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(blockIdx.x, block.thread_rank() / tileSize, tile.thread_rank());
    else if constexpr (func_traits::arity == 4
                       && std::is_pointer_v<
                           std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(reinterpret_cast<std::tuple_element_t<0, typename func_traits::arguments_t>>(shmem),
        blockIdx.x, block.thread_rank() / tileSize, tile.thread_rank());
  }

  namespace detail {
    template <typename, typename> struct function_traits_impl;

    template <auto F, unsigned int I, typename R, typename... Args>
    struct function_traits_impl<__nv_dl_tag<R (*)(Args...), F, I>> {
      static constexpr std::size_t arity = sizeof...(Args);
      using return_t = R;
      using arguments_t = std::tuple<Args...>;
    };
    template <class Tag, class... CapturedVarTypePack>
    struct function_traits_impl<__nv_dl_wrapper_t<Tag, CapturedVarTypePack...>>
        : function_traits_impl<Tag> {};
  }  // namespace detail

  struct CudaExecutionPolicy : ExecutionPolicyInterface<CudaExecutionPolicy> {
    using exec_tag = cuda_exec_tag;
    CudaExecutionPolicy &listen(ProcID incProc, StreamID incStream) {
      this->_wait = true;
      incomingProc = incProc;
      incomingStreamid = incStream;
      return *this;
    }
    CudaExecutionPolicy &stream(StreamID streamid_) {
      streamid = streamid_;
      return *this;
    }
    CudaExecutionPolicy &device(ProcID pid) {
      procid = pid;
      return *this;
    }
    CudaExecutionPolicy &shmem(std::size_t bytes) {
      shmemBytes = bytes;
      return *this;
    }
    CudaExecutionPolicy &block(std::size_t tpb) {
      blockSize = tpb;
      return *this;
    }
#if 0
    template <typename FTraits> static constexpr unsigned computeArity() noexcept {
      unsigned res{0};
      if constexpr (FTraits::arity != 0)
        res = FTraits::arity
              - (std::is_pointer_v<std::tuple_element_t<0, typename FTraits::arguments_t>> ? 1 : 0);
      return res;
    }
#endif

    template <typename Ts, typename Is, typename F>
    void operator()(Collapse<Ts, Is> dims, F &&f,
                    const source_location &loc = source_location::current()) const {
      using namespace index_literals;
      constexpr auto dim = Collapse<Ts, Is>::dim;
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(context.streamSpare(streamid), loc);
      // need to work on __device__ func as well
      if constexpr (dim == 1) {
        LaunchConfig lc{};
        if (blockSize == 0)
          lc = LaunchConfig{std::true_type{}, dims.get(0_th), shmemBytes};
        else
          lc = LaunchConfig{(dims.get(0_th) + blockSize - 1) / blockSize, blockSize, shmemBytes};
        cuda_safe_launch(loc, context, streamid, std::move(lc), thread_launch, dims.get(0_th), f);
      }
      // else if constexpr (arity == 2)
      else if constexpr (dim == 2) {
        cuda_safe_launch(loc, context, streamid, {dims.get(0_th), dims.get(1_th), shmemBytes},
                         block_thread_launch, f);
      }
      // else if constexpr (arity == 3)
      else if constexpr (dim == 3) {
        cuda_safe_launch(loc, context, streamid,
                         {dims.get(0_th), dims.get(1_th) * dims.get(2_th), shmemBytes},
                         block_tile_lane_launch, dims.get(2_th), f);
      }
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    template <typename Range, typename F>
    auto operator()(Range &&range, F &&f,
                    const source_location &loc = source_location::current()) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));

      // need to work on __device__ func as well
      auto iter = std::begin(range);
      using IterT = remove_cvref_t<decltype(iter)>;
      using DiffT = typename std::iterator_traits<IterT>::difference_type;
      const DiffT dist = std::end(range) - iter;
      using RefT = typename std::iterator_traits<IterT>::reference;

      LaunchConfig lc{};
      if (blockSize == 0)
        lc = LaunchConfig{std::true_type{}, dist, shmemBytes};
      else
        lc = LaunchConfig{(dist + blockSize - 1) / blockSize, blockSize, shmemBytes};

      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(context.streamSpare(streamid), loc);

      if constexpr (is_std_tuple<RefT>::value) {
        cuda_safe_launch(loc, context, streamid, std::move(lc), range_launch, dist, f, iter);
      } else if constexpr (is_zip_iterator_v<IterT>) {
        cuda_safe_launch(loc, context, streamid, std::move(lc), range_launch, dist, f,
                         std::begin(FWD(range)));
      } else {  // wrap the non-zip range in a zip range
        cuda_safe_launch(loc, context, streamid, std::move(lc), range_launch, dist, f,
                         std::begin(zip(FWD(range))));
      }

      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }

    /// for_each
    template <class ForwardIt, class UnaryFunction>
    void for_each_impl(std::random_access_iterator_tag, ForwardIt &&first, ForwardIt &&last,
                       UnaryFunction &&f, const source_location &loc) const {
      using IterT = remove_cvref_t<ForwardIt>;
      const auto dist = last - first;
      (*this)(
          {dist},
          [first, f](typename std::iterator_traits<IterT>::difference_type tid) {
            f(*(first + tid));
          },
          loc);
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
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      std::size_t temp_bytes = 0;
      auto stream = (cudaStream_t)context.streamSpare(streamid);
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(stream, loc);
      cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, first.operator->(), d_first.operator->(),
                                     binary_op, dist, stream);

      void *d_tmp = context.streamMemAlloc(temp_bytes, stream);
      cub::DeviceScan::InclusiveScan(d_tmp, temp_bytes, first, d_first, binary_op, dist, stream);
      context.streamMemFree(d_tmp, stream);
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
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
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      auto stream = (cudaStream_t)context.streamSpare(streamid);
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(stream, loc);
      std::size_t temp_bytes = 0;
      cub::DeviceScan::ExclusiveScan(nullptr, temp_bytes, first, d_first, binary_op, init, dist,
                                     stream);
      void *d_tmp = context.streamMemAlloc(temp_bytes, stream);
      cub::DeviceScan::ExclusiveScan(d_tmp, temp_bytes, first, d_first, binary_op, init, dist,
                                     stream);
      context.streamMemFree(d_tmp, stream);
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    template <class InputIt, class OutputIt,
              class T = remove_cvref_t<decltype(*std::declval<InputIt>())>,
              class BinaryOperation = std::plus<T>>
    void exclusive_scan(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                        T init = monoid_op<BinaryOperation>::e, BinaryOperation &&binary_op = {},
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
    template <class InputIt, class OutputIt, class T, class BinaryOperation>
    void reduce_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                     OutputIt &&d_first, T init, BinaryOperation &&binary_op,
                     const source_location &loc) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      std::size_t temp_bytes = 0;
      auto stream = (cudaStream_t)context.streamSpare(streamid);
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(stream, loc);
      cub::DeviceReduce::Reduce(nullptr, temp_bytes, first, d_first, dist, binary_op, init, stream);
      void *d_tmp = context.streamMemAlloc(temp_bytes, stream);
      cub::DeviceReduce::Reduce(d_tmp, temp_bytes, first, d_first, dist, binary_op, init, stream);
      context.streamMemFree(d_tmp, stream);
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    template <class InputIt, class OutputIt,
              class T = remove_cvref_t<decltype(*std::declval<InputIt>())>,
              class BinaryOp = std::plus<T>>
    void reduce(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                T init = monoid_op<BinaryOp>::e, BinaryOp &&binary_op = {},
                const source_location &loc = source_location::current()) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      reduce_impl(typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{},
                  FWD(first), FWD(last), FWD(d_first), init, FWD(binary_op), loc);
    }
    /// histogram sort
    /// radix sort pair
    template <class KeyIter, class ValueIter,
              typename Tn = typename std::iterator_traits<remove_cvref_t<KeyIter>>::difference_type>
    std::enable_if_t<std::is_convertible_v<
        typename std::iterator_traits<remove_cvref_t<KeyIter>>::iterator_category,
        std::random_access_iterator_tag>>
    radix_sort_pair(KeyIter &&keysIn, ValueIter &&valsIn, KeyIter &&keysOut, ValueIter &&valsOut,
                    Tn count = 0, int sbit = 0,
                    int ebit
                    = sizeof(typename std::iterator_traits<remove_cvref_t<KeyIter>>::value_type)
                      * 8,
                    const source_location &loc = source_location::current()) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      if (count) {
        std::size_t temp_bytes = 0;
        auto stream = (cudaStream_t)context.streamSpare(streamid);
        Cuda::CudaContext::StreamExecutionTimer *timer{};
        if (this->shouldProfile()) timer = context.tick(stream, loc);
        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keysIn.operator->(),
                                        keysOut.operator->(), valsIn.operator->(),
                                        valsOut.operator->(), count, sbit, ebit, stream);
        void *d_tmp = context.streamMemAlloc(temp_bytes, stream);
        cub::DeviceRadixSort::SortPairs(d_tmp, temp_bytes, keysIn.operator->(),
                                        keysOut.operator->(), valsIn.operator->(),
                                        valsOut.operator->(), count, sbit, ebit, stream);
        context.streamMemFree(d_tmp, stream);
        if (this->shouldProfile()) context.tock(timer, loc);
      }
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
    }
    /// radix sort
    template <class InputIt, class OutputIt>
    void radix_sort_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                         OutputIt &&d_first, int sbit, int ebit, const source_location &loc) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      const auto dist = last - first;
      std::size_t temp_bytes = 0;
      auto stream = (cudaStream_t)context.streamSpare(streamid);
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(stream, loc);
      cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, first.operator->(), d_first.operator->(),
                                     dist, sbit, ebit, stream);
      void *d_tmp = context.streamMemAlloc(temp_bytes, stream);
      cub::DeviceRadixSort::SortKeys(d_tmp, temp_bytes, first.operator->(), d_first.operator->(),
                                     dist, sbit, ebit, stream);
      context.streamMemFree(d_tmp, stream);
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.recordEventSpare(streamid);
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

    constexpr ProcID getProcid() const noexcept { return procid; }
    constexpr StreamID getStreamid() const noexcept { return streamid; }
    constexpr ProcID getIncomingProcid() const noexcept { return incomingProc; }
    constexpr StreamID getIncomingStreamid() const noexcept { return incomingStreamid; }

  protected:
    // bool do_launch(const ParallelTask &) const noexcept;
    friend struct ExecutionPolicyInterface<CudaExecutionPolicy>;
    // template <auto flagbit> friend struct CudaLibHandle<flagbit>;

    // std::size_t blockGranularity{128};
    StreamID incomingStreamid{0};
    StreamID streamid{0};
    std::size_t shmemBytes{0};  ///< amount of shared memory passed
    int blockSize{0};           ///< 0 to enable auto configure
    ProcID incomingProc{0};
    ProcID procid{0};  ///< 0-th gpu
  };

  constexpr bool is_backend_available(CudaExecutionPolicy) noexcept { return true; }
  constexpr bool is_backend_available(cuda_exec_tag) noexcept { return true; }

  constexpr CudaExecutionPolicy cuda_exec() noexcept { return CudaExecutionPolicy{}; }
  constexpr CudaExecutionPolicy par_exec(cuda_exec_tag) noexcept { return CudaExecutionPolicy{}; }

}  // namespace zs
