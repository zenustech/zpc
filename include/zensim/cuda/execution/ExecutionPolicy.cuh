#pragma once

#include <cooperative_groups.h>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <zensim/execution/ExecutionPolicy.hpp>

#include "zensim/cuda/Cuda.h"
#include "zensim/types/Tuple.h"
// #include <device_types.h>
#include <iterator>
#include <nvfunctional>

#include "zensim/resource/Resource.h"
#include "zensim/types/Function.h"

/// extracted from compiler error message...
template <class Tag, class... CapturedVarTypePack> struct __nv_dl_wrapper_t;
template <class U, U func, unsigned int> struct __nv_dl_tag;

namespace zs {

  // =========================  signature  ==============================
  // loopbody signature: (blockid, warpid, threadid, scratchpadMemory)
  template <typename Tn, typename F> __global__ void thread_launch(Tn n, F f) {
    extern __shared__ char shmem[];
    Tn id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
      using func_traits = function_traits<F>;
      if constexpr (func_traits::arity == 1)
        f(id);
      else if constexpr (func_traits::arity == 2
                         && std::is_pointer_v<
                             std::tuple_element_t<0, typename func_traits::arguments_t>>)
        f(shmem, id);
    }
  }
  template <typename F> __global__ void block_thread_launch(F f) {
    extern __shared__ char shmem[];
    using func_traits = function_traits<F>;
    if constexpr (func_traits::arity == 2
                  && !std::is_pointer_v<std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(blockIdx.x, threadIdx.x);
    else if constexpr (func_traits::arity == 3
                       && std::is_pointer_v<
                           std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(shmem, blockIdx.x, threadIdx.x);
  }
  namespace cg = cooperative_groups;
  template <typename Tn, typename F> __global__ void block_tile_lane_launch(Tn tileSize, F f) {
    extern __shared__ char shmem[];
    cg::thread_block block = cg::this_thread_block();
    cg::thread_group tile = cg::tiled_partition(block, tileSize);
    using func_traits = function_traits<F>;
    if constexpr (func_traits::arity == 3
                  && !std::is_pointer_v<std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(blockIdx.x, block.thread_rank() / tileSize, tile.thread_rank());
    else if constexpr (func_traits::arity == 4
                       && std::is_pointer_v<
                           std::tuple_element_t<0, typename func_traits::arguments_t>>)
      f(shmem, blockIdx.x, block.thread_rank() / tileSize, tile.thread_rank());
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
    template <typename FTraits> static constexpr unsigned computeArity() noexcept {
      unsigned res{0};
      if constexpr (FTraits::arity != 0)
        res = FTraits::arity
              - (std::is_pointer_v<std::tuple_element_t<0, typename FTraits::arguments_t>> ? 1 : 0);
      return res;
    }
    template <typename Tn, typename F>
    void operator()(std::initializer_list<Tn> range, const F &f) const {
      (*this)(std::vector<Tn>{range}, f);
    }
    template <typename Range, typename F,
              enable_if_t<((std::declval<Range &>.end() - std::declval<Range &>.begin()) > 0)> = 0>
    void operator()(Range &&range, const F &f) const {
      (*this)({range.end() - range.begin()}, f);
    }
    template <typename Tn, typename F> void operator()(const std::vector<Tn> &range, F &&f) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::ref_cuda_context(incomingProc).event_spare(incomingStreamid));
      // need to work on __device__ func as well
      // if constexpr (arity == 1)
      if (range.size() == 1)
        context.spare_launch(streamid, {(range[0] + 127) / 128, 128, shmemBytes}, thread_launch,
                             range[0], f);
      // else if constexpr (arity == 2)
      else if (range.size() == 2)
        context.spare_launch(streamid, {range[0], range[1], shmemBytes}, block_thread_launch, f);
      // else if constexpr (arity == 3)
      else if (range.size() == 3)
        context.spare_launch(streamid, {range[0], range[1] * range[2], shmemBytes},
                             block_tile_lane_launch, range[2], f);
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.spare_event_record(streamid);
    }

    /// for_each
    template <class ForwardIt, class UnaryFunction>
    void for_each_impl(std::random_access_iterator_tag, ForwardIt &&first, ForwardIt &&last,
                       UnaryFunction &&f) const {
      using IterT = remove_cvref_t<ForwardIt>;
      const auto dist = last - first;
      (*this)({dist}, [first, f](typename std::iterator_traits<IterT>::difference_type tid) {
        f(first + tid);
      });
    }
    template <class ForwardIt, class UnaryFunction>
    void for_each(ForwardIt &&first, ForwardIt &&last, UnaryFunction &&f) const {
      for_each_impl(typename std::iterator_traits<remove_cvref_t<ForwardIt>>::iterator_category{},
                    FWD(first), FWD(last), FWD(f));
    }
    /// inclusive scan
    template <class InputIt, class OutputIt, class BinaryOperation>
    constexpr void inclusive_scan_impl(std::random_access_iterator_tag, InputIt &&first,
                                       InputIt &&last, OutputIt &&d_first,
                                       BinaryOperation &&binary_op) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::ref_cuda_context(incomingProc).event_spare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      std::size_t temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes, first, d_first, binary_op, dist,
                                     context.stream_spare(streamid));
      void *d_tmp = context.borrow(temp_storage_bytes);
      cub::DeviceScan::InclusiveScan(d_tmp, temp_storage_bytes, first, d_first, binary_op, dist,
                                     context.stream_spare(streamid));
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.spare_event_record(streamid);
    }
    template <class InputIt, class OutputIt,
              class BinaryOperation = std::plus<remove_cvref_t<decltype(*std::declval<InputIt>())>>>
    constexpr void inclusive_scan(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                                  BinaryOperation &&binary_op = {}) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      inclusive_scan_impl(
          typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{}, FWD(first),
          FWD(last), FWD(d_first), FWD(binary_op));
    }
    /// exclusive scan
    template <class InputIt, class OutputIt, class T, class BinaryOperation>
    constexpr void exclusive_scan_impl(std::random_access_iterator_tag, InputIt &&first,
                                       InputIt &&last, OutputIt &&d_first, T init,
                                       BinaryOperation &&binary_op) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::ref_cuda_context(incomingProc).event_spare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      std::size_t temp_storage_bytes = 0;
      cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes, first, d_first, binary_op, init,
                                     dist, context.stream_spare(streamid));
      void *d_tmp = context.borrow(temp_storage_bytes);
      cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes, first, d_first, binary_op, init,
                                     dist, context.stream_spare(streamid));
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.spare_event_record(streamid);
    }
    template <class InputIt, class OutputIt,
              class T = remove_cvref_t<decltype(*std::declval<InputIt>())>,
              class BinaryOperation = std::plus<T>>
    constexpr void exclusive_scan(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                                  T init = monoid_op<BinaryOperation>::e,
                                  BinaryOperation &&binary_op = {}) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      exclusive_scan_impl(
          typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{}, FWD(first),
          FWD(last), FWD(d_first), init, FWD(binary_op));
    }
    /// reduce
    template <class InputIt, class OutputIt, class T, class BinaryOperation>
    constexpr void reduce_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                               OutputIt &&d_first, T init, BinaryOperation &&binary_op) const {
      auto &context = Cuda::context(procid);
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(
            streamid, Cuda::ref_cuda_context(incomingProc).event_spare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      std::size_t temp_storage_bytes = 0;
      cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, first, d_first, dist, binary_op, init,
                                context.stream_spare(streamid));
      void *d_tmp = context.borrow(temp_storage_bytes);
      cub::DeviceReduce::Reduce(d_tmp, temp_storage_bytes, first, d_first, dist, binary_op, init,
                                context.stream_spare(streamid));
      if (this->shouldSync()) context.syncStreamSpare(streamid);
      context.spare_event_record(streamid);
    }
    template <class InputIt, class OutputIt,
              class T = remove_cvref_t<decltype(*std::declval<InputIt>())>,
              class BinaryOp = std::plus<T>>
    constexpr void reduce(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                          T init = monoid_op<BinaryOp>::e, BinaryOp &&binary_op = {}) const {
      static_assert(
          is_same_v<typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category,
                    typename std::iterator_traits<remove_cvref_t<OutputIt>>::iterator_category>,
          "Input Iterator and Output Iterator should be from the same category");
      reduce_impl(typename std::iterator_traits<remove_cvref_t<InputIt>>::iterator_category{},
                  FWD(first), FWD(last), FWD(d_first), init, FWD(binary_op));
    }

  protected:
    // bool do_launch(const ParallelTask &) const noexcept;
    friend struct ExecutionPolicyInterface<CudaExecutionPolicy>;

    // std::size_t blockGranularity{128};
    ProcID incomingProc{0};
    StreamID incomingStreamid{0};
    StreamID streamid{0};
    ProcID procid{0};           ///< 0-th gpu
    std::size_t shmemBytes{0};  ///< amount of shared memory passed
  };

  constexpr CudaExecutionPolicy cuda_exec() noexcept { return CudaExecutionPolicy{}; }
  constexpr CudaExecutionPolicy par_exec(cuda_execution_tag) noexcept {
    return CudaExecutionPolicy{};
  }

}  // namespace zs
