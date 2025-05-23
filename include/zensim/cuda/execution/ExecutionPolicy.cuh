#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_merge_sort.cuh>
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

#include "zensim/ZpcFunction.hpp"

#if ZS_ENABLE_CUDA && !defined(__CUDACC__)
#  error "ZS_ENABLE_CUDA defined but the compiler is not defining the __CUDACC__ macro as expected"
// Some tooling environments will still function better if we do this here.
#  define __CUDACC__
#endif

/// extracted from compiler error message...
template <class Tag, class... CapturedVarTypePack> struct __nv_dl_wrapper_t;
template <class U, U func, unsigned int> struct __nv_dl_tag;

namespace zs {

  namespace detail {

    template <typename F, typename ArgSeq, typename = void> struct deduce_fts {
      static constexpr bool fts_available
          = is_valid([](auto t) -> decltype(zs::function_traits<typename decltype(t)::type>{},
                                            void()) {})(zs::wrapt<F>{});

      template <typename IterOrIndex, typename = void> struct iter_arg {
        using type = IterOrIndex;
      };
      template <typename Iter> struct iter_arg<Iter, void_t<decltype(*declval<Iter>())>> {
        using type = decltype(*declval<Iter>());
      };
      template <typename IterOrIndex> using iter_arg_t = typename iter_arg<IterOrIndex>::type;

      template <typename Seq> struct impl;
      template <typename... Args> struct impl<type_seq<Args...>> {
        static constexpr auto deduce_args_t() noexcept {
          if constexpr (fts_available)
            return typename function_traits<F>::arguments_t{};
          else {
            if constexpr (is_invocable_v<F, int, iter_arg_t<Args>...>)
              return type_seq<int, iter_arg_t<Args>...>{};
            else if constexpr (is_invocable_v<F, void *, iter_arg_t<Args>...>)
              return type_seq<void *, iter_arg_t<Args>...>{};
            else
              return type_seq<iter_arg_t<Args>...>{};
          }
        }
        static constexpr auto deduce_return_t() noexcept {
          if constexpr (fts_available)
            return wrapt<typename function_traits<F>::return_t>{};
          else {
            if constexpr (is_invocable_v<F, int, iter_arg_t<Args>...>)
              return wrapt<invoke_result_t<F, int, iter_arg_t<Args>...>>{};
            else if constexpr (is_invocable_v<F, void *, iter_arg_t<Args>...>)
              return wrapt<invoke_result_t<F, void *, iter_arg_t<Args>...>>{};
            else
              return wrapt<invoke_result_t<F, iter_arg_t<Args>...>>{};
          }
        }
        static constexpr size_t deduce_arity() noexcept {
          if constexpr (fts_available)
            return function_traits<F>::arity;
          else
            return decltype(deduce_args_t())::count;
        }
      };

      using arguments_t =
          typename decltype(impl<ArgSeq>::deduce_args_t())::template functor<zs::tuple>;
      using first_argument_t = typename decltype(impl<ArgSeq>::deduce_args_t())::template type<0>;

      using return_t = typename decltype(impl<ArgSeq>::deduce_return_t())::type;
      static constexpr size_t arity = impl<ArgSeq>::deduce_arity();

      static_assert(is_same_v<return_t, void>,
                    "callable for execution policy should only return void");
    };
    template <typename F, typename ArgSeq, typename... Ts>
    struct deduce_fts<F, ArgSeq, zs::tuple<Ts...>> {
      using param_arg_t = zs::tuple<Ts...>;
      static constexpr bool fts_available
          = is_valid([](auto t) -> decltype(zs::function_traits<typename decltype(t)::type>{},
                                            void()) {})(zs::wrapt<F>{});

      template <typename IterOrIndex, typename = void> struct iter_arg {
        using type = IterOrIndex;
      };
      template <typename Iter> struct iter_arg<Iter, void_t<decltype(*declval<Iter>())>> {
        using type = decltype(*declval<Iter>());
      };
      template <typename IterOrIndex> using iter_arg_t = typename iter_arg<IterOrIndex>::type;

      template <typename Seq> struct impl;
      template <typename... Args> struct impl<type_seq<Args...>> {
        static constexpr auto deduce_args_t() noexcept {
          if constexpr (fts_available)
            return typename function_traits<F>::arguments_t{};
          else {
            if constexpr (is_invocable_v<F, int, iter_arg_t<Args>..., param_arg_t>)
              return type_seq<int, iter_arg_t<Args>..., param_arg_t>{};
            else if constexpr (is_invocable_v<F, void *, iter_arg_t<Args>..., param_arg_t>)
              return type_seq<void *, iter_arg_t<Args>..., param_arg_t>{};
            else
              return type_seq<iter_arg_t<Args>..., param_arg_t>{};
          }
        }
        static constexpr auto deduce_return_t() noexcept {
          if constexpr (fts_available)
            return wrapt<typename function_traits<F>::return_t>{};
          else {
            if constexpr (is_invocable_v<F, int, iter_arg_t<Args>..., param_arg_t>)
              return wrapt<invoke_result_t<F, int, iter_arg_t<Args>..., param_arg_t>>{};
            else if constexpr (is_invocable_v<F, void *, iter_arg_t<Args>..., param_arg_t>)
              return wrapt<invoke_result_t<F, void *, iter_arg_t<Args>..., param_arg_t>>{};
            else
              return wrapt<invoke_result_t<F, iter_arg_t<Args>..., param_arg_t>>{};
          }
        }
        static constexpr size_t deduce_arity() noexcept {
          if constexpr (fts_available)
            return function_traits<F>::arity;
          else
            return decltype(deduce_args_t())::count;
        }
      };

      using arguments_t =
          typename decltype(impl<ArgSeq>::deduce_args_t())::template functor<zs::tuple>;
      using first_argument_t = typename decltype(impl<ArgSeq>::deduce_args_t())::template type<0>;

      using return_t = typename decltype(impl<ArgSeq>::deduce_return_t())::type;
      static constexpr size_t arity = impl<ArgSeq>::deduce_arity();

      static_assert(is_same_v<return_t, void>,
                    "callable for execution policy should only return void");
    };

    template <bool withIndex, typename Tn, typename F, typename ZipIter, size_t... Is>
    __forceinline__ __device__ void range_foreach(wrapv<withIndex>, Tn i, F &&f, ZipIter &&iter,
                                                  index_sequence<Is...>) {
      (zs::get<Is>(iter.iters).advance(i), ...);
      if constexpr (withIndex)
        f(i, *zs::get<Is>(iter.iters)...);
      else {
        f(*zs::get<Is>(iter.iters)...);
      }
    }
    template <bool withIndex, typename ShmT, typename Tn, typename F, typename ZipIter,
              size_t... Is>
    __forceinline__ __device__ void range_foreach(wrapv<withIndex>, ShmT *shmem, Tn i, F &&f,
                                                  ZipIter &&iter, index_sequence<Is...>) {
      (zs::get<Is>(iter.iters).advance(i), ...);
      using func_traits
          = detail::deduce_fts<remove_cvref_t<F>, typename RM_REF_T(iter.iters)::tuple_types>;
      using shmem_ptr_t = typename func_traits::first_argument_t;
      if constexpr (withIndex)
        f(reinterpret_cast<shmem_ptr_t>(shmem), i, *zs::get<Is>(iter.iters)...);
      else
        f(reinterpret_cast<shmem_ptr_t>(shmem), *zs::get<Is>(iter.iters)...);
    }

    template <bool withIndex, typename Tn, typename F, typename ZipIter, typename ParamTuple,
              size_t... Is>
    __forceinline__ __device__ void range_foreach_with_params(wrapv<withIndex>, Tn i, F &&f,
                                                              ZipIter &&iter, ParamTuple &&params,
                                                              index_sequence<Is...>) {
      ((void)zs::get<Is>(iter.iters).advance(i), ...);
      if constexpr (withIndex)
        f(i, *zs::get<Is>(iter.iters)..., FWD(params));
      else {
        f(*zs::get<Is>(iter.iters)..., FWD(params));
      }
    }
    template <bool withIndex, typename ShmT, typename Tn, typename F, typename ZipIter,
              typename ParamTuple, size_t... Is>
    __forceinline__ __device__ void range_foreach_with_params(wrapv<withIndex>, ShmT *shmem, Tn i,
                                                              F &&f, ZipIter &&iter,
                                                              ParamTuple &&params,
                                                              index_sequence<Is...>) {
      ((void)zs::get<Is>(iter.iters).advance(i), ...);
      using func_traits
          = detail::deduce_fts<remove_cvref_t<F>, typename RM_REF_T(iter.iters)::tuple_types>;
      using shmem_ptr_t = typename func_traits::first_argument_t;
      if constexpr (withIndex)
        f(reinterpret_cast<shmem_ptr_t>(shmem), i, *zs::get<Is>(iter.iters)..., FWD(params));
      else
        f(reinterpret_cast<shmem_ptr_t>(shmem), *zs::get<Is>(iter.iters)..., FWD(params));
    }
  }  // namespace detail

  // =========================  signature  ==============================
  // loopbody signature: (scratchpadMemory, blockid, warpid, threadid)
  template <typename Tn, typename F> __global__ void thread_launch(Tn n, F f) {
    extern __shared__ std::max_align_t shmem[];
    Tn id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
      using func_traits = detail::deduce_fts<F, type_seq<RM_CVREF_T(id)>>;
      static_assert(func_traits::arity >= 1 && func_traits::arity <= 2,
                    "thread_launch requires arity to be within [1, 2]");
      if constexpr (func_traits::arity == 1)
        f(id);
      else if constexpr (func_traits::arity == 2) {
        static_assert(std::is_pointer_v<typename func_traits::first_argument_t>,
                      "the first argument must be a shmem pointer");
        f(reinterpret_cast<typename func_traits::first_argument_t>(shmem), id);
      }
    }
  }
  template <typename F> __global__ void block_thread_launch(F f) {
    extern __shared__ std::max_align_t shmem[];
    using func_traits
        = detail::deduce_fts<F, type_seq<RM_CVREF_T(blockIdx.x), RM_CVREF_T(threadIdx.x)>>;
    static_assert(func_traits::arity >= 2 && func_traits::arity <= 3,
                  "block_thread_launch requires arity to be within [2, 3]");
    if constexpr (func_traits::arity == 2) {
      static_assert(!std::is_pointer_v<typename func_traits::first_argument_t>,
                    "the first argument must NOT be a shmem pointer");
      f(blockIdx.x, threadIdx.x);
    } else if constexpr (func_traits::arity == 3) {
      static_assert(std::is_pointer_v<typename func_traits::first_argument_t>,
                    "the first argument must be a shmem pointer");
      f(reinterpret_cast<typename func_traits::first_argument_t>(shmem), blockIdx.x, threadIdx.x);
    }
  }

  template <typename Tn, typename F, typename ZipIter> __global__
      enable_if_type<is_ra_iter_v<ZipIter>
                     && (is_tuple<typename std::iterator_traits<ZipIter>::reference>::value
                         || is_std_tuple<typename std::iterator_traits<ZipIter>::reference>::value)>
      range_launch(Tn n, F f, ZipIter iter) {
    extern __shared__ std::max_align_t shmem[];
    Tn id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
      using func_traits = detail::deduce_fts<F, typename RM_REF_T(iter.iters)::tuple_types>;
      constexpr auto numArgs = zs::tuple_size_v<typename std::iterator_traits<ZipIter>::reference>;
      constexpr auto indices = make_index_sequence<numArgs>{};
      static_assert(func_traits::arity >= numArgs && func_traits::arity <= numArgs + 2,
                    "range_launch arity does not match with numArgs");
      if constexpr (func_traits::arity == numArgs) {
        detail::range_foreach(false_c, id, f, iter, indices);
      } else if constexpr (func_traits::arity == numArgs + 1) {
        static_assert(
            is_integral_v<typename func_traits::first_argument_t>
                || is_pointer_v<typename func_traits::first_argument_t>,
            "when arity equals numArgs+1, the first argument should be a shmem pointer or an "
            "integer");
        if constexpr (is_integral_v<typename func_traits::first_argument_t>)
          detail::range_foreach(true_c, id, f, iter, indices);
        else if constexpr (is_pointer_v<typename func_traits::first_argument_t>)
          detail::range_foreach(false_c,
                                reinterpret_cast<typename func_traits::first_argument_t>(shmem), id,
                                f, iter, indices);
      } else if constexpr (func_traits::arity == numArgs + 2) {
        static_assert(is_pointer_v<typename func_traits::first_argument_t>,
                      "when arity equals numArgs+2, the first argument should be a shmem pointer");
        detail::range_foreach(true_c,
                              reinterpret_cast<typename func_traits::first_argument_t>(shmem), id,
                              f, iter, indices);
      }
    }
  }
  template <typename Tn, typename F, typename ZipIter, typename... Args> __global__
      enable_if_type<is_ra_iter_v<ZipIter>
                     && (is_tuple<typename std::iterator_traits<ZipIter>::reference>::value
                         || is_std_tuple<typename std::iterator_traits<ZipIter>::reference>::value)>
      range_launch_with_params(Tn n, F f, ZipIter iter, zs::tuple<Args...> params) {
    extern __shared__ std::max_align_t shmem[];
    Tn id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
      /// @note beware of the discrepency here
      using func_traits
          = detail::deduce_fts<F, typename RM_REF_T(iter.iters)::tuple_types, zs::tuple<Args...>>;
      constexpr auto numArgs
          = zs::tuple_size_v<typename std::iterator_traits<ZipIter>::reference> + 1;
      constexpr auto indices = make_index_sequence<numArgs - 1>{};

      static_assert(func_traits::arity >= numArgs && func_traits::arity <= numArgs + 2,
                    "range_launch_with_params arity does not match with numArgs");
      if constexpr (func_traits::arity == numArgs) {
        detail::range_foreach_with_params(false_c, id, f, iter, params, indices);
      } else if constexpr (func_traits::arity == numArgs + 1) {
        static_assert(is_integral_v<typename func_traits::first_argument_t>
                          || is_pointer_v<typename func_traits::first_argument_t>,
                      "when arity equals numArgs+1 (tail for params), the first argument should be "
                      "a shmem pointer or an integer");
        if constexpr (is_integral_v<typename func_traits::first_argument_t>)
          detail::range_foreach_with_params(true_c, id, f, iter, params, indices);
        else if constexpr (is_pointer_v<typename func_traits::first_argument_t>)
          detail::range_foreach_with_params(
              false_c, reinterpret_cast<typename func_traits::first_argument_t>(shmem), id, f, iter,
              params, indices);
        else
          static_assert(always_false<Tn>, "slot reserved...");
      } else if constexpr (func_traits::arity == numArgs + 2) {
        static_assert(is_pointer_v<typename func_traits::first_argument_t>,
                      "when arity equals numArgs+2 (tail for params), the first argument should be "
                      "a shmem pointer");
        detail::range_foreach_with_params(
            true_c, reinterpret_cast<typename func_traits::first_argument_t>(shmem), id, f, iter,
            params, indices);
      }
    }
  }
  namespace cg = cooperative_groups;
  template <typename Tn, typename F> __global__ void block_tile_lane_launch(Tn tileSize, F f) {
    extern __shared__ std::max_align_t shmem[];
    cg::thread_block block = cg::this_thread_block();
    cg::thread_group tile = cg::tiled_partition(block, tileSize);
    using func_traits = detail::deduce_fts<
        F, type_seq<RM_CVREF_T(blockIdx.x), RM_CVREF_T(block.thread_rank() / tileSize),
                    RM_CVREF_T(tile.thread_rank())>>;
    static_assert(func_traits::arity >= 3 && func_traits::arity <= 4,
                  "block_tile_lane_launch requires arity to be within [3, 4]");
    if constexpr (func_traits::arity == 3) {
      static_assert(!std::is_pointer_v<typename func_traits::first_argument_t>,
                    "the first argument must NOT be a shmem pointer");
      f(blockIdx.x, block.thread_rank() / tileSize, tile.thread_rank());
    } else if constexpr (func_traits::arity == 4) {
      static_assert(std::is_pointer_v<typename func_traits::first_argument_t>,
                    "the first argument must be a shmem pointer");
      f(reinterpret_cast<typename func_traits::first_argument_t>(shmem), blockIdx.x,
        block.thread_rank() / tileSize, tile.thread_rank());
    }
  }

  namespace detail {
    template <typename, typename> struct function_traits_impl;

    template <auto F, unsigned int I, typename R, typename... Args>
    struct function_traits_impl<__nv_dl_tag<R (*)(Args...), F, I>> {
      static constexpr size_t arity = sizeof...(Args);
      using return_t = R;
      using arguments_t = zs::tuple<Args...>;
    };
    template <class Tag, class... CapturedVarTypePack>
    struct function_traits_impl<__nv_dl_wrapper_t<Tag, CapturedVarTypePack...>>
        : function_traits_impl<Tag> {};
  }  // namespace detail

  struct CudaExecutionPolicy;
  ZPC_API extern ZSPmrAllocator<> get_temporary_memory_source(const CudaExecutionPolicy &pol);

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
    CudaExecutionPolicy &shmem(size_t bytes) {
      shmemBytes = bytes;
      return *this;
    }
    CudaExecutionPolicy &block(size_t tpb) {
      blockSize = tpb;
      return *this;
    }
#if 0
    template <typename FTraits> static constexpr unsigned computeArity() noexcept {
      unsigned res{0};
      if constexpr (FTraits::arity != 0)
        res = FTraits::arity
              - (std::is_pointer_v<zs::tuple_element_t<0, typename FTraits::arguments_t>> ? 1 : 0);
      return res;
    }
#endif

    void syncCtx(const source_location &loc = source_location::current()) const {
      auto &context = Cuda::context(getProcid());
      context.syncStreamSpare(streamid, loc);
    }

    template <typename Ts, typename Is, typename F>
    void operator()(Collapse<Ts, Is> dims, F &&f,
                    const source_location &loc = source_location::current()) const {
      using namespace index_literals;
      constexpr auto dim = Collapse<Ts, Is>::dim;
      auto &context = Cuda::context(getProcid());
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(context.streamSpare(streamid), loc);

      // need to work on __device__ func as well
      u32 ec = 0;
      if constexpr (dim == 1) {
        LaunchConfig lc{};
        if (blockSize == 0)
          lc = LaunchConfig{true_c, dims.get(0_th), shmemBytes};
        else
          lc = LaunchConfig{(dims.get(0_th) + blockSize - 1) / blockSize, blockSize, shmemBytes};
        ec = cuda_safe_launch(loc, context, streamid, std::move(lc), thread_launch, dims.get(0_th),
                              f);
      }
      // else if constexpr (arity == 2)
      else if constexpr (dim == 2) {
        ec = cuda_safe_launch(loc, context, streamid, {dims.get(0_th), dims.get(1_th), shmemBytes},
                              block_thread_launch, f);
      }
      // else if constexpr (arity == 3)
      else if constexpr (dim == 3) {
        ec = cuda_safe_launch(loc, context, streamid,
                              {dims.get(0_th), dims.get(1_th) * dims.get(2_th), shmemBytes},
                              block_tile_lane_launch, dims.get(2_th), f);
      }
      static_assert(dim >= 1 && dim <= 3,
                    "launch dimensions other than 1, 2, 3 are not supported yet");
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) {
        context.syncStreamSpare(streamid, loc);
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (ec == 0) ec = Cuda::get_last_cuda_rt_error();
#endif
      }
      checkCuKernelLaunchError(ec, context, fmt::format("Spare [{}]", streamid), loc);
      context.recordEventSpare(streamid, loc);
    }
    template <typename Range, typename F>
    auto operator()(Range &&range, F &&f,
                    const source_location &loc = source_location::current()) const {
      auto &context = Cuda::context(getProcid());
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
        lc = LaunchConfig{true_c, dist, shmemBytes};
      else
        lc = LaunchConfig{(dist + blockSize - 1) / blockSize, blockSize, shmemBytes};

      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(context.streamSpare(streamid), loc);

      u32 ec = 0;
      if constexpr (is_zip_iterator_v<IterT>) {
        ec = cuda_safe_launch(loc, context, streamid, std::move(lc), range_launch, dist, f,
                              std::begin(FWD(range)));
      } else {  // wrap the non-zip range in a zip range
        ec = cuda_safe_launch(loc, context, streamid, std::move(lc), range_launch, dist, f,
                              std::begin(zip(FWD(range))));
      }

      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) {
        context.syncStreamSpare(streamid, loc);
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (ec == 0) ec = Cuda::get_last_cuda_rt_error();
#endif
      }
      checkCuKernelLaunchError(ec, context, fmt::format("Spare [{}]", streamid), loc);
      context.recordEventSpare(streamid, loc);
    }
    template <typename Range, typename... Args, typename F>
    auto operator()(Range &&range, const zs::tuple<Args...> &params, F &&f,
                    const source_location &loc = source_location::current()) const {
      auto &context = Cuda::context(getProcid());
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
        lc = LaunchConfig{true_c, dist, shmemBytes};
      else
        lc = LaunchConfig{(dist + blockSize - 1) / blockSize, blockSize, shmemBytes};

      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(context.streamSpare(streamid), loc);

      u32 ec = 0;
      if constexpr (is_zip_iterator_v<IterT>) {
        ec = cuda_safe_launch(loc, context, streamid, std::move(lc), range_launch_with_params, dist,
                              f, std::begin(FWD(range)), params);
      } else {  // wrap the non-zip range in a zip range
        ec = cuda_safe_launch(loc, context, streamid, std::move(lc), range_launch_with_params, dist,
                              f, std::begin(zip(FWD(range))), params);
      }

      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) {
        context.syncStreamSpare(streamid, loc);
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (ec == 0) ec = Cuda::get_last_cuda_rt_error();
#endif
      }
      checkCuKernelLaunchError(ec, context, fmt::format("Spare [{}]", streamid), loc);
      context.recordEventSpare(streamid, loc);
    }

    /// for_each
    template <class ForwardIt, class UnaryFunction>
    void for_each_impl(std::random_access_iterator_tag, ForwardIt &&first, ForwardIt &&last,
                       UnaryFunction &&f, const source_location &loc) const {
      using IterT = remove_cvref_t<ForwardIt>;
      const auto dist = last - first;
      (*this)(
          Collapse{dist},
          [first = FWD(first), f = FWD(f)] __device__(
              typename std::iterator_traits<IterT>::difference_type tid) mutable {
            f(*(first + tid));
          },
          loc);
    }
    template <class ForwardIt, class UnaryFunction>
    void for_each(ForwardIt &&first, ForwardIt &&last, UnaryFunction &&f,
                  const source_location &loc = source_location::current()) const {
      static_assert(is_ra_iter_v<remove_cvref_t<ForwardIt>>,
                    "Iterator should be a random access iterator");
      for_each_impl(std::random_access_iterator_tag{}, FWD(first), FWD(last), FWD(f), loc);
    }
    /// inclusive scan
    template <class InputIt, class OutputIt, class BinaryOperation>
    void inclusive_scan_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                             OutputIt &&d_first, BinaryOperation &&binary_op,
                             const source_location &loc) const {
      auto &context = Cuda::context(getProcid());
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      size_t temp_bytes = 0;
      auto stream = (cudaStream_t)context.streamSpare(streamid);
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(stream, loc);
#if 0
      thrust::inclusive_scan(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(first.operator->()),
                             thrust::device_pointer_cast(first.operator->() + dist),
                             thrust::device_pointer_cast(d_first.operator->()), FWD(binary_op));
#else
      cub::DeviceScan::InclusiveScan(nullptr, temp_bytes, first, d_first, binary_op, dist, stream);

      void *d_tmp = context.streamMemAlloc(temp_bytes, stream, loc);
      cub::DeviceScan::InclusiveScan(d_tmp, temp_bytes, first, d_first, binary_op, dist, stream);
      context.streamMemFree(d_tmp, stream, loc);
#endif
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid, loc);
      context.recordEventSpare(streamid, loc);
    }
    template <class InputIt, class OutputIt,
              class BinaryOperation = plus<remove_cvref_t<decltype(*declval<InputIt>())>>>
    void inclusive_scan(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                        BinaryOperation &&binary_op = {},
                        const source_location &loc = source_location::current()) const {
      static_assert(is_ra_iter_v<remove_cvref_t<InputIt>> && is_ra_iter_v<remove_cvref_t<OutputIt>>,
                    "Input Iterator and Output Iterator should both be random access iterators");
      inclusive_scan_impl(std::random_access_iterator_tag{}, FWD(first), FWD(last), FWD(d_first),
                          FWD(binary_op), loc);
    }
    /// exclusive scan
    template <class InputIt, class OutputIt, class T, class BinaryOperation>
    void exclusive_scan_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                             OutputIt &&d_first, T init, BinaryOperation &&binary_op,
                             const source_location &loc) const {
      auto &context = Cuda::context(getProcid());
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      const auto dist = last - first;
      auto stream = (cudaStream_t)context.streamSpare(streamid);
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(stream, loc);
#if 0
      thrust::exclusive_scan(
          thrust::cuda::par.on(stream), thrust::device_pointer_cast(first.operator->()),
          thrust::device_pointer_cast(first.operator->() + dist),
          thrust::device_pointer_cast(d_first.operator->()), init, FWD(binary_op));
#else
      size_t temp_bytes = 0;
      cub::DeviceScan::ExclusiveScan(nullptr, temp_bytes, first, d_first, binary_op, init, dist,
                                     stream);
      void *d_tmp = context.streamMemAlloc(temp_bytes, stream, loc);
      cub::DeviceScan::ExclusiveScan(d_tmp, temp_bytes, first, d_first, binary_op, init, dist,
                                     stream);
      context.streamMemFree(d_tmp, stream, loc);
#endif
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid, loc);
      context.recordEventSpare(streamid, loc);
    }
    template <class InputIt, class OutputIt,
              class BinaryOperation
              = plus<typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>>
    void exclusive_scan(
        InputIt &&first, InputIt &&last, OutputIt &&d_first,
        typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type init
        = deduce_identity<BinaryOperation,
                          typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>(),
        BinaryOperation &&binary_op = {},
        const source_location &loc = source_location::current()) const {
      static_assert(is_ra_iter_v<remove_cvref_t<InputIt>> && is_ra_iter_v<remove_cvref_t<OutputIt>>,
                    "Input Iterator and Output Iterator should both be random access iterators");
      exclusive_scan_impl(std::random_access_iterator_tag{}, FWD(first), FWD(last), FWD(d_first),
                          init, FWD(binary_op), loc);
    }
    /// reduce
    template <class InputIt, class OutputIt, class T, class BinaryOperation>
    void reduce_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                     OutputIt &&d_first, T init, BinaryOperation &&binary_op,
                     const source_location &loc) const {
      auto &context = Cuda::context(getProcid());
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      using IterT = remove_cvref_t<InputIt>;
      using ValueT = typename std::iterator_traits<IterT>::value_type;
      const auto dist = last - first;
      size_t temp_bytes = 0;
      auto stream = (cudaStream_t)context.streamSpare(streamid);
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      if (this->shouldProfile()) timer = context.tick(stream, loc);
#if 0
      ValueT res
          = thrust::reduce(thrust::device, thrust::device_pointer_cast(first.operator->()),
                           thrust::device_pointer_cast(last.operator->()), (ValueT)init, binary_op);
      (*this)(
          Collapse{1}, [d_first = FWD(d_first), res] __device__(int) mutable { *d_first = res; },
          loc);
#else
      cub::DeviceReduce::Reduce(nullptr, temp_bytes, first, d_first, dist, binary_op, init, stream);
      void *d_tmp = context.streamMemAlloc(temp_bytes, stream, loc);
      cub::DeviceReduce::Reduce(d_tmp, temp_bytes, first, d_first, dist, binary_op, init, stream);
      context.streamMemFree(d_tmp, stream, loc);
#endif
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid, loc);
      context.recordEventSpare(streamid, loc);
    }
    template <class InputIt, class OutputIt,
              // class T = remove_cvref_t<decltype(*declval<InputIt>())>,
              class BinaryOp
              = plus<typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>>
    void reduce(InputIt &&first, InputIt &&last, OutputIt &&d_first,
                typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type init
                = deduce_identity<
                    BinaryOp, typename std::iterator_traits<remove_cvref_t<InputIt>>::value_type>(),
                BinaryOp &&binary_op = {},
                const source_location &loc = source_location::current()) const {
      static_assert(is_ra_iter_v<remove_cvref_t<InputIt>> && is_ra_iter_v<remove_cvref_t<OutputIt>>,
                    "Input Iterator and Output Iterator should both be random access iterators");
      reduce_impl(std::random_access_iterator_tag{}, FWD(first), FWD(last), FWD(d_first), init,
                  FWD(binary_op), loc);
    }
    /// merge sort
    template <class KeyIter, class ValueIter, typename CompareOpT>
    enable_if_type<is_ra_iter_v<remove_reference_t<KeyIter>>
                   && is_ra_iter_v<remove_reference_t<ValueIter>>>
    merge_sort_pair(
        KeyIter &&keys, ValueIter &&vals,
        typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type count,
        CompareOpT &&compOp, const source_location &loc = source_location::current()) const {
      auto &context = Cuda::context(getProcid());
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      if (count) {
        size_t temp_bytes = 0;
        auto stream = (cudaStream_t)context.streamSpare(streamid);
        Cuda::CudaContext::StreamExecutionTimer *timer{};
        if (this->shouldProfile()) timer = context.tick(stream, loc);
        cub::DeviceMergeSort::StableSortPairs(nullptr, temp_bytes, keys, vals, count, compOp,
                                              stream);
        // context.syncStreamSpare(streamid, loc);
        void *d_tmp = context.streamMemAlloc(temp_bytes, stream, loc);
        // cuMemAllocAsync((CUdeviceptr *)&d_tmp, temp_bytes, stream);
        cub::DeviceMergeSort::StableSortPairs(d_tmp, temp_bytes, keys, vals, count, compOp, stream);
        // context.syncStreamSpare(streamid, loc);
        // cuMemFreeAsync((CUdeviceptr)d_tmp, stream);
        context.streamMemFree(d_tmp, stream, loc);
        if (this->shouldProfile()) context.tock(timer, loc);
      }
      if (this->shouldSync()) context.syncStreamSpare(streamid, loc);
      context.recordEventSpare(streamid, loc);
    }
    template <class KeyIter, typename CompareOpT>
    enable_if_type<is_ra_iter_v<remove_reference_t<KeyIter>>> merge_sort(
        KeyIter &&first, KeyIter &&last, CompareOpT &&compOp,
        const source_location &loc = source_location::current()) const {
      auto &context = Cuda::context(getProcid());
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      const auto dist = last - first;
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      auto stream = (cudaStream_t)context.streamSpare(streamid);
      if (this->shouldProfile()) timer = context.tick(stream, loc);

      size_t temp_bytes = 0;
      cub::DeviceMergeSort::StableSortKeys(nullptr, temp_bytes, first, dist, compOp, stream);
      void *d_tmp = context.streamMemAlloc(temp_bytes, stream, loc);
      cub::DeviceMergeSort::StableSortKeys(d_tmp, temp_bytes, first, dist, compOp, stream);
      context.streamMemFree(d_tmp, stream, loc);

      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid, loc);
      context.recordEventSpare(streamid, loc);
    }
    /// histogram sort
    /// radix sort pair
    template <class KeyIter, class ValueIter,
              typename Tn
              = typename std::iterator_traits<remove_reference_t<KeyIter>>::difference_type>
    enable_if_type<is_ra_iter_v<remove_reference_t<KeyIter>>
                   && is_ra_iter_v<remove_reference_t<ValueIter>>>
    radix_sort_pair(KeyIter &&keysIn, ValueIter &&valsIn, KeyIter &&keysOut, ValueIter &&valsOut,
                    Tn count = 0, int sbit = 0,
                    int ebit
                    = sizeof(typename std::iterator_traits<remove_reference_t<KeyIter>>::value_type)
                      * 8,
                    const source_location &loc = source_location::current()) const {
      using KeyIterT = remove_cvref_t<KeyIter>;
      using ValueIterT = remove_cvref_t<ValueIter>;
      using DiffT = typename std::iterator_traits<KeyIterT>::difference_type;
      using KeyT = typename std::iterator_traits<KeyIterT>::value_type;
      using ValueT = typename std::iterator_traits<ValueIterT>::value_type;
      auto &context = Cuda::context(getProcid());
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      if (count) {
        size_t temp_bytes = 0;
        auto stream = (cudaStream_t)context.streamSpare(streamid);
        Cuda::CudaContext::StreamExecutionTimer *timer{};
        if (this->shouldProfile()) timer = context.tick(stream, loc);
#if 0
        (*this)(
            Collapse{count},
            [keysIn, keysOut, valsIn, valsOut] __device__(Tn i) mutable {
              *(keysOut + i) = *(keysIn + i);
              *(valsOut + i) = *(valsIn + i);
            },
            loc);
        thrust::sort_by_key(thrust::cuda::par.on(stream),
                            thrust::device_pointer_cast(keysOut.operator->()),
                            thrust::device_pointer_cast(keysOut.operator->() + count),
                            thrust::device_pointer_cast(valsOut.operator->()));
#else
        auto allocator = get_temporary_memory_source(*this);
        Vector<KeyT> ksIn{allocator, (size_t)count}, ksOut{allocator, (size_t)count};
        Vector<ValueT> vsIn{allocator, (size_t)count}, vsOut{allocator, (size_t)count};

        (*this)(zip(range(keysIn, keysIn + count), ksIn), zs::make_tuple(),
                _zs_policy_assign_operator{}, loc);
        (*this)(zip(range(valsIn, valsIn + count), vsIn), zs::make_tuple(),
                _zs_policy_assign_operator{}, loc);

        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, ksIn.data(), ksOut.data(), vsIn.data(),
                                        vsOut.data(), count, sbit, ebit, stream);
        // context.syncStreamSpare(streamid, loc);
        void *d_tmp = context.streamMemAlloc(temp_bytes, stream, loc);
#  if 0
        void *d_tmp;
        cuMemAllocAsync((CUdeviceptr *)&d_tmp, temp_bytes, stream);
#  endif
        cub::DeviceRadixSort::SortPairs(d_tmp, temp_bytes, ksIn.data(), ksOut.data(), vsIn.data(),
                                        vsOut.data(), count, sbit, ebit, stream);
        // context.syncStreamSpare(streamid, loc);
        // cuMemFreeAsync((CUdeviceptr)d_tmp, stream);
        context.streamMemFree(d_tmp, stream, loc);

        (*this)(zip(ksOut, range(keysOut, keysOut + count)), zs::make_tuple(),
                _zs_policy_assign_operator{}, loc);
        (*this)(zip(vsOut, range(valsOut, valsOut + count)), zs::make_tuple(),
                _zs_policy_assign_operator{}, loc);
#endif
        if (this->shouldProfile()) context.tock(timer, loc);
      }
      if (this->shouldSync()) context.syncStreamSpare(streamid, loc);
      context.recordEventSpare(streamid, loc);
    }
    /// radix sort
    template <class InputIt, class OutputIt>
    void radix_sort_impl(std::random_access_iterator_tag, InputIt &&first, InputIt &&last,
                         OutputIt &&d_first, int sbit, int ebit, const source_location &loc) const {
      using KeyIterT = remove_cvref_t<InputIt>;
      using KeyT = typename std::iterator_traits<KeyIterT>::value_type;
      auto &context = Cuda::context(getProcid());
      context.setContext();
      if (this->shouldWait())
        context.spareStreamWaitForEvent(streamid,
                                        Cuda::context(incomingProc).eventSpare(incomingStreamid));
      const auto dist = last - first;
      Cuda::CudaContext::StreamExecutionTimer *timer{};
      auto stream = (cudaStream_t)context.streamSpare(streamid);
      if (this->shouldProfile()) timer = context.tick(stream, loc);
#if 0
      (*this)(
          Collapse{dist},
          [first, d_first] __device__(auto i) mutable { *(d_first + i) = *(first + i); }, loc);
      thrust::sort(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_first.operator->()),
                   thrust::device_pointer_cast(d_first.operator->() + dist));
#else
      auto allocator = get_temporary_memory_source(*this);
      Vector<KeyT> ksIn{allocator, (size_t)dist}, ksOut{allocator, (size_t)dist};
      (*this)(zip(range(first, last), ksIn), zs::make_tuple(), _zs_policy_assign_operator{}, loc);

      size_t temp_bytes = 0;
      cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, ksIn.data(), ksOut.data(), dist, sbit,
                                     ebit, stream);
      void *d_tmp = context.streamMemAlloc(temp_bytes, stream, loc);
      cub::DeviceRadixSort::SortKeys(d_tmp, temp_bytes, ksIn.data(), ksOut.data(), dist, sbit, ebit,
                                     stream);
      context.streamMemFree(d_tmp, stream, loc);

      (*this)(zip(ksOut, range(d_first, d_first + dist)), zs::make_tuple(),
              _zs_policy_assign_operator{}, loc);
#endif
      if (this->shouldProfile()) context.tock(timer, loc);
      if (this->shouldSync()) context.syncStreamSpare(streamid, loc);
      context.recordEventSpare(streamid, loc);
    }
    template <class InputIt, class OutputIt>
    void radix_sort(InputIt &&first, InputIt &&last, OutputIt &&d_first, int sbit = 0,
                    int ebit
                    = sizeof(typename std::iterator_traits<remove_reference_t<InputIt>>::value_type)
                      * 8,
                    const source_location &loc = source_location::current()) const {
      static_assert(is_ra_iter_v<remove_cvref_t<InputIt>> && is_ra_iter_v<remove_cvref_t<OutputIt>>,
                    "Input Iterator and Output Iterator should both be random access iterators");
      static_assert(is_same_v<typename std::iterator_traits<remove_reference_t<InputIt>>::pointer,
                              typename std::iterator_traits<remove_reference_t<OutputIt>>::pointer>,
                    "Input iterator pointer different from output iterator\'s");
      radix_sort_impl(std::random_access_iterator_tag{}, FWD(first), FWD(last), FWD(d_first), sbit,
                      ebit, loc);
    }

    constexpr ProcID getProcid() const noexcept {
      if (procid < 0) return Cuda::get_default_device();
      return procid;
    }
    constexpr StreamID getStreamid() const noexcept { return streamid; }
    void *getStream() const noexcept {
      return Cuda::context(getProcid()).streamSpare(getStreamid());
    }
    decltype(auto) context() { return Cuda::context(getProcid()); }
    decltype(auto) context() const { return Cuda::context(getProcid()); }
    void *getContext() const { return context().getContext(); }

    constexpr ProcID getIncomingProcid() const noexcept { return incomingProc; }
    constexpr StreamID getIncomingStreamid() const noexcept { return incomingStreamid; }

    constexpr size_t getShmemSize() const noexcept { return shmemBytes; }

  protected:
    // bool do_launch(const ParallelTask &) const noexcept;
    friend struct ExecutionPolicyInterface<CudaExecutionPolicy>;
    // template <auto flagbit> friend struct CudaLibHandle<flagbit>;

    // size_t blockGranularity{128};
    StreamID incomingStreamid{-1};
    StreamID streamid{-1};  ///< @note use CUDA default stream by default
    size_t shmemBytes{0};   ///< amount of shared memory passed
    int blockSize{0};       ///< 0 to enable auto configure
    ProcID incomingProc{0};
    ProcID procid{-1};  ///< use current default CUDA device
  };

  constexpr bool is_backend_available(CudaExecutionPolicy) noexcept { return true; }
  constexpr bool is_backend_available(cuda_exec_tag) noexcept { return true; }

  constexpr CudaExecutionPolicy cuda_exec() noexcept { return CudaExecutionPolicy{}; }
  constexpr CudaExecutionPolicy par_exec(cuda_exec_tag) noexcept { return CudaExecutionPolicy{}; }

}  // namespace zs
