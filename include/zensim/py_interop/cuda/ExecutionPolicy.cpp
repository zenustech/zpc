#include "zensim/cuda/execution/ExecutionPolicy.cuh"

#include "zensim/cuda/Cuda.h"
#include "zensim/py_interop/GenericIterator.hpp"

extern "C" {

ZPC_EXPORT zs::CudaExecutionPolicy *policy__device() { return new zs::CudaExecutionPolicy; }
ZPC_EXPORT void del_policy__device(zs::CudaExecutionPolicy *v) { delete v; }

ZPC_EXPORT void launch__device(zs::CudaExecutionPolicy *ppol, void *kernel, zs::size_t dim,
                               void **args) {
  using namespace zs;
  CudaExecutionPolicy &pol = *ppol;

  const int blockDim = 128;
  const int gridDim = (dim + blockDim - 1) / blockDim;

  auto &context = pol.context();
  Cuda::ContextGuard guard(context.getContext());

  CUresult ec = cuLaunchKernel((CUfunction)kernel, gridDim, 1, 1, blockDim, 1, 1, 0,
                               (CUstream)pol.getStream(), args, 0);

  if (pol.shouldSync()) {
    context.syncStreamSpare(pol.getStreamid(), source_location::current());
  }

  if (ec != CUDA_SUCCESS) {
    const char *errString = nullptr;
    if (cuGetErrorString) {
      cuGetErrorString(ec, &errString);
      checkCuApiError((u32)ec, source_location::current(), "[cuLaunchKernel]", errString);
    } else
      checkCuApiError((u32)ec, source_location::current(), "[cuLaunchKernel]");
  }
}

#define ZS_DEFINE_PARALLEL_PRIMITIVES(T)                                                          \
  /* reduce */                                                                                    \
  ZPC_EXPORT void reduce_sum__cuda##_##T##_1(                                                     \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                           \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::reduce(*pol, first, last, output, (T)0, zs::plus<T>{});                                   \
  }                                                                                               \
  ZPC_EXPORT void reduce_prod__cuda##_##T##_1(                                                    \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                           \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::reduce(*pol, first, last, output, (T)1, zs::multiplies<T>{});                             \
  }                                                                                               \
  ZPC_EXPORT void reduce_min__cuda##_##T##_1(                                                     \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                           \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::reduce(*pol, first, last, output, zs::detail::deduce_numeric_max<T>(), zs::getmin<T>{});  \
  }                                                                                               \
  ZPC_EXPORT void reduce_max__cuda##_##T##_1(                                                     \
      zs::CudaExecutionPolicy *p, aosoa_iterator_const_##T##_1 first,                             \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::reduce(*p, first, last, output, zs::detail::deduce_numeric_lowest<T>(), zs::getmax<T>{}); \
  }                                                                                               \
  /* exclusive scan */                                                                            \
  ZPC_EXPORT void exclusive_scan_sum__cuda##_##T##_1(                                             \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                           \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::exclusive_scan(*pol, first, last, output, (T)0, zs::plus<T>{});                           \
  }                                                                                               \
  ZPC_EXPORT void exclusive_scan_prod__cuda##_##T##_1(                                            \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                           \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::exclusive_scan(*pol, first, last, output, (T)1, zs::multiplies<T>{});                     \
  }                                                                                               \
  /* inclusive scan */                                                                            \
  ZPC_EXPORT void inclusive_scan_sum__cuda##_##T##_1(                                             \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                           \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::inclusive_scan(*pol, first, last, output, zs::plus<T>{});                                 \
  }                                                                                               \
  ZPC_EXPORT void inclusive_scan_prod__cuda##_##T##_1(                                            \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                           \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::inclusive_scan(*pol, first, last, output, zs::multiplies<T>{});                           \
  }                                                                                               \
  /* merge sort */                                                                                \
  ZPC_EXPORT void merge_sort__cuda##_##T##_1(                                                     \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_##T##_1 first, aosoa_iterator_##T##_1 last) {  \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::merge_sort(*pol, first, last);                                                            \
  }                                                                                               \
  ZPC_EXPORT void merge_sort_pair__cuda##_##T##_1(zs::CudaExecutionPolicy *pol,                   \
                                                  aosoa_iterator_##T##_1 keys,                    \
                                                  aosoa_iterator_##int##_1 vals, size_t count) {  \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::merge_sort_pair(*pol, keys, vals, count);                                                 \
  }                                                                                               \
  /* radix sort */                                                                                \
  ZPC_EXPORT void radix_sort__cuda##_##T##_1(                                                     \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_##T##_1 first, aosoa_iterator_##T##_1 last,    \
      aosoa_iterator_##T##_1 output) {                                                            \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    if constexpr (zs::is_integral_v<T>)                                                           \
      zs::radix_sort(*pol, first, last, output, 0, sizeof(T) * 8);                                \
  }                                                                                               \
  ZPC_EXPORT void radix_sort_pair__cuda##_##T##_1(                                                \
      zs::CudaExecutionPolicy *pol, aosoa_iterator_##T##_1 keysIn,                                \
      aosoa_iterator_##int##_1 valsIn, aosoa_iterator_##T##_1 keysOut,                            \
      aosoa_iterator_##int##_1 valsOut, size_t count) {                                           \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    if constexpr (zs::is_integral_v<T>)                                                           \
      zs::radix_sort_pair(*pol, keysIn, valsIn, keysOut, valsOut, count, 0, sizeof(T) * 8);       \
  }

ZS_DEFINE_PARALLEL_PRIMITIVES(int)
ZS_DEFINE_PARALLEL_PRIMITIVES(float)
ZS_DEFINE_PARALLEL_PRIMITIVES(double)
}