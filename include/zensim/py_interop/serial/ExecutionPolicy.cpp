#include "zensim/execution/ExecutionPolicy.hpp"

#include "zensim/py_interop/GenericIterator.hpp"

extern "C" {

ZPC_EXPORT zs::SequentialExecutionPolicy *policy__serial() {
  return new zs::SequentialExecutionPolicy;
}
ZPC_EXPORT void del_policy__serial(zs::SequentialExecutionPolicy *v) { delete v; }

#define ZS_DEFINE_PARALLEL_PRIMITIVES(T)                                                          \
  /* reduce */                                                                                    \
  ZPC_EXPORT void reduce_sum__seq##_##T##_1(                                                      \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                     \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::reduce(*pol, first, last, output, (T)0, zs::plus<T>{});                                   \
  }                                                                                               \
  ZPC_EXPORT void reduce_prod__seq##_##T##_1(                                                     \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                     \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::reduce(*pol, first, last, output, (T)1, zs::multiplies<T>{});                             \
  }                                                                                               \
  ZPC_EXPORT void reduce_min__seq##_##T##_1(                                                      \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                     \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::reduce(*pol, first, last, output, zs::detail::deduce_numeric_max<T>(), zs::getmin<T>{});  \
  }                                                                                               \
  ZPC_EXPORT void reduce_max__seq##_##T##_1(                                                      \
      zs::SequentialExecutionPolicy *p, aosoa_iterator_const_##T##_1 first,                       \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::reduce(*p, first, last, output, zs::detail::deduce_numeric_lowest<T>(), zs::getmax<T>{}); \
  }                                                                                               \
  /* exclusive scan */                                                                            \
  ZPC_EXPORT void exclusive_scan_sum__seq##_##T##_1(                                              \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                     \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::exclusive_scan(*pol, first, last, output, (T)0, zs::plus<T>{});                           \
  }                                                                                               \
  ZPC_EXPORT void exclusive_scan_prod__seq##_##T##_1(                                             \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                     \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::exclusive_scan(*pol, first, last, output, (T)1, zs::multiplies<T>{});                     \
  }                                                                                               \
  /* inclusive scan */                                                                            \
  ZPC_EXPORT void inclusive_scan_sum__seq##_##T##_1(                                              \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                     \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::inclusive_scan(*pol, first, last, output, zs::plus<T>{});                                 \
  }                                                                                               \
  ZPC_EXPORT void inclusive_scan_prod__seq##_##T##_1(                                             \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                     \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                         \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::inclusive_scan(*pol, first, last, output, zs::multiplies<T>{});                           \
  }                                                                                               \
  /* merge sort */                                                                                \
  ZPC_EXPORT void merge_sort__seq##_##T##_1(zs::SequentialExecutionPolicy *pol,                   \
                                            aosoa_iterator_##T##_1 first,                         \
                                            aosoa_iterator_##T##_1 last) {                        \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::merge_sort(*pol, first, last);                                                            \
  }                                                                                               \
  ZPC_EXPORT void merge_sort_pair__seq##_##T##_1(zs::SequentialExecutionPolicy *pol,              \
                                                 aosoa_iterator_##T##_1 keys,                     \
                                                 aosoa_iterator_##int##_1 vals, size_t count) {   \
    static_assert(zs::is_arithmetic_v<T>,                                                         \
                  "parallel primitives only available for arithmetic types");                     \
    zs::merge_sort_pair(*pol, keys, vals, count);                                                 \
  }

#define ZS_DEFINE_PARALLEL_PRIMITIVES_RADIX_SORT(T)                                       \
  /* radix sort */                                                                        \
  ZPC_EXPORT void radix_sort__seq##_##T##_1(                                              \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_##T##_1 first,                   \
      aosoa_iterator_##T##_1 last, aosoa_iterator_##T##_1 output) {                       \
    static_assert(zs::is_arithmetic_v<T>,                                                 \
                  "parallel primitives only available for arithmetic types");             \
    zs::radix_sort(*pol, first, last, output, 0, sizeof(T) * 8);                          \
  }                                                                                       \
  ZPC_EXPORT void radix_sort_pair__seq##_##T##_1(                                         \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_##T##_1 keysIn,                  \
      aosoa_iterator_##int##_1 valsIn, aosoa_iterator_##T##_1 keysOut,                    \
      aosoa_iterator_##int##_1 valsOut, size_t count) {                                   \
    static_assert(zs::is_arithmetic_v<T>,                                                 \
                  "parallel primitives only available for arithmetic types");             \
    zs::radix_sort_pair(*pol, keysIn, valsIn, keysOut, valsOut, count, 0, sizeof(T) * 8); \
  }

ZS_DEFINE_PARALLEL_PRIMITIVES(int)
ZS_DEFINE_PARALLEL_PRIMITIVES(float)
ZS_DEFINE_PARALLEL_PRIMITIVES(double)

ZS_DEFINE_PARALLEL_PRIMITIVES_RADIX_SORT(int)
}
