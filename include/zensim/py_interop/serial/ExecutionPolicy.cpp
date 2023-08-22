#include "zensim/execution/ExecutionPolicy.hpp"

#include "zensim/py_interop/GenericIterator.hpp"

extern "C" {

zs::SequentialExecutionPolicy *policy__serial() { return new zs::SequentialExecutionPolicy; }
void del_policy__serial(zs::SequentialExecutionPolicy *v) { delete v; }

#define ZS_DEFINE_PARALLEL_PRIMITIVES(T)                                                           \
  /* reduce */                                                                                     \
  void reduce_sum__seq##_##T##_1(                                                                  \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                      \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                          \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    zs::reduce(*pol, first, last, output, (T)0, zs::plus<T>{});                                    \
  }                                                                                                \
  void reduce_prod__seq##_##T##_1(                                                                 \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                      \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                          \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    zs::reduce(*pol, first, last, output, (T)1, zs::multiplies<T>{});                              \
  }                                                                                                \
  void reduce_min__seq##_##T##_1(                                                                  \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                      \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                          \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    zs::reduce(*pol, first, last, output, zs::detail::deduce_numeric_max<T>(), zs::getmin<T>{});   \
  }                                                                                                \
  void reduce_max__seq##_##T##_1(                                                                  \
      zs::SequentialExecutionPolicy *p, aosoa_iterator_const_##T##_1 first,                        \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                          \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    zs::reduce(*p, first, last, output, zs::detail::deduce_numeric_lowest<T>(), zs::getmax<T>{});  \
  }                                                                                                \
  /* exclusive scan */                                                                             \
  void exclusive_scan_sum__seq##_##T##_1(                                                          \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                      \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                          \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    zs::exclusive_scan(*pol, first, last, output, (T)0, zs::plus<T>{});                            \
  }                                                                                                \
  void exclusive_scan_prod__seq##_##T##_1(                                                         \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                      \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                          \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    zs::exclusive_scan(*pol, first, last, output, (T)1, zs::multiplies<T>{});                      \
  }                                                                                                \
  /* inclusive scan */                                                                             \
  void inclusive_scan_sum__seq##_##T##_1(                                                          \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                      \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                          \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    zs::inclusive_scan(*pol, first, last, output, zs::plus<T>{});                                  \
  }                                                                                                \
  void inclusive_scan_prod__seq##_##T##_1(                                                         \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_const_##T##_1 first,                      \
      aosoa_iterator_const_##T##_1 last, aosoa_iterator_##T##_1 output) {                          \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    zs::inclusive_scan(*pol, first, last, output, zs::multiplies<T>{});                            \
  }                                                                                                \
  /* merge sort */                                                                                 \
  void merge_sort__seq##_##T##_1(zs::SequentialExecutionPolicy *pol, aosoa_iterator_##T##_1 first, \
                                 aosoa_iterator_##T##_1 last) {                                    \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    zs::merge_sort(*pol, first, last);                                                             \
  }                                                                                                \
  /* radix sort */                                                                                 \
  void radix_sort__seq##_##T##_1(zs::SequentialExecutionPolicy *pol, aosoa_iterator_##T##_1 first, \
                                 aosoa_iterator_##T##_1 last, aosoa_iterator_##T##_1 output) {     \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    if constexpr (zs::is_integral_v<T>)                                                            \
      zs::radix_sort(*pol, first, last, output, 0, sizeof(T) * 8);                                 \
  }                                                                                                \
  void radix_sort_pair__seq##_##T##_1(                                                             \
      zs::SequentialExecutionPolicy *pol, aosoa_iterator_##T##_1 keysIn,                           \
      aosoa_iterator_##int##_1 valsIn, aosoa_iterator_##T##_1 keysOut,                             \
      aosoa_iterator_##int##_1 valsOut, size_t count) {                                            \
    static_assert(zs::is_arithmetic_v<T>,                                                          \
                  "parallel primitives only available for arithmetic types");                      \
    if constexpr (zs::is_integral_v<T>)                                                            \
      zs::radix_sort_pair(*pol, keysIn, valsIn, keysOut, valsOut, count, 0, sizeof(T) * 8);        \
  }

ZS_DEFINE_PARALLEL_PRIMITIVES(int)
ZS_DEFINE_PARALLEL_PRIMITIVES(float)
ZS_DEFINE_PARALLEL_PRIMITIVES(double)
}