#include "zensim/math/Vec.h"
#include "zensim/py_interop/SmallVec.hpp"

extern "C" {
#define INSTANTIATE_SMALL_VEC_CAPIS_2D(T, M, N)                                     \
  ZPC_EXPORT zs::SmallVec* small_vec__##T##_##M##_##N() {                           \
    return new zs::SmallVec{zs::vec<T, M, N>{}};                                    \
  }                                                                                 \
  ZPC_EXPORT void del_small_vec__##T##_##M##_##N(zs::SmallVec* vec) { delete vec; } \
  ZPC_EXPORT T* small_vec_data_ptr__##T##_##M##_##N(zs::SmallVec* vec) {            \
    T* ptr;                                                                         \
    std::visit(                                                                     \
        [&ptr](auto& v) {                                                           \
          if constexpr (!zs::is_scalar_v<T>) {                                      \
            using vec_t = typename std::decay_t<decltype(v)>;                       \
            using val_t = typename vec_t::value_type;                               \
            if constexpr (std::is_same_v<val_t, T>) ptr = v._data;                  \
          }                                                                         \
        },                                                                          \
        *vec);                                                                      \
    return ptr;                                                                     \
  }

#define INSTANTIATE_SMALL_VEC_CAPIS_2D_WITH_ROW(T, M) \
  INSTANTIATE_SMALL_VEC_CAPIS_2D(T, M, 1)             \
  INSTANTIATE_SMALL_VEC_CAPIS_2D(T, M, 2)             \
  INSTANTIATE_SMALL_VEC_CAPIS_2D(T, M, 3)             \
  INSTANTIATE_SMALL_VEC_CAPIS_2D(T, M, 4)

#define INSTANTIATE_SMALL_VEC_CAPIS_2D_ALL(T)   \
  INSTANTIATE_SMALL_VEC_CAPIS_2D_WITH_ROW(T, 1) \
  INSTANTIATE_SMALL_VEC_CAPIS_2D_WITH_ROW(T, 2) \
  INSTANTIATE_SMALL_VEC_CAPIS_2D_WITH_ROW(T, 3) \
  INSTANTIATE_SMALL_VEC_CAPIS_2D_WITH_ROW(T, 4)

#define INSTANTIATE_SMALL_VEC_CAPIS_1D(T, N)                                                    \
  ZPC_EXPORT zs::SmallVec* small_vec__##T##_##N() { return new zs::SmallVec{zs::vec<T, N>{}}; } \
  ZPC_EXPORT void del_small_vec__##T##_##N(zs::SmallVec* vec) { delete vec; }                   \
  ZPC_EXPORT T* small_vec_data_ptr__##T##_##N(zs::SmallVec* vec) {                              \
    T* ptr;                                                                                     \
    std::visit(                                                                                 \
        [&ptr](auto& v) {                                                                       \
          if constexpr (!zs::is_scalar_v<T>) {                                                  \
            using vec_t = typename std::decay_t<decltype(v)>;                                   \
            using val_t = typename vec_t::value_type;                                           \
            if constexpr (std::is_same_v<val_t, T>) ptr = v._data;                              \
          }                                                                                     \
        },                                                                                      \
        *vec);                                                                                  \
    return ptr;                                                                                 \
  }

#define INSTANTIATE_SMALL_VEC_CAPIS_1D_ALL(T) \
  INSTANTIATE_SMALL_VEC_CAPIS_1D(T, 1)        \
  INSTANTIATE_SMALL_VEC_CAPIS_1D(T, 2)        \
  INSTANTIATE_SMALL_VEC_CAPIS_1D(T, 3)        \
  INSTANTIATE_SMALL_VEC_CAPIS_1D(T, 4)

#define INSTANTIATE_SMALL_VEC_CAPIS_SCALAR(T)                                 \
  ZPC_EXPORT zs::SmallVec* small_vec__##T() { return new zs::SmallVec{T{}}; } \
  ZPC_EXPORT void del_small_vec__##T(zs::SmallVec* vec) { delete vec; }       \
  ZPC_EXPORT void* small_vec_data_ptr__##T(zs::SmallVec* vec) {               \
    void* ptr;                                                                \
    std::visit([&ptr](auto& v) { ptr = static_cast<void*>(&v); }, *vec);      \
    return ptr;                                                               \
  }

#define INSTANTIATE_SMALL_VEC_CAPIS(T)  \
  INSTANTIATE_SMALL_VEC_CAPIS_2D_ALL(T) \
  INSTANTIATE_SMALL_VEC_CAPIS_1D_ALL(T) \
  INSTANTIATE_SMALL_VEC_CAPIS_SCALAR(T)

INSTANTIATE_SMALL_VEC_CAPIS(int)
INSTANTIATE_SMALL_VEC_CAPIS(float)
INSTANTIATE_SMALL_VEC_CAPIS(double)
}