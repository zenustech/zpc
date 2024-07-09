#include "zensim/container/Bht.hpp"
#include "zensim/py_interop/BhtView.hpp"

extern "C" {

#define INSTANTIATE_BHT_CAPIS(Tn, Dim, Index, B)                                                 \
  /* container */                                                                                \
  ZPC_EXPORT zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>>                               \
      *container##__##bht##_##Tn##_##Dim##_##Index##_##B(                                        \
          const zs::ZSPmrAllocator<false> *allocator, zs::size_t numExpectedEntries) {           \
    return new zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>>{*allocator,                 \
                                                                     numExpectedEntries};        \
  }                                                                                              \
  ZPC_EXPORT zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>>                                \
      *container##__##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual(                            \
          const zs::ZSPmrAllocator<true> *allocator, zs::size_t numExpectedEntries) {            \
    return new zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>>{*allocator,                  \
                                                                    numExpectedEntries};         \
  }                                                                                              \
  ZPC_EXPORT void del_container##__##bht##_##Tn##_##Dim##_##Index##_##B(                         \
      zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v) {                                \
    delete v;                                                                                    \
  }                                                                                              \
  ZPC_EXPORT void del_container##__##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual(             \
      zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>> * v) {                                \
    delete v;                                                                                    \
  }                                                                                              \
  ZPC_EXPORT void relocate_container##__##bht##_##Tn##_##Dim##_##Index##_##B(                    \
      zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v, zs::memsrc_e mre,                \
      zs::ProcID devid) {                                                                        \
    *v = v->clone({mre, devid});                                                                 \
  }                                                                                              \
  ZPC_EXPORT void relocate_container##__##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual(        \
      zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>> * v, zs::memsrc_e mre,                \
      zs::ProcID devid) {                                                                        \
    *v = v->clone({mre, devid});                                                                 \
  }                                                                                              \
  ZPC_EXPORT void reset_container##__##bht##_##Tn##_##Dim##_##Index##_##B(                       \
      zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v, int clearCnt) {                  \
    v->reset(clearCnt);                                                                          \
  }                                                                                              \
  ZPC_EXPORT void reset_container##__##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual(           \
      zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>> * v, int clearCnt) {                  \
    v->reset(clearCnt);                                                                          \
  }                                                                                              \
  ZPC_EXPORT size_t container_size##__##bht##_##Tn##_##Dim##_##Index##_##B(                      \
      const zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v) {                          \
    return v->size();                                                                            \
  }                                                                                              \
  ZPC_EXPORT size_t container_size##__##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual(          \
      const zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>> *v) {                           \
    return v->size();                                                                            \
  }                                                                                              \
  ZPC_EXPORT size_t container_capacity##__##bht##_##Tn##_##Dim##_##Index##_##B(                  \
      const zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v) {                          \
    return v->_tableSize;                                                                        \
  }                                                                                              \
  ZPC_EXPORT size_t container_capacity##__##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual(      \
      const zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>> *v) {                           \
    return v->_tableSize;                                                                        \
  }                                                                                              \
  /* custom */                                                                                   \
  /* pyview */                                                                                   \
  ZPC_EXPORT zs::BhtViewLite<Tn, Dim, Index, B> *pyview##__##bht##_##Tn##_##Dim##_##Index##_##B( \
      zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v) {                                \
    return new zs::BhtViewLite<Tn, Dim, Index, B>{v->self().keys.data(),                         \
                                                  v->self().indices.data(),                      \
                                                  v->self().status.data(),                       \
                                                  v->_activeKeys.data(),                         \
                                                  v->_cnt.data(),                                \
                                                  v->_buildSuccess.data(),                       \
                                                  v->_tableSize,                                 \
                                                  v->_hf0,                                       \
                                                  v->_hf1,                                       \
                                                  v->_hf2};                                      \
  }                                                                                              \
  ZPC_EXPORT zs::BhtViewLite<const Tn, Dim, Index, B>                                            \
      *pyview##__##bht##_##const##_##Tn##_##Dim##_##Index##_##B(                                 \
          const zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v) {                      \
    return new zs::BhtViewLite<const Tn, Dim, Index, B>{v->self().keys.data(),                   \
                                                        v->self().indices.data(),                \
                                                        v->self().status.data(),                 \
                                                        v->_activeKeys.data(),                   \
                                                        v->_cnt.data(),                          \
                                                        v->_buildSuccess.data(),                 \
                                                        v->_tableSize,                           \
                                                        v->_hf0,                                 \
                                                        v->_hf1,                                 \
                                                        v->_hf2};                                \
  }                                                                                              \
  ZPC_EXPORT zs::BhtViewLite<Tn, Dim, Index, B>                                                  \
      *pyview##__##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual(                               \
          zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>> * v) {                            \
    return new zs::BhtViewLite<Tn, Dim, Index, B>{v->self().keys.data(),                         \
                                                  v->self().indices.data(),                      \
                                                  v->self().status.data(),                       \
                                                  v->_activeKeys.data(),                         \
                                                  v->_cnt.data(),                                \
                                                  v->_buildSuccess.data(),                       \
                                                  v->_tableSize,                                 \
                                                  v->_hf0,                                       \
                                                  v->_hf1,                                       \
                                                  v->_hf2};                                      \
  }                                                                                              \
  ZPC_EXPORT zs::BhtViewLite<const Tn, Dim, Index, B>                                            \
      *pyview##__##bht##_##const##_##Tn##_##Dim##_##Index##_##B##_##virtual(                     \
          const zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>> *v) {                       \
    return new zs::BhtViewLite<const Tn, Dim, Index, B>{v->self().keys.data(),                   \
                                                        v->self().indices.data(),                \
                                                        v->self().status.data(),                 \
                                                        v->_activeKeys.data(),                   \
                                                        v->_cnt.data(),                          \
                                                        v->_buildSuccess.data(),                 \
                                                        v->_tableSize,                           \
                                                        v->_hf0,                                 \
                                                        v->_hf1,                                 \
                                                        v->_hf2};                                \
  }

INSTANTIATE_BHT_CAPIS(int, 1, int, 32)
INSTANTIATE_BHT_CAPIS(int, 1, int, 16)
INSTANTIATE_BHT_CAPIS(int, 2, int, 32)
INSTANTIATE_BHT_CAPIS(int, 2, int, 16)
INSTANTIATE_BHT_CAPIS(int, 3, int, 32)
INSTANTIATE_BHT_CAPIS(int, 3, int, 16)
INSTANTIATE_BHT_CAPIS(int, 4, int, 32)
INSTANTIATE_BHT_CAPIS(int, 4, int, 16)
}
