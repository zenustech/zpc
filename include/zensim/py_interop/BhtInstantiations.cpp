#include "zensim/container/Bht.hpp"
#include "zensim/py_interop/BhtView.hpp"

extern "C" {

#define INSTANTIATE_BHT_CAPIS(Tn, Dim, Index, B)                                          \
  /* container */                                                                         \
  zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>>                                   \
      *container##__##bht##_##Tn##_##Dim##_##Index##_##B(                                 \
          const zs::ZSPmrAllocator<false> *allocator, zs::size_t numExpectedEntries) {    \
    return new zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>>{*allocator,          \
                                                                     numExpectedEntries}; \
  }                                                                                       \
  zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>>                                    \
      *container##__##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual(                     \
          const zs::ZSPmrAllocator<true> *allocator, zs::size_t numExpectedEntries) {     \
    return new zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>>{*allocator,           \
                                                                    numExpectedEntries};  \
  }                                                                                       \
  zs::BhtViewLite<Tn, Dim, Index, B> *pyview##__##bht##_##Tn##_##Dim##_##Index##_##B(     \
      zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v) {                         \
    return new zs::BhtViewLite<Tn, Dim, Index, B>{v->self().keys.data(),                  \
                                                  v->self().indices.data(),               \
                                                  v->self().status.data(),                \
                                                  v->_activeKeys.data(),                  \
                                                  v->_cnt.data(),                         \
                                                  v->_buildSuccess.data(),                \
                                                  v->_tableSize,                          \
                                                  v->_hf0,                                \
                                                  v->_hf1,                                \
                                                  v->_hf2};                               \
  }                                                                                       \
  zs::BhtViewLite<const Tn, Dim, Index, B>                                                \
      *pyview##__##bht##_##const##_##Tn##_##Dim##_##Index##_##B(                          \
          const zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v) {               \
    return new zs::BhtViewLite<const Tn, Dim, Index, B>{v->self().keys.data(),            \
                                                        v->self().indices.data(),         \
                                                        v->self().status.data(),          \
                                                        v->_activeKeys.data(),            \
                                                        v->_cnt.data(),                   \
                                                        v->_buildSuccess.data(),          \
                                                        v->_tableSize,                    \
                                                        v->_hf0,                          \
                                                        v->_hf1,                          \
                                                        v->_hf2};                         \
  }

INSTANTIATE_BHT_CAPIS(int, 2, int, 32)
INSTANTIATE_BHT_CAPIS(int, 2, int, 16)
INSTANTIATE_BHT_CAPIS(int, 3, int, 32)
INSTANTIATE_BHT_CAPIS(int, 3, int, 16)
}