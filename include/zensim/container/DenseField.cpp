#include "DenseField.hpp"

namespace zs {

  template struct DenseField<u8, ZSPmrAllocator<>>;
  template struct DenseField<u32, ZSPmrAllocator<>>;
  template struct DenseField<u64, ZSPmrAllocator<>>;
  template struct DenseField<i8, ZSPmrAllocator<>>;
  template struct DenseField<i32, ZSPmrAllocator<>>;
  template struct DenseField<i64, ZSPmrAllocator<>>;
  template struct DenseField<f32, ZSPmrAllocator<>>;
  template struct DenseField<f64, ZSPmrAllocator<>>;

  template struct DenseField<u8, ZSPmrAllocator<true>>;
  template struct DenseField<u32, ZSPmrAllocator<true>>;
  template struct DenseField<u64, ZSPmrAllocator<true>>;
  template struct DenseField<i8, ZSPmrAllocator<true>>;
  template struct DenseField<i32, ZSPmrAllocator<true>>;
  template struct DenseField<i64, ZSPmrAllocator<true>>;
  template struct DenseField<f32, ZSPmrAllocator<true>>;
  template struct DenseField<f64, ZSPmrAllocator<true>>;

}  // namespace zs