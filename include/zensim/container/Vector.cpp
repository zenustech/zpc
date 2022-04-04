#include "Vector.hpp"

namespace zs {

  template struct Vector<u8, ZSPmrAllocator<>>;
  template struct Vector<u32, ZSPmrAllocator<>>;
  template struct Vector<u64, ZSPmrAllocator<>>;
  template struct Vector<i8, ZSPmrAllocator<>>;
  template struct Vector<i32, ZSPmrAllocator<>>;
  template struct Vector<i64, ZSPmrAllocator<>>;
  template struct Vector<f32, ZSPmrAllocator<>>;
  template struct Vector<f64, ZSPmrAllocator<>>;

  template struct Vector<u8, ZSPmrAllocator<true>>;
  template struct Vector<u32, ZSPmrAllocator<true>>;
  template struct Vector<u64, ZSPmrAllocator<true>>;
  template struct Vector<i8, ZSPmrAllocator<true>>;
  template struct Vector<i32, ZSPmrAllocator<true>>;
  template struct Vector<i64, ZSPmrAllocator<true>>;
  template struct Vector<f32, ZSPmrAllocator<true>>;
  template struct Vector<f64, ZSPmrAllocator<true>>;

}  // namespace zs