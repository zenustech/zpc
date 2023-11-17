#include "Vector.hpp"

namespace zs {

  ZPC_INSTANTIATE_STRUCT Vector<u8, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT Vector<u32, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT Vector<u64, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT Vector<i8, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT Vector<i32, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT Vector<i64, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT Vector<f32, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT Vector<f64, ZSPmrAllocator<>>;

  ZPC_INSTANTIATE_STRUCT Vector<u8, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT Vector<u32, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT Vector<u64, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT Vector<i8, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT Vector<i32, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT Vector<i64, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT Vector<f32, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT Vector<f64, ZSPmrAllocator<true>>;

}  // namespace zs