#include "DenseField.hpp"

namespace zs {

  ZPC_INSTANTIATE_STRUCT DenseField<u8, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT DenseField<u32, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT DenseField<u64, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT DenseField<i8, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT DenseField<i32, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT DenseField<i64, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT DenseField<f32, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT DenseField<f64, ZSPmrAllocator<>>;

  ZPC_INSTANTIATE_STRUCT DenseField<u8, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT DenseField<u32, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT DenseField<u64, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT DenseField<i8, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT DenseField<i32, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT DenseField<i64, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT DenseField<f32, ZSPmrAllocator<true>>;
  ZPC_INSTANTIATE_STRUCT DenseField<f64, ZSPmrAllocator<true>>;

}  // namespace zs