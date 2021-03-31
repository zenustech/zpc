#pragma once
#include <zensim/resource/Resource.h>

#include <zensim/container/Vector.hpp>

namespace zs {

  template <typename Property = void, typename ValueType = float, typename IndexType = int>
  struct YaleMatrixStorage : MemoryHandle {
    using Index = IndexType;
    using Value = ValueType;
    Vector<Index> offsets, indices;
    Vector<Value> vals;
  };

  template <typename Storage> struct Matrix : Storage { ; };

}  // namespace zs