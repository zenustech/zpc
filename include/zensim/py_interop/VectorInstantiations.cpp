#include "zensim/container/Vector.hpp"
#include "zensim/py_interop/VectorView.hpp"

namespace zs {

  VectorViewLite<float> pyview(Vector<float, ZSPmrAllocator<false>> &v) {
    return VectorViewLite<float>{v.data()};
  }
  VectorViewLite<const float> pyview(const Vector<float, ZSPmrAllocator<false>> &v) {
    return VectorViewLite<const float>{v.data()};
  }

}  // namespace zs
