#include "Matrix.hpp"

#include <stdexcept>

#include "zensim/Logger.hpp"

namespace zs {

  template <typename ValueType, typename IndexType>
  ValueType& YaleSparseMatrix<ValueType, IndexType>::do_coeff(IndexType r, IndexType c) {
    index_type i = c;
    index_type j = r;
    if (base_t::isRowMajor()) {
      i = r;
      j = c;
    }
    // if (offsets.size() < i)
    //     throw std::runtime_error("");
    for (index_type st = offsets[i], ed = offsets[i + 1]; st < ed; ++st)
      if (indices[st] == j) return vals[st];
    // throw std::runtime_error("");
  }

  template <typename ValueType, typename IndexType>
  const ValueType& YaleSparseMatrix<ValueType, IndexType>::do_coeff(IndexType r,
                                                                    IndexType c) const {
    index_type i = c;
    index_type j = r;
    if (base_t::isRowMajor()) {
      i = r;
      j = c;
    }
    // if (offsets.size() < i)
    //     throw std::runtime_error("");
    for (index_type st = offsets[i], ed = offsets[i + 1]; st < ed; ++st)
      if (indices[st] == j) return vals[st];
    // throw std::runtime_error("");
  }

  template struct YaleSparseMatrix<f32, i32>;
  template struct YaleSparseMatrix<f32, i64>;
  template struct YaleSparseMatrix<f32, u32>;
  template struct YaleSparseMatrix<f32, u64>;
  template struct YaleSparseMatrix<f64, i32>;
  template struct YaleSparseMatrix<f64, i64>;
  template struct YaleSparseMatrix<f64, u32>;
  template struct YaleSparseMatrix<f64, u64>;
}  // namespace zs