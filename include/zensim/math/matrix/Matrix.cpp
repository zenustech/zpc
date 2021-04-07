#include "Matrix.hpp"

#include <stdexcept>

#include "zensim/Logger.hpp"

namespace zs {

  template struct YaleSparseMatrix<f32, i32>;
  template struct YaleSparseMatrix<f32, i64>;
  template struct YaleSparseMatrix<f32, u32>;
  template struct YaleSparseMatrix<f32, u64>;
  template struct YaleSparseMatrix<f64, i32>;
  template struct YaleSparseMatrix<f64, i64>;
  template struct YaleSparseMatrix<f64, u32>;
  template struct YaleSparseMatrix<f64, u64>;
}  // namespace zs