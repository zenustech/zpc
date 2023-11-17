#pragma once
#include "zensim/container/Vector.hpp"
#include "zensim/memory/MemoryResource.h"
#include "zensim/resource/Resource.h"

namespace zs {

  enum struct matrix_order_e : char { rowMajor = 0, colMajor };

  /// matrix base
  template <typename ValueType = float, typename IndexType = int> struct MatrixBase {
    using value_type = ValueType;
    using index_type = IndexType;
    using size_type = zs::make_unsigned_t<index_type>;
    using difference_type = zs::make_signed_t<size_type>;

    constexpr index_type rows() const noexcept { return nrows; }
    constexpr index_type cols() const noexcept { return ncols; }
    constexpr index_type size() const noexcept { return rows() * cols(); }
    constexpr bool isVector() const noexcept { return rows() == 1 || cols() == 1; }
    constexpr bool isRowMajor() const noexcept { return order == matrix_order_e::rowMajor; }

    constexpr index_type outerSize() const noexcept {
      return isVector() ? 1 : (isRowMajor() ? rows() : cols());
    }
    constexpr index_type innerSize() const noexcept {
      return isVector() ? size() : (isRowMajor() ? cols() : rows());
    }

    index_type nrows{1}, ncols{1};
    matrix_order_e order{matrix_order_e::rowMajor};
  };

  /// matrix access
  template <typename Derived> struct MatrixAccessor {
    // using value_type = typename Derived::value_type;
    // using index_type = typename Derived::index_type;
    template <typename Ti> constexpr decltype(auto) coeff(Ti r, Ti c) {
      return static_cast<Derived *>(this)->do_coeff(r, c);
    }
    template <typename Ti> constexpr decltype(auto) coeff(Ti r, Ti c) const {
      return static_cast<const Derived *>(this)->do_coeff(r, c);
    }
  };

  /// matrix

  /// matrix
  template <typename ValueType = float, typename IndexType = int> struct IdentityMatrix
      : MatrixBase<ValueType, IndexType>,
        MatrixAccessor<IdentityMatrix<ValueType, IndexType>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;
    using size_type = typename base_t::size_type;
    using difference_type = typename base_t::difference_type;

    constexpr value_type do_coeff(size_type r, size_type c) const { return r == c ? identity : 0; }

    value_type identity{1};
  };
  template <typename ValueType = float, typename IndexType = int> struct DiagonalMatrix
      : MatrixBase<ValueType, IndexType>,
        MatrixAccessor<DiagonalMatrix<ValueType, IndexType>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;
    using size_type = typename base_t::size_type;
    using difference_type = typename base_t::difference_type;

    constexpr value_type do_coeff(size_type r, size_type c) const {
      return r == c ? diagEntries[r] : 0;
    }
    Vector<value_type> diagEntries{};
  };
  template <typename ValueType = float, typename IndexType = int,
            typename AllocatorT = ZSPmrAllocator<>>
  struct YaleSparseMatrix : MatrixBase<ValueType, IndexType>,
                            MatrixAccessor<YaleSparseMatrix<ValueType, IndexType, AllocatorT>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using allocator_type = AllocatorT;
    using value_type = ValueType;
    using index_type = IndexType;
    using size_type = typename base_t::size_type;
    using difference_type = typename base_t::difference_type;

    static_assert(is_zs_allocator<allocator_type>::value,
                  "YaleSparseMatrix only works with zspmrallocator for now.");

    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (size_t)1 << (size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

    YaleSparseMatrix(const allocator_type &allocator, size_type nrows = 1, size_type ncols = 1,
                     matrix_order_e order = matrix_order_e::rowMajor)
        : base_t{(index_type)nrows, (index_type)ncols, order},
          _allocator{allocator},
          offsets{_allocator, nrows + 1},
          indices{_allocator, 0},
          vals{_allocator, 0} {}
    YaleSparseMatrix(size_type nrows = 1, size_type ncols = 1,
                     matrix_order_e order = matrix_order_e::rowMajor, memsrc_e mre = memsrc_e::host,
                     ProcID devid = -1)
        : YaleSparseMatrix{get_default_allocator(mre, devid), nrows, ncols, order} {}

    constexpr auto nnz() const noexcept { return vals.size(); }
    constexpr value_type do_coeff(size_type r, size_type c) const {
      size_type i = c;
      size_type j = r;
      if (base_t::isRowMajor()) {
        i = r;
        j = c;
      }
      if ((size_type)offsets.size() <= i) return 0;
      for (index_type st = offsets[i], ed = offsets[i + 1]; st != ed; ++st)
        if (indices[st] == j) return vals[st];
      return 0;
    }

    allocator_type _allocator{};
    Vector<index_type> offsets{}, indices{};
    Vector<value_type> vals{};
  };
  template <typename ValueType = float, typename IndexType = int,
            typename AllocatorT = ZSPmrAllocator<>>
  struct CooSparseMatrix : MatrixBase<ValueType, IndexType>,
                           MatrixAccessor<CooSparseMatrix<ValueType, IndexType, AllocatorT>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using allocator_type = AllocatorT;
    using value_type = ValueType;
    using index_type = IndexType;
    using size_type = typename base_t::size_type;
    using difference_type = typename base_t::difference_type;

    static_assert(is_zs_allocator<allocator_type>::value,
                  "CooSparseMatrix only works with zspmrallocator for now.");

    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (size_t)1 << (size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

    CooSparseMatrix(const allocator_type &allocator, size_type nrows = 1, size_type ncols = 1,
                    matrix_order_e order = matrix_order_e::rowMajor)
        : base_t{nrows, ncols, order},
          _allocator{allocator},
          rowInds{_allocator, 0},
          colInds{_allocator, 0},
          vals{_allocator, 0} {}
    CooSparseMatrix(size_type nrows = 1, size_type ncols = 1,
                    matrix_order_e order = matrix_order_e::rowMajor, memsrc_e mre = memsrc_e::host,
                    ProcID devid = -1)
        : CooSparseMatrix{get_default_allocator(mre, devid), nrows, ncols, order} {}

    constexpr auto nnz() const noexcept { return vals.size(); }

    allocator_type _allocator{};
    Vector<index_type> rowInds{}, colInds{};
    Vector<value_type> vals{};
  };
  template <typename ValueType = float, typename IndexType = int> struct DenseMatrix
      : MatrixBase<ValueType, IndexType>,
        MatrixAccessor<DenseMatrix<ValueType, IndexType>> {
    using base_t = MatrixBase<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    constexpr auto nnz() const noexcept { return base_t::size(); }
    Vector<value_type> vals{};
  };

}  // namespace zs