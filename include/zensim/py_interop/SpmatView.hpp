#pragma once
#include "zensim/py_interop/VectorView.hpp"

namespace zs {

  template <typename T_, bool RowMajor, typename Ti_, typename Tn_>
  struct SpmatViewLite {  // T may be const
    static constexpr bool is_row_major = RowMajor;
    using value_type = remove_const_t<T_>;
    using index_type = Ti_;
    using size_type = zs::make_unsigned_t<Tn_>;
    using difference_type = zs::make_signed_t<size_type>;

    static constexpr bool is_const_structure = is_const<T_>::value;

    ///
    template <typename T> using decorate_t
        = conditional_t<is_const_structure, zs::add_const_t<T>, T>;

    ///

    constexpr SpmatViewLite() noexcept = default;
    ~SpmatViewLite() = default;
    SpmatViewLite(index_type nrows, index_type ncols, decorate_t<size_type>* const ptrs,
                  decorate_t<index_type>* const inds, decorate_t<value_type>* const vals) noexcept
        : _nrows{nrows}, _ncols{ncols}, _ptrs{ptrs}, _inds{inds}, _vals{vals} {}

    constexpr index_type rows() const noexcept { return _nrows; }
    constexpr index_type cols() const noexcept { return _ncols; }
    constexpr zs::tuple<index_type, index_type> shape() const noexcept {
      return zs::make_tuple(rows(), cols());
    }
    constexpr size_type size() const noexcept { return rows() * cols(); }
    constexpr size_type outerSize() const noexcept {
      if constexpr (is_row_major)
        return rows();
      else
        return cols();
    }
    constexpr size_type innerSize() const noexcept {
      if constexpr (is_row_major)
        return cols();
      else
        return rows();
    }
    constexpr size_type nnz() const noexcept {
      if constexpr (is_row_major)
        return _ptrs[_nrows];
      else
        return _ptrs[_ncols];
    }
    constexpr size_type locate(index_type i, index_type j) const noexcept {
      size_type id{}, ed{};
      if constexpr (is_row_major) {
        id = _ptrs[i];
        ed = _ptrs[i + 1];
      } else {
        id = _ptrs[j];
        ed = _ptrs[j + 1];
      }
      for (; id != ed; ++id) {
        if constexpr (is_row_major) {
          if (j == _inds[id]) break;
        } else {
          if (i == _inds[id]) break;
        }
      }
      if (id != ed)
        return id;
      else {
        printf("cannot find the spmat entry at (%d, %d)\n", (int)i, (int)j);
        return detail::deduce_numeric_max<size_type>();
      }
    }
    /// @note binary search
    constexpr size_type locate(index_type i, index_type j, true_type) const noexcept {
      size_type st{}, ed{}, mid{};
      if constexpr (is_row_major) {
        st = _ptrs[i];
        ed = _ptrs[i + 1] - 1;
      } else {
        st = _ptrs[j];
        ed = _ptrs[j + 1] - 1;
      }
      while (ed >= st) {
        mid = st + (ed - st) / 2;
        if constexpr (is_row_major) {
          if (j == _inds[mid]) return mid;
          if (j < _inds[mid])
            ed = mid - 1;
          else
            st = mid + 1;
        } else {
          if (i == _inds[mid]) return mid;
          if (i < _inds[mid])
            ed = mid - 1;
          else
            st = mid + 1;
        }
      }
      return detail::deduce_numeric_max<size_type>();
    }
    constexpr bool exist(index_type i, index_type j, true_type) const noexcept {
      size_type st{}, ed{}, mid{};
      if constexpr (is_row_major) {
        st = _ptrs[i];
        ed = _ptrs[i + 1] - 1;
      } else {
        st = _ptrs[j];
        ed = _ptrs[j + 1] - 1;
      }
      while (ed >= st) {
        mid = st + (ed - st) / 2;
        if constexpr (is_row_major) {
          if (j == _inds[mid]) break;
          if (j < _inds[mid])
            ed = mid - 1;
          else
            st = mid + 1;
        } else {
          if (i == _inds[mid]) break;
          if (i < _inds[mid])
            ed = mid - 1;
          else
            st = mid + 1;
        }
      }
      return ed >= st;
    }

    index_type _nrows{}, _ncols{};
    VectorViewLite<decorate_t<size_type>> _ptrs{};
    VectorViewLite<decorate_t<index_type>> _inds{};
    VectorViewLite<decorate_t<value_type>> _vals{};
  };

}  // namespace zs