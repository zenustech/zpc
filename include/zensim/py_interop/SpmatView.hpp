#pragma once
#include "zensim/VectorView.hpp"

namespace zs {

  template <typename T_, bool RowMajor, typename Ti_, typename Tn_>
  struct SpmatViewLite {  // T may be const
    static constexpr bool is_row_major = RowMajor;
    using value_type = remove_const_t<T_>;
    using index_type = Ti_;
    using size_type = zs::make_unsigned_t<Tn_>;
    using difference_type = zs::make_signed_t<size_type>;

    static constexpr bool is_const_structure = is_const<value_type>::value;

    ///
    template <typename T> using decorate_t
        = conditional_t<is_const_structure, zs::add_const_t<T>, T>;

    ///

    SpmatViewLite() noexcept = default;
    SpmatViewLite(index_type nrows, index_type ncols, decorate_t<size_type>* const ptrs,
                  decorate_t<index_type>* const inds, decorate_t<value_type>* const vals) noexcept
        : _rows{nrows}, _ncols{ncols}, _ptrs{ptrs}, _inds{inds}, _vals{vals} {}

    index_type _nrows{}, _ncols{};
    VectorViewLite<decorate_t<size_type>> _ptrs{};
    VectorViewLite<decorate_t<index_type>> _inds{};
    VectorViewLite<decorate_t<value_type>> _vals{};
  };

}  // namespace zs