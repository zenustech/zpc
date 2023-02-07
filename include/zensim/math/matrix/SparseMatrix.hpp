#pragma once
#include "zensim/container/Bcht.hpp"
#include "zensim/container/Vector.hpp"

namespace zs {

  template <typename T = float, bool RowMajor = true, typename Ti = int, typename Tn = int,
            typename AllocatorT = ZSPmrAllocator<>>
  struct SparseMatrix {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "SparseMatrix only works with zspmrallocator for now.");
    static_assert(std::is_default_constructible_v<T> && std::is_trivially_copyable_v<T>,
                  "element is not default-constructible or trivially-copyable!");

    static constexpr bool is_row_major = RowMajor;
    using value_type = T;
    using allocator_type = AllocatorT;
    using size_type = std::make_unsigned_t<Tn>;
    using difference_type = std::make_signed_t<size_type>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    using index_type = Ti;
    using subscript_type = zs::vec<index_type, 2>;
    using table_type
        = zs::bcht<subscript_type, sint_t, true, zs::universal_hash<subscript_type>, 16>;

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    decltype(auto) memoryLocation() const noexcept { return _ptrs.get_allocator().location; }
    ProcID devid() const noexcept { return memoryLocation().devid(); }
    memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) get_allocator() const noexcept { return _ptrs.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (std::size_t)1 << (std::size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

    /// allocator-aware
    SparseMatrix(const allocator_type &allocator, Ti ni, Ti nj)
        : _nrows{ni}, _ncols{nj}, _ptrs{allocator, 2}, _inds{allocator, 0}, _vals{allocator, 0} {
      _ptrs.reset(0);
    }
    explicit SparseMatrix(Ti ni, Ti nj, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : SparseMatrix{get_default_allocator(mre, devid), ni, nj} {}
    SparseMatrix(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : SparseMatrix{get_default_allocator(mre, devid), 0, 0} {}

    constexpr index_type rows() const noexcept { return _nrows; }
    constexpr index_type cols() const noexcept { return _ncols; }
    constexpr size_type size() const noexcept { return rows() * cols(); }
    constexpr size_type nnz() const noexcept { return _inds.size(); }

    /// @note invalidates all entries
    void resize(Ti ni, Ti nj) {
      _nrows = ni;
      _ncols = nj;
      if constexpr (is_row_major)
        _ptrs.resize(ni + 1);
      else
        _ptrs.resize(nj + 1);
      _ptrs.reset(0);
    }

    /// @brief iterators
    struct iterator_impl : IteratorInterface<iterator_impl> {
      constexpr iterator_impl(index_type line, index_type *ptrs, index_type *inds, value_type *vals)
          : _idx{ptrs[line]}, _inds{inds}, _vals{vals} {}

      constexpr tuple<index_type &, value_type &> dereference() {
        return forward_as_tuple(_inds[_idx], _vals[_idx]);
      }
      constexpr bool equal_to(iterator_impl it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

      constexpr index_type index() const { return _inds[_idx]; }
      constexpr value_type value() const { return _vals[_idx]; }
      constexpr value_type &value() { return _vals[_idx]; }

    protected:
      size_type _idx{0};
      index_type *_inds;
      value_type *_vals;
    };
    using iterator = LegacyIterator<iterator_impl>;

    struct const_iterator_impl : IteratorInterface<const_iterator_impl> {
      constexpr const_iterator_impl(index_type line, const index_type *ptrs, const index_type *inds,
                                    const value_type *vals)
          : _idx{ptrs[line]}, _inds{inds}, _vals{vals} {}

      constexpr tuple<const index_type &, const value_type &> dereference() {
        return forward_as_tuple(_inds[_idx], _vals[_idx]);
      }
      constexpr bool equal_to(const_iterator_impl it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(const_iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

      constexpr index_type index() const { return _inds[_idx]; }
      constexpr value_type value() const { return _vals[_idx]; }

    protected:
      size_type _idx{0};
      const index_type *_inds;
      const value_type *_vals;
    };
    using const_iterator = LegacyIterator<const_iterator_impl>;

    constexpr auto begin(index_type no) noexcept {
      return make_iterator<iterator_impl>(no, _ptrs.data(), _inds.data(), _vals.data());
    }
    constexpr auto end(index_type no) noexcept {
      return make_iterator<iterator_impl>(no + 1, _ptrs.data(), _inds.data(), _vals.data());
    }

    constexpr auto begin(index_type no) const noexcept {
      return make_iterator<const_iterator_impl>(no, _ptrs.data(), _inds.data(), _vals.data());
    }
    constexpr auto end(index_type no) const noexcept {
      return make_iterator<const_iterator_impl>(no + 1, _ptrs.data(), _inds.data(), _vals.data());
    }

    index_type _nrows = 0, _ncols = 0;  // for square matrix, nrows = ncols
    zs::Vector<size_type, allocator_type> _ptrs{};
    zs::Vector<index_type, allocator_type> _inds{};
    zs::Vector<value_type, allocator_type> _vals{};  // maybe empty, e.g. bidirectional graph
  };

  template <execspace_e Space, typename SpMatT, typename = void> struct SparseMatrixView {
    static constexpr bool is_const_structure = std::is_const_v<SpMatT>;
    static constexpr auto space = Space;
    template <typename T> using decorate_t
        = conditional_t<is_const_structure, std::add_const_t<T>, T>;
    using sparse_matrix_type = std::remove_const_t<SpMatT>;
    using const_sparse_matrix_type = std::add_const_t<sparse_matrix_type>;

    static constexpr auto is_row_major = sparse_matrix_type::is_row_major;

    using allocator_type = typename sparse_matrix_type::allocator_type;
    using value_type = typename sparse_matrix_type::value_type;
    using size_type = typename sparse_matrix_type::size_type;
    using index_type = typename sparse_matrix_type::index_type;
    using difference_type = typename sparse_matrix_type::difference_type;

    SparseMatrixView() noexcept = default;
    explicit constexpr SparseMatrixView(SpMatT &spmat)
        : _nrows{spmat._nrows},
          _ncols{spmat._ncols},
          _ptrs{view<space>(spmat._ptrs, true_c)},
          _inds{view<space>(spmat._inds, true_c)},
          _vals{view<space>(spmat._vals, true_c)} {}

    constexpr auto operator()(index_type i, index_type j) const {
      size_type offset{}, ed{};
      index_type target{};
      if constexpr (is_row_major) {
        offset = _ptrs[i];
        ed = _ptrs[i + 1];
        target = j;
      } else {
        offset = _ptrs[j];
        ed = _ptrs[j + 1];
        target = i;
      }
      for (; offset != ed; ++offset) {
        if (_inds[offset] == target) return _vals[offset];
      }
      return value_type{};
    }

    constexpr auto begin(index_type no) const {
      return
          typename sparse_matrix_type::const_iterator{no, _ptrs.data(), _inds.data(), _vals.data()};
    }
    constexpr auto end(index_type no) const {
      return typename sparse_matrix_type::const_iterator{no + 1, _ptrs.data(), _inds.data(),
                                                         _vals.data()};
    }
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr auto begin(index_type no) {
      return conditional_t<is_const_structure, typename sparse_matrix_type::const_iterator,
                           typename sparse_matrix_type::iterator>{no, _ptrs.data(), _inds.data(),
                                                                  _vals.data()};
    }
    template <bool V = is_const_structure, enable_if_t<!V> = 0> constexpr auto end(index_type no) {
      return conditional_t<is_const_structure, typename sparse_matrix_type::const_iterator,
                           typename sparse_matrix_type::iterator>{no + 1, _ptrs.data(),
                                                                  _inds.data(), _vals.data()};
    }

    constexpr index_type rows() const noexcept { return _nrows; }
    constexpr index_type cols() const noexcept { return _ncols; }
    constexpr size_type size() const noexcept { return rows() * cols(); }
    constexpr size_type nnz() const noexcept {
      if constexpr (is_row_major)
        return _ptrs[_nrows];
      else
        return _ptrs[_ncols];
    }

    index_type _nrows, _ncols;
    zs::VectorView<space, decorate_t<Vector<size_type, allocator_type>>, true> _ptrs;
    zs::VectorView<space, decorate_t<Vector<index_type, allocator_type>>, true> _inds;
    zs::VectorView<space, decorate_t<Vector<value_type, allocator_type>>, true> _vals;
  };

  template <execspace_e ExecSpace, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT>
  constexpr decltype(auto) proxy(SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat) {
    return SparseMatrixView<ExecSpace, SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT>>{spmat};
  }
  template <execspace_e ExecSpace, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT>
  constexpr decltype(auto) proxy(const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat) {
    return SparseMatrixView<ExecSpace, const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT>>{spmat};
  }

  template <execspace_e ExecSpace, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT>
  constexpr decltype(auto) proxy(SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                 const SmallString &tagName) {
    auto ret = SparseMatrixView<ExecSpace, SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT>>{spmat};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._ptrs._nameTag = tagName + SmallString{":ptrs"};
    ret._inds._nameTag = tagName + SmallString{":inds"};
    ret._vals._nameTag = tagName + SmallString{":vals"};
#endif
    return ret;
  }
  template <execspace_e ExecSpace, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT>
  constexpr decltype(auto) proxy(const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                 const SmallString &tagName) {
    auto ret
        = SparseMatrixView<ExecSpace, const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT>>{spmat};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._ptrs._nameTag = tagName + SmallString{":ptrs"};
    ret._inds._nameTag = tagName + SmallString{":inds"};
    ret._vals._nameTag = tagName + SmallString{":vals"};
#endif
    return ret;
  }

}  // namespace zs