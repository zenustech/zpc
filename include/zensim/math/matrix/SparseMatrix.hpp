#pragma once
#include "zensim/container/Bcht.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"

namespace zs {

  template <typename T = float, bool RowMajor = true, typename Ti = int, typename Tn = int,
            typename AllocatorT = ZSPmrAllocator<>>
  struct SparseMatrix {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "SparseMatrix only works with zspmrallocator for now.");
    static_assert(std::is_fundamental_v<T> || is_vec<T>::value,
                  "only fundamental types and vec are allowed to be used as value_type.");
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
    constexpr size_type outerSize() const noexcept {
      if constexpr (is_row_major)
        return rows();
      else
        return cols();
    }
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
    template <bool is_const = false> struct iterator_impl
        : IteratorInterface<iterator_impl<is_const>> {
      template <typename TT> using decorated_t = conditional_t<is_const, std::add_const_t<TT>, TT>;
      constexpr iterator_impl(index_type line, decorated_t<index_type> *ptrs,
                              decorated_t<index_type> *inds, decorated_t<value_type> *vals)
          : _idx{ptrs[line]}, _inds{inds}, _vals{vals} {}

      constexpr tuple<index_type &, value_type &> dereference() {
        return tie(_inds[_idx], _vals[_idx]);
      }
      constexpr bool equal_to(iterator_impl it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

      constexpr index_type index() const { return _inds[_idx]; }
      constexpr value_type value() const { return _vals[_idx]; }
      constexpr decorated_t<value_type> &value() { return _vals[_idx]; }

    protected:
      size_type _idx{0};
      decorated_t<index_type> *_inds;
      decorated_t<value_type> *_vals;
    };
    using iterator = LegacyIterator<iterator_impl<false>>;
    using const_iterator = LegacyIterator<iterator_impl<true>>;

    constexpr auto begin(index_type no) noexcept {
      return make_iterator<iterator_impl<true>>(no, _ptrs.data(), _inds.data(), _vals.data());
    }
    constexpr auto end(index_type no) noexcept {
      return make_iterator<iterator_impl<true>>(no + 1, _ptrs.data(), _inds.data(), _vals.data());
    }

    constexpr auto begin(index_type no) const noexcept {
      return make_iterator<iterator_impl<true>>(no, _ptrs.data(), _inds.data(), _vals.data());
    }
    constexpr auto end(index_type no) const noexcept {
      return make_iterator<iterator_impl<true>>(no + 1, _ptrs.data(), _inds.data(), _vals.data());
    }
    void print() {
      size_type offset = 0;
      for (index_type i = 0; i != outerSize(); ++i) {
        auto ed = _ptrs[i + 1];
        fmt::print("#\tline [{}] ({} entries):\t", i, ed - offset);
        for (; offset != ed; ++offset) fmt::print("{}\t", _inds[offset]);
        fmt::print("\n");
      }
    }

    template <typename Policy, typename IRange, typename JRange, typename VRange>
    void build(Policy &&policy, index_type nrows, index_type ncols, IRange &&is, JRange &&js,
               VRange &&vs);
    template <typename Policy, typename IRange, typename JRange, bool Mirror = false>
    void build(Policy &&policy, index_type nrows, index_type ncols, IRange &&is, JRange &&js,
               wrapv<Mirror> = {});

    index_type _nrows = 0, _ncols = 0;  // for square matrix, nrows = ncols
    zs::Vector<size_type, allocator_type> _ptrs{};
    zs::Vector<index_type, allocator_type> _inds{};
    zs::Vector<value_type, allocator_type> _vals{};  // maybe empty, e.g. bidirectional graph
  };

  /// @brief conventional csr sparse matrix build
  template <typename T, bool RowMajor, typename Ti, typename Tn, typename AllocatorT>
  template <typename Policy, typename IRange, typename JRange, typename VRange>
  void SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT>::build(Policy &&policy, Ti nrows, Ti ncols,
                                                            IRange &&is, JRange &&js, VRange &&vs) {
    using Tr = RM_CVREF_T(*std::begin(is));
    using Tc = RM_CVREF_T(*std::begin(js));
    using Tv = RM_CVREF_T(*std::begin(vs));
    static_assert(
        std::is_convertible_v<Tr,
                              Ti> && std::is_convertible_v<Tr, Ti> && std::is_convertible_v<Tv, T>,
        "input triplet types are not convertible to types of this sparse matrix.");

    auto size = range_size(is);
    if (size != range_size(js) || size != range_size(vs))
      throw std::runtime_error(fmt::format("is size: {}, while js size ({}), vs size ({})\n", size,
                                           range_size(js), range_size(vs)));

    /// @brief initial hashing
    _nrows = nrows;
    _ncols = ncols;
    Ti nsegs = is_row_major ? nrows : ncols;
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    constexpr auto execTag = wrapv<space>{};
    using ICoord = zs::vec<Ti, 2>;

    zs::bcht<ICoord, index_type, true, zs::universal_hash<ICoord>, 16> tab{get_allocator(),
                                                                           (std::size_t)size};
    Vector<size_type> cnts{get_allocator(), (std::size_t)(nsegs + 1)};
    Vector<index_type> localOffsets{get_allocator(), (std::size_t)size};
    cnts.reset(0);
    policy(range(size), [tab = proxy<space>(tab), cnts = view<space>(cnts),
                         localOffsets = view<space>(localOffsets), is = std::begin(is),
                         js = std::begin(js), execTag] ZS_LAMBDA(size_type k) mutable {
      using tab_t = RM_CVREF_T(tab);
      Ti i = is[k], j = js[k];
      // insertion success
      if (auto id = tab.insert(ICoord{i, j}); id != tab_t::sentinel_v) {
        if constexpr (RowMajor)
          localOffsets[id] = atomic_add(execTag, &cnts[i], (size_type)1);
        else
          localOffsets[id] = atomic_add(execTag, &cnts[j], (size_type)1);
      }
    });

    /// @brief _ptrs
    _ptrs.resize(nsegs + 1);
    exclusive_scan(policy, std::begin(cnts), std::end(cnts), std::begin(_ptrs));

    auto numEntries = _ptrs.getVal(nsegs);
    fmt::print("{} entries activated in total from {} triplets.\n", numEntries, size);

    /// @brief _inds, _vals
    static_assert(std::is_fundamental_v<value_type> || is_vec<value_type>::value,
                  "value_type not supported");
    _inds.resize(numEntries);
    _vals.resize(numEntries);
    if constexpr (std::is_fundamental_v<value_type>)
      _vals.reset(0);
    else if constexpr (is_vec<value_type>::value) {
      _vals.reset(0);
#if 0
      policy(range(numEntries), [vals = view<space>(_vals)] ZS_LAMBDA(size_type k) mutable {
        vals[k] = value_type::zeros();
      });
#endif
    }
    policy(range(size), [tab = proxy<space>(tab), localOffsets = view<space>(localOffsets),
                         is = std::begin(is), js = std::begin(js), vs = std::begin(vs),
                         ptrs = view<space>(_ptrs), inds = view<space>(_inds),
                         vals = view<space>(_vals), execTag] ZS_LAMBDA(size_type k) mutable {
      using tab_t = RM_CVREF_T(tab);
      Ti i = is[k], j = js[k];
      auto id = tab.query(ICoord{i, j});
      auto loc = localOffsets[id];
      size_type offset = 0;
      if constexpr (RowMajor) {
        offset = ptrs[i] + loc;
        inds[offset] = j;
      } else {
        offset = ptrs[j] + loc;
        inds[offset] = i;
      }
      if constexpr (std::is_fundamental_v<value_type>)
        atomic_add(execTag, &vals[offset], (value_type)vs[k]);
      else if constexpr (is_vec<value_type>::value) {
        auto &val = vals[offset];
        const auto &e = vs[k];
        for (typename value_type::index_type i = 0; i != value_type::extent; ++i)
          atomic_add(execTag, &val.val(i), (typename value_type::value_type)e.val(i));
      }
    });
  }

  /// @brief topology only csr sparse matrix build
  template <typename T, bool RowMajor, typename Ti, typename Tn, typename AllocatorT>
  template <typename Policy, typename IRange, typename JRange, bool Mirror>
  void SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT>::build(Policy &&policy, Ti nrows, Ti ncols,
                                                            IRange &&is, JRange &&js,
                                                            wrapv<Mirror>) {
    using Tr = RM_CVREF_T(*std::begin(is));
    using Tc = RM_CVREF_T(*std::begin(js));
    static_assert(std::is_convertible_v<Tr, Ti> && std::is_convertible_v<Tr, Ti>,
                  "input doublet types are not convertible to types of this sparse matrix.");

    auto size = range_size(is);
    if (size != range_size(js))
      throw std::runtime_error(
          fmt::format("is size: {}, while js size ({})\n", size, range_size(js)));

    /// @brief initial hashing
    _nrows = nrows;
    _ncols = ncols;
    Ti nsegs = is_row_major ? nrows : ncols;
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    using ICoord = zs::vec<Ti, 2>;

    zs::bcht<ICoord, index_type, true, zs::universal_hash<ICoord>, 16> tab{
        get_allocator(), Mirror ? (std::size_t)size * 2 : (std::size_t)size};
    Vector<index_type> localOffsets{get_allocator(),
                                    Mirror ? (std::size_t)size * 2 : (std::size_t)size};
    Vector<size_type> cnts{get_allocator(), (std::size_t)(nsegs + 1)};
    cnts.reset(0);
    policy(range(size),
           [tab = proxy<space>(tab), cnts = view<space>(cnts),
            localOffsets = view<space>(localOffsets), is = std::begin(is), js = std::begin(js),
            execTag = wrapv<space>{}] ZS_LAMBDA(size_type k) mutable {
             using tab_t = RM_CVREF_T(tab);
             Ti i = is[k], j = js[k];
             // insertion success
             if (auto id = tab.insert(ICoord{i, j}); id != tab_t::sentinel_v) {
               if constexpr (RowMajor)
                 localOffsets[id] = atomic_add(execTag, &cnts[i], (size_type)1);
               else
                 localOffsets[id] = atomic_add(execTag, &cnts[j], (size_type)1);

               /// @note spawn symmetric entries
               if constexpr (Mirror) {
                 if (i != j) {
                   if (id = tab.insert(ICoord{j, i}); id != tab_t::sentinel_v) {
                     if constexpr (RowMajor)
                       localOffsets[id] = atomic_add(execTag, &cnts[j], (size_type)1);
                     else
                       localOffsets[id] = atomic_add(execTag, &cnts[i], (size_type)1);
                   }
                 }
               }
             }
           });

    /// @brief _ptrs
    _ptrs.resize(nsegs + 1);
    exclusive_scan(policy, std::begin(cnts), std::end(cnts), std::begin(_ptrs));

    auto numEntries = _ptrs.getVal(nsegs);
    fmt::print("{} entries activated in total from {} doublets.\n", numEntries, size);

    if (auto ntab = tab.size(); numEntries != ntab) {
      throw std::runtime_error(fmt::format(
          "computed number of entries {} not equal to the number of active table entries {}\n",
          numEntries, ntab));
    }

    /// @brief _inds
    _inds.resize(numEntries);
    policy(range(numEntries),
           [tab = proxy<space>(tab), localOffsets = view<space>(localOffsets),
            ptrs = view<space>(_ptrs), inds = view<space>(_inds)] ZS_LAMBDA(size_type k) mutable {
             auto ij = tab._activeKeys[k];
             auto loc = localOffsets[k];
             if constexpr (RowMajor)
               inds[ptrs[ij[0]] + loc] = ij[1];
             else
               inds[ptrs[ij[1]] + loc] = ij[0];
           });
  }

  template <execspace_e Space, typename SpMatT, bool Base = false, typename = void>
  struct SparseMatrixView {
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
          _ptrs{view<space>(spmat._ptrs, wrapv<Base>{})},
          _inds{view<space>(spmat._inds, wrapv<Base>{})},
          _vals{view<space>(spmat._vals, wrapv<Base>{})} {}

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
    constexpr size_type outerSize() const noexcept {
      if constexpr (is_row_major)
        return rows();
      else
        return cols();
    }
    constexpr size_type nnz() const noexcept {
      if constexpr (is_row_major)
        return _ptrs[_nrows];
      else
        return _ptrs[_ncols];
    }

    index_type _nrows, _ncols;
    zs::VectorView<space, decorate_t<Vector<size_type, allocator_type>>, Base> _ptrs;
    zs::VectorView<space, decorate_t<Vector<index_type, allocator_type>>, Base> _inds;
    zs::VectorView<space, decorate_t<Vector<value_type, allocator_type>>, Base> _vals;
  };

  template <execspace_e ExecSpace, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT, bool Base = true>
  constexpr decltype(auto) view(SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                wrapv<Base> = {}) {
    return SparseMatrixView<ExecSpace, SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT>>{spmat};
  }
  template <execspace_e ExecSpace, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT, bool Base = true>
  constexpr decltype(auto) view(const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                wrapv<Base> = {}) {
    return SparseMatrixView<ExecSpace, const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT>>{spmat};
  }

  template <execspace_e ExecSpace, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT, bool Base = true>
  constexpr decltype(auto) view(SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat, wrapv<Base>,
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
            typename AllocatorT, bool Base = true>
  constexpr decltype(auto) view(const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                wrapv<Base>, const SmallString &tagName) {
    auto ret
        = SparseMatrixView<ExecSpace, const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT>>{spmat};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._ptrs._nameTag = tagName + SmallString{":ptrs"};
    ret._inds._nameTag = tagName + SmallString{":inds"};
    ret._vals._nameTag = tagName + SmallString{":vals"};
#endif
    return ret;
  }

  template <execspace_e space, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT>
  constexpr decltype(auto) proxy(SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat) {
    return view<space>(spmat, false_c);
  }
  template <execspace_e space, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT>
  constexpr decltype(auto) proxy(const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat) {
    return view<space>(spmat, false_c);
  }

  template <execspace_e space, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT>
  constexpr decltype(auto) proxy(SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                 const SmallString &tagName) {
    return view<space>(spmat, false_c, tagName);
  }
  template <execspace_e space, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT>
  constexpr decltype(auto) proxy(const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                 const SmallString &tagName) {
    return view<space>(spmat, false_c, tagName);
  }

}  // namespace zs