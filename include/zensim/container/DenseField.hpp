#pragma once
#include "zensim/container/Vector.hpp"
#include "zensim/resource/Resource.h"

namespace zs {

  template <typename T, typename AllocatorT = ZSPmrAllocator<>> struct DenseField {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "DenseField only works with zspmrallocator for now.");
    static_assert(is_same_v<T, remove_cvref_t<T>>, "T is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<T>, "element is not default-constructible!");
    // static_assert(zs::is_trivially_copyable_v<T>, "element is not trivially-copyable!");
    static_assert(zs::is_trivially_copyable_v<T> || std::is_copy_assignable_v<T>,
                  "element is not copyable!");

    using allocator_type = AllocatorT;

    using value_type = T;
    using field_type = Vector<value_type, allocator_type>;
    using size_type = typename field_type::size_type;
    using shape_type = Vector<size_type, allocator_type>;

    using difference_type = typename field_type::difference_type;
    using reference = typename field_type::reference;
    using const_reference = typename field_type::const_reference;
    using pointer = typename field_type::pointer;
    using const_pointer = typename field_type::const_pointer;

    constexpr decltype(auto) memoryLocation() const noexcept { return _field.memoryLocation(); }
    constexpr ProcID devid() const noexcept { return _field.devid(); }
    constexpr memsrc_e memspace() const noexcept { return _field.memspace(); }
    decltype(auto) get_allocator() const noexcept { return _field.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      return _field.get_default_allocator(mre, devid);
    }

    static size_type shape_size(const std::vector<size_type> &shape) noexcept {
      size_type sz = 1;
      for (size_type d : shape) sz *= d;
      return sz;
    }
    /// allocator-aware
    DenseField(const allocator_type &allocator, const std::vector<size_type> &shape)
        : _field{allocator, shape_size(shape)}, _shape{shape.size() + 1} {
      size_type base = 1;
      _shape[shape.size()] = 1;
      for (sint_t i = shape.size() - 1; i >= 0;) {
        base *= shape[i];
        _shape[i--] = base;
      }
      if (allocator.location.memspace() != memsrc_e::host)
        _shape = _shape.clone(allocator.location);
    }
    explicit DenseField(const std::vector<size_type> &shape, memsrc_e mre = memsrc_e::host,
                        ProcID devid = -1)
        : DenseField{get_default_allocator(mre, devid), shape} {}
    DenseField(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : DenseField{get_default_allocator(mre, devid), {}} {}

    ~DenseField() = default;

    inline value_type getVal(size_type i = 0) const { return _field.getVal(i); }
    inline void retrieveVals(value_type *dst) const { _field.retrieveVals(dst); }
    inline void setVal(value_type v, size_type i = 0) const { _field.setVal(v, i); }

    struct iterator_impl : IteratorInterface<iterator_impl> {
      constexpr iterator_impl(pointer base, size_type idx, const size_type *bases, size_type dim)
          : _iter{base, idx}, _bases{bases}, _dim{dim} {}

      constexpr reference dereference() { return _iter.dereference(); }
      constexpr bool equal_to(iterator_impl it) const noexcept { return it._iter.equal_to(_iter); }
      constexpr void advance(difference_type offset) noexcept { _iter.advance(offset); }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return _iter.distance_to(it._iter);
      }

    protected:
      typename field_type::iterator_impl _iter;
      const size_type *_bases{nullptr};
      size_type _dim{};
    };
    struct const_iterator_impl : IteratorInterface<const_iterator_impl> {
      constexpr const_iterator_impl(const_pointer base, size_type idx, const size_type *bases,
                                    size_type dim)
          : _iter{base, idx}, _bases{bases}, _dim{dim} {}

      constexpr const_reference dereference() { return _iter.dereference(); }
      constexpr bool equal_to(const_iterator_impl it) const noexcept {
        return it._iter.equal_to(_iter);
      }
      constexpr void advance(difference_type offset) noexcept { _iter.advance(offset); }
      constexpr difference_type distance_to(const_iterator_impl it) const noexcept {
        return _iter.distance_to(it._iter);
      }

    protected:
      typename field_type::const_iterator_impl _iter;
      const size_type *_bases{nullptr};
      size_type _dim{};
    };
    using iterator = LegacyIterator<iterator_impl>;
    using const_iterator = LegacyIterator<const_iterator_impl>;

    constexpr auto begin() noexcept {
      return make_iterator<iterator_impl>(_field.data(), (size_type)0, _shape.data(),
                                          _shape.size());
    }
    constexpr auto end() noexcept {
      return make_iterator<iterator_impl>(_field.data(), size(), _shape.data(), _shape.size());
    }
    constexpr auto begin() const noexcept {
      return make_iterator<const_iterator_impl>(_field.data(), (size_type)0, _shape.data(),
                                                _shape.size());
    }
    constexpr auto end() const noexcept {
      return make_iterator<const_iterator_impl>(_field.data(), size(), _shape.data(),
                                                _shape.size());
    }

    /// capacity
    constexpr size_type dims() const noexcept { return _shape.size() - 1; }
    constexpr size_type size() const noexcept { return _field.size(); }
    constexpr size_type capacity() const noexcept { return _field.capacity(); }
    constexpr bool empty() const noexcept { return size() == 0; }
    constexpr pointer data() noexcept { return _field.data(); }
    constexpr const_pointer data() const noexcept { return _field.data(); }

    constexpr reference front() noexcept {
      // if (this->onHost()) [[likely]]
      return _field.front();
    }
    constexpr const_reference front() const noexcept {
      // if (this->onHost()) [[likely]]
      return _field.front();
    }
    constexpr reference back() noexcept {
      // if (this->onHost()) [[likely]]
      return _field.back();
    }
    constexpr const_reference back() const noexcept {
      // if (this->onHost()) [[likely]]
      return _field.back();
    }

    /// element access
    template <typename... Args, enable_if_all<is_integral_v<Args>...> = 0>
    constexpr size_type linearOffset(Args... is) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (sizeof...(is) + 1 != _shape.size())
        printf("densefield ofb! num indices (%d) does not equal the dim of shape (%d)\n",
               (int)(sizeof...(is)), (int)_shape.size() - 1);
#endif
      std::vector<size_type> bases(_shape.size());
      _shape.retrieveVals(bases.data());
      size_type offset = 0, i = 0;
      ((void)(offset += (size_type)is * bases[++i]), ...);
      return offset;
    }

    template <typename... Args, enable_if_all<is_integral_v<Args>...> = 0>
    constexpr reference operator()(Args... is) noexcept {
      size_type offset = linearOffset(zs::move(is)...);
      return _field[offset];
    }
    template <typename... Args, enable_if_all<is_integral_v<Args>...> = 0>
    constexpr conditional_t<is_fundamental_v<value_type>, value_type, const_reference> operator()(
        Args... is) const noexcept {
      size_type offset = linearOffset(zs::move(is)...);
      return _field[offset];
    }
    constexpr reference operator[](size_type offset) noexcept { return _field[offset]; }
    constexpr conditional_t<is_fundamental_v<value_type>, value_type, const_reference> operator[](
        size_type offset) const noexcept {
      return _field[offset];
    }
    /// ctor, assignment operator
    DenseField(const DenseField &o) : _field{o._field}, _shape{o._shape} {}
    DenseField &operator=(const DenseField &o) {
      if (this == &o) return *this;
      DenseField tmp(o);
      swap(tmp);
      return *this;
    }
    DenseField clone(const allocator_type &allocator) const {
      DenseField ret{};
      ret._field = _field.clone(allocator);
      ret._shape = _shape.clone(allocator);
      return ret;
    }
    DenseField clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }
    DenseField(DenseField &&o) noexcept {
      const DenseField defaultVector{};
      _field = zs::exchange(o._field, defaultVector._field);
      _shape = zs::exchange(o._shape, defaultVector._shape);
    }
    /// make move-assignment safe for self-assignment
    DenseField &operator=(DenseField &&o) noexcept {
      if (this == &o) return *this;
      DenseField tmp(zs::move(o));
      swap(tmp);
      return *this;
    }
    void swap(DenseField &o) noexcept {
      std::swap(_field, o._field);
      std::swap(_shape, o._shape);
    }
    friend void swap(DenseField &a, DenseField &b) noexcept { a.swap(b); }

    void reset(int ch) { _field.reset(ch); }
    /// @note field is invalidated

    void reshape(const std::vector<size_type> &newShape) {
      shape_type shape{newShape.size() + 1};
      size_type base = 1;
      shape[newShape.size()] = 1;
      for (sint_t i = newShape.size() - 1; i >= 0;) {
        base *= newShape[i];
        shape[i--] = base;
      }
      _field.resize(base);
      if (_shape.memspace() != memsrc_e::host)
        _shape = shape.clone(_shape.memoryLocation());
      else
        _shape = zs::move(shape);
    }

    field_type _field;
    shape_type _shape;
  };

  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<u8, ZSPmrAllocator<>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<u32, ZSPmrAllocator<>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<u64, ZSPmrAllocator<>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<i8, ZSPmrAllocator<>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<i32, ZSPmrAllocator<>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<i64, ZSPmrAllocator<>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<f32, ZSPmrAllocator<>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<f64, ZSPmrAllocator<>>;

  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<u8, ZSPmrAllocator<true>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<u32, ZSPmrAllocator<true>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<u64, ZSPmrAllocator<true>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<i8, ZSPmrAllocator<true>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<i32, ZSPmrAllocator<true>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<i64, ZSPmrAllocator<true>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<f32, ZSPmrAllocator<true>>;
  ZPC_FWD_DECL_TEMPLATE_STRUCT DenseField<f64, ZSPmrAllocator<true>>;

  template <execspace_e S, typename DenseFieldT, bool Base, typename = void> struct DenseFieldView {
    static constexpr auto space = S;
    static constexpr bool is_const_structure = is_const_v<DenseFieldT>;
    using container_type = remove_const_t<DenseFieldT>;
    using const_container_type = add_const_t<container_type>;
    using pointer = conditional_t<is_const_structure, typename container_type::const_pointer,
                                  typename container_type::pointer>;
    using value_type = typename container_type::value_type;
    using size_type = typename container_type::size_type;
    using difference_type = typename container_type::difference_type;

    /// @note may not need to embed 'Base' variant, for the shape info already includes the size.
    template <typename VectorT, bool B> using vector_view_type = decltype(view<space>(
        declval<conditional_t<is_const_structure, const VectorT &, VectorT &>>(), wrapv<B>{}));

    DenseFieldView() noexcept = default;
    explicit constexpr DenseFieldView(DenseFieldT &df)
        : field{view<space>(df._field, true_c)}, shape{view<space>(df._shape, wrapv<Base>{})} {}

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator[](size_type offset) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (offset >= shape[0]) {
        printf("densefield [%s] ofb! accessing %lld out of [0, %lld)\n", field._nameTag.asChars(),
               (long long)offset, (long long)shape[0]);
        return field[detail::deduce_numeric_max<std::uintptr_t>()];
      }
#endif
      return field[offset];
    }
    constexpr decltype(auto) operator[](size_type offset) const {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (offset >= shape[0]) {
        printf("densefield [%s] ofb! accessing %lld out of [0, %lld)\n", field._nameTag.asChars(),
               (long long)offset, (long long)shape[0]);
        return field[detail::deduce_numeric_max<std::uintptr_t>()];
      }
#endif
      return field[offset];
    }
    template <typename... Args, enable_if_all<is_integral_v<Args>...> = 0>
    constexpr size_type linearOffset(Args... is) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if constexpr (!Base) {
        if (sizeof...(is) + 1 != shape.size())
          printf("densefield ofb! num indices (%d) does not equal the dim of shape (%d)\n",
                 (int)(sizeof...(is)), (int)shape.size() - 1);
      }
#endif
      size_type offset = 0, i = 0;
      ((void)(offset += (size_type)is * shape[++i]), ...);
      return offset;
    }
    template <typename... Args, bool V = !is_const_structure && (... && is_integral_v<Args>),
              enable_if_t<V> = 0>
    constexpr decltype(auto) operator()(Args... is) {
      const size_type offset = linearOffset(zs::move(is)...);
      return operator[](offset);
    }
    template <typename... Args, enable_if_all<is_integral_v<Args>...> = 0>
    constexpr decltype(auto) operator()(Args... is) const {
      const size_type offset = linearOffset(zs::move(is)...);
      return operator[](offset);
    }

    constexpr pointer data() noexcept { return field.data(); }
    constexpr typename container_type::const_pointer data() const noexcept { return field.data(); }

    vector_view_type<typename DenseFieldT::field_type, true> field;
    vector_view_type<typename DenseFieldT::shape_type, Base> shape;
  };

  template <execspace_e ExecSpace, typename T, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(DenseField<T, Allocator> &df, wrapv<Base> = {}) {
    return DenseFieldView<ExecSpace, DenseField<T, Allocator>, Base>{df};
  }
  template <execspace_e ExecSpace, typename T, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const DenseField<T, Allocator> &df, wrapv<Base> = {}) {
    return DenseFieldView<ExecSpace, const DenseField<T, Allocator>, Base>{df};
  }

  template <execspace_e ExecSpace, typename T, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(DenseField<T, Allocator> &df, wrapv<Base>, const SmallString &tagName) {
    auto ret = DenseFieldView<ExecSpace, DenseField<T, Allocator>, Base>{df};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret.field._nameTag = tagName;
    ret.shape._nameTag = tagName;
#endif
    return ret;
  }
  template <execspace_e ExecSpace, typename T, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const DenseField<T, Allocator> &df, wrapv<Base>, const SmallString &tagName) {
    auto ret = DenseFieldView<ExecSpace, const DenseField<T, Allocator>, Base>{df};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret.field._nameTag = tagName;
    ret.shape._nameTag = tagName;
#endif
    return ret;
  }

  template <execspace_e space, typename T, typename Allocator>
  decltype(auto) proxy(DenseField<T, Allocator> &df) {
    return view<space>(df, false_c);
  }
  template <execspace_e space, typename T, typename Allocator>
  decltype(auto) proxy(const DenseField<T, Allocator> &df) {
    return view<space>(df, false_c);
  }

  template <execspace_e space, typename T, typename Allocator>
  decltype(auto) proxy(DenseField<T, Allocator> &df, const SmallString &tagName) {
    return view<space>(df, false_c, tagName);
  }
  template <execspace_e space, typename T, typename Allocator>
  decltype(auto) proxy(const DenseField<T, Allocator> &df, const SmallString &tagName) {
    return view<space>(df, false_c, tagName);
  }

#if ZS_ENABLE_SERIALIZATION
  template <typename S, typename T> void serialize(S &s, DenseField<T, ZSPmrAllocator<>> &df) {
    if (!df.memoryLocation().onHost()) {
      df = df.clone({memsrc_e::host, -1});
    }

    serialize(s, df._field);
    serialize(s, df._shape);
  }
#endif

}  // namespace zs