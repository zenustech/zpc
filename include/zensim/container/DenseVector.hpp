#pragma once
#include "zensim/container/Vector.hpp"
#include "zensim/resource/Resource.h"

namespace zs {

  template <typename T, typename AllocatorT = ZSPmrAllocator<>> struct DenseVector {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "DenseVector only works with zspmrallocator for now.");
    static_assert(is_same_v<T, remove_cvref_t<T>>, "T is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<T>, "element is not default-constructible!");
    // static_assert(zs::is_trivially_copyable_v<T>, "element is not trivially-copyable!");
    static_assert(zs::is_trivially_copyable_v<T> || std::is_copy_assignable_v<T>,
                  "element is not copyable!");

    using allocator_type = AllocatorT;

    using value_type = T;
    using field_type = Vector<value_type, allocator_type>;
    using shape_type = Vector<size_type, allocator_type>;

    using size_type = typename field_type::size_type;
    using difference_type = typename field_type::difference_type;
    using reference = typename field_type::reference;
    using const_reference = typename field_type::const_reference;
    using pointer = typename field_type::pointer;
    using const_pointer = typename field_type::const_pointer;

    constexpr decltype(auto) memoryLocation() const noexcept { return _field.location; }
    constexpr ProcID devid() const noexcept { return _field.devid(); }
    constexpr memsrc_e memspace() const noexcept { return _field.memspace(); }
    decltype(auto) get_allocator() const noexcept { return _field.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      return _field.get_default_allocator(mre, devid);
    }

    static size_type shape_size(const std::vector<size_type> &shape) const noexcept {
      size_type sz = 1;
      for (size_type d : shape) sz *= d;
      return sz;
    }
    /// allocator-aware
    DenseVector(const allocator_type &allocator, const std::vector<size_type> &shape)
        : _field{allocator, shape_size(shape)}, _shape{shape.size()} {
      size_type base = 1;
      for (sint_t i = shape.size() - 1; i >= 0; --i) {
        base *= shape[i];
        _shape[i] = base;
      }
      if (allocator.location.memspace() != memsrc_e::host)
        _shape = _shape.clone(allocator.location);
    }
    explicit DenseVector(const std::vector<size_type> &shape, memsrc_e mre = memsrc_e::host,
                         ProcID devid = -1)
        : DenseVector{get_default_allocator(mre, devid), shape} {}
    DenseVector(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : DenseVector{get_default_allocator(mre, devid), {}} {}

    ~DenseVector() = default;

    inline value_type getVal(size_type i = 0) const { return _field.getVal(i); }
    inline void retrieveVals(value_type *dst) const { _field.retrieveVals(dst); }
    inline void setVal(value_type v, size_type i = 0) const { _field.setVal(v, i); }

    struct iterator_impl : typename field_type::iterator_impl {
      using base_t = typename field_type::iterator_impl;
      constexpr iterator_impl(pointer base, size_type idx, const size_type *bases, size_type dim)
          : base_t{base, idx}, _bases{bases}, _dim{dim} {}

    protected:
      const size_type *_bases{nullptr};
      size_type _dim{};
    };
    struct const_iterator_impl : typename field_type::const_iterator_impl {
      using base_t = typename field_type::const_iterator_impl;
      constexpr const_iterator_impl(const_pointer base, size_type idx, const size_type *bases,
                                    size_type dim)
          : base_t{base, idx}, _bases{bases}, _dim{dim} {}

    protected:
      const size_type *_bases{nullptr};
      size_type _dim{};
    };
    using iterator = LegacyIterator<iterator_impl>;
    using const_iterator = LegacyIterator<const_iterator_impl>;

    constexpr auto begin() noexcept { return make_iterator<iterator_impl>(_base, 0); }
    constexpr auto end() noexcept { return make_iterator<iterator_impl>(_base, size()); }
    constexpr auto begin() const noexcept { return make_iterator<const_iterator_impl>(_base, 0); }
    constexpr auto end() const noexcept {
      return make_iterator<const_iterator_impl>(_base, size());
    }

    /// capacity
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
      if (sizeof...(is) != _shape.size())
        printf("densevector ofb! num indices (%d) does not equal the dim of shape (%d)\n",
               (int)(sizeof...(is)), (int)_shape.size());
#endif
      std::vector<size_type> bases(_shape.size());
      _shape.retrieveVals(bases.data());
      size_type offset = 1, i = 0;
      (void)((offset += (size_type)is * (i != _shape.size() - 1 ? bases[i + 1] : (size_type)1)),
             ...);
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
    DenseVector(const DenseVector &o) : _field{o._field}, _shape{o._shape} {}
    DenseVector &operator=(const DenseVector &o) {
      if (this == &o) return *this;
      DenseVector tmp(o);
      swap(tmp);
      return *this;
    }
    DenseVector clone(const allocator_type &allocator) const {
      DenseVector ret{};
      ret._field = _field.clone(allocator);
      ret._shape = _shape.clone(allocator);
      return ret;
    }
    DenseVector clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }
    DenseVector(DenseVector &&o) noexcept {
      const DenseVector defaultVector{};
      _field = std::exchange(o._field, defaultVector._field);
      _shape = std::exchange(o._shape, defaultVector._shape);
    }
    /// make move-assignment safe for self-assignment
    DenseVector &operator=(DenseVector &&o) noexcept {
      if (this == &o) return *this;
      DenseVector tmp(zs::move(o));
      swap(tmp);
      return *this;
    }
    void swap(DenseVector &o) noexcept {
      std::swap(_field, o._field);
      std::swap(_shape, o._shape);
    }
    friend void swap(DenseVector &a, DenseVector &b) noexcept { a.swap(b); }

    void reset(int ch) { _field.reset(ch); }
    /// @note field is invalidated

    void reshape(const std::vector<size_type> &newShape) {
      // _shape = newShape;
      size_type base = 1;
      shape_type shape{newShape.size()};
      for (sint_t i = newShape.size() - 1; i >= 0; --i) {
        base *= newShape[i];
        shape[i] = base;
      }
      _field.resize(base);
      _shape = shape.clone(_shape.memoryLocation());
    }

  protected:
    field_type _field;
    shape_type _shape;
  };

  extern template struct DenseVector<u8, ZSPmrAllocator<>>;
  extern template struct DenseVector<u32, ZSPmrAllocator<>>;
  extern template struct DenseVector<u64, ZSPmrAllocator<>>;
  extern template struct DenseVector<i8, ZSPmrAllocator<>>;
  extern template struct DenseVector<i32, ZSPmrAllocator<>>;
  extern template struct DenseVector<i64, ZSPmrAllocator<>>;
  extern template struct DenseVector<f32, ZSPmrAllocator<>>;
  extern template struct DenseVector<f64, ZSPmrAllocator<>>;

  extern template struct DenseVector<u8, ZSPmrAllocator<true>>;
  extern template struct DenseVector<u32, ZSPmrAllocator<true>>;
  extern template struct DenseVector<u64, ZSPmrAllocator<true>>;
  extern template struct DenseVector<i8, ZSPmrAllocator<true>>;
  extern template struct DenseVector<i32, ZSPmrAllocator<true>>;
  extern template struct DenseVector<i64, ZSPmrAllocator<true>>;
  extern template struct DenseVector<f32, ZSPmrAllocator<true>>;
  extern template struct DenseVector<f64, ZSPmrAllocator<true>>;

}  // namespace zs