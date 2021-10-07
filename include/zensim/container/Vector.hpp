#pragma once
#include <type_traits>

#include "zensim/memory/Allocator.h"
#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"

namespace zs {

  template <typename T, typename Index = std::size_t> struct Vector {
    static_assert(is_same_v<T, remove_cvref_t<T>>, "T is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<T>, "element is not default-constructible!");
    static_assert(std::is_trivially_copyable_v<T>, "element is not trivially-copyable!");

    using value_type = T;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = std::make_unsigned_t<Index>;
    using difference_type = std::make_signed_t<size_type>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    constexpr decltype(auto) memoryLocation() noexcept { return _allocator.location; }
    constexpr decltype(auto) memoryLocation() const noexcept { return _allocator.location; }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) allocator() const noexcept { return _allocator; }

    /// allocator-aware
    Vector(const allocator_type &allocator, size_type count)
        : _allocator{allocator},
          _base{(pointer)_allocator.allocate(count * sizeof(value_type),
                                             std::alignment_of_v<value_type>)},
          _size{count},
          _capacity{count} {}
    explicit Vector(size_type count, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Vector{get_memory_source(mre, devid), count} {}
    Vector(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Vector{get_memory_source(mre, devid), 0} {}

    ~Vector() {
      if (_base && _capacity > 0)
        _allocator.deallocate(_base, _capacity * sizeof(value_type),
                              std::alignment_of_v<value_type>);
    }

    struct iterator_impl : IteratorInterface<iterator_impl> {
      template <typename Ti> constexpr iterator_impl(pointer base, Ti &&idx)
          : _base{base}, _idx{static_cast<size_type>(idx)} {}

      constexpr reference dereference() { return _base[_idx]; }
      constexpr bool equal_to(iterator_impl it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      pointer _base{nullptr};
      size_type _idx{0};
    };
    struct const_iterator_impl : IteratorInterface<const_iterator_impl> {
      template <typename Ti> constexpr const_iterator_impl(const_pointer base, Ti &&idx)
          : _base{base}, _idx{static_cast<size_type>(idx)} {}

      constexpr const_reference dereference() { return _base[_idx]; }
      constexpr bool equal_to(const_iterator_impl it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(const_iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      const_pointer _base{nullptr};
      size_type _idx{0};
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
    constexpr size_type size() const noexcept { return _size; }
    constexpr size_type capacity() const noexcept { return _capacity; }
    constexpr bool empty() const noexcept { return size() == 0; }
    constexpr pointer data() noexcept { return _base; }
    constexpr const_pointer data() const noexcept { return _base; }

    constexpr reference front() noexcept {
      // if (this->onHost()) [[likely]]
      return _base[0];
    }
    constexpr const_reference front() const noexcept {
      // if (this->onHost()) [[likely]]
      return _base[0];
    }
    constexpr reference back() noexcept {
      // if (this->onHost()) [[likely]]
      return _base[size() - 1];
    }
    constexpr const_reference back() const noexcept {
      // if (this->onHost()) [[likely]]
      return _base[size() - 1];
    }

    /// element access
    constexpr reference operator[](size_type idx) noexcept { return _base[idx]; }
    constexpr conditional_t<std::is_fundamental_v<value_type>, value_type, const_reference>
    operator[](size_type idx) const noexcept {
      return _base[idx];
    }
    /// ctor, assignment operator
    Vector(const Vector &o)
        : _allocator{o._allocator},
          _base{(pointer)_allocator.allocate(sizeof(value_type) * o._capacity,
                                             std::alignment_of_v<value_type>)},
          _size{o.size()},
          _capacity{o._capacity} {
      if (o.data() && o.size() > 0)
        copy(MemoryEntity{memoryLocation(), (void *)data()},
             MemoryEntity{o.memoryLocation(), (void *)o.data()}, o.usedBytes());
    }
    Vector &operator=(const Vector &o) {
      if (this == &o) return *this;
      Vector tmp(o);
      swap(tmp);
      return *this;
    }
    Vector clone(const allocator_type &allocator) const {
      Vector ret{allocator, capacity()};
      copy(MemoryEntity{allocator.location, (void *)ret.data()},
           MemoryEntity{memoryLocation(), (void *)this->data()}, usedBytes());
      return ret;
    }
    Vector clone(const MemoryLocation &mloc) const {
      return clone(get_memory_source(mloc.memspace(), mloc.devid()));
    }
    /// assignment or destruction after std::move
    /// https://www.youtube.com/watch?v=ZG59Bqo7qX4
    /// explicit noexcept
    /// leave the source object in a valid (default constructed) state
    Vector(Vector &&o) noexcept {
      const Vector defaultVector{};
      _allocator = std::exchange(o._allocator, defaultVector._allocator);
      _base = std::exchange(o._base, defaultVector._base);
      _size = std::exchange(o._size, defaultVector.size());
      _capacity = std::exchange(o._capacity, defaultVector._capacity);
    }
    /// make move-assignment safe for self-assignment
    Vector &operator=(Vector &&o) noexcept {
      if (this == &o) return *this;
      Vector tmp(std::move(o));
      swap(tmp);
      return *this;
    }
    void swap(Vector &o) noexcept {
      std::swap(_allocator, o._allocator);
      std::swap(_base, o._base);
      std::swap(_size, o._size);
      std::swap(_capacity, o._capacity);
    }
    friend void swap(Vector &a, Vector &b) { a.swap(b); }

    void clear() { resize(0); }
    void resize(size_type newSize) {
      const auto oldSize = size();
      if (newSize < oldSize) {
        _size = newSize;
        return;
      }
      if (newSize > oldSize) {
        const auto oldCapacity = capacity();
        if (newSize > oldCapacity) {
          /// virtual memory way
          /// conventional way
          Vector tmp{_allocator, geometric_size_growth(newSize)};
          if (size())
            copy(MemoryEntity{tmp.memoryLocation(), (void *)tmp.data()},
                 MemoryEntity{memoryLocation(), (void *)data()}, usedBytes());
          tmp._size = newSize;
          swap(tmp);
          return;
        }
      }
    }

    void push_back(const value_type &val) {
      if (size() >= capacity()) resize(size() + 1);
      (*this)[_size++] = val;
    }
    void push_back(value_type &&val) {
      if (size() >= capacity()) resize(size() + 1);
      (*this)[_size++] = std::move(val);
    }

    void append(const Vector &other) {
      difference_type count = other.size();  //< def standard iterator
      if (count <= 0) return;
      size_type unusedCapacity = capacity() - size();
      if (count > unusedCapacity)
        resize(size() + count);
      else
        _size += count;
      copy(MemoryEntity{memoryLocation(), (void *)(_base + size())},
           MemoryEntity{other.memoryLocation(), (void *)other.data()}, sizeof(T) * count);
    }

  protected:
    constexpr std::size_t usedBytes() const noexcept { return sizeof(T) * size(); }

    constexpr size_type geometric_size_growth(size_type newSize) noexcept {
      size_type geometricSize = capacity();
      geometricSize = geometricSize + geometricSize / 2;
      if (newSize > geometricSize) return newSize;
      return geometricSize;
    }

    allocator_type _allocator{};  // alignment should be inside allocator
    pointer _base{nullptr};
    size_type _size{0}, _capacity{0};
  };

  template <execspace_e, typename VectorT, typename = void> struct VectorView {
    using vector_t = typename VectorT::pointer;
    using size_type = typename VectorT::size_type;

    constexpr VectorView() = default;
    ~VectorView() = default;
    explicit constexpr VectorView(VectorT &vector)
        : _vector{vector.data()}, _vectorSize{vector.size()} {}

    constexpr decltype(auto) operator[](size_type i) { return _vector[i]; }
    constexpr decltype(auto) operator[](size_type i) const { return _vector[i]; }
    constexpr decltype(auto) operator()(size_type i) { return _vector[i]; }
    constexpr decltype(auto) operator()(size_type i) const { return _vector[i]; }
    constexpr size_type size() const noexcept { return _vectorSize; }

    vector_t _vector{nullptr};
    size_type _vectorSize{0};
  };

  template <execspace_e Space, typename VectorT> struct VectorView<Space, const VectorT> {
    using vector_t = typename VectorT::const_pointer;
    using size_type = typename VectorT::size_type;

    constexpr VectorView() = default;
    ~VectorView() = default;
    explicit constexpr VectorView(const VectorT &vector)
        : _vector{vector.data()}, _vectorSize{vector.size()} {}

    constexpr decltype(auto) operator[](size_type i) { return _vector[i]; }
    constexpr decltype(auto) operator[](size_type i) const { return _vector[i]; }
    constexpr decltype(auto) operator()(size_type i) { return _vector[i]; }
    constexpr decltype(auto) operator()(size_type i) const { return _vector[i]; }
    constexpr size_type size() const noexcept { return _vectorSize; }

    vector_t _vector{nullptr};
    size_type _vectorSize{0};
  };

  template <execspace_e ExecSpace, typename T, typename Index>
  constexpr decltype(auto) proxy(Vector<T, Index> &vec) {  // currently ignore constness
    return VectorView<ExecSpace, Vector<T, Index>>{vec};
  }
  template <execspace_e ExecSpace, typename T, typename Index>
  constexpr decltype(auto) proxy(const Vector<T, Index> &vec) {  // currently ignore constness
    return VectorView<ExecSpace, const Vector<T, Index>>{vec};
  }

}  // namespace zs