#pragma once
#include <type_traits>

#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"

namespace zs {

  template <typename T, typename AllocatorT = ZSPmrAllocator<>> struct Vector {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "Vector only works with zspmrallocator for now.");
    static_assert(is_same_v<T, remove_cvref_t<T>>, "T is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<T>, "element is not default-constructible!");
    static_assert(std::is_trivially_copyable_v<T>, "element is not trivially-copyable!");

    using value_type = T;
    using allocator_type = AllocatorT;
    using size_type = std::size_t;
    using difference_type = sint_t;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    constexpr decltype(auto) memoryLocation() const noexcept { return _allocator.location; }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) get_allocator() const noexcept { return _allocator; }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (std::size_t)1 << (std::size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }
    pointer allocate(std::size_t bytes) {
      /// virtual memory way
      if constexpr (is_virtual_zs_allocator<allocator_type>::value) {
        _allocator.commit(0, bytes);
        return (pointer)_allocator.address(0);
      }
      /// conventional way
      else
        return (pointer)_allocator.allocate(bytes, std::alignment_of_v<value_type>);
    }

    /// allocator-aware
    Vector(const allocator_type &allocator, size_type count)
        : _allocator{allocator},
          _base{allocate(sizeof(value_type) * count)},
          _size{count},
          _capacity{count} {}
    explicit Vector(size_type count, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Vector{get_default_allocator(mre, devid), count} {}
    Vector(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Vector{get_default_allocator(mre, devid), 0} {}

    ~Vector() {
      if (_base && capacity() > 0)
        _allocator.deallocate(_base, capacity() * sizeof(value_type),
                              std::alignment_of_v<value_type>);
    }

    inline value_type getVal(size_type i = 0) const {
      value_type res[1];
      Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)res},
                     MemoryEntity{memoryLocation(), (void *)(data() + i)}, sizeof(value_type));
      return res[0];
    }
    inline void setVal(value_type v, size_type i = 0) const {
      Resource::copy(MemoryEntity{memoryLocation(), (void *)(data() + i)},
                     MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)&v},
                     sizeof(value_type));
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
          _base{allocate(sizeof(value_type) * o._capacity)},
          _size{o.size()},
          _capacity{o._capacity} {
      if (o.data() && o.size() > 0)
        Resource::copy(MemoryEntity{memoryLocation(), (void *)data()},
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
      Resource::copy(MemoryEntity{allocator.location, (void *)ret.data()},
                     MemoryEntity{memoryLocation(), (void *)this->data()}, usedBytes());
      return ret;
    }
    Vector clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
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
    friend void swap(Vector &a, Vector &b) noexcept { a.swap(b); }

    void clear() { resize(0); }
    void reset(int ch) {
      Resource::memset(MemoryEntity{memoryLocation(), (void *)data()}, ch, usedBytes());
    }
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
          if constexpr (is_virtual_zs_allocator<allocator_type>::value) {
            _capacity = geometric_size_growth(newSize);
            _allocator.commit(_capacity * sizeof(value_type));
            _size = newSize;
          }
          /// conventional way
          else {
            Vector tmp{_allocator, geometric_size_growth(newSize)};
            if (size())
              Resource::copy(MemoryEntity{tmp.memoryLocation(), (void *)tmp.data()},
                             MemoryEntity{memoryLocation(), (void *)data()}, usedBytes());
            tmp._size = newSize;
            swap(tmp);
          }
          return;
        } else
          _size = newSize;
      }
    }
    void resize(size_type newSize, int ch) {
      const auto oldSize = size();
      if (newSize < oldSize) {
        _size = newSize;
        // Resource::memset(MemoryEntity{memoryLocation(), data() + newSize}, ch,
        //                 sizeof(T) * (capacity() - newSize));
        return;
      }
      if (newSize > oldSize) {
        const auto oldCapacity = capacity();
        if (newSize > oldCapacity) {
          /// virtual memory way
          if constexpr (is_virtual_zs_allocator<allocator_type>::value) {
            _capacity = geometric_size_growth(newSize);
            _allocator.commit(_capacity * sizeof(value_type));
            Resource::memset(MemoryEntity{memoryLocation(), (void *)(data() + _size)}, ch,
                             sizeof(T) * (newSize - _size));
            _size = newSize;
          }
          /// conventional way
          else {
            Vector tmp{_allocator, geometric_size_growth(newSize)};
            if (size())
              Resource::copy(MemoryEntity{tmp.memoryLocation(), (void *)tmp.data()},
                             MemoryEntity{memoryLocation(), (void *)data()}, usedBytes());
            Resource::memset(MemoryEntity{tmp.memoryLocation(), (void *)(tmp.data() + _size)}, ch,
                             sizeof(T) * (newSize - _size));
            tmp._size = newSize;
            swap(tmp);
          }
          return;
        } else {
          Resource::memset(MemoryEntity{memoryLocation(), (void *)(data() + _size)}, ch,
                           sizeof(T) * (newSize - _size));
          _size = newSize;
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
      Resource::copy(MemoryEntity{memoryLocation(), (void *)(_base + size())},
                     MemoryEntity{other.memoryLocation(), (void *)other.data()}, sizeof(T) * count);
    }

  protected:
    constexpr std::size_t usedBytes() const noexcept { return sizeof(T) * size(); }

    constexpr size_type geometric_size_growth(size_type newSize,
                                              size_type capacity) const noexcept {
      size_type geometricSize = capacity;
      geometricSize = geometricSize + geometricSize / 2;
      if (newSize > geometricSize) return newSize;
      return geometricSize;
    }
    constexpr size_type geometric_size_growth(size_type newSize) const noexcept {
      return geometric_size_growth(newSize, capacity());
    }

    allocator_type _allocator{};  // alignment should be inside allocator
    pointer _base{nullptr};
    size_type _size{0}, _capacity{0};
  };

  extern template struct Vector<u8, ZSPmrAllocator<>>;
  extern template struct Vector<u32, ZSPmrAllocator<>>;
  extern template struct Vector<u64, ZSPmrAllocator<>>;
  extern template struct Vector<i8, ZSPmrAllocator<>>;
  extern template struct Vector<i32, ZSPmrAllocator<>>;
  extern template struct Vector<i64, ZSPmrAllocator<>>;
  extern template struct Vector<f32, ZSPmrAllocator<>>;
  extern template struct Vector<f64, ZSPmrAllocator<>>;

  extern template struct Vector<u8, ZSPmrAllocator<true>>;
  extern template struct Vector<u32, ZSPmrAllocator<true>>;
  extern template struct Vector<u64, ZSPmrAllocator<true>>;
  extern template struct Vector<i8, ZSPmrAllocator<true>>;
  extern template struct Vector<i32, ZSPmrAllocator<true>>;
  extern template struct Vector<i64, ZSPmrAllocator<true>>;
  extern template struct Vector<f32, ZSPmrAllocator<true>>;
  extern template struct Vector<f64, ZSPmrAllocator<true>>;

  template <typename T,
            enable_if_all<is_same_v<T, remove_cvref_t<T>>, std::is_default_constructible_v<T>,
                          std::is_trivially_copyable_v<T>>
            = 0>
  auto from_std_vector(const std::vector<T> &vs,
                       const MemoryLocation &mloc = {memsrc_e::host, -1}) {
    auto allocator = get_memory_source(mloc.memspace(), mloc.devid());
    Vector<T> ret{allocator, vs.size()};
    Resource::copy(MemoryEntity{allocator.location, (void *)ret.data()},
                   MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)vs.data()},
                   sizeof(T) * vs.size());
    return ret;
  }

  template <execspace_e, typename VectorT, bool Base = false, typename = void> struct VectorView {
    static constexpr bool is_const_structure = std::is_const_v<VectorT>;
    using vector_type = std::remove_const_t<VectorT>;
    using const_vector_type = std::add_const_t<vector_type>;
    using pointer = conditional_t<is_const_structure, typename vector_type::const_pointer,
                                  typename vector_type::pointer>;
    using value_type = typename vector_type::value_type;
    using size_type = typename vector_type::size_type;
    using difference_type = typename vector_type::difference_type;

    VectorView() noexcept = default;
    explicit constexpr VectorView(VectorT &vector)
        : _vector{vector.data()}, _vectorSize{vector.size()} {}

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator[](size_type i) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      using RetT = decltype(_vector[i]);
      if (i >= _vectorSize) {
        printf("vector [%s] ofb! accessing %lld out of [0, %lld)\n", _nameTag.asChars(),
               (long long)i, (long long)_vectorSize);
        return (RetT)(*((value_type *)(limits<std::uintptr_t>::max() - sizeof(value_type) + 1)));
      }
#endif
      return _vector[i];
    }
    constexpr decltype(auto) operator[](size_type i) const {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      using RetT = decltype(_vector[i]);
      if (i >= _vectorSize) {
        printf("vector [%s] ofb! accessing %lld out of [0, %lld)\n", _nameTag.asChars(),
               (long long)i, (long long)_vectorSize);
        return (
            RetT)(*((const value_type *)(limits<std::uintptr_t>::max() - sizeof(value_type) + 1)));
      }
#endif
      return _vector[i];
    }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator()(size_type i) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      using RetT = decltype(_vector[i]);
      if (i >= _vectorSize) {
        printf("vector [%s] ofb! accessing %lld out of [0, %lld)\n", _nameTag.asChars(),
               (long long)i, (long long)_vectorSize);
        return (RetT)(*((value_type *)(limits<std::uintptr_t>::max() - sizeof(value_type) + 1)));
      }
#endif
      return _vector[i];
    }
    constexpr decltype(auto) operator()(size_type i) const {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      using RetT = decltype(_vector[i]);
      if (i >= _vectorSize) {
        printf("vector [%s] ofb! accessing %lld out of [0, %lld)\n", _nameTag.asChars(),
               (long long)i, (long long)_vectorSize);
        return (
            RetT)(*((const value_type *)(limits<std::uintptr_t>::max() - sizeof(value_type) + 1)));
      }
#endif
      return _vector[i];
    }

    constexpr size_type size() const noexcept { return _vectorSize; }

    constexpr pointer data() noexcept { return _vector; }
    constexpr typename vector_type::const_pointer data() const noexcept { return _vector; }

    pointer _vector{nullptr};
    size_type _vectorSize{0};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    SmallString _nameTag{};
#endif
  };

  template <execspace_e S, typename VectorT> struct VectorView<S, VectorT, true, void> {
    static constexpr bool is_const_structure = std::is_const_v<VectorT>;
    using vector_type = std::remove_const_t<VectorT>;
    using const_vector_type = std::add_const_t<vector_type>;
    using pointer = conditional_t<is_const_structure, typename vector_type::const_pointer,
                                  typename vector_type::pointer>;
    using value_type = typename vector_type::value_type;
    using size_type = typename vector_type::size_type;
    using difference_type = typename vector_type::difference_type;

    VectorView() noexcept = default;
    explicit constexpr VectorView(VectorT &vector) : _vector{vector.data()} {}

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator[](size_type i) {
      return _vector[i];
    }
    constexpr decltype(auto) operator[](size_type i) const { return _vector[i]; }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator()(size_type i) {
      return _vector[i];
    }
    constexpr decltype(auto) operator()(size_type i) const { return _vector[i]; }

    constexpr pointer data() noexcept { return _vector; }
    constexpr typename vector_type::const_pointer data() const noexcept { return _vector; }

    pointer _vector{nullptr};
  };

  template <execspace_e ExecSpace, typename T, typename Allocator>
  constexpr decltype(auto) proxy(Vector<T, Allocator> &vec) {
    return VectorView<ExecSpace, Vector<T, Allocator>>{vec};
  }
  template <execspace_e ExecSpace, typename T, typename Allocator>
  constexpr decltype(auto) proxy(const Vector<T, Allocator> &vec) {
    return VectorView<ExecSpace, const Vector<T, Allocator>>{vec};
  }

  template <execspace_e ExecSpace, typename T, typename Allocator>
  constexpr decltype(auto) proxy(Vector<T, Allocator> &vec, const SmallString &tagName) {
    auto ret = VectorView<ExecSpace, Vector<T, Allocator>>{vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }
  template <execspace_e ExecSpace, typename T, typename Allocator>
  constexpr decltype(auto) proxy(const Vector<T, Allocator> &vec, const SmallString &tagName) {
    auto ret = VectorView<ExecSpace, const Vector<T, Allocator>>{vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }

  template <execspace_e ExecSpace, typename T, typename Allocator>
  constexpr decltype(auto) proxy(std::true_type, Vector<T, Allocator> &vec) {
    return VectorView<ExecSpace, Vector<T, Allocator>, true>{vec};
  }
  template <execspace_e ExecSpace, typename T, typename Allocator>
  constexpr decltype(auto) proxy(std::true_type, const Vector<T, Allocator> &vec) {
    return VectorView<ExecSpace, const Vector<T, Allocator>, true>{vec};
  }

  template <execspace_e ExecSpace, typename T, typename Allocator>
  constexpr decltype(auto) proxy(std::true_type, Vector<T, Allocator> &vec,
                                 const SmallString &tagName) {
    return proxy(true_c, vec);
  }
  template <execspace_e ExecSpace, typename T, typename Allocator>
  constexpr decltype(auto) proxy(std::true_type, const Vector<T, Allocator> &vec,
                                 const SmallString &tagName) {
    return proxy(true_c, vec);
  }

}  // namespace zs