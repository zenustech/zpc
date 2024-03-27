#pragma once
#include "zensim/ZpcImplPattern.hpp"
#include "zensim/ZpcTuple.hpp"

namespace zs {

  /// strongly-typed handle
  template <typename Id, typename TypeName> struct Handle {
    static_assert(is_integral_v<Id>, "Id should be an integral type for the moment.");
    using value_type = Id;

    static constexpr auto type_name() { return get_type_str<TypeName>(); }

    constexpr value_type get() const noexcept { return id; }
    explicit constexpr operator value_type() const noexcept { return id; }

    constexpr bool operator==(const Handle& o) const noexcept { return id == o.get(); }

    Id id;
  };

  /// unique ptr
  /// @ref: microsoft/STL
  template <typename T> struct DefaultDelete {
    constexpr DefaultDelete() noexcept = default;
    template <typename TT, enable_if_t<is_convertible_v<TT*, T*>> = 0>
    constexpr DefaultDelete(const DefaultDelete<TT>&) noexcept {}
    constexpr void operator()(T* p) { delete p; };
  };
  template <typename T> struct DefaultDelete<T[]> {
    constexpr DefaultDelete() noexcept = default;
    template <typename TT, enable_if_t<is_convertible_v<TT (*)[], T (*)[]>> = 0>
    constexpr DefaultDelete(const DefaultDelete<TT[]>&) noexcept {}
    constexpr void operator()(T* p) { delete[] p; };
  };
  template <typename T, typename D = DefaultDelete<T>> struct Unique {
  public:
    using pointer = T*;
    using element_type = T;
    using deleter_type = D;

    template <typename DD = D, enable_if_all<!is_pointer_v<DD>, is_default_constructible_v<DD>> = 0>
    constexpr Unique() noexcept : _storage{} {}

    /// nullptr
    template <typename DD = D, enable_if_all<!is_pointer_v<DD>, is_default_constructible_v<DD>> = 0>
    constexpr Unique(decltype(nullptr)) noexcept : _storage{} {}

    constexpr Unique& operator=(decltype(nullptr)) noexcept {
      reset();
      return *this;
    }

    ///
    template <typename DD = D,
              enable_if_all<!is_pointer_v<DD>, !is_reference_v<DD>, is_default_constructible_v<DD>>
              = 0>
    constexpr explicit Unique(pointer p) noexcept {
      _storage.template get<1>() = p;
    }

    template <typename DD = D, enable_if_t<is_constructible_v<DD, const DD&>> = 0>
    constexpr Unique(pointer p, const D& d) noexcept : _storage{d, p} {}

    template <typename DD = D, enable_if_all<!is_reference_v<DD>, is_constructible_v<DD, DD>> = 0>
    constexpr Unique(pointer p, D&& d) noexcept : _storage{zs::move(d), p} {}

    template <typename DD = D,
              enable_if_all<is_reference_v<DD>, is_constructible_v<DD, remove_reference_t<DD>>> = 0>
    Unique(pointer, remove_reference_t<D>&&) = delete;

    ///
    Unique(const Unique&) = delete;
    Unique& operator=(const Unique&) = delete;

    ///
    template <typename DD = D, enable_if_t<is_move_constructible_v<DD>> = 0>
    constexpr Unique(Unique&& o) noexcept
        : _storage(zs::forward<D>(o.get_deleter()), o.release()) {}
    template <typename DD = D, enable_if_t<is_move_assignable_v<DD>> = 0>
    constexpr Unique& operator=(Unique&& o) noexcept {
      reset(o.release());
      _storage.template get<0>() = zs::forward<D>(o._storage.template get<0>());
      return *this;
    }

    ///
    template <
        typename TT, typename DD,
        enable_if_all<!is_array_v<TT>, is_convertible_v<typename Unique<TT, DD>::pointer, pointer>,
                      (is_reference_v<D> ? is_same_v<DD, D> : is_convertible_v<DD, D>)>
        = 0>
    constexpr Unique(Unique<TT, DD>&& o) noexcept
        : _storage{zs::forward<DD>(o.get_deleter()), o.release()} {}
    template <
        typename TT, typename DD,
        enable_if_all<!is_array_v<TT>, is_convertible_v<typename Unique<TT, DD>::pointer, pointer>,
                      is_assignable_v<D&, DD>>
        = 0>
    constexpr Unique& operator=(Unique<TT, DD>&& o) noexcept {
      reset(o.release());
      _storage.template get<0>() = zs::forward<DD>(o._storage.template get<0>());
      return *this;
    }

    constexpr void swap(Unique& o) noexcept {
      zs_swap(_storage.template get<0>(), o._storage.template get<0>());
      zs_swap(_storage.template get<1>(), o._storage.template get<1>());
    }

    ~Unique() noexcept {
      if (auto p = _storage.template get<1>(); p) {
        _storage.template get<0>()(p);
        _storage.template get<1>() = nullptr;
      }
    }

    [[nodiscard]] constexpr D& get_deleter() noexcept { return _storage.template get<0>(); }

    [[nodiscard]] constexpr const D& get_deleter() const noexcept {
      return _storage.template get<0>();
    }

    [[nodiscard]] constexpr add_lvalue_reference_t<T> operator*() const
        noexcept(noexcept(*zs::declval<pointer>())) {
      return *_storage.template get<1>();
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept {
      return _storage.template get<1>();
    }

    [[nodiscard]] constexpr pointer get() const noexcept { return _storage.template get<1>(); }

    constexpr explicit operator bool() const noexcept {
      return static_cast<bool>(_storage.template get<1>());
    }

    constexpr pointer release() noexcept {
      return zs::exchange(_storage.template get<1>(), nullptr);
    }

    constexpr void reset(pointer p = nullptr) noexcept {
      pointer old = zs::exchange(_storage.template get<1>(), p);
      if (old) _storage.template get<0>()(old);
    }

    tuple<D, pointer> _storage;
  };

}  // namespace zs