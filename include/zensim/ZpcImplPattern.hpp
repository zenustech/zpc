#pragma once
#include "zensim/ZpcMeta.hpp"
#include "zensim/ZpcReflection.hpp"

namespace zs {

  template <template <typename> class... Skills> struct Object
      : private Skills<Object<Skills...>>... {};
  template <typename Derived, template <typename> class... Skills> struct
#if defined(ZS_COMPILER_MSVC)
      ///  ref:
      ///  https://stackoverflow.com/questions/12701469/why-is-the-empty-base-class-optimization-ebo-is-not-working-in-msvc
      ///  ref:
      ///  https://devblogs.microsoft.com/cppblog/optimizing-the-layout-of-empty-base-classes-in-vs2015-update-2-3/
      __declspec(empty_bases)
#endif
          Mixin : public Skills<Derived>... {
  };

  struct ObjectConcept {
    virtual ~ObjectConcept() = default;
  };
  // struct NodeConcept {
  // NodeConcept(NodeConcept* parent) noexcept : _parent{parent} {}
  // NodeConcept* _parent{nullptr};
  // };

#define ZS_SUPPLEMENT_IMPL_PATTERN_DERIVED_ACCESS                                                \
  constexpr auto derivedPtr() noexcept { return static_cast<Derived*>(this); }                   \
  constexpr auto derivedPtr() const noexcept { return static_cast<const Derived*>(this); }       \
  constexpr auto derivedPtr() volatile noexcept { return static_cast<volatile Derived*>(this); } \
  constexpr auto derivedPtr() const volatile noexcept {                                          \
    return static_cast<const volatile Derived*>(this);                                           \
  }

  template <typename Derived> struct Tagged {
    constexpr auto typeName() const { return get_type_str<Derived>(); }
  };

  struct ObjectVisitor;
  struct VisitableObjectConcept : ObjectConcept {
    virtual void accept(ObjectVisitor&);
  };

  template <typename Derived> struct Visitee {
    ZS_SUPPLEMENT_IMPL_PATTERN_DERIVED_ACCESS
    constexpr void accept(...) {}
    constexpr void accept(...) const {}

    template <typename Visitor> constexpr auto accept(Visitor&& visitor)
        -> decltype(FWD(visitor)(declval<Derived&>())) {
      return FWD(visitor)(*derivedPtr());
    }
    template <typename Visitor> constexpr auto accept(Visitor&& visitor) const
        -> decltype(FWD(visitor)(declval<const Derived&>())) {
      return FWD(visitor)(*derivedPtr());
    }
    template <typename Policy, typename Visitor>
    constexpr auto accept(Policy&& pol, Visitor&& visitor)
        -> decltype(FWD(visitor)(FWD(pol), declval<Derived&>())) {
      return FWD(visitor)(FWD(pol), *derivedPtr());
    }
    template <typename Policy, typename Visitor>
    constexpr auto accept(Policy&& pol, Visitor&& visitor) const
        -> decltype(FWD(visitor)(FWD(pol), declval<const Derived&>())) {
      return FWD(visitor)(FWD(pol), *derivedPtr());
    }
  };

  template <typename Derived> struct Observable {
    ZS_SUPPLEMENT_IMPL_PATTERN_DERIVED_ACCESS
    // constexpr void subscribe(...) {}
    // constexpr void unsubscribe(...) {}
  };

  template <typename Derived> struct Observer {
    ZS_SUPPLEMENT_IMPL_PATTERN_DERIVED_ACCESS
    // constexpr void onNext(...) {}
    // constexpr void onComplete(...) {}
    // constexpr void onError(...) {}
  };

  template <typename T> constexpr void destroy_at(T* p) {
    if constexpr (zs::is_array_v<T>)
      for (auto& elem : *p) (destroy_at)(addressof(elem));
    else
      p->~T();
  }
  template <typename T, typename... Args> constexpr T* construct_at(T* p, Args&&... args) {
    return ::new (static_cast<void*>(p)) T(FWD(args)...);
  }

  template <typename T, typename RefT = T*, typename = void> struct ValueOrRef {
    // T must be trivially destructible
    static constexpr size_t num_bytes = sizeof(T) > sizeof(RefT) ? sizeof(T) : sizeof(RefT);

    explicit constexpr ValueOrRef(RefT const ptr) noexcept : _isValue{false}, _destroyed{false} {
      *reinterpret_cast<RefT*>(_buffer) = ptr;
    }
#if 0
    ~ValueOrRef() { destroy(); }

    template <bool V = is_move_constructible_v<T>, enable_if_t<V> = 0>
    constexpr ValueOrRef(ValueOrRef&& o) noexcept(noexcept(construct_at(declval<ValueOrRef*>(),
                                                                        zs::move(o.get())))) {
      if (o.isValid()) {
        construct_at(pimpl(), zs::move(o.get()));
        _isValue = true;
        _destroyed = false;
        o._destroyed = true;
        return;
      }
      // _isValue actually does not matter here
      _isValue = false;
      _destroyed = true;
    }
    template <bool V = is_move_assignable_v<T>>
    constexpr enable_if_type<V, ValueOrRef&> operator=(ValueOrRef&& o) noexcept(
        noexcept(declval<T&>() = zs::move(o.get())) && noexcept(declval<ValueOrRef&>().destroy())) {
      // _isValue should not change here
      destroy();
      if (o.isValid()) {
        get() = zs::move(o.get());
        _destroyed = false;
        o._destroyed = true;
      }
      return *this;
    }

    template <bool V = is_copy_constructible_v<T>, enable_if_t<V> = 0>
    constexpr ValueOrRef(const ValueOrRef& o) noexcept(noexcept(construct_at(declval<ValueOrRef*>(),
                                                                             o.get()))) {
      if (o.isValid()) {
        construct_at(pimpl(), o.get());
        _isValue = true;
        _destroyed = false;
        return;
      }
      _isValue = false;
      _destroyed = true;
    }
    template <bool V = is_copy_assignable_v<T>>
    constexpr enable_if_type<V, ValueOrRef&> operator=(const ValueOrRef& o) noexcept(
        noexcept(declval<T&>() = o.get()) && noexcept(declval<ValueOrRef&>().destroy())) {
      destroy();
      if (o.isValid()) {
        get() = o.get();
        _destroyed = false;
      }
      return *this;
    }

    constexpr void overwrite(T* ptr) noexcept {
      destroy();
      _isValue = false;
      *reinterpret_cast<T**>(_buffer) = ptr;
      if (ptr)
        _destroyed = false;
      else
        _destroyed = true;
    }
    constexpr void overwrite(T& obj) noexcept { overwrite(&obj); }
    template <typename... Args> constexpr void overwrite(Args&&... args) {
      destroy();
      _isValue = true;
      construct_at(pimpl(), FWD(args)...);
      _destroyed = false;
    }

    constexpr T& get() { return *pimpl(); }
    constexpr const T& get() const { return *pimpl(); }
    constexpr bool holdsValue() const noexcept { return _isValue; }
    constexpr bool holdsReference() const noexcept { return !_isValue; }
    constexpr bool isValid() const noexcept { return !_destroyed; }

  protected:
    constexpr void destroy() {
      if (!_destroyed) {
        if (_isValue) destroy_at(pimpl());
        _destroyed = true;
      }
    }
    constexpr T* pimpl() {
      if (_isValue)
        return reinterpret_cast<T*>(_buffer);
      else
        return *reinterpret_cast<T**>(_buffer);
    }
    constexpr T const* pimpl() const {
      if (_isValue)
        return reinterpret_cast<T const*>(_buffer);
      else
        return *reinterpret_cast<T* const*>(_buffer);
    }
#endif
    alignas(alignof(T) > alignof(RefT) ? alignof(T) : alignof(RefT)) byte _buffer[num_bytes] = {};
    bool _isValue{false}, _destroyed{false};
  };

  template <typename T> struct ValueOrRef<T, T*> {
    // T must be trivially destructible
    static constexpr size_t num_bytes = sizeof(T) > sizeof(T*) ? sizeof(T) : sizeof(T*);

    explicit constexpr ValueOrRef(T* ptr) noexcept : _isValue{false}, _destroyed{false} {
      if (ptr)
        *reinterpret_cast<T**>(_buffer) = ptr;
      else {
        *reinterpret_cast<T**>(_buffer) = nullptr;
        _destroyed = true;
      }
    }
    explicit constexpr ValueOrRef(T& obj) noexcept : ValueOrRef{&obj} {}
    template <typename... Args> constexpr ValueOrRef(Args&&... args)
        : _isValue{true}, _destroyed{false} {
      construct_at(pimpl(), FWD(args)...);
    }
    ~ValueOrRef() { destroy(); }

    template <bool V = is_move_constructible_v<T>, enable_if_t<V> = 0>
    constexpr ValueOrRef(ValueOrRef&& o) noexcept(noexcept(construct_at(declval<ValueOrRef*>(),
                                                                        zs::move(o.get())))) {
      if (o.isValid()) {
        construct_at(pimpl(), zs::move(o.get()));
        _isValue = true;
        _destroyed = false;
        o._destroyed = true;
        return;
      }
      // _isValue actually does not matter here
      _isValue = false;
      _destroyed = true;
    }
    template <bool V = is_move_assignable_v<T>>
    constexpr enable_if_type<V, ValueOrRef&> operator=(ValueOrRef&& o) noexcept(
        noexcept(declval<T&>() = zs::move(o.get())) && noexcept(declval<ValueOrRef&>().destroy())) {
      // _isValue should not change here
      destroy();
      if (o.isValid()) {
        get() = zs::move(o.get());
        _destroyed = false;
        o._destroyed = true;
      }
      return *this;
    }

    template <bool V = is_copy_constructible_v<T>, enable_if_t<V> = 0>
    constexpr ValueOrRef(const ValueOrRef& o) noexcept(noexcept(construct_at(declval<ValueOrRef*>(),
                                                                             o.get()))) {
      if (o.isValid()) {
        construct_at(pimpl(), o.get());
        _isValue = true;
        _destroyed = false;
        return;
      }
      _isValue = false;
      _destroyed = true;
    }
    template <bool V = is_copy_assignable_v<T>>
    constexpr enable_if_type<V, ValueOrRef&> operator=(const ValueOrRef& o) noexcept(
        noexcept(declval<T&>() = o.get()) && noexcept(declval<ValueOrRef&>().destroy())) {
      destroy();
      if (o.isValid()) {
        get() = o.get();
        _destroyed = false;
      }
      return *this;
    }

    constexpr void overwrite(T* ptr) noexcept {
      destroy();
      _isValue = false;
      *reinterpret_cast<T**>(_buffer) = ptr;
      if (ptr)
        _destroyed = false;
      else
        _destroyed = true;
    }
    constexpr void overwrite(T& obj) noexcept { overwrite(&obj); }
    template <typename... Args> constexpr void overwrite(Args&&... args) {
      destroy();
      _isValue = true;
      construct_at(pimpl(), FWD(args)...);
      _destroyed = false;
    }

    constexpr T& get() { return *pimpl(); }
    constexpr const T& get() const { return *pimpl(); }
    constexpr bool holdsValue() const noexcept { return _isValue; }
    constexpr bool holdsReference() const noexcept { return !_isValue; }
    constexpr bool isValid() const noexcept { return !_destroyed; }

  protected:
    constexpr void destroy() {
      if (!_destroyed) {
        if (_isValue) destroy_at(pimpl());
        _destroyed = true;
      }
    }
    constexpr T* pimpl() {
      if (_isValue)
        return reinterpret_cast<T*>(_buffer);
      else
        return *reinterpret_cast<T**>(_buffer);
    }
    constexpr T const* pimpl() const {
      if (_isValue)
        return reinterpret_cast<T const*>(_buffer);
      else
        return *reinterpret_cast<T* const*>(_buffer);
    }
    alignas(alignof(T) > alignof(T*) ? alignof(T) : alignof(T*)) byte _buffer[num_bytes] = {};
    bool _isValue{false}, _destroyed{false};
  };

  ///
  /// @note this version is usable in kernel
  template <typename T>
  struct ValueOrRef<T, T*, enable_if_type<is_trivially_destructible_v<T>, void>> {
    // T must be trivially destructible
    static constexpr size_t num_bytes = sizeof(T) > sizeof(T*) ? sizeof(T) : sizeof(T*);

    explicit constexpr ValueOrRef(T* ptr) noexcept : _isValue{false}, _destroyed{false} {
      if (ptr)
        *reinterpret_cast<T**>(_buffer) = ptr;
      else {
        *reinterpret_cast<T**>(_buffer) = nullptr;
        _destroyed = true;
      }
    }
    explicit constexpr ValueOrRef(T& obj) noexcept : ValueOrRef{&obj} {}
    template <typename... Args> constexpr ValueOrRef(Args&&... args)
        : _isValue{true}, _destroyed{false} {
      construct_at(pimpl(), FWD(args)...);
    }
    ~ValueOrRef() noexcept = default;  /// NOTICE THIS

    template <bool V = is_move_constructible_v<T>, enable_if_t<V> = 0>
    constexpr ValueOrRef(ValueOrRef&& o) noexcept(noexcept(construct_at(declval<ValueOrRef*>(),
                                                                        zs::move(o.get())))) {
      if (o.isValid()) {
        construct_at(pimpl(), zs::move(o.get()));
        _isValue = true;
        _destroyed = false;
        o._destroyed = true;
        return;
      }
#if 0
        /// @note extended behavior
        if (o.isValid()) {
          for (size_t i = 0; i != num_bytes; ++i) _buffer[i] = o._buffer[i];
          _isValue = o._isValue;
          _destroyed = o._destroyed;
          o.destroy();  // move
        }
#endif
      // _isValue actually does not matter here
      _isValue = false;
      _destroyed = true;
    }
    template <bool V = is_move_assignable_v<T>>
    constexpr enable_if_type<V, ValueOrRef&> operator=(ValueOrRef&& o) noexcept(
        noexcept(declval<T&>() = zs::move(o.get())) && noexcept(declval<ValueOrRef&>().destroy())) {
      // _isValue should not change here
      destroy();
      if (o.isValid()) {
        get() = zs::move(o.get());
        _destroyed = false;
        o._destroyed = true;
      }
      return *this;
    }

    template <bool V = is_copy_constructible_v<T>, enable_if_t<V> = 0>
    constexpr ValueOrRef(const ValueOrRef& o) noexcept(noexcept(construct_at(declval<ValueOrRef*>(),
                                                                             o.get()))) {
      if (o.isValid()) {
        construct_at(pimpl(), o.get());
        _isValue = true;
        _destroyed = false;
        return;
      }
      _isValue = false;
      _destroyed = true;
    }
    template <bool V = is_copy_assignable_v<T>>
    constexpr enable_if_type<V, ValueOrRef&> operator=(const ValueOrRef& o) noexcept(
        noexcept(declval<T&>() = o.get()) && noexcept(declval<ValueOrRef&>().destroy())) {
      destroy();
      if (o.isValid()) {
        get() = o.get();
        _destroyed = false;
      }
      return *this;
    }

    constexpr void overwrite(T* ptr) noexcept {
      destroy();
      _isValue = false;
      *reinterpret_cast<T**>(_buffer) = ptr;
      if (ptr)
        _destroyed = false;
      else
        _destroyed = true;
    }
    constexpr void overwrite(T& obj) noexcept { overwrite(&obj); }
    template <typename... Args> constexpr void overwrite(Args&&... args) {
      destroy();
      _isValue = true;
      construct_at(pimpl(), FWD(args)...);
      _destroyed = false;
    }

    constexpr T& get() { return *pimpl(); }
    constexpr const T& get() const { return *pimpl(); }
    constexpr bool holdsValue() const noexcept { return _isValue; }
    constexpr bool holdsReference() const noexcept { return !_isValue; }
    constexpr bool isValid() const noexcept { return !_destroyed; }

  protected:
    constexpr void destroy() {
      if (!_destroyed) {
        /// @note since trivially destructible, no need for manual destruction
        // if (_isValue) destroy_at(pimpl());
        _destroyed = true;
      }
    }
    constexpr T* pimpl() {
      if (_isValue)
        return reinterpret_cast<T*>(_buffer);
      else
        return *reinterpret_cast<T**>(_buffer);
    }
    constexpr T const* pimpl() const {
      if (_isValue)
        return reinterpret_cast<T const*>(_buffer);
      else
        return *reinterpret_cast<T* const*>(_buffer);
    }
    alignas(alignof(T) > alignof(T*) ? alignof(T) : alignof(T*)) byte _buffer[num_bytes] = {};
    bool _isValue{false}, _destroyed{false};
  };

  /// @ref <c++ software design> Klaus Iglberger
  struct DynamicStorage {
    template <typename T, typename... Args>
    [[maybe_unused]] constexpr T* create(Args&&... args) const {
      return _ptr = ::new T(FWD(args)...);
    }
    template <typename T> constexpr void destroy() const noexcept { ::delete data<T>(); }

    template <typename T> constexpr T* data() noexcept {
      return const_cast<T*>(reinterpret_cast<T const*>(_ptr));
    }
    template <typename T> constexpr const T* data() const noexcept {
      return reinterpret_cast<T const*>(_ptr);
    }

  private:
    void* _ptr{nullptr};
  };

  template <size_t Capacity, size_t Alignment> struct InplaceStorage {
    template <typename T, typename... Args>
    [[maybe_unused]] constexpr T* create(Args&&... args) const {
      static_assert(sizeof(T) <= Capacity, "The given type is too large.");
      static_assert(alignof(T) <= Alignment, "The given type is misaligned.");
      T* addr = const_cast<T*>(reinterpret_cast<T const*>(_buffer));
      return construct_at(addr, FWD(args)...);
    }
    template <typename T> constexpr void destroy() const noexcept { destroy_at(data<T>()); }

    template <typename T = void> constexpr T* data() noexcept {
      return const_cast<T*>(reinterpret_cast<T const*>(_buffer));
    }
    template <typename T = void> constexpr const T* data() const noexcept {
      return reinterpret_cast<T const*>(_buffer);
    }

  private:
    alignas(Alignment) byte _buffer[Capacity] = {};
  };

  template <typename T, size_t Cap = 128>  // 128 bytes as cap
  struct DefaultStorage
      : conditional_t<sizeof(T) <= Cap, InplaceStorage<sizeof(T), alignof(T)>, DynamicStorage> {};

  template <typename Type, typename StoragePolicy = DefaultStorage<Type>> struct Owner {
    using storage_type = StoragePolicy;

    template <bool V = is_copy_constructible_v<Type>, enable_if_t<V> = 0>
    Owner(const Type& obj) noexcept(is_nothrow_copy_constructible_v<Type>) {
      _storage.template create<Type>(obj);
    }
    template <bool V = is_move_constructible_v<Type>, enable_if_t<V> = 0>
    Owner(Type&& obj) noexcept(is_nothrow_move_constructible_v<Type>) {
      _storage.template create<Type>(zs::move(obj));
    }

    template <bool V = is_move_constructible_v<Type>, enable_if_t<V> = 0>
    Owner(Owner&& o) noexcept(is_nothrow_move_constructible_v<Type>) : Owner(zs::move(o.get())) {}
    template <bool V = is_copy_constructible_v<Type>, enable_if_t<V> = 0>
    Owner(const Owner& o) noexcept(is_nothrow_copy_constructible_v<Type>) : Owner(o.get()) {}

    ~Owner() noexcept(is_nothrow_destructible_v<Type>) { _storage.destroy(); };

    template <bool V = is_copy_assignable_v<Type>, enable_if_t<V> = 0>
    Owner& operator=(const Type& obj) noexcept(is_nothrow_copy_assignable_v<Type>) {
      get() = obj;
    }
    template <bool V = is_move_assignable_v<Type>, enable_if_t<V> = 0>
    Owner& operator=(Type&& obj) noexcept(is_nothrow_move_assignable_v<Type>) {
      get() = zs::move(obj);
    }

    template <bool V = is_copy_assignable_v<Type>, enable_if_t<V> = 0>
    Owner& operator=(const Owner& obj) noexcept(is_nothrow_copy_assignable_v<Type>) {
      operator=(obj.get());
    }
    template <bool V = is_move_assignable_v<Type>, enable_if_t<V> = 0>
    Owner& operator=(Owner&& obj) noexcept(is_nothrow_move_assignable_v<Type>) {
      operator=(zs::move(obj.get()));
    }

    Type& get() { return *_storage.data(); }
    const Type& get() const { return *_storage.data(); }
    operator Type&() { return *_storage.data(); }
    operator const Type&() const { return *_storage.data(); }

  private:
    storage_type _storage;
  };

}  // namespace zs