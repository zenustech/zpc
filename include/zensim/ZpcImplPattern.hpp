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

  /// @ref from cppreference
  enum class byte : unsigned char {};
  template <typename IntT>
  constexpr enable_if_type<is_integral_v<IntT>, IntT> to_integer(byte b) noexcept {
    return IntT(b);
  }
  template <typename IntT>
  constexpr enable_if_type<is_integral_v<IntT>, byte&> operator<<=(byte& b, IntT shift) noexcept {
    return b = b << shift;
  }
  template <typename IntT>
  constexpr enable_if_type<is_integral_v<IntT>, byte&> operator>>=(byte& b, IntT shift) noexcept {
    return b = b >> shift;
  }
  template <typename IntT>
  constexpr enable_if_type<is_integral_v<IntT>, byte> operator<<(byte b, IntT shift) noexcept {
    // cpp17 relaxed enum class initialization rules
    return byte(static_cast<unsigned int>(b) << shift);
  }
  template <typename IntT>
  constexpr enable_if_type<is_integral_v<IntT>, byte> operator>>(byte b, IntT shift) noexcept {
    return byte(static_cast<unsigned int>(b) >> shift);
  }
  constexpr byte operator|(byte l, byte r) noexcept {
    return byte(static_cast<unsigned int>(l) | static_cast<unsigned int>(r));
  }
  constexpr byte operator&(byte l, byte r) noexcept {
    return byte(static_cast<unsigned int>(l) & static_cast<unsigned int>(r));
  }
  constexpr byte operator^(byte l, byte r) noexcept {
    return byte(static_cast<unsigned int>(l) ^ static_cast<unsigned int>(r));
  }
  constexpr byte operator~(byte b) noexcept { return byte(~static_cast<unsigned int>(b)); }
  constexpr byte& operator|=(byte& l, byte r) noexcept { return l = l | r; }
  constexpr byte& operator&=(byte& l, byte r) noexcept { return l = l & r; }
  constexpr byte& operator^=(byte& l, byte r) noexcept { return l = l ^ r; }

  template <typename T> constexpr void destroy_at(T* p) {
    if constexpr (zs::is_array_v<T>)
      for (auto& elem : *p) (destroy_at)(addressof(elem));
    else
      p->~T();
  }
  template <typename T, typename... Args> constexpr T* construct_at(T* p, Args&&... args) {
    return ::new (static_cast<void*>(p)) T(FWD(args)...);
  }

  template <typename T, typename = void> struct ValueOrRef {
    // T must be trivially destructible
    static constexpr size_t num_bytes = sizeof(T) > sizeof(T*) ? sizeof(T) : sizeof(T*);

    explicit constexpr ValueOrRef(T* ptr) noexcept : _isValue{false}, _destroyed{false} {
      *reinterpret_cast<T**>(_buffer) = ptr;
    }
    explicit constexpr ValueOrRef(T& obj) noexcept : ValueOrRef{&obj} {}
    template <typename... Args> constexpr ValueOrRef(Args&&... args)
        : _isValue{true}, _destroyed{false} {
      construct_at(pimpl(), FWD(args)...);
    }
    ~ValueOrRef() { destroy(); }

    constexpr ValueOrRef(ValueOrRef&& o) {
      for (size_t i = 0; i != num_bytes; ++i) _buffer[i] = o._buffer[i];
      _isValue = o._isValue;
      _destroyed = o._destroyed;
      o.destroy();  // move
    }
    constexpr ValueOrRef& operator=(ValueOrRef&& o) {
      destroy();  // assignment
      for (size_t i = 0; i != num_bytes; ++i) _buffer[i] = o._buffer[i];
      _isValue = o._isValue;
      _destroyed = o._destroyed;
      o.destroy();
    }
    constexpr ValueOrRef(const ValueOrRef& o) {
      for (size_t i = 0; i != num_bytes; ++i) _buffer[i] = o._buffer[i];
      _isValue = o._isValue;
      _destroyed = o._destroyed;
    }
    constexpr ValueOrRef& operator=(const ValueOrRef& o) {
      destroy();
      for (size_t i = 0; i != num_bytes; ++i) _buffer[i] = o._buffer[i];
      _isValue = o._isValue;
      _destroyed = o._destroyed;
    }

    constexpr T& get() { return *pimpl(); }
    constexpr const T& get() const { return *pimpl(); }
    constexpr bool holdsValue() const noexcept { return _isValue; }
    constexpr bool holdsReference() const noexcept { return !_isValue; }

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

  template <typename T> struct ValueOrRef<T, enable_if_type<is_trivially_destructible_v<T>, void>> {
    // T must be trivially destructible
    static constexpr size_t num_bytes = sizeof(T) > sizeof(T*) ? sizeof(T) : sizeof(T*);

    explicit constexpr ValueOrRef(T* ptr) noexcept : _isValue{false}, _destroyed{false} {
      *reinterpret_cast<T**>(_buffer) = ptr;
    }
    explicit constexpr ValueOrRef(T& obj) noexcept : ValueOrRef{&obj} {}
    template <typename... Args> constexpr ValueOrRef(Args&&... args)
        : _isValue{true}, _destroyed{false} {
      construct_at(pimpl(), FWD(args)...);
    }
    ~ValueOrRef() noexcept = default;  /// NOTICE THIS
    constexpr ValueOrRef(ValueOrRef&& o) {
      for (size_t i = 0; i != num_bytes; ++i) _buffer[i] = o._buffer[i];
      _isValue = o._isValue;
      _destroyed = o._destroyed;
      o.destroy();  // move
    }
    constexpr ValueOrRef& operator=(ValueOrRef&& o) {
      destroy();  // assignment
      for (size_t i = 0; i != num_bytes; ++i) _buffer[i] = o._buffer[i];
      _isValue = o._isValue;
      _destroyed = o._destroyed;
      o.destroy();
    }
    constexpr ValueOrRef(const ValueOrRef& o) {
      for (size_t i = 0; i != num_bytes; ++i) _buffer[i] = o._buffer[i];
      _isValue = o._isValue;
      _destroyed = o._destroyed;
    }
    constexpr ValueOrRef& operator=(const ValueOrRef& o) {
      destroy();
      for (size_t i = 0; i != num_bytes; ++i) _buffer[i] = o._buffer[i];
      _isValue = o._isValue;
      _destroyed = o._destroyed;
    }

    constexpr T& get() { return *pimpl(); }
    constexpr const T& get() const { return *pimpl(); }
    constexpr bool holdsValue() const noexcept { return _isValue; }
    constexpr bool holdsReference() const noexcept { return !_isValue; }
    constexpr bool isValid() const noexcept { return _destroyed; }

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

}  // namespace zs