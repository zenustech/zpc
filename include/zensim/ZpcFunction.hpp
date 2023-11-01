#pragma once
#include "zensim/ZpcImplPattern.hpp"

namespace zs {

  ///
  /// traits
  ///
  namespace detail {
    /// currently only support free functions & lambdas/functors
    /// exclude member functions
    template <typename, typename = void> struct function_traits_impl;
    // free-function
    template <typename R, typename... Args> struct function_traits_impl<R(Args...)> {
      static constexpr size_t arity = sizeof...(Args);
      using return_t = R;
      using arguments_t = type_seq<Args...>;
    };
    template <typename R, typename... Args> struct function_traits_impl<R (*)(Args...)>
        : function_traits_impl<R(Args...)> {};
    // member function
    template <typename C, typename R, typename... Args>  // function member pointer
    struct function_traits_impl<R (C::*)(Args...)> : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R>  // data member pointer
    struct function_traits_impl<R(C::*)> : function_traits_impl<R(C &)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) const> : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) volatile>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) noexcept>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) const volatile>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) const noexcept>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) volatile noexcept>
        : function_traits_impl<R(C &, Args...)> {};
    template <typename C, typename R, typename... Args>
    struct function_traits_impl<R (C::*)(Args...) const volatile noexcept>
        : function_traits_impl<R(C &, Args...)> {};
    // lambda/ functor
    template <typename Functor>
    struct function_traits_impl<Functor, void_t<decltype(&Functor::operator())>> {
    protected:
      using calltype = function_traits_impl<decltype(&Functor::operator())>;
      template <typename... Ts, size_t... Is>
      static auto extract_arguments(type_seq<Ts...>, index_sequence<Is...>)
          -> type_seq<select_type<Is + 1, typename calltype::arguments_t>...>;

    public:
      static constexpr size_t arity = calltype::arity - 1;
      using return_t = typename calltype::return_t;
      using arguments_t = decltype(extract_arguments(declval<typename calltype::arguments_t>(),
                                                     make_index_sequence<arity>{}));
    };

  }  // namespace detail
  template <typename F> using function_traits = detail::function_traits_impl<decay_t<F>>;

  // template <class R, class... Args> using function = std::function<R(Args...)>;

  template <typename F> struct recursive_lambda {
    F f;

    template <typename F_> constexpr recursive_lambda(F_ &&f) : f{FWD(f)} {}

    template <typename... Args> constexpr decltype(auto) operator()(Args &&...args) {
      return zs::invoke(f, *this, FWD(args)...);
    }
  };
  template <typename F> recursive_lambda(F &&) -> recursive_lambda<F>;

  ///
  /// function ref
  ///

  /// @ref vittorio romeo, sy brand
  template <class Signature> struct function;
  template <class Signature> struct function_ref;
  namespace detail {
    struct function_ref_dummy {
      void foo();
      byte boo;
    };

    static_assert(alignof(function_ref_dummy) == 1, "this is not anticipated...");
    static constexpr size_t function_ref_dummy_member_pointer_size
        = sizeof(void(function_ref_dummy::*)()) > sizeof(int(function_ref_dummy::*))
              ? sizeof(void (function_ref_dummy::*)())
              : sizeof(int(function_ref_dummy::*));
    static constexpr size_t function_ref_dummy_member_pointer_alignment
        = alignof(void(function_ref_dummy::*)()) > alignof(int(function_ref_dummy::*))
              ? alignof(void (function_ref_dummy::*)())
              : alignof(int(function_ref_dummy::*));
    static_assert(function_ref_dummy_member_pointer_size >= sizeof(void *),
                  "this is not anticipated...");
    static_assert(function_ref_dummy_member_pointer_alignment >= alignof(void *),
                  "this is not anticipated...");
  }  // namespace detail

  /// @note trade space for time (one less indirection due to the omission of vtable)
  /// @note currently ignoring const handling
  template <class R, class... Args> struct function<R(Args...)> {
    enum class manage_op_e { destruct = 0, clone = 1 };
    using function_storage = InplaceStorage<detail::function_ref_dummy_member_pointer_size,
                                            detail::function_ref_dummy_member_pointer_alignment>;
    using callable_type = R(const void *, Args...);
    using manager_fn = void(void *self, void *o, manage_op_e);

    template <typename Callable> struct Owner {
      Owner(Callable callable) noexcept : _callable{zs::move(callable)} {}
      ~Owner() = default;
      Owner(const Owner &) = default;
      Owner &operator=(const Owner &) = default;

      constexpr R operator()(Args... args) const {
        return zs::invoke(const_cast<Callable &>(_callable), zs::forward<Args>(args)...);
      }
      Callable _callable;
    };

    function() noexcept = default;
    ~function() {
      if (_manageFn) (*_manageFn)(_storage.data(), nullptr, manage_op_e::destruct);
    }

    template <typename F,
              enable_if_t<!is_same_v<decay_t<F>, function> && is_invocable_r_v<R, F &&, Args...>>
              = 0>
    function(F &&f) noexcept : _storage{}, _erasedFn{nullptr}, _manageFn{nullptr} {
      operator=(FWD(f));
    }
    template <typename F, enable_if_t<is_invocable_r_v<R, F &&, Args...>> = 0>
    function &operator=(F &&f) noexcept {
      if (_manageFn) (*_manageFn)(_storage.data(), nullptr, manage_op_e::destruct);

      constexpr bool fit = sizeof(f) <= function_storage::capacity
                           && alignof(decay_t<F>) <= function_storage::alignment;

      if constexpr (fit) {
        using FuncOwner = Owner<decay_t<F>>;
        _storage.template create<FuncOwner>(FWD(f));

        _erasedFn = [](const void *obj, Args... args) -> R {
          zs::invoke(*static_cast<const function_storage *>(obj)->template data<FuncOwner>(),
                     zs::forward<Args>(args)...);
        };
        _manageFn = [](void *self, void *o, manage_op_e op) {
          if (op == manage_op_e::destruct) {
            static_cast<const function_storage *>(self)->template destroy<FuncOwner>();
          } else {  // manage_op_e::clone
            if constexpr (is_copy_constructible_v<FuncOwner>) {
              const auto &me
                  = *static_cast<const function_storage *>(self)->template data<FuncOwner>();
              static_cast<const function_storage *>(o)->template create<FuncOwner>(me);
            } else {
              throw StaticException();
            }
          }
        };
      } else {
        using FuncOwner = Unique<decay_t<F>>;
        _storage.template create<FuncOwner>(FuncOwner::make(FWD(f)));

        _erasedFn = [](const void *obj, Args... args) -> R {
          zs::invoke(**static_cast<const function_storage *>(obj)->template data<FuncOwner>(),
                     zs::forward<Args>(args)...);
        };
        _manageFn = [](void *self, void *o, manage_op_e op) {
          if (op == manage_op_e::destruct) {
            static_cast<const function_storage *>(self)->template destroy<FuncOwner>();
          } else {  // manage_op_e::clone
            if constexpr (is_copy_constructible_v<decay_t<F>>) {
              const auto &me
                  = **static_cast<const function_storage *>(self)->template data<FuncOwner>();
              static_cast<const function_storage *>(o)->template create<FuncOwner>(
                  FuncOwner::make(me));
            } else {
              // currently hold functor is not copyable!
              throw StaticException();
            }
          }
        };
      }
      return *this;
    }

    function(const function &o) { operator=(o); }
    function &operator=(const function &o) {
      o._manageFn(o._storage.data(), _storage.data(), manage_op_e::clone);
      _erasedFn = o._erasedFn;
      _manageFn = o._manageFn;
      return *this;
    }

    function(function &&o) noexcept { operator=(zs::move(o)); }
    function &operator=(function &&o) noexcept {
      memcpy(_storage.data(), o._storage.data(), function_storage::capacity);
      exchange(_erasedFn, o._erasedFn, nullptr);
      exchange(_manageFn, o._manageFn, nullptr);
      return *this;
    }

    void swap(function &o) noexcept {
      function_storage tmp = o._storage;
      _storage = tmp;
      o._storage = zs::move(tmp);
      swap(_erasedFn, o._erasedFn);
      swap(_manageFn, o._manageFn);
    }

    R operator()(Args... args) const {
      if (!_erasedFn) throw StaticException();
      return _erasedFn(_storage.data(), zs::forward<Args>(args)...);
    }

  private:
    function_storage _storage{};
    callable_type *_erasedFn = nullptr;
    manager_fn *_manageFn = nullptr;
  };

  template <class R, class... Args> struct function_ref<R(Args...)> {
    using function_ref_storage
        = InplaceStorage<detail::function_ref_dummy_member_pointer_size,
                         detail::function_ref_dummy_member_pointer_alignment>;
    // using non_member_callable_signature = R(Args...);
    // using function_member_pointer_type = void (detail::function_ref_dummy::*)();
    // using data_member_pointer_type = byte(detail::function_ref_dummy::*);
    using callable_type = R(const void *, Args...);

    template <typename Handle> struct Ref {
      static_assert(true, "...");
      Ref(Handle handle) noexcept : _handle{zs::move(handle)} {}
      ~Ref() = default;
      Ref(const Ref &) = default;
      Ref &operator=(const Ref &) = default;

      constexpr R operator()(Args... args) const {
        if constexpr (is_member_pointer_v<Handle>)
          return zs::invoke(const_cast<Handle &>(_handle), zs::forward<Args>(args)...);
        else
          return zs::invoke(*const_cast<Handle &>(_handle), zs::forward<Args>(args)...);
      }
      Handle _handle;
    };

    /// @note rule of three
    constexpr function_ref() noexcept = delete;

    constexpr function_ref(const function_ref &rhs) noexcept = default;
    constexpr function_ref &operator=(const function_ref &rhs) noexcept = default;

    template <
        typename F,
        enable_if_t<!is_same_v<decay_t<F>, function_ref> && is_invocable_r_v<R, F &&, Args...>> = 0>
    constexpr function_ref(F &&f) noexcept : _storage{}, _erasedFn{nullptr} {
      operator=(FWD(f));
    }
    template <typename F, enable_if_t<is_invocable_r_v<R, F &&, Args...>> = 0>
    constexpr function_ref &operator=(F &&f) noexcept {
      if constexpr (!is_member_pointer_v<decay_t<F>>) {
        using FuncRef = Ref<add_pointer_t<F>>;
        _storage.template create<add_pointer_t<F>>(zs::addressof(f));
        _erasedFn = [](const void *obj, Args... args) -> R {
          zs::invoke(*static_cast<const function_ref_storage *>(obj)->template data<FuncRef>(),
                     zs::forward<Args>(args)...);
        };
      } else {
        using FuncRef = Ref<decay_t<F>>;
        _storage.template create<FuncRef>(FWD(f));
        _erasedFn = [](const void *obj, Args... args) -> R {
          zs::invoke(*static_cast<const function_ref_storage *>(obj)->template data<FuncRef>(),
                     zs::forward<Args>(args)...);
        };
      }
      return *this;
    }

    constexpr R operator()(Args... args) const {
      return _erasedFn(_storage.data(), zs::forward<Args>(args)...);
    }

  private:
    // union {
    //   void *_object{nullptr};
    //   function_member_pointer_type _funcMember;
    //   data_member_pointer_type _dataMember;
    // };
    function_ref_storage _storage{};
    callable_type *_erasedFn = nullptr;
  };

}  // namespace zs
