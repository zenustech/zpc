#pragma once
#include "zensim/ZpcMeta.hpp"

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

  template <class R, class... Args> struct function_ref<R(Args...)> {
    using non_member_callable_signature = R(Args...);

    using function_member_pointer_type = void (detail::function_ref_dummy::*)();
    using data_member_pointer_type = byte(detail::function_ref_dummy::*);
    using callable_type = R(const function_ref *, Args...);

    /// @note rule of three
    constexpr function_ref() noexcept = delete;

    constexpr function_ref(const function_ref &rhs) noexcept = default;
    constexpr function_ref &operator=(const function_ref &rhs) noexcept = default;

    template <typename F,
              enable_if_t<!is_same_v<decay_t<F>, function_ref> && !is_member_pointer_v<decay_t<F>>
                          && is_invocable_r_v<R, F &&, Args...>>
              = 0>
    constexpr function_ref(F &&f) noexcept : _object{nullptr}, _erasedFn{nullptr} {
      operator=(FWD(f));
    }
    template <typename F,
              enable_if_t<!is_member_pointer_v<decay_t<F>> && is_invocable_r_v<R, F &&, Args...>>
              = 0>
    constexpr function_ref &operator=(F &&f) noexcept {
      _object = const_cast<void *>(reinterpret_cast<const void *>(zs::addressof(f)));

      _erasedFn = [](const function_ref *self, Args... args) -> R {
        return zs::invoke(*reinterpret_cast<add_pointer_t<F>>(const_cast<void *>(self->_object)),
                          zs::forward<Args>(args)...);
      };
      return *this;
    }

    template <typename Pointed, class C,
              enable_if_t<zs::is_invocable_r_v<R, Pointed(C::*), Args...>> = 0>
    constexpr function_ref(Pointed C::*f) noexcept {
      operator=(f);
    }
    template <typename Pointed, class C,
              enable_if_t<zs::is_invocable_r_v<R, Pointed(C::*), Args...>> = 0>
    constexpr function_ref &operator=(Pointed C::*f) noexcept {
      using F = Pointed(C::*);
      if constexpr (is_function_v<Pointed>) {
        _funcMember = *reinterpret_cast<function_member_pointer_type *>(&f);

        _erasedFn = [](const function_ref *self, Args... args) -> R {
          return zs::invoke(*reinterpret_cast<F *>(&const_cast<function_ref *>(self)->_funcMember),
                            zs::forward<Args>(args)...);
        };
      } else {
        static_assert(is_object_v<Pointed> && sizeof...(Args) == 1, "???");
        _dataMember = *reinterpret_cast<data_member_pointer_type *>(&f);

        _erasedFn = [](const function_ref *self, Args... args) -> R {
          return zs::invoke(*reinterpret_cast<F *>(&const_cast<function_ref *>(self)->_dataMember),
                            zs::forward<Args>(args)...);
        };
      }
      return *this;
    }

    constexpr R operator()(Args... args) const {
      return _erasedFn(this, zs::forward<Args>(args)...);
    }

  private:
    union {
      void *_object{nullptr};
      function_member_pointer_type _funcMember;
      data_member_pointer_type _dataMember;
    };
    // detail::function_ref_storage _storage{};
    callable_type *_erasedFn = nullptr;
  };

}  // namespace zs
