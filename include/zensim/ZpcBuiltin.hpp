#pragma once

#include "zensim/ZpcFunctional.hpp"
#include "zensim/ZpcIterator.hpp"
#include "zensim/ZpcMathUtils.hpp"
#include "zensim/ZpcReflection.hpp"
#include "zensim/ZpcTuple.hpp"
#include "zensim/math/Tensor.hpp"
#include "zensim/math/Vec.h"
//
#include "zensim/py_interop/TileVectorView.hpp"
#include "zensim/py_interop/VectorView.hpp"
//
#include "zensim/ZpcFunction.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/types/Property.h"
#include "zensim/types/SmallVector.hpp"
// #include "zensim/types/SourceLocation.hpp"
#if defined(ZS_ENABLE_OPENMP) && ZS_ENABLE_OPENMP == 1
#  include "zensim/omp/Omp.h"
#endif

namespace zs {

  template <typename Signature> struct Bind;
  /// @note bind expression predicate
  template <typename Expr> struct is_bind_expression : false_type {};
  template <typename Signature> struct is_bind_expression<Bind<Signature>> : true_type {};
  template <typename Signature> struct is_bind_expression<const Bind<Signature>> : true_type {};
  template <typename Signature> struct is_bind_expression<volatile Bind<Signature>> : true_type {};
  template <typename Signature> struct is_bind_expression<const volatile Bind<Signature>>
      : true_type {};

  template <typename Expr> constexpr bool is_bind_expression_v = is_bind_expression<Expr>::value;

  /// @note Bind
  template <typename Functor, typename... BoundArgs> struct Bind<Functor(BoundArgs...)> {
    using bound_indices = index_sequence_for<BoundArgs...>;

    template <typename BArg, typename... Args, zs::size_t... Is>
    static constexpr decltype(auto) dereference_bind_expr_argument(BArg &&arg,
                                                                   zs::tuple<Args...> &callArgs,
                                                                   zs::index_sequence<Is...>) {
      return arg(zs::get<Is>(zs::move(callArgs))...);
    }
    template <typename BArg, typename Tuple>
    static constexpr decltype(auto) dereference_argument(BArg &&arg, Tuple &callArgs) {
      using T = remove_cvref_t<BArg>;
      if constexpr (detail::is_reference_wrapper_v<T>) {
        return arg.get();
      } else if constexpr (is_bind_expression_v<T>) {
        return dereference_bind_expr_argument(
            arg, callArgs, make_index_sequence<tuple_size_v<remove_cv_t<Tuple>>>{});
      } else if constexpr (is_value_wrapper_v<T>) {
        return zs::get<(T::value - 1)>(zs::move(callArgs));  // ref: gcc11
      } else {
        return zs::forward<BArg>(arg);
      }
    }

    template <typename... Args> explicit constexpr Bind(Functor &f, Args &&...args)
        : _f{f}, _boundArgs{FWD(args)...} {}
    template <typename... Args> explicit constexpr Bind(Functor &&f, Args &&...args)
        : _f{zs::move(f)}, _boundArgs{FWD(args)...} {}
    constexpr Bind(const Bind &) = default;
    constexpr Bind(Bind &&) = default;

    template <typename... Args, zs::size_t... Is>
    constexpr decltype(auto) callImpl(zs::tuple<Args...> &&callArgs, zs::index_sequence<Is...>) {
      return zs::invoke(_f, dereference_argument(zs::get<Is>(_boundArgs), callArgs)...);
    }
    template <typename... Args, zs::size_t... Is>
    constexpr decltype(auto) callImpl(zs::tuple<Args...> &&callArgs,
                                      zs::index_sequence<Is...>) const {
      return zs::invoke(_f, dereference_argument(zs::get<Is>(_boundArgs), callArgs)...);
    }

    template <typename... Args> constexpr decltype(auto) operator()(Args &&...args) {
      return callImpl(zs::forward_as_tuple(zs::forward<Args>(args)...), bound_indices{});
    }
    template <typename... Args> constexpr decltype(auto) operator()(Args &&...args) const {
      return callImpl(zs::forward_as_tuple(zs::forward<Args>(args)...), bound_indices{});
    }

    Functor _f;
    tuple<BoundArgs...> _boundArgs;
  };

  namespace detail {
    template <typename Functor, typename... BoundArgs> struct bind_helper {
      using func_type = decay_t<Functor>;
      using bind_type = Bind<func_type(decay_t<BoundArgs>...)>;
    };
  }  // namespace detail

  template <typename Functor, typename... BArgs>
  constexpr auto bind(Functor &&functor, BArgs &&...bargs) {
    return typename detail::bind_helper<Functor, BArgs...>::bind_type(zs::forward<Functor>(functor),
                                                                      zs::forward<BArgs>(bargs)...);
  }

  template <class T, class = int> struct printf_target;

  template <class T> struct printf_target<T, enable_if_t<is_floating_point_v<T>>> {
    using type = float;
    constexpr static char placeholder[] = "%f";
  };

  template <class T> struct printf_target<T, enable_if_t<is_integral_v<T>>> {
    using type = int;
    constexpr static char placeholder[] = "%d";
  };

  template <class T> struct printf_target<
      T, enable_if_t<is_same_v<decay_t<T>, char *> || is_same_v<decay_t<T>, const char *>>> {
    using type = const char *;
    constexpr static char placeholder[] = "%s";
  };

  ZS_FUNCTION SmallString join(const SmallString &joinStr) { return SmallString{}; }

  template <class T> ZS_FUNCTION SmallString join(const SmallString &joinStr, T &&s0) { return s0; }

  template <class T, class... Types>
  ZS_FUNCTION SmallString join(const SmallString &joinStr, T &&s0, Types &&...args) {
    return s0 + joinStr + join(joinStr, FWD(args)...);
  }

  template <class... Types> ZS_FUNCTION void print_internal(Types &&...args) {
    auto formatStr = join(" ", SmallString{printf_target<remove_cvref_t<Types>>::placeholder}...);
    printf(formatStr.asChars(),
           static_cast<typename printf_target<remove_cvref_t<Types>>::type>(args)...);
  }

  template <class... Types> ZS_FUNCTION void print(Types &&...args) {
    print_internal(FWD(args)..., "\n");
  }

#ifdef ZPC_JIT_MODE
  constexpr auto tid = [
#  if defined(SYCL_LANGUAGE_VERSION)
                           &__item
#  endif
  ]() {
#  if __CUDA_ARCH__ || __MUSA_ARCH__ || __HIP_ARCH__
    return blockIdx.x * blockDim.x + threadIdx.x;
#  elif defined(_OPENMP)
    return ::omp_get_thread_num();
#  elif defined(SYCL_LANGUAGE_VERSION)
    return __item.get_linear_id();
#  else
    return 0;
#  endif
  };
#endif

}  // namespace zs
using vec2i = zs::vec<int, 2>;
using vec3i = zs::vec<int, 3>;
using vec4i = zs::vec<int, 4>;
using vec2f = zs::vec<float, 2>;
using vec3f = zs::vec<float, 3>;
using vec4f = zs::vec<float, 4>;
using vec2d = zs::vec<double, 2>;
using vec3d = zs::vec<double, 3>;
using vec4d = zs::vec<double, 4>;
using mat2i = zs::vec<int, 2, 2>;
using mat3i = zs::vec<int, 3, 3>;
using mat4i = zs::vec<int, 4, 4>;
using mat2f = zs::vec<float, 2, 2>;
using mat3f = zs::vec<float, 3, 3>;
using mat4f = zs::vec<float, 4, 4>;
using mat2d = zs::vec<double, 2, 2>;
using mat3d = zs::vec<double, 3, 3>;
using mat4d = zs::vec<double, 4, 4>;