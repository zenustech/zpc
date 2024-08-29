#pragma once
#include <tuple>
#include <type_traits>
#include <variant>

#include "zensim/ZpcFunctional.hpp"
#include "zensim/ZpcMeta.hpp"

namespace zs {

  /// https://github.com/SuperV1234/ndctechtown2020/blob/master/7_a_match.pdf
  template <typename... Fs> struct overload_set : Fs... {
    template <typename... Xs> constexpr overload_set(Xs &&...xs) : Fs{zs::forward<Xs>(xs)}... {}
    using Fs::operator()...;
  };
  /// class template argument deduction
  template <typename... Xs> overload_set(Xs &&...xs) -> overload_set<remove_cvref_t<Xs>...>;

  template <typename... Fs> constexpr auto make_overload_set(Fs &&...fs) {
    return overload_set<typename decay<Fs>::type...>(zs::forward<Fs>(fs)...);
  }

  template <typename... Ts> using variant = std::variant<Ts...>;

  template <typename... Fs> constexpr auto match(Fs &&...fs) {
#if 0
  return [visitor = overload_set{zs::forward<Fs>(fs)...}](
             auto &&...vs) -> decltype(auto) {
    return std::visit(visitor, zs::forward<decltype(vs)>(vs)...);
  };
#else
    return [visitor = make_overload_set(zs::forward<Fs>(fs)...)](auto &&...vs) -> decltype(auto) {
      return std::visit(visitor, zs::forward<decltype(vs)>(vs)...);
    };
#endif
  }

  template <typename> struct is_variant : false_type {};
  template <typename... Ts> struct is_variant<variant<Ts...>> : true_type {};

  template <typename Visitor> struct VariantTaskExecutor {
    Visitor visitor;

    VariantTaskExecutor() = default;
    template <typename F> constexpr VariantTaskExecutor(F &&f) : visitor{FWD(f)} {}

    template <typename Fn, typename... Args> struct CheckCallable {
    private:
      template <typename F, typename... Ts> static constexpr false_type test(...) { return {}; }
      template <typename F, typename... Ts>
      static constexpr true_type test(void_t<decltype(declval<Fn>()(declval<Args>()...))> *) {
        return {};
      }

    public:
      static constexpr bool value = test<Fn, Args...>(nullptr);
    };

    template <zs::size_t No, typename Args, size_t... Ns, size_t i, size_t... js, size_t I,
              size_t... Js>
    constexpr void traverse(bool &tagMatch, Args &args,
                            const std::array<zs::size_t, sizeof...(Ns)> &varIndices,
                            index_sequence<Ns...> dims, index_sequence<i, js...> indices,
                            index_sequence<I, Js...>) {
      if constexpr (No == 0) {
        if constexpr (CheckCallable<
                          Visitor,
                          std::variant_alternative_t<I,
                                                     remove_cvref_t<std::tuple_element_t<i, Args>>>,
                          std::variant_alternative_t<
                              Js, remove_cvref_t<std::tuple_element_t<js, Args>>>...>::value) {
          if ((varIndices[i] == I) && ((varIndices[js] == Js) && ...)) {
            tagMatch = true;
            visitor(std::get<I>(std::get<i>(args)), std::get<Js>(std::get<js>(args))...);
            // std::invoke(visitor, std::get<I>(std::get<i>(args)),
            //            std::get<Js>(std::get<js>(args))...);
            return;
          }
        }
      } else {
        traverse<No - 1>(
            tagMatch, args, varIndices, dims, indices,
            index_sequence<select_indexed_value<No - 1, Ns...>::value - 1, I, Js...>{});
        if (tagMatch) return;
      }
      if constexpr (I > 0) {  // next loop
        traverse<No>(tagMatch, args, varIndices, dims, indices, index_sequence<I - 1, Js...>{});
        if (tagMatch) return;
      }
    }

    template <typename... Args> static constexpr bool all_variant() {
      return (is_variant<remove_cvref_t<Args>>::value && ...);
    }

    template <typename... Args>
    constexpr enable_if_type<all_variant<Args...>()> operator()(Args &&...args) {
      using variant_sizes = index_sequence<std::variant_size_v<remove_cvref_t<Args>>...>;
      constexpr auto narg = sizeof...(Args);
      constexpr auto lastVariantSize
          = std::variant_size_v<select_indexed_type<narg - 1, remove_cvref_t<Args>...>>;
      auto packedArgs = std::forward_as_tuple(FWD(args)...);
      std::array<zs::size_t, narg> varIndices{(args.index())...};
      bool tagMatch{false};

      traverse<narg - 1>(tagMatch, packedArgs, varIndices, variant_sizes{},
                         index_sequence_for<Args...>{}, index_sequence<lastVariantSize - 1>{});
    }
  };
  template <typename Visitor> VariantTaskExecutor(Visitor) -> VariantTaskExecutor<Visitor>;

}  // namespace zs
