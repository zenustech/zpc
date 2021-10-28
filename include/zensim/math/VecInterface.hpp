#pragma once
#include <tuple>

#include "MathUtils.h"
#include "zensim/meta/Meta.h"
#include "zensim/meta/Sequence.h"

namespace zs {

  template <typename, typename, typename> struct indexer_impl;

  /// indexer
  template <typename Tn, Tn... Ns, std::size_t... StorageOrders, std::size_t... Is>
  struct indexer_impl<integer_seq<Tn, Ns...>, index_seq<StorageOrders...>, index_seq<Is...>> {
    static_assert(sizeof...(Ns) == sizeof...(StorageOrders), "dimension mismatch");
    static constexpr auto dim = sizeof...(Ns);
    static constexpr auto extent = (Ns * ...);
    using index_type = Tn;
    using extents = integer_seq<Tn, Ns...>;
    using storage_orders = value_seq<select_indexed_value<Is, StorageOrders...>::value...>;
    using lookup_orders = value_seq<storage_orders::template index<wrapv<Is>>::value...>;

    using storage_extents_impl
        = integer_seq<index_type, select_indexed_value<StorageOrders, Ns...>::value...>;
    using storage_bases = value_seq<excl_suffix_mul(Is, storage_extents_impl{})...>;
    using lookup_bases = value_seq<storage_bases::template type<lookup_orders::template type<Is>::value>::value...>;

    template <std::size_t I> static constexpr index_type storage_range() noexcept {
      return select_indexed_value<storage_orders::template type<I>::value, Ns...>::value;
    }
    template <std::size_t I> static constexpr index_type range() noexcept {
      return select_indexed_value<I, Ns...>::value;
    }

    static constexpr index_type offset(std::enable_if_t<Is == Is, index_type>... is) noexcept {
      return ((is * lookup_bases::template type<Is>::value) + ...);
    }
    template <std::size_t... Js, typename... Args>
    static constexpr index_type offset_impl(index_seq<Js...>, Args &&...args) noexcept {
      return (... + ((index_type)args * lookup_bases::template type<Js>::value));
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    static constexpr index_type offset(Args &&...args) noexcept {
      return offset_impl(std::index_sequence_for<Args...>{}, FWD(args)...);
    }
  };

  template <typename Tn, Tn... Ns> using indexer
      = indexer_impl<integer_seq<Tn, Ns...>, std::make_index_sequence<sizeof...(Ns)>,
                     std::make_index_sequence<sizeof...(Ns)>>;

  template <typename Orders, typename Tn, Tn... Ns> using ordered_indexer
      = indexer_impl<integer_seq<Tn, Ns...>, Orders,
                     std::make_index_sequence<sizeof...(Ns)>>;

  template <typename Derived, typename T = typename Derived::value_type,
            typename Tn = typename Derived::index_type,
            typename Indexer = typename Derived::indexer_type>
  struct VecInterface {
    using value_type = T;
    using index_type = Tn;
    using indexer_type = Indexer;
    static constexpr int dim = indexer_type::dim;

    constexpr auto data() noexcept -> value_type* { return static_cast<Derived*>(this)->do_data(); }
    constexpr auto data() volatile noexcept -> volatile value_type* {
      return static_cast<volatile Derived*>(this)->do_data();
    }
    constexpr auto data() const noexcept -> const value_type* {
      return static_cast<const Derived*>(this)->do_data();
    }
    constexpr auto data() const volatile noexcept -> const volatile value_type* {
      return static_cast<const volatile Derived*>(this)->do_data();
    }
    /// property query
    // template <std::size_t I> static constexpr index_type range() noexcept {
    //   return select_indexed_value<I, Ns...>::value;
    // }

    /// entry access
    template <typename... Tis, enable_if_t<sizeof...(Tis) <= dim> = 0>
    constexpr decltype(auto) operator()(Tis&&... is) noexcept {
      return static_cast<Derived*>(this)->operator()(FWD(is)...);
    }
    template <typename... Tis, enable_if_t<sizeof...(Tis) <= dim> = 0>
    constexpr decltype(auto) operator()(Tis&&... is) const noexcept {
      return static_cast<const Derived*>(this)->operator()(FWD(is)...);
    }
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) operator[](Ti is) noexcept {
      return static_cast<Derived*>(this)->operator[](is);
    }
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) operator[](Ti is) const noexcept {
      return static_cast<const Derived*>(this)->operator[](is);
    }
    // tuple as index
    template <typename... Ts, enable_if_all<sizeof...(Ts) <= dim,
                                            (std::is_integral_v<remove_cvref_t<Ts>>, ...)> = 0>
    constexpr decltype(auto) operator()(const std::tuple<Ts...>& is) noexcept {
      return std::apply(static_cast<Derived&>(*this), is);
    }
    template <typename... Ts, enable_if_all<(sizeof...(Ts) <= dim),
                                            (std::is_integral_v<remove_cvref_t<Ts>>, ...)> = 0>
    constexpr decltype(auto) operator()(const std::tuple<Ts...>& is) const noexcept {
      return std::apply(static_cast<const Derived&>(*this), is);
    }
  };

}  // namespace zs