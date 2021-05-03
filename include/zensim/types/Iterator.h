#pragma once
#include <functional>
#include <iterator>
#include <limits>
#include <type_traits>

#include "zensim/meta/Sequence.h"
#include "zensim/types/Tuple.h"

namespace zs::detail {
  /// reference type helper
  template <class T> struct arrow_proxy {
  private:
    using TPlain = typename std::remove_reference<T>::type;
    T r;

  public:
    constexpr arrow_proxy(T &&r) : r(FWD(r)) {}
    constexpr TPlain *operator->() { return &r; }
  };
  /// check equal to
  template <typename T> struct has_equal_to {
  private:
    template <typename U> static std::false_type test(...);
#if 0
    /// this definition is not good
    template <typename U> static std::true_type test(
        enable_if_t<std::is_convertible_v<std::invoke_result_t<decltype(&U::equal_to), U>, bool>>
            *);
#else
    /// this method allows for template function
    template <typename U> static auto test(char) -> std::enable_if_t<
        std::is_convertible_v<decltype(std::declval<U &>().equal_to(std::declval<U>())), bool>,
        std::true_type>;
#endif
    // template <typename U> static std::true_type test(decltype(&U::equal_to));

  public:
    static constexpr bool value = decltype(test<T>(0))();
  };
  /// check increment impl
  template <typename T> struct has_increment {
  private:
    template <typename U> static std::false_type test(...);
    template <typename U> static std::true_type test(decltype(&U::increment));

  public:
    static constexpr bool value = decltype(test<T>(0))();
  };
  /// check decrement impl
  template <typename T> struct has_decrement {
  private:
    template <typename U> static std::false_type test(...);
    template <typename U> static std::true_type test(decltype(&U::decrement));

  public:
    static constexpr bool value = decltype(test<T>(0))();
  };
  /// check advance impl
  template <typename T> struct has_distance_to {
  private:
    template <typename U> static std::false_type test(...);
    template <typename U> static auto test(char)
        -> decltype(std::declval<U &>().distance_to(std::declval<U>()), std::true_type{});

  public:
    static constexpr bool value = decltype(test<T>(0))();
  };
  /// check sentinel support
  template <typename T> struct has_sentinel {
  private:
    template <typename U> static std::false_type test(...);
    template <typename U> static std::true_type test(
        decltype(&U::sentinel_type),
        enable_if_t<std::is_convertible_v<std::invoke_result_t<decltype(&U::at_end)>, bool>>);

  public:
    static constexpr bool value = decltype(test<T>(0, 0))();
  };
  /// check single pass declaration
  template <typename T> struct decl_single_pass {
  private:
    template <typename U> static std::false_type test(...);
    template <typename U> static std::true_type test(decltype(&U::single_pass_iterator));

  public:
    static constexpr bool value = decltype(test<T>(0))();
  };
  /// infer difference type
  template <typename T> struct has_difference_type {
  private:
    template <typename U> static std::false_type test(...);
    template <typename U> static std::true_type test(void_t<typename U::difference_type> *);

  public:
    static constexpr bool value = decltype(test<T>(nullptr))();
  };
  template <typename, typename = void> struct infer_difference_type {
    using type = std::ptrdiff_t;
  };
  template <typename T> struct infer_difference_type<T, void_t<typename T::difference_type>> {
    using type = typename T::difference_type;
  };
  template <typename T> struct infer_difference_type<
      T, std::enable_if_t<!has_difference_type<T>::value && has_distance_to<T>::value>> {
    using type = decltype(std::declval<T &>().distance_to(std::declval<T &>()));
  };
  template <typename T> using infer_difference_type_t = typename infer_difference_type<T>::type;
  /// check advance impl
  template <typename T> struct has_advance {
  private:
    template <typename U> static std::false_type test(...);
    template <typename U> static auto test(char)
        -> decltype(std::declval<U &>().advance(std::declval<infer_difference_type_t<U>>()),
                    std::true_type{});

  public:
    static constexpr bool value = decltype(test<T>(0))();
  };
  /// infer value type
  template <typename T, typename = void> struct infer_value_type {
    using type = remove_cvref_t<decltype(*std::declval<T &>())>;
  };
  template <typename T> struct infer_value_type<T, void_t<decltype(&T::value_type)>> {
    using type = typename T::value_type;
  };
  template <typename T> using infer_value_type_t = typename infer_value_type<T>::type;

}  // namespace zs::detail

namespace zs {

  /// standard-conforming iterator
  // https://vector-of-bool.github.io/2020/06/13/cpp20-iter-facade.html
  template <typename> struct IteratorInterface;
  /// iterator traits helper
  template <typename, typename = void> struct iterator_traits;
  template <typename Iter>
  struct iterator_traits<Iter,
                         void_t<enable_if_t<std::is_base_of_v<IteratorInterface<Iter>, Iter>>>> {
    // reference
    using reference = decltype(*std::declval<Iter &>());
    // pointer (should also infer this)
    using pointer = decltype(std::declval<Iter &>().operator->());
    // difference type
    using difference_type = typename detail::infer_difference_type<Iter>::type;
    // value type
    using value_type = typename detail::infer_value_type<Iter>::type;

    using iterator_category = conditional_t<
        detail::has_advance<Iter>::value && detail::has_distance_to<Iter>::value,
        std::random_access_iterator_tag,
        conditional_t<detail::has_decrement<Iter>::value, std::bidirectional_iterator_tag,
                      conditional_t<detail::decl_single_pass<Iter>::value, std::input_iterator_tag,
                                    std::forward_iterator_tag>>>;
  };
  template <typename Derived> struct IteratorInterface {
    /// dereference
    constexpr decltype(auto) operator*() { return self().dereference(); }

  protected:
    template <typename T, enable_if_t<std::is_reference_v<T>> = 0>
    constexpr auto getAddress(T &&v) {
      // `ref` is a true reference, and we're safe to take its address
      return std::addressof(v);
    }
    template <typename T, enable_if_t<!std::is_reference_v<T>> = 0>
    constexpr auto getAddress(T &&v) {
      // `ref` is *not* a reference. Returning its address would be the
      // address of a local. Return that thing wrapped in an arrow_proxy.
      return detail::arrow_proxy(std::move(v));
    }

  public:
    constexpr decltype(auto) operator->() { return getAddress(**this); }
    /// compare
    friend constexpr bool operator==(const Derived &left, const Derived &right) {
      if constexpr (detail::has_equal_to<Derived>::value)
        return left.equal_to(right);
      else if constexpr (detail::has_distance_to<Derived>::value)
        return left.distance_to(right) == 0;
      else
        static_assert(detail::has_distance_to<Derived>::value,
                      "Iterator equality comparator missing");
      return false;
    }
    friend constexpr bool operator!=(const Derived &left, const Derived &right) {
      static_assert(detail::has_equal_to<Derived>::value,
                    "Iterator should implement \"bool equal_to()\"");
      return !left.equal_to(right);
    }
    /// increment (forward iterator)
    constexpr Derived &operator++() {
      if constexpr (detail::has_increment<Derived>::value) {
        self().increment();
      } else {
        static_assert(detail::has_advance<Derived>::value,
                      "Iterator must provide either .advance() or .increment()");
        self() += 1;
      }
      return self();
    }
    constexpr auto operator++(int) {
      if constexpr (detail::decl_single_pass<Derived>::value) {
        return ++*this;
      } else {
        auto copy = self();
        ++*this;
        return copy;
      }
    }
    /// decrement (bidirectional iterator)
    template <typename T = Derived, enable_if_t<detail::has_decrement<T>::value> = 0>
    constexpr Derived &operator--() {
      if constexpr (detail::has_decrement<Derived>::value) {
        self().decrement();
      } else {
        static_assert(detail::has_advance<Derived>::value,
                      "Iterator must provide either .advance() or .decrement()");
        self() -= 1;
      }
      return self();
    }
    template <typename T = Derived, enable_if_t<detail::has_decrement<T>::value> = 0>
    constexpr Derived operator--(int) {
      auto copy = self();
      --*this;
      return copy;
    }
    /// advance (random access iterator)
    template <typename D, typename Self = Derived,
              enable_if_all<detail::has_advance<Self>::value,
                            std::is_convertible_v<D, detail::infer_difference_type_t<Self>>> = 0>
    friend constexpr Derived &operator+=(Derived &it, D offset) {
      it.advance(offset);
      return it;
    }
    template <typename D, typename Self = Derived,
              enable_if_all<detail::has_advance<Self>::value,
                            std::is_convertible_v<D, detail::infer_difference_type_t<Self>>> = 0>
    friend constexpr Derived &operator+(Derived it, D offset) {
      return it += offset;
    }
    template <typename D, typename Self = Derived,
              enable_if_all<detail::has_advance<Self>::value,
                            std::is_convertible_v<D, detail::infer_difference_type_t<Self>>> = 0>
    friend constexpr Derived &operator+(D offset, Derived it) {
      return it += offset;
    }
    template <typename D, typename Self = Derived,
              enable_if_all<detail::has_advance<Self>::value,
                            std::is_convertible_v<D, detail::infer_difference_type_t<Self>>> = 0>
    friend constexpr Derived &operator-(Derived it, D offset) {
      return it + (-offset);
    }
    template <typename D, typename Self = Derived,
              enable_if_all<detail::has_advance<Self>::value,
                            std::is_convertible_v<D, detail::infer_difference_type_t<Self>>> = 0>
    friend constexpr Derived &operator-=(Derived &it, D offset) {
      return it = it - offset;
    }
    template <typename D, typename Self = Derived,
              enable_if_all<detail::has_advance<Self>::value,
                            std::is_convertible_v<D, detail::infer_difference_type_t<Self>>> = 0>
    constexpr decltype(auto) operator[](D pos) {
      return *(self() + pos);
    }
    ///
    template <typename Self = Derived, enable_if_t<detail::has_distance_to<Self>::value> = 0>
    friend constexpr auto operator-(const Derived &left, const Derived &right) {
      /// the deducted type must be inside the function body or template parameters
      return static_cast<detail::infer_difference_type_t<Derived>>(right.distance_to(left));
    }
#define LEGACY_ITERATOR_OP(OP)                                                        \
  template <typename T = Derived, enable_if_t<detail::has_distance_to<T>::value> = 0> \
  friend constexpr bool operator OP(const Derived &a, const Derived &b) {             \
    return (a - b) OP 0;                                                              \
  }
    LEGACY_ITERATOR_OP(<);
    LEGACY_ITERATOR_OP(<=);
    LEGACY_ITERATOR_OP(>);
    LEGACY_ITERATOR_OP(>=);
    /// sentinel support
    template <typename T = Derived, enable_if_t<detail::has_sentinel<T>::value> = 0>
    friend constexpr bool operator==(const Derived &self, typename T::sentinel_type sentinel) {
      return self.at_end();
    }

  private:
    constexpr auto &self() noexcept { return static_cast<Derived &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const Derived &>(*this); }
  };  // namespace zs

  template <typename Iterator> struct LegacyIterator : Iterator {
    using reference = typename iterator_traits<Iterator>::reference;
    using pointer = typename iterator_traits<Iterator>::pointer;
    using difference_type = typename iterator_traits<Iterator>::difference_type;
    using value_type = typename iterator_traits<Iterator>::value_type;
    using iterator_category = typename iterator_traits<Iterator>::iterator_category;

    template <typename... Args> constexpr LegacyIterator(Args &&...args) : Iterator{FWD(args)...} {}
  };
  template <typename T, typename... Args>
  constexpr LegacyIterator<T> make_iterator(Args &&...args) {
    return LegacyIterator<T>{FWD(args)...};
  }
  template <template <typename...> class T, typename... Args>
  constexpr auto make_iterator(Args &&...args) {
    using IterT = decltype(T{FWD(args)...});
    return LegacyIterator<IterT>{T{FWD(args)...}};
  }

  namespace detail {
    template <typename IB, typename IE = IB> struct WrappedIterator {
      mutable IB beginIterator;
      mutable IE endIterator;
      constexpr IB &&begin() const &&noexcept { return std::move(beginIterator); }
      constexpr IE &&end() const &&noexcept { return std::move(endIterator); }
      constexpr IB begin() const &noexcept { return beginIterator; }
      constexpr IE end() const &noexcept { return endIterator; }
      constexpr WrappedIterator(IB &&begin, IE &&end)
          : beginIterator(std::move(begin)), endIterator(std::move(end)) {}
      constexpr WrappedIterator(const IB &begin, const IE &end)
          : beginIterator(begin), endIterator(end) {}
    };
    template <typename IB, typename IE> constexpr auto iter_range(IB &&bg, IE &&ed) {
      return WrappedIterator<IB, IE>(FWD(bg), FWD(ed));
    }
  }  // namespace detail

  /// iterator
  template <typename Data, typename StrideT> struct IndexIterator
      : IteratorInterface<IndexIterator<Data, StrideT>> {
    using T = Data;
    using DiffT = std::make_signed_t<Data>;
    static_assert(std::is_integral_v<T>, "Index type must be integral");

    constexpr IndexIterator(T base = 0, DiffT stride = 1) : base{base}, stride{stride} {}
    constexpr T dereference() const noexcept { return base; }
    template <typename Iter> constexpr bool equal_to(Iter it) const noexcept {
      return base == it.base;
    }
    constexpr void advance(DiffT offset) noexcept { base += stride; }
    template <typename Iter> constexpr DiffT distance_to(Iter &&it) const noexcept {
      return static_cast<DiffT>(it.base) - static_cast<DiffT>(base);
    }

    T base;
    DiffT stride;
  };
  template <typename Data, auto Stride>
  struct IndexIterator<Data, integral_v<decltype(Stride), Stride>>
      : IteratorInterface<IndexIterator<Data, integral_v<decltype(Stride), Stride>>> {
    using T = std::make_signed_t<Data>;
    static_assert(std::is_integral_v<T>, "Index type must be integral");
    static_assert(
        std::is_integral_v<decltype(Stride)> && std::is_convertible_v<decltype(Stride), T>,
        "Stride type must be integral and compatible with the index type");

    constexpr IndexIterator(T base = 0) : base{base} {}
    constexpr T dereference() const noexcept { return base; }
    template <typename Iter> constexpr bool equal_to(Iter it) const noexcept {
      return base == it.base;
    }
    constexpr void advance(T offset) noexcept { base += static_cast<T>(Stride); }
    template <typename Iter> constexpr T distance_to(Iter &&it) const noexcept {
      return it.base - base;
    }

    T base;
  };
  template <typename BaseT> IndexIterator(BaseT) -> IndexIterator<BaseT, integral_v<int, 1>>;
  template <typename BaseT, typename DiffT, enable_if_t<!is_integral_constant<DiffT>::value> = 0>
  IndexIterator(BaseT, DiffT) -> IndexIterator<BaseT, DiffT>;
  template <typename BaseT, auto Diff> IndexIterator(BaseT, integral_v<decltype(Diff), Diff>)
      -> IndexIterator<BaseT, integral_v<decltype(Diff), Diff>>;

  struct CounterIterator : IteratorInterface<CounterIterator> {
    using T = std::size_t;

    constexpr CounterIterator(T base = 0) : base{base} {}
    constexpr void increment() noexcept { ++base; }
    constexpr T dereference() const noexcept { return base; }
    template <typename Iter> constexpr bool equal_to(Iter it) const noexcept { return false; }

    T base;
  };

  template <typename Ts, typename Indices, typename = void> struct Collapse;
  template <typename... Tn, std::size_t... Is>
  struct Collapse<type_seq<Tn...>, index_seq<Is...>,
                  std::enable_if_t<(std::is_integral_v<Tn> && ...)>> {
    template <std::size_t I> using index_t = std::tuple_element_t<I, std::tuple<Tn...>>;
    static constexpr std::size_t dim = sizeof...(Tn);
    constexpr Collapse(Tn... ns) : ns{ns...} {}

    struct iterator : IteratorInterface<iterator> {
      constexpr iterator(wrapv<0>, const std::tuple<Tn...> &ns)
          : ns{ns}, it{(Is + 1 > 0 ? 0 : 0)...} {}
      constexpr iterator(wrapv<1>, const std::tuple<Tn...> &ns)
          : ns{ns}, it{(Is == 0 ? std::get<0>(ns) : 0)...} {}

      constexpr auto dereference() { return it; }
      constexpr bool equal_to(iterator o) const noexcept {
        return ((std::get<Is>(it) == std::get<Is>(o.it)) && ...);
      }
      template <std::size_t I> constexpr void increment_impl() noexcept {
        index_t<I> n = ++std::get<I>(it);
        if constexpr (I)
          if (n >= std::get<I>(ns)) {
            std::get<I>(it) = 0;
            increment_impl<I - 1>();
          }
      }
      constexpr void increment() noexcept { increment_impl<dim - 1>(); }

      std::tuple<Tn...> ns, it;
    };

    constexpr auto begin() const noexcept { return make_iterator<iterator>(wrapv<0>{}, ns); }
    constexpr auto end() const noexcept { return make_iterator<iterator>(wrapv<1>{}, ns); }

    std::tuple<Tn...> ns;
  };

  template <typename... Tn> Collapse(Tn...)
      -> Collapse<type_seq<Tn...>, std::index_sequence_for<Tn...>>;

  template <typename... Tn> constexpr auto ndrange(Tn &&...ns) { return Collapse{FWD(ns)...}; }
  namespace detail {
    template <typename T, std::size_t... Is> constexpr auto ndrange_impl(T n, index_seq<Is...>) {
      return Collapse{(Is + 1 ? n : n)...};
    }
  }  // namespace detail
  template <auto d> constexpr auto ndrange(decltype(d) n) {
    return detail::ndrange_impl(n, std::make_index_sequence<d>{});
  }

  // zip
  template <typename, typename, typename = void> struct zip_iterator;

  /// for serial execution
  template <typename... Iters, std::size_t... Is>
  struct zip_iterator<std::tuple<Iters...>, index_seq<Is...>,
                      std::enable_if_t<!((std::is_convertible_v<
                                          typename std::iterator_traits<Iters>::iterator_category,
                                          std::random_access_iterator_tag>)&&...)>>
      : IteratorInterface<zip_iterator<
            std::tuple<Iters...>, index_seq<Is...>,
            std::enable_if_t<!(
                (std::is_convertible_v<typename std::iterator_traits<Iters>::iterator_category,
                                       std::random_access_iterator_tag>)&&...)>>> {
    constexpr zip_iterator(Iters &&...its) : iters{std::make_tuple<Iters...>(FWD(its)...)} {}

    template <size_t Idx, class T> constexpr auto getElement(T &v) {
      if constexpr (std::is_reference<decltype(*std::get<Idx>(v))>::value) {
        return std::reference_wrapper(*std::get<Idx>(v));
      } else {
        return *std::get<Idx>(v);
      }
    }
    // constexpr auto dereference() { return std::forward_as_tuple((*std::get<Is>(iters))...); }
    constexpr auto dereference() { return std::make_tuple(getElement<Is>(iters)...); }
    constexpr bool equal_to(const zip_iterator &it) const {
      return ((std::get<Is>(iters) == std::get<Is>(it.iters)) || ...);
    }
    constexpr void increment() { ((void)(++std::get<Is>(iters)), ...); }

  protected:
    std::tuple<Iters...> iters;
  };
  /// for parallel execution
  template <typename... Iters, std::size_t... Is>
  struct zip_iterator<std::tuple<Iters...>, index_seq<Is...>,
                      std::enable_if_t<((std::is_convertible_v<
                                         typename std::iterator_traits<Iters>::iterator_category,
                                         std::random_access_iterator_tag>)&&...)>>
      : IteratorInterface<zip_iterator<
            std::tuple<Iters...>, index_seq<Is...>,
            std::enable_if_t<(
                (std::is_convertible_v<typename std::iterator_traits<Iters>::iterator_category,
                                       std::random_access_iterator_tag>)&&...)>>> {
    using difference_type
        = std::common_type_t<typename std::iterator_traits<Iters>::difference_type...>;

    constexpr zip_iterator(Iters &&...its) : iters{std::make_tuple<Iters...>(FWD(its)...)} {}

  protected:
    template <typename DerefT, enable_if_t<std::is_reference<DerefT>::value> = 0>
    constexpr auto getRef(DerefT &&deref) {
      return std::reference_wrapper(deref);
    }
    template <typename DerefT, enable_if_t<!std::is_reference<DerefT>::value> = 0>
    constexpr auto getRef(DerefT &&deref) {
      return deref;
    }

  public:
    // constexpr auto dereference() { return std::forward_as_tuple((*std::get<Is>(iters))...); }
    constexpr auto dereference() { return std::make_tuple(getRef(*std::get<Is>(iters))...); }
    constexpr bool equal_to(const zip_iterator &it) const {
      return ((std::get<Is>(iters) == std::get<Is>(it.iters)) || ...);
    }
    constexpr void advance(difference_type offset) { ((void)(std::get<Is>(iters) += offset), ...); }
    constexpr difference_type distance_to(const zip_iterator &it) const {
      return static_cast<difference_type>(std::get<sizeof...(Is) - 1>(it.iters)
                                          - std::get<sizeof...(Is) - 1>(iters));
    }

  protected:
    std::tuple<Iters...> iters;
  };

  template <typename... Iters> zip_iterator(Iters...)
      -> zip_iterator<std::tuple<Iters...>, std::index_sequence_for<Iters...>>;

  /// range
  template <typename T> constexpr auto range(T begin, T end, T increment) {
    auto actualEnd = end - ((end - begin) % increment);
    // static_assert(detail::has_distance_to<LegacyIterator<IndexIterator<T>>>::value, "WTF1???");
    // static_assert(detail::has_distance_to<IndexIterator<T>>::value, "WTF2???");
    using DiffT = std::make_signed_t<T>;
    return detail::iter_range(
        make_iterator<IndexIterator<T, DiffT>>(begin, static_cast<DiffT>(increment)),
        make_iterator<IndexIterator<T, DiffT>>(actualEnd, static_cast<DiffT>(increment)));
  }
  template <typename T> constexpr auto range(T begin, T end) {
    return range<T>(begin, end, begin < end ? 1 : -1);
  }
  template <typename T> constexpr auto range(T end) { return range<T>(0, end); }

  template <typename... Args> constexpr auto zip(Args &&...args) {
    auto begin = make_iterator<zip_iterator>(std::begin(FWD(args))...);
    auto end = make_iterator<zip_iterator>(std::end(FWD(args))...);
    return detail::iter_range(std::move(begin), std::move(end));
  }

  template <typename... Args> constexpr auto enumerate(Args &&...args) {
    auto begin = make_iterator<zip_iterator>(make_iterator<CounterIterator>((std::size_t)0),
                                             std::begin(FWD(args))...);
    auto end = make_iterator<zip_iterator>(
        make_iterator<CounterIterator>(std::numeric_limits<std::size_t>::max()),
        std::end(FWD(args))...);
    return detail::iter_range(std::move(begin), std::move(end));
  }

}  // namespace zs