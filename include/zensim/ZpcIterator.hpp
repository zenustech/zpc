#pragma once

#include "zensim/ZpcTuple.hpp"

namespace zs::detail {
  /// reference type helper
  template <class T> struct arrow_proxy {
  private:
    using TPlain = remove_reference_t<T>;
    T r;

  public:
    constexpr arrow_proxy(T &&r) : r(FWD(r)) {}
    constexpr TPlain *operator->() { return &r; }
  };
  /// check equal to
  template <typename T> struct has_equal_to {
  private:
    template <typename U> static false_type test(...);
#if 0
    /// this definition is not good
    template <typename U> static true_type test(
        enable_if_t<is_convertible_v<invoke_result_t<decltype(&U::equal_to), U>, bool>>
            *);
#else
    /// this method allows for template function
    template <typename U> static auto test(char)
        -> enable_if_type<is_convertible_v<decltype(declval<U &>().equal_to(declval<U>())), bool>,
                          true_type>;
#endif
    // template <typename U> static true_type test(decltype(&U::equal_to));

  public:
    static constexpr bool value = decltype(test<T>(0))::value;
  };
  /// check increment impl
  template <typename T> struct has_increment {
  private:
    template <typename U> static false_type test(...);
    template <typename U> static true_type test(decltype(&U::increment));

  public:
    static constexpr bool value = decltype(test<T>(0))::value;
  };
  /// check decrement impl
  template <typename T> struct has_decrement {
  private:
    template <typename U> static false_type test(...);
    template <typename U> static true_type test(decltype(&U::decrement));

  public:
    static constexpr bool value = decltype(test<T>(0))::value;
  };
  /// check advance impl
  template <typename T> struct has_distance_to {
  private:
    template <typename U> static false_type test(...);
    template <typename U>
    static auto test(char) -> decltype(declval<U &>().distance_to(declval<U>()), true_type{});

  public:
    static constexpr bool value = decltype(test<T>(0))::value;
  };
  /// check sentinel support
  template <typename T> struct has_sentinel {
  private:
    template <typename U> static false_type test(...);
    template <typename U> static true_type test(
        decltype(&U::sentinel_type),
        enable_if_t<is_convertible_v<invoke_result_t<decltype(&U::at_end)>, bool>>);

  public:
    static constexpr bool value = decltype(test<T>(0, 0))::value;
  };
  /// check single pass declaration
  template <typename T> struct decl_single_pass {
  private:
    template <typename U> static false_type test(...);
    template <typename U> static true_type test(decltype(&U::single_pass_iterator));

  public:
    static constexpr bool value = decltype(test<T>(0))::value;
  };
  /// infer difference type
  template <typename T> struct has_difference_type {
  private:
    template <typename U> static false_type test(...);
    template <typename U> static true_type test(void_t<typename U::difference_type> *);

  public:
    static constexpr bool value = decltype(test<T>(nullptr))::value;
  };
  template <typename, typename = void> struct infer_difference_type {
    using type = sint_t;
  };
  template <typename T> struct infer_difference_type<T, void_t<typename T::difference_type>> {
    using type = typename T::difference_type;
  };
  template <typename T> struct infer_difference_type<
      T, enable_if_type<!has_difference_type<T>::value && has_distance_to<T>::value>> {
    using type = decltype(declval<T &>().distance_to(declval<T &>()));
  };
  template <typename T> using infer_difference_type_t = typename infer_difference_type<T>::type;
  /// check advance impl
  template <typename T> struct has_advance {
  private:
    template <typename U> static false_type test(...);
    template <typename U> static auto test(char)
        -> decltype(declval<U &>().advance(declval<infer_difference_type_t<U>>()), true_type{});

  public:
    static constexpr bool value = decltype(test<T>(0))::value;
  };
  /// infer value type
  template <typename T, typename = void> struct infer_value_type {
    using type = remove_cvref_t<decltype(*declval<T &>())>;
  };
  template <typename T> struct infer_value_type<T, void_t<decltype(&T::value_type)>> {
    using type = typename T::value_type;
  };
  template <typename T> using infer_value_type_t = typename infer_value_type<T>::type;

}  // namespace zs::detail

namespace zs {

  template <class C> constexpr auto begin(C &c) -> decltype(c.begin()) { return c.begin(); }
  template <class C> constexpr auto begin(const C &c) -> decltype(c.begin()) { return c.begin(); }
  template <class C>
  constexpr auto cbegin(const C &c) noexcept(noexcept(begin(c))) -> decltype(begin(c)) {
    return begin(c);
  }
  template <class C> constexpr auto end(C &c) -> decltype(c.end()) { return c.end(); }
  template <class C> constexpr auto end(const C &c) -> decltype(c.end()) { return c.end(); }
  template <class C>
  constexpr auto cend(const C &c) noexcept(noexcept(end(c))) -> decltype(end(c)) {
    return end(c);
  }

  struct input_iterator_tag {
    // template <typename T> constexpr operator T();
#if 0
    {
      static_assert(always_false<T>, "should not include this default conversion overload");
    }
#endif
  };
  struct output_iterator_tag {
    // template <typename T> constexpr operator T();
  };
  struct forward_iterator_tag : public input_iterator_tag {
    // template <typename T> constexpr operator T();
  };
  struct bidirectional_iterator_tag : public forward_iterator_tag {
    // template <typename T> constexpr operator T();
  };
  struct random_access_iterator_tag : public bidirectional_iterator_tag {
    // template <typename T> constexpr operator T();
  };
  struct contiguous_iterator_tag : public random_access_iterator_tag {
    // template <typename T> constexpr operator T();
  };

  /// standard-conforming iterator
  // https://vector-of-bool.github.io/2020/06/13/cpp20-iter-facade.html
  template <typename> struct IteratorInterface;
  /// iterator traits helper
  template <typename, typename = void> struct iterator_traits;
  template <typename Iter>
  struct iterator_traits<Iter, enable_if_type<is_base_of_v<IteratorInterface<Iter>, Iter>, void>> {
    // reference
    using reference = decltype(*declval<Iter &>());
    // pointer (should also infer this)
    using pointer = decltype(declval<Iter &>().operator->());
    // difference type
    using difference_type = typename detail::infer_difference_type<Iter>::type;
    static_assert(is_signed_v<difference_type>, "difference_type should be a signed integer.");
    // value type
    using value_type = typename detail::infer_value_type<Iter>::type;

    using iterator_category = conditional_t<
        detail::has_advance<Iter>::value && detail::has_distance_to<Iter>::value,
        random_access_iterator_tag,
        conditional_t<detail::has_decrement<Iter>::value, bidirectional_iterator_tag,
                      conditional_t<detail::decl_single_pass<Iter>::value, input_iterator_tag,
                                    forward_iterator_tag>>>;
  };

  // ref: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
  template <typename Iter, typename = void> struct is_ra_iter : false_type {};
  template <typename Iter>
  struct is_ra_iter<Iter, void_t<Iter &, typename iterator_traits<Iter>::difference_type>> {
    using diff_t = typename iterator_traits<Iter>::difference_type;
    static constexpr bool value
        = is_same_v<decltype(declval<Iter &>() += declval<diff_t>()), Iter &>
          && is_same_v<decltype(declval<Iter &>() -= declval<diff_t>()), Iter &>
          && is_same_v<decltype(declval<Iter &>() + declval<diff_t>()), Iter>
          && is_same_v<decltype(declval<diff_t>() + declval<Iter &>()), Iter>
          && is_same_v<decltype(declval<Iter &>() - declval<diff_t>()), Iter>
          && is_same_v<decltype(declval<Iter>() - declval<Iter>()), diff_t>
          && is_same_v<decltype(declval<Iter &>()[declval<diff_t>()]),
                       typename iterator_traits<Iter>::reference>;
  };
  template <typename Iter> constexpr bool is_ra_iter_v = is_ra_iter<Iter>::value;

  template <typename Derived> struct IteratorInterface {
    /// dereference
    constexpr decltype(auto) operator*() { return self().dereference(); }

  protected:
    template <typename T, enable_if_t<is_reference_v<T>> = 0> constexpr auto getAddress(T &&v) {
      // `ref` is a true reference, and we're safe to take its address
      return zs::addressof(v);
    }
    template <typename T, enable_if_t<!is_reference_v<T>> = 0> constexpr auto getAddress(T &&v) {
      // `ref` is *not* a reference. Returning its address would be the
      // address of a local. Return that thing wrapped in an arrow_proxy.
      return detail::arrow_proxy(zs::move(v));
    }

  public:
    constexpr auto operator->() { return getAddress(**this); }
    /// compare
    constexpr bool operator==(const Derived &right) const {
      if constexpr (detail::has_equal_to<Derived>::value)
        return self().equal_to(right);
      else if constexpr (detail::has_distance_to<Derived>::value)
        return self().distance_to(right) == 0;
      else
        static_assert(detail::has_distance_to<Derived>::value,
                      "Iterator equality comparator missing");
      return false;
    }
    constexpr bool operator!=(const Derived &right) const {
      static_assert(
          detail::has_equal_to<Derived>::value,
          "Iterator should implement \"bool equal_to(Iter)\" or \"Integral distance_to(Iter)\"");
      return !self().equal_to(right);
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
    template <typename T = Derived,
              enable_if_t<detail::has_decrement<T>::value || detail::has_advance<T>::value> = 0>
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
    template <typename T = Derived,
              enable_if_t<detail::has_decrement<T>::value || detail::has_advance<T>::value> = 0>
    constexpr Derived operator--(int) {
      auto copy = self();
      --*this;
      return copy;
    }
/// advance (random access iterator)
#if 1
    template <typename Self = Derived, enable_if_t<detail::has_advance<Self>::value> = 0>
    constexpr Derived &operator+=(detail::infer_difference_type_t<Self> offset) {
      static_cast<Derived *>(this)->advance(offset);
      return static_cast<Derived &>(*this);
    }
#else
    template <typename D, typename Self = Derived,
              enable_if_all<detail::has_advance<Self>::value,
                            is_convertible_v<D, detail::infer_difference_type_t<Self>>>
              = 0>
    friend constexpr Derived &operator+=(Derived &it, D offset) {
      it.advance(offset);
      return it;
    }
#endif
    template <typename Self = Derived, enable_if_t<detail::has_advance<Self>::value> = 0>
    constexpr Derived operator+(detail::infer_difference_type_t<Self> offset) const {
      auto it = self();
      return it += offset;
    }
    template <typename Self = Derived, enable_if_t<detail::has_advance<Self>::value> = 0>
    friend constexpr Derived operator+(detail::infer_difference_type_t<Self> offset, Derived it) {
      return it += offset;
    }
    template <typename Self = Derived, enable_if_t<detail::has_advance<Self>::value> = 0>
    constexpr Derived operator-(detail::infer_difference_type_t<Self> offset) const {
      return (*this) + (-offset);
    }
    template <typename Self = Derived, enable_if_t<detail::has_advance<Self>::value> = 0>
    constexpr Derived &operator-=(detail::infer_difference_type_t<Self> offset) {
      static_cast<Derived &>(*this).advance(-offset);
      return static_cast<Derived &>(*this);
    }
    template <typename Self = Derived, enable_if_t<detail::has_advance<Self>::value> = 0>
    constexpr decltype(auto) operator[](detail::infer_difference_type_t<Self> pos) {
      return *(self() + pos);
    }
    ///
    template <typename Self = Derived, enable_if_t<detail::has_distance_to<Self>::value> = 0>
    friend constexpr detail::infer_difference_type_t<Self> operator-(const Self &left,
                                                                     const Derived &right) {
      /// the deducted type must be inside the function body or template parameters
      return static_cast<detail::infer_difference_type_t<Derived>>(right.distance_to(left));
    }
#define LEGACY_ITERATOR_OP(OP)                                                        \
  template <typename T = Derived, enable_if_t<detail::has_distance_to<T>::value> = 0> \
  friend constexpr bool operator OP(const T &a, const Derived &b) {                   \
    return (a - b) OP 0;                                                              \
  }
    LEGACY_ITERATOR_OP(<)
    LEGACY_ITERATOR_OP(<=)
    LEGACY_ITERATOR_OP(>)
    LEGACY_ITERATOR_OP(>=)
    /// sentinel support
    template <typename T = Derived, enable_if_t<detail::has_sentinel<T>::value> = 0>
    friend constexpr bool operator==(const Derived &self,
                                     [[maybe_unused]] typename T::sentinel_type sentinel) {
      return self.at_end();
    }

  private:
    constexpr auto &self() noexcept { return static_cast<Derived &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const Derived &>(*this); }
  };  // namespace zs

  template <typename Iterator> struct LegacyIterator : Iterator {
    using base_t = Iterator;
    using reference = typename iterator_traits<Iterator>::reference;
    using pointer = typename iterator_traits<Iterator>::pointer;
    using difference_type = typename iterator_traits<Iterator>::difference_type;
    using value_type = typename iterator_traits<Iterator>::value_type;
    using iterator_category = typename iterator_traits<Iterator>::iterator_category;

    template <typename... Args> constexpr LegacyIterator(Args &&...args) : Iterator{FWD(args)...} {}

    /// @brief dereference
    constexpr decltype(auto) operator*() const {
      return *(*const_cast<base_t *>(static_cast<const base_t *>(this)));
    }

    /// @brief prefix-increment
    constexpr LegacyIterator &operator++() {
      ++(*static_cast<base_t *>(this));
      return *this;
    }
    /// @brief post-increment
    constexpr LegacyIterator operator++(int) {
      return LegacyIterator{(*static_cast<base_t *>(this))++};
    }
    /// @brief prefix-decrement
    constexpr LegacyIterator &operator--() {
      --(*static_cast<base_t *>(this));
      return *this;
    }
    /// @brief post-decrement
    constexpr LegacyIterator operator--(int) {
      return LegacyIterator{(*static_cast<base_t *>(this))--};
    }
    /// @brief advance (random access iterator)
    template <typename Self = base_t, enable_if_t<detail::has_advance<Self>::value> = 0>
    constexpr LegacyIterator &operator+=(detail::infer_difference_type_t<Iterator> offset) {
      static_cast<base_t *>(this)->advance(offset);
      return static_cast<LegacyIterator &>(*this);
    }
    template <typename Self = base_t, enable_if_t<detail::has_advance<Self>::value> = 0>
    constexpr LegacyIterator operator+(detail::infer_difference_type_t<Self> offset) const {
      LegacyIterator it = *this;
      return it += offset;
    }
    template <typename Self = base_t, enable_if_t<detail::has_advance<Self>::value> = 0>
    friend constexpr LegacyIterator operator+(detail::infer_difference_type_t<Iterator> offset,
                                              LegacyIterator it) {
      return it += offset;
    }
    template <typename Self = base_t, enable_if_t<detail::has_advance<Self>::value> = 0>
    constexpr LegacyIterator operator-(detail::infer_difference_type_t<Iterator> offset) const {
      return (*this) + (-(difference_type)offset);
    }
    template <typename Self = base_t, enable_if_t<detail::has_advance<Self>::value> = 0>
    constexpr LegacyIterator &operator-=(detail::infer_difference_type_t<Iterator> offset) {
      static_cast<base_t *>(this)->advance(-(difference_type)offset);
      return static_cast<LegacyIterator &>(*this);
    }
  };
  template <typename T, typename... Args>
  constexpr LegacyIterator<T> make_iterator(Args &&...args) {
    return LegacyIterator<T>(FWD(args)...);
  }
  template <template <typename...> class T, typename... Args>
  constexpr auto make_iterator(Args &&...args) {
    using IterT = decltype(T(FWD(args)...));
    return LegacyIterator<IterT>(T(FWD(args)...));
  }

  template <typename Iter> struct is_custom_iterator : false_type {};
  template <typename Iter> struct is_custom_iterator<LegacyIterator<Iter>> : true_type {};

  template <typename Iter> constexpr bool is_custom_iterator_v = is_custom_iterator<Iter>::value;

  namespace detail {
    template <typename IB, typename IE = IB> struct WrappedIterator {
      IB beginIterator;
      IE endIterator;
      constexpr IB begin() const noexcept { return beginIterator; }
      constexpr IE end() const noexcept { return endIterator; }
      constexpr WrappedIterator(IB &&begin, IE &&end)
          : beginIterator(zs::move(begin)), endIterator(zs::move(end)) {}
      constexpr WrappedIterator(const IB &begin, const IE &end)
          : beginIterator(begin), endIterator(end) {}
    };
    template <typename IB, typename IE> constexpr auto iter_range(IB &&bg, IE &&ed) {
      return WrappedIterator<remove_cvref_t<IB>, remove_cvref_t<IE>>(FWD(bg), FWD(ed));
    }
  }  // namespace detail

  ///
  /// iterator
  ///
  // index iterator
  template <typename Data, typename StrideT> struct IndexIterator
      : IteratorInterface<IndexIterator<Data, StrideT>> {
    using T = Data;
    using DiffT
        = conditional_t<sizeof(StrideT) <= 1, i8,
                        conditional_t<sizeof(StrideT) <= 2, i16,
                                      conditional_t<sizeof(StrideT) <= 4, i32,
                                                    sint_t>>>;  // zs::make_signed_t<StrideT>;
    static_assert(is_integral_v<T> && is_integral_v<DiffT>, "Index type must be integral");
    static_assert(is_convertible_v<T, DiffT>, "Stride type not compatible with the index type");

    constexpr IndexIterator(T base = 0, DiffT stride = 1) : base{base}, stride{stride} {}
    constexpr T dereference() const noexcept { return base; }
    constexpr bool equal_to(IndexIterator it) const noexcept { return base == it.base; }
    constexpr void advance(DiffT offset) noexcept { base += stride * offset; }
    constexpr DiffT distance_to(IndexIterator it) const noexcept {
      return static_cast<DiffT>(it.base) - static_cast<DiffT>(base);
    }

    T base;
    DiffT stride;
  };
  template <typename Data, sint_t Stride> struct IndexIterator<Data, integral<sint_t, Stride>>
      : IteratorInterface<IndexIterator<Data, integral<sint_t, Stride>>> {
    // using T = zs::make_signed_t<Data>;
    using T = conditional_t<
        sizeof(Data) <= 1, i8,
        conditional_t<sizeof(Data) <= 2, i16, conditional_t<sizeof(Data) <= 4, i32, sint_t>>>;
    static_assert(is_integral_v<T>, "Index type must be integral");
    static_assert(is_convertible_v<sint_t, T>, "Stride type not compatible with the index type");

    constexpr IndexIterator(T base = 0, wrapv<Stride> = {}) : base{base} {}

    constexpr T dereference() const noexcept { return base; }
    constexpr bool equal_to(IndexIterator it) const noexcept { return base == it.base; }
    constexpr void advance(T offset) noexcept { base += static_cast<T>(Stride * offset); }
    constexpr T distance_to(IndexIterator it) const noexcept { return it.base - base; }

    T base;
  };
  template <typename BaseT, typename DiffT, enable_if_t<!is_integral_constant<DiffT>::value> = 0>
  IndexIterator(BaseT, DiffT) -> IndexIterator<BaseT, DiffT>;
  template <typename BaseT>
  IndexIterator(BaseT) -> IndexIterator<BaseT, integral<sint_t, (sint_t)1>>;
  template <typename BaseT, sint_t Diff>
  IndexIterator(BaseT, integral<sint_t, Diff>) -> IndexIterator<BaseT, integral<sint_t, Diff>>;

  // pointer iterator
  template <typename Data> struct PointerIterator : IteratorInterface<PointerIterator<Data>> {
    using T = Data;
    using DiffT = sint_t;

    constexpr PointerIterator(T *base = nullptr) : base{base} {}
    constexpr decltype(auto) dereference() noexcept { return *base; }
    constexpr bool equal_to(PointerIterator it) const noexcept { return base == it.base; }
    constexpr void advance(DiffT offset) noexcept { base += offset; }
    constexpr DiffT distance_to(PointerIterator it) const noexcept { return it.base - base; }

    T *base;
  };
  template <typename BaseT> PointerIterator(BaseT *) -> PointerIterator<BaseT>;

  // collapse iterator
  template <typename Ts, typename Indices> struct Collapse;

  template <typename... Tn> constexpr bool all_integral() { return (is_integral_v<Tn> && ...); }
  template <typename... Tn, size_t... Is> struct Collapse<type_seq<Tn...>, index_sequence<Is...>> {
    static_assert(all_integral<Tn...>(), "not all types in Collapse is integral!");
    template <zs::size_t I> using index_t = zs::tuple_element_t<I, zs::tuple<Tn...>>;
    static constexpr size_t dim = sizeof...(Tn);
    constexpr Collapse(Tn... ns) : ns{ns...} {}
    template <typename VecT, enable_if_t<is_integral_v<typename VecT::value_type>> = 0>
    constexpr Collapse(const VecInterface<VecT> &v) : ns{v.to_tuple()} {}

    template <zs::size_t I = 0> constexpr auto get(wrapv<I> = {}) const noexcept {
      return zs::get<I>(ns);
    }

    struct iterator : IteratorInterface<iterator> {
      constexpr iterator(wrapv<0>, const zs::tuple<Tn...> &ns)
          : ns{ns}, it{(Is + 1 > 0 ? 0 : 0)...} {}
      constexpr iterator(wrapv<1>, const zs::tuple<Tn...> &ns)
          : ns{ns}, it{(Is == 0 ? zs::get<0>(ns) : 0)...} {}

      constexpr auto dereference() { return it; }
      constexpr bool equal_to(iterator o) const noexcept {
        return ((zs::get<Is>(it) == zs::get<Is>(o.it)) && ...);
      }
      template <zs::size_t I> constexpr void increment_impl() noexcept {
        index_t<I> n = ++zs::get<I>(it);
        if constexpr (I > 0)
          if (n >= zs::get<I>(ns)) {
            zs::get<I>(it) = 0;
            increment_impl<I - 1>();
          }
      }
      constexpr void increment() noexcept { increment_impl<dim - 1>(); }

      zs::tuple<Tn...> ns, it;
    };

    constexpr auto begin() const noexcept { return make_iterator<iterator>(wrapv<0>{}, ns); }
    constexpr auto end() const noexcept { return make_iterator<iterator>(wrapv<1>{}, ns); }

    zs::tuple<Tn...> ns;
  };

  template <typename Tn, int dim> using collapse_t
      = Collapse<typename build_seq<dim>::template uniform_types_t<type_seq, int>,
                 make_index_sequence<dim>>;

  template <typename VecT, enable_if_t<is_integral_v<typename VecT::value_type>> = 0>
  Collapse(const VecInterface<VecT> &)
      -> Collapse<typename build_seq<VecT::extent>::template uniform_types_t<
                      type_seq, typename VecT::value_type>,
                  make_index_sequence<VecT::extent>>;
  template <typename... Tn, enable_if_all<is_integral_v<Tn>...> = 0>
  Collapse(Tn...) -> Collapse<type_seq<Tn...>, index_sequence_for<Tn...>>;

  template <typename... Tn> constexpr auto ndrange(Tn... ns) { return Collapse{ns...}; }
  namespace detail {
    template <typename T, size_t... Is> constexpr auto ndrange_impl(T n, index_sequence<Is...>) {
      return Collapse{(Is + 1 ? n : n)...};
    }
  }  // namespace detail
  template <auto d> constexpr auto ndrange(decltype(d) n) {
    return detail::ndrange_impl(n, make_index_sequence<d>{});
  }

  // zip iterator
  template <typename, typename> struct zip_iterator;
  template <typename... Ts, enable_if_all<is_ra_iter_v<Ts>...> = 0>
  static true_type all_convertible_to_raiter(int);
  static false_type all_convertible_to_raiter(...);

  template <typename... Iters, size_t... Is>
  struct zip_iterator<zs::tuple<Iters...>, index_sequence<Is...>>
      : IteratorInterface<zip_iterator<zs::tuple<Iters...>, index_sequence<Is...>>> {
    static constexpr bool all_random_access_iter
        = decltype(all_convertible_to_raiter<Iters...>(0))::value;
    using difference_type = common_type_t<typename iterator_traits<Iters>::difference_type...>;

    zip_iterator() = default;
    constexpr zip_iterator(Iters &&...its) : iters{zs::make_tuple<Iters...>(FWD(its)...)} {}

    template <typename DerefT, enable_if_t<is_reference_v<DerefT>> = 0>
    constexpr auto getRef(DerefT &&deref) {
      return zs::ref(deref);
    }
    template <typename DerefT, enable_if_t<!is_reference_v<DerefT>> = 0>
    constexpr decltype(auto) getRef(DerefT &&deref) {
      return FWD(deref);
    }
    constexpr auto dereference() { return zs::make_tuple(getRef(*zs::get<Is>(iters))...); }

    constexpr bool equal_to(const zip_iterator &it) const {
      return ((zs::get<Is>(iters) == zs::get<Is>(it.iters)) || ...);
    }
    template <bool Cond = !all_random_access_iter, enable_if_t<Cond> = 0>
    constexpr void increment() {
      ((++zs::get<Is>(iters)), ...);
    }
    template <bool Cond = all_random_access_iter, enable_if_t<Cond> = 0>
    constexpr void advance(difference_type offset) {
      ((zs::get<Is>(iters) += offset), ...);
    }
    template <bool Cond = all_random_access_iter, enable_if_t<Cond> = 0>
    constexpr difference_type distance_to(const zip_iterator &it) const {
      difference_type dist = detail::deduce_numeric_max<difference_type>();
      ((dist = dist < (zs::get<Is>(it.iters) - zs::get<Is>(iters))
                   ? dist
                   : (zs::get<Is>(it.iters) - zs::get<Is>(iters))),
       ...);
      return dist;
    }

    zs::tuple<Iters...> iters;
  };

  template <typename... Iters>
  zip_iterator(Iters...) -> zip_iterator<zs::tuple<Iters...>, index_sequence_for<Iters...>>;

  template <typename Iter> struct is_zip_iterator : false_type {};
  template <typename Iter, typename Indices>
  struct is_zip_iterator<LegacyIterator<zip_iterator<Iter, Indices>>> : true_type {};

  template <typename Iter, typename Indices> struct is_zip_iterator<zip_iterator<Iter, Indices>>
      : true_type {};

  template <typename Iter> constexpr bool is_zip_iterator_v = is_zip_iterator<Iter>::value;

  ///
  /// ranges
  ///
  // index range
  template <typename T, enable_if_t<is_integral_v<T>> = 0>
  constexpr auto range(T begin, T end, T increment) {
    auto actualEnd = end - ((end - begin) % increment);
    // using DiffT = zs::make_signed_t<T>;
    using DiffT = conditional_t<
        sizeof(T) <= 1, i8,
        conditional_t<sizeof(T) <= 2, i16, conditional_t<sizeof(T) <= 4, i32, sint_t>>>;
    return detail::iter_range(
        make_iterator<IndexIterator>(begin, static_cast<DiffT>(increment)),
        make_iterator<IndexIterator>(actualEnd, static_cast<DiffT>(increment)));
  }
  template <typename T, enable_if_t<is_integral_v<T>> = 0> constexpr auto range(T begin, T end) {
    // using DiffT = zs::make_signed_t<T>;
    using DiffT = conditional_t<
        sizeof(T) <= 1, i8,
        conditional_t<sizeof(T) <= 2, i16, conditional_t<sizeof(T) <= 4, i32, sint_t>>>;
    return range<DiffT>(begin, end, (DiffT)(begin < end ? 1 : -1));
  }
  template <typename T, enable_if_t<is_integral_v<T>> = 0> constexpr auto range(T end) {
    return range((T)0, end);
  }

  // pointer range
  template <typename T> constexpr auto range(T *begin, T *end) {
    return detail::iter_range(make_iterator<PointerIterator>(begin),
                              make_iterator<PointerIterator>(end));
  }
  template <typename T, typename Ti, enable_if_t<is_integral_v<Ti>> = 0>
  constexpr auto range(T *st, Ti n) {
    return detail::iter_range(make_iterator<PointerIterator>(st),
                              make_iterator<PointerIterator>(st + n));
  }

  // container
  template <typename IB, typename IE, typename... Args>
  constexpr decltype(auto) range(const detail::WrappedIterator<IB, IE> &r, Args &&...args) {
    return r;
  }
  template <typename IB, typename IE, typename... Args>
  constexpr decltype(auto) range(detail::WrappedIterator<IB, IE> &&r, Args &&...args) {
    return zs::move(r);
  }

  template <typename Container, typename... Args>
  constexpr auto range(Container &&container, Args &&...args)
      -> decltype(detail::iter_range(container.begin(args...), container.end(args...))) {
    return detail::iter_range(container.begin(args...), container.end(args...));
  }

  template <typename Iter>
  constexpr auto range(const LegacyIterator<Iter> &st, const LegacyIterator<Iter> &ed) {
    return detail::iter_range(st, ed);
  }

  // zip range
  template <typename... Args> constexpr auto zip(Args &&...args) {
    auto bg = make_iterator<zip_iterator>(zs::begin(FWD(args))...);
    auto ed = make_iterator<zip_iterator>(zs::end(FWD(args))...);
    return detail::iter_range(zs::move(bg), zs::move(ed));
  }

  // enumeration range
  template <typename... Args> constexpr auto enumerate(Args &&...args) {
    auto bg = make_iterator<zip_iterator>(make_iterator<IndexIterator>((sint_t)0),
                                          zs::begin(FWD(args))...);
    auto ed = make_iterator<zip_iterator>(
        make_iterator<IndexIterator>(detail::deduce_numeric_max<sint_t>()), zs::end(FWD(args))...);
    return detail::iter_range(zs::move(bg), zs::move(ed));
  }

  // helper
  template <typename Range>
  constexpr auto range_size(const Range &range) -> decltype(zs::end(range) - zs::begin(range)) {
    return zs::end(range) - zs::begin(range);
  }

  template <typename RaRange, typename Tn> struct chunk_view {
    constexpr chunk_view(RaRange &&r, Tn n) noexcept : _originalRange(r), _chunkSize{n} {}
    constexpr chunk_view(RaRange &r, Tn n) noexcept : _originalRange(r), _chunkSize{n} {}
    using RaIter = decltype(declval<RaRange &>().begin());
    using ConstRaIter = decltype(declval<add_const_t<RaRange> &>().begin());
    using difference_type = typename RaIter::difference_type;

    template <bool IsConst> struct iterator_impl : IteratorInterface<iterator_impl<IsConst>> {
      using Iter = conditional_t<IsConst, ConstRaIter, RaIter>;
      constexpr iterator_impl(Iter iter, Tn chunkNo, Tn chunkSize, Tn originalSize)
          : _originalIter{iter},
            _originalSize{originalSize},
            _chunkNo{chunkNo},
            _chunkSize{chunkSize} {}

      /// @note also a (sub)range
      constexpr auto dereference() {
        Tn st = _chunkNo * _chunkSize;
        Tn ed = st + _chunkSize;
        if (ed > _originalSize) ed = _originalSize;
        return detail::iter_range(_originalIter + st, _originalIter + ed);
      }
      constexpr bool equal_to(iterator_impl it) const noexcept {
        return it._chunkNo == it._chunkNo;
      }
      constexpr void advance(difference_type offset) noexcept { _chunkNo += offset; }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return it._chunkNo - _chunkNo;
      }

    protected:
      Iter _originalIter;
      difference_type _originalSize;
      Tn _chunkNo, _chunkSize;
    };

    constexpr auto begin() noexcept {
      static_assert(is_ra_iter<decltype(_originalRange.begin())>::value,
                    "currently chunk_view only accepts a random access range.");
      return make_iterator<iterator_impl<false>>(_originalRange.begin(), 0, _chunkSize,
                                                 range_size(_originalRange));
    }
    constexpr auto end() noexcept {
      auto numRequiredChunks = (range_size(_originalRange) + _chunkSize - 1) / _chunkSize;
      return make_iterator<iterator_impl<false>>(_originalRange.begin(), numRequiredChunks,
                                                 _chunkSize, range_size(_originalRange));
    }
    constexpr auto begin() const noexcept {
      static_assert(is_ra_iter<decltype(_originalRange.begin())>::value,
                    "currently chunk_view only accepts a random access range.");
      return make_iterator<iterator_impl<true>>(_originalRange.begin(), 0, _chunkSize,
                                                range_size(_originalRange));
    }
    constexpr auto end() const noexcept {
      auto numRequiredChunks = (range_size(_originalRange) + _chunkSize - 1) / _chunkSize;
      return make_iterator<iterator_impl<true>>(_originalRange.begin(), numRequiredChunks,
                                                _chunkSize, range_size(_originalRange));
    }

    /// data members
    RaRange &_originalRange;
    Tn _chunkSize;
  };
  template <typename RaRange, typename Tn>
  chunk_view(RaRange &&, Tn) -> chunk_view<remove_reference_t<RaRange>, Tn>;

}  // namespace zs

#if ZS_ENABLE_SERIALIZATION
namespace bitsery {
  namespace traits {
    template <typename T> struct ContainerTraits;
    template <typename T> struct BufferAdapterTraits;

    template <typename IterT> struct ContainerTraits<zs::detail::WrappedIterator<IterT, IterT>> {
      static_assert(zs::detail::has_advance<IterT>::value
                        && zs::detail::has_distance_to<IterT>::value,
                    "must use a random-access-iterator range to assume contiguous for now...");
      using TValue = typename zs::iterator_traits<IterT>::value_type;
      static constexpr bool isResizable = false;
      static constexpr bool isContiguous = true;  // may not be true
      static zs::size_t size(const zs::detail::WrappedIterator<IterT, IterT> &container) {
        return zs::range_size(container);
      }
    };
    template <typename IterT>
    struct BufferAdapterTraits<zs::detail::WrappedIterator<IterT, IterT>> {
      using TIterator = IterT;
      using TConstIterator = IterT;
      using TValue = typename ContainerTraits<zs::detail::WrappedIterator<IterT, IterT>>::TValue;
    };
  }  // namespace traits
}  // namespace bitsery
#endif