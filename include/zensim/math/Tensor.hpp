#pragma once

#include "zensim/math/VecInterface.hpp"

namespace zs {

  template <typename Tensor, typename Extents> struct tensor_view;
  template <typename T, typename Extents, typename Indexer> struct tensor_impl;

  template <typename Tensor, typename Tn, Tn... Ns>
  struct tensor_view<Tensor, integer_sequence<Tn, Ns...>>
      : VecInterface<tensor_view<Tensor, integer_sequence<Tn, Ns...>>> {
    using base_t = VecInterface<tensor_view<Tensor, integer_sequence<Tn, Ns...>>>;
    static constexpr bool is_const_structure = is_const_v<Tensor>;
    using tensor_type = remove_const_t<Tensor>;
    using primitive_type = typename tensor_type::primitive_type;
    using value_type = typename tensor_type::value_type;
    using extents = integer_sequence<Tn, Ns...>;
    using index_type = typename vseq_t<extents>::value_type;
    using indexer_type = typename tensor_type::indexer_type;

    SUPPLEMENT_VEC_STATIC_ATTRIBUTES

    static constexpr auto prefix_dim = tensor_type::dim - dim;

    using prefix_type = typename build_seq<prefix_dim>::template uniform_types_t<tuple, index_type>;
    using base_type = typename build_seq<dim>::template uniform_types_t<tuple, index_type>;

    template <typename OtherT, typename ExtentsT> using variant_vec =
        typename tensor_type::template variant_vec<OtherT, ExtentsT>;

    constexpr tensor_view() noexcept = default;
    ~tensor_view() = default;
    constexpr tensor_view(Tensor &tensor, extents) noexcept
        : _tensorPtr{&tensor},
          _prefix{make_uniform_tuple<prefix_dim>(0)},
          _base{make_uniform_tuple<dim>(0)} {}
    template <typename... Ps, typename... Bs>
    constexpr tensor_view(Tensor &tensor, extents, const tuple<Ps...> &prefix,
                          const tuple<Bs...> &base) noexcept
        : _tensorPtr{&tensor},
          _prefix{make_from_tuple<prefix_type>(prefix)},
          _base{make_from_tuple<base_type>(base)} {}

    /// helper
    using indices = make_index_sequence<dim>;
    template <typename... Args, size_t... Is, enable_if_t<dim == sizeof...(Args)> = 0>
    constexpr auto getTensorCoord(tuple<Args...> c, index_sequence<Is...>) const noexcept {
      return tuple_cat(_prefix, make_tuple((get<Is>(_base) + get<Is>(c))...));
    }

    /// random access
    // ()
    template <typename... Args,
              enable_if_t<sizeof...(Args) <= dim && (is_integral_v<remove_cvref_t<Args>> && ...)>
              = 0>
    constexpr decltype(auto) operator()(Args &&...args) noexcept {
      return _tensorPtr->val(getTensorCoord(forward_as_tuple(FWD(args)...), indices{}));
    }
    template <typename... Args,
              enable_if_t<sizeof...(Args) <= dim && (is_integral_v<remove_cvref_t<Args>> && ...)>
              = 0>
    constexpr decltype(auto) operator()(Args &&...args) const noexcept {
      return _tensorPtr->val(getTensorCoord(forward_as_tuple(FWD(args)...), indices{}));
    }
    // []
    template <typename Index, enable_if_t<is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) noexcept {
      if constexpr (dim == 1)
        return _tensorPtr->val(getTensorCoord(make_tuple(index), indices{}));
      else
        return tensor_view<Tensor,
                           gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>>{
            *_tensorPtr, gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>{},
            tuple_cat(_prefix, make_tuple((index_type)index + get<0>(_base))),
            _base.shuffle(typename build_seq<dim - 1>::template arithmetic<1>{})};
    }
    template <typename Index, enable_if_t<is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) const noexcept {
      if constexpr (dim == 1)
        return _tensorPtr->val(getTensorCoord(make_tuple(index), indices{}));
      else
        return tensor_view<add_const_t<Tensor>,
                           gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>>{
            *_tensorPtr, gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>{},
            tuple_cat(_prefix, make_tuple((index_type)index + get<0>(_base))),
            _base.shuffle(typename build_seq<dim - 1>::template arithmetic<1>{})};
    }
    // val
    template <typename Index, enable_if_t<is_integral_v<Index>> = 0>
    constexpr decltype(auto) do_val(Index index) noexcept {
      return _tensorPtr->val(getTensorCoord(index_to_coord(index, vseq_t<extents>{}), indices{}));
      // return zs::apply((*_tensorPtr), getTensorCoord(index_to_coord(index, vseq_t<extents>{}),
      // indices{}));
    }
    template <typename Index, enable_if_t<is_integral_v<Index>> = 0>
    constexpr decltype(auto) do_val(Index index) const noexcept {
      return _tensorPtr->val(getTensorCoord(index_to_coord(index, vseq_t<extents>{}), indices{}));
    }

    Tensor *_tensorPtr{nullptr};
    prefix_type _prefix{};
    base_type _base{};
  };
  template <typename Tensor, typename Extents, typename... Args>
  tensor_view(Tensor &, Extents, Args...) -> tensor_view<Tensor, Extents>;

  template <typename T, typename Tn, Tn... Ns, typename Tm, Tm... Ms, size_t... Is, size_t... Js>
  struct tensor_impl<
      T, integer_sequence<Tn, Ns...>,
      indexer_impl<integer_sequence<Tm, Ms...>, index_sequence<Is...>, index_sequence<Js...>>>
      : VecInterface<tensor_impl<T, integer_sequence<Tn, Ns...>,
                                 indexer_impl<integer_sequence<Tm, Ms...>, index_sequence<Is...>,
                                              index_sequence<Js...>>>> {
    using self_t = tensor_impl<
        T, integer_sequence<Tn, Ns...>,
        indexer_impl<integer_sequence<Tm, Ms...>, index_sequence<Is...>, index_sequence<Js...>>>;
    using base_t = VecInterface<self_t>;
    using primitive_type = T;
    using value_type = remove_pointer_t<T>;
    using index_type = Tn;
    using indexer_type
        = indexer_impl<integer_sequence<Tm, Ms...>, index_sequence<Is...>, index_sequence<Js...>>;
    using extents = integer_sequence<index_type, Ns...>;

    static_assert(sizeof...(Ns) == sizeof...(Ms),
                  "access dimension and storage dimension mismatch!");
    static_assert(((Ns <= Ms) && ...), "access dimension and storage dimension mismatch!");

    template <zs::size_t I> static constexpr auto truncate_storage_orders() noexcept {
      constexpr auto marks = value_seq<(Is < I ? 1 : 0)...>{};
      constexpr auto offsets = marks.scan();  // exclusive scan
      constexpr auto tags = marks.pair(offsets);
      constexpr auto seq = tags.filter(
          typename vseq_t<
              typename build_seq<marks.reduce(plus<int>{}).value>::ascend>::template to_iseq<int>{});
      return value_seq<Is...>{}.shuffle(seq);
    }

    SUPPLEMENT_VEC_STATIC_ATTRIBUTES

    template <typename Extents> using deduce_indexer_type = conditional_t<
        is_same_v<extents, Extents>, indexer_type,
        conditional_t<
            (vseq_t<Extents>::count <= dim),
            //
            indexer_impl<Extents,
                         typename decltype(truncate_storage_orders<vseq_t<Extents>::count>())::iseq,
                         make_index_sequence<vseq_t<Extents>::count>>,
            //
            indexer_impl<
                Extents,
                decltype(value_seq<Is...>{}.concat(
                    typename build_seq<vseq_t<Extents>::count - dim>::template arithmetic<dim>{})),
                make_index_sequence<vseq_t<Extents>::count>>>>;

    template <typename OtherT, typename ExtentsT> using variant_vec
        = tensor_impl<OtherT, ExtentsT, deduce_indexer_type<ExtentsT>>;

    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr T &operator()(Args &&...args) noexcept {
      return _data[indexer_type::offset(FWD(args)...)];
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr const T &operator()(Args &&...args) const noexcept {
      return _data[indexer_type::offset(FWD(args)...)];
    }
    // []
    template <typename Index, enable_if_t<is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) noexcept {
      if constexpr (dim == 1)
        return _data[indexer_type::offset(index)];
      else
        return tensor_view{
            *this, gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>{},
            make_tuple((index_type)index), make_uniform_tuple<dim - 1>((index_type)0)};
    }
    template <typename Index, enable_if_t<is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) const noexcept {
      if constexpr (dim == 1)
        return _data[indexer_type::offset(index)];
      else
        return tensor_view{
            *this, gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>{},
            make_tuple((index_type)index), make_uniform_tuple<dim - 1>((index_type)0)};
    }
    // val (in ascending access order rather than memory storage order)
    decltype(auto) do_val(index_type index) noexcept {
      return _data[indexer_type::offset(index_to_coord(index, vseq_t<extents>{}))];
    }
    decltype(auto) do_val(index_type index) const noexcept {
      return _data[indexer_type::offset(index_to_coord(index, vseq_t<extents>{}))];
    }

    T _data[indexer_type::extent];
  };

}  // namespace zs