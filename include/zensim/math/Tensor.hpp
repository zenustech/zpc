#pragma once

#include <utility>

#include "VecInterface.hpp"
#include "zensim/math/MathUtils.h"
#include "zensim/types/Tuple.h"

namespace zs {

  template <typename Tensor, typename Extents> struct tensor_view;
  template <typename T, typename Extents, typename Indexer> struct tensor_impl;

  template <typename Tensor, typename Tn, Tn... Ns>
  struct tensor_view<Tensor, integer_seq<Tn, Ns...>>
      : VecInterface<tensor_view<Tensor, integer_seq<Tn, Ns...>>> {
    using base_t = VecInterface<tensor_view<Tensor, integer_seq<Tn, Ns...>>>;
    static constexpr bool is_const_structure = std::is_const_v<Tensor>;
    using tensor_type = std::remove_const_t<Tensor>;
    using value_type = typename tensor_type::value_type;
    using extents = integer_seq<Tn, Ns...>;
    using index_type = typename vseq_t<extents>::value_type;
    using indexer_type = typename tensor_type::indexer_type;

    SUPPLEMENT_VEC_STATIC_ATTRIBUTES

    static constexpr auto prefix_dim = tensor_type::dim - dim;

    using prefix_type = typename gen_seq<prefix_dim>::template uniform_types_t<tuple, index_type>;
    using base_type = typename gen_seq<dim>::template uniform_types_t<tuple, index_type>;

    template <typename OtherT, typename ExtentsT> using variant_vec =
        typename tensor_type::template variant_vec<OtherT, ExtentsT>;

    tensor_view() noexcept = default;

    /// helper
    using indices = std::make_index_sequence<dim>;
    template <typename... Args, std::size_t... Is, enable_if_t<dim == sizeof...(Args)> = 0>
    constexpr auto getTensorCoord(tuple<Args...> c, index_seq<Is...>) const noexcept {
      return tuple_cat(_prefix, make_tuple((get<Is>(_base) + get<Is>(c))...));
    }

    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr decltype(auto) operator()(Args &&...args) noexcept {
      return _tensorPtr->val(getTensorCoord(forward_as_tuple(FWD(args)...), indices{}));
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr decltype(auto) operator()(Args &&...args) const noexcept {
      return _tensorPtr->val(getTensorCoord(forward_as_tuple(FWD(args)...), indices{}));
    }
    // []
    template <typename Index, enable_if_t<std::is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) noexcept {
      if constexpr (dim == 1)
        return _tensorPtr->val(getTensorCoord(make_tuple(index), indices{}));
      else
        return tensor_view<Tensor,
                           gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>{
            *_tensorPtr, getTensorCoord(make_tuple(index), index_seq<0>{}),
            _base.shuffle(typename gen_seq<dim - 1>::template arithmetic<1>{})};
    }
    template <typename Index, enable_if_t<std::is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) const noexcept {
      if constexpr (dim == 1)
        return _tensorPtr->val(getTensorCoord(make_tuple(index), indices{}));
      else
        return tensor_view<std::add_const_t<Tensor>,
                           gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>{
            *_tensorPtr, getTensorCoord(make_tuple(index), index_seq<0>{}),
            _base.shuffle(typename gen_seq<dim - 1>::template arithmetic<1>{})};
    }
    // val
    template <typename Index, enable_if_t<std::is_integral_v<Index>> = 0>
    constexpr decltype(auto) do_val(Index index) noexcept {
      return _tensorPtr->val(getTensorCoord(index_to_coord(index, vseq_t<extents>{}), indices{}));
    }
    template <typename Index, enable_if_t<std::is_integral_v<Index>> = 0>
    constexpr decltype(auto) do_val(Index index) const noexcept {
      return _tensorPtr->val(getTensorCoord(index_to_coord(index, vseq_t<extents>{}), indices{}));
    }

    Tensor *_tensorPtr{nullptr};
    prefix_type _prefix{};
    base_type _base{};
  };

  template <typename T, typename Tn, Tn... Ns, typename Tm, Tm... Ms, std::size_t... Is,
            std::size_t... Js>
  struct tensor_impl<T, integer_seq<Tn, Ns...>,
                     indexer_impl<integer_seq<Tm, Ms...>, index_seq<Is...>, index_seq<Js...>>>
      : VecInterface<
            tensor_impl<T, integer_seq<Tn, Ns...>,
                        indexer_impl<integer_seq<Tm, Ms...>, index_seq<Is...>, index_seq<Js...>>>> {
    using self_t
        = tensor_impl<T, integer_seq<Tn, Ns...>,
                      indexer_impl<integer_seq<Tm, Ms...>, index_seq<Is...>, index_seq<Js...>>>;
    using base_t = VecInterface<self_t>;
    using value_type = T;
    using index_type = Tn;
    using indexer_type = indexer_impl<integer_seq<Tm, Ms...>, index_seq<Is...>, index_seq<Js...>>;
    using extents = integer_seq<index_type, Ns...>;

    static_assert(sizeof...(Ns) == sizeof...(Ms),
                  "access dimension and storage dimension mismatch!");
    static_assert(((Ns <= Ms) && ...), "access dimension and storage dimension mismatch!");

    template <std::size_t I> static constexpr auto truncate_storage_orders() noexcept {
      constexpr auto marks = value_seq<(Is < I ? 1 : 0)...>{};
      constexpr auto N = marks.reduce(plus<int>{}).value;
      constexpr auto offsets = marks.scan();  // exclusive scan
      constexpr auto tags = marks.pair(offsets);
      constexpr auto seq
          = tags.filter(typename vseq_t<typename gen_seq<N>::ascend>::template to_iseq<int>{});
      return value_seq<Is...>{}.shuffle(seq);
    }

    SUPPLEMENT_VEC_STATIC_ATTRIBUTES

    template <typename Extents>
    using deduce_indexer_type = conditional_t<is_same_v<extents, Extents>, indexer_type, 
      conditional_t<(vseq_t<Extents>::count <= dim), 
                     //
                     indexer_impl<Extents,
                     typename decltype(truncate_storage_orders<vseq_t<Extents>::count>())::iseq,
                     std::make_index_sequence<vseq_t<Extents>::count>>, 
                     // 
                     indexer_impl<Extents, 
                     decltype(value_seq<Is...>{}.concat(typename gen_seq<vseq_t<Extents>::count - dim>::template arithmetic<dim>{})),
                     std::make_index_sequence<vseq_t<Extents>::count>>
      >
    >;

    template <typename OtherT, typename ExtentsT> using variant_vec = tensor_impl<
        OtherT, ExtentsT, deduce_indexer_type<ExtentsT>>;

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
    template <typename Index, enable_if_t<std::is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) noexcept {
      if constexpr (dim == 1)
        return _data[indexer_type::offset(index)];
      else
        return tensor_view<self_t,
                           gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>{
            this, make_tuple(index), make_uniform_tuple<dim - 1>((index_type)0)};
    }
    template <typename Index, enable_if_t<std::is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) const noexcept {
      if constexpr (dim == 1)
        return _data[indexer_type::offset(index)];
      else
        return tensor_view<std::add_const_t<self_t>,
                           gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>{
            this, make_tuple(index), make_uniform_tuple<dim - 1>((index_type)0)};
    }
    // val (in ascending access order rather than memory storage order)
    template <typename Index> constexpr T &do_val(Index index) noexcept {
      return _data[indexer_type::offset(index_to_coord(index, vseq_t<extents>{}))];
    }
    template <typename Index> constexpr const T &do_val(Index index) const noexcept {
      return _data[indexer_type::offset(index_to_coord(index, vseq_t<extents>{}))];
    }
#if 0
    static void print() {
      printf("indexer_type: %s\n", get_type_str<indexer_type>().data());
      printf("tensor size check: %d\n", (int)indexer_type::extent);
    }
#endif

    T _data[indexer_type::extent];
  };

}  // namespace zs