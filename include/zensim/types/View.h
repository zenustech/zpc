#pragma once
#include "Property.h"
#include "zensim/meta/Meta.h"

namespace zs {

  template <execspace_e, typename T> constexpr decltype(auto) proxy(T &&v) {
    throw std::runtime_error("proxy of T not implemented");
    return FWD(v);
  }

  template <execspace_e space, typename Structure> using view_t
      = decltype(proxy<space>(std::declval<Structure &>()));
  template <execspace_e space, typename Structure> using const_view_t
      = decltype(proxy<space>(std::declval<const Structure &>()));
  ///
  /// dof traits for dofview
  ///
  template <execspace_e space, typename Structure, typename = void> struct dof_traits {
    struct dof_detail {
      template <typename _Tp> using value_t = typename _Tp::value_type;
      template <typename _Tp> using T_t = typename _Tp::T;
      template <typename _Tp> using size_t = typename _Tp::size_type;
      template <typename _Tp> using index_t = typename _Tp::index_t;
      template <typename _Tp> using dim_t = typename std::integral_constant<int, _Tp::dim>;
    };

    using value_type = detected_or_t<detected_or_t<float, dof_detail::T_t, Structure>,
                                     dof_detail::value_t, Structure>;
    using size_type = detected_or_t<detected_or_t<std::size_t, dof_detail::index_t, Structure>,
                                    dof_detail::size_t, Structure>;
    static constexpr int dim
        = detected_or_t<std::integral_constant<int, 3>, dof_detail::dim_t, Structure>::value;
    static constexpr attrib_e preferred_entry_e
        = std::is_floating_point_v<value_type> ? attrib_e::scalar : attrib_e::vector;

    template <execspace_e es = space, typename s = Structure, typename View = view_t<es, s>>
    static constexpr auto get(View obj, size_type i = 0) -> RM_CVREF_T(obj(i)) {
      return obj(i);
    }
    template <execspace_e es = space, typename s = Structure, typename View = view_t<es, s>>
    static constexpr decltype(std::declval<View>()((size_type)0)) ref(View obj, size_type i = 0) {
      return obj(i);
    }
    template <execspace_e es = space, typename s = Structure, typename View = view_t<es, s>,
              typename V = value_type>
    static constexpr auto set(View obj, size_type i = 0, V &&v = {}) -> std::enable_if_t<
        std::is_reference_v<decltype(
            ref(obj, i))> && std::is_convertible_v<RM_CVREF_T(v), RM_CVREF_T(ref(obj, i))>> {
      ref(obj, i) = FWD(v);
    }
  };

}  // namespace zs