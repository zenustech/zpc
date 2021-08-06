#pragma once
#include "Property.h"
#include "zensim/meta/Meta.h"

namespace zs {

  template <execspace_e, typename T> constexpr decltype(auto) proxy(T &&v) {
    throw std::runtime_error("proxy of T not implemented");
    return FWD(v);
  }
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

    using view_t = decltype(proxy<space>(std::declval<Structure &>()));
    using const_view_t = decltype(proxy<space>(std::declval<const Structure &>()));

#if 0
    template <typename Obj = view_t> static ZS_FUNCTION auto get(Obj obj, size_type i)
        -> decltype(obj(i)) {
      return obj(i);
    }
#endif
    template <typename Obj = view_t> static constexpr auto get(Obj obj, size_type i)
        -> decltype(obj(i)) {
      return obj(i);
    }
    template <typename Obj = view_t>
    static ZS_FUNCTION decltype(std::declval<Obj>()((size_type)0)) ref(Obj obj, size_type i) {
      return obj(i);
    }
    template <typename Obj = view_t, typename V = value_type>
    static ZS_FUNCTION auto set(Obj obj, size_type i, V &&v) -> decltype(obj.set(FWD(v)), void()) {
      set(i);
    }
  };

}  // namespace zs