#pragma once
#include "Property.h"
#include "zensim/meta/Meta.h"

namespace zs {

  ///
  /// dof traits for dofview
  ///
  template <execspace_e space, typename Structure, typename = void> struct dof_traits {
    struct dof_detail {
      template <typename _Tp> using value_t = typename _Tp::value_type;
      template <typename _Tp> using T_t = typename _Tp::T;
      template <typename _Tp> using size_t = typename _Tp::size_type;
      template <typename _Tp> using index_t = typename _Tp::index_t;
      template <typename _Tp> using counter_t = typename _Tp::channel_counter_type;
      template <typename _Tp> using dim_t = typename std::integral_constant<int, _Tp::dim>;
    };

    using structure_view_t = decltype(proxy<space>(std::declval<Structure>()));
    using structure_type = remove_cvref_t<Structure>;
    using value_type = detected_or_t<detected_or_t<float, dof_detail::T_t, structure_type>,
                                     dof_detail::value_t, structure_type>;
    using size_type = detected_or_t<detected_or_t<std::size_t, dof_detail::index_t, structure_type>,
                                    dof_detail::size_t, structure_type>;
    using channel_counter_type
        = detected_or_t<unsigned char, dof_detail::counter_t, structure_type>;
    static constexpr int dim
        = detected_or_t<std::integral_constant<int, 3>, dof_detail::dim_t, structure_type>::value;
    static constexpr attrib_e default_entry_e
        = std::is_floating_point_v<value_type> ? attrib_e::scalar : attrib_e::vector;

    /// access by entry index
    template <typename svt = structure_view_t> static constexpr auto get(svt obj, size_type i)
        -> std::enable_if_t<std::is_convertible_v<RM_CVREF_T(obj(i)), value_type>, value_type> {
      return obj(i);
    }
    template <typename svt = structure_view_t> static constexpr auto ref(svt obj, size_type i)
        -> std::enable_if_t<
            std::is_reference_v<decltype(obj(i))> && is_same_v<decltype(get(obj, i)), value_type>,
            decltype(obj(i))> {
      return obj(i);
    }
    template <typename svt = structure_view_t>
    static constexpr auto set(svt obj, size_type i, const value_type& v)
        -> std::enable_if_t<std::is_assignable_v<decltype(ref(obj, i)), value_type>> {
      ref(obj, i) = v;
    }

    /// access by channel and entry index
    template <typename svt = structure_view_t>
    static constexpr auto get(svt obj, channel_counter_type chn, size_type i)
        -> std::enable_if_t<std::is_convertible_v<RM_CVREF_T(obj(chn, i)), value_type>,
                            value_type> {
      return obj(chn, i);
    }
    template <typename svt = structure_view_t>
    static constexpr auto ref(svt obj, channel_counter_type chn, size_type i)
        -> std::enable_if_t<std::is_reference_v<decltype(obj(
                                chn, i))> && is_same_v<decltype(get(obj, chn, i)), value_type>,
                            decltype(obj(chn, i))> {
      return obj(chn, i);
    }
    template <typename svt = structure_view_t>
    static constexpr auto set(svt obj, channel_counter_type chn, size_type i, const value_type& v)
        -> std::enable_if_t<std::is_assignable_v<decltype(ref(obj, chn, i)), value_type>> {
      ref(obj, chn, i) = v;
    }
  };

  template <execspace_e space, typename Structure, attrib_e ae> struct DofView {
    using structure_t = Structure;
    using structure_view_t = decltype(proxy<space>(std::declval<Structure>()));
    using traits = dof_traits<space, structure_t>;
    using value_type = typename traits::value_type;
    using size_type = typename traits::size_type;
    using channel_counter_type = typename traits::channel_counter_type;
    static constexpr int dim = traits::dim;
    static constexpr auto entry_e = ae;

    DofView(wrapv<space>, Structure structure, wrapv<ae> = {})
        : _structure{proxy<space>(structure)} {}

    structure_view_t _structure;

    constexpr decltype(auto) ref(size_type i) { return traits::ref(_structure, i); }

    template <attrib_e AccessEntry = ae>
    constexpr value_type get(size_type i, wrapv<AccessEntry> = {}) const {
      if constexpr (AccessEntry == ae)
        return traits::get(_structure, i);
      else if (AccessEntry == attrib_e::scalar && ae == attrib_e::vector)
        return traits::get(_structure, i / dim)[i % dim];
      else if (AccessEntry == attrib_e::vector && ae == attrib_e::scalar) {
        vec<value_type, dim> ret{};
        const size_type base = i * dim;
        for (int d = 0; d != dim; ++d) ret(d) = traits::get(_structure, base + d);
        return ret;
      }
    }
    template <typename V> constexpr auto set(size_type i, V&& v = {})
        -> std::enable_if_t<std::is_lvalue_reference_v<decltype(ref(i))>> {
      if constexpr (std::is_assignable_v<decltype(ref(i)), V>)
        traits::set(_structure, i, FWD(v));
      else if (ae == attrib_e::vector)  // V is a scalar
        ref(i)[i % dim] = FWD(v);
      else if (ae == attrib_e::scalar) {  // V is a vector
        const size_type base = i * dim;
        for (int d = 0; d != dim; ++d) ref(base + d) = v[d];
      }
    }
  };

  template <execspace_e space, typename Structure, attrib_e ae> struct DofChannelView {
    using structure_t = Structure;
    using structure_view_t = decltype(proxy<space>(std::declval<Structure>()));
    using traits = dof_traits<space, structure_t>;
    using value_type = typename traits::value_type;
    using size_type = typename traits::size_type;
    using channel_counter_type = typename traits::channel_counter_type;
    static constexpr int dim = traits::dim;
    static constexpr auto default_entry_e = traits::default_entry_e;
    static constexpr auto entry_e = ae;

    DofChannelView(wrapv<space>, Structure structure, channel_counter_type chn, wrapv<ae> = {})
        : _structure{proxy<space>(structure)}, _chn{chn} {}

    structure_view_t _structure;
    channel_counter_type _chn;

    constexpr decltype(auto) ref(size_type i) { return traits::ref(_structure, _chn, i); }

    template <attrib_e AccessEntry = ae>
    constexpr value_type get(size_type i, wrapv<AccessEntry> = {}) const {
      if constexpr (AccessEntry == ae)
        return traits::get(_structure, _chn, i);
      else if (AccessEntry == attrib_e::scalar && ae == attrib_e::vector)
        return traits::get(_structure, _chn, i / dim)[i % dim];
      else if (AccessEntry == attrib_e::vector && ae == attrib_e::scalar) {
        // different
        vec<value_type, dim> ret{};
        for (int d = 0; d != dim; ++d) ret(d) = traits::get(_structure, _chn + d, i);
        return ret;
      }
    }
    template <typename V> constexpr auto set(size_type i, V&& v = {})
        -> std::enable_if_t<std::is_lvalue_reference_v<decltype(ref(i))>> {
      if constexpr (std::is_assignable_v<decltype(ref(i)), V>)
        traits::set(_structure, _chn, i, FWD(v));
      else if (ae == attrib_e::vector)  // V is a scalar
        ref(i / dim)[i % dim] = FWD(v);
      else if (ae == attrib_e::scalar)  // V is a vector
        // different
        for (int d = 0; d != dim; ++d) ref(_chn + d, i) = v[d];
    }
  };

  ///
  template <execspace_e space, typename T> constexpr decltype(auto) dof_view(T&& t) {
    return DofView<space, T, dof_traits<space, T>::default_entry_e>{wrapv<space>{}, t};
  }
  template <execspace_e space, attrib_e ae, typename T> constexpr decltype(auto) dof_view(T&& t) {
    return DofView<space, T, ae>{wrapv<space>{}, t};
  }

  template <execspace_e space, typename T>
  constexpr decltype(auto) dof_view(T&& t, typename dof_traits<space, T>::channel_counter_type c) {
    return DofChannelView<space, T, dof_traits<space, T>::default_entry_e>{wrapv<space>{}, t, c};
  }
  template <execspace_e space, attrib_e ae, typename T>
  constexpr decltype(auto) dof_view(T&& t, typename dof_traits<space, T>::channel_counter_type c) {
    return DofChannelView<space, T, ae>{wrapv<space>{}, t, c};
  }

}  // namespace zs