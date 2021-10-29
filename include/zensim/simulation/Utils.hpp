#pragma once
#include "zensim/math/Vec.h"
#include "zensim/tpls/magic_enum/magic_enum.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/Property.h"

namespace zs {

#if 0
  template <typename Tn, int dim, typename Ti, typename Table>
  constexpr auto unpack_coord_in_grid(const vec<Tn, dim> &coord, Ti sideLength,
                                      const Table &table) {
    using IV = vec<Tn, dim>;
    IV blockCoord = coord;
    for (int d = 0; d < dim; ++d) blockCoord[d] += (coord[d] < 0 ? -(Tn)sideLength + 1 : 0);
    blockCoord = blockCoord / (Tn)sideLength;
    return std::make_tuple(table.query(blockCoord), coord - blockCoord * (Tn)sideLength);
  }
  template <typename Tn, int dim, typename Ti>
  constexpr auto unpack_coord_in_grid(const vec<Tn, dim> &coord, Ti sideLength) {
    using IV = vec<Tn, dim>;
    IV blockCoord = coord;
    for (int d = 0; d < dim; ++d) blockCoord[d] += (coord[d] < 0 ? -(Tn)sideLength + 1 : 0);
    blockCoord = blockCoord / (Tn)sideLength;
    return std::make_tuple(blockCoord, coord - blockCoord * (Tn)sideLength);
  }
  template <typename Tn, int dim, typename Ti, typename Table, typename Grid>
  constexpr auto unpack_coord_in_grid(const vec<Tn, dim> &coord, Ti sideLength, const Table &table,
                                      Grid &&grid) {
    using IV = vec<Tn, dim>;
    IV blockCoord = coord;
    for (int d = 0; d < dim; ++d) blockCoord[d] += (coord[d] < 0 ? -(Tn)sideLength + 1 : 0);
    blockCoord = blockCoord / (Tn)sideLength;
    return std::forward_as_tuple(grid[table.query(blockCoord)],
                                 coord - blockCoord * (Tn)sideLength);
  }
#else
  template <typename Tn, int dim, typename Ti>
  constexpr auto unpack_coord_in_grid(const vec<Tn, dim> &coord, Ti sideLength) {
    auto loc = coord & (Tn)(sideLength - 1);
    return std::make_tuple((coord - loc) / (Tn)sideLength, loc);
  }
  template <typename Tn, int dim, typename Ti, typename Table>
  constexpr auto unpack_coord_in_grid(const vec<Tn, dim> &coord, Ti sideLength,
                                      const Table &table) {
    auto loc = coord & (Tn)(sideLength - 1);
    return std::make_tuple(table.query((coord - loc) / (Tn)sideLength), loc);
  }
  template <typename Tn, int dim, typename Ti, typename Table, typename Grid>
  constexpr auto unpack_coord_in_grid(const vec<Tn, dim> &coord, Ti sideLength, const Table &table,
                                      Grid &&grid) {
    auto loc = coord & (Tn)(sideLength - 1);
    if constexpr (std::is_reference_v<decltype(grid[0])>)
      return std::make_tuple(std::ref(grid[table.query((coord - loc) / (Tn)sideLength)]), loc);
    else
      return std::make_tuple(grid[table.query((coord - loc) / (Tn)sideLength)], loc);
  }
#endif

  template <grid_e gt = grid_e::collocated, kernel_e kt = kernel_e::quadratic, typename T = f32,
            int dim_ = 3, typename Ti = int>
  struct LocalArena {
    using value_type = T;
    using index_type = Ti;
    static constexpr int dim = dim_;
    static constexpr index_type width = magic_enum::enum_integer(kt);
    using TV = vec<value_type, dim>;
    using TM = vec<value_type, dim, width>;
    using IV = vec<index_type, dim>;

    constexpr IV world_to_index_base(const TV &pos) const noexcept {
      const auto dxInv = (T)1 / dx;
      constexpr index_type offset
          = kt == kernel_e::quadratic ? -1 : (kt == kernel_e::linear ? 0 : -2);
      IV ret{};
      for (int d = 0; d != dim; ++d)
        if constexpr (gt == grid_e::collocated)
          ret[d] = lower_trunc(pos[d] * dxInv + (T)0.5) + offset;
        else if constexpr (gt == grid_e::cellcentered)
          ret[d] = lower_trunc(pos[d] * dxInv) + offset;
        else if constexpr (gt == grid_e::staggered)
          ret[d] = lower_trunc(pos[d] * dxInv) + offset;
      return ret;
    }

    constexpr void init(const value_type dx_, const TV &pos) {
      dx = dx_;
      corner = world_to_index_base(pos);
      if constexpr (gt == grid_e::collocated)
        localPos = pos - corner * dx;
      else if constexpr (gt == grid_e::cellcentered)
        localPos = pos - (corner + (T)0.5) * dx;
      else if constexpr (gt == grid_e::staggered)
        localPos = pos - corner * dx;

      weights = bspline_weight(localPos, (T)1 / dx);
    }

    constexpr auto range() const noexcept { return ndrange<dim>(width); }

  protected:
    template <typename... Tn, std::size_t... Is,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim)> = 0>
    constexpr auto weight_impl(const std::tuple<Tn...> &loc, index_seq<Is...>) const noexcept {
      value_type ret{1};
      ((void)(ret *= weights(Is, std::get<Is>(loc))), ...);
      return ret;
    }

  public:
    template <typename... Tn> constexpr IV offset(const std::tuple<Tn...> &loc) const noexcept {
      return make_vec<index_type>(loc);
    }

    template <typename... Tn,
              enable_if_all<(!is_std_tuple<Tn>() && ... && (sizeof...(Tn) == dim))> = 0>
    constexpr auto weight(Tn &&...is) const noexcept {
      return weight(std::forward_as_tuple(FWD(is)...));
    }

    template <typename... Tn> constexpr T weight(const std::tuple<Tn...> &loc) const noexcept {
      return weight_impl(loc, std::index_sequence_for<Tn...>{});
    }
    template <typename... Tn> constexpr TV diff(const std::tuple<Tn...> &pos) const noexcept {
      return offset(pos) * dx - localPos;
    }
    template <typename... Tn> constexpr IV coord(const std::tuple<Tn...> &pos) const noexcept {
      return offset(pos) + corner;
    }

    TV localPos{TV::zeros()};
    TM weights{TM::zeros()};
    IV corner{IV::zeros()};
    value_type dx{0};
  };

  template <grid_e gt = grid_e::collocated, kernel_e kt = kernel_e::quadratic, typename T = f32,
            int dim = 3, typename Ti = int, typename TT = T>
  constexpr LocalArena<gt, kt, T, dim, Ti> make_local_arena(TT dx, const vec<T, dim> &pos) {
    LocalArena<gt, kt, T, dim, Ti> ret{};
    ret.init(dx, pos);
    return ret;
  }

}  // namespace zs