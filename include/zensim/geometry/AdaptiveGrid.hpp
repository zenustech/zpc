#pragma once
#include "SparseGrid.hpp"

namespace zs {

  template <int dim, typename ValueT, typename SideLengthBits = index_sequence<3, 4, 5>,
            typename Indices = index_sequence<0, 1, 2>, typename AllocatorT = ZSPmrAllocator<>>
  struct AdaptiveGridImpl;

  template <int dim, typename ValueT, size_t... Ns> using AdaptiveGrid
      = AdaptiveGridImpl<dim, ValueT, index_sequence<Ns...>, make_index_sequence<sizeof...(Ns)>,
                         ZSPmrAllocator<>>;

  /// @brief stores all leaf blocks of an adaptive octree including halo regions
  template <int dim_, typename ValueT, size_t... SideLengthBits, size_t... Is>
  struct AdaptiveGridImpl<dim_, ValueT, index_sequence<SideLengthBits...>, index_sequence<Is...>,
                          ZSPmrAllocator<>> {
    using value_type = ValueT;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = size_t;
    using index_type = zs::make_signed_t<size_type>;  // associated with the number of blocks
    using integer_coord_component_type = int;

    static constexpr auto deduce_basic_value_type() noexcept {
      if constexpr (is_vec<value_type>::value)
        return wrapt<typename value_type::value_type>{};
      else
        return wrapt<value_type>{};
    }
    using coord_component_type = typename RM_CVREF_T(deduce_basic_value_type())::type;
    static_assert(is_floating_point_v<coord_component_type>,
                  "coord type should be floating point.");
    ///
    static constexpr int dim = dim_;
    static constexpr size_t num_levels = sizeof...(SideLengthBits);

    using length_bits_type = value_seq<SideLengthBits...>;
    static constexpr integer_coord_component_type length_bits[num_levels]
        = {(integer_coord_component_type)length_bits_type::template value<Is>...};
    using global_length_bits_type = decltype(declval<length_bits_type>().template scan<1>());
    static constexpr integer_coord_component_type global_length_bits[num_levels]
        = {(integer_coord_component_type)global_length_bits_type::template value<Is>...};

    struct impl_two_pow {
      constexpr integer_coord_component_type operator()(integer_coord_component_type b) noexcept {
        return (integer_coord_component_type)1 << b;
      }
    };
    using side_lengths_type
        = decltype(declval<length_bits_type>().transform(declval<impl_two_pow>()));
    static constexpr integer_coord_component_type side_lengths[num_levels]
        = {side_lengths_type::template value<Is>...};
    using global_side_lengths_type
        = decltype(declval<global_length_bits_type>().transform(declval<impl_two_pow>()));
    static constexpr integer_coord_component_type global_side_lengths[num_levels]
        = {global_side_lengths_type::template value<Is>...};

    struct impl_dim_pow {
      constexpr size_t operator()(size_t sl) noexcept { return math::pow_integral(sl, dim); }
    };
    using block_sizes_type
        = decltype(declval<side_lengths_type>().transform(declval<impl_dim_pow>()));
    static constexpr size_t block_sizes[num_levels]
        = {(size_t)block_sizes_type::template value<Is>...};
    using global_block_sizes_type
        = decltype(declval<global_side_lengths_type>().transform(declval<impl_dim_pow>()));
    static constexpr size_t global_block_sizes[num_levels]
        = {(size_t)global_block_sizes_type::template value<Is>...};

    using integer_coord_type = vec<integer_coord_component_type, dim>;
    using coord_type = vec<coord_component_type, dim>;
    using packed_value_type = vec<value_type, dim>;

    template <size_type bs> using grid_storage_type = TileVector<value_type, bs, allocator_type>;
    template <size_type bs> using mask_storage_type
        = TileVector<u64, (bs + (sizeof(u64) * 8 - 1)) / (sizeof(u64) * 8), allocator_type>;
    using table_type = bht<integer_coord_component_type, dim, int, 16, allocator_type>;
    template <size_type bs> struct Level {
      auto numBlocks() const { return table.size(); }
      auto numReservedBlocks() const noexcept { return grid.numReservedTiles(); }

      Level clone(const allocator_type &allocator) const {
        Level ret{};
        ret.table = table.clone(allocator);
        ret.grid = grid.clone(allocator);
        ret.mask = mask.clone(allocator);
        return ret;
      }
      table_type table;
      grid_storage_type<bs> grid;
      mask_storage_type<bs> mask;  // active/inactive, inside/outside
    };
    using transform_type = math::Transform<coord_component_type, dim>;

    template <size_t I = 0> Level<block_sizes[I]> &level(wrapv<I> = {}) { return get<I>(_levels); }
    template <size_t I = 0> const Level<block_sizes[I]> &level(wrapv<I> = {}) const {
      return get<I>(_levels);
    }

    constexpr MemoryLocation memoryLocation() const noexcept {
      return get<0>(_levels)._table.memoryLocation();
    }
    constexpr ProcID devid() const noexcept { return get<0>(_levels)._table.devid(); }
    constexpr memsrc_e memspace() const noexcept { return get<0>(_levels)._table.memspace(); }
    decltype(auto) get_allocator() const noexcept { return get<0>(_levels)._table.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      return get_memory_source(mre, devid);
    }

    template <size_t I = 0> constexpr auto numBlocks() const noexcept {
      return get<I>(_levels).numBlocks();
    }
    constexpr size_t numTotalBlocks() const noexcept {
      size_t ret = 0;
      (void)((ret += numBlocks<Is>()), ...);
      return ret;
    }
    template <size_t I = 0> constexpr auto numReservedBlocks() const noexcept {
      return get<I>(_levels).numReservedBlocks();
    }

    /// @brief maintenance
    AdaptiveGridImpl() = default;

    AdaptiveGridImpl clone(const allocator_type &allocator) const {
      AdaptiveGridImpl ret{};
      (void)((get<Is>(ret._levels) = get<Is>(_levels).clone(allocator)), ...);
      ret._transform = _transform;
      ret._background = _background;
      return ret;
    }
    AdaptiveGridImpl clone(const zs::MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    /// @brief transformation
    template <typename VecTM,
              enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == dim + 1,
                            VecTM::template range_t<1>::value == dim + 1,
                            is_floating_point_v<typename VecTM::value_type>>
              = 0>
    void resetTransformation(const VecInterface<VecTM> &i2w) {
      _transform.self() = i2w;
    }
    auto getIndexToWorldTransformation() const { return _transform.self(); }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    void translate(const VecInterface<VecT> &t) noexcept {
      _transform.postTranslate(t);
    }
    template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == dim,
                                           VecT::template range_t<1>::value == dim>
                             = 0>
    void rotate(const VecInterface<VecT> &r) noexcept {
      _transform.preRotate(Rotation<typename VecT::value_type, dim>{r});
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    void scale(const VecInterface<VecT> &s) {
      _transform.preScale(s);
    }
    void scale(const value_type s) { scale(s * coord_type::constant(1)); }

    zs::tuple<Level<block_sizes[Is]>...> _levels;
    transform_type _transform;
    value_type _background;  // background value
  };

  // special construct for blockwise access (with halos)

}  // namespace zs