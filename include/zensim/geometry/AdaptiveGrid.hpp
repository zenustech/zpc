#pragma once
#include "SparseGrid.hpp"
#include "zensim/types/Mask.hpp"

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
    static constexpr int num_levels = sizeof...(SideLengthBits);

    // length_bits
    using length_bit_counts_type = value_seq<SideLengthBits...>;
    static constexpr integer_coord_component_type length_bit_counts[num_levels]
        = {(integer_coord_component_type)length_bit_counts_type::template value<Is>...};
    using global_length_bit_counts_type
        = decltype(declval<length_bit_counts_type>().template scan<1>());
    static constexpr integer_coord_component_type global_length_bit_counts[num_levels]
        = {(integer_coord_component_type)global_length_bit_counts_type::template value<Is>...};

    // side_lengths
    struct impl_two_pow {
      constexpr integer_coord_component_type operator()(integer_coord_component_type b) noexcept {
        return (integer_coord_component_type)1 << b;
      }
    };
    using side_lengths_type
        = decltype(declval<length_bit_counts_type>().transform(declval<impl_two_pow>()));
    static constexpr integer_coord_component_type side_lengths[num_levels]
        = {side_lengths_type::template value<Is>...};
    using global_side_lengths_type
        = decltype(declval<global_length_bit_counts_type>().transform(declval<impl_two_pow>()));
    static constexpr integer_coord_component_type global_side_lengths[num_levels]
        = {global_side_lengths_type::template value<Is>...};

    // block_sizes
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
    template <size_type bs> using mask_storage_type = Vector<bit_mask<bs>, allocator_type>;
    using table_type = bht<integer_coord_component_type, dim, index_type, 16, allocator_type>;
    static constexpr index_type sentinel_v = table_type::sentinel_v;

    template <int level_> struct Level {
      static constexpr int level = level_;
      static constexpr integer_coord_component_type length_bit_count = length_bit_counts[level];
      static constexpr integer_coord_component_type global_length_bit_count
          = global_length_bit_counts[level];
      static constexpr integer_coord_component_type side_length = side_lengths[level];
      static constexpr integer_coord_component_type global_side_length = global_side_lengths[level];
      static constexpr size_t block_size = block_sizes[level];

      /// @note used for global coords
      static constexpr make_unsigned_t<integer_coord_component_type> cell_mask
          = global_side_length - 1;
      static constexpr make_unsigned_t<integer_coord_component_type> origin_mask = ~cell_mask;

      using grid_type = grid_storage_type<block_size>;
      using mask_type = mask_storage_type<block_size>;
      Level(const allocator_type &allocator, const std::vector<PropertyTag> &propTags, size_t count)
          : table{allocator, count},
            grid{allocator, propTags, count * block_size},
            valueMask{allocator, count},
            childMask{allocator, count} {
        grid.reset(0);
        childMask.reset(0);
        valueMask.reset(0);
      }
      Level(const allocator_type &allocator, size_t count)
          : Level(allocator, {{"sdf", 1}}, count) {}
      Level(size_t count = 0, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
          : Level(get_default_allocator(mre, devid), count) {}
      Level(const std::vector<PropertyTag> &propTags, size_t count, memsrc_e mre = memsrc_e::host,
            ProcID devid = -1)
          : Level(get_default_allocator(mre, devid), propTags, count) {}

      ~Level() = default;

      Level(const Level &o) = default;
      Level(Level &&o) noexcept = default;
      Level &operator=(const Level &o) = default;
      Level &operator=(Level &&o) noexcept = default;

      auto numBlocks() const { return table.size(); }
      auto numReservedBlocks() const noexcept { return grid.numReservedTiles(); }

      template <typename ExecPolicy>
      void resize(ExecPolicy &&policy, size_type numBlocks, bool resizeGrid = true) {
        table.resize(FWD(policy), numBlocks);
        if (resizeGrid) grid.resize(numBlocks * (size_type)block_size);
      }
      template <typename ExecPolicy>
      void resizePartition(ExecPolicy &&policy, size_type numBlocks) {
        table.resize(FWD(policy), numBlocks);
      }
      void resizeGrid(size_type numBlocks) { grid.resize(numBlocks * (size_type)block_size); }
      template <typename Policy>
      void append_channels(Policy &&policy, const std::vector<PropertyTag> &tags) {
        grid.append_channels(FWD(policy), tags);
      }
      // byte-wise reset
      void reset(value_type val) { grid.reset(val); }
      // value-wise reset
      template <typename Policy> void reset(Policy &&policy, value_type val) {
        grid.reset(FWD(policy), val);
      }

      Level clone(const allocator_type &allocator) const {
        Level ret{};
        ret.table = table.clone(allocator);
        ret.grid = grid.clone(allocator);
        ret.childMask = childMask.clone(allocator);
        ret.valueMask = valueMask.clone(allocator);
        return ret;
      }

      auto originRange() const {
        auto bg = table._activeKeys.begin();
        return detail::iter_range(bg, bg + numBlocks());
      }
      auto originRange() {
        auto bg = table._activeKeys.begin();
        return detail::iter_range(bg, bg + numBlocks());
      }

      table_type table;
      grid_type grid;
      /// @note for levelset, valueMask indicates inside/outside
      /// @note for leaf level, childMask reserved for special use cases
      mask_type valueMask, childMask;
    };
    using transform_type = math::Transform<coord_component_type, dim>;

    template <auto I = 0> Level<I> &level(wrapv<I>) { return zs::get<I>(_levels); }
    template <auto I = 0> const Level<I> &level(wrapv<I>) const { return zs::get<I>(_levels); }
    template <auto I = 0> Level<I> &level(value_seq<I>) { return zs::get<I>(_levels); }
    template <auto I = 0> const Level<I> &level(value_seq<I>) const { return zs::get<I>(_levels); }

    constexpr MemoryLocation memoryLocation() const noexcept {
      return get<0>(_levels)._table.memoryLocation();
    }
    constexpr ProcID devid() const noexcept { return get<0>(_levels).table.devid(); }
    constexpr memsrc_e memspace() const noexcept { return get<0>(_levels).table.memspace(); }
    decltype(auto) get_allocator() const noexcept { return get<0>(_levels).table.get_allocator(); }
    static decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) {
      return get_memory_source(mre, devid);
    }

    template <auto I = 0> constexpr auto numBlocks(wrapv<I> = {}) const noexcept {
      return get<I>(_levels).numBlocks();
    }
    template <auto I = 0> constexpr auto numBlocks(value_seq<I> = {}) const noexcept {
      return get<I>(_levels).numBlocks();
    }
    constexpr size_t numTotalBlocks() const noexcept {
      size_t ret = 0;
      (void)((ret += numBlocks(wrapv<Is>{})), ...);
      return ret;
    }
    template <size_t I = 0> constexpr auto numReservedBlocks() const noexcept {
      return get<I>(_levels).numReservedBlocks();
    }
    template <typename Policy>
    void append_channels(Policy &&policy, const std::vector<PropertyTag> &tags) {
      (void)(get<Is>(_levels).append_channels(FWD(policy), tags), ...);
    }
    void reset(value_type val) { (void)(get<Is>(_levels).reset(val), ...); }
    // value-wise reset
    template <typename Policy> void reset(Policy &&policy, value_type val) {
      (void)(get<Is>(_levels).reset(policy, val), ...);
    }
    constexpr auto numChannels() const noexcept {
      return level(dim_c<num_levels - 1>).grid.numChannels();
    }
    bool hasProperty(const SmallString &str) const noexcept {
      return level(dim_c<num_levels - 1>).grid.hasProperty(str);
    }
    constexpr size_type getPropertySize(const SmallString &str) const {
      return level(dim_c<num_levels - 1>).grid.getPropertySize(str);
    }
    constexpr size_type getPropertyOffset(const SmallString &str) const {
      return level(dim_c<num_levels - 1>).grid.getPropertyOffset(str);
    }
    constexpr PropertyTag getPropertyTag(size_type i = 0) const {
      return level(dim_c<num_levels - 1>).grid.getPropertyTag(i);
    }
    constexpr const auto &getPropertyTags() const {
      return level(dim_c<num_levels - 1>).grid.getPropertyTags();
    }

    constexpr coord_type voxelSize() const {
      // does not consider shearing here
      coord_type ret{};
      for (int i = 0; i != dim; ++i) {
        coord_component_type sum = 0;
        for (int d = 0; d != dim; ++d) sum += zs::sqr(_transform(i, d));
        ret.val(i) = std::sqrt(sum);
      }
      return ret;
    }
    static constexpr auto zeroValue() noexcept {
      if constexpr (is_vec<value_type>::value)
        return value_type::zeros();
      else
        return (value_type)0;
    }

    /// @brief maintenance
    AdaptiveGridImpl() = default;
    ~AdaptiveGridImpl() = default;

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

    zs::tuple<Level<Is>...> _levels;
    transform_type _transform;
    value_type _background;  // background value
  };

  template <typename T, typename = void> struct is_ag : false_type {};
  template <int dim, typename ValueT, size_t... SideLengthBits, size_t... Is, typename AllocatorT>
  struct is_ag<AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>,
                                index_sequence<Is...>, AllocatorT>> : true_type {};
  template <typename T> constexpr bool is_ag_v = is_ag<T>::value;

  ///
  /// @brief adaptive grid unnamed view
  ///
  template <execspace_e Space, typename AdaptiveGridT, bool IsConst, bool Base = false>
  struct AdaptiveGridUnnamedView;

  /// @note Base here indicates whether grid should use lite version
  /// @note this specializations is targeting ZSPmrAllocator<false> specifically
  template <execspace_e Space, int dim_, typename ValueT, size_t... SideLengthBits, size_t... Is,
            bool IsConst, bool Base>
  struct AdaptiveGridUnnamedView<Space,
                                 AdaptiveGridImpl<dim_, ValueT, index_sequence<SideLengthBits...>,
                                                  index_sequence<Is...>, ZSPmrAllocator<>>,
                                 IsConst, Base>
      : LevelSetInterface<AdaptiveGridUnnamedView<
            Space,
            AdaptiveGridImpl<dim_, ValueT, index_sequence<SideLengthBits...>, index_sequence<Is...>,
                             ZSPmrAllocator<>>,
            IsConst, Base>> {
    static constexpr bool is_const_structure = IsConst;
    static constexpr auto space = Space;
    using container_type = AdaptiveGridImpl<dim_, ValueT, index_sequence<SideLengthBits...>,
                                            index_sequence<Is...>, ZSPmrAllocator<>>;
    static constexpr int dim = container_type::dim;
    static constexpr int num_levels = container_type::num_levels;
    using value_type = typename container_type::value_type;
    using size_type = typename container_type::size_type;
    using index_type = typename container_type::index_type;

    using integer_coord_component_type = typename container_type::integer_coord_component_type;
    using integer_coord_type = typename container_type::integer_coord_type;
    using coord_component_type = typename container_type::coord_component_type;
    using coord_type = typename container_type::coord_type;
    using packed_value_type = typename container_type::packed_value_type;

    template <int LevelNo> using level_type = typename container_type::template Level<LevelNo>;
    using table_type = typename container_type::table_type;
    using table_view_type = RM_CVREF_T(
        view<space>(declval<conditional_t<is_const_structure, const table_type &, table_type &>>(),
                    wrapv<Base>{}));
    static constexpr index_type sentinel_v = table_type::sentinel_v;

    // grid
    template <int LevelNo> using grid_storage_type = typename level_type<LevelNo>::grid_type;
    template <int LevelNo> using unnamed_grid_view_type = RM_CVREF_T(
        view<space>(declval<conditional_t<is_const_structure, const grid_storage_type<LevelNo> &,
                                          grid_storage_type<LevelNo> &>>(),
                    wrapv<Base>{}));

    // mask
    template <int LevelNo> using mask_storage_type = typename level_type<LevelNo>::mask_type;
    template <int LevelNo> using mask_view_type = RM_CVREF_T(
        view<space>(declval<conditional_t<is_const_structure, const mask_storage_type<LevelNo> &,
                                          mask_storage_type<LevelNo> &>>(),
                    wrapv<Base>{}));

    template <int LevelNo> struct level_view_type {
      static constexpr int level = LevelNo;
      using level_t = level_type<level>;
      using grid_type = unnamed_grid_view_type<level>;
      using mask_type = mask_view_type<level>;

      constexpr level_view_type() noexcept = default;
      ~level_view_type() = default;
      level_view_type(conditional_t<is_const_structure, add_const_t<level_t>, level_t> &level)
          : table{view<Space>(level.table, wrapv<Base>{})},
            grid{view<Space>(level.grid, wrapv<Base>{})},
            valueMask{view<Space>(level.valueMask, wrapv<Base>{})},
            childMask{view<Space>(level.childMask, wrapv<Base>{})} {}

      static constexpr integer_coord_component_type length_bit_count = level_t::length_bit_count;
      static constexpr integer_coord_component_type global_length_bit_count
          = level_t::global_length_bit_count;
      static constexpr integer_coord_component_type side_length = level_t::side_length;
      static constexpr integer_coord_component_type global_side_length
          = level_t::global_side_length;
      static constexpr size_t block_size = level_t::block_size;

      /// @note used for global coords
      static constexpr make_unsigned_t<integer_coord_component_type> cell_mask = level_t::cell_mask;
      static constexpr make_unsigned_t<integer_coord_component_type> origin_mask
          = level_t::origin_mask;

      constexpr auto numBlocks() const { return table.size(); }

      table_view_type table;
      grid_type grid;
      mask_type valueMask, childMask;
    };

    using transform_type = typename container_type::transform_type;

    template <auto I = 0, bool isConst = is_const_structure>
    enable_if_type<!isConst, level_view_type<I> &> level(wrapv<I>) {
      return zs::get<I>(_levels);
    }
    template <auto I = 0> const level_view_type<I> &level(wrapv<I>) const {
      return zs::get<I>(_levels);
    }
    template <auto I = 0, bool isConst = is_const_structure>
    enable_if_type<!isConst, level_view_type<I> &> level(value_seq<I>) {
      return zs::get<I>(_levels);
    }
    template <auto I = 0> const level_view_type<I> &level(value_seq<I>) const {
      return zs::get<I>(_levels);
    }

    AdaptiveGridUnnamedView() noexcept = default;
    ~AdaptiveGridUnnamedView() noexcept = default;
    constexpr AdaptiveGridUnnamedView(
        conditional_t<is_const_structure, add_const_t<container_type>, container_type> &ag)
        : _levels{}, _transform{ag._transform}, _background{ag._background} {
      (void)((zs::get<Is>(_levels) = level_view_type<Is>{ag.level(dim_c<Is>)}), ...);
    }

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto indexToWorld(const VecInterface<VecT> &X) const {
      return X * _transform;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto worldToIndex(const VecInterface<VecT> &x) const {
      return x * inverse(_transform);
    }

    template <int I>
    static constexpr integer_coord_type coord_to_key(const integer_coord_type &c) noexcept {
      return c & level_view_type<I>::origin_mask;
    }
    template <int I>
    static constexpr integer_coord_component_type get_global_length_bit_count() noexcept {
      if constexpr (I < 0)
        return 0;
      else
        return level_view_type<I>::global_length_bit_count;
    }
    template <int I> static constexpr integer_coord_component_type coord_to_offset(
        const integer_coord_type &c) noexcept {
      integer_coord_component_type ret
          = (c[0] & level_view_type<I>::cell_mask) >> get_global_length_bit_count<I - 1>();
      for (int d = 1; d != dim; ++d) {
        ret = (ret << level_view_type<I>::length_bit_count)
              | ((c[d] & level_view_type<I>::cell_mask) >> get_global_length_bit_count<I - 1>());
      }
      return ret;
    }
    template <typename T, int I = num_levels - 1>
    constexpr enable_if_type<!is_const_v<T>, bool> probeValue(size_type chn,
                                                              const integer_coord_type &coord,
                                                              T &val, index_type bno = sentinel_v,
                                                              wrapv<I> = {}) const {
      constexpr bool IsVec = is_vec<T>::value;
      auto &lev = level(dim_c<I>);
      if (bno == sentinel_v) {
        auto c = coord_to_key<I>(coord);
        bno = lev.table.query(c);
        if (bno == sentinel_v) {
          if constexpr (IsVec) {
            val = T::constant(_background);
          } else
            val = _background;
          return false;
        }
      }
      const integer_coord_component_type n = coord_to_offset<I>(coord);
      auto block = lev.grid.tile(bno);
      if constexpr (I > 0) {
        // printf("found int-%d [%d] (%d, %d, %d) ch[%d]\n", I, bno, lev.table._activeKeys[bno][0],
        //        lev.table._activeKeys[bno][1], lev.table._activeKeys[bno][2], n);
        /// @note internal level
        if (lev.childMask[bno].isOff(n)) {
          if constexpr (IsVec) {
            for (int d = 0; d != T::extent; ++d) val.val(d) = block(chn + d, n);
          } else
            val = block(chn, n);
          return lev.valueMask[bno].isOn(n);
        }
        /// TODO: an optimal layout should directly give child-n position
        return probeValue(chn, coord, val, sentinel_v, wrapv<I - 1>{});
      } else {
        /// @note leaf level
        if constexpr (IsVec) {
          for (int d = 0; d != T::extent; ++d) val.val(d) = block(chn + d, n);
        } else
          val = block(chn, n);
        // printf("found leaf [%d] (%d, %d, %d), slot[%d (%d, %d, %d)] val: %f\n", bno,
        //        lev.table._activeKeys[bno][0], lev.table._activeKeys[bno][1],
        //        lev.table._activeKeys[bno][2], n, coord[0], coord[1], coord[2], (float)val);
        return lev.valueMask[bno].isOn(n);
      }
    }

    zs::tuple<level_view_type<Is>...> _levels;
    transform_type _transform;
    value_type _background;
  };

  ///
  /// @brief view variants for AdaptiveUnnamedView
  ///
  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... SideLengthBits, size_t... Is,
            typename AllocatorT, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  constexpr decltype(auto) view(
      const AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>, index_sequence<Is...>,
                             AllocatorT> &ag,
      wrapv<Base> = {}) {
    return AdaptiveGridUnnamedView<ExecSpace,
                                   AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>,
                                                    index_sequence<Is...>, AllocatorT>,
                                   /*IsConst*/ true, Base>{ag};
  }
  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... SideLengthBits, size_t... Is,
            typename AllocatorT, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  constexpr decltype(auto) view(AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>,
                                                 index_sequence<Is...>, AllocatorT> &ag,
                                wrapv<Base> = {}) {
    return AdaptiveGridUnnamedView<ExecSpace,
                                   AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>,
                                                    index_sequence<Is...>, AllocatorT>,
                                   /*IsConst*/ false, Base>{ag};
  }

  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... SideLengthBits, size_t... Is,
            typename AllocatorT, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  constexpr decltype(auto) view(
      const AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>, index_sequence<Is...>,
                             AllocatorT> &ag,
      wrapv<Base>, const SmallString &tagName) {
    auto ret
        = AdaptiveGridUnnamedView<ExecSpace,
                                  AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>,
                                                   index_sequence<Is...>, AllocatorT>,
                                  /*IsConst*/ true, Base>{ag};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    (void)((ret.level(dim_c<Is>).grid._nameTag = tagName)...);
#endif
    return ret;
  }
  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... SideLengthBits, size_t... Is,
            typename AllocatorT, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  constexpr decltype(auto) view(AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>,
                                                 index_sequence<Is...>, AllocatorT> &ag,
                                wrapv<Base>, const SmallString &tagName) {
    auto ret
        = AdaptiveGridUnnamedView<ExecSpace,
                                  AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>,
                                                   index_sequence<Is...>, AllocatorT>,
                                  /*IsConst*/ false, Base>{ag};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    (void)((ret.level(dim_c<Is>).grid._nameTag = tagName)...);
#endif
    return ret;
  }

  template <execspace_e space, int dim, typename ValueT, size_t... SideLengthBits, size_t... Is,
            typename AllocatorT>
  constexpr decltype(auto) proxy(
      const AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>, index_sequence<Is...>,
                             AllocatorT> &ag) {
    return view<space>(ag, false_c);
  }
  template <execspace_e space, int dim, typename ValueT, size_t... SideLengthBits, size_t... Is,
            typename AllocatorT>
  constexpr decltype(auto) proxy(AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>,
                                                  index_sequence<Is...>, AllocatorT> &ag) {
    return view<space>(ag, false_c);
  }

  template <execspace_e space, int dim, typename ValueT, size_t... SideLengthBits, size_t... Is,
            typename AllocatorT>
  constexpr decltype(auto) proxy(
      const AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>, index_sequence<Is...>,
                             AllocatorT> &ag,
      const SmallString &tagName) {
    return view<space>(ag, false_c, tagName);
  }
  template <execspace_e space, int dim, typename ValueT, size_t... SideLengthBits, size_t... Is,
            typename AllocatorT>
  constexpr decltype(auto) proxy(AdaptiveGridImpl<dim, ValueT, index_sequence<SideLengthBits...>,
                                                  index_sequence<Is...>, AllocatorT> &ag,
                                 const SmallString &tagName) {
    return view<space>(ag, false_c, tagName);
  }

  ///
  /// @brief adaptive grid view with name tag
  ///
  template <execspace_e Space, typename AdaptiveGridT, bool IsConst, bool Base = false>
  struct AdaptiveGridView;

  /// @note Base here indicates whether grid should use lite version
  /// @note no allocator distinguish here
  template <execspace_e Space, int dim_, typename ValueT, size_t... SideLengthBits, size_t... Is,
            typename AllocatorT, bool IsConst, bool Base>
  struct AdaptiveGridView<Space,
                          AdaptiveGridImpl<dim_, ValueT, index_sequence<SideLengthBits...>,
                                           index_sequence<Is...>, AllocatorT>,
                          IsConst, Base>
      : AdaptiveGridUnnamedView<Space,
                                AdaptiveGridImpl<dim_, ValueT, index_sequence<SideLengthBits...>,
                                                 index_sequence<Is...>, AllocatorT>,
                                IsConst, Base>,
        LevelSetInterface<
            AdaptiveGridView<Space,
                             AdaptiveGridImpl<dim_, ValueT, index_sequence<SideLengthBits...>,
                                              index_sequence<Is...>, AllocatorT>,
                             IsConst, Base>> {
    /// @brief unnamed base view
    using base_t
        = AdaptiveGridUnnamedView<Space,
                                  AdaptiveGridImpl<dim_, ValueT, index_sequence<SideLengthBits...>,
                                                   index_sequence<Is...>, AllocatorT>,
                                  IsConst, Base>;
    static constexpr bool is_const_structure = IsConst;
    static constexpr auto space = Space;
    using container_type = typename base_t::container_type;
    static constexpr int dim = base_t::dim;
    using value_type = typename base_t::value_type;
    using size_type = typename base_t::size_type;
    using index_type = typename base_t::index_type;

    using integer_coord_component_type = typename base_t::integer_coord_component_type;
    using integer_coord_type = typename base_t::integer_coord_type;
    using coord_component_type = typename base_t::coord_component_type;
    using coord_type = typename base_t::coord_type;
    using packed_value_type = typename base_t::packed_value_type;

    // property
    template <int LevelNo> using level_type = typename base_t::template level_type<LevelNo>;
    template <int LevelNo> using level_view_type =
        typename base_t::template level_view_type<LevelNo>;

    using table_type = typename base_t::table_type;
    using table_view_type = typename base_t::table_view_type;

    // grid
    template <int LevelNo> using grid_storage_type =
        typename base_t::template grid_storage_type<LevelNo>;
    template <int LevelNo> using unnamed_grid_view_type =
        typename base_t::template unnamed_grid_view_type<LevelNo>;
    template <int LevelNo> using grid_view_type = RM_CVREF_T(
        view<space>(declval<conditional_t<is_const_structure, const grid_storage_type<LevelNo> &,
                                          grid_storage_type<LevelNo> &>>(),
                    wrapv<Base>{}));

    // mask
    template <int LevelNo> using mask_storage_type =
        typename base_t::template mask_storage_type<LevelNo>;
    template <int LevelNo> using mask_view_type = typename base_t::template mask_view_type<LevelNo>;

    using channel_counter_type = typename grid_storage_type<0>::channel_counter_type;

    /// @note TileVectorView-alike
    const SmallString *_tagNames;
    const channel_counter_type *_tagOffsets;
    const channel_counter_type *_tagSizes;
    channel_counter_type _N;
  };

}  // namespace zs