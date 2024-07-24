#pragma once
#include <memory>

#include "SparseGrid.hpp"
#include "zensim/types/Mask.hpp"

namespace zs {

  /// @note scaling bits k indicates 2^k cells (in 1D) corresponds to 1 cell in the parent level
  template <int dim, typename ValueT, typename TileBits = index_sequence<3, 4, 5>,
            typename ScalingBits = index_sequence<3, 4, 5>,
            typename Indices = index_sequence<0, 1, 2>, typename AllocatorT = ZSPmrAllocator<>>
  struct AdaptiveGridImpl;

  template <typename AdaptiveGridViewT, int NumDepths = 1> struct AdaptiveGridAccessor;

  /// @brief stores all leaf blocks of an adaptive octree including halo regions
  template <int dim_, typename ValueT, size_t... TileBits, size_t... ScalingBits, size_t... Is>
  struct AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                          index_sequence<Is...>, ZSPmrAllocator<>> {
    using value_type = ValueT;
    using allocator_type = ZSPmrAllocator<>;
    // using size_type = size_t;
    using size_type = u32;
    using index_type = zs::make_signed_t<size_type>;  // associated with the number of blocks
    using integer_coord_component_type = int;

    using coord_mask_type = make_unsigned_t<integer_coord_component_type>;

    /// @note sanity checks
    static_assert(sizeof...(TileBits) == sizeof...(ScalingBits)
                      && sizeof...(TileBits) == sizeof...(Is),
                  "???");
    static_assert(((ScalingBits <= TileBits && ScalingBits >= 0) && ...),
                  "scaling bits should be in [0, num_tile_bits]");

    static constexpr auto deduce_basic_value_type() noexcept {
      if constexpr (is_vec<value_type>::value)
        return wrapt<typename value_type::value_type>{};
      else
        return wrapt<value_type>{};
    }
    using coord_component_type = typename RM_CVREF_T(deduce_basic_value_type())::type;
    static_assert(is_floating_point_v<coord_component_type>,
                  "coord type should be floating point.");

    static constexpr int dim = dim_;
    static constexpr int num_levels = sizeof...(TileBits);

    /// @note hierarchy/ tile
    /// @note bits (bit_count)/ dim/ size

    /// @note tile_bits
    /// determine tile size at each level

    // tile + bits
    // using tile_bits_type = value_seq<TileBits...>;
    template <int I> static constexpr integer_coord_component_type get_tile_bits() noexcept {
      static_assert(I < num_levels, "queried level exceeds the range.");
      if constexpr (I < 0)
        return 0;
      else
        return (integer_coord_component_type)value_seq<TileBits...>::template value<I>;
    }
    template <int I> static constexpr integer_coord_component_type get_accum_tile_bits() noexcept {
      // static_assert(I >= 0 && I < num_levels, "???");
      // return decltype(declval<tile_bits_type>().template scan<1>())::template value<I>;
      integer_coord_component_type ret = 0;
      ((void)(ret += (Is <= I ? get_tile_bits<Is>() : (integer_coord_component_type)0)), ...);
      return ret;
    }

    /// @note  hierarchy_bits from child scaling_bits and tile_dim
    /// determine branch factor at each level

    // hierarchy + bits
    // using hierarchy_bits_type = value_seq<get_hierarchy_bits<Is>()...>;
    template <int I> static constexpr integer_coord_component_type get_scaling_bits() noexcept {
      static_assert(I < num_levels, "queried level exceeds the range.");
      if constexpr (I < 0)
        return 0;
      else
        return (integer_coord_component_type)value_seq<ScalingBits...>::template value<I>;
    }

    template <int I> static constexpr integer_coord_component_type get_hierarchy_bits() noexcept {
      static_assert(I < num_levels, "queried level exceeds the range.");
      if constexpr (I < 0)
        return 0;
      else
        return get_tile_bits<I>() - (get_tile_bits<I - 1>() - get_scaling_bits<I - 1>());
    }

    template <int I>
    static constexpr integer_coord_component_type get_accum_hierarchy_bits() noexcept {
      // return decltype(declval<hierarchy_bits_type>().template scan<1>())::template value<I>;
      integer_coord_component_type ret = 0;
      ((void)(ret += (Is <= I ? get_hierarchy_bits<Is>() : (integer_coord_component_type)0)), ...);
      return ret;
    }
    static_assert(((get_hierarchy_bits<Is>() > 0) && ...),
                  "parent node dimension should be greater than its child node.");

    /// @brief dim series
    template <int I> static constexpr integer_coord_component_type get_hierarchy_dim() noexcept {
      return (integer_coord_component_type)1 << get_hierarchy_bits<I>();
    }
    template <int I> static constexpr integer_coord_component_type get_tile_dim() noexcept {
      return (integer_coord_component_type)1 << get_tile_bits<I>();
    }
    template <int I>
    static constexpr integer_coord_component_type get_accum_hierarchy_dim() noexcept {
      return (integer_coord_component_type)1 << get_accum_hierarchy_bits<I>();
    }
    template <int I> static constexpr integer_coord_component_type get_accum_tile_dim() noexcept {
      return (integer_coord_component_type)1 << get_accum_tile_bits<I>();
    }

    /// @brief hierarchy
    template <int I> static constexpr integer_coord_component_type get_cell_bit_offset() noexcept {
      return get_accum_hierarchy_bits<I>() - get_tile_bits<I>();
    }

    /// @brief size series
    template <int I> static constexpr size_type get_hierarchy_size() noexcept {
      return math::pow_integral((size_type)get_hierarchy_dim<I>(), dim);
    }
    template <int I> static constexpr size_type get_tile_size() noexcept {
      return math::pow_integral((size_type)get_tile_dim<I>(), dim);
    }
    template <int I> static constexpr size_type get_accum_hierarchy_size() noexcept {
      return math::pow_integral((size_type)get_accum_hierarchy_dim<I>(), dim);
    }
    template <int I> static constexpr size_type get_accum_tile_size() noexcept {
      return math::pow_integral((size_type)get_accum_tile_dim<I>(), dim);
    }

    using integer_coord_type = vec<integer_coord_component_type, dim>;
    using coord_type = vec<coord_component_type, dim>;
    using packed_value_type = vec<value_type, dim>;

    /// @note beware of the difference (tile/ hierarchy) here!
    template <int I> using grid_storage_type
        = TileVector<value_type, get_tile_size<I>(), allocator_type>;
    template <int I> using tile_mask_storage_type
        = Vector<bit_mask<get_tile_size<I>()>, allocator_type>;
    template <int I> using hierarchy_mask_storage_type
        = Vector<bit_mask<get_hierarchy_size<I>()>, allocator_type>;

    using child_offset_type = Vector<size_type, allocator_type>;

    using table_type = bht<integer_coord_component_type, dim, index_type, 16, allocator_type>;
    static constexpr index_type sentinel_v = table_type::sentinel_v;

    using transform_type = math::Transform<coord_component_type, dim>;

    template <int level_> struct Level {
      static constexpr int level = level_;
      static constexpr integer_coord_component_type tile_bits = get_tile_bits<level_>();
      static constexpr integer_coord_component_type tile_dim = get_tile_dim<level_>();
      static constexpr integer_coord_component_type accum_tile_dim = get_accum_tile_dim<level_>();
      static constexpr size_type block_size = get_tile_size<level_>();

      /// @note used for global coords
      /// @note [sbit, ebit) is the bit region covered within this level
      static constexpr integer_coord_component_type sbit = get_cell_bit_offset<level_>();
      static constexpr integer_coord_component_type ebit = get_accum_hierarchy_bits<level_>();
      static constexpr coord_mask_type cell_mask = get_accum_hierarchy_dim<level_>() - 1;
      static constexpr coord_mask_type origin_mask = ~cell_mask;

      using grid_type = grid_storage_type<level_>;
      using tile_mask_type = tile_mask_storage_type<level_>;
      using hierarchy_mask_type = hierarchy_mask_storage_type<level_>;
      static_assert(block_size == grid_type::lane_width, "???");
      Level(const allocator_type &allocator, const std::vector<PropertyTag> &propTags, size_t count)
          : table{allocator, count},
            grid{allocator, propTags, count * block_size},
            valueMask{allocator, count},
            childMask{allocator, count},
            childOffset{allocator, count + 1} {
        defaultInitialize();
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

      size_type numBlocks() const { return table.size(); }
      size_type numReservedBlocks() const noexcept { return grid.numReservedTiles(); }

      void refitToPartition() {
        size_type nbs = numBlocks();
        grid.resize(nbs * block_size);
        resizeTopo(nbs);
      }
      void defaultInitialize() {
        grid.reset(0);
        valueMask.reset(0);
        childMask.reset(0);
        // childOffset will be maintained during reorder
      }
      template <typename ExecPolicy>
      void resize(ExecPolicy &&policy, size_type numBlocks, bool resizeGrid_ = true) {
        if (resizeGrid_) resizeGrid(numBlocks);
        resizePartition(FWD(policy), numBlocks);
        resizeTopo(numBlocks);
      }
      template <typename ExecPolicy>
      void resizePartition(ExecPolicy &&policy, size_type numBlocks) {
        table.resize(FWD(policy), numBlocks);
      }
      void resizeGrid(size_type numBlocks) { grid.resize(numBlocks * (size_type)block_size); }
      void resizeTopo(size_type numBlocks) {
        valueMask.resize(numBlocks);
        childMask.resize(numBlocks);
        childOffset.resize(numBlocks + 1);
      }

      template <typename Policy>
      void append_channels(Policy &&policy, const std::vector<PropertyTag> &tags,
                           const source_location &loc = source_location::current()) {
        grid.append_channels(FWD(policy), tags, loc);
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
        ret.childOffset = childOffset.clone(allocator);
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
#if ZS_ENABLE_SERIALIZATION
      template <typename S> void serialize(S &s) {
        serialize(s, table);
        serialize(s, grid);
        serialize(s, valueMask);
        serialize(s, childMask);
        serialize(s, childOffset);
      }
#endif

      table_type table;
      grid_type grid;
      /// @note for levelset, valueMask indicates inside/outside
      /// @note for leaf level, childMask reserved for special use cases
      tile_mask_type valueMask;
      hierarchy_mask_type childMask;
      child_offset_type childOffset;
    };

    template <auto I = 0> Level<I> &level(wrapv<I>) { return zs::get<I>(_levels); }
    template <auto I = 0> const Level<I> &level(wrapv<I>) const { return zs::get<I>(_levels); }
    template <auto I = 0> Level<I> &level(value_seq<I>) { return zs::get<I>(_levels); }
    template <auto I = 0> const Level<I> &level(value_seq<I>) const { return zs::get<I>(_levels); }

    template <int I> static constexpr integer_coord_component_type get_end_bits() noexcept {
      if constexpr (I < 0)
        return (integer_coord_component_type)0;
      else
        return Level<I>::ebit;
    }
    template <int I>
    static constexpr integer_coord_type coord_to_key(const integer_coord_type &c) noexcept {
      return c & Level<I>::origin_mask;
    }
    template <int I> static constexpr integer_coord_component_type coord_to_hierarchy_offset(
        const integer_coord_type &c) noexcept {
      integer_coord_component_type ret = (c[0] & Level<I>::cell_mask) >> get_end_bits<I - 1>();
      for (int d = 1; d != dim; ++d) {
        ret = (ret << get_hierarchy_bits<I>())
              | ((c[d] & Level<I>::cell_mask) >> get_end_bits<I - 1>());
      }
      return ret;
    }
    template <int I> static constexpr integer_coord_component_type coord_to_tile_offset(
        const integer_coord_type &c) noexcept {
      integer_coord_component_type ret = (c[0] & Level<I>::cell_mask) >> Level<I>::sbit;
      for (int d = 1; d != dim; ++d) {
        ret = (ret << get_tile_bits<I>()) | ((c[d] & Level<I>::cell_mask) >> Level<I>::sbit);
      }
      return ret;
    }
    template <int I>
    static constexpr integer_coord_type hierarchy_offset_to_coord(size_type offset) noexcept {
      integer_coord_type ret{};
      for (auto d = dim - 1; d >= 0; --d, offset >>= get_hierarchy_bits<I>())
        ret[d] = (integer_coord_component_type)(offset & (get_hierarchy_dim<I>() - 1))
                 << get_end_bits<I - 1>();
      return ret;
    }
    template <int I>
    static constexpr integer_coord_type tile_offset_to_coord(size_type offset) noexcept {
      integer_coord_type ret{};
      for (auto d = dim - 1; d >= 0; --d, offset >>= get_tile_bits<I>())
        ret[d] = (integer_coord_component_type)(offset & (get_tile_dim<I>() - 1)) << Level<I>::sbit;
      return ret;
    }

    constexpr MemoryLocation memoryLocation() const noexcept {
      return get<0>(_levels).table.memoryLocation();
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
    constexpr void nodeCount(std::vector<size_type> &cnts) {
      ((void)(cnts[Is] = numBlocks(wrapv<Is>{})), ...);
    }
    constexpr size_t numTotalBlocks() const noexcept {
      size_t ret = 0;
      ((void)(ret += numBlocks(wrapv<Is>{})), ...);
      return ret;
    }
    template <size_t I = 0> constexpr auto numReservedBlocks() const noexcept {
      return get<I>(_levels).numReservedBlocks();
    }
    template <typename Policy>
    void append_channels(Policy &&policy, const std::vector<PropertyTag> &tags,
                         const source_location &loc = source_location::current()) {
      ((void)get<Is>(_levels).append_channels(FWD(policy), tags, loc), ...);
    }
    void reset(value_type val) { ((void)get<Is>(_levels).reset(val), ...); }
    // value-wise reset
    template <typename Policy> void reset(Policy &&policy, value_type val) {
      ((void)get<Is>(_levels).reset(policy, val), ...);
    }

    ///
    ///
    /// @brief maintain topo (i.e. childMask, table) in a bottom-up fashion
    /// @note mostly used for restructure and conversion from sparsegrid
    ///
    ///
    struct _build_level_entry {
      template <typename ParamT> constexpr void operator()(size_type i, ParamT &&params) {
        auto &[keys, parTab, agTag, lvlTag] = params;
        constexpr int level = RM_REF_T(lvlTag)::value;
        using AgT = typename RM_REF_T(agTag)::type;
        parTab.insert(AgT::template coord_to_key<level>(keys[i]));
      }
    };
    struct _update_topo_update_childmask {
      template <typename ParamT> constexpr void operator()(size_type wordNo, ParamT &&params) {
        auto &[chTab, origins, childMask, lvlTag] = params;
        constexpr int level_no = RM_REF_T(lvlTag)::value;
        using AgT = AdaptiveGridImpl;
        using hierarchy_bitmask_type = typename Level<level_no>::hierarchy_mask_type::value_type;
        using word_type = typename hierarchy_bitmask_type::word_type;
        size_type bno = wordNo / hierarchy_bitmask_type::word_count;
        integer_coord_type origin = origins[bno];
        wordNo %= hierarchy_bitmask_type::word_count;
        constexpr int bits_per_word = hierarchy_bitmask_type::bits_per_word;
        size_type hierarchyOffset = wordNo * bits_per_word;
        word_type mask = 0;
        for (int i = 0;
             i != bits_per_word && hierarchyOffset < AgT::template get_hierarchy_size<level_no>();
             ++i, ++hierarchyOffset) {
          auto coord = origin + AgT::template hierarchy_offset_to_coord<level_no>(hierarchyOffset);
          if (chTab.query(coord) != sentinel_v) mask |= ((word_type)1 << (word_type)i);
        }
        childMask[bno].words[wordNo] = mask;
      }
    };
#if 0
    struct _update_topo_update_parent_childmask {
      template <typename ParamT> constexpr void operator()(size_type i, ParamT &&params) {
        auto [keys, parTab, parChMask, agTag, parLvlTag] = params;
        constexpr int level = RM_REF_T(parLvlTag)::value;
        using AgT = typename RM_REF_T(agTag)::type;
        integer_coord_type key = keys[i];
        index_type bno = parTab.query(AgT::template coord_to_key<level>(key));
        parChMask[bno].setOn(AgT::template coord_to_hierarchy_offset<level>(key),
                             wrapv<RM_REF_T(keys)::space>{});
      }
    };
#endif
    template <typename Policy, bool SortGridData = true,
              execspace_e space = remove_reference_t<Policy>::exec_tag::value>
    void complementTopo(Policy &&pol, wrapv<SortGridData> = {});

    ///
    ///
    /// @brief restructure (compute childMask upon hash tables)
    ///
    ///
    template <integer_coord_component_type CellBits, int I = num_levels - 1>
    static constexpr int closest_child_level(wrapv<CellBits> parCbs, wrapv<I> = {}) {
      if constexpr (I <= 0)
        return 0;
      else if constexpr (Level<I>::sbit <= CellBits)
        return I;
      else
        return closest_child_level(parCbs, wrapv<I - 1>{});
    }
    struct _restructure_hash_active_cell {
      template <typename ParamT> constexpr void operator()(size_type i, ParamT &&params) {
        auto &[valueMask, keys, dstTab, srcAgTag, srcLvlTag, dstAgTag, dstLvlTag] = params;
        constexpr int level = RM_REF_T(srcLvlTag)::value;
        constexpr int dst_level = RM_REF_T(dstLvlTag)::value;
        using AgT = typename RM_REF_T(srcAgTag)::type;
        using DstAgT = typename RM_REF_T(dstAgTag)::type;
        constexpr size_type block_size = AgT::template Level<level>::block_size;
        auto bno = i / block_size;
        const auto &vm = valueMask[bno];
        auto cno = i & (block_size - 1);
        if (vm.isOn(cno)) {
          auto coord = keys[bno] + AgT::template tile_offset_to_coord<level>(cno);
          dstTab.insert(DstAgT::template coord_to_key<dst_level>(coord));
        }
      }
    };
#if 0
    struct _restructure_0 {
      template <typename Tup> constexpr void operator()(int i, integer_coord_type &c, Tup &&) {
        u64 sd = i;
        for (int d = 0; d != dim_; ++d) {
          auto val = zs::PCG::pcg32_random_r(sd, 1442695040888963407ull);
          c[d] = val % 10000;
        }
      }
    };
#endif
    struct _restructure_assign_values {
      template <typename ParamT> constexpr void operator()(size_type i, ParamT &&params) {
        auto &[agv, lvDst] = params;

        constexpr size_type block_size = RM_REF_T(lvDst)::block_size;
        constexpr int level = RM_REF_T(lvDst)::level;
        using AgvDstT = typename RM_REF_T(lvDst)::ag_view_type;
        auto bno = i / block_size;
        auto cno = i & (block_size - 1);
        auto coord
            = lvDst.table._activeKeys[bno] + AgvDstT::template tile_offset_to_coord<level>(cno);
        auto acc = agv.getAccessor();
        bool hasValue = false;
        value_type val{};  // beware, this is a constexpr func
        auto block = lvDst.grid.tile(bno);
        for (int d = 0; d != lvDst.numChannels(); ++d) {
          // hasValue = agv.probeValue(d, coord, val);
          hasValue |= acc.probeValue(d, coord, val);
          block(d, cno) = val;
        }
        if (hasValue) {
          lvDst.valueMask[bno].setOn(cno, wrapv<AgvDstT::space>{});
        }
      }
    };
#if 0
    struct _restructure_register_target_blocks {
      template <typename ParamT> constexpr void operator()(size_type i, ParamT &&params) {
        auto [keys, dstTab, num_child_dim_c, shift_bits_c] = params;

        auto key = keys[i];
        for (auto iter : ndrange<dim>(RM_CVREF_T(num_child_dim_c)::value)) {
          auto offset = make_vec<integer_coord_component_type>(iter)
                        << RM_CVREF_T(shift_bits_c)::value;
          dstTab.insert(key + offset);
        }
      }
    };
    struct _restructure_assign_values {
      template <typename ParamT> constexpr void operator()(size_type i, ParamT &&params) {
        auto [origins, srcValueMask, srcGrid, dstTab, dstValueMask, dstGrid, bsTag, rel_dim_c,
              srcAgTag, srcLvlTag, dstAgTag, dstLvlTag, execTag]
            = params;
        constexpr auto block_size = RM_REF_T(bsTag)::value;
        constexpr auto rel_dim = RM_REF_T(rel_dim_c)::value;
        constexpr int level = RM_REF_T(srcLvlTag)::value;
        constexpr int dst_level = RM_REF_T(dstLvlTag)::value;
        using AgT = typename RM_REF_T(srcAgTag)::type;
        using DstAgT = typename RM_REF_T(dstAgTag)::type;
        auto bno = i / block_size;
        const auto &vm = srcValueMask[bno];
        auto cno = i & (block_size - 1);
        if (vm.isOn(cno)) {
          auto origin = origins[bno] + AgT::template tile_offset_to_coord<level>(cno);
          auto &dstVm
              = dstValueMask[dstTab.query(DstAgT::template coord_to_key<dst_level>(origin))];
          for (auto iter : ndrange<dim>(rel_dim)) {
            auto coord = origin
                         + (make_vec<integer_coord_component_type>(iter)
                            << DstAgT::template Level<dst_level>::sbit);
            auto dstCno = DstAgT::template coord_to_tile_offset<dst_level>(coord);
            dstVm.setOn(dstCno);
            for (int d = 0; d != srcGrid.numChannels(); ++d) {
              dstGrid(d, dstCno) = srcGrid(d, cno);
            }
          }
        }
      }
    };
#endif
    template <typename Policy, typename ValueDst, size_t... TileBitsDst, size_t... ScalingBitsDst,
              size_t... Js, execspace_e space = remove_reference_t<Policy>::exec_tag::value>
    void restructure(Policy &&pol,
                     AdaptiveGridImpl<dim, ValueDst, index_sequence<TileBitsDst...>,
                                      index_sequence<ScalingBitsDst...>, index_sequence<Js...>,
                                      ZSPmrAllocator<>> &agDst);

    ///
    ///
    /// @brief reorder (compute childMask upon hash tables)
    ///
    ///
    template <int I> struct _reorder_init_kv {
      constexpr void operator()(size_type i, const integer_coord_type &coord, u32 &key,
                                size_type &id) const noexcept {
        auto c = ((coord >> Level<I>::sbit) & 1023).template cast<f32>() / 1024;
        key = morton_code<dim>(c);
        id = i;
      }
    };
    struct _reorder_count_on {
      template <typename MaskT, typename ParamT>
      constexpr void operator()(size_type &n, const MaskT &mask, ParamT &&params) noexcept {
        auto &[tag] = params;
        n = mask.countOn(tag);
      }
    };
    template <int I> struct _reorder_locate_children {
      template <typename ParamT>
      constexpr void operator()(size_type i, const integer_coord_type &coord, size_type &dst,
                                const ParamT &params) noexcept {
        auto &[tb, offsets, masks] = params;
        auto parentOrigin = coord_to_key<I>(coord);
        auto parentBno = tb.query(parentOrigin);
        auto childOffset = coord_to_hierarchy_offset<I>(coord);
        dst = offsets[parentBno]
              + masks[parentBno].countOffset(childOffset, wrapv<RM_REF_T(tb)::space>{});
      }
    };
    template <typename Policy, bool SortGridData = true, int I = num_levels - 1>
    void reorder(Policy &&pol, wrapv<SortGridData> = {}, wrapv<I> = {}) {
      constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;
      static_assert(I > 0 && I < num_levels, "???");

      if (!valid_memspace_for_execution(pol, get_allocator()))
        throw std::runtime_error(
            "[AdaptiveGrid::reorder] current memory location not compatible with the execution "
            "policy");

      auto &l = level(wrapv<I>{});
      auto &lc = level(wrapv<I - 1>{});
      /// @note implicit sync here
      auto nbs = l.numBlocks();
      auto nchbs = lc.numBlocks();

      auto allocator = get_temporary_memory_source(pol);
      /// pre-sort highest level
      if constexpr (I == num_levels - 1) {
        Vector<u32> codes{allocator, nbs};
        Vector<u32> sortedMcs{allocator, nbs};
        Vector<size_type> indices{allocator, nbs};
        Vector<size_type> sortedIndices{allocator, nbs};

        pol(enumerate(l.table._activeKeys, codes, indices), _reorder_init_kv<I>{});
        radix_sort_pair(pol, codes.begin(), indices.begin(), sortedMcs.begin(),
                        sortedIndices.begin(), nbs);
        /// reorder block entries
        l.table.reorder(pol, range(sortedIndices), false_c);
        /// reorder grid data
        if constexpr (SortGridData) l.grid.reorderTiles(pol, range(sortedIndices), false_c);
        l.valueMask.reorder(pol, range(sortedIndices), false_c);
        l.childMask.reorder(pol, range(sortedIndices), false_c);
      }
      /// histogram sort
      Vector<size_type> numActiveChildren{allocator, nbs + 1};
      pol(zip(numActiveChildren, l.childMask), zs::make_tuple(wrapv<space>{}), _reorder_count_on{});
      exclusive_scan(pol, numActiveChildren.begin(), numActiveChildren.end(),
                     l.childOffset.begin());
      // fmt::print("done level [{}] histogram sort.\n", I);

      /// DEBUG
      if constexpr (false) {
        Vector<size_type> numOn{allocator, 1};
        reduce(pol, numActiveChildren.begin(), numActiveChildren.end() - 1, numOn.begin());
        fmt::print("computed [{}] num active children, in reality [{}] children\n", numOn.getVal(),
                   nchbs);
      }
      /// DEBUG
      /// compute children reorder mapping
      Vector<size_type> dsts{allocator, nchbs};
      {
        const auto &params = zs::make_tuple(view<space>(l.table), view<space>(l.childOffset),
                                            view<space>(l.childMask));
        pol(enumerate(lc.table._activeKeys, dsts), params, _reorder_locate_children<I>{});
      }
      // fmt::print("done level [{}] locating children.\n", I);

      // pol(enumerate(dsts),
      //    [nchbs] ZS_LAMBDA(size_type i, size_type & dst) { dst = nchbs - 1 - i; });
      /// reorder block entries
      lc.table.reorder(pol, range(dsts), true_c);
      /// reorder grid data
      if constexpr (SortGridData) lc.grid.reorderTiles(pol, range(dsts), true_c);
      lc.valueMask.reorder(pol, range(dsts), true_c);
      lc.childMask.reorder(pol, range(dsts), true_c);

      /// recurse
      if constexpr (I > 1) {
        reorder(FWD(pol), wrapv<SortGridData>{}, wrapv<I - 1>{});
      }
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
      ((void)(get<Is>(ret._levels) = get<Is>(_levels).clone(allocator)), ...);
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

  // openvdb float grid
  ZPC_FWD_DECL_TEMPLATE_STRUCT
  AdaptiveGridImpl<3, f32, index_sequence<3, 4, 5>, index_sequence<3, 4, 5>,
                   index_sequence<0, 1, 2>, ZSPmrAllocator<>>;
  // bifrost adaptive tile tree
  ZPC_FWD_DECL_TEMPLATE_STRUCT
  AdaptiveGridImpl<3, f32, index_sequence<2, 2, 2>, index_sequence<1, 1, 1>,
                   index_sequence<0, 1, 2>, ZSPmrAllocator<>>;

  /// @note floatgrid: <3, f32, 3, 4, 5>
  template <int dim, typename ValueT, typename BitSeq, typename AllocatorT = ZSPmrAllocator<>>
  struct VdbGrid;

  template <int dim, typename ValueT, size_t... Ns, typename AllocatorT>
  struct VdbGrid<dim, ValueT, index_sequence<Ns...>, AllocatorT>
      : AdaptiveGridImpl<dim, ValueT, index_sequence<Ns...>, index_sequence<Ns...>,
                         make_index_sequence<sizeof...(Ns)>, AllocatorT> {
    using base_t = AdaptiveGridImpl<dim, ValueT, index_sequence<Ns...>, index_sequence<Ns...>,
                                    make_index_sequence<sizeof...(Ns)>, AllocatorT>;
    VdbGrid() = default;
    ~VdbGrid() = default;
    VdbGrid(base_t &&ag) : base_t{zs::move(ag)} {}
    VdbGrid(const base_t &ag) : base_t{ag} {}

    VdbGrid clone(const typename base_t::allocator_type &allocator) const {
      return base_t::clone(allocator);
    }
    VdbGrid clone(const zs::MemoryLocation &mloc) const { return base_t::clone(mloc); }
  };

  ZPC_FWD_DECL_TEMPLATE_STRUCT
  VdbGrid<3, f32, index_sequence<3, 4, 5>, ZSPmrAllocator<>>;

  /// @note bifrost adaptive tile tree: <3, f32, 3, 2>
  template <int dim, typename ValueT, int NumLevels, size_t N,
            typename AllocatorT = ZSPmrAllocator<>>
  struct AdaptiveTileTree
      : AdaptiveGridImpl<dim, ValueT, typename build_seq<NumLevels>::template constant<N>,
                         typename build_seq<NumLevels>::template constant<(size_t)1>,
                         typename build_seq<NumLevels>::ascend, AllocatorT> {
    using base_t = AdaptiveGridImpl<dim, ValueT, typename build_seq<NumLevels>::template constant<N>,
                                    typename build_seq<NumLevels>::template constant<(size_t)1>,
                                    typename build_seq<NumLevels>::ascend, AllocatorT>;
    AdaptiveTileTree() = default;
    ~AdaptiveTileTree() = default;
    AdaptiveTileTree(base_t &&ag) : base_t{zs::move(ag)} {}
    AdaptiveTileTree(const base_t &ag) : base_t{ag} {}

    AdaptiveTileTree clone(const typename base_t::allocator_type &allocator) const {
      return base_t::clone(allocator);
    }
    AdaptiveTileTree clone(const zs::MemoryLocation &mloc) const { return base_t::clone(mloc); }
  };

  ZPC_FWD_DECL_TEMPLATE_STRUCT
  AdaptiveTileTree<3, f32, 3, 2, ZSPmrAllocator<>>;

  /// @brief adaptive grid predicate
  template <int dim, typename ValueT, size_t... TileBits, size_t... ScalingBits, size_t... Is,
            typename AllocatorT>
  struct is_ag<AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>,
                                index_sequence<ScalingBits...>, index_sequence<Is...>, AllocatorT>>
      : true_type {};
  // vdb
  template <int dim, typename ValueT, typename BitSeq, typename AllocatorT>
  struct is_ag<VdbGrid<dim, ValueT, BitSeq, AllocatorT>> : true_type {};
  // adaptive tile tree
  template <int dim, typename ValueT, int NumLevels, size_t N, typename AllocatorT>
  struct is_ag<AdaptiveTileTree<dim, ValueT, NumLevels, N, AllocatorT>> : true_type {};

  ///
  /// @brief adaptive grid unnamed view
  ///
  template <execspace_e Space, typename AdaptiveGridT, bool IsConst, bool Base = false>
  struct AdaptiveGridUnnamedView;

  /// @note Base here indicates whether grid should use lite version
  /// @note this specializations is targeting ZSPmrAllocator<false> specifically
  template <execspace_e Space, int dim_, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, bool IsConst, bool Base>
  struct AdaptiveGridUnnamedView<
      Space,
      AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                       index_sequence<Is...>, ZSPmrAllocator<>>,
      IsConst, Base>
      : LevelSetInterface<
            AdaptiveGridUnnamedView<Space,
                                    AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>,
                                                     index_sequence<ScalingBits...>,
                                                     index_sequence<Is...>, ZSPmrAllocator<>>,
                                    IsConst, Base>> {
    static constexpr bool is_const_structure = IsConst;
    static constexpr auto space = Space;
    using container_type
        = AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>,
                           index_sequence<ScalingBits...>, index_sequence<Is...>, ZSPmrAllocator<>>;
    static constexpr int dim = container_type::dim;
    static constexpr int num_levels = container_type::num_levels;
    using value_type = typename container_type::value_type;
    using size_type = typename container_type::size_type;
    using index_type = typename container_type::index_type;

    using integer_coord_component_type = typename container_type::integer_coord_component_type;
    using coord_mask_type = typename container_type::coord_mask_type;
    using integer_coord_type = typename container_type::integer_coord_type;
    using coord_component_type = typename container_type::coord_component_type;
    using coord_type = typename container_type::coord_type;
    using packed_value_type = typename container_type::packed_value_type;

    template <int LevelNo> using level_type = typename container_type::template Level<LevelNo>;

    using child_offset_type = typename container_type::child_offset_type;
    using child_offset_view_type = decltype(view<space>(
        declval<
            conditional_t<is_const_structure, const child_offset_type &, child_offset_type &>>(),
        wrapv<Base>{}));

    using table_type = typename container_type::table_type;
    using table_view_type = decltype(view<space>(
        declval<conditional_t<is_const_structure, const table_type &, table_type &>>(),
        wrapv<Base>{}));
    static constexpr index_type sentinel_v = table_type::sentinel_v;

    // grid
    template <int LevelNo> using grid_storage_type = typename level_type<LevelNo>::grid_type;
    template <int LevelNo> using unnamed_grid_view_type = decltype(view<space>(
        declval<conditional_t<is_const_structure, const grid_storage_type<LevelNo> &,
                              grid_storage_type<LevelNo> &>>(),
        wrapv<Base>{}));

    // mask
    template <int LevelNo> using tile_mask_storage_type =
        typename level_type<LevelNo>::tile_mask_type;
    template <int LevelNo> using tile_mask_view_type = decltype(view<space>(
        declval<conditional_t<is_const_structure, const tile_mask_storage_type<LevelNo> &,
                              tile_mask_storage_type<LevelNo> &>>(),
        wrapv<Base>{}));
    template <int LevelNo> using hierarchy_mask_storage_type =
        typename level_type<LevelNo>::hierarchy_mask_type;
    template <int LevelNo> using hierarchy_mask_view_type = decltype(view<space>(
        declval<conditional_t<is_const_structure, const hierarchy_mask_storage_type<LevelNo> &,
                              hierarchy_mask_storage_type<LevelNo> &>>(),
        wrapv<Base>{}));

    template <int LevelNo> struct level_view_type {
      static constexpr int level = LevelNo;
      using level_t = level_type<level>;
      using ag_view_type = AdaptiveGridUnnamedView;
      using grid_type = unnamed_grid_view_type<level>;
      using tile_mask_type = tile_mask_view_type<level>;
      using hierarchy_mask_type = hierarchy_mask_view_type<level>;

      constexpr level_view_type() noexcept = default;
      ~level_view_type() = default;
      level_view_type(conditional_t<is_const_structure, add_const_t<level_t>, level_t> &level)
          : table{view<Space>(level.table, wrapv<Base>{})},
            grid{view<Space>(level.grid, wrapv<Base>{})},
            valueMask{view<Space>(level.valueMask, wrapv<Base>{})},
            childMask{view<Space>(level.childMask, wrapv<Base>{})},
            childOffset{view<Space>(level.childOffset, wrapv<Base>{})} {}

      static constexpr integer_coord_component_type tile_bits = level_t::tile_bits;
      static constexpr integer_coord_component_type tile_dim = level_t::tile_dim;
      static constexpr integer_coord_component_type accum_tile_dim = level_t::accum_tile_dim;
      static constexpr size_type block_size = level_t::block_size;

      /// @note used for global coords
      static constexpr integer_coord_component_type sbit = level_t::sbit;
      static constexpr integer_coord_component_type ebit = level_t::ebit;
      static constexpr coord_mask_type cell_mask = level_t::cell_mask;
      static constexpr coord_mask_type origin_mask = level_t::origin_mask;

      constexpr auto numBlocks() const { return table.size(); }
      constexpr auto numChannels() const { return grid.numChannels(); }

      table_view_type table;
      grid_type grid;
      tile_mask_type valueMask;
      hierarchy_mask_type childMask;
      child_offset_view_type childOffset;
    };

    using transform_type = typename container_type::transform_type;

    template <auto I = 0, bool isConst = is_const_structure, enable_if_t<!isConst> = 0>
    constexpr decltype(auto) level(wrapv<I>) {
      return zs::get<I>(_levels);
    }
    template <auto I = 0> constexpr decltype(auto) level(wrapv<I>) const {
      return zs::get<I>(_levels);
    }
    template <auto I = 0, bool isConst = is_const_structure, enable_if_t<!isConst> = 0>
    constexpr decltype(auto) level(value_seq<I>) {
      return zs::get<I>(_levels);
    }
    template <auto I = 0> constexpr decltype(auto) level(value_seq<I>) const {
      return zs::get<I>(_levels);
    }

    constexpr AdaptiveGridUnnamedView() noexcept = default;
    ~AdaptiveGridUnnamedView() noexcept = default;
    constexpr AdaptiveGridUnnamedView(
        conditional_t<is_const_structure, add_const_t<container_type>, container_type> &ag)
        : _levels{}, _transform{ag._transform}, _background{ag._background} {
      ((void)(zs::get<Is>(_levels) = level_view_type<Is>{ag.level(dim_c<Is>)}), ...);
    }

    template <int L = num_levels> constexpr auto getAccessor(wrapv<L> = {}) const {
      return AdaptiveGridAccessor<const AdaptiveGridUnnamedView, L>(this);
    }
    template <int L = num_levels> constexpr auto getAccessor(wrapv<L> = {}) {
      return AdaptiveGridAccessor<AdaptiveGridUnnamedView, L>(this);
    }

    /// @brief utility
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
      return container_type::template coord_to_key<I>(c);
    }
    template <int I> static constexpr integer_coord_component_type coord_to_tile_offset(
        const integer_coord_type &c) noexcept {
      return container_type::template coord_to_tile_offset<I>(c);
    }
    template <int I> static constexpr integer_coord_component_type coord_to_hierarchy_offset(
        const integer_coord_type &c) noexcept {
      return container_type::template coord_to_hierarchy_offset<I>(c);
    }
    template <int I>
    static constexpr integer_coord_type hierarchy_offset_to_coord(size_type offset) noexcept {
      return container_type::template hierarchy_offset_to_coord<I>(offset);
    }
    template <int I>
    static constexpr integer_coord_type tile_offset_to_coord(size_type offset) noexcept {
      return container_type::template tile_offset_to_coord<I>(offset);
    }

    template <bool Ordered = false, int I = num_levels - 1>
    constexpr int getValueLevel(const integer_coord_type &coord, index_type bno = sentinel_v,
                                wrapv<Ordered> = {}, wrapv<I> = {}) const {
      auto &lev = level(dim_c<I>);
      if (bno == sentinel_v) {
        auto c = coord_to_key<I>(coord);
        bno = lev.table.query(c);
        if (bno == sentinel_v) {
          return I;
        }
      }
      const integer_coord_component_type n = coord_to_hierarchy_offset<I>(coord);
      auto block = lev.grid.tile(bno);
      if constexpr (I > 0) {
        if (lev.childMask[bno].isOff(n)) {
          return I;
        }
        if constexpr (Ordered)
          return getValueLevel(
              coord, lev.childOffset[bno] + lev.childMask[bno].countOffset(n, wrapv<space>{}),
              wrapv<Ordered>{}, wrapv<I - 1>{});
        else
          return getValueLevel(coord, sentinel_v, wrapv<Ordered>{}, wrapv<I - 1>{});
      } else {
        return 0;
      }
    }
    /// @note only requires the validity of table, valueMask and grid data
    template <typename T, int I = num_levels - 1>
    constexpr enable_if_type<!is_const_v<T>, bool> topoObliviousProbeValue(
        size_type chn, const integer_coord_type &coord, T &val, index_type bno = sentinel_v,
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
      const integer_coord_component_type n = coord_to_tile_offset<I>(coord);
      auto block = lev.grid.tile(bno);
      if constexpr (I > 0) {
        if (lev.valueMask[bno].isOn(n)) {
          if constexpr (IsVec) {
            for (int d = 0; d != T::extent; ++d) val.val(d) = block(chn + d, n);
          } else
            val = block(chn, n);
          return true;
        } else {
          return topoObliviousProbeValue(chn, coord, val, sentinel_v, wrapv<I - 1>{});
        }
      } else {
        /// @note leaf level
        if constexpr (IsVec) {
          for (int d = 0; d != T::extent; ++d) val.val(d) = block(chn + d, n);
        } else
          val = block(chn, n);
        return lev.valueMask[bno].isOn(n);
      }
    }
    template <typename T, bool Ordered = false, int I = num_levels - 1>
    constexpr enable_if_type<!is_const_v<T>, bool> probeValue(size_type chn,
                                                              const integer_coord_type &coord,
                                                              T &val, index_type bno = sentinel_v,
                                                              wrapv<Ordered> = {},
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
      auto block = lev.grid.tile(bno);
      if constexpr (I > 0) {
        integer_coord_component_type n = coord_to_hierarchy_offset<I>(coord);
        // printf("found int-%d [%d] (%d, %d, %d) ch[%d]\n", I, bno, lev.table._activeKeys[bno][0],
        //        lev.table._activeKeys[bno][1], lev.table._activeKeys[bno][2], n);
        /// @note internal level
        if (lev.childMask[bno].isOff(n)) {
          n = coord_to_tile_offset<I>(coord);
          if constexpr (IsVec) {
            for (int d = 0; d != T::extent; ++d) val.val(d) = block(chn + d, n);
          } else
            val = block(chn, n);
          return lev.valueMask[bno].isOn(n);
        }
        /// TODO: an optimal layout should directly give child-n position
        if constexpr (Ordered)
          return probeValue(
              chn, coord, val,
              lev.childOffset[bno] + lev.childMask[bno].countOffset(n, wrapv<space>{}),
              wrapv<Ordered>{}, wrapv<I - 1>{});
        else
          return probeValue(chn, coord, val, sentinel_v, wrapv<Ordered>{}, wrapv<I - 1>{});
      } else {
        const integer_coord_component_type n = coord_to_tile_offset<I>(coord);
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
    template <typename AccessorAgView, int AccessorDepths, typename T, bool Ordered = false,
              int I = num_levels - 1, bool RequireHash = true, enable_if_t<!is_const_v<T>> = 0>
    constexpr bool probeValueAndCache(AdaptiveGridAccessor<AccessorAgView, AccessorDepths> &acc,
                                      size_type chn, const integer_coord_type &coord, T &val,
                                      index_type bno, wrapv<Ordered>, wrapv<I>,
                                      wrapv<RequireHash>) const {
      constexpr bool IsVec = is_vec<T>::value;
      auto &lev = level(dim_c<I>);
      if constexpr (RequireHash) {
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
      if constexpr (I == 0) {
        const integer_coord_component_type n = coord_to_tile_offset<I>(coord);
        acc.insert(coord, bno, wrapv<num_levels - 1 - I>{});
        auto block = lev.grid.tile(bno);
        /// @note leaf level
        if constexpr (IsVec) {
          for (int d = 0; d != T::extent; ++d) val.val(d) = block(chn + d, n);
        } else
          val = block(chn, n);
        return lev.valueMask[bno].isOn(n);
      }
      if constexpr (I > 0) {
        integer_coord_component_type n = coord_to_hierarchy_offset<I>(coord);
        /// @note internal level
        if (lev.childMask[bno].isOn(n)) {
          acc.insert(coord, bno, wrapv<num_levels - 1 - I>{});
          /// TODO: an optimal layout should directly give child-n position
          if constexpr (Ordered)
            return probeValueAndCache(
                acc, chn, coord, val,
                lev.childOffset[bno] + lev.childMask[bno].countOffset(n, wrapv<space>{}),
                wrapv<Ordered>{}, wrapv<I - 1>{}, wrapv<false>{});
          else
            return probeValueAndCache(acc, chn, coord, val, sentinel_v, wrapv<Ordered>{},
                                      wrapv<I - 1>{}, wrapv<true>{});
        } else {
          auto block = lev.grid.tile(bno);
          n = coord_to_tile_offset<I>(coord);
          if constexpr (IsVec) {
            for (int d = 0; d != T::extent; ++d) val.val(d) = block(chn + d, n);
          } else
            val = block(chn, n);
          return lev.valueMask[bno].isOn(n);  // isTileOn (root), isValueMaskOn (internal)
        }
      }
    }
    
    template <int I> constexpr value_type valueOr(size_type chn,
                                                  typename table_type::index_type blockno,
                                                  integer_coord_component_type cellno,
                                                  value_type defaultVal, wrapv<I>) const noexcept {
      const auto &l = level(wrapv<I>{});
      return blockno == table_type::sentinel_v ? defaultVal : l.grid(chn, blockno, cellno);
    }
    constexpr auto valueOr(false_type, size_type chn, const integer_coord_type &indexCoord,
                           value_type defaultVal) const noexcept {
      value_type ret{};
      auto found = probeValue(chn, indexCoord, ret, sentinel_v, wrapv<true>{});
      if (found)
        return ret;
      else
        return defaultVal;
    }
    constexpr value_type valueOr(true_type, size_type chn, integer_coord_type indexCoord,
                                 int orientation, value_type defaultVal) const noexcept {
      /// 0, ..., dim-1: within cell
      /// dim, ..., dim+dim-1: neighbor cell
      if (int f = orientation % (dim + dim); f >= dim) ++indexCoord[f - dim];
      return valueOr(false_c, chn, indexCoord, defaultVal);
    }
    template <int I> constexpr value_type value(size_type chn,
                                                typename table_type::index_type blockno,
                                                integer_coord_component_type cellno,
                                                value_type defaultVal, wrapv<I>) const noexcept {
      const auto &l = level(wrapv<I>{});
      return blockno == table_type::sentinel_v ? defaultVal : l.grid(chn, blockno, cellno);
    }
    constexpr auto value(false_type, size_type chn,
                         const integer_coord_type &indexCoord) const noexcept {
      value_type ret{};
      auto found = probeValue(chn, indexCoord, ret, sentinel_v, wrapv<true>{});
      return ret;
    }
    constexpr value_type value(true_type, size_type chn, integer_coord_type indexCoord,
                               int orientation) const noexcept {
      /// 0, ..., dim-1: within cell
      /// dim, ..., dim+dim-1: neighbor cell
      if (int f = orientation % (dim + dim); f >= dim) ++indexCoord[f - dim];
      return value(false_c, chn, indexCoord);
    }

    // arena
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iArena(const VecInterface<VecT> &X, wrapv<kt> = {}) const {
      return GridArena<const AdaptiveGridUnnamedView, kt, 0>(false_c, this, X);
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wArena(const VecInterface<VecT> &x, wrapv<kt> = {}) const {
      return GridArena<const AdaptiveGridUnnamedView, kt, 0>(false_c, this, worldToIndex(x));
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iArena(const VecInterface<VecT> &X, int f, wrapv<kt> = {}) const {
      return GridArena<const AdaptiveGridUnnamedView, kt, 0>(false_c, this, X, f);
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wArena(const VecInterface<VecT> &x, int f, wrapv<kt> = {}) const {
      return GridArena<const AdaptiveGridUnnamedView, kt, 0>(false_c, this, worldToIndex(x), f);
    }

    /// @brief voxel size
    constexpr coord_type voxelSize() const {
      // neglect shearing here
      coord_type ret{};
      for (int i = 0; i != dim; ++i) {
        coord_component_type sum = 0;
        for (int d = 0; d != dim; ++d) sum += zs::sqr(_transform(i, d));
        ret.val(i) = zs::sqrt(sum, wrapv<space>{});
      }
      return ret;
    }
    constexpr coord_component_type voxelSize(int i) const {
      // neglect shearing here
      coord_component_type sum = 0;
      for (int d = 0; d != dim; ++d) sum += zs::sqr(_transform(i, d));
      return zs::sqrt(sum, wrapv<space>{});
    }

    /// @brief linear index to coordinate
    template <int I = 0>
    constexpr integer_coord_type iCoord(size_type bno, integer_coord_component_type tileOffset,
                                        wrapv<I> = {}) const {
      return level(wrapv<I>{}).table._activeKeys[bno] + tile_offset_to_coord<I>(tileOffset);
    }
    template <int I = 0>
    constexpr integer_coord_type iCoord(size_type cellno, wrapv<I> = {}) const {
      return level(wrapv<I>{}).table._activeKeys[cellno / level_view_type<I>::block_size]
             + tile_offset_to_coord<I>(cellno & (level_view_type<I>::block_size - 1));
    }
    template <int I = 0> constexpr coord_type wCoord(size_type bno,
                                                     integer_coord_component_type tileOffset,
                                                     wrapv<I> = {}) const {
      return indexToWorld(iCoord(bno, tileOffset, wrapv<I>{}));
    }
    template <int I = 0> constexpr coord_type wCoord(size_type cellno, wrapv<I> = {}) const {
      return indexToWorld(iCoord(cellno, wrapv<I>{}));
    }

    template <int I = 0>
    constexpr coord_type iStaggeredCoord(size_type bno, integer_coord_component_type tileOffset,
                                         int f, wrapv<I> = {}) const {
      // f must be within [0, dim)
      return iCoord(bno, tileOffset, wrapv<I>{}) + coord_type::init([f](int d) {
               return d == f
                          ? (coord_component_type)-0.5
                                * (coord_component_type)((integer_coord_component_type)1
                                                         << container_type::
                                                                template get_cell_bit_offset<I>())
                          : (coord_component_type)0;
             });
    }
    template <int I = 0>
    constexpr coord_type iStaggeredCoord(size_type cellno, int f, wrapv<I> = {}) const {
      return iCoord(cellno, wrapv<I>{}) + coord_type::init([f](int d) {
               return d == f
                          ? (coord_component_type)-0.5
                                * (coord_component_type)((integer_coord_component_type)1
                                                         << container_type::
                                                                template get_cell_bit_offset<I>())
                          : (coord_component_type)0;
             });
    }
    template <int I = 0>
    constexpr coord_type wStaggeredCoord(size_type bno, integer_coord_component_type tileOffset,
                                         int f, wrapv<I> = {}) const {
      return indexToWorld(iStaggeredCoord(bno, tileOffset, f, wrapv<I>{}));
    }
    template <int I = 0>
    constexpr coord_type wStaggeredCoord(size_type cellno, int f, wrapv<I> = {}) const {
      return indexToWorld(iStaggeredCoord(cellno, f, wrapv<I>{}));
    }

    // insert/ query

    /// @brief sample
    /// @note collocated
    template <typename AccessorAgView, int AccessorDepths, kernel_e kt = kernel_e::linear,
              typename VecT = int,
              enable_if_all<is_same_v<AdaptiveGridUnnamedView, remove_const_t<AccessorAgView>>,
                            VecT::dim == 1, VecT::extent == dim>
              = 0>
    constexpr auto iSample(AdaptiveGridAccessor<AccessorAgView, AccessorDepths> &acc, size_type chn,
                           const VecInterface<VecT> &X, wrapv<kt> = {}) const {
      auto pad = GridArena<RM_CVREF_T(acc), kt, 0>(false_c, &acc, X);
      return pad.isample(chn, _background);
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int, bool UseAccessor = true,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iSample(size_type chn, const VecInterface<VecT> &X, wrapv<kt> = {},
                           wrapv<UseAccessor> = {}) const {
      if constexpr (UseAccessor) {
        auto acc = getAccessor();
        return iSample(acc, chn, X, wrapv<kt>{});
      } else {
        auto pad = iArena(X, wrapv<kt>{});
        return pad.isample(chn, _background);
      }
    }

    template <kernel_e kt = kernel_e::linear, typename VecT = int, bool UseAccessor = true,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wSample(size_type chn, const VecInterface<VecT> &x, wrapv<kt> = {},
                           wrapv<UseAccessor> = {}) const {
      return iSample(chn, worldToIndex(x), wrapv<kt>{}, wrapv<UseAccessor>{});
    }
    template <typename AccessorAgView, int AccessorDepths, kernel_e kt = kernel_e::linear,
              typename VecT = int,
              enable_if_all<is_same_v<AdaptiveGridUnnamedView, remove_const_t<AccessorAgView>>,
                            VecT::dim == 1, VecT::extent == dim>
              = 0>
    constexpr auto wSample(AdaptiveGridAccessor<AccessorAgView, AccessorDepths> &acc, size_type chn,
                           const VecInterface<VecT> &x, wrapv<kt> = {}) const {
      return iSample(acc, chn, worldToIndex(x), wrapv<kt>{});
    }

    /// @brief pack
    /// @note collocated
    template <typename AccessorAgView, int AccessorDepths, int N = dim,
              kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<is_same_v<AdaptiveGridUnnamedView, remove_const_t<AccessorAgView>>,
                            VecT::dim == 1, VecT::extent == dim, (N <= dim)>
              = 0>
    constexpr auto iPack(AdaptiveGridAccessor<AccessorAgView, AccessorDepths> &acc, size_type chn,
                         const VecInterface<VecT> &X, wrapv<N> = {}, wrapv<kt> = {}) const {
      zs::vec<value_type, N> ret{};
      auto pad = GridArena<RM_CVREF_T(acc), kt, 0>(false_c, &acc, X);
      for (int i = 0; i != N; ++i) ret.val(i) = pad.isample(chn + i, _background);
      return ret;
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto iPack(size_type chn, const VecInterface<VecT> &X, wrapv<N> = {},
                         wrapv<kt> = {}) const {
      auto acc = getAccessor();
      return iPack(acc, chn, X, wrapv<N>{}, wrapv<kt>{});
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT = int,
              bool UseAccessor = true,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto wPack(size_type chn, const VecInterface<VecT> &x, wrapv<N> = {},
                         wrapv<kt> = {}) const {
      auto acc = getAccessor();
      return iPack(acc, chn, worldToIndex(x), wrapv<N>{}, wrapv<kt>{});
    }
    template <typename AccessorAgView, int AccessorDepths, int N = dim,
              kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<is_same_v<AdaptiveGridUnnamedView, remove_const_t<AccessorAgView>>,
                            VecT::dim == 1, VecT::extent == dim, (N <= dim)>
              = 0>
    constexpr auto wPack(AdaptiveGridAccessor<AccessorAgView, AccessorDepths> &acc, size_type chn,
                         const VecInterface<VecT> &x, wrapv<N> = {}, wrapv<kt> = {}) const {
      return iPack(acc, chn, worldToIndex(x), wrapv<N>{}, wrapv<kt>{});
    }

    /// @brief levelset interface impl
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      return wSample(0, x);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getNormal(const VecInterface<VecT> &x) const noexcept {
      typename VecT::template variant_vec<value_type, typename VecT::extents> diff{}, v1{}, v2{};
      value_type eps = (value_type)(voxelSize(0) / 4);
      /// compute a local partial derivative
      for (int i = 0; i != dim; i++) {
        v1 = x;
        v2 = x;
        v1(i) = x(i) + eps;
        v2(i) = x(i) - eps;
        diff(i) = (getSignedDistance(v1) - getSignedDistance(v2)) / (eps + eps);
      }
      return diff.normalized();
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      return packed_value_type::constant(0);
    }

    zs::tuple<level_view_type<Is>...> _levels;
    transform_type _transform;
    value_type _background;
  };

  ///
  /// @brief view variants for AdaptiveUnnamedView
  ///
  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... TileBits,
            size_t... ScalingBits, size_t... Is, typename AllocatorT,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(
      const AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>,
                             index_sequence<ScalingBits...>, index_sequence<Is...>, AllocatorT> &ag,
      wrapv<Base> = {}) {
    return AdaptiveGridUnnamedView<
        ExecSpace,
        AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                         index_sequence<Is...>, AllocatorT>,
        /*IsConst*/ true, Base>{ag};
  }
  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... TileBits,
            size_t... ScalingBits, size_t... Is, typename AllocatorT,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(
      AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                       index_sequence<Is...>, AllocatorT> &ag,
      wrapv<Base> = {}) {
    return AdaptiveGridUnnamedView<
        ExecSpace,
        AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                         index_sequence<Is...>, AllocatorT>,
        /*IsConst*/ false, Base>{ag};
  }

  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... TileBits,
            size_t... ScalingBits, size_t... Is, typename AllocatorT,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(
      const AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>,
                             index_sequence<ScalingBits...>, index_sequence<Is...>, AllocatorT> &ag,
      wrapv<Base>, const SmallString &tagName) {
    auto ret = AdaptiveGridUnnamedView<
        ExecSpace,
        AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                         index_sequence<Is...>, AllocatorT>,
        /*IsConst*/ true, Base>{ag};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ((void)(ret.level(dim_c<Is>).grid._nameTag = tagName), ...);
#endif
    return ret;
  }
  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... TileBits,
            size_t... ScalingBits, size_t... Is, typename AllocatorT,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(
      AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                       index_sequence<Is...>, AllocatorT> &ag,
      wrapv<Base>, const SmallString &tagName) {
    auto ret = AdaptiveGridUnnamedView<
        ExecSpace,
        AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                         index_sequence<Is...>, AllocatorT>,
        /*IsConst*/ false, Base>{ag};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ((void)(ret.level(dim_c<Is>).grid._nameTag = tagName), ...);
#endif
    return ret;
  }

  template <execspace_e space, int dim, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, typename AllocatorT>
  decltype(auto) proxy(const AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>,
                                              index_sequence<ScalingBits...>, index_sequence<Is...>,
                                              AllocatorT> &ag) {
    return view<space>(ag, false_c);
  }
  template <execspace_e space, int dim, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, typename AllocatorT>
  decltype(auto) proxy(
      AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                       index_sequence<Is...>, AllocatorT> &ag) {
    return view<space>(ag, false_c);
  }

  template <execspace_e space, int dim, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, typename AllocatorT>
  decltype(auto) proxy(
      const AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>,
                             index_sequence<ScalingBits...>, index_sequence<Is...>, AllocatorT> &ag,
      const SmallString &tagName) {
    return view<space>(ag, false_c, tagName);
  }
  template <execspace_e space, int dim, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, typename AllocatorT>
  decltype(auto) proxy(
      AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
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
  template <execspace_e Space, int dim_, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, typename AllocatorT, bool IsConst, bool Base>
  struct AdaptiveGridView<
      Space,
      AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                       index_sequence<Is...>, AllocatorT>,
      IsConst, Base>
      : AdaptiveGridUnnamedView<
            Space,
            AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>,
                             index_sequence<ScalingBits...>, index_sequence<Is...>, AllocatorT>,
            IsConst, Base>,
        LevelSetInterface<AdaptiveGridView<
            Space,
            AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>,
                             index_sequence<ScalingBits...>, index_sequence<Is...>, AllocatorT>,
            IsConst, Base>> {
    /// @brief unnamed base view
    using base_t = AdaptiveGridUnnamedView<
        Space,
        AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                         index_sequence<Is...>, AllocatorT>,
        IsConst, Base>;
    static constexpr bool is_const_structure = IsConst;
    static constexpr auto space = Space;
    using container_type = typename base_t::container_type;
    static constexpr int dim = base_t::dim;
    static constexpr int num_levels = base_t::num_levels;
    using value_type = typename base_t::value_type;
    using size_type = typename base_t::size_type;
    using index_type = typename base_t::index_type;

    using integer_coord_component_type = typename base_t::integer_coord_component_type;
    using coord_mask_type = typename base_t::coord_mask_type;
    using integer_coord_type = typename base_t::integer_coord_type;
    using coord_component_type = typename base_t::coord_component_type;
    using coord_type = typename base_t::coord_type;
    using packed_value_type = typename base_t::packed_value_type;

    // property
    template <int LevelNo> using level_type = typename base_t::template level_type<LevelNo>;
    template <int LevelNo> using level_view_type =
        typename base_t::template level_view_type<LevelNo>;

    using child_offset_type = typename container_type::child_offset_type;
    using child_offset_view_type = typename container_type::child_offset_view_type;

    using table_type = typename base_t::table_type;
    using table_view_type = typename base_t::table_view_type;
    static constexpr index_type sentinel_v = base_t::sentinel_v;

    // grid
    template <int LevelNo> using grid_storage_type =
        typename base_t::template grid_storage_type<LevelNo>;
    template <int LevelNo> using unnamed_grid_view_type =
        typename base_t::template unnamed_grid_view_type<LevelNo>;
    template <int LevelNo> using grid_view_type = decltype(view<space>(
        declval<conditional_t<is_const_structure, const grid_storage_type<LevelNo> &,
                              grid_storage_type<LevelNo> &>>(),
        wrapv<Base>{}));

    // mask
    template <int LevelNo> using tile_mask_storage_type =
        typename base_t::template tile_mask_storage_type<LevelNo>;
    template <int LevelNo> using tile_mask_view_type =
        typename base_t::template tile_mask_view_type<LevelNo>;
    template <int LevelNo> using hierarchy_mask_storage_type =
        typename base_t::template hierarchy_mask_storage_type<LevelNo>;
    template <int LevelNo> using hierarchy_mask_view_type =
        typename base_t::template hierarchy_mask_view_type<LevelNo>;

    using channel_counter_type = typename grid_storage_type<0>::channel_counter_type;

    constexpr auto getPropertyNames() const noexcept { return _tagNames; }
    constexpr auto getPropertyOffsets() const noexcept { return _tagOffsets; }
    constexpr auto getPropertySizes() const noexcept { return _tagSizes; }
    constexpr auto numProperties() const noexcept { return _N; }
    constexpr auto propertyIndex(const SmallString &propName) const noexcept {
      channel_counter_type i = 0;
      for (; i != _N; ++i)
        if (_tagNames[i] == propName) break;
      return i;
    }
    constexpr auto propertySize(const SmallString &propName) const noexcept {
      return getPropertySizes()[propertyIndex(propName)];
    }
    constexpr auto propertyOffset(const SmallString &propName) const noexcept {
      return getPropertyOffsets()[propertyIndex(propName)];
    }
    constexpr bool hasProperty(const SmallString &propName) const noexcept {
      return propertyIndex(propName) != _N;
    }

    /// @note TileVectorView-alike
    const SmallString *_tagNames;
    const channel_counter_type *_tagOffsets;
    const channel_counter_type *_tagSizes;
    channel_counter_type _N;
  };

  ///
  /// @brief view variants for AdaptiveView
  ///
  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... TileBits,
            size_t... ScalingBits, size_t... Is, typename AllocatorT,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(
      const std::vector<SmallString> &tagNames,
      const AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>,
                             index_sequence<ScalingBits...>, index_sequence<Is...>, AllocatorT> &ag,
      wrapv<Base> = {}) {
    return AdaptiveGridView<
        ExecSpace,
        AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                         index_sequence<Is...>, AllocatorT>,
        /*IsConst*/ true, Base>{ag};
  }
  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... TileBits,
            size_t... ScalingBits, size_t... Is, typename AllocatorT,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(
      const std::vector<SmallString> &tagNames,
      AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                       index_sequence<Is...>, AllocatorT> &ag,
      wrapv<Base> = {}) {
    return AdaptiveGridView<
        ExecSpace,
        AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                         index_sequence<Is...>, AllocatorT>,
        /*IsConst*/ false, Base>{ag};
  }

  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... TileBits,
            size_t... ScalingBits, size_t... Is, typename AllocatorT,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(
      const std::vector<SmallString> &tagNames,
      const AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>,
                             index_sequence<ScalingBits...>, index_sequence<Is...>, AllocatorT> &ag,
      wrapv<Base>, const SmallString &tagName) {
    auto ret = AdaptiveGridView<
        ExecSpace,
        AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                         index_sequence<Is...>, AllocatorT>,
        /*IsConst*/ true, Base>{ag};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ((void)(ret.level(dim_c<Is>).grid._nameTag = tagName), ...);
#endif
    return ret;
  }
  template <execspace_e ExecSpace, int dim, typename ValueT, size_t... TileBits,
            size_t... ScalingBits, size_t... Is, typename AllocatorT,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(
      const std::vector<SmallString> &tagNames,
      AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                       index_sequence<Is...>, AllocatorT> &ag,
      wrapv<Base>, const SmallString &tagName) {
    auto ret = AdaptiveGridView<
        ExecSpace,
        AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                         index_sequence<Is...>, AllocatorT>,
        /*IsConst*/ false, Base>{ag};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ((void)(ret.level(dim_c<Is>).grid._nameTag = tagName), ...);
#endif
    return ret;
  }

  template <execspace_e space, int dim, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, typename AllocatorT>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       const AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>,
                                              index_sequence<ScalingBits...>, index_sequence<Is...>,
                                              AllocatorT> &ag) {
    for (auto &&tag : tagNames)
      if (!ag.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("adaptive grid property [\"{}\"] does not exist", (std::string)tag));
    return view<space>(tagNames, ag, false_c);
  }
  template <execspace_e space, int dim, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, typename AllocatorT>
  decltype(auto) proxy(
      const std::vector<SmallString> &tagNames,
      AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                       index_sequence<Is...>, AllocatorT> &ag) {
    for (auto &&tag : tagNames)
      if (!ag.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("adaptive grid property [\"{}\"] does not exist", (std::string)tag));
    return view<space>(tagNames, ag, false_c);
  }

  template <execspace_e space, int dim, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, typename AllocatorT>
  decltype(auto) proxy(
      const std::vector<SmallString> &tagNames,
      const AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>,
                             index_sequence<ScalingBits...>, index_sequence<Is...>, AllocatorT> &ag,
      const SmallString &tagName) {
    for (auto &&tag : tagNames)
      if (!ag.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("adaptive grid property [\"{}\"] does not exist", (std::string)tag));
    return view<space>(tagNames, ag, false_c, tagName);
  }
  template <execspace_e space, int dim, typename ValueT, size_t... TileBits, size_t... ScalingBits,
            size_t... Is, typename AllocatorT>
  decltype(auto) proxy(
      const std::vector<SmallString> &tagNames,
      AdaptiveGridImpl<dim, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                       index_sequence<Is...>, AllocatorT> &ag,
      const SmallString &tagName) {
    for (auto &&tag : tagNames)
      if (!ag.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("adaptive grid property [\"{}\"] does not exist", (std::string)tag));
    return view<space>(tagNames, ag, false_c, tagName);
  }

  ///
  ///
  ///
  /// @brief restructure
  ///
  ///
  ///
  template <int dim_, typename ValueT, size_t... TileBits, size_t... ScalingBits, size_t... Is>
  template <typename Policy, typename ValueDst, size_t... TileBitsDst, size_t... ScalingBitsDst,
            size_t... Js, execspace_e space>
  void AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                        index_sequence<Is...>, ZSPmrAllocator<>>::
      restructure(Policy &&pol, AdaptiveGridImpl<dim, ValueDst, index_sequence<TileBitsDst...>,
                                                 index_sequence<ScalingBitsDst...>,
                                                 index_sequence<Js...>, ZSPmrAllocator<>> &agDst) {
    if (!valid_memspace_for_execution(pol, get_allocator()))
      throw std::runtime_error(
          "[AdaptiveGrid::reorder] current memory location not compatible with the execution "
          "policy");

    using AgDstT = RM_CVREF_T(agDst);
    /// @brief prepare allocation
    const auto &tags = getPropertyTags();
    // if (agDst.memoryLocation() != memoryLocation())
    {
      auto allocateTargetLevel = [&tags, &agDst, this](auto lNo) {
        auto &lDst = agDst.level(lNo);
        lDst = RM_CVREF_T(lDst)(get_allocator(), tags, 0);
      };
      ((void)allocateTargetLevel(wrapv<Js>{}), ...);
    }

    bool shouldSync = pol.shouldSync();
    pol.sync(true);

    /// @brief prepare partition
    auto hashTargetLevel = [&pol, &agDst, this](auto lNo) {
      constexpr int level_no = RM_CVREF_T(lNo)::value;
      constexpr integer_coord_component_type cell_bits = Level<level_no>::sbit;
      constexpr int dst_level_no = AgDstT::closest_child_level(wrapv<cell_bits>{});
      static_assert(cell_bits >= AgDstT::template Level<dst_level_no>::sbit, "???");

#if 0
      fmt::print(fg(fmt::color::light_blue), "hashing level-{} [{}, {}) to dst-level-{} [{}, {})\n",
                 level_no, cell_bits, Level<level_no>::ebit, dst_level_no,
                 AgDstT::template Level<dst_level_no>::sbit,
                 AgDstT::template Level<dst_level_no>::ebit);
#endif

      auto &l = level(lNo);
      auto &lDst = agDst.level(wrapv<dst_level_no>{});

      /// @note reserve enough space for the assignment
      constexpr integer_coord_component_type num_child_bits
          = Level<level_no>::ebit - AgDstT::template Level<dst_level_no>::ebit;
      constexpr size_type side_scale
          = num_child_bits < 0 ? (size_type)1 : ((size_type)1 << num_child_bits);
      auto nbs = l.numBlocks();
      auto prevDstNbs = lDst.numBlocks();

      auto estDstNbs = prevDstNbs + nbs * math::pow_integral(side_scale, dim);
      lDst.resizePartition(pol, estDstNbs);

      {
        /// @note register active cells to target level
        auto params = zs::make_tuple(view<space>(l.valueMask), view<space>(l.table._activeKeys),
                                     view<space>(lDst.table), wrapt<AdaptiveGridImpl>{},
                                     wrapv<level_no>{}, wrapt<AgDstT>{}, wrapv<dst_level_no>{});
        pol(range(nbs * Level<level_no>::block_size), params, _restructure_hash_active_cell{});
      }
    };
    ((void)hashTargetLevel(wrapv<Is>{}), ...);

    /// @brief default initialize target grid
    ((void)agDst.level(wrapv<Js>{}).refitToPartition(), ...);
    ((void)agDst.level(wrapv<Js>{}).defaultInitialize(), ...);

    /// @brief assign values
    auto agv = view<space>(*this);
    auto agvDst = view<space>(agDst);
    /// @note assume every value-active cell in the target adaptive grid is exactly covered by only
    /// one active cell in the source adaptive grid
    auto buildTargetLevel = [&pol, &agv, &agDst, &agvDst](auto lNo) {
      auto &lDst = agDst.level(lNo);
      auto lvDst = agvDst.level(lNo);
      auto nbsDst = lDst.numBlocks();
      {
        auto params = zs::make_tuple(agv, lvDst);
        pol(range(nbsDst * AgDstT::template Level<RM_REF_T(lNo)::value>::block_size), params,
            _restructure_assign_values{});
      }
    };
    ((void)buildTargetLevel(wrapv<Js>{}), ...);

    /// DEBUG
#if 0
    auto agv = view<space>(*this);
    auto agvDst = view<space>(agDst);
    Vector<integer_coord_type> coords{allocator, 100000};
    pol(enumerate(coords), zs::tuple<>{}, _restructure_0{});
    {
      auto tup = zs::make_tuple(agv, agvDst);
      pol(enumerate(coords), tup, _restructure_1{});
    }
    fmt::print(fg(fmt::color::red), "done dst ag value check\n");
#endif
    /// DEBUG

    /// @brief complete topo build
    agDst.complementTopo(FWD(pol), true_c);  // this includes [reorder] at the end
    /// @brief remaining info
    agDst._transform = _transform;
    agDst._background = _background;

    pol.sync(shouldSync);
  }

  ///
  ///
  ///
  /// @brief complement topology
  ///
  ///
  ///
  template <int dim_, typename ValueT, size_t... TileBits, size_t... ScalingBits, size_t... Is>
  template <typename Policy, bool SortGridData, execspace_e space> void
  AdaptiveGridImpl<dim_, ValueT, index_sequence<TileBits...>, index_sequence<ScalingBits...>,
                   index_sequence<Is...>, ZSPmrAllocator<>>::complementTopo(Policy &&pol,
                                                                            wrapv<SortGridData>) {
    if (!valid_memspace_for_execution(pol, get_allocator()))
      throw std::runtime_error(
          "[AdaptiveGrid::updateTopo] current memory location not compatible with the "
          "execution policy");
    // ...
    bool shouldSync = pol.shouldSync();
    pol.sync(true);
    /// DEBUG
    /// DEBUG
    auto complementParentTopo = [&pol, this](auto lNoc) {
      constexpr int level_no = RM_CVREF_T(lNoc)::value;
      if constexpr (level_no != num_levels - 1) {
        auto &lc = level(wrapv<level_no>{});
        auto nbs = lc.numBlocks();
        auto &lp = level(wrapv<level_no + 1>{});
        using ParentLevelT = RM_CVREF_T(lp);
        size_t prevParNbs = lp.numBlocks();
        /// @note worst case scenario, assume every child block generates a new parent block
        lp.resizePartition(pol, prevParNbs + nbs);

        /// DEBUG
        if constexpr (false) {
          auto allocator = get_temporary_memory_source(pol);
          Vector<size_type> numActiveChildren{allocator, prevParNbs};
          pol(zip(numActiveChildren, lp.childMask), zs::make_tuple(wrapv<space>{}),
              _reorder_count_on{});
          Vector<size_type> numOn{allocator, 1};
          reduce(pol, numActiveChildren.begin(), numActiveChildren.end(), numOn.begin());
          fmt::print("upon topo complementation level [{}]: initial [{}] num active children\n",
                     level_no + 1, numOn.getVal());
        }
        /// DEBUG

        {
          auto params = zs::make_tuple(view<space>(lc.table._activeKeys), view<space>(lp.table),
                                       wrapt<AdaptiveGridImpl>{}, wrapv<level_no + 1>{});
          pol(range(nbs), params, _build_level_entry{});
        }
        size_t parNbs = lp.numBlocks();
        // fmt::print("complementing level [{}] topo: {} -> {}\n", level_no + 1, prevParNbs,
        // parNbs);
        {
          /// @note init (child/value) mask for newly spawned blocks. childOffset uninitialized
          lp.resizeTopo(parNbs);
          // value
          Resource::memset(
              MemoryEntity{this->memoryLocation(), (void *)(lp.valueMask.data() + prevParNbs)}, 0,
              /*num bytes*/ (parNbs - prevParNbs)
                  * sizeof(typename ParentLevelT::tile_mask_type::value_type));
          // child
          Resource::memset(
              MemoryEntity{this->memoryLocation(), (void *)(lp.childMask.data() + prevParNbs)}, 0,
              /*num bytes*/ (parNbs - prevParNbs)
                  * sizeof(typename ParentLevelT::hierarchy_mask_type::value_type));

          /// @note init grid data
          lp.resizeGrid(parNbs);
          Resource::memset(MemoryEntity{this->memoryLocation(), lp.grid.tileOffset(prevParNbs)}, 0,
                           /*num bytes*/ (parNbs - prevParNbs) * lp.grid.tileBytes());
        }
        // fmt::print("complementing level [{}]: done default init\n", level_no + 1);
        /// DEBUG
        if constexpr (false) {
          auto allocator = get_temporary_memory_source(pol);
          Vector<size_type> numActiveChildren{allocator, parNbs};
          pol(zip(numActiveChildren, lp.childMask), zs::make_tuple(wrapv<space>{}),
              _reorder_count_on{});
          Vector<size_type> numOn{allocator, 1};
          reduce(pol, numActiveChildren.begin(), numActiveChildren.end(), numOn.begin());
          fmt::print("upon topo complementation level [{}]: after init, [{}] num active children\n",
                     level_no + 1, numOn.getVal());
        }
        /// DEBUG
        {
          /// @note maintain child mask
          using hierarchy_bitmask_type =
              typename Level<level_no + 1>::hierarchy_mask_type::value_type;

          auto params = zs::make_tuple(view<space>(lc.table), view<space>(lp.table._activeKeys),
                                       view<space>(lp.childMask), wrapv<level_no + 1>{});
          /// @note parallel in mask element (word)
          pol(range(parNbs * hierarchy_bitmask_type::word_count), params,
              _update_topo_update_childmask{});
        }
        // fmt::print("complementing level [{}]: done update child mask\n", level_no + 1);

        /// DEBUG
        if constexpr (false) {
          auto allocator = get_temporary_memory_source(pol);
          Vector<size_type> numActiveChildren{allocator, parNbs};
          pol(zip(numActiveChildren, lp.childMask), zs::make_tuple(wrapv<space>{}),
              _reorder_count_on{});
          Vector<size_type> numOn{allocator, 1};
          reduce(pol, numActiveChildren.begin(), numActiveChildren.end(), numOn.begin());
          fmt::print(
              "upon topo complementation level [{}]: in the end [{}] num active children, in "
              "reality [{}] "
              "children\n",
              level_no + 1, numOn.getVal(), nbs);
        }
        /// DEBUG
      }
    };
    ((void)complementParentTopo(wrapv<Is>{}), ...);  // since c++17, sequenced execution
    pol.sync(shouldSync);
    /// @note maintain childMask info
    reorder(FWD(pol), wrapv<SortGridData>{});
  }

#if ZS_ENABLE_SERIALIZATION
  template <typename S, int dim, typename T, typename TileBits, typename ScalingBits,
            typename Indices>
  void serialize(S &s,
                 AdaptiveGridImpl<dim, T, TileBits, ScalingBits, Indices, ZSPmrAllocator<>> &ag) {
    if (!ag.memoryLocation().onHost()) {
      ag = ag.clone({memsrc_e::host, -1});
    }

    serialize(s, ag._levels);
    serialize(s, ag._transform);
    s.template value<sizeof(ag._background)>(ag._background);
  }
#endif

}  // namespace zs