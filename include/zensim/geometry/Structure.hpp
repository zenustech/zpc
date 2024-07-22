#pragma once
#include <map>
#include <stdexcept>

#include "zensim/TypeAlias.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Polymorphism.h"
#include "zensim/zpc_tpls/magic_enum/magic_enum.hpp"

namespace zs {

#if 0
  template <typename V = f32, typename I = i32, int d = 3> struct MeshObject {
    using ElemT = I[d + 1];
    using TV = V[d];
    using TM = V[d][d];
    using TMAffine = V[d + 1][d + 1];
    static constexpr int dim = d;
    Vector<V> M;
    Vector<TV> X;
    Vector<ElemT> Elems;
    Vector<TM> F;
  };

  using GeneralMesh
      = variant<MeshObject<f32, i32, 2>, MeshObject<f32, i64, 2>, MeshObject<f32, i32, 3>,
                MeshObject<f32, i64, 3>, MeshObject<f64, i32, 2>, MeshObject<f64, i64, 2>,
                MeshObject<f64, i32, 3>, MeshObject<f64, i64, 3>>;
#endif

  template <typename ValueT, int d_, auto SideLength, grid_e category, typename allocator>
  struct Grid;
  template <typename ValueT, int d_, auto SideLength, typename allocator> struct Grids;

  template <typename ValueT = f32, int d_ = 3, auto SideLength = 4,
            grid_e category_ = grid_e::collocated, typename AllocatorT = ZSPmrAllocator<>>
  struct Grid {
    static_assert(d_ > 0, "dimension must be positive!");
    using value_type = ValueT;
    using allocator_type = AllocatorT;
    using coord_index_type = conditional_t<(sizeof(value_type) <= 4), i32, i64>;
    // using cell_index_type = zs::make_unsigned_t<decltype(SideLength)>;
    using cell_index_type = coord_index_type;
    static constexpr auto category = category_;
    static constexpr int dim = d_;
    static constexpr cell_index_type side_length = SideLength;
    static constexpr auto block_size = math::pow_integral(side_length, (int)dim);
    using grids_t = Grids<value_type, dim, side_length, allocator_type>;
    static constexpr cell_index_type block_space() noexcept {
      auto ret = side_length;
      for (int d = 1; d != dim; ++d) ret *= side_length;
      return ret;
    }
    /// ninja optimization
    static constexpr bool is_power_of_two
        = side_length > 0 && ((side_length & (side_length - 1)) == 0);
    static constexpr auto num_cell_bits = bit_count(side_length);

    using grid_storage_t = TileVector<value_type, (size_t)block_size, allocator_type>;
    using size_type = typename grid_storage_t::size_type;
    using channel_counter_type = typename grid_storage_t::channel_counter_type;

    using CellIV = vec<cell_index_type, dim>;
    using IV = vec<coord_index_type, dim>;
    using TV = vec<value_type, dim>;

    constexpr MemoryLocation memoryLocation() const noexcept { return blocks.memoryLocation(); }
    constexpr ProcID devid() const noexcept { return blocks.devid(); }
    constexpr memsrc_e memspace() const noexcept { return blocks.memspace(); }
    constexpr auto size() const noexcept { return blocks.size(); }
    decltype(auto) get_allocator() const noexcept { return blocks.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (size_t)1 << (size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }
    constexpr decltype(auto) numBlocks() const noexcept { return blocks.numTiles(); }
    constexpr decltype(auto) numReservedBlocks() const noexcept {
      return blocks.numReservedTiles();
    }
    constexpr channel_counter_type numChannels() const noexcept { return blocks.numChannels(); }

    Grid(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
         value_type dx, size_type count = 0)
        : blocks{allocator, channelTags, count * (size_t)block_size}, dx{dx} {}
    Grid(const std::vector<PropertyTag> &channelTags, value_type dx, size_type count,
         memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Grid{get_default_allocator(mre, devid), channelTags, dx, count} {}
    Grid(channel_counter_type numChns, value_type dx, size_type count,
         memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Grid{get_default_allocator(mre, devid), {{"unnamed", numChns}}, dx, count} {}
    Grid(value_type dx = 1.f, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Grid{get_default_allocator(mre, devid), {{"m", 1}, {"v", dim}}, dx, 0} {}

    Grid clone(const allocator_type &allocator) const {
      Grid ret{};
      ret.blocks = blocks.clone(allocator);
      ret.dx = dx;
      return ret;
    }
    Grid clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    void resize(size_type numBlocks) { blocks.resize(numBlocks * (size_type)block_size); }
    template <typename Policy>
    void append_channels(Policy &&policy, const std::vector<PropertyTag> &tags,
                         const source_location &loc = source_location::current()) {
      blocks.append_channels(FWD(policy), tags, loc);
    }
    template <typename Policy> void reset(Policy &&policy, value_type val) {
      blocks.reset(FWD(policy), val);
    }

    bool hasProperty(const SmallString &str) const noexcept { return blocks.hasProperty(str); }
    constexpr channel_counter_type getPropertySize(const SmallString &str) const {
      return blocks.getPropertySize(str);
    }
    constexpr channel_counter_type getPropertyOffset(const SmallString &str) const {
      return blocks.getPropertyOffset(str);
    }
    constexpr PropertyTag getPropertyTag(size_t i = 0) const { return blocks.getPropertyTag(i); }
    constexpr const auto &getPropertyTags() const { return blocks.getPropertyTags(); }

    grid_storage_t blocks;
    value_type dx;
  };
  template <typename ValueT = f32, int d_ = 3, auto SideLength = 4,
            typename AllocatorT = ZSPmrAllocator<>>
  struct Grids {
    template <grid_e category = grid_e::collocated> using grid_t
        = Grid<ValueT, d_, SideLength, category, AllocatorT>;
    using collocated_grid_t = grid_t<grid_e::collocated>;
    using value_type = typename collocated_grid_t::value_type;
    using allocator_type = typename collocated_grid_t::allocator_type;
    using cell_index_type = typename collocated_grid_t::cell_index_type;
    using coord_index_type = typename collocated_grid_t::coord_index_type;
    static constexpr int dim = d_;
    static constexpr cell_index_type side_length = collocated_grid_t::side_length;
    static constexpr cell_index_type block_space() noexcept {
      return collocated_grid_t::block_size;
    }
    /// ninja optimization
    static constexpr bool is_power_of_two = collocated_grid_t::is_power_of_two;
    static constexpr auto num_cell_bits = collocated_grid_t::num_cell_bits;

    using grid_storage_t = typename collocated_grid_t::grid_storage_t;
    using size_type = typename collocated_grid_t::size_type;
    using channel_counter_type = typename collocated_grid_t::channel_counter_type;

    using CellIV = typename collocated_grid_t::CellIV;
    using IV = typename collocated_grid_t::IV;
    using TV = typename collocated_grid_t::TV;

    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (size_t)1 << (size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

    template <typename F> constexpr decltype(auto) gridApply(grid_e category, F &&f) {
      if (category == grid_e::collocated)
        return std::invoke(f, grid(collocated_c));
      else if (category == grid_e::cellcentered)
        return std::invoke(f, grid(cellcentered_c));
      else
        return std::invoke(f, grid(staggered_c));
    }
    template <typename F> constexpr decltype(auto) gridApply(grid_e category, F &&f) const {
      if (category == grid_e::collocated)
        return std::invoke(f, grid(collocated_c));
      else if (category == grid_e::cellcentered)
        return std::invoke(f, grid(cellcentered_c));
      else
        return std::invoke(f, grid(staggered_c));
    }
    constexpr MemoryLocation memoryLocation() const noexcept {
      return gridApply(_primaryGrid,
                       [](auto &&grid) -> MemoryLocation { return grid.memoryLocation(); });
    }
    constexpr memsrc_e space() const noexcept {
      return gridApply(_primaryGrid, [](auto &&grid) -> memsrc_e { return grid.memspace(); });
    }
    constexpr ProcID devid() const noexcept {
      return gridApply(_primaryGrid, [](auto &&grid) -> ProcID { return grid.devid(); });
    }
    constexpr size_type size() const noexcept {
      return gridApply(_primaryGrid, [](auto &&grid) -> size_type { return grid.size(); });
    }
    constexpr const allocator_type &get_allocator() const {
      return gridApply(_primaryGrid, [](auto &&grid) -> decltype(grid.get_allocator()) {
        return grid.get_allocator();
      });
      throw std::runtime_error(
          fmt::format("primary grid \"{}\" not known", magic_enum::enum_name(_primaryGrid)));
    }
    constexpr size_type numBlocks() const noexcept {
      return gridApply(_primaryGrid, [](auto &&grid) -> size_type { return grid.numTiles(); });
    }

    Grids(const allocator_type &allocator,
          const std::vector<PropertyTag> &channelTags = {{"m", 1}, {"v", dim}}, value_type dx = 1.f,
          size_type numBlocks = 0, grid_e ge = grid_e::collocated)
        : _collocatedGrid{allocator, channelTags, dx},
          _cellcenteredGrid{allocator, channelTags, dx},
          _staggeredGrid{allocator, channelTags, dx},
          _dx{dx},
          _primaryGrid{ge} {
      if (ge == grid_e::collocated)
        _collocatedGrid.resize(numBlocks);
      else if (ge == grid_e::cellcentered)
        _cellcenteredGrid.resize(numBlocks);
      else if (ge == grid_e::staggered)
        _staggeredGrid.resize(numBlocks);
    }
    Grids(const std::vector<PropertyTag> &channelTags = {{"m", 1}, {"v", dim}}, value_type dx = 1.f,
          size_type numBlocks = 0, memsrc_e mre = memsrc_e::host, ProcID devid = -1,
          grid_e ge = grid_e::collocated)
        : Grids{get_default_allocator(mre, devid), channelTags, dx, numBlocks, ge} {}

    void align(grid_e targetGrid) {
      if (targetGrid == _primaryGrid) return;
      const auto nblocks = numBlocks();
      gridApply(targetGrid, [nblocks](auto &&grid) { return grid.resize(nblocks); });
    }

    template <grid_e category = grid_e::collocated>
    constexpr auto &grid(wrapv<category> = {}) noexcept {
      if constexpr (category == grid_e::collocated)
        return _collocatedGrid;
      else if constexpr (category == grid_e::cellcentered)
        return _cellcenteredGrid;
      else if constexpr (category == grid_e::staggered)
        return _staggeredGrid;
    }
    template <grid_e category = grid_e::collocated>
    constexpr const auto &grid(wrapv<category> = {}) const noexcept {
      if constexpr (category == grid_e::collocated)
        return _collocatedGrid;
      else if constexpr (category == grid_e::cellcentered)
        return _cellcenteredGrid;
      else if constexpr (category == grid_e::staggered)
        return _staggeredGrid;
    }
    template <grid_e category = grid_e::collocated>
    constexpr auto numCells(wrapv<category> = {}) const noexcept {
      if constexpr (category == grid_e::collocated)
        return _collocatedGrid.size() * _collocatedGrid.numChannels();
      else if constexpr (category == grid_e::cellcentered)
        return _cellcenteredGrid.size() * _cellcenteredGrid.numChannels();
      else if constexpr (category == grid_e::staggered)
        return _staggeredGrid.size() * _staggeredGrid.numChannels();
    }

    grid_t<grid_e::collocated> _collocatedGrid{};
    grid_t<grid_e::cellcentered> _cellcenteredGrid{};
    grid_t<grid_e::staggered> _staggeredGrid{};
    value_type _dx{};
    grid_e _primaryGrid{};
  };

  using GeneralGrids
      = variant<Grids<f32, 3, 4>, Grids<f32, 3, 8>, Grids<f32, 2, 4>, Grids<f32, 2, 8>>;

  /// GridT can be const decorated
  template <typename GridT, typename = void> struct grid_traits {
    static constexpr bool is_const_structure = is_const_v<GridT>;
    using grid_t = remove_const_t<GridT>;
    using value_type = typename grid_t::value_type;
    using size_type = typename grid_t::size_type;  // basically size_t
    using channel_counter_type = typename grid_t::channel_counter_type;
    using coord_index_type = typename grid_t::coord_index_type;
    using grid_storage_t
        = remove_cvref_t<typename grid_t::grid_storage_t>;  // should be a tilevector

    static_assert(is_signed_v<coord_index_type>,
                  "coordinate index type should be a signed integer!");

    static constexpr grid_e category = grid_t::category;
    static constexpr int dim = grid_t::dim;
    static constexpr auto side_length = grid_t::side_length;

    /// deduced
    // https://listengine.tuxfamily.org/lists.tuxfamily.org/eigen/2017/01/msg00126.html
    using cell_index_type = zs::make_signed_t<RM_CVREF_T(
        side_length)>;  // this should be signed integer for indexing convenience
    static constexpr auto block_size = math::pow_integral(side_length, (int)dim);
    static constexpr bool is_power_of_two
        = side_length > 0 && ((side_length & (side_length - 1)) == 0);
    static constexpr auto num_cell_bits = bit_count(side_length);
    static constexpr auto num_block_bits = bit_count(block_size);

    template <execspace_e space, bool with_name = true> using grid_view_t = conditional_t<
        with_name,
        decltype(proxy<space>({}, declval<conditional_t<is_const_structure, const grid_storage_t &,
                                                        grid_storage_t &>>())),
        decltype(proxy<space>(declval<conditional_t<is_const_structure, const grid_storage_t &,
                                                    grid_storage_t &>>()))>;
    template <execspace_e space, bool with_name = true> using grid_block_view_t =
        typename grid_view_t<space, with_name>::tile_view_type;  // tilevector view property

    using coord_t = vec<coord_index_type, dim>;  // this is the default storage choice
    using pos_t = vec<value_type, dim>;

    static constexpr coord_t cellid_to_coord(cell_index_type cellid) noexcept {
      coord_t ret{};
      if constexpr (is_power_of_two)
        for (auto d = dim - 1; d >= 0; --d, cellid >>= num_cell_bits)
          ret[d] = (coord_index_type)(cellid & (side_length - 1));
      else
        for (auto d = dim - 1; d >= 0; --d, cellid /= side_length)
          ret[d] = (coord_index_type)(cellid % side_length);
      return ret;
    }
    template <typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    static constexpr auto coord_to_cellid(const VecInterface<VecT> &coord) noexcept {
      using Ti = math::op_result_t<cell_index_type, typename VecT::index_type>;
      Ti ret{0};
      if constexpr (is_power_of_two)
        for (int d = 0; d != dim; ++d) ret = (ret << (Ti)num_cell_bits) | (Ti)coord[d];
      else
        for (int d = 0; d != dim; ++d) ret = (ret * (Ti)side_length) + (Ti)coord[d];
      return ret;
    }
    template <typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    static constexpr auto global_coord_to_cellid(const VecInterface<VecT> &coord) noexcept {
      using Ti = math::op_result_t<cell_index_type, typename VecT::index_type>;
      Ti ret{0};
      if constexpr (is_power_of_two)
        for (int d = 0; d != dim; ++d)
          ret = (ret << (Ti)num_cell_bits) | ((Ti)coord[d] & (Ti)(side_length - 1));
      else
        for (int d = 0; d != dim; ++d)
          ret = (ret * (Ti)side_length) + ((Ti)coord[d] % (Ti)side_length);
      return ret;
    }
  };

  template <execspace_e space, typename GridT, bool with_name = true, bool block_scope = false,
            typename = void>
  struct GridView : grid_traits<GridT> {
    using traits = grid_traits<GridT>;
    using traits::category;
    using traits::cellid_to_coord;
    using traits::coord_to_cellid;
    using traits::global_coord_to_cellid;

    using traits::block_size;
    using traits::is_power_of_two;
    using traits::num_block_bits;
    using traits::num_cell_bits;
    using traits::side_length;

    using value_type = typename traits::value_type;
    using size_type = typename traits::size_type;
    using channel_counter_type = typename traits::channel_counter_type;
    using coord_index_type = typename traits::coord_index_type;
    using cell_index_type = typename traits::cell_index_type;
    using view_t
        = conditional_t<block_scope, typename traits::template grid_block_view_t<space, with_name>,
                        typename traits::template grid_view_t<space, with_name>>;
    static constexpr auto is_const_structure = traits::is_const_structure;
    static constexpr auto dim = traits::dim;

    static constexpr auto block_space() noexcept { return traits::block_size; }
    GridView() noexcept = default;
    constexpr GridView(view_t g, value_type dx) noexcept : grid{g}, dx{dx} {}

    template <auto V = with_name>
    constexpr enable_if_type<V, const SmallString *> getPropertyNames() const noexcept {
      return grid.getPropertyNames();
    }
    template <auto V = with_name>
    constexpr enable_if_type<V, const channel_counter_type *> getPropertyOffsets() const noexcept {
      return grid.getPropertyOffsets();
    }
    template <auto V = with_name>
    constexpr enable_if_type<V, const channel_counter_type *> getPropertySizes() const noexcept {
      return grid.getPropertySizes();
    }
    template <auto V = with_name>
    constexpr enable_if_type<V, channel_counter_type> numProperties() const noexcept {
      return grid.numProperties();
    }
    template <auto V = with_name> constexpr enable_if_type<V, channel_counter_type> propertyIndex(
        const SmallString &propName) const noexcept {
      return grid.propertyIndex(propName);
    }
    template <auto V = with_name> constexpr enable_if_type<V, channel_counter_type> propertySize(
        const SmallString &propName) const noexcept {
      return grid.propertySize(propName);
    }
    template <auto V = with_name> constexpr enable_if_type<V, channel_counter_type> propertyOffset(
        const SmallString &propName) const noexcept {
      return grid.propertyOffset(propName);
    }
    template <auto V = with_name>
    constexpr enable_if_type<V, bool> hasProperty(const SmallString &propName) const noexcept {
      return grid.hasProperty(propName);
    }

    /// grid -> block
    template <auto in_block = block_scope, enable_if_t<!in_block> = 0>
    constexpr auto block(const size_type i) noexcept {
      return GridView<space, GridT, with_name, true>{grid.tile(i), dx};
    }
    template <auto in_block = block_scope, enable_if_t<!in_block> = 0>
    constexpr auto block(const size_type i) const noexcept {
      return GridView<space, std::add_const_t<GridT>, with_name, true>{grid.tile(i), dx};
    }
    template <auto in_block = block_scope, enable_if_t<!in_block> = 0>
    constexpr auto operator[](const size_type i) noexcept {
      return block(i);
    }
    template <auto in_block = block_scope, enable_if_t<!in_block> = 0>
    constexpr auto operator[](const size_type i) const noexcept {
      return block(i);
    }

    ///
    /// voxel ([block, cell] pair)
    /// cell can be index or coord
    ///
    /// unnamed access
    // block
    template <typename VecT, bool is_const = is_const_structure, auto in_block = block_scope,
              enable_if_all<!is_const, in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto &operator()(channel_counter_type c, const VecInterface<VecT> &loc) noexcept {
      return grid(c, coord_to_cellid(loc));
    }
    template <typename VecT, auto in_block = block_scope,
              enable_if_all<in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto operator()(channel_counter_type c,
                              const VecInterface<VecT> &loc) const noexcept {
      return grid(c, coord_to_cellid(loc));
    }
    template <bool is_const = is_const_structure, auto in_block = block_scope,
              enable_if_all<!is_const, in_block> = 0>
    constexpr auto &operator()(channel_counter_type chn, cell_index_type cellid) noexcept {
      return grid(chn, cellid);
    }
    template <auto in_block = block_scope, enable_if_all<in_block> = 0>
    constexpr auto operator()(channel_counter_type chn, cell_index_type cellid) const noexcept {
      return grid(chn, cellid);
    }
    // grid
    template <typename VecT, bool is_const = is_const_structure, auto in_block = block_scope,
              enable_if_all<!is_const, !in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto &operator()(channel_counter_type c, const size_type blockno,
                               const VecInterface<VecT> &loc) noexcept {
      if constexpr (is_power_of_two)
        return grid(c, (blockno << (size_type)num_block_bits) | (size_type)coord_to_cellid(loc));
      else
        return grid(c, blockno * (size_type)block_size + (size_type)coord_to_cellid(loc));
    }
    template <typename VecT, auto in_block = block_scope,
              enable_if_all<!in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto operator()(channel_counter_type c, const size_type blockno,
                              const VecInterface<VecT> &loc) const noexcept {
      if constexpr (is_power_of_two)
        return grid(c, (blockno << (size_type)num_block_bits) | (size_type)coord_to_cellid(loc));
      else
        return grid(c, blockno * (size_type)block_size + (size_type)coord_to_cellid(loc));
    }
    template <bool is_const = is_const_structure, auto in_block = block_scope,
              enable_if_all<!is_const, !in_block> = 0>
    constexpr auto &operator()(channel_counter_type c, const size_type blockno,
                               cell_index_type cellno) noexcept {
      if constexpr (is_power_of_two)
        return grid(c, (blockno << (size_type)num_block_bits) | (size_type)cellno);
      else
        return grid(c, blockno * (size_type)block_size + (size_type)cellno);
    }
    template <auto in_block = block_scope, enable_if_all<!in_block> = 0>
    constexpr auto operator()(channel_counter_type c, const size_type blockno,
                              cell_index_type cellno) const noexcept {
      if constexpr (is_power_of_two)
        return grid(c, (blockno << (size_type)num_block_bits) | (size_type)cellno);
      else
        return grid(c, blockno * (size_type)block_size + (size_type)cellno);
    }
    template <bool is_const = is_const_structure, auto in_block = block_scope,
              enable_if_all<!is_const, !in_block> = 0>
    constexpr auto &operator()(channel_counter_type chn, size_type id) noexcept {
      return grid(chn, id);
    }
    template <auto in_block = block_scope, enable_if_all<!in_block> = 0>
    constexpr auto operator()(channel_counter_type chn, size_type id) const noexcept {
      return grid(chn, id);
    }

    // block
    template <typename VecT, bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope,
              enable_if_all<!is_const, has_name, in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto &operator()(const SmallString &propName,
                               const VecInterface<VecT> &loc) noexcept {
      return grid(propName, coord_to_cellid(loc));
    }
    template <typename VecT, bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto operator()(const SmallString &propName,
                              const VecInterface<VecT> &loc) const noexcept {
      return grid(propName, coord_to_cellid(loc));
    }
    template <bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope, enable_if_all<!is_const, has_name, in_block> = 0>
    constexpr auto &operator()(const SmallString &propName, cell_index_type cellid) noexcept {
      return grid(propName, cellid);
    }
    template <bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, in_block> = 0>
    constexpr auto operator()(const SmallString &propName, cell_index_type cellid) const noexcept {
      return grid(propName, cellid);
    }
    template <typename VecT, bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope,
              enable_if_all<!is_const, has_name, in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto &operator()(const SmallString &propName, channel_counter_type chn,
                               const VecInterface<VecT> &loc) noexcept {
      return grid(propName, chn, coord_to_cellid(loc));
    }
    template <typename VecT, bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto operator()(const SmallString &propName, channel_counter_type chn,
                              const VecInterface<VecT> &loc) const noexcept {
      return grid(propName, chn, coord_to_cellid(loc));
    }
    template <bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope, enable_if_all<!is_const, has_name, in_block> = 0>
    constexpr auto &operator()(const SmallString &propName, channel_counter_type chn,
                               cell_index_type cellid) noexcept {
      return grid(propName, chn, cellid);
    }
    template <bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, in_block> = 0>
    constexpr auto operator()(const SmallString &propName, channel_counter_type chn,
                              cell_index_type cellid) const noexcept {
      return grid(propName, chn, cellid);
    }
    // grid
    // name, bno, ccoord
    template <typename VecT, bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope,
              enable_if_all<!is_const, has_name, !in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto &operator()(const SmallString &propName, const size_type blockno,
                               const VecInterface<VecT> &loc) noexcept {
      if constexpr (is_power_of_two)
        return grid(propName,
                    (blockno << (size_type)num_block_bits) | (size_type)coord_to_cellid(loc));
      else
        return grid(propName, blockno * (size_type)block_size + (size_type)coord_to_cellid(loc));
    }
    template <typename VecT, bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, !in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto operator()(const SmallString &propName, const size_type blockno,
                              const VecInterface<VecT> &loc) const noexcept {
      if constexpr (is_power_of_two)
        return grid(propName,
                    (blockno << (size_type)num_block_bits) | (size_type)coord_to_cellid(loc));
      else
        return grid(propName, blockno * (size_type)block_size + (size_type)coord_to_cellid(loc));
    }
    // name, bno, cno
    template <bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope, enable_if_all<!is_const, has_name, !in_block> = 0>
    constexpr auto &operator()(const SmallString &propName, size_type blockno,
                               const cell_index_type cellid) noexcept {
      if constexpr (is_power_of_two)
        return grid(propName, (blockno << (size_type)num_block_bits) | (size_type)cellid);
      else
        return grid(propName, blockno * (size_type)block_size + (size_type)cellid);
    }
    template <bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, !in_block> = 0>
    constexpr auto operator()(const SmallString &propName, size_type blockno,
                              const cell_index_type cellid) const noexcept {
      if constexpr (is_power_of_two)
        return grid(propName, (blockno << (size_type)num_block_bits) | (size_type)cellid);
      else
        return grid(propName, blockno * (size_type)block_size + (size_type)cellid);
    }
    // name, chn, bno, ccoord
    template <typename VecT, bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope,
              enable_if_all<!is_const, has_name, !in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto &operator()(const SmallString &propName, channel_counter_type chn,
                               const size_type blockno, const VecInterface<VecT> &loc) noexcept {
      if constexpr (is_power_of_two)
        return grid(propName, chn,
                    (blockno << (size_type)num_block_bits) | (size_type)coord_to_cellid(loc));
      else
        return grid(propName, chn,
                    blockno * (size_type)block_size + (size_type)coord_to_cellid(loc));
    }
    template <typename VecT, bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, !in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto operator()(const SmallString &propName, channel_counter_type chn,
                              const size_type blockno,
                              const VecInterface<VecT> &loc) const noexcept {
      if constexpr (is_power_of_two)
        return grid(propName, chn,
                    (blockno << (size_type)num_block_bits) | (size_type)coord_to_cellid(loc));
      else
        return grid(propName, chn,
                    blockno * (size_type)block_size + (size_type)coord_to_cellid(loc));
    }
    // name, chn, bno, cno
    template <bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope, enable_if_all<!is_const, has_name, !in_block> = 0>
    constexpr auto &operator()(const SmallString &propName, const channel_counter_type chn,
                               const size_type blockno, const cell_index_type cellid) noexcept {
      if constexpr (is_power_of_two)
        return grid(propName, chn, (blockno << (size_type)num_block_bits) | (size_type)cellid);
      else
        return grid(propName, chn, blockno * (size_type)block_size + (size_type)cellid);
    }
    template <bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, !in_block> = 0>
    constexpr auto operator()(const SmallString &propName, const channel_counter_type chn,
                              const size_type blockno,
                              const cell_index_type cellid) const noexcept {
      if constexpr (is_power_of_two)
        return grid(propName, chn, (blockno << (size_type)num_block_bits) | (size_type)cellid);
      else
        return grid(propName, chn, blockno * (size_type)block_size + (size_type)cellid);
    }

    ///
    /// voxel (direct voxel index)
    ///
    // chn, no
    template <bool is_const = is_const_structure, auto in_block = block_scope,
              enable_if_all<!is_const, in_block> = 0>
    constexpr auto &voxel(const channel_counter_type chn, const cell_index_type no) noexcept {
      return grid(chn, no);
    }
    template <auto in_block = block_scope, enable_if_all<in_block> = 0>
    constexpr auto voxel(const channel_counter_type chn, const cell_index_type no) const noexcept {
      return grid(chn, no);
    }
    template <bool is_const = is_const_structure, auto in_block = block_scope,
              enable_if_all<!is_const, !in_block> = 0>
    constexpr auto &voxel(const channel_counter_type chn, const size_type no) noexcept {
      return grid(chn, no);
    }
    template <auto in_block = block_scope, enable_if_all<!in_block> = 0>
    constexpr auto voxel(const channel_counter_type chn, const size_type no) const noexcept {
      return grid(chn, no);
    }
    // name, chn, no
    template <bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope, enable_if_all<!is_const, has_name, in_block> = 0>
    constexpr auto &voxel(const SmallString &propName, const channel_counter_type chn,
                          const cell_index_type no) noexcept {
      return grid(propName, chn, no);
    }
    template <bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, in_block> = 0>
    constexpr auto voxel(const SmallString &propName, const channel_counter_type chn,
                         const cell_index_type no) const noexcept {
      return grid(propName, chn, no);
    }
    template <bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope, enable_if_all<!is_const, has_name, !in_block> = 0>
    constexpr auto &voxel(const SmallString &propName, const channel_counter_type chn,
                          const size_type no) noexcept {
      return grid(propName, chn, no);
    }
    template <bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, !in_block> = 0>
    constexpr auto voxel(const SmallString &propName, const channel_counter_type chn,
                         const size_type no) const noexcept {
      return grid(propName, chn, no);
    }
    template <bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope, enable_if_all<!is_const, has_name, in_block> = 0>
    constexpr auto &voxel(const SmallString &propName, const cell_index_type no) noexcept {
      return grid(propName, no);
    }
    template <bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, in_block> = 0>
    constexpr auto voxel(const SmallString &propName, const cell_index_type no) const noexcept {
      return grid(propName, no);
    }
    template <bool is_const = is_const_structure, bool has_name = with_name,
              auto in_block = block_scope, enable_if_all<!is_const, has_name, !in_block> = 0>
    constexpr auto &voxel(const SmallString &propName, const size_type no) noexcept {
      return grid(propName, no);
    }
    template <bool has_name = with_name, auto in_block = block_scope,
              enable_if_all<has_name, !in_block> = 0>
    constexpr auto voxel(const SmallString &propName, const size_type no) const noexcept {
      return grid(propName, no);
    }

    // rw
    template <auto... Ns, typename Tn, enable_if_t<is_integral_v<Tn>> = 0>
    constexpr auto pack(channel_counter_type chn, Tn cellid) const noexcept {
      return grid.template pack<Ns...>(chn, cellid);
    }
    template <auto... Ns, typename Tn, auto has_name = with_name,
              enable_if_all<is_integral_v<Tn>, has_name> = 0>
    constexpr auto pack(const SmallString &propName, Tn cellid) const noexcept {
      return grid.template pack<Ns...>(propName, cellid);
    }
    template <auto... Ns, auto in_block = block_scope, enable_if_all<!in_block> = 0>
    constexpr auto pack(channel_counter_type chn, size_type blockno,
                        const cell_index_type cellid) const noexcept {
      if constexpr (is_power_of_two)
        return grid.template pack<Ns...>(
            chn, (blockno << (size_type)num_block_bits) | (size_type)cellid);
      else
        return grid.template pack<Ns...>(chn, blockno * (size_type)block_size + (size_type)cellid);
    }
    template <auto... Ns, auto has_name = with_name, auto in_block = block_scope,
              enable_if_all<!in_block, has_name> = 0>
    constexpr auto pack(const SmallString &propName, size_type blockno,
                        const cell_index_type cellid) const noexcept {
      if constexpr (is_power_of_two)
        return grid.template pack<Ns...>(
            propName, (blockno << (size_type)num_block_bits) | (size_type)cellid);
      else
        return grid.template pack<Ns...>(propName,
                                         blockno * (size_type)block_size + (size_type)cellid);
    }
    template <auto... Ns, typename VecT, auto in_block = block_scope,
              enable_if_all<!in_block, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto pack(channel_counter_type chn, size_type blockno,
                        const VecInterface<VecT> &loc) const noexcept {
      if constexpr (is_power_of_two)
        return grid.template pack<Ns...>(
            chn, (blockno << (size_type)num_block_bits) | (size_type)coord_to_cellid(loc));
      else
        return grid.template pack<Ns...>(
            chn, blockno * (size_type)block_size + (size_type)coord_to_cellid(loc));
    }
    template <auto... Ns, typename VecT, auto has_name = with_name, auto in_block = block_scope,
              enable_if_all<!in_block, has_name, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>>
              = 0>
    constexpr auto pack(const SmallString &propName, size_type blockno,
                        const VecInterface<VecT> &loc) const noexcept {
      if constexpr (is_power_of_two)
        return grid.template pack<Ns...>(
            propName, (blockno << (size_type)num_block_bits) | (size_type)coord_to_cellid(loc));
      else
        return grid.template pack<Ns...>(
            propName, blockno * (size_type)block_size + (size_type)coord_to_cellid(loc));
    }
    template <typename VecT, bool is_const = is_const_structure,
              enable_if_all<!is_const, std::is_convertible_v<typename VecT::value_type, value_type>>
              = 0>
    constexpr void set(channel_counter_type chn, cell_index_type cellid,
                       const VecInterface<VecT> &val) noexcept {
      grid.template tuple<VecT::extent>(chn, cellid) = val;
    }
    template <typename VecT, bool is_const = is_const_structure, bool has_name = with_name,
              enable_if_all<!is_const, has_name,
                            std::is_convertible_v<typename VecT::value_type, value_type>>
              = 0>
    constexpr void set(const SmallString &propName, cell_index_type cellid,
                       const VecInterface<VecT> &val) noexcept {
      grid.template tuple<VecT::extent>(propName, cellid) = val;
    }

    constexpr auto numCells() const noexcept { return grid.size(); }
    constexpr auto numBlocks() const noexcept { return grid.numTiles(); }
    constexpr auto numChannels() const noexcept { return grid.numChannels(); }

    view_t grid{};
    value_type dx{0};
  };

  template <execspace_e, typename GridsT, typename = void> struct GridsView;
  template <execspace_e space, typename GridsT> struct GridsView<space, GridsT> {
    static constexpr bool is_const_structure = is_const_v<GridsT>;
    using grids_t = remove_const_t<GridsT>;
    using value_type = typename grids_t::value_type;
    static constexpr int dim = grids_t::dim;
    static constexpr auto side_length = grids_t::side_length;
    static constexpr auto block_space() noexcept { return grids_t::block_space(); }
    static constexpr bool is_power_of_two = grids_t::is_power_of_two;
    static constexpr auto num_cell_bits = grids_t::num_cell_bits;

    using grid_storage_t = typename grids_t::grid_storage_t;
    using grid_view_t = RM_REF_T(proxy<space>(
        {},
        declval<conditional_t<is_const_structure, const grid_storage_t &, grid_storage_t &>>()));
    using grid_block_view_t = typename grid_view_t::tile_view_type;  // tilevector view property
    using size_type = typename grids_t::size_type;
    using channel_counter_type = typename grids_t::channel_counter_type;
    using cell_index_type = typename grids_t::cell_index_type;
    using coord_index_type = typename grids_t::coord_index_type;

    using CellIV = typename grids_t::CellIV;
    using IV = typename grids_t::IV;
    using TV = typename grids_t::TV;

    static constexpr auto cellid_to_coord(cell_index_type cellid) noexcept {
      CellIV ret{CellIV::zeros()};
      if constexpr (is_power_of_two) {
        for (int d = dim - 1; d >= 0 && cellid; --d, cellid >>= num_cell_bits)
          ret[d] = cellid & (side_length - 1);
      } else {
        for (int d = dim - 1; d >= 0 && cellid; --d, cellid /= side_length)
          ret[d] = cellid % side_length;
      }

      return ret;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           is_integral_v<typename VecT::index_type>>
                             = 0>
    static constexpr auto coord_to_cellid(const VecInterface<VecT> &coord) noexcept {
      using Ti = typename VecT::index_type;
      cell_index_type ret{0};
      if constexpr (is_power_of_two)
        for (int d = 0; d != dim; ++d) ret = (ret << num_cell_bits) | coord[d];
      else
        for (int d = 0; d != dim; ++d) ret = (ret * (Ti)side_length) + coord[d];
      return ret;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           is_integral_v<typename VecT::index_type>>
                             = 0>
    static constexpr auto global_coord_to_cellid(const VecInterface<VecT> &coord) noexcept {
      using Ti = typename VecT::index_type;
      cell_index_type ret{0};
      if constexpr (is_power_of_two)
        for (int d = 0; d != dim; ++d)
          ret = (ret << num_cell_bits) | (coord[d] & (Ti)(side_length - 1));
      else
        for (int d = 0; d != dim; ++d) ret = (ret * (Ti)side_length) + (coord[d] % (Ti)side_length);
      return ret;
    }

    constexpr GridsView() = default;
    ~GridsView() = default;
    explicit GridsView(GridsT &grids)
        : _collocatedGrid{proxy<space>({}, grids.grid(collocated_c).blocks)},
          _cellcenteredGrid{proxy<space>({}, grids.grid(cellcentered_c).blocks)},
          _staggeredGrid{proxy<space>({}, grids.grid(staggered_c).blocks)},
          _dx{grids._dx} {}

    template <grid_e category = grid_e::collocated> struct Block {
      static constexpr auto block_space() noexcept { return GridsView::block_space(); }
      constexpr Block(grid_block_view_t tile, value_type dx) noexcept : block{tile}, dx{dx} {}

      constexpr bool hasProperty(const SmallString &propName) const {
        return block.hasProperty(propName);
      }
      template <typename Ti>
      constexpr auto &operator()(channel_counter_type c, const vec<Ti, dim> &loc) noexcept {
        return block(c, coord_to_cellid(loc));
      }
      template <typename Ti>
      constexpr auto operator()(channel_counter_type c, const vec<Ti, dim> &loc) const noexcept {
        return block(c, coord_to_cellid(loc));
      }
      template <typename Ti>
      constexpr auto &operator()(const SmallString &propName, const vec<Ti, dim> &loc) noexcept {
        return block(propName, coord_to_cellid(loc));
      }
      template <typename Ti> constexpr auto operator()(const SmallString &propName,
                                                       const vec<Ti, dim> &loc) const noexcept {
        return block(propName, coord_to_cellid(loc));
      }
      constexpr auto &operator()(channel_counter_type chn, cell_index_type cellid) {
        return block(chn, cellid);
      }
      constexpr auto operator()(channel_counter_type chn, cell_index_type cellid) const {
        return block(chn, cellid);
      }
      constexpr auto &operator()(const SmallString &propName, cell_index_type cellid) {
        return block(propName, cellid);
      }
      constexpr auto operator()(const SmallString &propName, cell_index_type cellid) const {
        return block(propName, cellid);
      }
      template <auto N>
      constexpr auto pack(channel_counter_type chn, cell_index_type cellid) const {
        return block.template pack<N>(chn, cellid);
      }
      template <auto N>
      constexpr auto pack(const SmallString &propName, cell_index_type cellid) const {
        return block.template pack<N>(propName, cellid);
      }
      template <auto N, typename V, bool b = is_const_structure, enable_if_t<!b> = 0>
      constexpr void set(channel_counter_type chn, cell_index_type cellid, const vec<V, N> &val) {
        block.template tuple<N>(chn, cellid) = val;
      }
      template <auto N, typename V, bool b = is_const_structure, enable_if_t<!b> = 0>
      constexpr void set(const SmallString &propName, cell_index_type cellid,
                         const vec<V, N> &val) {
        block.template tuple<N>(propName, cellid) = val;
      }

      constexpr auto size() const noexcept { return block.size(); }

      grid_block_view_t block;
      value_type dx;
    };
    template <grid_e category = grid_e::collocated> struct Grid {
      using size_type = typename GridsView::size_type;
      using cell_index_type = typename GridsView::cell_index_type;
      using value_type = typename GridsView::value_type;
      using channel_counter_type = typename GridsView::channel_counter_type;
      static constexpr int dim = GridsView::dim;
      static constexpr auto side_length = GridsView::side_length;
      static constexpr auto block_space() noexcept { return GridsView::block_space(); }
      static constexpr auto cellid_to_coord(cell_index_type cellid) noexcept {
        return GridsView::cellid_to_coord(cellid);
      }
      template <typename Ti>
      static constexpr auto coord_to_cellid(const vec<Ti, dim> &coord) noexcept {
        return GridsView::coord_to_cellid(coord);
      }
      template <typename Ti>
      static constexpr auto global_coord_to_cellid(const vec<Ti, dim> &coord) noexcept {
        return GridsView::global_coord_to_cellid(coord);
      }
      constexpr Grid(grid_view_t grid, value_type dx) noexcept : grid{grid}, dx{dx} {}

      constexpr bool hasProperty(const SmallString &propName) const {
        return grid.hasProperty(propName);
      }
      constexpr auto block(size_type i) { return Block<category>{grid.tile(i), dx}; }
      constexpr auto block(size_type i) const { return Block<category>{grid.tile(i), dx}; }
      constexpr auto operator[](size_type i) { return block(i); }
      constexpr auto operator[](size_type i) const { return block(i); }
      template <typename Ti> constexpr auto &operator()(channel_counter_type c, size_type blockid,
                                                        const vec<Ti, dim> &loc) noexcept {
        return grid(c, blockid * block_space() + coord_to_cellid(loc));
      }
      template <typename Ti> constexpr auto operator()(channel_counter_type c, size_type blockid,
                                                       const vec<Ti, dim> &loc) const noexcept {
        return grid(c, blockid * block_space() + coord_to_cellid(loc));
      }
      template <typename Ti> constexpr auto &operator()(const SmallString &propName,
                                                        size_type blockid,
                                                        const vec<Ti, dim> &loc) noexcept {
        return grid(propName, blockid * block_space() + coord_to_cellid(loc));
      }
      template <typename Ti> constexpr auto operator()(const SmallString &propName,
                                                       size_type blockid,
                                                       const vec<Ti, dim> &loc) const noexcept {
        return grid(propName, blockid * block_space() + coord_to_cellid(loc));
      }
      constexpr auto &operator()(channel_counter_type chn, size_type cellid) {
        return grid(chn, cellid);
      }
      constexpr auto operator()(channel_counter_type chn, size_type cellid) const {
        return grid(chn, cellid);
      }
      constexpr auto &operator()(const SmallString &propName, size_type cellid) {
        return grid(propName, cellid);
      }
      constexpr auto operator()(const SmallString &propName, size_type cellid) const {
        return grid(propName, cellid);
      }
      template <auto N> constexpr auto pack(channel_counter_type chn, size_type cellid) const {
        return grid.template pack<N>(chn, cellid);
      }
      template <auto N> constexpr auto pack(const SmallString &propName, size_type cellid) const {
        return grid.template pack<N>(propName, cellid);
      }
      template <auto N, typename Ti> constexpr auto pack(channel_counter_type chn,
                                                         size_type blockid,
                                                         const vec<Ti, dim> &loc) const {
        return grid.template pack<N>(chn, blockid * block_space() + coord_to_cellid(loc));
      }
      template <auto N, typename Ti> constexpr auto pack(const SmallString &propName,
                                                         size_type blockid,
                                                         const vec<Ti, dim> &loc) const {
        return grid.template pack<N>(propName, blockid * block_space() + coord_to_cellid(loc));
      }
      template <auto N, typename V, bool b = is_const_structure, enable_if_t<!b> = 0>
      constexpr void set(channel_counter_type chn, size_type cellid, const vec<V, N> &val) {
        grid.template tuple<N>(chn, cellid) = val;
      }
      template <auto N, typename V, bool b = is_const_structure, enable_if_t<!b> = 0>
      constexpr void set(const SmallString &propName, size_type cellid, const vec<V, N> &val) {
        grid.template tuple<N>(propName, cellid) = val;
      }

      constexpr auto size() const noexcept { return grid.size(); }
      constexpr auto numChannels() const noexcept { return grid.numChannels(); }

      grid_view_t grid;
      value_type dx;
    };

    /// grid
    template <grid_e category = grid_e::collocated, bool b = is_const_structure,
              enable_if_t<!b> = 0>
    constexpr auto grid(wrapv<category> = {}) {
      if constexpr (category == grid_e::collocated)
        return Grid<category>{_collocatedGrid, _dx};
      else if constexpr (category == grid_e::cellcentered)
        return Grid<category>{_cellcenteredGrid, _dx};
      else if constexpr (category == grid_e::staggered)
        return Grid<category>{_staggeredGrid, _dx};
    }
    template <grid_e category = grid_e::collocated>
    constexpr auto grid(wrapv<category> = {}) const {
      if constexpr (category == grid_e::collocated)
        return Grid<category>{_collocatedGrid, _dx};
      else if constexpr (category == grid_e::cellcentered)
        return Grid<category>{_cellcenteredGrid, _dx};
      else if constexpr (category == grid_e::staggered)
        return Grid<category>{_staggeredGrid, _dx};
    }

    /// block
    template <grid_e category = grid_e::collocated, bool b = is_const_structure,
              enable_if_t<!b> = 0>
    constexpr auto block(size_type i) {
      if constexpr (category == grid_e::collocated)
        return Block<category>{_collocatedGrid.tile(i), _dx};
      else if constexpr (category == grid_e::cellcentered)
        return Block<category>{_cellcenteredGrid.tile(i), _dx};
      else if constexpr (category == grid_e::staggered)
        return Block<category>{_staggeredGrid.tile(i), _dx};
    }
    template <grid_e category = grid_e::collocated> constexpr auto block(size_type i) const {
      if constexpr (category == grid_e::collocated)
        return Block<category>{_collocatedGrid.tile(i), _dx};
      else if constexpr (category == grid_e::cellcentered)
        return Block<category>{_cellcenteredGrid.tile(i), _dx};
      else if constexpr (category == grid_e::staggered)
        return Block<category>{_staggeredGrid.tile(i), _dx};
    }
    template <grid_e category = grid_e::collocated, bool b = is_const_structure,
              enable_if_t<!b> = 0>
    constexpr auto operator[](size_type i) {
      return block(i);
    }
    template <grid_e category = grid_e::collocated> constexpr auto operator[](size_type i) const {
      return block(i);
    }

    /// cell
    template <grid_e category = grid_e::collocated, bool b = is_const_structure,
              enable_if_t<!b> = 0>
    constexpr value_type &cell(channel_counter_type chn, size_type bid, cell_index_type cid) {
      if constexpr (category == grid_e::collocated)
        return _collocatedGrid(chn, bid * block_space() + cid);
      else if constexpr (category == grid_e::cellcentered)
        return _cellcenteredGrid(chn, bid * block_space() + cid);
      else if constexpr (category == grid_e::staggered)
        return _staggeredGrid(chn, bid * block_space() + cid);
    }
    template <grid_e category = grid_e::collocated>
    constexpr const value_type &cell(channel_counter_type chn, size_type bid,
                                     cell_index_type cid) const {
      if constexpr (category == grid_e::collocated)
        return _collocatedGrid(chn, bid * block_space() + cid);
      else if constexpr (category == grid_e::cellcentered)
        return _cellcenteredGrid(chn, bid * block_space() + cid);
      else if constexpr (category == grid_e::staggered)
        return _staggeredGrid(chn, bid * block_space() + cid);
    }
    template <grid_e category = grid_e::collocated, bool b = is_const_structure,
              enable_if_t<!b> = 0>
    constexpr value_type &operator()(channel_counter_type chn, size_type dof) {
      if constexpr (category == grid_e::collocated)
        return _collocatedGrid(chn, dof);
      else if constexpr (category == grid_e::cellcentered)
        return _cellcenteredGrid(chn, dof);
      else if constexpr (category == grid_e::staggered)
        return _staggeredGrid(chn, dof);
    }
    template <grid_e category = grid_e::collocated>
    constexpr const value_type &operator()(channel_counter_type chn, size_type dof) const {
      if constexpr (category == grid_e::collocated)
        return _collocatedGrid(chn, dof);
      else if constexpr (category == grid_e::cellcentered)
        return _cellcenteredGrid(chn, dof);
      else if constexpr (category == grid_e::staggered)
        return _staggeredGrid(chn, dof);
    }

    grid_view_t _collocatedGrid;
    grid_view_t _cellcenteredGrid;
    grid_view_t _staggeredGrid;
    value_type _dx;
  };

  template <execspace_e space, typename V, int d, auto SideLength, typename Allocator>
  decltype(auto) proxy(Grids<V, d, SideLength, Allocator> &grids) {
    return GridsView<space, Grids<V, d, SideLength, Allocator>>{grids};
  }
  template <execspace_e space, typename V, int d, auto SideLength, typename Allocator>
  decltype(auto) proxy(const Grids<V, d, SideLength, Allocator> &grids) {
    return GridsView<space, const Grids<V, d, SideLength, Allocator>>{grids};
  }

  template <execspace_e space, typename V, int d, auto SideLength, grid_e category,
            typename Allocator>
  decltype(auto) proxy(Grid<V, d, SideLength, category, Allocator> &grid) {
    return GridView<space, Grid<V, d, SideLength, category, Allocator>, true, false>{
        proxy<space>({}, grid.blocks), grid.dx};
  }
  template <execspace_e space, typename V, int d, auto SideLength, grid_e category,
            typename Allocator>
  decltype(auto) proxy(const Grid<V, d, SideLength, category, Allocator> &grid) {
    return GridView<space, const Grid<V, d, SideLength, category, Allocator>, true, false>{
        proxy<space>({}, grid.blocks), grid.dx};
  }

  template <execspace_e space, typename V, int d, auto SideLength, grid_e category,
            typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 Grid<V, d, SideLength, category, Allocator> &grid) {
    return GridView<space, Grid<V, d, SideLength, category, Allocator>, true, false>{
        proxy<space>({}, grid.blocks), grid.dx};
  }
  template <execspace_e space, typename V, int d, auto SideLength, grid_e category,
            typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 const Grid<V, d, SideLength, category, Allocator> &grid) {
    return GridView<space, const Grid<V, d, SideLength, category, Allocator>, true, false>{
        proxy<space>({}, grid.blocks), grid.dx};
  }

}  // namespace zs