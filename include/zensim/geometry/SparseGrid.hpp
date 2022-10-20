#pragma once
#include <utility>

#include "zensim/container/Bcht.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/geometry/LevelSetInterface.h"
#include "zensim/math/Vec.h"
#include "zensim/math/curve/InterpolationKernel.hpp"
#include "zensim/math/matrix/Transform.hpp"
#include "zensim/types/Property.h"
#include "zensim/zpc_tpls/fmt/color.h"

namespace zs {

  template <int dim_ = 3, typename ValueT = f32, int SideLength = 8,
            typename AllocatorT = ZSPmrAllocator<>, typename IndexT = i32>
  struct SparseGrid {
    using value_type = ValueT;
    using allocator_type = AllocatorT;
    using coord_index_type = std::make_signed_t<IndexT>;  // coordinate index type
    using index_type = ssize_t;
    using size_type = std::size_t;

    ///
    static constexpr int dim = dim_;
    static constexpr size_type side_length = SideLength;
    static constexpr size_type block_size = math::pow_integral(side_length, dim);
    using grid_storage_type = TileVector<value_type, block_size, allocator_type>;
    using coord_type = vec<coord_index_type, dim>;
    using affine_matrix_type = vec<value_type, dim + 1, dim + 1>;
    ///
    using table_type = bcht<coord_type, int, true, universal_hash<coord_type>, 16>;

    static constexpr bool value_is_vec = is_vec<value_type>::value;

    constexpr MemoryLocation memoryLocation() const noexcept { return _grid.memoryLocation(); }
    constexpr ProcID devid() const noexcept { return _grid.devid(); }
    constexpr memsrc_e memspace() const noexcept { return _grid.memspace(); }
    constexpr auto size() const noexcept { return _grid.size(); }
    decltype(auto) get_allocator() const noexcept { return _grid.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (std::size_t)1 << (std::size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

    /// query
    constexpr decltype(auto) numBlocks() const noexcept { return _table.size(); }
    constexpr decltype(auto) numReservedBlocks() const noexcept { return _grid.numReservedTiles(); }
    constexpr auto numChannels() const noexcept { return _grid.numChannels(); }
    static constexpr auto zeroValue() noexcept {
      if constexpr (value_is_vec)
        return value_type::zeros();
      else
        return (value_type)0;
    }

    SparseGrid(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
               size_type numBlocks = 0)
        : _table{allocator, numBlocks},
          _grid{allocator, channelTags, numBlocks * block_size},
          _transform{affine_matrix_type::identity()},
          _background{zeroValue()} {}
    SparseGrid(const std::vector<PropertyTag> &channelTags, size_type numBlocks,
               memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : SparseGrid{get_default_allocator(mre, devid), channelTags, numBlocks} {}
    SparseGrid(size_type numChns, size_type numBlocks, memsrc_e mre = memsrc_e::host,
               ProcID devid = -1)
        : SparseGrid{get_default_allocator(mre, devid), {{"unnamed", numChns}}, numBlocks} {}
    SparseGrid(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : SparseGrid{get_default_allocator(mre, devid), {{"sdf", 1}}, 0} {}

    SparseGrid clone(const allocator_type &allocator) const {
      SparseGrid ret{};
      ret._table = _table.clone(allocator);
      ret._grid = _grid.clone(allocator);
      ret._transform = _transform;
      ret._background = _background;
      return ret;
    }
    SparseGrid clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    template <typename ExecPolicy> void resize(ExecPolicy &&policy, size_type numBlocks) {
      _table.resize(FWD(policy), numBlocks);
      _grid.resize(numBlocks * block_size);
    }
    template <typename Policy>
    void append_channels(Policy &&policy, const std::vector<PropertyTag> &tags) {
      _grid.append_channels(FWD(policy), tags);
    }
    template <typename Policy> void reset(Policy &&policy, value_type val) {
      _grid.reset(FWD(policy), val);
    }

    bool hasProperty(const SmallString &str) const noexcept { return _grid.hasProperty(str); }
    constexpr size_type getPropertySize(const SmallString &str) const {
      return _grid.getPropertySize(str);
    }
    constexpr size_type getPropertyOffset(const SmallString &str) const {
      return _grid.getPropertyOffset(str);
    }
    constexpr PropertyTag getPropertyTag(std::size_t i = 0) const {
      return _grid.getPropertyTag(i);
    }
    constexpr const auto &getPropertyTags() const { return _grid.getPropertyTags(); }

    void printTransformation(std::string_view msg = {}) const {
      const auto &a = _transform;
      fmt::print(fg(fmt::color::aquamarine),
                 "[{}] inspecting {} transform:\n[{}, {}, {}, {};\n {}, {}, {}, {};\n {}, {}, {}, "
                 "{};\n {}, {}, {}, {}].\n",
                 msg, get_type_str<SparseGrid>(), a(0, 0), a(0, 1), a(0, 2), a(0, 3), a(1, 0),
                 a(1, 1), a(1, 2), a(1, 3), a(2, 0), a(2, 1), a(2, 2), a(2, 3), a(3, 0), a(3, 1),
                 a(3, 2), a(3, 3));
    }
    template <typename VecTM,
              enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == dim + 1,
                            VecTM::template range_t<1>::value == dim + 1,
                            std::is_floating_point_v<typename VecTM::value_type>> = 0>
    void resetTransformation(const VecInterface<VecTM> &i2w) {
      _transform.self() = i2w;
    }
    auto getIndexToWorldTransformation() const { return _transform.transform; }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    void translate(const VecInterface<VecT> &t) noexcept {
      _transform.postTranslate(t);
    }
    template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == dim,
                                           VecT::template range_t<1>::value == dim> = 0>
    void rotate(const VecInterface<VecT> &r) noexcept {
      _transform.preRotate(r);
    }
    template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == dim,
                                           VecT::template range_t<1>::value == dim> = 0>
    void scale(const VecInterface<VecT> &s) {
      _transform.preScale(s);
    }
    void scale(const value_type s) { scale(s * affine_matrix_type::identity()); }

    table_type _table;
    grid_storage_type _grid;
    math::Transform<value_type, dim> _transform;
    value_type _background;  // background value
  };

  template <typename T, typename = void> struct is_spg : std::false_type {};
  template <int dim, typename ValueT, int SideLength, typename AllocatorT, typename IndexT>
  struct is_spg<SparseGrid<dim, ValueT, SideLength, AllocatorT, IndexT>> : std::true_type {};
  template <typename T> constexpr bool is_spg_v = is_spg<T>::value;

  // forward decl
  template <typename GridT, kernel_e kt, int drv_order> struct GridArena;

  template <execspace_e Space, typename SparseGridT> struct SparseGridView
      : LevelSetInterface<SparseGridView<Space, SparseGridT>> {
    static constexpr bool is_const_structure = std::is_const_v<SparseGridT>;
    static constexpr auto space = Space;
    using container_type = std::remove_const_t<SparseGridT>;
    using value_type = typename container_type::value_type;
    using size_type = typename container_type::size_type;
    using index_type = typename container_type::index_type;

    using coord_index_type = typename container_type::coord_index_type;
    using coord_type = typename container_type::coord_type;

    using table_type = typename container_type::table_type;
    using table_view_type = RM_CVREF_T(proxy<space>(
        {}, std::declval<conditional_t<is_const_structure, const table_type &, table_type &>>()));
    using grid_storage_type = typename container_type::grid_storage_type;
    using grid_view_type = RM_CVREF_T(proxy<space>(
        {},
        std::declval<
            conditional_t<is_const_structure, const grid_storage_type &, grid_storage_type &>>()));

    static constexpr int dim = container_type::dim;
    static constexpr auto side_length = container_type::side_length;
    static constexpr auto block_size = container_type::block_size;

    SparseGridView() noexcept = default;
    ~SparseGridView() noexcept = default;
    constexpr SparseGridView(SparseGridT &sg)
        : _table{proxy<space>(sg._table)},
          _grid{proxy<space>({}, sg._grid)},
          _transform{sg._transform},
          _background{sg._background} {}

    /// helper functions
    // index space <-> world space
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto indexToWorld(const VecInterface<VecT> &X) const {
      return X * _transform;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto worldToIndex(const VecInterface<VecT> &x) const {
      return x * inverse(_transform);
    }
    // linear index <-> node coordinate
    static constexpr coord_type cellid_to_coord(coord_index_type cellid) noexcept {
      coord_type ret{};
      for (auto d = dim - 1; d >= 0; --d, cellid /= side_length)
        ret[d] = (coord_index_type)(cellid % side_length);
      return ret;
    }
    template <typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            std::is_signed_v<typename VecT::value_type>,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>> = 0>
    static constexpr auto local_coord_to_cellid(const VecInterface<VecT> &coord) noexcept {
      using Ti = math::op_result_t<coord_index_type, typename VecT::value_type>;
      Ti ret{coord[0]};
      for (int d = 1; d < dim; ++d) ret = (ret * (Ti)side_length) + (Ti)coord[d];
      return ret;
    }
    template <typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            std::is_signed_v<typename VecT::value_type>,
                            std::is_convertible_v<typename VecT::value_type, coord_index_type>> = 0>
    static constexpr auto global_coord_to_cellid(const VecInterface<VecT> &coord) noexcept {
      // [x, y, z]
      using Ti = math::op_result_t<coord_index_type, typename VecT::value_type>;
      Ti ret{coord[0]};
      for (int d = 1; d < dim; ++d)
        ret = (ret * (Ti)side_length) + ((Ti)coord[d] % (Ti)side_length);
      return ret;
    }
    // node value access
    template <typename VecTI, enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                                            std::is_integral_v<typename VecTI::index_type>> = 0>
    constexpr auto decomposeCoord(const VecInterface<VecTI> &indexCoord) const noexcept {
      auto cellid = indexCoord % side_length;
      auto blockid = indexCoord - cellid;
      return make_tuple(_table.query(blockid), local_coord_to_cellid(cellid));
    }
    constexpr value_type valueOr(coord_index_type chn, typename table_type::index_type blockno,
                                 size_type cellno, value_type defaultVal) const noexcept {
      return blockno == table_type::sentinel_v ? defaultVal : _grid(chn, blockno, cellno);
    }
    template <typename VecTI, enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                                            std::is_integral_v<typename VecTI::value_type>> = 0>
    constexpr auto valueOr(std::false_type, size_type chn, const VecInterface<VecTI> &indexCoord,
                           value_type defaultVal) const noexcept {
      auto [bno, cno] = decomposeCoord(indexCoord);
      return valueOr(chn, bno, cno, defaultVal);
    }
    template <typename VecTI, enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                                            std::is_integral_v<typename VecTI::value_type>> = 0>
    constexpr value_type valueOr(std::true_type, size_type chn,
                                 const VecInterface<VecTI> &indexCoord, size_type orientation,
                                 value_type defaultVal) const noexcept {
      /// 0, ..., dim-1: within cell
      /// dim, ..., dim+dim-1: neighbor cell
      auto coord = indexCoord.clone();
      if (auto f = orientation % (dim + dim); f >= dim) ++coord[f - dim];
      auto [bno, cno] = decomposeCoord(coord);
      return valueOr(chn, bno, cno, defaultVal);
    }

#if 0
    /// sample
    // collocated
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iSample(const SmallString &prop, const VecInterface<VecT> &X) const {
      ...;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wSample(const SmallString &prop, const VecInterface<VecT> &x) const {
      return iSample(worldToIndex(x));
    }
    // staggered
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iStaggeredSample(const SmallString &prop, const VecInterface<VecT> &X,
                                    int f = 0) const {
      ...;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wStaggeredSample(const SmallString &prop, const VecInterface<VecT> &x,
                                    int f = 0) const {
      return iStaggeredSample(worldToIndex(x), f);
    }

    /// packed sample
    // staggered
    template <int N, typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iStaggeredPack(const SmallString &prop, const VecInterface<VecT> &X, int f = 0,
                                  wrapv<N> = {}) const {
      ...;
    }
    template <int N, typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wStaggeredPack(const SmallString &prop, const VecInterface<VecT> &x, int f = 0,
                                  wrapv<N> tag = {}) const {
      return iStaggeredPack<N>(worldToIndex(x), f, tag);
    }
    // collocated
    template <int N, typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iPack(const SmallString &prop, const VecInterface<VecT> &X,
                         wrapv<N> = {}) const {
      iSample(X, )
    }
    template <int N, typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wPack(const SmallString &prop, const VecInterface<VecT> &x,
                         wrapv<N> tag = {}) const {
      return iPack(worldToIndex(x), tag);
    }

    // should provide arena-based (instead of x) sampler

    // ref
    template <typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, index_type>> = 0>
    constexpr decltype(auto) operator()(size_type chn, const VecInterface<VecT> &X) {
      ;
    }
    template <typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, index_type>> = 0>
    constexpr decltype(auto) operator()(const SmallString &prop, const VecInterface<VecT> &X) {
      ;
    }
#endif

    table_view_type _table;
    grid_view_type _grid;
    math::Transform<value_type, dim> _transform;
    value_type _background;
  };

#if 0
  template <typename GridT, kernel_e kt_ = kernel_e::quadratic, int drv_order = 0>
  struct GridArena {
    using grid_type = GridT;
    using value_type = typename grid_type::value_type;
    using index_type = typename grid_type::index_type;

    static_assert(std::is_signed_v<index_type>, "index_type should be a signed integer.");
    static constexpr grid_e category = grid_type::category;
    static constexpr int dim = grid_type::dim;
    static constexpr kernel_e kt = kt_;
    static constexpr int width = (kt == kernel_e::linear ? 2 : (kt == kernel_e::quadratic ? 3 : 4));
    static constexpr int deriv_order = drv_order;

    using TV = typename grid_type::TV;
    using TWM = vec<value_type, grid_type::dim, width>;
    using IV = typename grid_type::IV;

    using coord_index_type = typename grid_type::coord_index_type;
    using channel_counter_type = typename grid_type::channel_counter_type;
    using cell_index_type = typename grid_type::cell_index_type;

    static_assert(deriv_order >= 0 && deriv_order <= 2,
                  "weight derivative order should be a integer within [0, 2]");

    using WeightScratchPad
        = conditional_t<deriv_order == 0, tuple<TWM>,
                        conditional_t<deriv_order == 1, tuple<TWM, TWM>, tuple<TWM, TWM, TWM>>>;

    template <typename Val, std::size_t... Is>
    static constexpr auto arena_type_impl(index_seq<Is...>) {
      return vec<Val, (Is + 1 > 0 ? width : width)...>{};
    }
    template <typename Val, int d> static constexpr auto arena_type() {
      return arena_type_impl<Val>(std::make_index_sequence<d>{});
    }
    template <typename Val> using Arena = RM_CVREF_T(arena_type<Val, grid_type::dim>());

    /// constructors
    // index-space
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate != grid_e::staggered> = 0>
    constexpr GridArena(std::false_type, grid_type &lsv, const VecInterface<VecT> &X) noexcept
        : gridPtr{&lsv}, weights{}, iLocalPos{}, iCorner{} {
      constexpr int lerp_degree
          = (kt == kernel_e::linear ? 0 : (kt == kernel_e::quadratic ? 1 : 2));
      for (int d = 0; d != dim; ++d) iCorner[d] = base_node<lerp_degree>(X[d]);
      iLocalPos = X - iCorner;
      if constexpr (kt == kernel_e::linear)
        weights = linear_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::quadratic)
        weights = quadratic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::cubic)
        weights = cubic_bspline_weights<deriv_order>(iLocalPos);
    }
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate == grid_e::staggered> = 0>
    constexpr GridArena(std::false_type, grid_type &lsv, const VecInterface<VecT> &X,
                            int f) noexcept
        : gridPtr{&lsv}, weights{}, iLocalPos{}, iCorner{} {
      constexpr int lerp_degree
          = (kt == kernel_e::linear ? 0 : (kt == kernel_e::quadratic ? 1 : 2));
      auto delta = TV::init([f](int d) { return d != f ? (value_type)0.5 : (value_type)0; });
      for (int d = 0; d != dim; ++d) iCorner[d] = base_node<lerp_degree>(X[d] - delta[d]);
      iLocalPos = X - (iCorner + delta);
      if constexpr (kt == kernel_e::linear)
        weights = linear_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::quadratic)
        weights = quadratic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::cubic)
        weights = cubic_bspline_weights<deriv_order>(iLocalPos);
    }
    // world-space
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate != grid_e::staggered> = 0>
    constexpr GridArena(std::true_type, grid_type &lsv, const VecInterface<VecT> &x) noexcept
        : GridArena{false_c, lsv, lsv.worldToIndex(x)} {}
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate == grid_e::staggered> = 0>
    constexpr GridArena(std::true_type, grid_type &lsv, const VecInterface<VecT> &x, int f) noexcept
        : GridArena{false_c, lsv, lsv.worldToIndex(x), f} {}
    /// for CTAD
    template <typename VecT, auto cate = category, bool WS = true,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate == grid_e::staggered> = 0>
    constexpr GridArena(wrapv<kt>, wrapv<deriv_order>, grid_type &lsv, const VecInterface<VecT> &x,
                            int f, integral_t<bool, WS> tag = {}) noexcept
        : GridArena{tag, lsv, x, f} {}
    template <typename VecT, auto cate = category, bool WS = true,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate != grid_e::staggered> = 0>
    constexpr GridArena(wrapv<kt>, wrapv<deriv_order>, grid_type &lsv, const VecInterface<VecT> &x,
                            integral_t<bool, WS> tag = {}) noexcept
        : GridArena{tag, lsv, x} {}

    /// scalar arena
    constexpr Arena<value_type> arena(typename grid_type::channel_counter_type chn,
                                      typename grid_type::value_type defaultVal = 0) const noexcept {
      // ensure that chn's orientation is aligned with initialization if within a staggered grid
      Arena<value_type> pad{};
      for (auto offset : ndrange<dim>(width)) {
        pad.val(offset) = gridPtr->value_or(
            chn, iCorner + make_vec<typename grid_type::coord_index_type>(offset), defaultVal);
      }
      return pad;
    }
    constexpr Arena<value_type> arena(const SmallString &propName,
                                      typename grid_type::channel_counter_type chn = 0,
                                      typename grid_type::value_type defaultVal = 0) const noexcept {
      return arena(gridPtr->_grid.propertyOffset(propName) + chn, defaultVal);
    }

    /// minimum
    constexpr value_type minimum(typename grid_type::channel_counter_type chn = 0) const noexcept {
      auto pad = arena(chn, limits<value_type>::max());
      value_type ret = limits<value_type>::max();
      for (auto offset : ndrange<dim>(width))
        if (const auto &v = pad.val(offset); v < ret) ret = v;
      return ret;
    }
    constexpr value_type minimum(const SmallString &propName,
                                 typename grid_type::channel_counter_type chn = 0) const noexcept {
      return minimum(gridPtr->_grid.propertyOffset(propName) + chn);
    }

    /// maximum
    constexpr value_type maximum(typename grid_type::channel_counter_type chn = 0) const noexcept {
      auto pad = arena(chn, limits<value_type>::lowest());
      value_type ret = limits<value_type>::lowest();
      for (auto offset : ndrange<dim>(width))
        if (const auto &v = pad.val(offset); v > ret) ret = v;
      return ret;
    }
    constexpr value_type maximum(const SmallString &propName,
                                 typename grid_type::channel_counter_type chn = 0) const noexcept {
      return maximum(gridPtr->_grid.propertyOffset(propName) + chn);
    }

    /// isample
    constexpr value_type isample(typename grid_type::channel_counter_type chn,
                                 typename grid_type::value_type defaultVal) const noexcept {
      auto pad = arena(chn, defaultVal);
      if constexpr (kt == kernel_e::linear)
        return xlerp(iLocalPos, pad);
      else {
        value_type ret = 0;
        for (auto offset : ndrange<dim>(width)) ret += weight(offset) * pad.val(offset);
        return ret;
      }
    }
    constexpr value_type isample(const SmallString &propName,
                                 typename grid_type::channel_counter_type chn,
                                 typename grid_type::value_type defaultVal) const noexcept {
      return isample(gridPtr->_grid.propertyOffset(propName) + chn, defaultVal);
    }

    /// weight
    template <typename... Tn>
    constexpr value_type weight(const std::tuple<Tn...> &loc) const noexcept {
      return weight_impl(loc, std::index_sequence_for<Tn...>{});
    }
    template <typename... Tn,
              enable_if_all<(!is_std_tuple<Tn>() && ... && (sizeof...(Tn) == dim))> = 0>
    constexpr auto weight(Tn &&...is) const noexcept {
      return weight(std::forward_as_tuple(FWD(is)...));
    }
    /// weight gradient
    template <std::size_t I, typename... Tn, auto ord = deriv_order>
    constexpr std::enable_if_t<(ord > 0), value_type> weightGradient(
        const std::tuple<Tn...> &loc) const noexcept {
      return weightGradient_impl<I>(loc, std::index_sequence_for<Tn...>{});
    }
    void printWeights() {
      value_type sum = 0;
      for (int d = 0; d != dim; ++d) {
        for (int w = 0; w != width; ++w) {
          sum += get<0>(weights)(d, w);
          fmt::print("weights({}, {}): [{}]\t", d, w, get<0>(weights)(d, w));
          if constexpr (deriv_order > 0) fmt::print("[{}]\t", get<1>(weights)(d, w));
          if constexpr (deriv_order > 1) fmt::print("[{}]\t", get<2>(weights)(d, w));
          fmt::print("\n");
        }
      }
      fmt::print("weight sum: {}\n", sum);
    }

    grid_type *gridPtr{nullptr};
    WeightScratchPad weights{};
    TV iLocalPos{TV::zeros()};  // index-space local offset
    IV iCorner{IV::zeros()};    // index-space global coord

  protected:
    template <typename... Tn, std::size_t... Is,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim)> = 0>
    constexpr value_type weight_impl(const std::tuple<Tn...> &loc,
                                     index_seq<Is...>) const noexcept {
      value_type ret{1};
      ((void)(ret *= get<0>(weights)(Is, std::get<Is>(loc))), ...);
      return ret;
    }
    template <std::size_t I, typename... Tn, std::size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr value_type weightGradient_impl(const std::tuple<Tn...> &loc,
                                             index_seq<Is...>) const noexcept {
      value_type ret{1};
      ((void)(ret *= (I == Is ? get<1>(weights)(Is, std::get<Is>(loc))
                              : get<0>(weights)(Is, std::get<Is>(loc)))),
       ...);
      return ret;
    }
    template <typename... Tn, std::size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr TV weightGradients_impl(const std::tuple<Tn...> &loc,
                                      index_seq<Is...>) const noexcept {
      return TV{weightGradient_impl<Is>(loc, index_seq<Is...>{})...};
    }
  };
#endif

}  // namespace zs