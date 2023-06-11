#pragma once
#include <utility>

// #include "zensim/container/Bcht.hpp"
#include "zensim/container/Bht.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/geometry/LevelSetInterface.h"
#include "zensim/math/Vec.h"
#include "zensim/math/curve/InterpolationKernel.hpp"
#include "zensim/math/matrix/Transform.hpp"
#include "zensim/types/Property.h"
#include "zensim/zpc_tpls/fmt/color.h"

namespace zs {

  template <int dim_ = 3, typename ValueT = f32, int SideLength = 8,
            typename AllocatorT = ZSPmrAllocator<>, typename IntegerCoordT = i32>
  struct SparseGrid {
    using value_type = ValueT;
    using allocator_type = AllocatorT;
    using size_type = size_t;
    using index_type = zs::make_signed_t<size_type>;  // associated with the number of blocks

    using integer_coord_component_type = zs::make_signed_t<IntegerCoordT>;
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
    static constexpr integer_coord_component_type side_length = SideLength;
    static_assert(((side_length & (side_length - 1)) == 0) && (side_length > 0),
                  "side length must be power of 2");
    static constexpr integer_coord_component_type block_size = math::pow_integral(side_length, dim);
    using integer_coord_type = vec<integer_coord_component_type, dim>;
    using coord_type = vec<coord_component_type, dim>;
    using packed_value_type = vec<value_type, dim>;
    using grid_storage_type = TileVector<value_type, block_size, allocator_type>;
    ///
    using transform_type = math::Transform<coord_component_type, dim>;
    // using table_type = bcht<integer_coord_type, int, true, universal_hash<integer_coord_type>,
    // 16>;
    using table_type = bht<integer_coord_component_type, dim, int, 16, allocator_type>;

    constexpr MemoryLocation memoryLocation() const noexcept { return _grid.memoryLocation(); }
    constexpr ProcID devid() const noexcept { return _grid.devid(); }
    constexpr memsrc_e memspace() const noexcept { return _grid.memspace(); }
    constexpr auto size() const noexcept { return _grid.size(); }
    decltype(auto) get_allocator() const noexcept { return _grid.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (size_t)1 << (size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

    /// query
    constexpr decltype(auto) numBlocks() const noexcept { return _table.size(); }
    constexpr decltype(auto) numReservedBlocks() const noexcept { return _grid.numReservedTiles(); }
    constexpr auto numChannels() const noexcept { return _grid.numChannels(); }
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

    SparseGrid(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
               size_type numBlocks = 0)
        : _table{allocator, numBlocks},
          _grid{allocator, channelTags, numBlocks * (size_type)block_size},
          _transform{transform_type::identity()},
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

    template <typename ExecPolicy>
    void resize(ExecPolicy &&policy, size_type numBlocks, bool resizeGrid = true) {
      _table.resize(FWD(policy), numBlocks);
      if (resizeGrid) _grid.resize(numBlocks * (size_type)block_size);
    }
    template <typename ExecPolicy> void resizePartition(ExecPolicy &&policy, size_type numBlocks) {
      _table.resize(FWD(policy), numBlocks);
    }
    void resizeGrid(size_type numBlocks) { _grid.resize(numBlocks * (size_type)block_size); }
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
    constexpr PropertyTag getPropertyTag(size_type i = 0) const { return _grid.getPropertyTag(i); }
    constexpr const auto &getPropertyTags() const { return _grid.getPropertyTags(); }

    void printTransformation(std::string_view msg = {}) const {
      const auto &a = _transform;
      if constexpr (dim == 3) {
        fmt::print(
            fg(fmt::color::aquamarine),
            "[{}] inspecting {} transform:\n[{}, {}, {}, {};\n {}, {}, {}, {};\n {}, {}, {}, "
            "{};\n {}, {}, {}, {}].\n",
            msg, get_type_str<SparseGrid>(), a(0, 0), a(0, 1), a(0, 2), a(0, 3), a(1, 0), a(1, 1),
            a(1, 2), a(1, 3), a(2, 0), a(2, 1), a(2, 2), a(2, 3), a(3, 0), a(3, 1), a(3, 2),
            a(3, 3));
      } else if constexpr (dim == 2) {
        fmt::print(fg(fmt::color::aquamarine),
                   "[{}] inspecting {} transform:\n[{}, {}, {};\n {}, {}, {};\n {}, {}, {}].\n",
                   msg, get_type_str<SparseGrid>(), a(0, 0), a(0, 1), a(0, 2), a(1, 0), a(1, 1),
                   a(1, 2), a(2, 0), a(2, 1), a(2, 2));
      } else if constexpr (dim == 1) {
        fmt::print(fg(fmt::color::aquamarine),
                   "[{}] inspecting {} transform:\n[{}, {};\n {}, {}].\n", msg,
                   get_type_str<SparseGrid>(), a(0, 0), a(0, 1), a(1, 0), a(1, 1));
      }
    }
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

    table_type _table;
    grid_storage_type _grid;
    transform_type _transform;
    value_type _background;  // background value
  };

  template <typename T, typename = void> struct is_spg : false_type {};
  template <int dim, typename ValueT, int SideLength, typename AllocatorT, typename IndexT>
  struct is_spg<SparseGrid<dim, ValueT, SideLength, AllocatorT, IndexT>> : true_type {};
  template <typename T> constexpr bool is_spg_v = is_spg<T>::value;

  // forward decl
  template <typename GridViewT, kernel_e kt, int drv_order> struct GridArena;

  template <execspace_e Space, typename SparseGridT> struct SparseGridView
      : LevelSetInterface<SparseGridView<Space, SparseGridT>> {
    static constexpr bool is_const_structure = std::is_const_v<SparseGridT>;
    static constexpr auto space = Space;
    using container_type = remove_const_t<SparseGridT>;
    using value_type = typename container_type::value_type;
    using size_type = typename container_type::size_type;
    using index_type = typename container_type::index_type;

    using integer_coord_component_type = typename container_type::integer_coord_component_type;
    using integer_coord_type = typename container_type::integer_coord_type;
    using coord_component_type = typename container_type::coord_component_type;
    using coord_type = typename container_type::coord_type;
    using packed_value_type = typename container_type::packed_value_type;

    using table_type = typename container_type::table_type;
    using table_view_type = RM_CVREF_T(proxy<space>(
        declval<conditional_t<is_const_structure, const table_type &, table_type &>>()));
    using grid_storage_type = typename container_type::grid_storage_type;
    using grid_view_type = RM_CVREF_T(view<space>(
        {},
        declval<
            conditional_t<is_const_structure, const grid_storage_type &, grid_storage_type &>>()));
    using transform_type = typename container_type::transform_type;

    static constexpr int dim = container_type::dim;
    static constexpr auto side_length = container_type::side_length;
    static constexpr auto block_size = container_type::block_size;

    SparseGridView() noexcept = default;
    ~SparseGridView() noexcept = default;
    constexpr SparseGridView(SparseGridT &sg)
        : _table{proxy<space>(sg._table)},
          _grid{view<space>({}, sg._grid)},
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
    // number of active blocks
    constexpr size_type numActiveBlocks() const { return _table.size(); }
    // linear index <-> node coordinate
    static constexpr integer_coord_type local_offset_to_coord(
        integer_coord_component_type offset) noexcept {
      integer_coord_type ret{};
      for (auto d = dim - 1; d >= 0; --d, offset /= side_length) ret[d] = offset % side_length;
      return ret;
    }
    template <typename VecT,
              enable_if_all<
                  VecT::dim == 1, VecT::extent == dim, is_signed_v<typename VecT::value_type>,
                  std::is_convertible_v<typename VecT::value_type, integer_coord_component_type>>
              = 0>
    static constexpr integer_coord_component_type local_coord_to_offset(
        const VecInterface<VecT> &coord) noexcept {
      integer_coord_component_type ret{coord[0]};
      for (int d = 1; d < dim; ++d)
        ret = (ret * side_length) + (integer_coord_component_type)coord[d];
      return ret;
    }
    template <typename VecT,
              enable_if_all<
                  VecT::dim == 1, VecT::extent == dim, is_signed_v<typename VecT::value_type>,
                  std::is_convertible_v<typename VecT::value_type, integer_coord_component_type>>
              = 0>
    static constexpr integer_coord_component_type global_coord_to_local_offset(
        const VecInterface<VecT> &coord) noexcept {
      // [x, y, z]
      integer_coord_component_type ret{coord[0]};
      for (int d = 1; d < dim; ++d)
        ret = (ret * side_length) + ((integer_coord_component_type)coord[d] % side_length);
      return ret;
    }
    // node value access (used for GridArena::arena_type init)
    template <typename VecTI, enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                                            is_integral_v<typename VecTI::index_type>>
                              = 0>
    constexpr auto decomposeCoord(const VecInterface<VecTI> &indexCoord) const noexcept {
      auto cellid = indexCoord & (side_length - 1);
      auto blockid = indexCoord - cellid;
      return make_tuple(_table.query(blockid), local_coord_to_offset(cellid));
    }
    constexpr value_type valueOr(size_type chn, typename table_type::index_type blockno,
                                 integer_coord_component_type cellno,
                                 value_type defaultVal) const noexcept {
      return blockno == table_type::sentinel_v ? defaultVal : _grid(chn, blockno, cellno);
    }
    template <typename VecTI, enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                                            is_integral_v<typename VecTI::value_type>>
                              = 0>
    constexpr auto valueOr(false_type, size_type chn, const VecInterface<VecTI> &indexCoord,
                           value_type defaultVal) const noexcept {
      auto [bno, cno] = decomposeCoord(indexCoord);
      return valueOr(chn, bno, cno, defaultVal);
    }
    template <typename VecTI, enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                                            is_integral_v<typename VecTI::value_type>>
                              = 0>
    constexpr value_type valueOr(true_type, size_type chn, const VecInterface<VecTI> &indexCoord,
                                 int orientation, value_type defaultVal) const noexcept {
      /// 0, ..., dim-1: within cell
      /// dim, ..., dim+dim-1: neighbor cell
      auto coord = indexCoord.clone();
      if (int f = orientation % (dim + dim); f >= dim) ++coord[f - dim];
      auto [bno, cno] = decomposeCoord(coord);
      return valueOr(chn, bno, cno, defaultVal);
    }

    // arena
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iArena(const VecInterface<VecT> &X, wrapv<kt> = {}) const {
      return GridArena<const SparseGridView, kt, 0>(false_c, *this, X);
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wArena(const VecInterface<VecT> &x, wrapv<kt> = {}) const {
      return GridArena<const SparseGridView, kt, 0>(false_c, *this, worldToIndex(x));
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iArena(const VecInterface<VecT> &X, int f, wrapv<kt> = {}) const {
      return GridArena<const SparseGridView, kt, 0>(false_c, *this, X, f);
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wArena(const VecInterface<VecT> &x, int f, wrapv<kt> = {}) const {
      return GridArena<const SparseGridView, kt, 0>(false_c, *this, worldToIndex(x), f);
    }

    // voxel size
    constexpr coord_type voxelSize() const {
      // does not consider shearing here
      coord_type ret{};
      for (int i = 0; i != dim; ++i) {
        coord_component_type sum = 0;
        for (int d = 0; d != dim; ++d) sum += zs::sqr(_transform(i, d));
        ret.val(i) = zs::sqrt(sum, wrapv<space>{});
      }
      return ret;
    }

    // linear index to coordinate
    constexpr integer_coord_type iCoord(size_type bno, integer_coord_component_type cno) const {
      return _table._activeKeys[bno] + local_offset_to_coord(cno);
    }
    constexpr integer_coord_type iCoord(size_type cellno) const {
      return _table._activeKeys[cellno / block_size] + local_offset_to_coord(cellno % block_size);
    }
    constexpr coord_type wCoord(size_type bno, integer_coord_component_type cno) const {
      return indexToWorld(iCoord(bno, cno));
    }
    constexpr coord_type wCoord(size_type cellno) const { return indexToWorld(iCoord(cellno)); }

    constexpr coord_type iStaggeredCoord(size_type bno, integer_coord_component_type cno,
                                         int f) const {
      // f must be within [0, dim)
      return iCoord(bno, cno) + coord_type::init([f](int d) {
               return d == f ? (coord_component_type)-0.5 : (coord_component_type)0;
             });
    }
    constexpr coord_type iStaggeredCoord(size_type cellno, int f) const {
      return iCoord(cellno) + coord_type::init([f](int d) {
               return d == f ? (coord_component_type)-0.5 : (coord_component_type)0;
             });
    }
    constexpr coord_type wStaggeredCoord(size_type bno, integer_coord_component_type cno,
                                         int f) const {
      return indexToWorld(iStaggeredCoord(bno, cno, f));
    }
    constexpr coord_type wStaggeredCoord(size_type cellno, int f) const {
      return indexToWorld(iStaggeredCoord(cellno, f));
    }

    /// delegate to hash table
    template <typename VecT, enable_if_t<is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto insert(const VecInterface<VecT> &x) {
      auto X_ = worldToIndex(x) + (coord_component_type)0.5;
      auto X = integer_coord_type::init(
          [&X_](int i) { return lower_trunc(X_[i], wrapt<integer_coord_component_type>{}); });
      X -= (X & (side_length - 1));
      return _table.insert(X);
    }
    template <typename VecT, enable_if_t<is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto query(const VecInterface<VecT> &x) const {
      auto X_ = worldToIndex(x) + (coord_component_type)0.5;
      auto X = integer_coord_type::init(
          [&X_](int i) { return lower_trunc(X_[i], wrapt<integer_coord_component_type>{}); });
      X -= (X & (side_length - 1));
      return _table.query(X);
    }

    /// sample
    // collocated
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iSample(size_type chn, const VecInterface<VecT> &X, wrapv<kt> = {}) const {
      auto pad = GridArena<const SparseGridView, kt, 0>(false_c, *this, X);
      return pad.isample(chn, _background);
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iSample(const SmallString &prop, const VecInterface<VecT> &X,
                           wrapv<kt> = {}) const {
      auto pad = GridArena<const SparseGridView, kt, 0>(false_c, *this, X);
      return pad.isample(prop, 0, _background);
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iSample(const SmallString &prop, size_type chn, const VecInterface<VecT> &X,
                           wrapv<kt> = {}) const {
      auto pad = GridArena<const SparseGridView, kt, 0>(false_c, *this, X);
      return pad.isample(prop, chn, _background);
    }

    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wSample(size_type chn, const VecInterface<VecT> &x, wrapv<kt> = {}) const {
      return iSample(chn, worldToIndex(x), wrapv<kt>{});
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wSample(const SmallString &prop, const VecInterface<VecT> &x,
                           wrapv<kt> = {}) const {
      return iSample(prop, worldToIndex(x), wrapv<kt>{});
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wSample(const SmallString &prop, size_type chn, const VecInterface<VecT> &x,
                           wrapv<kt> = {}) const {
      return iSample(prop, chn, worldToIndex(x), wrapv<kt>{});
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      return wSample("sdf", x);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getNormal(const VecInterface<VecT> &x) const noexcept {
      typename VecT::template variant_vec<value_type, typename VecT::extents> diff{}, v1{}, v2{};
      value_type eps = (value_type)1e-6;
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
    // staggered
    template <typename VecT = int, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                                 is_signed_v<typename VecT::value_type>>
                                   = 0>
    constexpr value_type iStaggeredCellSample(size_type propOffset, int d,
                                              const VecInterface<VecT> &X,
                                              typename table_type::index_type blockno,
                                              integer_coord_component_type cellno, int f) const {
      static_assert(dim == 2 || dim == 3, "only implements 2d & 3d for now.");
      using Ti = integer_coord_component_type;
      if (d == f) return valueOr(propOffset + d, blockno, cellno, _background);
      return (valueOr(propOffset + d, blockno, cellno, _background)
              + valueOr(false_c, propOffset + d,
                        X + integer_coord_type::init([d](int i) -> Ti { return i == d ? 1 : 0; }),
                        _background)
              + valueOr(false_c, propOffset + d,
                        X + integer_coord_type::init([f](int i) -> Ti { return i == f ? -1 : 0; }),
                        _background)
              + valueOr(false_c, propOffset + d, X + integer_coord_type::init([d, f](int i) -> Ti {
                                                   return i == d ? 1 : (i == f ? -1 : 0);
                                                 }),
                        _background))
             * (coord_component_type)0.25;
    }
    template <typename VecT = int, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                                 is_signed_v<typename VecT::value_type>>
                                   = 0>
    constexpr auto iStaggeredCellSample(const SmallString &prop, int chn,
                                        const VecInterface<VecT> &X, int f) const {
      static_assert(dim == 2 || dim == 3, "only implements 2d & 3d for now.");
      auto coord = X.clone();
      if (f >= dim) ++coord[f -= dim];
      auto [bno, cno] = decomposeCoord(coord);
      return iStaggeredCellSample(_grid.propertyOffset(prop), chn, coord, bno, cno, f);
    }

    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iStaggeredSample(size_type chn, int f, const VecInterface<VecT> &X,
                                    wrapv<kt> = {}) const {
      auto pad = GridArena<const SparseGridView, kt, 0>(false_c, *this, X, f);
      return pad.isample(chn + f, _background);
    }
    template <kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto iStaggeredSample(const SmallString &prop, int f, const VecInterface<VecT> &X,
                                    wrapv<kt> = {}) const {
      auto pad = GridArena<const SparseGridView, kt, 0>(false_c, *this, X, f);
      return pad.isample(prop, f, _background);
    }

    template <kernel_e kt = kernel_e::linear, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wStaggeredSample(size_type chn, int f, const VecInterface<VecT> &x,
                                    wrapv<kt> = {}) const {
      return iStaggeredSample(chn, f, worldToIndex(x), wrapv<kt>{});
    }
    template <kernel_e kt = kernel_e::linear, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto wStaggeredSample(const SmallString &prop, int f, const VecInterface<VecT> &x,
                                    wrapv<kt> = {}) const {
      return iStaggeredSample(prop, f, worldToIndex(x), wrapv<kt>{});
    }

    /// packed sample
    // staggered
    template <typename VecT = int, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                                 is_signed_v<typename VecT::value_type>>
                                   = 0>
    constexpr packed_value_type iStaggeredCellPack(size_type propOffset,
                                                   const VecInterface<VecT> &X,
                                                   typename table_type::index_type blockno,
                                                   integer_coord_component_type cellno,
                                                   int f) const {
      static_assert(dim == 2 || dim == 3, "only implements 2d & 3d for now.");
      packed_value_type ret{};
      using Ti = integer_coord_component_type;
      for (int d = 0; d != dim; ++d) {
        if (d == f)
          ret.val(d) = valueOr(propOffset + d, blockno, cellno, _background);
        else
          ret.val(d)
              = (valueOr(propOffset + d, blockno, cellno, _background)
                 + valueOr(false_c, propOffset + d, X + integer_coord_type::init([d](int i) -> Ti {
                                                      return i == d ? 1 : 0;
                                                    }),
                           _background)
                 + valueOr(false_c, propOffset + d, X + integer_coord_type::init([f](int i) -> Ti {
                                                      return i == f ? -1 : 0;
                                                    }),
                           _background)
                 + valueOr(false_c, propOffset + d,
                           X + integer_coord_type::init([d, f](int i) -> Ti {
                             return i == d ? 1 : (i == f ? -1 : 0);
                           }),
                           _background))
                * (coord_component_type)0.25;
      }
      return ret;
    }
    template <typename VecT = int, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                                 is_signed_v<typename VecT::value_type>>
                                   = 0>
    constexpr packed_value_type iStaggeredCellPack(const SmallString &propName,
                                                   const VecInterface<VecT> &X, int f) const {
      static_assert(dim == 2 || dim == 3, "only implements 2d & 3d for now.");
      auto coord = X.clone();
      if (f >= dim) ++coord[f -= dim];
      auto [bno, cno] = decomposeCoord(coord);
      return iStaggeredCellPack(_grid.propertyOffset(propName), coord, bno, cno, f);
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto iStaggeredPack(size_type chnOffset, const VecInterface<VecT> &X, wrapv<N> = {},
                                  wrapv<kt> = {}) const {
      zs::vec<value_type, N> ret{};
      for (int i = 0; i != N; ++i) {
        const auto pad = GridArena<const SparseGridView, kt, 0>(false_c, *this, X, i);
        ret.val(i) = pad.isample(chnOffset + i, _background);
      }
      return ret;
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto iStaggeredPack(const SmallString &prop, const VecInterface<VecT> &X,
                                  wrapv<N> = {}, wrapv<kt> = {}) const {
      return iStaggeredPack(_grid.propertyOffset(prop), X, wrapv<N>{}, wrapv<kt>{});
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto iStaggeredPack(const SmallString &prop, size_type chn,
                                  const VecInterface<VecT> &X, wrapv<N> = {},
                                  wrapv<kt> = {}) const {
      return iStaggeredPack(_grid.propertyOffset(prop) + chn, X, wrapv<N>{}, wrapv<kt>{});
    }

    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto wStaggeredPack(size_type chnOffset, const VecInterface<VecT> &x, wrapv<N> = {},
                                  wrapv<kt> = {}) const {
      return iStaggeredPack(chnOffset, worldToIndex(x), wrapv<N>{}, wrapv<kt>{});
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto wStaggeredPack(const SmallString &prop, const VecInterface<VecT> &x,
                                  wrapv<N> = {}, wrapv<kt> = {}) const {
      return iStaggeredPack(_grid.propertyOffset(prop), worldToIndex(x), wrapv<N>{}, wrapv<kt>{});
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto wStaggeredPack(const SmallString &prop, size_type chn,
                                  const VecInterface<VecT> &x, wrapv<N> = {},
                                  wrapv<kt> = {}) const {
      return iStaggeredPack(_grid.propertyOffset(prop) + chn, worldToIndex(x), wrapv<N>{},
                            wrapv<kt>{});
    }
    // collocated
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto iPack(size_type chnOffset, const VecInterface<VecT> &X, wrapv<N> = {},
                         wrapv<kt> = {}) const {
      zs::vec<value_type, N> ret{};
      const auto pad = GridArena<const SparseGridView, kt, 0>(false_c, *this, X);
      for (int i = 0; i != N; ++i) ret.val(i) = pad.isample(chnOffset + i, _background);
      return ret;
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto iPack(const SmallString &prop, const VecInterface<VecT> &X, wrapv<N> = {},
                         wrapv<kt> = {}) const {
      return iPack(_grid.propertyOffset(prop), X, wrapv<N>{}, wrapv<kt>{});
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT = int,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto iPack(const SmallString &prop, size_type chn, const VecInterface<VecT> &X,
                         wrapv<N> = {}, wrapv<kt> = {}) const {
      return iPack(_grid.propertyOffset(prop) + chn, X, wrapv<N>{}, wrapv<kt>{});
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto wPack(size_type chnOffset, const VecInterface<VecT> &x, wrapv<N> = {},
                         wrapv<kt> = {}) const {
      return iPack(chnOffset, worldToIndex(x), wrapv<N>{}, wrapv<kt>{});
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto wPack(const SmallString &prop, const VecInterface<VecT> &x, wrapv<N> = {},
                         wrapv<kt> = {}) const {
      return iPack(_grid.propertyOffset(prop), worldToIndex(x), wrapv<N>{}, wrapv<kt>{});
    }
    template <int N = dim, kernel_e kt = kernel_e::linear, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, (N <= dim)> = 0>
    constexpr auto wPack(const SmallString &prop, size_type chn, const VecInterface<VecT> &x,
                         wrapv<N> = {}, wrapv<kt> = {}) const {
      return iPack(_grid.propertyOffset(prop) + chn, worldToIndex(x), wrapv<N>{}, wrapv<kt>{});
    }

    /// node access
    // ref
    constexpr decltype(auto) operator()(size_type chn, size_type cellno) {
      return _grid(chn, cellno);
    }
    constexpr decltype(auto) operator()(size_type chn, size_type blockno, size_type cellno) {
      return _grid(chn, blockno, cellno);
    }
    constexpr decltype(auto) operator()(const SmallString &prop, size_type blockno,
                                        size_type cellno) {
      return this->operator()(_grid.propertyOffset(prop), blockno, cellno);
    }
    constexpr decltype(auto) operator()(const SmallString &prop, size_type chn, size_type blockno,
                                        size_type cellno) {
      return this->operator()(_grid.propertyOffset(prop) + chn, blockno, cellno);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr decltype(auto) operator()(size_type chn, const VecInterface<VecT> &X) {
      auto [blockno, cellno] = decomposeCoord(X);
      if (blockno == table_type::sentinel_v) printf("accessing an inactive voxel (block)!\n");
      return _grid(chn, blockno, cellno);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr decltype(auto) operator()(const SmallString &prop, const VecInterface<VecT> &X) {
      return this->operator()(_grid.propertyOffset(prop), X);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr decltype(auto) operator()(const SmallString &prop, size_type chn,
                                        const VecInterface<VecT> &X) {
      return this->operator()(_grid.propertyOffset(prop) + chn, X);
    }
    // value
    constexpr auto operator()(size_type chn, size_type cellno) const { return _grid(chn, cellno); }
    constexpr auto operator()(size_type chn, size_type blockno, size_type cellno) const {
      return _grid(chn, blockno, cellno);
    }
    constexpr auto operator()(const SmallString &prop, size_type blockno, size_type cellno) const {
      return this->operator()(_grid.propertyOffset(prop), blockno, cellno);
    }
    constexpr auto operator()(const SmallString &prop, size_type chn, size_type blockno,
                              size_type cellno) const {
      return this->operator()(_grid.propertyOffset(prop) + chn, blockno, cellno);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr auto operator()(size_type chn, const VecInterface<VecT> &X) const {
      auto [blockno, cellno] = decomposeCoord(X);
      if (blockno == table_type::sentinel_v) return _background;
      return _grid(chn, blockno, cellno);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr auto operator()(const SmallString &prop, const VecInterface<VecT> &X) const {
      return this->operator()(_grid.propertyOffset(prop), X);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr auto operator()(const SmallString &prop, size_type chn,
                              const VecInterface<VecT> &X) const {
      return this->operator()(_grid.propertyOffset(prop) + chn, X);
    }

    /// @brief access value by index
    constexpr auto value(size_type chn, size_type cellno) const { return _grid(chn, cellno); }
    constexpr auto value(size_type chn, size_type blockno, size_type cellno) const {
      return _grid(chn, blockno, cellno);
    }
    constexpr auto value(const SmallString &prop, size_type blockno, size_type cellno) const {
      return value(_grid.propertyOffset(prop), blockno, cellno);
    }
    constexpr auto value(const SmallString &prop, size_type chn, size_type blockno,
                         size_type cellno) const {
      return value(_grid.propertyOffset(prop) + chn, blockno, cellno);
    }
    /// @brief access value by coord
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr auto value(size_type chn, const VecInterface<VecT> &X) const {
      auto [blockno, cellno] = decomposeCoord(X);
      if (blockno == table_type::sentinel_v) return _background;
      return _grid(chn, blockno, cellno);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr auto value(const SmallString &prop, const VecInterface<VecT> &X) const {
      return value(_grid.propertyOffset(prop), X);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr auto value(const SmallString &prop, size_type chn,
                         const VecInterface<VecT> &X) const {
      return value(_grid.propertyOffset(prop) + chn, X);
    }
    /// @note default value override
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr auto value(size_type chn, const VecInterface<VecT> &X,
                         value_type backgroundVal) const {
      auto [blockno, cellno] = decomposeCoord(X);
      if (blockno == table_type::sentinel_v) return backgroundVal;
      return _grid(chn, blockno, cellno);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr auto value(const SmallString &prop, const VecInterface<VecT> &X,
                         value_type backgroundVal) const {
      return value(_grid.propertyOffset(prop), X, backgroundVal);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr auto value(const SmallString &prop, size_type chn, const VecInterface<VecT> &X,
                         value_type backgroundVal) const {
      return value(_grid.propertyOffset(prop) + chn, X, backgroundVal);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type,
                                                                 integer_coord_component_type>>
                             = 0>
    constexpr bool hasVoxel(const VecInterface<VecT> &X) const {
      auto [blockno, cellno] = decomposeCoord(X);
      return blockno != table_type::sentinel_v;
    }
    // more delegations
    template <auto... Ns> constexpr auto pack(size_type chnOffset, size_type cellno) const {
      return _grid.template pack<Ns...>(chnOffset, cellno);
    }
    template <auto... Ns>
    constexpr auto pack(size_type chnOffset, size_type blockno, size_type cellno) const {
      return _grid.template pack<Ns...>(chnOffset, blockno, cellno);
    }
    template <auto... Ns>
    constexpr auto pack(value_seq<Ns...>, size_type chnOffset, size_type cellno) const {
      return pack<Ns...>(chnOffset, cellno);
    }
    template <auto... Ns> constexpr auto pack(value_seq<Ns...>, size_type chnOffset,
                                              size_type blockno, size_type cellno) const {
      return pack<Ns...>(chnOffset, blockno, cellno);
    }
    // acquire block / tile
    constexpr auto block(size_type blockno) { return _grid.tile(blockno); }
    constexpr auto block(size_type blockno) const { return _grid.tile(blockno); }

    table_view_type _table;
    grid_view_type _grid;
    transform_type _transform;
    value_type _background;
  };

  template <execspace_e ExecSpace, int dim, typename ValueT, int SideLength, typename AllocatorT,
            typename IntegerCoordT>
  constexpr decltype(auto) proxy(
      const std::vector<SmallString> &tagNames,
      SparseGrid<dim, ValueT, SideLength, AllocatorT, IntegerCoordT> &spg) {
    return SparseGridView<ExecSpace,
                          SparseGrid<dim, ValueT, SideLength, AllocatorT, IntegerCoordT>>{spg};
  }
  template <execspace_e ExecSpace, int dim, typename ValueT, int SideLength, typename AllocatorT,
            typename IntegerCoordT>
  constexpr decltype(auto) proxy(
      const std::vector<SmallString> &tagNames,
      const SparseGrid<dim, ValueT, SideLength, AllocatorT, IntegerCoordT> &spg) {
    return SparseGridView<ExecSpace,
                          const SparseGrid<dim, ValueT, SideLength, AllocatorT, IntegerCoordT>>{
        spg};
  }

  template <execspace_e ExecSpace, int dim, typename ValueT, int SideLength, typename AllocatorT,
            typename IntegerCoordT>
  constexpr decltype(auto) proxy(
      SparseGrid<dim, ValueT, SideLength, AllocatorT, IntegerCoordT> &spg) {
    return SparseGridView<ExecSpace,
                          SparseGrid<dim, ValueT, SideLength, AllocatorT, IntegerCoordT>>{spg};
  }
  template <execspace_e ExecSpace, int dim, typename ValueT, int SideLength, typename AllocatorT,
            typename IntegerCoordT>
  constexpr decltype(auto) proxy(
      const SparseGrid<dim, ValueT, SideLength, AllocatorT, IntegerCoordT> &spg) {
    return SparseGridView<ExecSpace,
                          const SparseGrid<dim, ValueT, SideLength, AllocatorT, IntegerCoordT>>{
        spg};
  }

  template <typename GridViewT, kernel_e kt_ = kernel_e::linear, int drv_order = 0>
  struct GridArena {
    using grid_view_type = GridViewT;
    using value_type = typename grid_view_type::value_type;
    using size_type = typename grid_view_type::size_type;

    using integer_coord_component_type = typename grid_view_type::integer_coord_component_type;
    using integer_coord_type = typename grid_view_type::integer_coord_type;
    using coord_component_type = typename grid_view_type::coord_component_type;
    using coord_type = typename grid_view_type::coord_type;

    static constexpr int dim = grid_view_type::dim;
    static constexpr kernel_e kt = kt_;
    static constexpr int width = [](kernel_e kt) {
      if (kt == kernel_e::linear || kt == kernel_e::delta2)
        return 2;
      else if (kt == kernel_e::quadratic || kt == kernel_e::delta3)
        return 3;
      else if (kt == kernel_e::cubic || kt == kernel_e::delta4)
        return 4;
    }(kt);
    static constexpr int deriv_order = drv_order;

    using TWM = vec<coord_component_type, dim, width>;

    static_assert(deriv_order >= 0 && deriv_order <= 2,
                  "weight derivative order should be an integer within [0, 2]");
    static_assert(((kt == kernel_e::delta2 || kt == kernel_e::delta3 || kt == kernel_e::delta4)
                   && deriv_order == 0)
                      || (kt == kernel_e::linear || kt == kernel_e::quadratic
                          || kt == kernel_e::cubic),
                  "weight derivative order should be 0 when using delta kernel");

    using WeightScratchPad
        = conditional_t<deriv_order == 0, tuple<TWM>,
                        conditional_t<deriv_order == 1, tuple<TWM, TWM>, tuple<TWM, TWM, TWM>>>;

    template <typename ValT, size_t... Is>
    static constexpr auto deduce_arena_type_impl(index_sequence<Is...>) {
      return vec<ValT, (Is + 1 > 0 ? width : width)...>{};
    }
    template <typename ValT, int d> static constexpr auto deduce_arena_type() {
      return deduce_arena_type_impl<ValT>(make_index_sequence<d>{});
    }
    template <typename ValT> using arena_type = RM_CVREF_T(deduce_arena_type<ValT, dim>());

    /// constructors
    /// index-space ctors
    // collocated grid
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr GridArena(false_type, grid_view_type &sgv, const VecInterface<VecT> &X) noexcept
        : gridPtr{&sgv}, weights{}, iLocalPos{}, iCorner{} {
      constexpr int lerp_degree
          = ((kt == kernel_e::linear || kt == kernel_e::delta2)
                 ? 0
                 : ((kt == kernel_e::quadratic || kt == kernel_e::delta3) ? 1 : 2));
      for (int d = 0; d != dim; ++d)
        iCorner[d] = base_node<lerp_degree>(X[d], wrapt<integer_coord_component_type>{});
      iLocalPos = X - iCorner;
      if constexpr (kt == kernel_e::linear)
        weights = linear_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::quadratic)
        weights = quadratic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::cubic)
        weights = cubic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::delta2)
        weights = delta_2point_weights(iLocalPos);
      else if constexpr (kt == kernel_e::delta3)
        weights = delta_3point_weights(iLocalPos);
      else if constexpr (kt == kernel_e::delta4)
        weights = delta_4point_weights(iLocalPos);
    }
    // staggered grid
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr GridArena(false_type, grid_view_type &sgv, const VecInterface<VecT> &X,
                        int f) noexcept
        : gridPtr{&sgv}, weights{}, iLocalPos{}, iCorner{} {
      constexpr int lerp_degree
          = ((kt == kernel_e::linear || kt == kernel_e::delta2)
                 ? 0
                 : ((kt == kernel_e::quadratic || kt == kernel_e::delta3) ? 1 : 2));
      const auto delta = coord_type::init([f = f % dim](int d) {
        return d != f ? (coord_component_type)0 : (coord_component_type)-0.5;
      });
      for (int d = 0; d != dim; ++d) iCorner[d] = base_node<lerp_degree>(X[d] - delta[d]);
      iLocalPos = X - (iCorner + delta);
      if constexpr (kt == kernel_e::linear)
        weights = linear_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::quadratic)
        weights = quadratic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::cubic)
        weights = cubic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::delta2)
        weights = delta_2point_weights(iLocalPos);
      else if constexpr (kt == kernel_e::delta3)
        weights = delta_3point_weights(iLocalPos);
      else if constexpr (kt == kernel_e::delta4)
        weights = delta_4point_weights(iLocalPos);
    }
    /// world-space ctors
    // collocated grid
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr GridArena(true_type, grid_view_type &sgv, const VecInterface<VecT> &x) noexcept
        : GridArena{false_c, sgv, sgv.worldToIndex(x)} {}
    // staggered grid
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr GridArena(true_type, grid_view_type &sgv, const VecInterface<VecT> &x, int f) noexcept
        : GridArena{false_c, sgv, sgv.worldToIndex(x), f} {}

    /// scalar arena
    constexpr arena_type<value_type> arena(size_type chn,
                                           value_type defaultVal = {}) const noexcept {
      // ensure that chn's orientation is aligned with initialization if within a staggered grid
      arena_type<value_type> pad{};
      for (auto offset : ndrange<dim>(width)) {
        pad.val(offset) = gridPtr->valueOr(
            false_c, chn, iCorner + make_vec<integer_coord_component_type>(offset), defaultVal);
      }
      return pad;
    }
    constexpr arena_type<value_type> arena(const SmallString &propName, size_type chn = 0,
                                           value_type defaultVal = {}) const noexcept {
      return arena(gridPtr->_grid.propertyOffset(propName) + chn, defaultVal);
    }

    /// helpers
    constexpr auto range() const noexcept { return ndrange<dim>(width); }

    template <typename... Tn> constexpr auto offset(const std::tuple<Tn...> &loc) const noexcept {
      return make_vec<int>(loc);
    }
    template <typename... Tn> constexpr auto offset(const tuple<Tn...> &loc) const noexcept {
      return make_vec<int>(loc);
    }

    template <typename... Tn> constexpr auto coord(const std::tuple<Tn...> &loc) const noexcept {
      return iCorner + offset(loc);
    }
    template <typename... Tn> constexpr auto coord(const tuple<Tn...> &loc) const noexcept {
      return iCorner + offset(loc);
    }

    /// minimum
    constexpr value_type minimum(size_type chn = 0) const noexcept {
      auto pad = arena(chn, limits<value_type>::max());
      value_type ret = limits<value_type>::max();
      for (auto offset : ndrange<dim>(width))
        if (const auto &v = pad.val(offset); v < ret) ret = v;
      return ret;
    }
    constexpr value_type minimum(const SmallString &propName, size_type chn = 0) const noexcept {
      return minimum(gridPtr->_grid.propertyOffset(propName) + chn);
    }

    /// maximum
    constexpr value_type maximum(size_type chn = 0) const noexcept {
      auto pad = arena(chn, limits<value_type>::lowest());
      value_type ret = limits<value_type>::lowest();
      for (auto offset : ndrange<dim>(width))
        if (const auto &v = pad.val(offset); v > ret) ret = v;
      return ret;
    }
    constexpr value_type maximum(const SmallString &propName, size_type chn = 0) const noexcept {
      return maximum(gridPtr->_grid.propertyOffset(propName) + chn);
    }

    /// isample
    constexpr value_type isample(size_type chn, value_type defaultVal = {}) const noexcept {
      auto pad = arena(chn, defaultVal);
      if constexpr (kt == kernel_e::linear)
        return xlerp(iLocalPos, pad);
      else {
        value_type ret = 0;
        for (auto offset : ndrange<dim>(width)) ret += weight(offset) * pad.val(offset);
        return ret;
      }
    }
    constexpr value_type isample(const SmallString &propName, size_type chn,
                                 value_type defaultVal = {}) const noexcept {
      return isample(gridPtr->_grid.propertyOffset(propName) + chn, defaultVal);
    }

    /// weight
    template <typename... Tn>
    constexpr value_type weight(const std::tuple<Tn...> &loc) const noexcept {
      return weight_impl(loc, index_sequence_for<Tn...>{});
    }
    template <typename... Tn>
    constexpr value_type weight(const zs::tuple<Tn...> &loc) const noexcept {
      return weight_impl(loc, index_sequence_for<Tn...>{});
    }
    template <typename... Tn, enable_if_all<((!is_tuple_v<Tn> && !is_std_tuple<Tn>()) && ...
                                             && (sizeof...(Tn) == dim))>
                              = 0>
    constexpr auto weight(Tn &&...is) const noexcept {
      return weight(zs::forward_as_tuple(FWD(is)...));
    }
    /// weight gradient
    template <zs::size_t I, typename... Tn, auto ord = deriv_order>
    constexpr enable_if_type<(ord > 0), coord_component_type> weightGradient(
        const std::tuple<Tn...> &loc) const noexcept {
      return weightGradient_impl<I>(loc, index_sequence_for<Tn...>{});
    }
    template <zs::size_t I, typename... Tn, auto ord = deriv_order>
    constexpr enable_if_type<(ord > 0), coord_component_type> weightGradient(
        const zs::tuple<Tn...> &loc) const noexcept {
      return weightGradient_impl<I>(loc, index_sequence_for<Tn...>{});
    }
    /// weight gradient
    template <typename... Tn, auto ord = deriv_order>
    constexpr enable_if_type<(ord > 0), coord_type> weightsGradient(
        const std::tuple<Tn...> &loc) const noexcept {
      return weightGradients_impl(loc, index_sequence_for<Tn...>{});
    }
    template <typename... Tn, auto ord = deriv_order>
    constexpr enable_if_type<(ord > 0), coord_type> weightsGradient(
        const zs::tuple<Tn...> &loc) const noexcept {
      return weightGradients_impl(loc, index_sequence_for<Tn...>{});
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

    grid_view_type *gridPtr{nullptr};
    WeightScratchPad weights{};
    coord_type iLocalPos{coord_type::zeros()};                // index-space local offset
    integer_coord_type iCorner{integer_coord_type::zeros()};  // index-space global coord

  protected:
    /// std tuple
    template <typename... Tn, size_t... Is,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim)> = 0>
    constexpr coord_component_type weight_impl(const std::tuple<Tn...> &loc,
                                               index_sequence<Is...>) const noexcept {
      coord_component_type ret{1};
      ((void)(ret *= get<0>(weights)(Is, std::get<Is>(loc))), ...);
      return ret;
    }
    template <zs::size_t I, typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr coord_component_type weightGradient_impl(const std::tuple<Tn...> &loc,
                                                       index_sequence<Is...>) const noexcept {
      coord_component_type ret{1};
      ((void)(ret *= (I == Is ? get<1>(weights)(Is, std::get<Is>(loc))
                              : get<0>(weights)(Is, std::get<Is>(loc)))),
       ...);
      return ret;
    }
    template <typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr coord_type weightGradients_impl(const std::tuple<Tn...> &loc,
                                              index_sequence<Is...>) const noexcept {
      return coord_type{weightGradient_impl<Is>(loc, index_sequence<Is...>{})...};
    }
    /// zs tuple
    template <typename... Tn, size_t... Is,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim)> = 0>
    constexpr coord_component_type weight_impl(const zs::tuple<Tn...> &loc,
                                               index_sequence<Is...>) const noexcept {
      coord_component_type ret{1};
      ((void)(ret *= get<0>(weights)(Is, zs::get<Is>(loc))), ...);
      return ret;
    }
    template <zs::size_t I, typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr coord_component_type weightGradient_impl(const zs::tuple<Tn...> &loc,
                                                       index_sequence<Is...>) const noexcept {
      coord_component_type ret{1};
      ((void)(ret *= (I == Is ? get<1>(weights)(Is, zs::get<Is>(loc))
                              : get<0>(weights)(Is, zs::get<Is>(loc)))),
       ...);
      return ret;
    }
    template <typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr coord_type weightGradients_impl(const zs::tuple<Tn...> &loc,
                                              index_sequence<Is...>) const noexcept {
      return coord_type{weightGradient_impl<Is>(loc, index_sequence<Is...>{})...};
    }
  };

}  // namespace zs