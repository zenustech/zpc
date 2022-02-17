#pragma once
#include <utility>

#include "Structure.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/geometry/LevelSetInterface.h"
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/Transform.hpp"
#include "zensim/tpls/fmt/color.h"

namespace zs {

  template <int dim_ = 3, grid_e category_ = grid_e::collocated> struct SparseLevelSet {
    static constexpr int dim = dim_;
    static constexpr int side_length = 8;
    static constexpr auto category = category_;
    using value_type = f32;
    using allocator_type = ZSPmrAllocator<>;
    using index_type = i32;
    using IV = vec<index_type, dim>;
    using TV = vec<value_type, dim>;
    using TM = vec<value_type, dim, dim>;
    using Affine = vec<value_type, dim + 1, dim + 1>;
    using grid_t = Grid<value_type, dim, side_length, category>;
    using size_type = typename grid_t::size_type;
    using table_t = HashTable<index_type, dim, size_type>;

    using coord_index_type = typename grid_t::coord_index_type;
    using channel_counter_type = typename grid_t::channel_counter_type;
    // using cell_index_type = std::make_unsigned_t<decltype(SideLength)>;
    using cell_index_type = typename grid_t::cell_index_type;
    static constexpr auto block_size = grid_traits<grid_t>::block_size;

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
    /// do not use grid's numBlocks()
    constexpr decltype(auto) numBlocks() const noexcept { return _table.size(); }
    constexpr channel_counter_type numChannels() const noexcept { return _grid.numChannels(); }

    SparseLevelSet(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
                   value_type dx, size_type count = 0)
        : _backgroundValue{(value_type)0},
          _backgroundVecValue{TV::zeros()},
          _table{allocator, count},
          _grid{allocator, channelTags, dx, count},
          _min{TV::uniform(limits<value_type>::max())},
          _max{TV::uniform(limits<value_type>::lowest())},
          _i2wSinv{TM::identity() / dx},
          _i2wRinv{TM::identity()},
          _i2wT{TV::zeros()},  // origin offset
          _i2wShat{TM::identity() * dx},
          _i2wRhat{TM::identity()} {}
    SparseLevelSet(const std::vector<PropertyTag> &channelTags, value_type dx, size_type count,
                   memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : SparseLevelSet{get_default_allocator(mre, devid), channelTags, dx, count} {}
    SparseLevelSet(channel_counter_type numChns, value_type dx, size_type count,
                   memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : SparseLevelSet{get_default_allocator(mre, devid), {{"unnamed", numChns}}, dx, count} {}
    SparseLevelSet(value_type dx = 1.f, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : SparseLevelSet{get_default_allocator(mre, devid), {{"sdf", 1}}, dx, 0} {}

    SparseLevelSet clone(const allocator_type &allocator) const {
      SparseLevelSet ret{};
      ret._backgroundValue = _backgroundValue;
      ret._backgroundVecValue = _backgroundVecValue;
      ret._table = _table.clone(allocator);
      ret._grid = _grid.clone(allocator);
      ret._min = _min;
      ret._max = _max;
      ret._i2wSinv = _i2wSinv;
      ret._i2wRinv = _i2wRinv;
      ret._i2wT = _i2wT;
      ret._i2wShat = _i2wShat;
      ret._i2wRhat = _i2wRhat;
      return ret;
    }
    SparseLevelSet clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    template <typename ExecPolicy> void resize(ExecPolicy &&policy, size_type numBlocks) {
      _table.resize(FWD(policy), numBlocks);
      _grid.resize(numBlocks);
    }
    template <typename Policy>
    void append_channels(Policy &&policy, const std::vector<PropertyTag> &tags) {
      _grid.append_channels(FWD(policy), tags);
    }
    template <typename Policy> void reset(Policy &&policy, value_type val) {
      _grid.reset(FWD(policy), val);
    }

    bool hasProperty(const SmallString &str) const noexcept { return _grid.hasProperty(str); }
    constexpr channel_counter_type getChannelSize(const SmallString &str) const {
      return _grid.getChannelSize(str);
    }
    constexpr channel_counter_type getChannelOffset(const SmallString &str) const {
      return _grid.getChannelOffset(str);
    }
    constexpr PropertyTag getPropertyTag(std::size_t i = 0) const {
      return _grid.getPropertyTag(i);
    }
    constexpr const auto &getPropertyTags() const { return _grid.getPropertyTags(); }

    template <typename VecTM,
              enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == dim + 1,
                            VecTM::template range_t<1>::value == dim + 1,
                            std::is_floating_point_v<typename VecTM::value_type>> = 0>
    void resetTransformation(const VecInterface<VecTM> &i2w) {
      math::decompose_transform(i2w, _i2wSinv, _i2wRinv, _i2wT, 0);
      _i2wSinv = inverse(_i2wSinv);
      _i2wRinv = _i2wRinv.transpose();  // equal to inverse
      _i2wShat = TM::identity();
      _i2wRhat = TM::identity();
    }
    auto getIndexToWorldTransformation() const {
      Affine ret{Affine::identity()};
      {
        auto S = inverse(_i2wSinv);
        for (int i = 0; i != dim; ++i)
          for (int j = 0; j != dim; ++j) ret(i, j) = S(i, j);
      }
      {
        Affine tmp{Affine::identity()};
        auto R = _i2wRinv.transpose();
        for (int i = 0; i != dim; ++i)
          for (int j = 0; j != dim; ++j) tmp(i, j) = R(i, j);
        ret = ret * tmp;
      }
      {
        Affine tmp{Affine::identity()};
        for (int j = 0; j != dim; ++j) tmp(dim, j) = _i2wT[j];
        ret = ret * tmp;
      }
      return ret;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    void translate(const VecInterface<VecT> &t) noexcept {
      _i2wT += t;
    }
    template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == dim,
                                           VecT::template range_t<1>::value == dim> = 0>
    void rotate(const VecInterface<VecT> &r) noexcept {
      _i2wRhat = _i2wRhat * r;
      _i2wRinv = r.transpose() * _i2wRinv;
    }
    template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == dim,
                                           VecT::template range_t<1>::value == dim> = 0>
    void scale(const VecInterface<VecT> &s) {
      _i2wShat = _i2wShat * s;
      _i2wSinv = inverse(s) * _i2wSinv;
    }
    void scale(const value_type s) { scale(s * TM::identity()); }

    value_type _backgroundValue{0};
    TV _backgroundVecValue{TV::zeros()};
    table_t _table{};
    grid_t _grid{};
    TV _min{TV::uniform(limits<value_type>::max())},
        _max{TV::uniform(limits<value_type>::lowest())};
    // initial index-to-world affine transformation
    TM _i2wSinv{TM::identity()}, _i2wRinv{TM::identity()};
    TV _i2wT{TV::zeros()};
    // additional
    TM _i2wShat{TM::identity()}, _i2wRhat{TM::identity()};
  };

  template <typename T, typename = void> struct is_spls : std::false_type {};
  template <int dim, grid_e category> struct is_spls<SparseLevelSet<dim, category>>
      : std::true_type {};
  template <typename T> constexpr bool is_spls_v = is_spls<T>::value;

  using GeneralSparseLevelSet
      = variant<SparseLevelSet<3, grid_e::collocated>, SparseLevelSet<2, grid_e::collocated>>;

  template <execspace_e, typename SparseLevelSetT, typename = void> struct SparseLevelSetView;

  template <execspace_e Space, typename SparseLevelSetT>
  struct SparseLevelSetView<Space, SparseLevelSetT>
      : LevelSetInterface<SparseLevelSetView<Space, SparseLevelSetT>>,
        RM_CVREF_T(
            proxy<Space>({}, std::declval<conditional_t<std::is_const_v<SparseLevelSetT>,
                                                        const typename SparseLevelSetT::grid_t &,
                                                        typename SparseLevelSetT::grid_t &>>())) {
    static constexpr bool is_const_structure = std::is_const_v<SparseLevelSetT>;
    static constexpr auto space = Space;
    using ls_t = std::remove_const_t<SparseLevelSetT>;
    using value_type = typename ls_t::value_type;
    using size_type = typename ls_t::size_type;
    using index_type = typename ls_t::index_type;
    using table_t = typename ls_t::table_t;
    using table_view_t = RM_CVREF_T(proxy<Space>(
        std::declval<conditional_t<is_const_structure, const table_t &, table_t &>>()));
    using grid_t = typename ls_t::grid_t;
    using grid_view_t = RM_CVREF_T(proxy<Space>(
        {}, std::declval<conditional_t<is_const_structure, const grid_t &, grid_t &>>()));

    using T = typename ls_t::value_type;
    using Ti = typename ls_t::index_type;
    using TV = typename ls_t::TV;
    using TM = typename ls_t::TM;
    using IV = typename ls_t::IV;
    using Affine = typename ls_t::Affine;

    using coord_index_type = typename ls_t::coord_index_type;
    using channel_counter_type = typename ls_t::channel_counter_type;
    using cell_index_type = typename ls_t::cell_index_type;

    static constexpr grid_e category = ls_t::category;
    static constexpr int dim = ls_t::dim;
    static constexpr auto side_length = ls_t::side_length;
    static constexpr auto block_size = ls_t::block_size;
    static_assert(grid_t::is_power_of_two, "block_size should be power of 2");

    template <typename Val, std::size_t... Is>
    static constexpr auto arena_type_impl(index_seq<Is...>) {
      return vec<Val, (Is + 1 > 0 ? 2 : 2)...>{};
    }
    template <typename Val, int d> static constexpr auto arena_type() {
      return arena_type_impl<Val>(std::make_index_sequence<d>{});
    }

    template <typename Val> using Arena = decltype(arena_type<Val, dim>());

    SparseLevelSetView() noexcept = default;
    ~SparseLevelSetView() noexcept = default;
    constexpr SparseLevelSetView(SparseLevelSetT &ls)
        : _table{proxy<Space>(ls._table)},
          _grid{proxy<Space>({}, ls._grid)},
          _backgroundValue{ls._backgroundValue},
          _backgroundVecValue{ls._backgroundVecValue},
          _min{ls._min},
          _max{ls._max},
          _i2wT{ls._i2wT},
          _i2wRinv{ls._i2wRinv},
          _i2wSinv{ls._i2wSinv},
          _i2wRhat{ls._i2wRhat},
          _i2wShat{ls._i2wShat} {}

    template <auto S = Space, enable_if_t<S == execspace_e::host> = 0> void print() {
      if constexpr (dim == 2) {
        auto blockCnt = *_table._cnt;
        using Ti = RM_CVREF_T(blockCnt);
        using IV = vec<Ti, dim>;
        for (Ti bno = 0; bno != blockCnt; ++bno) {
          auto blockCoord = _table._activeKeys[bno];
          fmt::print(fg(fmt::color::orange), "\nblock [{}] ({}, {})\n", bno, blockCoord[0],
                     blockCoord[1]);
          if (blockCoord[0] >= 0 || blockCoord[1] >= 0) {
            fmt::print("skip\n");
            continue;
          }

          auto block = _grid.block(bno);
          for (Ti cno = 0, ed = grid_t::block_space(); cno != ed; ++cno) {
            IV cellCoord{cno % side_length, side_length - 1 - cno / side_length};
            auto val = block("sdf", cellCoord);
            auto tag = block("tag", cellCoord);
            auto mask = block("mask", cellCoord);
            auto tagmask = block("tagmask", cellCoord);
            auto c = fg(fmt::color::white);
            if (mask == 0 || tagmask == 0) {
            } else {
              if (tag > 1)
                c = fg(fmt::color::dark_olive_green);
              else if (tag > 0)
                c = fg(fmt::color::light_sea_green);
              else
                c = fg(fmt::color::yellow_green);
            }
            // if (tag < 0) fmt::print("WTF at {} ({}, {})??\n", (int)bno, (int)cellCoord[0],
            // (int)cellCoord[1]);
            auto candi = fmt::format("{:.4f}", val);
            auto candi1 = fmt::format("{}", tag);
            fmt::print(c, "[{}{}({})] ", val < 0 ? "" : " ", mask ? candi : "------",
                       tagmask ? candi1 : " ");
            if (cno % side_length == side_length - 1) fmt::print("\n");
          }
          fmt::print("\n");
        }
      }
    }

    constexpr const SmallString *getPropertyNames() const noexcept {
      return _grid.getPropertyNames();
    }
    constexpr const channel_counter_type *getPropertyOffsets() const noexcept {
      return _grid.getPropertyOffsets();
    }
    constexpr const channel_counter_type *getPropertySizes() const noexcept {
      return _grid.getPropertySizes();
    }
    constexpr channel_counter_type numProperties() const noexcept { return _grid.numProperties(); }
    constexpr channel_counter_type propertyIndex(const SmallString &propName) const noexcept {
      return _grid.propertyIndex(propName);
    }
    constexpr channel_counter_type propertySize(const SmallString &propName) const noexcept {
      return _grid.propertySize(propName);
    }
    constexpr channel_counter_type propertyOffset(const SmallString &propName) const noexcept {
      return _grid.propertyOffset(propName);
    }
    constexpr bool hasProperty(const SmallString &propName) const noexcept {
      return _grid.hasProperty(propName);
    }
    constexpr auto numCells() const noexcept { return _grid.numCells(); }
    constexpr auto numBlocks() const noexcept { return _table.size(); }
    constexpr auto numChannels() const noexcept { return _grid.numChannels(); }

    constexpr auto do_getBoundingBox() const noexcept { return std::make_tuple(_min, _max); }

    /// coordinate transformation
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto worldToIndex(const VecInterface<VecT> &x) const noexcept {
      // world-to-view: minus trans, div rotation, div scale
      if constexpr (category == grid_e::cellcentered)
        return (x - _i2wT) * _i2wRinv * _i2wSinv - (value_type)0.5;
      else
        return (x - _i2wT) * _i2wRinv * _i2wSinv;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto indexToWorld(const VecInterface<VecT> &X) const noexcept {
      // view-to-index: scale, rotate, trans
      if constexpr (category == grid_e::cellcentered)
        return (X + (value_type)0.5) * _i2wShat * _i2wRhat + _i2wT;
      else
        return X * _i2wShat * _i2wRhat + _i2wT;
    }
    constexpr auto indexToWorld(size_type bno, cell_index_type cno) const noexcept {
      return indexToWorld(_table._activeKeys[bno] + grid_view_t::cellid_to_coord(cno));
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto indexToWorld(size_type bno, const VecInterface<VecT> &cid) const noexcept {
      return indexToWorld(_table._activeKeys[bno] + cid);
    }
    // face center position (staggered grid only)
    template <auto cate = category, enable_if_all<cate == grid_e::staggered> = 0>
    constexpr auto indexToWorld(size_type bno, cell_index_type cno,
                                int orientation) const noexcept {
      auto offset = TV::uniform((value_type)0.5);
      offset(orientation) = (value_type)0;
      return indexToWorld(_table._activeKeys[bno] + grid_view_t::cellid_to_coord(cno) + offset);
    }
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate == grid_e::staggered> = 0>
    constexpr auto indexToWorld(size_type bno, const VecInterface<VecT> &cid,
                                int orientation) const noexcept {
      auto offset = TV::uniform((value_type)0.5);
      offset(orientation) = (value_type)0;
      return indexToWorld(_table._activeKeys[bno] + cid + offset);
    }

    /// helper functions
    template <typename VecTI, enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                                            std::is_integral_v<typename VecTI::index_type>> = 0>
    constexpr auto decompose_coord(const VecInterface<VecTI> &indexCoord) const noexcept {
      auto cellid = indexCoord & (side_length - 1);
      auto blockid = indexCoord - cellid;
      return make_tuple(_table.query(blockid), grid_view_t::coord_to_cellid(cellid));
    }
    constexpr value_type value_or(channel_counter_type chn, typename table_t::value_t blockno,
                                  cell_index_type cellno, value_type defaultVal) const noexcept {
      return blockno < (typename table_t::value_t)0 ? defaultVal
                                                    : _grid(chn, (size_type)blockno, cellno);
    }

    /// sample
    // cell-wise staggered grid sampling
    template <typename VecT, kernel_e kt = kernel_e::linear, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            std::is_floating_point_v<typename VecT::value_type>,
                            cate == grid_e::staggered> = 0>
    constexpr TV isample(const SmallString &propName, const VecInterface<VecT> &X,
                         const value_type defaultVal, wrapv<kt> = {}) const noexcept {
      static_assert(kt == kernel_e::linear, "only linear interop implemented so far");
      if (!_grid.hasProperty(propName)) return TV::uniform(defaultVal);
      /// world to local
      const auto propOffset = _grid.propertyOffset(propName);
      IV loc{};
      for (int d = 0; d != dim; ++d) loc(d) = zs::floor(X(d));
      TV diff = X - loc;
      TV ret{};
      auto [blockno, cellno] = decompose_coord(loc);
      for (int d = 0; d != dim; ++d) {
        auto neighborLoc = loc;
        ++neighborLoc(d);
        auto [bno, cno] = decompose_coord(neighborLoc);
        ret(d) = linear_interop(diff(d), value_or(propOffset + d, blockno, cellno, defaultVal),
                                value_or(propOffset + d, bno, cno, defaultVal));
      }
      return ret;
    }
#if 0
    template <
        typename VecT, kernel_e kt = kernel_e::linear, auto cate = category,
        enable_if_all<VecT::dim == 1, VecT::extent == dim,
                      std::is_integral_v<typename VecT::value_type>, cate == grid_e::staggered> = 0>
    constexpr TV isample(const SmallString &propName, const VecInterface<VecT> &X, int orientation,
                         const value_type defaultVal, wrapv<kt> = {}) const noexcept {
      static_assert(kt == kernel_e::linear, "only linear interop implemented so far");
      if (!_grid.hasProperty(propName)) return TV::uniform(defaultVal);
      /// world to local
      const auto propOffset = _grid.propertyOffset(propName);
      ;
    }
#endif
    template <typename VecT, kernel_e kt = kernel_e::linear, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate == grid_e::staggered> = 0>
    constexpr TV wsample(const SmallString &propName, const VecInterface<VecT> &x,
                         const value_type defaultVal, wrapv<kt> ktTag = {}) const noexcept {
      return isample(propName, worldToIndex(x), defaultVal, ktTag);
    }
    template <typename VecT, kernel_e kt = kernel_e::linear,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type isample(const SmallString &propName, channel_counter_type chn,
                                 const VecInterface<VecT> &X, const value_type defaultVal,
                                 wrapv<kt> = {}) const noexcept {
      static_assert(kt == kernel_e::linear, "only linear interop implemented so far");
      {  // channel index check
        auto pid = _grid.propertyIndex(propName);
        if (pid == _grid.numProperties())
          return defaultVal;
        else if (chn >= _grid.getPropertySizes()[pid])
          return defaultVal;
      }
      /// world to local
      if constexpr (category == grid_e::staggered) {
        const auto orientation = chn % dim;
        const auto propOffset = _grid.propertyOffset(propName);
        IV loc{};
        value_type diff = X(orientation) - zs::floor(X(orientation));
        auto [blockno, cellno] = decompose_coord(loc);
        auto neighborLoc = loc;
        ++neighborLoc(orientation);
        auto [bno, cno] = decompose_coord(neighborLoc);
        return linear_interop(diff, value_or(propOffset + orientation, blockno, cellno, defaultVal),
                              value_or(propOffset + orientation, bno, cno, defaultVal));
      } else {
        const auto propOffset = _grid.propertyOffset(propName);
        Arena<value_type> arena{};
        IV loc{};
        for (int d = 0; d != dim; ++d) loc(d) = zs::floor(X(d));
        TV diff = X - loc;
        for (auto &&offset : ndrange<dim>(2)) {
          auto [bno, cno] = decompose_coord(loc + make_vec<index_type>(offset));
          arena.val(offset) = value_or(propOffset + chn, bno, cno, defaultVal);
        }
        return xlerp<0>(diff, arena);
      }
    }
    template <typename VecT, kernel_e kt = kernel_e::linear,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type wsample(const SmallString &propName, channel_counter_type chn,
                                 const VecInterface<VecT> &x, const value_type defaultVal,
                                 wrapv<kt> ktTag = {}) const noexcept {
      return isample(propName, chn, worldToIndex(x), defaultVal, ktTag);
    }

    /// levelset interface
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      return wsample("sdf", 0, x, _backgroundValue);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getNormal(const VecInterface<VecT> &x) const noexcept {
      typename VecT::template variant_vec<value_type, typename VecT::extents> diff{}, v1{}, v2{};
      // TV diff{}, v1{}, v2{};
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
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      if (!_grid.hasProperty("vel")) return TV::zeros();
      /// world to local
      using TV = typename VecT::template variant_vec<value_type, typename VecT::extents>;
      auto arena = Arena<TV>::uniform(_backgroundVecValue);
      IV loc{};
      TV X = worldToIndex(x);
      for (int d = 0; d < dim; ++d) loc(d) = zs::floor(X(d));
      auto diff = X - loc;
      for (auto &&offset : ndrange<dim>(2)) {
        auto coord = loc + make_vec<index_type>(offset);
        auto blockid = coord - (coord & (side_length - 1));
        auto blockno = _table.query(blockid);
        if (blockno != table_t::sentinel_v)
          arena.val(offset) = _grid.template pack<dim>("vel", blockno, coord - blockid);
      }
      return xlerp<0>(diff, arena);  //  * _i2wShat * _i2wRhat
      // don't be a smartass...
    }

    table_view_t _table{};
    grid_view_t _grid{};
    T _backgroundValue{limits<T>::max()};
    TV _backgroundVecValue{TV::uniform(limits<T>::max())};
    TV _min{TV::uniform(limits<T>::max())}, _max{TV::uniform(limits<T>::lowest())};

    TV _i2wT{TV::zeros()};
    TM _i2wRinv{TM::identity()}, _i2wSinv{TM::identity()};
    TM _i2wRhat{TM::identity()}, _i2wShat{TM::identity()};
  };

  template <execspace_e ExecSpace, int dim, grid_e category>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 SparseLevelSet<dim, category> &levelset) {
    return SparseLevelSetView<ExecSpace, SparseLevelSet<dim, category>>{levelset};
  }
  template <execspace_e ExecSpace, int dim, grid_e category>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 const SparseLevelSet<dim, category> &levelset) {
    return SparseLevelSetView<ExecSpace, const SparseLevelSet<dim, category>>{levelset};
  }

  template <execspace_e ExecSpace, int dim, grid_e category>
  constexpr decltype(auto) proxy(SparseLevelSet<dim, category> &levelset) {
    return SparseLevelSetView<ExecSpace, SparseLevelSet<dim, category>>{levelset};
  }
  template <execspace_e ExecSpace, int dim, grid_e category>
  constexpr decltype(auto) proxy(const SparseLevelSet<dim, category> &levelset) {
    return SparseLevelSetView<ExecSpace, const SparseLevelSet<dim, category>>{levelset};
  }

}  // namespace zs