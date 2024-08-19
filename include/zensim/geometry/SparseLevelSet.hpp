#pragma once
#include <utility>

#include "Structure.hpp"
#include "zensim/container/Bht.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/geometry/LevelSetInterface.h"
#include "zensim/math/Vec.h"
#include "zensim/math/curve/InterpolationKernel.hpp"
#include "zensim/math/matrix/Transform.hpp"
#include "zensim/types/Property.h"
#include "zensim/zpc_tpls/fmt/color.h"

namespace zs {

  template <typename LsvT, kernel_e kt, int drv_order> struct LevelSetArena;

  template <int dim_ = 3, grid_e category_ = grid_e::collocated> struct SparseLevelSet {
    static constexpr int dim = dim_;
    static constexpr int side_length = 8;
    static constexpr auto category = category_;
    using value_type = f32;
    using allocator_type = ZSPmrAllocator<>;
    using index_type = i32;
    using grid_t = Grid<value_type, dim, side_length, category>;
    using size_type = typename grid_t::size_type;
    // using table_t = HashTable<index_type, dim, size_type>;
    using table_t = bht<index_type, dim, int, 16>;

    using IV = typename table_t::key_type;
    using TV = vec<value_type, dim>;
    using TM = vec<value_type, dim, dim>;
    using Affine = vec<value_type, dim + 1, dim + 1>;

    using coord_index_type = typename grid_t::coord_index_type;
    using channel_counter_type = typename grid_t::channel_counter_type;
    // using cell_index_type = zs::make_unsigned_t<decltype(SideLength)>;
    using cell_index_type = typename grid_t::cell_index_type;
    static constexpr auto block_size = grid_traits<grid_t>::block_size;

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
    /// do not use grid's numBlocks()
    constexpr decltype(auto) numBlocks() const noexcept { return _table.size(); }
    constexpr decltype(auto) numReservedBlocks() const noexcept {
      return _grid.numReservedBlocks();
    }
    constexpr channel_counter_type numChannels() const noexcept { return _grid.numChannels(); }

    SparseLevelSet(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
                   value_type dx, size_type count = 0)
        : _backgroundValue{(value_type)0},
          _backgroundVecValue{TV::zeros()},
          _table{allocator, count},
          _grid{allocator, channelTags, dx, count},
          _min{TV::constant(detail::deduce_numeric_max<value_type>())},
          _max{TV::constant(detail::deduce_numeric_lowest<value_type>())},
          _i2wSinv{TM::identity() / dx},
          _i2wRinv{TM::identity()},
          _i2wT{TV::zeros()},  // origin offset
          _i2wShat{TM::identity()},
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
    void append_channels(Policy &&policy, const std::vector<PropertyTag> &tags,
                         const source_location &loc = source_location::current()) {
      _grid.append_channels(FWD(policy), tags, loc);
    }
    template <typename Policy> void reset(Policy &&policy, value_type val) {
      _grid.reset(FWD(policy), val);
    }

    bool hasProperty(const SmallString &str) const noexcept { return _grid.hasProperty(str); }
    constexpr channel_counter_type getPropertySize(const SmallString &str) const {
      return _grid.getPropertySize(str);
    }
    constexpr channel_counter_type getPropertyOffset(const SmallString &str) const {
      return _grid.getPropertyOffset(str);
    }
    constexpr PropertyTag getPropertyTag(size_t i = 0) const { return _grid.getPropertyTag(i); }
    constexpr const auto &getPropertyTags() const { return _grid.getPropertyTags(); }

    void printTransformation(std::string_view msg = {}) const {
      auto r = _i2wRinv.transpose();
      auto [mi, ma] = proxy<execspace_e::host>(*this).getBoundingBox();
      fmt::print(fg(fmt::color::aquamarine),
                 "[ls<dim {}, cate {}> {}] dx: {}. ibox: [{}, {}, {} ~ {}, {}, {}]; wbox: [{}, {}, "
                 "{} ~ {}, {}, {}]. trans: {}, {}, "
                 "{}. \nrotation [{}, {}, {}; {}, {}, {}; {}, {}, {}].\n",
                 dim, category, msg, (value_type)1 / _i2wSinv(0), _min[0], _min[1], _min[2],
                 _max[0], _max[1], _max[2], mi[0], mi[1], mi[2], ma[0], ma[1], ma[2], _i2wT(0),
                 _i2wT(1), _i2wT(2), r(0, 0), r(0, 1), r(0, 2), r(1, 0), r(1, 1), r(1, 2), r(2, 0),
                 r(2, 1), r(2, 2));
    }
    template <typename VecTM,
              enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == dim + 1,
                            VecTM::template range_t<1>::value == dim + 1,
                            is_floating_point_v<typename VecTM::value_type>>
              = 0>
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
                                           VecT::template range_t<1>::value == dim>
                             = 0>
    void rotate(const VecInterface<VecT> &r) noexcept {
      _i2wRhat = _i2wRhat * r;
      _i2wRinv = r.transpose() * _i2wRinv;
    }
    template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == dim,
                                           VecT::template range_t<1>::value == dim>
                             = 0>
    void scale(const VecInterface<VecT> &s) {
      _i2wShat = _i2wShat * s;
      _i2wSinv = inverse(s) * _i2wSinv;
      _grid.dx *= s(0, 0);
    }
    void scale(const value_type s) { scale(s * TM::identity()); }

    value_type _backgroundValue{0};
    TV _backgroundVecValue{TV::zeros()};
    table_t _table{};
    grid_t _grid{};
    TV _min{TV::constant(detail::deduce_numeric_max<value_type>())},
        _max{TV::constant(detail::deduce_numeric_lowest<value_type>())};
    // initial index-to-world affine transformation
    TM _i2wSinv{TM::identity()}, _i2wRinv{TM::identity()};
    TV _i2wT{TV::zeros()};
    // additional
    TM _i2wShat{TM::identity()}, _i2wRhat{TM::identity()};
  };

  template <typename T, typename = void> struct is_spls : false_type {};
  template <int dim, grid_e category> struct is_spls<SparseLevelSet<dim, category>> : true_type {};
  template <typename T> constexpr bool is_spls_v = is_spls<T>::value;

  using GeneralSparseLevelSet
      = variant<SparseLevelSet<3, grid_e::collocated>, SparseLevelSet<2, grid_e::collocated>>;

  template <execspace_e, typename SparseLevelSetT, typename = void> struct SparseLevelSetView;

  template <execspace_e Space, typename SparseLevelSetT>
  struct SparseLevelSetView<Space, SparseLevelSetT>
      : LevelSetInterface<SparseLevelSetView<Space, SparseLevelSetT>> {
    static constexpr bool is_const_structure = is_const_v<SparseLevelSetT>;
    static constexpr auto space = Space;
    using ls_t = remove_const_t<SparseLevelSetT>;
    using value_type = typename ls_t::value_type;
    using size_type = typename ls_t::size_type;
    using index_type = typename ls_t::index_type;
    using table_t = typename ls_t::table_t;
    using table_view_t = RM_REF_T(
        proxy<Space>(declval<conditional_t<is_const_structure, const table_t &, table_t &>>()));
    using grid_t = typename ls_t::grid_t;
    using grid_view_t = RM_REF_T(
        proxy<Space>({}, declval<conditional_t<is_const_structure, const grid_t &, grid_t &>>()));

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

    template <typename Val, size_t... Is>
    static constexpr auto arena_type_impl(index_sequence<Is...>) {
      return vec<Val, (Is + 1 > 0 ? 2 : 2)...>{};
    }
    template <typename Val, int d> static constexpr auto arena_type() {
      return arena_type_impl<Val>(make_index_sequence<d>{});
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

    constexpr SparseLevelSetView(table_view_t tablev, grid_view_t gridv)
        : _table{tablev},
          _grid{gridv},
          _backgroundValue{0},
          _backgroundVecValue{TV::zeros()},
          _min{TV::zeros()},
          _max{TV::zeros()},
          _i2wT{TV::zeros()},
          _i2wRinv{TM::identity()},
          _i2wSinv{TM::identity() / gridv.dx},
          _i2wRhat{TM::identity()},
          _i2wShat{TM::identity()} {}

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

    constexpr auto do_getBoundingBox() const noexcept {
      auto mi = TV::constant(detail::deduce_numeric_max<value_type>());
      auto ma = TV::constant(detail::deduce_numeric_lowest<value_type>());
      auto length = _max - _min;
      for (auto loc : ndrange<dim>(2)) {
        auto coord = _min + make_vec<value_type>(loc) * length;
        auto pos = indexToWorld(coord);
        for (int d = 0; d != dim; ++d) {
          mi[d] = pos[d] < mi[d] ? pos[d] : mi[d];
          ma[d] = pos[d] > ma[d] ? pos[d] : ma[d];
        }
      }
      return zs::make_tuple(mi, ma);
    }

    /// coordinate transformation
    /// world space to index space
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto worldToIndex(const VecInterface<VecT> &x) const noexcept {
      // world-to-view: minus trans, div rotation, div scale
      if constexpr (category == grid_e::cellcentered)
        return (x - _i2wT) * _i2wRinv * _i2wSinv - (value_type)0.5;
      else
        return (x - _i2wT) * _i2wRinv * _i2wSinv;
    }
    /// index space to world space
    // cell-corresponding positions
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           is_integral_v<typename VecT::value_type>>
                             = 0>
    constexpr auto cellToIndex(const VecInterface<VecT> &X) const noexcept {
      if constexpr (category == grid_e::cellcentered)
        return (X + (value_type)0.5);
      else
        return X.template cast<value_type>();
    }
    template <typename VecT, auto cate = category,
              enable_if_all<cate == grid_e::staggered, VecT::dim == 1, VecT::extent == dim,
                            is_integral_v<typename VecT::value_type>>
              = 0>
    constexpr auto cellToIndex(const VecInterface<VecT> &X, int f) const noexcept {
      using RetT = typename VecT::template variant_vec<value_type, typename VecT::extents>;
      return RetT::init([&X, f](int d) {
        return d != f ? (value_type)X[d] + (value_type)0.5 : (value_type)X[d];
      });
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto indexToCell(const VecInterface<VecT> &x) const noexcept {
      using T = decltype(lower_trunc((value_type)x[0]));
      using RetT = typename VecT::template variant_vec<T, typename VecT::extents>;
      if constexpr (category == grid_e::cellcentered)
        return RetT::init([&x](int d) { return lower_trunc((value_type)x[d]); });
      else
        return RetT::init([&x](int d) { return lower_trunc((value_type)x[d] + (value_type)0.5); });
    }
    template <typename VecT, auto cate = category,
              enable_if_all<cate == grid_e::staggered, VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto indexToCell(const VecInterface<VecT> &x, int f) const noexcept {
      using T = decltype(lower_trunc((value_type)x[0]));
      using RetT = typename VecT::template variant_vec<T, typename VecT::extents>;
      return RetT::init([&x, f](int d) {
        return d != f ? lower_trunc((value_type)x[d])
                      : lower_trunc((value_type)x[d] + (value_type)0.5);
      });
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto worldToCell(const VecInterface<VecT> &x) const noexcept {
      return indexToCell((x - _i2wT) * _i2wRinv * _i2wSinv);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto indexToWorld(const VecInterface<VecT> &X) const noexcept {
      // view-to-index: scale, rotate, trans
      if constexpr (category == grid_e::cellcentered)
        return (X + (value_type)0.5) * inverse(_i2wSinv) * inverse(_i2wRinv) + _i2wT;
      else
        return X * inverse(_i2wSinv) * _i2wRinv.transpose() + _i2wT;
    }
    constexpr auto indexToWorld(size_type bno, cell_index_type cno) const noexcept {
      return indexToWorld(_table._activeKeys[bno] + grid_view_t::cellid_to_coord(cno));
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto indexToWorld(size_type bno, const VecInterface<VecT> &cid) const noexcept {
      return indexToWorld(_table._activeKeys[bno] + cid);
    }
    // face center position (staggered grid only)
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            is_integral_v<typename VecT::value_type>, cate == grid_e::staggered>
              = 0>
    constexpr auto indexToWorld(const VecInterface<VecT> &coord, int orientation) const noexcept {
      auto offset = TV::constant((value_type)0.5);
      offset(orientation) = (value_type)0;
      return indexToWorld(coord + offset);
    }
    template <auto cate = category, enable_if_all<cate == grid_e::staggered> = 0>
    constexpr auto indexToWorld(size_type bno, cell_index_type cno,
                                int orientation) const noexcept {
      return indexToWorld(_table._activeKeys[bno] + grid_view_t::cellid_to_coord(cno), orientation);
    }
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            is_integral_v<typename VecT::value_type>, cate == grid_e::staggered>
              = 0>
    constexpr auto indexToWorld(size_type bno, const VecInterface<VecT> &cid,
                                int orientation) const noexcept {
      return indexToWorld(_table._activeKeys[bno] + cid, orientation);
    }

    /// helper functions
    template <typename VecTI, enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                                            is_integral_v<typename VecTI::index_type>>
                              = 0>
    constexpr auto decompose_coord(const VecInterface<VecTI> &indexCoord) const noexcept {
      auto cellid = indexCoord & (side_length - 1);
      auto blockid = indexCoord - cellid;
      return make_tuple(_table.query(blockid), grid_view_t::coord_to_cellid(cellid));
    }
    template <typename VecTI, auto cate = category,
              enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                            is_integral_v<typename VecTI::index_type>, cate == grid_e::staggered>
              = 0>
    constexpr value_type value_or(channel_counter_type chn, const VecInterface<VecTI> &indexCoord,
                                  int orientation, value_type defaultVal) const noexcept {
      /// 0, ..., dim-1: within cell
      /// dim, ..., dim+dim-1: neighbor cell
      auto coord = indexCoord.clone();
      if (auto f = orientation % (dim + dim); f >= dim) ++coord[f - dim];
      auto [bno, cno] = decompose_coord(coord);
      return bno == table_t::sentinel_v ? defaultVal : _grid(chn, (size_type)bno, cno);
    }
    constexpr value_type value_or(channel_counter_type chn, typename table_t::index_type blockno,
                                  cell_index_type cellno, value_type defaultVal) const noexcept {
      return blockno == table_t::sentinel_v ? defaultVal : _grid(chn, (size_type)blockno, cellno);
    }
    template <typename VecTI, enable_if_all<VecTI::dim == 1, VecTI::extent == dim,
                                            is_integral_v<typename VecTI::index_type>>
                              = 0>
    constexpr auto value_or(channel_counter_type chn, const VecInterface<VecTI> &indexCoord,
                            value_type defaultVal) const noexcept {
      auto [bno, cno] = decompose_coord(indexCoord);
      return value_or(chn, bno, cno, defaultVal);
    }

    template <kernel_e kt = kernel_e::linear, int order = 0, typename VecT, auto cate = category,
              bool WS = true, enable_if_t<cate != grid_e::staggered> = 0>
    constexpr auto arena(const VecInterface<VecT> &x, wrapv<kt> ktFlag = {},
                         wrapv<order> orderFlag = {}, wrapv<WS> tag = {}) noexcept {
      return LevelSetArena{ktFlag, orderFlag, *this, x, tag};
    }
    template <kernel_e kt = kernel_e::linear, int order = 0, typename VecT, auto cate = category,
              bool WS = true, enable_if_t<cate != grid_e::staggered> = 0>
    constexpr auto arena(const VecInterface<VecT> &x, wrapv<kt> ktFlag = {},
                         wrapv<order> orderFlag = {}, wrapv<WS> tag = {}) const noexcept {
      return LevelSetArena{ktFlag, orderFlag, *this, x, tag};
    }

    template <kernel_e kt = kernel_e::linear, int order = 0, typename VecT, auto cate = category,
              bool WS = true, enable_if_t<cate == grid_e::staggered> = 0>
    constexpr auto arena(const VecInterface<VecT> &x, int f, wrapv<kt> ktFlag = {},
                         wrapv<order> orderFlag = {}, wrapv<WS> tag = {}) noexcept {
      return LevelSetArena{ktFlag, orderFlag, *this, x, f, tag};
    }
    template <kernel_e kt = kernel_e::linear, int order = 0, typename VecT, auto cate = category,
              bool WS = true, enable_if_t<cate == grid_e::staggered> = 0>
    constexpr auto arena(const VecInterface<VecT> &x, int f, wrapv<kt> ktFlag = {},
                         wrapv<order> orderFlag = {}, wrapv<WS> tag = {}) const noexcept {
      return LevelSetArena{ktFlag, orderFlag, *this, x, f, tag};
    }

    ///
    /// tensor sampling
    ///
    // cell-wise staggered grid sampling
    template <
        typename VecT, kernel_e kt = kernel_e::linear, auto cate = category,
        enable_if_all<VecT::dim == 1, VecT::extent == dim,
                      is_floating_point_v<typename VecT::value_type>, cate == grid_e::staggered>
        = 0>
    constexpr TV ipack(const SmallString &propName, const VecInterface<VecT> &X,
                       const value_type defaultVal, wrapv<kt> = {}) const noexcept {
      static_assert(kt == kernel_e::linear, "only linear interop implemented so far");
      if (!_grid.hasProperty(propName)) return TV::constant(defaultVal);
      const auto propOffset = _grid.propertyOffset(propName);
#if 0
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
#else
      TV ret{};
      for (int d = 0; d != dim; ++d) {
        auto pad = arena(X, d, kernel_linear_c, wrapv<0>{}, false_c);
        ret(d) = pad.isample(propOffset + d, defaultVal);
      }
#endif
      return ret;
    }
    /// this serves as an optimized impl (access 4 instead of 8)
    template <typename VecT, kernel_e kt = kernel_e::linear, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            is_integral_v<typename VecT::value_type>, cate == grid_e::staggered>
              = 0>
    constexpr TV ipack(const SmallString &propName, const VecInterface<VecT> &coord, int f,
                       const value_type defaultVal, wrapv<kt> = {}) const noexcept {
      static_assert(kt == kernel_e::linear, "only linear interop implemented so far");
      if (!_grid.hasProperty(propName)) return TV::constant(defaultVal);
      const auto propOffset = _grid.propertyOffset(propName);
      /// world to local
      auto [blockno, cellno] = decompose_coord(coord);
      TV ret{};
      for (int d = 0; d != dim; ++d) {
        if (d == f)
          ret(d) = value_or(propOffset + d, blockno, cellno, defaultVal);
        else
          ret(d) = (value_or(propOffset + d, blockno, cellno, defaultVal)
                    + value_or(propOffset + d,
                               coord + IV::init([d](int i) noexcept { return i == d ? 1 : 0; }),
                               defaultVal)
                    + value_or(propOffset + d,
                               coord + IV::init([f](int i) noexcept { return i == f ? -1 : 0; }),
                               defaultVal)
                    + value_or(propOffset + d, coord + IV::init([d, f](int i) noexcept {
                                                 return i == d ? 1 : (i == f ? -1 : 0);
                                               }),
                               defaultVal))
                   * (value_type)0.25;
      }
      return ret;
    }
    template <typename VecT, kernel_e kt = kernel_e::linear, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate == grid_e::staggered> = 0>
    constexpr TV wpack(const SmallString &propName, const VecInterface<VecT> &x,
                       const value_type defaultVal, wrapv<kt> ktTag = {}) const noexcept {
      return ipack(propName, worldToIndex(x), defaultVal, ktTag);
    }
    template <auto... Ns, typename VecT, kernel_e kt = kernel_e::linear,
              typename RetT = typename VecT::template variant_vec<
                  typename VecT::value_type, integer_sequence<typename VecT::index_type, Ns...>>,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto ipack(channel_counter_type chn, const VecInterface<VecT> &X,
                         const value_type defaultVal, wrapv<kt> = {}) const noexcept {
      static_assert(kt == kernel_e::linear, "only linear interop implemented so far");
      constexpr auto extent = RetT::extent;
      RetT ret{};
#if 0
      IV loc{};
      for (int d = 0; d != dim; ++d) loc(d) = zs::floor(X(d));
      auto diff = X - loc;
#endif
      if constexpr (category == grid_e::staggered) {
#if 0
        auto [blockno, cellno] = decompose_coord(loc);
        for (channel_counter_type d = 0; d != extent; ++d) {
          const auto orientation = (chn + d) % dim;
          auto neighborLoc = loc;
          ++neighborLoc(orientation);
          ret.val(d)
              = linear_interop(diff(orientation), value_or(chn + d, blockno, cellno, defaultVal),
                               value_or(chn + d, neighborLoc, defaultVal));
        }
#else
        for (channel_counter_type d = 0; d != extent; ++d) {
          const auto orientation = (chn + d) % dim;
          auto pad = arena(X, orientation, kernel_linear_c, wrapv<0>{}, false_c);
          ret.val(d) = pad.isample(chn + d, defaultVal);
        }
#endif
      } else {
#if 0
        Arena<RetT> arena{};
        for (auto &&offset : ndrange<dim>(2))
          arena.val(offset) = value_or(chn, loc + make_vec<index_type>(offset), defaultVal);
        ret = xlerp<0>(diff, arena);
#else
        auto pad = arena(X, kernel_linear_c, wrapv<0>{}, false_c);
        for (channel_counter_type d = 0; d != extent; ++d)
          ret.val(d) = pad.isample(chn + d, defaultVal);
#endif
      }
      return ret;
    }
    template <auto... Ns, typename VecT, kernel_e kt = kernel_e::linear, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate != grid_e::staggered> = 0>
    constexpr auto ipack(const SmallString &propName, const VecInterface<VecT> &X,
                         const value_type defaultVal, wrapv<kt> ktTag = {}) const noexcept {
      using RetT = decltype(ipack<Ns...>(0, X, defaultVal, ktTag));
      if (!_grid.hasProperty(propName)) return RetT::constant(defaultVal);
      const auto propOffset = _grid.propertyOffset(propName);
      return ipack<Ns...>(propOffset, X, defaultVal, ktTag);
    }
    template <auto... Ns, typename VecT, kernel_e kt = kernel_e::linear, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate != grid_e::staggered> = 0>
    constexpr auto wpack(const SmallString &propName, const VecInterface<VecT> &x,
                         const value_type defaultVal, wrapv<kt> ktTag = {}) const noexcept {
      using RetT = decltype(ipack<Ns...>(0, worldToIndex(x), defaultVal, ktTag));
      if (!_grid.hasProperty(propName)) return RetT::constant(defaultVal);
      const auto propOffset = _grid.propertyOffset(propName);
      return ipack<Ns...>(propOffset, worldToIndex(x), defaultVal, ktTag);
    }

    ///
    /// scalar sampling
    ///
    template <typename VecT, kernel_e kt = kernel_e::linear,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type isample(channel_counter_type chn, const VecInterface<VecT> &X,
                                 const value_type defaultVal, wrapv<kt> = {}) const noexcept {
      static_assert(kt == kernel_e::linear, "only linear interop implemented so far");
      /// world to local
      IV loc{};
      for (int d = 0; d != dim; ++d) loc(d) = zs::floor(X(d));
      if constexpr (category == grid_e::staggered) {
#if 0
        const auto orientation = chn % dim;
        value_type diff = X(orientation) - loc(orientation);
        auto neighborLoc = loc;
        ++neighborLoc(orientation);
        return linear_interop(diff, value_or(chn, loc, defaultVal),
                              value_or(chn, neighborLoc, defaultVal));
#else
        const auto orientation = chn % dim;
        auto pad = arena(X, orientation, kernel_linear_c, wrapv<0>{}, false_c);
        return pad.isample(chn, defaultVal);
#endif
      } else {
#if 0
        Arena<value_type> arena{};
        TV diff = X - loc;
        for (auto &&offset : ndrange<dim>(2))
          arena.val(offset) = value_or(chn, loc + make_vec<index_type>(offset), defaultVal);
        return xlerp<0>(diff, arena);
#else
        auto pad = arena(X, kernel_linear_c, wrapv<0>{}, false_c);
        return pad.isample(chn, defaultVal);
#endif
      }
    }
    template <typename VecT, kernel_e kt = kernel_e::linear,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type isample(const SmallString &propName, channel_counter_type chn,
                                 const VecInterface<VecT> &X, const value_type defaultVal,
                                 wrapv<kt> ktTag = {}) const noexcept {
      {  // channel index check
        auto pid = _grid.propertyIndex(propName);
        if (pid == _grid.numProperties())
          return defaultVal;
        else if (chn >= _grid.getPropertySizes()[pid])
          return defaultVal;
      }
      const auto propOffset = _grid.propertyOffset(propName);
      return isample(propOffset + chn, X, defaultVal, ktTag);
    }
    template <typename VecT, kernel_e kt = kernel_e::linear,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type wsample(const SmallString &propName, channel_counter_type chn,
                                 const VecInterface<VecT> &x, const value_type defaultVal,
                                 wrapv<kt> ktTag = {}) const noexcept {
      {  // channel index check
        auto pid = _grid.propertyIndex(propName);
        if (pid == _grid.numProperties())
          return defaultVal;
        else if (chn >= _grid.getPropertySizes()[pid])
          return defaultVal;
      }
      const auto propOffset = _grid.propertyOffset(propName);
      return isample(propOffset + chn, worldToIndex(x), defaultVal, ktTag);
    }
    template <typename VecT, kernel_e kt = kernel_e::linear,
              enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type wsample(channel_counter_type chn, const VecInterface<VecT> &x,
                                 const value_type defaultVal, wrapv<kt> ktTag = {}) const noexcept {
      // channel index check
      if (chn >= _grid.numChannels()) return defaultVal;
      return isample(chn, worldToIndex(x), defaultVal, ktTag);
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
      if constexpr (category == grid_e::staggered)
        return wpack("v", x, 0) * _i2wShat * _i2wRhat;
      else
        return wpack<dim>("v", x, 0) * _i2wShat * _i2wRhat;
    }

    table_view_t _table{};
    grid_view_t _grid{};
    T _backgroundValue{detail::deduce_numeric_max<T>()};
    TV _backgroundVecValue{TV::constant(detail::deduce_numeric_max<T>())};
    TV _min{TV::constant(detail::deduce_numeric_max<T>())},
        _max{TV::constant(detail::deduce_numeric_lowest<T>())};

    TV _i2wT{TV::zeros()};
    TM _i2wRinv{TM::identity()}, _i2wSinv{TM::identity()};
    TM _i2wRhat{TM::identity()}, _i2wShat{TM::identity()};
  };

  // directly
  template <execspace_e ExecSpace, typename HashTableT, typename GridT>
  decltype(auto) proxy(HashTableView<ExecSpace, HashTableT> tablev,
                       GridView<ExecSpace, GridT, true, false> gridv) {
    constexpr int dim = RM_CVREF_T(tablev)::dim;
    constexpr auto category = RM_CVREF_T(gridv)::category;
    return SparseLevelSetView<ExecSpace, SparseLevelSet<dim, category>>{tablev, gridv};
  }
  template <execspace_e ExecSpace, int dim, grid_e category>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       SparseLevelSet<dim, category> &levelset) {
    return SparseLevelSetView<ExecSpace, SparseLevelSet<dim, category>>{levelset};
  }
  template <execspace_e ExecSpace, int dim, grid_e category>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       const SparseLevelSet<dim, category> &levelset) {
    return SparseLevelSetView<ExecSpace, const SparseLevelSet<dim, category>>{levelset};
  }

  template <execspace_e ExecSpace, int dim, grid_e category>
  decltype(auto) proxy(SparseLevelSet<dim, category> &levelset) {
    return SparseLevelSetView<ExecSpace, SparseLevelSet<dim, category>>{levelset};
  }
  template <execspace_e ExecSpace, int dim, grid_e category>
  decltype(auto) proxy(const SparseLevelSet<dim, category> &levelset) {
    return SparseLevelSetView<ExecSpace, const SparseLevelSet<dim, category>>{levelset};
  }

  template <typename LsvT, kernel_e kt_ = kernel_e::quadratic, int drv_order = 0>
  struct LevelSetArena {
    using lsv_t = LsvT;
    using value_type = typename lsv_t::value_type;
    using index_type = typename lsv_t::index_type;

    static_assert(is_signed_v<index_type>, "index_type should be a signed integer.");
    static constexpr grid_e category = lsv_t::category;
    static constexpr int dim = lsv_t::dim;
    static constexpr kernel_e kt = kt_;
    static constexpr int width = (kt == kernel_e::linear ? 2 : (kt == kernel_e::quadratic ? 3 : 4));
    static constexpr int deriv_order = drv_order;

    using TV = typename lsv_t::TV;
    using TWM = vec<value_type, lsv_t::dim, width>;
    using IV = typename lsv_t::IV;

    using coord_index_type = typename lsv_t::coord_index_type;
    using channel_counter_type = typename lsv_t::channel_counter_type;
    using cell_index_type = typename lsv_t::cell_index_type;

    static_assert(deriv_order >= 0 && deriv_order <= 2,
                  "weight derivative order should be a integer within [0, 2]");

    using WeightScratchPad
        = conditional_t<deriv_order == 0, tuple<TWM>,
                        conditional_t<deriv_order == 1, tuple<TWM, TWM>, tuple<TWM, TWM, TWM>>>;

    template <typename Val, size_t... Is>
    static constexpr auto arena_type_impl(index_sequence<Is...>) {
      return vec<Val, (Is + 1 > 0 ? width : width)...>{};
    }
    template <typename Val, int d> static constexpr auto arena_type() {
      return arena_type_impl<Val>(make_index_sequence<d>{});
    }
    template <typename Val> using Arena = RM_CVREF_T(arena_type<Val, lsv_t::dim>());

    /// constructors
    // index-space
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate != grid_e::staggered> = 0>
    constexpr LevelSetArena(false_type, lsv_t &lsv, const VecInterface<VecT> &X) noexcept
        : lsPtr{&lsv}, weights{}, iLocalPos{}, iCorner{} {
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
    constexpr LevelSetArena(false_type, lsv_t &lsv, const VecInterface<VecT> &X, int f) noexcept
        : lsPtr{&lsv}, weights{}, iLocalPos{}, iCorner{} {
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
    constexpr LevelSetArena(true_type, lsv_t &lsv, const VecInterface<VecT> &x) noexcept
        : LevelSetArena{false_c, lsv, lsv.worldToIndex(x)} {}
    template <typename VecT, auto cate = category,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate == grid_e::staggered> = 0>
    constexpr LevelSetArena(true_type, lsv_t &lsv, const VecInterface<VecT> &x, int f) noexcept
        : LevelSetArena{false_c, lsv, lsv.worldToIndex(x), f} {}
    /// for CTAD
    template <typename VecT, auto cate = category, bool WS = true,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate == grid_e::staggered> = 0>
    constexpr LevelSetArena(wrapv<kt>, wrapv<deriv_order>, lsv_t &lsv, const VecInterface<VecT> &x,
                            int f, wrapv<WS> tag = {}) noexcept
        : LevelSetArena{tag, lsv, x, f} {}
    template <typename VecT, auto cate = category, bool WS = true,
              enable_if_all<VecT::dim == 1, VecT::extent == dim, cate != grid_e::staggered> = 0>
    constexpr LevelSetArena(wrapv<kt>, wrapv<deriv_order>, lsv_t &lsv, const VecInterface<VecT> &x,
                            wrapv<WS> tag = {}) noexcept
        : LevelSetArena{tag, lsv, x} {}

    /// scalar arena
    constexpr Arena<value_type> arena(typename lsv_t::channel_counter_type chn,
                                      typename lsv_t::value_type defaultVal = 0) const noexcept {
      // ensure that chn's orientation is aligned with initialization if within a staggered grid
      Arena<value_type> pad{};
      for (auto offset : ndrange<dim>(width)) {
        pad.val(offset) = lsPtr->value_or(
            chn, iCorner + make_vec<typename lsv_t::coord_index_type>(offset), defaultVal);
      }
      return pad;
    }
    constexpr Arena<value_type> arena(const SmallString &propName,
                                      typename lsv_t::channel_counter_type chn = 0,
                                      typename lsv_t::value_type defaultVal = 0) const noexcept {
      return arena(lsPtr->_grid.propertyOffset(propName) + chn, defaultVal);
    }

    /// minimum
    constexpr value_type minimum(typename lsv_t::channel_counter_type chn = 0) const noexcept {
      auto pad = arena(chn, detail::deduce_numeric_max<value_type>());
      value_type ret = detail::deduce_numeric_max<value_type>();
      for (auto offset : ndrange<dim>(width))
        if (const auto &v = pad.val(offset); v < ret) ret = v;
      return ret;
    }
    constexpr value_type minimum(const SmallString &propName,
                                 typename lsv_t::channel_counter_type chn = 0) const noexcept {
      return minimum(lsPtr->_grid.propertyOffset(propName) + chn);
    }

    /// maximum
    constexpr value_type maximum(typename lsv_t::channel_counter_type chn = 0) const noexcept {
      auto pad = arena(chn, detail::deduce_numeric_lowest<value_type>());
      value_type ret = detail::deduce_numeric_lowest<value_type>();
      for (auto offset : ndrange<dim>(width))
        if (const auto &v = pad.val(offset); v > ret) ret = v;
      return ret;
    }
    constexpr value_type maximum(const SmallString &propName,
                                 typename lsv_t::channel_counter_type chn = 0) const noexcept {
      return maximum(lsPtr->_grid.propertyOffset(propName) + chn);
    }

    /// isample
    constexpr value_type isample(typename lsv_t::channel_counter_type chn,
                                 typename lsv_t::value_type defaultVal) const noexcept {
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
                                 typename lsv_t::channel_counter_type chn,
                                 typename lsv_t::value_type defaultVal) const noexcept {
      return isample(lsPtr->_grid.propertyOffset(propName) + chn, defaultVal);
    }

    /// weight
    template <typename... Tn>
    constexpr value_type weight(const std::tuple<Tn...> &loc) const noexcept {
      return weight_impl(loc, index_sequence_for<Tn...>{});
    }
    template <typename... Tn, enable_if_all<((!is_tuple_v<Tn> && !is_std_tuple<Tn>()) && ...
                                             && (sizeof...(Tn) == dim))>
                              = 0>
    constexpr auto weight(Tn &&...is) const noexcept {
      return weight(std::forward_as_tuple(FWD(is)...));
    }
    /// weight gradient
    template <zs::size_t I, typename... Tn, auto ord = deriv_order>
    constexpr enable_if_type<(ord > 0), value_type> weightGradient(
        const std::tuple<Tn...> &loc) const noexcept {
      return weightGradient_impl<I>(loc, index_sequence_for<Tn...>{});
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

    lsv_t *lsPtr{nullptr};
    WeightScratchPad weights{};
    TV iLocalPos{TV::zeros()};  // index-space local offset
    IV iCorner{IV::zeros()};    // index-space global coord

  protected:
    template <typename... Tn, size_t... Is,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim)> = 0>
    constexpr value_type weight_impl(const std::tuple<Tn...> &loc,
                                     index_sequence<Is...>) const noexcept {
      value_type ret{1};
      ((void)(ret *= get<0>(weights)(Is, std::get<Is>(loc))), ...);
      return ret;
    }
    template <zs::size_t I, typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr value_type weightGradient_impl(const std::tuple<Tn...> &loc,
                                             index_sequence<Is...>) const noexcept {
      value_type ret{1};
      ((void)(ret *= (I == Is ? get<1>(weights)(Is, std::get<Is>(loc))
                              : get<0>(weights)(Is, std::get<Is>(loc)))),
       ...);
      return ret;
    }
    template <typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr TV weightGradients_impl(const std::tuple<Tn...> &loc,
                                      index_sequence<Is...>) const noexcept {
      return TV{weightGradient_impl<Is>(loc, index_sequence<Is...>{})...};
    }
  };

  template <kernel_e kt = kernel_e::linear, int deriv_order = 0, bool WS = true, typename LsvT,
            typename VecT>
  constexpr auto make_levelset_arena(LsvT &lsv, const VecInterface<VecT> &x, wrapv<WS> tag = {}) {
    static_assert(LsvT::category != grid_e::staggered,
                  "this method is not for staggered levelset.");
    return LevelSetArena<LsvT, kt, deriv_order>{tag, lsv, x};
  }
  template <kernel_e kt = kernel_e::linear, int deriv_order = 0, bool WS = true, typename LsvT,
            typename VecT>
  constexpr auto make_levelset_arena(LsvT &lsv, const VecInterface<VecT> &x, int f,
                                     wrapv<WS> tag = {}) {
    static_assert(LsvT::category == grid_e::staggered,
                  "this method is for staggered levelset only.");
    return LevelSetArena<LsvT, kt, deriv_order>{tag, lsv, x, f};
  }

  template <kernel_e kt, int drv_order, typename LsvT, typename... Args>
  LevelSetArena(wrapv<kt>, wrapv<drv_order>, LsvT &,
                Args...) -> LevelSetArena<remove_reference_t<LsvT>, kt, drv_order>;

}  // namespace zs