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

  template <typename LsvT, kernel_e kt, int drv_order> struct LevelSetArena;

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

}  // namespace zs