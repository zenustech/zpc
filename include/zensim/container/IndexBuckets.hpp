#pragma once
#include <utility>

#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  template <int dim_ = 3, typename Index = i64, typename Tn = i32,
            grid_e category_ = grid_e::collocated, typename AllocatorT = ZSPmrAllocator<>>
  struct IndexBuckets {
    static_assert(is_integral_v<Index> && is_integral_v<Tn>,
                  "index and coord_index should be integrals");
    static constexpr int dim = dim_;
    static constexpr auto category = category_;
    using allocator_type = AllocatorT;
    using value_type = f32;
    using size_type = zs::make_unsigned_t<Index>;
    using index_type = zs::make_signed_t<Index>;
    using coord_index_type = zs::make_signed_t<Tn>;
    using table_t = HashTable<Tn, dim, index_type, allocator_type>;
    using vector_t = Vector<index_type, allocator_type>;

    constexpr MemoryLocation memoryLocation() const noexcept { return _table.memoryLocation(); }
    constexpr memsrc_e memspace() const noexcept { return _table.memspace(); }
    constexpr ProcID devid() const noexcept { return _table.devid(); }
    constexpr auto size() const noexcept { return _table.size(); }
    decltype(auto) get_allocator() const noexcept { return _table.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (size_t)1 << (size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

    constexpr IndexBuckets() = default;

    IndexBuckets clone(const allocator_type &allocator) const {
      IndexBuckets ret{};
      ret._table = _table.clone(allocator);
      ret._indices = _indices.clone(allocator);
      ret._offsets = _offsets.clone(allocator);
      ret._counts = _counts.clone(allocator);
      ret._dx = _dx;
      return ret;
    }
    IndexBuckets clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    constexpr auto numEntries() const noexcept { return _indices.size(); }
    constexpr auto numBuckets() const noexcept { return _counts.size() - 1; }

    table_t _table{};
    vector_t _indices{};
    vector_t _offsets{}, _counts{};
    value_type _dx{1};
  };

  using GeneralIndexBuckets
      = variant<IndexBuckets<3, i32, i32>, IndexBuckets<3, i64, i32>, IndexBuckets<3, i32, i64>,
                IndexBuckets<3, i64, i64>, IndexBuckets<2, i32, i32>, IndexBuckets<2, i64, i32>,
                IndexBuckets<2, i32, i64>, IndexBuckets<2, i64, i64>>;

  template <execspace_e Space, typename IndexBucketsT, typename = void> struct IndexBucketsView {
    static constexpr bool is_const_structure = is_const_v<IndexBucketsT>;
    static constexpr auto space = Space;
    using ib_t = remove_const_t<IndexBucketsT>;
    static constexpr int dim = ib_t::dim;
    static constexpr auto category = ib_t::category;
    using value_type = typename ib_t::value_type;
    using size_type = typename ib_t::size_type;
    using index_type = typename ib_t::index_type;
    using coord_index_type = typename ib_t::coord_index_type;
    using table_t = typename ib_t::table_t;
    using table_view_t = RM_REF_T(
        proxy<space>(declval<conditional_t<is_const_structure, const table_t &, table_t &>>()));
    using vector_t = typename ib_t::vector_t;
    using vector_view_t = RM_REF_T(
        proxy<space>(declval<conditional_t<is_const_structure, const vector_t &, vector_t &>>()));

    static constexpr auto coord_offset
        = category == grid_e::collocated ? (value_type)0.5 : (value_type)0;

    IndexBucketsView() noexcept = default;
    ~IndexBucketsView() noexcept = default;
    constexpr IndexBucketsView(IndexBucketsT &ibs)
        : table{proxy<Space>(ibs._table)},
          indices{proxy<Space>(ibs._indices)},
          offsets{proxy<Space>(ibs._offsets)},
          counts{proxy<Space>(ibs._counts)},
          dx{ibs._dx} {}

    constexpr auto coord(const size_type bucketno) const noexcept {
      return table._activeKeys[bucketno];
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           is_floating_point_v<typename VecT::value_type>>
                             = 0>
    constexpr auto bucketCoord(const VecInterface<VecT> &pos) const noexcept {
      const auto dxinv = (value_type)1.0 / dx;
      typename VecT::template variant_vec<coord_index_type, typename VecT::extents> coord{};
      for (typename VecT::index_type d = 0; d != VecT::extent; ++d)
        coord[d] = lower_trunc(pos[d] * dxinv + coord_offset, number_c<coord_index_type>);
      return coord;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           is_integral_v<typename VecT::value_type>>
                             = 0>
    constexpr auto bucketNo(const VecInterface<VecT> &coord) const noexcept {
      return table.query(coord);
    }

    table_view_t table{};  // activekeys, table
    vector_view_t indices{}, offsets{}, counts{};
    value_type dx{0};
  };

  template <execspace_e ExecSpace, int dim, typename Index, typename Tn, grid_e category,
            typename AllocatorT>
  decltype(auto) proxy(IndexBuckets<dim, Index, Tn, category, AllocatorT> &indexBuckets) {
    return IndexBucketsView<ExecSpace, IndexBuckets<dim, Index, Tn, category, AllocatorT>>{
        indexBuckets};
  }
  template <execspace_e ExecSpace, int dim, typename Index, typename Tn, grid_e category,
            typename AllocatorT>
  decltype(auto) proxy(
      const IndexBuckets<dim, Index, Tn, category, AllocatorT> &indexBuckets) {
    return IndexBucketsView<ExecSpace, const IndexBuckets<dim, Index, Tn, category, AllocatorT>>{
        indexBuckets};
  }

}  // namespace zs