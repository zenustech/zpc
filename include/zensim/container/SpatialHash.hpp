#pragma once

#include "Bht.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"

namespace zs {

  template <int dim_ = 3, typename Index = int, typename ValueT = zs::f32,
            typename AllocatorT = zs::ZSPmrAllocator<>>
  struct SpatialHash {
    static constexpr int dim = dim_;
    using allocator_type = AllocatorT;
    using value_type = ValueT;
    // must be signed integer, since we are using -1 as sentinel value
    using index_type = zs::make_signed_t<Index>;
    using size_type = zs::make_unsigned_t<Index>;
    static_assert(is_floating_point_v<value_type>, "value_type should be floating point");
    static_assert(is_integral_v<index_type>, "index_type should be an integral");

    using bv_t = zs::AABBBox<dim, value_type>;
    using coord_type = zs::vec<value_type, dim>;
    using integer_coord_type = zs::vec<int, dim>;
    using table_type = bht<int, dim, index_type, 16, allocator_type>;
    using indices_type = zs::Vector<index_type, allocator_type>;

    constexpr decltype(auto) memoryLocation() const noexcept { return _table.memoryLocation(); }
    constexpr zs::ProcID devid() const noexcept { return _table.devid(); }
    constexpr zs::memsrc_e memspace() const noexcept { return _table.memspace(); }
    decltype(auto) get_allocator() const noexcept { return _table.get_allocator(); }
    decltype(auto) get_default_allocator(zs::memsrc_e mre, zs::ProcID devid) const {
      return _table.get_default_allocator(mre, devid);
    }

    SpatialHash() = default;

    SpatialHash clone(const allocator_type &allocator) const {
      SpatialHash ret{};
      ret._dx = _dx;
      ret._table = _table.clone(allocator);
      ret._indices = _indices.clone(allocator);
      ret._offsets = _offsets.clone(allocator);
      return ret;
    }
    SpatialHash clone(const zs::MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    size_type numActiveCells() const { return _table.size(); }

    template <typename Policy>
    void build(Policy &&, value_type dx, const zs::Vector<bv_t> &primBvs);

    /// @brief cell side length
    value_type _dx;
    /// @brief sparse cells activated by input primitives
    table_type _table;
    /// @brief primitive indices grouped in cells
    indices_type _indices{};
    /// @brief primitive index offset of each cell
    indices_type _offsets{};
  };

  template <zs::execspace_e, typename ShT, typename = void> struct SpatialHashView;

  /// proxy to work within each backends
  template <zs::execspace_e Space, typename ShT> struct SpatialHashView<Space, ShT> {
    static constexpr bool is_const_structure = is_const_v<ShT>;
    static constexpr auto space = Space;
    using container_type = remove_const_t<ShT>;
    static constexpr int dim = ShT::dim;
    using index_type = typename ShT::index_type;
    using size_type = typename ShT::size_type;
    using value_type = typename ShT::value_type;

    using bv_t = typename ShT::bv_t;
    using coord_type = typename ShT::coord_type;
    using integer_coord_type = typename ShT::integer_coord_type;
    using table_type = typename ShT::table_type;
    using table_view_type = RM_REF_T(proxy<space>(
        declval<conditional_t<is_const_structure, const table_type &, table_type &>>()));
    using indices_type = typename ShT::indices_type;
    using indices_view_type = RM_REF_T(proxy<space>(
        declval<conditional_t<is_const_structure, const indices_type &, indices_type &>>()));

    constexpr SpatialHashView() = default;
    ~SpatialHashView() = default;

    constexpr SpatialHashView(ShT &sh)
        : _dx{sh._dx},
          _table{zs::proxy<space>(sh._table)},
          _indices{zs::proxy<space>(sh._indices)},
          _offsets{zs::proxy<space>(sh._offsets)} {}

    template <class F> constexpr void iter_neighbors(const bv_t &bv, F &&f) const {
      auto mi = integer_coord_type::init([&mi = bv._min, dxinv = 1 / _dx](int d) -> index_type {
        return lower_trunc(mi[d] * dxinv, zs::wrapt<index_type>{});
      });
      auto ma = integer_coord_type::init([&ma = bv._max, dxinv = 1 / _dx](int d) -> index_type {
        return zs::ceil(ma[d] * dxinv);
      });
      auto range = Collapse(ma - mi);
      for (auto loc : range) {
        auto cno = _table.query(mi + make_vec<int>(loc));
        if (cno < 0) continue;

        auto st = _offsets[cno];
        auto ed = _offsets[cno + 1];
        for (size_type no = st; no != ed; ++no) f(_indices[no]);
      }
    }

    template <typename VecT, class F>
    constexpr void iter_neighbors(const VecInterface<VecT> &p, F &&f) const {
      auto loc = integer_coord_type::init([&p, dxinv = 1 / _dx](int d) -> index_type {
        return lower_trunc(p[d] * dxinv, zs::wrapt<index_type>{});
      });
      auto cno = _table.query(loc);
      if (cno < 0) return;

      auto st = _offsets[cno];
      auto ed = _offsets[cno + 1];
      for (size_type no = st; no != ed; ++no) f(_indices[no]);
    }

    value_type _dx;
    table_view_type _table;
    indices_view_type _indices, _offsets;
  };

  template <zs::execspace_e space, int dim, typename Ti, typename T, typename Allocator>
  decltype(auto) proxy(const SpatialHash<dim, Ti, T, Allocator> &sh) {
    return SpatialHashView<space, const SpatialHash<dim, Ti, T, Allocator>>{sh};
  }

  template <int dim, typename Index, typename Value, typename Allocator> template <typename Policy>
  void SpatialHash<dim, Index, Value, Allocator>::build(
      Policy &&policy, value_type dx, const zs::Vector<zs::AABBBox<dim, Value>> &primBvs) {
    using namespace zs;
    using T = value_type;
    using Ti = index_type;
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    constexpr auto execTag = wrapv<space>{};

    _dx = dx;
    if (_dx < detail::deduce_numeric_epsilon<value_type>() * 10)
      throw std::runtime_error("cell side_length for spatial hashing should be greater than zero.");

    if (primBvs.size() == 0) return;

    auto prevStatus = _table._buildSuccess.getVal();
    auto expectedNumEntries = primBvs.size();
    int numTrialIters = 0;
    _table = table_type{primBvs.get_allocator(), expectedNumEntries};

    do {
      policy(range(primBvs.size()), [primBvs = proxy<space>(primBvs), table = proxy<space>(_table),
                                     dxinv = 1 / _dx] ZS_LAMBDA(size_type i) mutable {
        using table_t = RM_CVREF_T(table);
        auto bv = primBvs[i];
        auto mi = integer_coord_type::init([&mi = bv._min, dxinv](int d) -> int {
          return lower_trunc(mi[d] * dxinv, zs::wrapt<int>{});
        });
        auto ma = integer_coord_type::init(
            [&ma = bv._max, dxinv](int d) -> int { return (int)zs::ceil(ma[d] * dxinv); });
        auto range = Collapse(ma - mi);
        for (auto loc : range) table.insert(mi + make_vec<int>(loc));
      });

      if (auto buildSuccess = _table._buildSuccess.getVal(); buildSuccess) break;
      if (++numTrialIters > 4)
        throw std::runtime_error(
            fmt::format("using dx[{}] as the cell side_length results in excessive ({}) spatial "
                        "hashing table size!",
                        dx, expectedNumEntries));
      expectedNumEntries *= 4;
      _table.resize(policy, expectedNumEntries);
    } while (true);

    const auto numCells = _table.size();
    indices_type counts{primBvs.get_allocator(), numCells + 1};
    counts.reset(0);

    policy(range(primBvs.size()), [primBvs = proxy<space>(primBvs), table = proxy<space>(_table),
                                   counts = proxy<space>(counts), dxinv = 1 / _dx,
                                   tag = wrapv<space>{}] ZS_LAMBDA(size_type i) mutable {
      using table_t = RM_CVREF_T(table);
      auto bv = primBvs[i];
      auto mi = integer_coord_type::init([&mi = bv._min, dxinv](int d) -> int {
        return lower_trunc(mi[d] * dxinv, zs::wrapt<int>{});
      });
      auto ma = integer_coord_type::init(
          [&ma = bv._max, dxinv](int d) -> int { return (int)zs::ceil(ma[d] * dxinv); });
      auto range = Collapse(ma - mi);
      for (auto loc : range) {
        auto cno = table.query(mi + make_vec<int>(loc));
        if (cno < 0) printf("\n\tshould not miss spatial hashing here!\t\n");
        atomic_add(tag, &counts[cno], (Ti)1);
      }
    });

    _offsets = indices_type{primBvs.get_allocator(), numCells + 1};
    exclusive_scan(policy, std::begin(counts), std::end(counts), std::begin(_offsets));

    const size_t numEntries = _offsets.getVal(numCells);
    _indices = indices_type{primBvs.get_allocator(), numEntries};

    policy(range(primBvs.size()), [primBvs = proxy<space>(primBvs), table = proxy<space>(_table),
                                   offsets = proxy<space>(_offsets), counts = proxy<space>(counts),
                                   indices = proxy<space>(_indices), dxinv = 1 / _dx,
                                   tag = wrapv<space>{}] ZS_LAMBDA(size_type i) mutable {
      using table_t = RM_CVREF_T(table);
      auto bv = primBvs[i];
      auto mi = integer_coord_type::init([&mi = bv._min, dxinv](int d) -> int {
        return lower_trunc(mi[d] * dxinv, zs::wrapt<int>{});
      });
      auto ma = integer_coord_type::init(
          [&ma = bv._max, dxinv](int d) -> int { return (int)zs::ceil(ma[d] * dxinv); });
      auto range = Collapse(ma - mi);
      for (auto loc : range) {
        auto cno = table.query(mi + make_vec<int>(loc));
        auto offset = offsets[cno];
        auto no = atomic_add(tag, &counts[cno], (Ti)-1) - 1;
        indices[offset + no] = i;
      }
    });

    return;
  }

}  // namespace zs