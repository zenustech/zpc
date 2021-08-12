#pragma once
#include <map>

#include "zensim/TypeAlias.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/tpls/gcem/gcem_incl/pow.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

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

  /// sizeof(float) = 4
  /// bin_size = 64
  /// attrib_size = 16
  template <typename V = dat32, int dim_ = 3, int channel_bits = 4, int domain_bits = 2>
  struct GridBlock {
    using value_type = V;
    using size_type = int;
    static constexpr int dim = dim_;
    using IV = vec<size_type, dim>;
    static constexpr int num_chns() noexcept { return 1 << channel_bits; }
    static constexpr int side_length() noexcept { return 1 << domain_bits; }
    static constexpr int space() noexcept { return 1 << domain_bits * dim; }

    constexpr auto &operator()(int c, IV loc) noexcept { return _data[c][offset(loc)]; }
    constexpr auto operator()(int c, IV loc) const noexcept { return _data[c][offset(loc)]; }
    constexpr auto &operator()(int c, size_type cellid) noexcept { return _data[c][cellid]; }
    constexpr auto operator()(int c, size_type cellid) const noexcept { return _data[c][cellid]; }
    static constexpr IV to_coord(size_type cellid) {
      IV ret{IV::zeros()};
      for (int d = dim - 1; d >= 0 && cellid > 0; --d, cellid >>= domain_bits)
        ret[d] = cellid % side_length();
      return ret;
    }

  protected:
    constexpr int offset(const IV &loc) const noexcept {
      // using Seq = typename gen_seq<d>::template uniform_values_t<vseq_t, (1 << domain_bits)>;
      size_type ret{0};
      for (int d = 0; d < dim; ++d) ret = (ret << domain_bits) + loc[d];
      return ret;
    }
    V _data[num_chns()][space()];
  };

  template <typename Block> struct GridBlocks;
  template <typename V, int d, int chn_bits, int domain_bits>
  struct GridBlocks<GridBlock<V, d, chn_bits, domain_bits>> {
    using value_type = V;
    using block_t = GridBlock<V, d, chn_bits, domain_bits>;
    static constexpr int dim = block_t::dim;
    using IV = typename block_t::IV;
    using size_type = typename Vector<block_t>::size_type;

    constexpr GridBlocks(float dx = 1.f, std::size_t numBlocks = 0, memsrc_e mre = memsrc_e::host,
                         ProcID devid = -1)
        : blocks{numBlocks, mre, devid}, dx{dx} {}

    void resize(size_type newSize) { blocks.resize(newSize); }
    void capacity() const noexcept { blocks.capacity(); }
    Vector<block_t> blocks;
    V dx;
  };

#if 0
  using GeneralGridBlocks
      = variant<GridBlocks<GridBlock<dat32, 2, 3, 2>>, GridBlocks<GridBlock<dat32, 3, 3, 2>>,
                GridBlocks<GridBlock<dat64, 2, 3, 2>>, GridBlocks<GridBlock<dat64, 3, 3, 2>>,
                GridBlocks<GridBlock<dat32, 2, 4, 2>>, GridBlocks<GridBlock<dat32, 3, 4, 2>>,
                GridBlocks<GridBlock<dat64, 2, 4, 2>>, GridBlocks<GridBlock<dat64, 3, 4, 2>>>;
#else
  using GeneralGridBlocks = variant<GridBlocks<GridBlock<dat32, 3, 2, 2>>>;
#endif

  template <execspace_e, typename GridBlocksT, typename = void> struct GridBlocksView;
  template <execspace_e space, typename GridBlocksT> struct GridBlocksView<space, GridBlocksT> {
    using value_type = typename GridBlocksT::value_type;
    using block_t = typename GridBlocksT::block_t;
    static constexpr int dim = block_t::dim;
    using IV = typename block_t::IV;
    using size_type = typename GridBlocksT::size_type;

    constexpr GridBlocksView() = default;
    ~GridBlocksView() = default;
    explicit constexpr GridBlocksView(GridBlocksT &gridblocks)
        : _gridBlocks{gridblocks.blocks.data()},
          _blockCount{gridblocks.blocks.size()},
          _dx{gridblocks.dx} {}

    constexpr block_t &operator[](size_type i) { return _gridBlocks[i]; }
    constexpr const block_t &operator[](size_type i) const { return _gridBlocks[i]; }

    block_t *_gridBlocks;
    size_type _blockCount;
    value_type _dx;
  };

  template <execspace_e ExecSpace, typename V, int d, int chn_bits, int domain_bits>
  constexpr decltype(auto) proxy(GridBlocks<GridBlock<V, d, chn_bits, domain_bits>> &blocks) {
    return GridBlocksView<ExecSpace, GridBlocks<GridBlock<V, d, chn_bits, domain_bits>>>{blocks};
  }

  template <typename ValueT = f32, int d = 3, int SideLength = 4> struct Grids {
    enum struct grid_e : unsigned char { collocated = 0, cellcentered, staggered };

    using value_type = ValueT;
    using allocator_type = ZSPmrAllocator<>;
    static constexpr int dim = d;
    static constexpr int side_length = SideLength;
    static constexpr int block_space() noexcept { return gcem::pow(side_length, dim); }
    using grid_storage_t = TileVector<value_type, (std::size_t)block_space()>;
    using size_type = typename grid_storage_t::size_type;
    using channel_counter_type = typename grid_storage_t::channel_counter_type;

    using IV = vec<int, dim>;

    constexpr MemoryLocation memoryLocation() const noexcept {
      return match([](auto &&att) { return att.memoryLocation(); })(attr("mv"));
    }
    constexpr memsrc_e space() const noexcept {
      return match([](auto &&att) { return att.memspace(); })(attr("mv"));
    }
    constexpr ProcID devid() const noexcept {
      return match([](auto &&att) { return att.devid(); })(attr("mv"));
    }
    constexpr auto size() const noexcept {
      return match([](auto &&att) { return att.size(); })(attr("mv"));
    }
    constexpr decltype(auto) allocator() const noexcept {
      return match([](auto &&att) { return att.allocator(); })(attr("mv"));
    }
    constexpr decltype(auto) numBlocks() const noexcept {
      return match([](auto &&att) { return att.numTiles(); })(attr("mv"));
    }

    struct Grid {
      Grid(const allocator_type &allocator, const PropertyTag &channelTag, size_type count = 0,
           grid_e ge = grid_e::collocated)
          : blocks{allocator, {channelTag}, count}, category{ge} {}
      Grid(const PropertyTag &channelTag, size_type count, memsrc_e mre = memsrc_e::host,
           ProcID devid = -1, grid_e ge = grid_e::collocated)
          : Grid{get_memory_source(mre, devid), channelTag, count, ge} {}
      Grid(channel_counter_type numChns, size_type count, memsrc_e mre = memsrc_e::host,
           ProcID devid = -1, grid_e ge = grid_e::collocated)
          : Grid{get_memory_source(mre, devid), PropertyTag{"default", numChns}, count, ge} {}
      Grid(memsrc_e mre = memsrc_e::host, ProcID devid = -1, grid_e ge = grid_e::collocated)
          : Grid{get_memory_source(mre, devid), PropertyTag{"default", 1 + dim}, 0, ge} {}
      grid_storage_t blocks;
      grid_e category;
    };
    struct GridDescriptor {
      const SmallString name;
      const channel_counter_type nchns;
      const grid_e category;
    };

    static GridDescriptor get_grid_property(const Grid &grid) noexcept {
      auto prop = grid.getPropertyTag(0);
      return GridDescriptor{std::get<SmallString>(prop), (channel_counter_type)std::get<1>(prop),
                            grid.category};
    }

    constexpr Grids(const allocator_type &allocator, value_type dx = 1.f, size_type numBlocks = 0,
                    grid_e ge = grid_e::collocated) {
      _grids["mv"] = Grid{allocator, {"mv", 1 + dim}, numBlocks, ge};
    }
    constexpr Grids(value_type dx = 1.f, size_type numBlocks = 0, memsrc_e mre = memsrc_e::host,
                    ProcID devid = -1, grid_e ge = grid_e::collocated)
        : Grids{get_memory_source(mre, devid), dx, numBlocks, ge} {}

    constexpr decltype(auto) attr(const std::string &attrib) { return _grids.at(attrib); }
    constexpr decltype(auto) attr(const std::string &attrib) const { return _grids.at(attrib); }

    // void resize(size_type newSize) { blocks.resize(newSize); }
    // void capacity() const noexcept { blocks.capacity(); }

    std::map<std::string, Grid> _grids;
    value_type _dx;
  };

  using GeneralGrids = variant<Grids<f32, 3, 4>, Grids<f32, 3, 2>, Grids<f32, 3, 1>,
                               Grids<f32, 2, 4>, Grids<f32, 2, 2>, Grids<f32, 2, 1>>;

}  // namespace zs