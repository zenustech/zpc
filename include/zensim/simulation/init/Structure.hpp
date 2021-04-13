#pragma once
#include "zensim/TypeAlias.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  template <typename V = f32, typename I = i32, int d = 3> struct Mesh {
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
      = variant<Mesh<f32, i32, 2>, Mesh<f32, i64, 2>, Mesh<f32, i32, 3>, Mesh<f32, i64, 3>,
                Mesh<f64, i32, 2>, Mesh<f64, i64, 2>, Mesh<f64, i32, 3>, Mesh<f64, i64, 3>>;

  /// sizeof(float) = 4
  /// bin_size = 64
  /// attrib_size = 16
  template <typename V = dat32, int d = 3, int channel_bits = 4, int domain_bits = 2>
  struct GridBlock {
    static constexpr int dim = d;
    using IV = vec<int, dim>;

    constexpr auto &operator()(int c, IV loc) noexcept { return _data[c][offset(loc)]; }
    constexpr auto operator()(int c, IV loc) const noexcept { return _data[c][offset(loc)]; }

  protected:
    constexpr int offset(const IV &loc) const noexcept {
      // using Seq = typename gen_seq<d>::template uniform_values_t<vseq_t, (1 << domain_bits)>;
      int ret{0};
      if constexpr (d == 2)
        ret = (loc[0] << domain_bits) + loc[1];
      else if constexpr (d == 3)
        ret = (loc[0] << (domain_bits + domain_bits)) + (loc[1] << domain_bits) + loc[2];
      return ret;
    }
    V _data[1 << channel_bits][(1 << domain_bits) << dim];
  };

  template <typename Block> struct GridBlocks;
  template <typename V, int d, int chn_bits, int domain_bits>
  struct GridBlocks<GridBlock<V, d, chn_bits, domain_bits>> {
    using Block = GridBlock<V, d, chn_bits, domain_bits>;

    Vector<Block> blocks;
  };

  using GeneralGridBlocks
      = variant<GridBlocks<GridBlock<dat32, 2, 3, 2>>, GridBlocks<GridBlock<dat32, 3, 3, 2>>,
                GridBlocks<GridBlock<dat64, 2, 3, 2>>, GridBlocks<GridBlock<dat64, 3, 3, 2>>,
                GridBlocks<GridBlock<dat32, 2, 4, 2>>, GridBlocks<GridBlock<dat32, 3, 4, 2>>,
                GridBlocks<GridBlock<dat64, 2, 4, 2>>, GridBlocks<GridBlock<dat64, 3, 4, 2>>>;

  template <typename I = i32, int d = 3> struct Nodes {
    using IV = I[d];
    Vector<IV> nodes;
    f32 dx;
  };

  using GeneralNodes = variant<Nodes<i32, 2>, Nodes<i64, 2>, Nodes<i32, 3>, Nodes<i64, 3>>;

}  // namespace zs