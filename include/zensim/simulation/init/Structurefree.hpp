#include "zensim/TypeAlias.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Polymorphism.h"

namespace zs {

  template <typename ValueT = f32, int d = 3> struct Particles {
    using TV = ValueT[d];
    using TM = ValueT[d][d];
    using TMAffine = ValueT[d + 1][d + 1];
    static constexpr int dim = d;
    Vector<ValueT> M;
    Vector<TV> X, V;
    Vector<TM> F;
  };

  using GeneralParticles
      = variant<Particles<f32, 2>, Particles<f64, 2>, Particles<f32, 3>, Particles<f64, 3>>;

  /// sizeof(float) = 4
  /// bin_size = 64
  /// attrib_size = 16
  template <typename V = dat32, int channel_bits = 4, int counter_bits = 6> struct ParticleBin {
    constexpr decltype(auto) operator[](int c) noexcept { return _data[c]; }
    constexpr decltype(auto) operator[](int c) const noexcept { return _data[c]; }
    constexpr auto& operator()(int c, int pid) noexcept { return _data[c][pid]; }
    constexpr auto operator()(int c, int pid) const noexcept { return _data[c][pid]; }

  protected:
    V _data[1 << channel_bits][1 << counter_bits];
  };

  template <typename PBin, int bin_bits = 4> struct ParticleGrid;
  template <typename V, int chnbits, int cntbits, int bin_bits>
  struct ParticleGrid<ParticleBin<V, chnbits, cntbits>, bin_bits> {
    static constexpr int nbins = (1 << bin_bits);
    using Bin = ParticleBin<V, chnbits, cntbits>;
    using Block = Bin[nbins];

    Vector<Block> blocks;
    Vector<int> ppbs;
  };
  template <typename V, int chnbits, int cntbits>
  struct ParticleGrid<ParticleBin<V, chnbits, cntbits>, -1> {
    using Bin = ParticleBin<V, chnbits, cntbits>;

    Vector<Bin> bins;
    Vector<int> cnts, offsets;
    Vector<int> ppbs;
  };

}  // namespace zs