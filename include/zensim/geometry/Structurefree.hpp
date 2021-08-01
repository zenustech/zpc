#pragma once
#include "zensim/TypeAlias.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/resource/Resource.h"
#include "zensim/types/Polymorphism.h"

namespace zs {

  enum ParticleAttributeFlagBit : char {
    None = 0,
    Particle_J = 0x1,
    Particle_F = 0x2,
    Particle_C = 0x4
  };
  using ParticleAttributeFlagBits = char;

  template <typename ValueT = f32, int d = 3> struct Particles {
    using T = ValueT;
    // using TV = ValueT[d];
    using TV = vec<T, d>;
    // using TM = vec<T, d, d>;
    using TM = vec<T, d * d>;
    using TMAffine = vec<T, (d + 1) * (d + 1)>;
    using size_type = typename Vector<TV>::size_type;
    static constexpr int dim = d;

    constexpr MemoryLocation memoryLocation() const noexcept { return X.memoryLocation(); }
    constexpr decltype(auto) getAllocator() const noexcept { return X.allocator(); }
    constexpr memsrc_e space() const noexcept { return X.memspace(); }
    constexpr ProcID devid() const noexcept { return X.devid(); }
    constexpr auto size() const noexcept { return X.size(); }

    std::vector<std::array<ValueT, dim>> retrievePositions() const {
      Vector<TV> Xtmp{X.size(), memsrc_e::host, -1};
      copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)Xtmp.data()},
           MemoryEntity{X.memoryLocation(), (void *)X.data()}, X.size() * sizeof(TV));
      std::vector<std::array<ValueT, dim>> ret(X.size());
      memcpy(ret.data(), Xtmp.data(), sizeof(TV) * X.size());
      return ret;
    }
    std::vector<T> retrieveStressMagnitude() const {
      std::vector<T> ret(X.size());
      if (F.size()) {
        Vector<TM> Ftmp{X.size()};
        copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)Ftmp.data()},
             MemoryEntity{F.memoryLocation(), (void *)F.data()}, F.size() * sizeof(TM));
        for (size_type i = 0; i < Ftmp.size(); ++i) {
          const auto &v = Ftmp[i];
          ret[i] = v(0) * (v(4) * v(8) - v(5) * v(7)) - v(1) * (v(3) * v(8) - v(5) * v(6))
                   + v(2) * (v(3) * v(7) - v(4) * v(6));
        }
      } else if (J.size()) {
        copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)ret.data()},
             MemoryEntity{J.memoryLocation(), (void *)J.data()}, J.size() * sizeof(T));
      }
      return ret;
    }
#define CHECK_ATTRIBUTE(ATTRIB) \
  constexpr bool has##ATTRIB() const noexcept { return ATTRIB.size() > 0; }

    CHECK_ATTRIBUTE(M)
    CHECK_ATTRIBUTE(X)
    CHECK_ATTRIBUTE(V)
    CHECK_ATTRIBUTE(J)
    CHECK_ATTRIBUTE(logJp)
    CHECK_ATTRIBUTE(F)
    CHECK_ATTRIBUTE(C)

    void append(const Particles &other) {
      M.append(other.M);
      X.append(other.X);
      V.append(other.V);
      J.append(other.J);
      F.append(other.F);
      C.append(other.C);
    }
    void resize(std::size_t newSize) {
      M.resize(newSize);
      X.resize(newSize);
      V.resize(newSize);
      if (J.size()) J.resize(newSize);
      if (F.size()) F.resize(newSize);
      if (C.size()) C.resize(newSize);
    }

    Vector<T> M;
    Vector<TV> X, V;
    Vector<T> J, logJp;
    Vector<TM> F, C;
    /// aux channels
    // SoAVector<dat32> aux32;
    // SoAVector<dat64> aux64;
    TileVector<f32, 32> particleBins;  // should be optional (mass, pos, vel, J, F, C, logJp)
  };

#if 0
  using GeneralParticles
      = variant<Particles<f32, 2>, Particles<f64, 2>, Particles<f32, 3>, Particles<f64, 3>>;
#else
  using GeneralParticles = variant<Particles<f32, 3>>;
#endif

  template <execspace_e space, typename ParticlesT, typename = void> struct ParticlesView {
    using T = typename ParticlesT::T;
    using TV = typename ParticlesT::TV;
    using TM = typename ParticlesT::TM;
    static constexpr int dim = ParticlesT::dim;
    using size_type = typename ParticlesT::size_type;

    ParticlesView() = default;
    ~ParticlesView() = default;
    explicit constexpr ParticlesView(ParticlesT &particles)
        : _M{particles.M.data()},
          _X{particles.X.data()},
          _V{particles.V.data()},
          _J{particles.J.data()},
          _F{particles.F.data()},
          _C{particles.C.data()},
          _logJp{particles.logJp.data()},
          _particleCount{particles.size()} {}

    constexpr auto &mass(size_type parid) { return _M[parid]; }
    constexpr auto mass(size_type parid) const { return _M[parid]; }
    constexpr auto &pos(size_type parid) { return _X[parid]; }
    constexpr const auto &pos(size_type parid) const { return _X[parid]; }
    constexpr auto &vel(size_type parid) { return _V[parid]; }
    constexpr const auto &vel(size_type parid) const { return _V[parid]; }
    /// deformation for water
    constexpr auto &J(size_type parid) { return _J[parid]; }
    constexpr const auto &J(size_type parid) const { return _J[parid]; }
    /// deformation for solid
    constexpr auto &F(size_type parid) { return _F[parid]; }
    constexpr const auto &F(size_type parid) const { return _F[parid]; }
    /// for apic transfer only
    constexpr auto &C(size_type parid) { return _C[parid]; }
    constexpr const auto &C(size_type parid) const { return _C[parid]; }
    /// plasticity
    constexpr auto &logJp(size_type parid) { return _logJp[parid]; }
    constexpr const auto &logJp(size_type parid) const { return _logJp[parid]; }
    constexpr auto size() const noexcept { return _particleCount; }

  protected:
    T *_M;
    TV *_X, *_V;
    T *_J;
    TM *_F, *_C;
    T *_logJp;
    size_type _particleCount;
  };

  template <execspace_e space, typename ParticlesT> struct ParticlesView<space, const ParticlesT> {
    using T = typename ParticlesT::T;
    using TV = typename ParticlesT::TV;
    using TM = typename ParticlesT::TM;
    static constexpr int dim = ParticlesT::dim;
    using size_type = typename ParticlesT::size_type;

    ParticlesView() = default;
    ~ParticlesView() = default;
    explicit constexpr ParticlesView(const ParticlesT &particles)
        : _M{particles.M.data()},
          _X{particles.X.data()},
          _V{particles.V.data()},
          _J{particles.J.data()},
          _F{particles.F.data()},
          _C{particles.C.data()},
          _logJp{particles.logJp.data()},
          _particleCount{particles.size()} {}

    constexpr auto mass(size_type parid) const { return _M[parid]; }
    constexpr const auto &pos(size_type parid) const { return _X[parid]; }
    constexpr const auto &vel(size_type parid) const { return _V[parid]; }
    /// deformation for water
    constexpr const auto &J(size_type parid) const { return _J[parid]; }
    /// deformation for solid
    constexpr const auto &F(size_type parid) const { return _F[parid]; }
    /// for apic transfer only
    constexpr const auto &C(size_type parid) const { return _C[parid]; }
    /// plasticity
    constexpr const auto &logJp(size_type parid) const { return _logJp[parid]; }

    constexpr auto size() const noexcept { return _particleCount; }

  protected:
    T *_M;
    TV *_X, *_V;
    T *_J;
    TM *_F, *_C;
    T *_logJp;
    size_type _particleCount;
  };

  template <execspace_e ExecSpace, typename V, int d>
  constexpr decltype(auto) proxy(Particles<V, d> &particles) {
    return ParticlesView<ExecSpace, Particles<V, d>>{particles};
  }
  template <execspace_e ExecSpace, typename V, int d>
  constexpr decltype(auto) proxy(const Particles<V, d> &particles) {
    return ParticlesView<ExecSpace, const Particles<V, d>>{particles};
  }

  ///
  /// tiles particles
  ///
  template <auto Length, typename ValueT = f32, int d = 3> struct TiledParticles {
    using T = ValueT;
    using TV = vec<ValueT, d>;
    using TM = vec<ValueT, d, d>;
    using TMAffine = vec<ValueT, d + 1, d + 1>;
    using size_type = std::size_t;
    static constexpr int dim = d;
    static constexpr auto lane_width = Length;
    template <typename TT> using Tiles = TileVector<TT, lane_width, size_type, int>;
    template <typename TT> using tiles_t =
        typename TileVector<TT, lane_width, size_type, int>::base_t;

    constexpr MemoryHandle handle() const noexcept { return static_cast<MemoryHandle>(X); }
    constexpr memsrc_e space() const noexcept { return X.memspace(); }
    constexpr ProcID devid() const noexcept { return X.devid(); }
    constexpr auto size() const noexcept { return X.size(); }

    void resize(std::size_t newSize) { X.resize(newSize); }

    Tiles<TV> X;
  };

  template <execspace_e, typename ParticlesT, typename = void> struct TiledParticlesView;
  template <execspace_e space, typename TiledParticlesT>
  struct TiledParticlesView<space, TiledParticlesT> {
    using T = typename TiledParticlesT::T;
    using TV = typename TiledParticlesT::TV;
    using TM = typename TiledParticlesT::TM;
    static constexpr int dim = TiledParticlesT::dim;
    using size_type = typename TiledParticlesT::size_type;
    template <typename TT> using tiles_t = typename TiledParticlesT::template tiles_t<TT>;

    constexpr TiledParticlesView() = default;
    ~TiledParticlesView() = default;
    explicit constexpr TiledParticlesView(TiledParticlesT &particles)
        : _X{particles.X.data()}, _particleCount{particles.size()} {}

    constexpr auto &pos(size_type parid) { return _X[parid]; }
    constexpr const auto &pos(size_type parid) const { return _X[parid]; }
    constexpr auto size() const noexcept { return _particleCount; }

  protected:
    tiles_t<TV> _X;
    size_type _particleCount;
  };

  template <execspace_e ExecSpace, auto Length, typename V, int d>
  constexpr decltype(auto) proxy(TiledParticles<Length, V, d> &particles) {
    return TiledParticlesView<ExecSpace, TiledParticles<Length, V, d>>{particles};
  }

  /// sizeof(float) = 4
  /// bin_size = 64
  /// attrib_size = 16
  template <typename V = dat32, int channel_bits = 4, int counter_bits = 6> struct ParticleBin {
    constexpr decltype(auto) operator[](int c) noexcept { return _data[c]; }
    constexpr decltype(auto) operator[](int c) const noexcept { return _data[c]; }
    constexpr auto &operator()(int c, int pid) noexcept { return _data[c][pid]; }
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