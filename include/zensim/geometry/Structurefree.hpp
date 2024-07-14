#pragma once
#include <map>
#include <stdexcept>

#include "zensim/TypeAlias.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Optional.h"
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

    using Attribute = variant<Vector<T>, Vector<TV>, Vector<TM>, Vector<TMAffine>>;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = typename Vector<TV>::size_type;
    static constexpr int dim = d;
    template <typename TT> static constexpr attrib_e get_attribute_enum(const Vector<TT> &) {
      if constexpr (is_same_v<TT, T>)
        return attrib_e::scalar;
      else if constexpr (is_same_v<TT, TV>)
        return attrib_e::vector;
      else if constexpr (is_same_v<TT, TM>)
        return attrib_e::matrix;
      else if constexpr (is_same_v<TT, TMAffine>)
        return attrib_e::affine;
      throw std::runtime_error(
          fmt::format(" array of \"{}\" is not a known attribute type\n", demangle<T>()));
    }
    static constexpr attrib_e get_attribute_enum(const Attribute &att) {
      return match([](const auto &att) { return get_attribute_enum(att); })(att);
    }

    constexpr MemoryLocation memoryLocation() const noexcept {
      return match([](auto &&att) { return att.memoryLocation(); })(attr("x"));
    }
    constexpr memsrc_e space() const noexcept {
      return match([](auto &&att) { return att.memspace(); })(attr("x"));
    }
    constexpr ProcID devid() const noexcept {
      return match([](auto &&att) { return att.devid(); })(attr("x"));
    }
    constexpr auto size() const noexcept {
      return match([](auto &&att) { return att.size(); })(attr("x"));
    }
    decltype(auto) get_allocator() const noexcept {
      return match([](auto &&att) { return att.get_allocator(); })(attr("x"));
    }

    Particles(const allocator_type &allocator, size_type count) {
      /// "x" is a reserved key
      _attributes["x"] = Vector<TV>{allocator, count};
    }
    Particles(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Particles{get_memory_source(mre, devid), 0} {}
    Particles(size_type count, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Particles{get_memory_source(mre, devid), count} {}

    std::vector<std::array<ValueT, dim>> retrievePositions() const {
      const auto &X = attr<TV>("x");
      Vector<TV> Xtmp{X.size(), memsrc_e::host, -1};
      Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)Xtmp.data()},
                     MemoryEntity{X.memoryLocation(), (void *)X.data()}, X.size() * sizeof(TV));
      std::vector<std::array<ValueT, dim>> ret(X.size());
      memcpy(ret.data(), Xtmp.data(), sizeof(TV) * X.size());
      return ret;
    }
    std::vector<T> retrieveStressMagnitude() const {
      const auto &X = attr<TV>("x");
      std::vector<T> ret(X.size());
      if (hasAttr("F", true)) {
        const auto &F = attr<TM>("F");
        Vector<TM> Ftmp{X.size()};
        Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)Ftmp.data()},
                       MemoryEntity{F.memoryLocation(), (void *)F.data()}, F.size() * sizeof(TM));
        for (size_type i = 0; i < Ftmp.size(); ++i) {
          const auto &v = Ftmp[i];
          ret[i] = v(0) * (v(4) * v(8) - v(5) * v(7)) - v(1) * (v(3) * v(8) - v(5) * v(6))
                   + v(2) * (v(3) * v(7) - v(4) * v(6));
        }
      } else if (hasAttr("J", true)) {
        const auto &J = attr<TM>("F");
        Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)ret.data()},
                       MemoryEntity{J.memoryLocation(), (void *)J.data()}, J.size() * sizeof(T));
      }
      return ret;
    }

    constexpr decltype(auto) attrs() { return _attributes; }
    constexpr decltype(auto) attrs() const { return _attributes; }

    constexpr const Attribute *tryGet(const std::string &attrib) const noexcept {
      if (auto it = _attributes.find(attrib); it != _attributes.end()) return &it->second;
      return nullptr;
    }
    constexpr Attribute *tryGet(const std::string &attrib) noexcept {
      if (auto it = _attributes.find(attrib); it != _attributes.end()) return &it->second;
      return nullptr;
    }
    constexpr bool hasAttr(const std::string &attrib, bool checkEmpty = false) const noexcept {
      if (auto obj = tryGet(attrib); obj)
        if (!checkEmpty || match([](auto &&att) -> bool { return att.size() > 0; })(*obj))
          return true;
      return false;
    }
    // avoid [] because default ctor of Vector is not necessarily expected
    constexpr decltype(auto) attr(const std::string &attrib) { return _attributes.at(attrib); }
    constexpr decltype(auto) attr(const std::string &attrib) const {
      return _attributes.at(attrib);
    }
    template <typename AT> constexpr decltype(auto) attr(const std::string &attrib) {
      return std::get<Vector<AT>>(attr(attrib));
    }
    template <typename AT> constexpr decltype(auto) attr(const std::string &attrib) const {
      return std::get<Vector<AT>>(attr(attrib));
    }
    constexpr decltype(auto) attrScalar(const std::string &attrib) {
      return std::get<Vector<T>>(attr(attrib));
    }
    constexpr decltype(auto) attrScalar(const std::string &attrib) const {
      return std::get<Vector<T>>(attr(attrib));
    }
    constexpr decltype(auto) attrVector(const std::string &attrib) {
      return std::get<Vector<TV>>(attr(attrib));
    }
    constexpr decltype(auto) attrVector(const std::string &attrib) const {
      return std::get<Vector<TV>>(attr(attrib));
    }
    constexpr decltype(auto) attrMatrix(const std::string &attrib) {
      return std::get<Vector<TM>>(attr(attrib));
    }
    constexpr decltype(auto) attrMatrix(const std::string &attrib) const {
      return std::get<Vector<TM>>(attr(attrib));
    }
    constexpr decltype(auto) attrAffine(const std::string &attrib) {
      return std::get<Vector<TMAffine>>(attr(attrib));
    }
    constexpr decltype(auto) attrAffine(const std::string &attrib) const {
      return std::get<Vector<TMAffine>>(attr(attrib));
    }

    template <typename TT = void>
    constexpr decltype(auto) getAttrAddress(const std::string &attrib) {
      if (hasAttr(attrib))
        return match([](auto &&att) -> TT * { return att.data(); })(attr(attrib));
      return (TT *)nullptr;
    }
    template <typename TT = void>
    constexpr decltype(auto) getAttrAddress(const std::string &attrib) const {
      if (hasAttr(attrib))
        return match([](auto &&att) -> const TT * { return att.data(); })(attr(attrib));
      return (const TT *)nullptr;
    }

    constexpr Attribute &addAttr(const std::string &attrib, attrib_e ae) {
      if (auto obj = tryGet(attrib); obj) {
        if (get_attribute_enum(*obj) == ae) return *obj;
      }
      auto &att = _attributes[attrib];
      switch (ae) {
        case attrib_e::scalar:
          att = Vector<T>{get_allocator(), size()};
          break;
        case attrib_e::vector:
          att = Vector<TV>{get_allocator(), size()};
          break;
        case attrib_e::matrix:
          att = Vector<TM>{get_allocator(), size()};
          break;
        case attrib_e::affine:
          att = Vector<TMAffine>{get_allocator(), size()};
          break;
        default:;
      }
      return att;
    }

    void append(const Particles &other) {
      for (auto &&attrib : other.attrs())
        if (auto obj = tryGet(attrib.first); obj) {
          match(
              [](auto &&dst,
                 auto &&src) -> enable_if_type<is_same_v<RM_CVREF_T(dst), RM_CVREF_T(src)>> {
                dst.append(src);
              },
              [&attrib](auto &&dst, auto &&src)
                  -> enable_if_type<!is_same_v<RM_CVREF_T(dst), RM_CVREF_T(src)>> {
                throw std::runtime_error(
                    fmt::format("attributes of the same name \"{}\" are of type \"{}\"(dst) and "
                                "\"{}\"(src)\n",
                                attrib.first, demangle(dst), demangle(src)));
              })(*obj, attrib.second);
        } else
          _attributes[attrib.first] = attrib.second;
    }

    void resize(size_t newSize) {
      for (auto &&attrib : attrs()) match([newSize](auto &&att) { att.resize(newSize); })(attrib);
    }

    /// aux channels
    // SoAVector<dat32> aux32;
    // SoAVector<dat64> aux64;
    TileVector<T, 32> particleBins;  // should be optional (mass, pos, vel, J, F, C, logJp)

  protected:
    std::map<std::string, Attribute> _attributes;
  };

#if 0
  using GeneralParticles
      = variant<Particles<f32, 2>, Particles<f64, 2>, Particles<f32, 3>, Particles<f64, 3>>;
#else
  using GeneralParticles = variant<Particles<f32, 3>, Particles<f32, 2>>;
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
        : _M{(T *)particles.getAttrAddress("m")},
          _X{(TV *)particles.getAttrAddress("x")},
          _V{(TV *)particles.getAttrAddress("v")},
          _Dinv{(TV *)particles.getAttrAddress("Dinv")},
          _J{(T *)particles.getAttrAddress("J")},
          _F{(TM *)particles.getAttrAddress("F")},
          _C{(TM *)particles.getAttrAddress("C")},
          _logJp{(T *)particles.getAttrAddress("logJp")},
          _particleCount{particles.size()} {}

    constexpr auto &mass(size_type parid) { return _M[parid]; }
    constexpr auto mass(size_type parid) const { return _M[parid]; }
    constexpr auto &pos(size_type parid) { return _X[parid]; }
    constexpr const auto &pos(size_type parid) const { return _X[parid]; }
    constexpr auto &vel(size_type parid) { return _V[parid]; }
    constexpr const auto &vel(size_type parid) const { return _V[parid]; }
    constexpr auto &Dinv(size_type parid) { return _Dinv[parid]; }
    constexpr const auto &Dinv(size_type parid) const { return _Dinv[parid]; }
    /// deformation for water
    constexpr auto &J(size_type parid) { return _J[parid]; }
    constexpr const auto &J(size_type parid) const { return _J[parid]; }
    /// deformation for solid
    constexpr auto &F(size_type parid) { return _F[parid]; }
    constexpr const auto &F(size_type parid) const { return _F[parid]; }
    /// for apic transfer only
    constexpr auto &C(size_type parid) { return _C[parid]; }
    constexpr const auto &C(size_type parid) const { return _C[parid]; }
    constexpr auto &B(size_type parid) { return _C[parid]; }
    constexpr const auto &B(size_type parid) const { return _C[parid]; }
    /// plasticity
    constexpr auto &logJp(size_type parid) { return _logJp[parid]; }
    constexpr const auto &logJp(size_type parid) const { return _logJp[parid]; }
    constexpr auto size() const noexcept { return _particleCount; }

  protected:
    T *_M;
    TV *_X, *_V, *_Dinv;
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
        : _M{(const T *)particles.getAttrAddress("m")},
          _X{(const TV *)particles.getAttrAddress("x")},
          _V{(const TV *)particles.getAttrAddress("v")},
          _Dinv{(const TV *)particles.getAttrAddress("Dinv")},
          _J{(const T *)particles.getAttrAddress("J")},
          _F{(const TM *)particles.getAttrAddress("F")},
          _C{(const TM *)particles.getAttrAddress("C")},
          _logJp{(const T *)particles.getAttrAddress("logJp")},
          _particleCount{particles.size()} {}

    constexpr auto mass(size_type parid) const { return _M[parid]; }
    constexpr const auto &pos(size_type parid) const { return _X[parid]; }
    constexpr const auto &vel(size_type parid) const { return _V[parid]; }
    constexpr const auto &Dinv(size_type parid) const { return _Dinv[parid]; }
    /// deformation for water
    constexpr const auto &J(size_type parid) const { return _J[parid]; }
    /// deformation for solid
    constexpr const auto &F(size_type parid) const { return _F[parid]; }
    /// for apic transfer only
    constexpr const auto &C(size_type parid) const { return _C[parid]; }
    constexpr const auto &B(size_type parid) const { return _C[parid]; }
    /// plasticity
    constexpr const auto &logJp(size_type parid) const { return _logJp[parid]; }

    constexpr auto size() const noexcept { return _particleCount; }

  protected:
    const T *_M;
    const TV *_X;
    const TV *_V, *_Dinv;
    const T *_J;
    const TM *_F;
    const TM *_C;
    const T *_logJp;
    size_type _particleCount;
  };

  template <execspace_e ExecSpace, typename V, int d>
  decltype(auto) proxy(Particles<V, d> &particles) {
    return ParticlesView<ExecSpace, Particles<V, d>>{particles};
  }
  template <execspace_e ExecSpace, typename V, int d>
  decltype(auto) proxy(const Particles<V, d> &particles) {
    return ParticlesView<ExecSpace, const Particles<V, d>>{particles};
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