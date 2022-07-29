#pragma once

#include "Vector.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"

namespace zs {

  template <int dim_ = 3, typename Index = int, typename ValueT = f32,
            typename AllocatorT = ZSPmrAllocator<>>
  struct LBvs {
    static constexpr int dim = dim_;
    using allocator_type = AllocatorT;
    using value_type = ValueT;
    // must be signed integer, since we are using -1 as sentinel value
    using index_type = std::make_signed_t<Index>;
    using size_type = std::make_unsigned_t<Index>;
    static_assert(std::is_floating_point_v<value_type>, "value_type should be floating point");
    static_assert(std::is_integral_v<index_type>, "index_type should be an integral");

    using mc_t = conditional_t<is_same_v<value_type, f64>, u64, u32>;
    using Box = AABBBox<dim, value_type>;
    using TV = vec<value_type, dim>;
    using IV = vec<index_type, dim>;
    using bvs_t = Vector<Box, allocator_type>;
    using vector_t = Vector<value_type, allocator_type>;
    using indices_t = Vector<index_type, allocator_type>;

    constexpr decltype(auto) memoryLocation() const noexcept { return bvs.memoryLocation(); }
    constexpr ProcID devid() const noexcept { return bvs.devid(); }
    constexpr memsrc_e memspace() const noexcept { return bvs.memspace(); }
    decltype(auto) get_allocator() const noexcept { return bvs.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      return bvs.get_default_allocator(mre, devid);
    }

    LBvs() = default;

    LBvs clone(const allocator_type &allocator) const {
      LBvs ret{};
      ret.bvs = bvs.clone(allocator);
      ret.auxIndices = auxIndices.clone(allocator);
      ret.leafIndices = leafIndices.clone(allocator);
      return ret;
    }
    LBvs clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    constexpr auto getNumNodes() const noexcept { return bvs.size(); }
    constexpr auto getNumLeaves() const noexcept { return bvs.size(); }

    template <typename Policy>
    void build(Policy &&pol, const Vector<AABBBox<dim, value_type>> &primBvs) {
      constexpr auto space = std::remove_reference_t<Policy>::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
      // ZS_LAMBDA -> __device__
      static_assert(space == execspace_e::cuda, "specialized policy and compiler not match");
#else
      static_assert(space != execspace_e::cuda, "specialized policy and compiler not match");
#endif
      bvs = primBvs;
      Vector<Box> bv{primBvs.get_allocator(), 1};
      bv.setVal(Box{TV::init([](int) { return limits<value_type>::max(); }),
                    TV::init([](int) { return limits<value_type>::lowest(); })});
      pol(bvs, [gbv = proxy<space>(bv)] ZS_LAMBDA(const Box &bv) mutable {
        for (int d = 0; d != dim; ++d) atomic_min(wrapv<space>{}, &gbv[0]._min[d], bv._min[d]);
        for (int d = 0; d != dim; ++d) atomic_max(wrapv<space>{}, &gbv[0]._max[d], bv._max[d]);
      });
      gbv = bv.getVal();
    }
    template <typename Policy>
    void refit(Policy &&pol, const Vector<AABBBox<dim, value_type>> &primBvs) {
      constexpr auto space = std::remove_reference_t<Policy>::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
      // ZS_LAMBDA -> __device__
      static_assert(space == execspace_e::cuda, "specialized policy and compiler not match");
#else
      static_assert(space != execspace_e::cuda, "specialized policy and compiler not match");
#endif
      bvs = primBvs;
      Vector<Box> bv{primBvs.get_allocator(), 1};
      bv.setVal(Box{TV::init([](int) { return limits<value_type>::max(); }),
                    TV::init([](int) { return limits<value_type>::lowest(); })});
      pol(bvs, [gbv = proxy<space>(bv)] ZS_LAMBDA(const Box &bv) mutable {
        for (int d = 0; d != dim; ++d) atomic_min(wrapv<space>{}, &gbv[0]._min[d], bv._min[d]);
        for (int d = 0; d != dim; ++d) atomic_max(wrapv<space>{}, &gbv[0]._max[d], bv._max[d]);
      });
      gbv = bv.getVal();
    }

    bvs_t bvs;
    Box gbv;
    // escape index for internal nodes, primitive index for leaf nodes
    indices_t auxIndices;
    indices_t leafIndices;  // leaf indices within optimized lbvh
  };

  template <execspace_e, typename LBvsT, typename = void> struct LBvsView;

  /// proxy to work within each backends
  template <execspace_e space, typename LBvsT> struct LBvsView<space, const LBvsT> {
    static constexpr int dim = LBvsT::dim;
    static constexpr auto exectag = wrapv<space>{};
    using index_t = typename LBvsT::index_type;
    using bv_t = typename LBvsT::Box;
    using bvs_t = typename LBvsT::bvs_t;
    using indices_t = typename LBvsT::indices_t;

    constexpr LBvsView() = default;
    ~LBvsView() = default;

    explicit constexpr LBvsView(const LBvsT &lbvs)
        : _bvs{proxy<space>(lbvs.bvs)},
          _auxIndices{proxy<space>(lbvs.auxIndices)},
          _leafIndices{proxy<space>(lbvs.leafIndices)},
          _numNodes{static_cast<index_t>(lbvs.getNumNodes())} {}

    constexpr auto numNodes() const noexcept { return _numNodes; }
    constexpr auto numLeaves() const noexcept { return numNodes(); }

    constexpr bv_t getNodeBV(index_t node) const { return _bvs[node]; }

    // BV can be either VecInterface<VecT> or AABBBox<dim, T>
    template <typename BV, class F> constexpr void iter_neighbors(const BV &bv, F &&f) const {
      for (index_t i = 0; i != _numNodes; ++i) {
        if (overlaps(getNodeBV(i), bv)) f(i);
      }
      return;
    }

    VectorView<space, const bvs_t> _bvs;
    VectorView<space, const indices_t> _auxIndices, _leafIndices;
    index_t _numNodes;
  };

  template <execspace_e space, int dim, typename Ti, typename T, typename Allocator>
  constexpr decltype(auto) proxy(const LBvs<dim, Ti, T, Allocator> &lbvs) {
    return LBvsView<space, const LBvs<dim, Ti, T, Allocator>>{lbvs};
  }

  template <execspace_e space, typename LBvsT, typename BV, class F>
  constexpr void iter_neighbors(const LBvsView<space, const LBvsT> &bvs, const BV &bv, F &&f) {
    for (int i = 0; i != bvs._numNodes; ++i) {
      if (overlaps(bvs.getNodeBV(i), bv)) f(i);
    }
    return;
  }

}  // namespace zs
