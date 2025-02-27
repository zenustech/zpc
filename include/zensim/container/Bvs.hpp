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
    using index_type = zs::make_signed_t<Index>;
    using size_type = zs::make_unsigned_t<Index>;
    static_assert(is_floating_point_v<value_type>, "value_type should be floating point");
    static_assert(is_signed_v<index_type>, "index_type should be a signed integral");

    using mc_t = conditional_t<(sizeof(value_type) > sizeof(f64)), u64, u32>;
    using Box = AABBBox<dim, value_type>;
    using TV = vec<value_type, dim>;
    using IV = vec<index_type, dim>;
    // using bvs_t = Vector<Box, allocator_type>;
    using bvs_t = TileVector<value_type, 32, allocator_type>;
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
      ret.axis = axis;
      ret.sts = sts;
      ret.auxIndices = auxIndices.clone(allocator);
      return ret;
    }
    LBvs clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    constexpr auto getNumNodes() const noexcept { return bvs.size(); }
    constexpr auto getNumLeaves() const noexcept { return bvs.size(); }

    template <typename Policy> Box getBox(Policy &&pol) const {
      Box gbv;
      constexpr auto space = remove_reference_t<Policy>::exec_tag::value;
      Vector<value_type> ret{bvs.get_allocator(), 1};
      Vector<value_type> gmins{bvs.get_allocator(), bvs.size()},
          gmaxs{bvs.get_allocator(), bvs.size()};
      for (int d = 0; d != dim; ++d) {
        pol(enumerate(gmins, gmaxs), [bvs = view<space>(bvs), d] ZS_LAMBDA(
                                         index_type i, value_type & gmin, value_type & gmax) {
          gmin = bvs(d, i);
          gmax = bvs(dim + d, i);
        });
        reduce(pol, std::begin(gmins), std::end(gmins), std::begin(ret),
               detail::deduce_numeric_max<value_type>(), getmin<value_type>{});
        gbv._min[d] = ret.getVal();
        reduce(pol, std::begin(gmaxs), std::end(gmaxs), std::begin(ret),
               detail::deduce_numeric_lowest<value_type>(), getmax<value_type>{});
        gbv._max[d] = ret.getVal();
      }
      return gbv;
    }

    template <typename Policy> void build(Policy &&pol,
                                          const zs::Vector<zs::AABBBox<dim, value_type>> &primBvs,
                                          int axis = -1) {
      constexpr auto space = remove_reference_t<Policy>::exec_tag::value;
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
      // ZS_LAMBDA -> __device__
      static_assert(space == execspace_e::cuda, "specialized policy and compiler not match");
#elif ZS_ENABLE_MUSA && defined(__MUSACC__)
      static_assert(space == execspace_e::musa, "specialized policy and compiler not match");
#elif ZS_ENABLE_ROCM && defined(__HIPCC__)
      static_assert(space == execspace_e::rocm, "specialized policy and compiler not match");
#elif ZS_ENABLE_SYCL && defined(SYCL_LANGUAGE_VERSION)
      static_assert(space == execspace_e::sycl, "specialized policy and compiler not match");
#else
      static_assert(space == execspace_e::seq || space == execspace_e::openmp,
                    "specialized policy and compiler not match");
#endif

      // must call this in the beginning
      bvs = bvs_t{
          primBvs.get_allocator(), {{"min", dim}, {"max", dim}}, (size_t)range_size(primBvs)};

      if (!(axis >= 0 && axis < dim)) {
        // compute bvs and global bv
        pol(enumerate(primBvs), [bvs = view<space>(bvs)] ZS_LAMBDA(index_type i, auto bv) mutable {
          for (int d = 0; d != dim; ++d) {
            bvs(d, i) = bv._min[d];
            bvs(dim + d, i) = bv._max[d];
          }
        });
        auto gbv = getBox(pol);
        // pick the longest dimension
        axis = 0;  // x-axis by default
        auto dis = gbv._max[0] - gbv._min[0];
        for (int d = 1; d < dim; ++d) {
          if (auto tmp = gbv._max[d] - gbv._min[d]; tmp > dis) {  // select the longest orientation
            dis = tmp;
            axis = d;
          }
        }
      }

      this->axis = axis;

#if 1
      /// radix sort based
      vector_t keys{bvs.get_allocator(), bvs.size()}, sortedKeys{bvs.get_allocator(), bvs.size()};
      indices_t indices{bvs.get_allocator(), bvs.size()},
          sortedIndices{bvs.get_allocator(), bvs.size()};
      pol(enumerate(keys, indices),
          [primBvs = view<space>(primBvs), axis = axis] ZS_LAMBDA(index_type i, value_type & key,
                                                                  index_type & index) mutable {
            key = primBvs[i]._min[axis];
            index = i;
          });
      // sort in ascending order
      radix_sort_pair(pol, std::begin(keys), std::begin(indices), std::begin(sortedKeys),
                      std::begin(sortedIndices), bvs.size());
      sts = std::move(sortedKeys);
      auxIndices = std::move(sortedIndices);

#else
      /// merge sort based
      sts = vector_t{bvs.get_allocator(), bvs.size()};
      auxIndices = indices_t{bvs.get_allocator(), bvs.size()};
      pol(enumerate(sts, auxIndices),
          [primBvs = view<space>(primBvs), axis = axis] ZS_LAMBDA(index_type i, value_type & key,
                                                                  index_type & index) mutable {
            key = primBvs[i]._min[axis];
            index = i;
          });
      merge_sort_pair(pol, std::begin(sts), std::begin(auxIndices), bvs.size(),
                      std::less<value_type>{});

#endif
      pol(enumerate(auxIndices), [bvs = view<space>(bvs), primBvs = view<space>(primBvs)] ZS_LAMBDA(
                                     size_type i, size_type src) mutable {
        auto bv = primBvs[src];
        for (int d = 0; d != dim_; ++d) {
          bvs(d, i) = bv._min[d];
          bvs(dim_ + d, i) = bv._max[d];
        }
      });
    }

    template <typename Policy>
    void refit(Policy &&pol, const Vector<AABBBox<dim, value_type>> &primBvs) {
      // build(pol, primBvs);
      constexpr auto space = remove_reference_t<Policy>::exec_tag::value;
      if (range_size(primBvs) != bvs.size())
        throw std::runtime_error("the count of bounding volumes mismatch");
      if (!valid_memspace_for_execution(pol, primBvs.get_allocator()))
        throw std::runtime_error(
            "current memory location not compatible with the execution policy");

      pol(enumerate(sts, auxIndices),
          [primBvs = view<space>(primBvs), axis = axis] ZS_LAMBDA(
              size_type i, value_type & key, size_type src) { key = primBvs[src]._min[axis]; });
      merge_sort_pair(pol, std::begin(sts), std::begin(auxIndices), bvs.size(),
                      std::less<value_type>{});

      pol(enumerate(auxIndices), [bvs = view<space>(bvs), primBvs = view<space>(primBvs)] ZS_LAMBDA(
                                     size_type i, size_type src) mutable {
        auto bv = primBvs[src];
        for (int d = 0; d != dim_; ++d) {
          bvs(d, i) = bv._min[d];
          bvs(dim_ + d, i) = bv._max[d];
        }
      });
    }

    bvs_t bvs;
    int axis;
    vector_t sts;
    indices_t auxIndices;
  };

  template <execspace_e, typename LBvsT, typename = void> struct LBvsView;

  /// proxy to work within each backends
  template <execspace_e space, typename LBvsT> struct LBvsView<space, const LBvsT> {
    static constexpr bool is_const_structure = true;
    static constexpr int dim = LBvsT::dim;
    static constexpr auto exectag = wrapv<space>{};
    using value_type = typename LBvsT::value_type;
    using index_type = typename LBvsT::index_type;
    using size_type = typename LBvsT::size_type;
    using index_t = typename LBvsT::index_type;
    using bv_t = typename LBvsT::Box;
    using bvs_t = typename LBvsT::bvs_t;
    using bvs_view_type = decltype(view<space>(
        declval<conditional_t<is_const_structure, const bvs_t &, bvs_t &>>()));
    using vector_t = typename LBvsT::vector_t;
    using indices_t = typename LBvsT::indices_t;

    constexpr LBvsView() = default;
    ~LBvsView() = default;

    explicit constexpr LBvsView(const LBvsT &lbvs)
        : _bvs{view<space>(lbvs.bvs)},
          _auxIndices{proxy<space>(lbvs.auxIndices)},
          _axis{lbvs.axis},
          _numNodes{static_cast<index_t>(lbvs.getNumNodes())} {}

    constexpr auto numNodes() const noexcept { return _numNodes; }
    constexpr auto numLeaves() const noexcept { return numNodes(); }

    constexpr bv_t getNodeBV(index_t node) const {
      return bv_t{_bvs.pack(dim_c<dim>, 0, node), _bvs.pack(dim_c<dim>, dim, node)};
    }

    constexpr index_type locate(value_type v) const {
      // vs[i - 1] (if exist) < vs[i] <= v <= vs[i + 1]
      index_type left = 0, right = numNodes() - 1;
      while (left <= right) {
        auto mid = left + (right - left) / 2;
        if (_bvs(_axis, mid) > v)
          right = mid - 1;
        else
          left = mid + 1;
      }
      return right;
    }

    // BV can be either VecInterface<VecT> or AABBBox<dim, T>
    template <typename BV, class F> constexpr void iter_neighbors(const BV &bv, F &&f) const {
      auto ed = locate(bv._max[_axis]);
      for (index_t i = 0; i <= ed; ++i) {
        auto primid = _auxIndices[i];
        if (overlaps(getNodeBV(i), bv)) f(primid);
      }
      return;
    }
    template <class F> constexpr void self_iter_neighbors(index_t node, F &&f) const {
      auto bv = bv_t{_bvs.pack(dim_c<dim>, 0, node), _bvs.pack(dim_c<dim>, dim, node)};
      auto ed = locate(bv._max[_axis]);
      for (index_t i = node + 1; i <= ed; ++i) {
        auto primid = _auxIndices[i];
        if (overlaps(getNodeBV(i), bv)) f(primid);
      }
      return;
    }

    bvs_view_type _bvs;
    VectorView<space, const indices_t> _auxIndices;
    int _axis;
    index_t _numNodes;
  };

  template <execspace_e space, int dim, typename Ti, typename T, typename Allocator>
  decltype(auto) proxy(const LBvs<dim, Ti, T, Allocator> &lbvs) {
    return LBvsView<space, const LBvs<dim, Ti, T, Allocator>>{lbvs};
  }

  template <execspace_e space, typename LBvsT, typename BV, class F>
  constexpr void iter_neighbors(const LBvsView<space, const LBvsT> &bvs, const BV &bv, F &&f) {
    bvs.iter_neighbors(bv, FWD(f));
  }

}  // namespace zs
