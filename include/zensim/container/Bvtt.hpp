#pragma once
#include "Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename PrimIdT = size_t, typename NodeIdT = PrimIdT> struct BvttFront {
    using allocator_type = ZSPmrAllocator<>;
    using prim_id_t = PrimIdT;
    using node_id_t = NodeIdT;
    using prim_vector_t = Vector<prim_id_t>;
    using node_vector_t = Vector<node_id_t>;
    using index_t = zs::make_signed_t<math::op_result_t<prim_id_t, node_id_t>>;
    using counter_t = Vector<index_t>;

    constexpr decltype(auto) memoryLocation() const noexcept { return _cnt.memoryLocation(); }
    constexpr zs::ProcID devid() const noexcept { return _cnt.devid(); }
    constexpr zs::memsrc_e memspace() const noexcept { return _cnt.memspace(); }
    decltype(auto) get_allocator() const noexcept { return _cnt.get_allocator(); }
    decltype(auto) get_default_allocator(zs::memsrc_e mre, zs::ProcID devid) const {
      return _cnt.get_default_allocator(mre, devid);
    }

    BvttFront() = default;

    BvttFront clone(const allocator_type &allocator) const {
      BvttFront ret{};
      ret._primIds = _primIds.clone(allocator);
      ret._nodeIds = _nodeIds.clone(allocator);
      ret._offsets = _offsets.clone(allocator);
      ret._cnt = _cnt.clone(allocator);
      return ret;
    }
    BvttFront clone(const zs::MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    BvttFront(const allocator_type &allocator, node_id_t numNodes, index_t estimatedCount)
        : _primIds{allocator, (size_t)estimatedCount},
          _nodeIds{allocator, (size_t)estimatedCount},
          _offsets{allocator, (size_t)numNodes + 1},
          _cnt{allocator, 1} {
      _cnt.setVal(0);
    }
    BvttFront(node_id_t numNodes, index_t estimatedCount, memsrc_e mre = memsrc_e::host,
              ProcID devid = -1)
        : BvttFront{get_memory_source(mre, devid), numNodes, estimatedCount} {}

    auto size() const { return _cnt.getVal(0); }
    auto numNodes() const { return _offsets.size() - 1; }

    void setCounter(index_t newSize) { _cnt.setVal(newSize); }
    void getCounter(index_t newSize) const { _cnt.getVal(); }
    void reserve(index_t newSize) {
      if (newSize > _primIds.size()) {  // not to confuse OFB check
        _primIds.resize(newSize);
        _nodeIds.resize(newSize);
      }
    }

    template <typename Policy> void reorder(Policy &&policy);

    prim_vector_t _primIds;
    node_vector_t _nodeIds;
    counter_t _offsets;
    counter_t _cnt;
  };

  template <typename PrimIdT, typename NodeIdT> template <typename Policy>
  void BvttFront<PrimIdT, NodeIdT>::reorder(Policy &&policy) {
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    constexpr auto execTag = wrapv<space>{};
    using bvtt_t = BvttFront<PrimIdT, NodeIdT>;
    using index_t = typename bvtt_t::index_t;
    using counter_t = typename bvtt_t::counter_t;
    // count front nodes by node id
    const auto numFrontNodes = size();
    counter_t offsets{_cnt.get_allocator(), numNodes()};
    counter_t counts{_cnt.get_allocator(), numNodes()};
    counts.reset(0);
    auto frontView = proxy<space>(*this);
    fmt::print("{} current front nodes, {} num bvh nodes\n", numFrontNodes, numNodes());
    policy(range(numFrontNodes),
           [execTag, counts = proxy<space>(counts), front = frontView] ZS_LAMBDA(
               index_t i) mutable { atomic_add(execTag, &counts[front.node(i)], (index_t)1); });
    // scan
    exclusive_scan(policy, std::begin(counts), std::end(counts), std::begin(offsets));
    // reorder front
    auto primIds = _primIds;
    auto nodeIds = _nodeIds;
    counts.reset(0);
    policy(range(numFrontNodes), [counts = proxy<space>(counts), offsets = proxy<space>(offsets),
                                  primIds = proxy<space>(primIds), nodeIds = proxy<space>(nodeIds),
                                  front = frontView, execTag] ZS_LAMBDA(index_t i) mutable {
      auto nodeid = front.node(i);
      // auto loc = offsets[nodeid] + atomic_add(execTag, &counts[nodeid], (index_t)1);
      auto loc = offsets[nodeid] + atomic_add(execTag, &counts[nodeid], (index_t)1);
      primIds[loc] = front.prim(i);
      nodeIds[loc] = nodeid;
    });

    _primIds = std::move(primIds);
    _nodeIds = std::move(nodeIds);
  }

  template <execspace_e space, typename BvttFrontT, typename = void> struct BvttFrontView {
    static constexpr bool is_const_structure = is_const_v<BvttFrontT>;
    using prim_id_t = typename BvttFrontT::prim_id_t;
    using node_id_t = typename BvttFrontT::node_id_t;
    using prim_vector_t = typename BvttFrontT::prim_vector_t;
    using node_vector_t = typename BvttFrontT::node_vector_t;
    using index_t = typename BvttFrontT::index_t;
    using counter_t = typename BvttFrontT::counter_t;

    constexpr BvttFrontView() = default;
    ~BvttFrontView() = default;
    explicit constexpr BvttFrontView(BvttFrontT &bvfront)
        : _prims{bvfront._primIds.data()},
          _nodes{bvfront._nodeIds.data()},
          _cnt{bvfront._cnt.data()},
          _numFrontNodes{
              (index_t)std::min(bvfront._primIds.capacity(), bvfront._nodeIds.capacity())} {}

#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<
                  S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm, !V>
              = 0>
    __forceinline__ __device__ void push_back(prim_id_t prim, node_id_t node) {
      const auto no = atomic_add(wrapv<space>{}, _cnt, (index_t)1);
      if (no < _numFrontNodes) {
        _prims[no] = prim;
        _nodes[no] = node;
      }
    }
#endif
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<is_host_execution<S>(), !V> = 0>
    inline void push_back(prim_id_t prim, node_id_t node) {
      const auto no = atomic_add(wrapv<space>{}, _cnt, (index_t)1);
      if (no < _numFrontNodes) {
        _prims[no] = prim;
        _nodes[no] = node;
      }
    }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr void assign(index_t no, prim_id_t pid, node_id_t nid) {
      if (no < _numFrontNodes) {
        _prims[no] = pid;
        _nodes[no] = nid;
      } else {
        printf("bvtt front overflow! [%lld] exceeding cap [%lld]\n", (long long int)no,
               (long long int)_numFrontNodes);
      }
    }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr void assign(index_t no, node_id_t nid) {
      if (no < _numFrontNodes) {
        _nodes[no] = nid;
      } else {
        printf("bvtt front overflow! [%lld] exceeding cap [%lld]\n", (long long int)no,
               (long long int)_numFrontNodes);
      }
    }

    constexpr auto prim(index_t i) const {
      if (i < _numFrontNodes) return _prims[i];
      return ~(typename prim_vector_t::value_type)0;
    }
    constexpr auto node(index_t i) const {
      if (i < _numFrontNodes) return _nodes[i];
      return ~(typename node_vector_t::value_type)0;
    }

    conditional_t<is_const_structure, typename prim_vector_t::const_pointer,
                  typename prim_vector_t::pointer>
        _prims;
    conditional_t<is_const_structure, typename node_vector_t::const_pointer,
                  typename node_vector_t::pointer>
        _nodes;
    conditional_t<is_const_structure, typename counter_t::const_pointer,
                  typename counter_t::pointer>
        _cnt;
    const index_t _numFrontNodes;
  };

  template <execspace_e ExecSpace, typename PrimIdT, typename NodeIdT>
  decltype(auto) proxy(BvttFront<PrimIdT, NodeIdT> &front) {
    return BvttFrontView<ExecSpace, BvttFront<PrimIdT, NodeIdT>>{front};
  }
  template <execspace_e ExecSpace, typename PrimIdT, typename NodeIdT>
  decltype(auto) proxy(const BvttFront<PrimIdT, NodeIdT> &front) {
    return BvttFrontView<ExecSpace, const BvttFront<PrimIdT, NodeIdT>>{front};
  }

}  // namespace zs