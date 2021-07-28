#pragma once
#include "Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename Id1 = std::size_t, typename Id2 = Id1> struct BvttFront {
    using allocator_type = ZSPmrAllocator<>;
    BvttFront(std::size_t numPrims, std::size_t estimatedCount, memsrc_e mre = memsrc_e::host,
              ProcID devid = -1)
        : numPrims{numPrims},
          primIds{estimatedCount, mre, devid},
          nodeIds{estimatedCount, mre, devid},
          cnt{1, mre, devid} {}
#if 0
    BvttFront(const allocator_type &allocator, std::size_t estimatedCount,
              memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : numPrims{numPrims},
          primIds{allocator, estimatedCount, mre, devid},
          nodeIds{allocator, estimatedCount, mre, devid},
          cnt{allocator, 1, mre, devid} {}
#endif
    std::size_t numPrims;
    Vector<Id1> primIds;
    Vector<Id2> nodeIds;
    Vector<std::size_t> cnt;
  };

  template <execspace_e space, typename Id1, typename Id2>
  inline void reorder_bvtt_front(BvttFront<Id1, Id2> &front) {
    // count front nodes by prim id;
    // scan
    // reorder front
  }

}  // namespace zs