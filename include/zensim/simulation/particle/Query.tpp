#include "zensim/container/IndexBuckets.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/simulation/sparsity/SparsityOp.hpp"

namespace zs {

  template <typename ExecPolicy>
  GeneralIndexBuckets index_buckets_for_particles(const ExecPolicy &execPol,
                                                  const GeneralParticles &particles, float dx,
                                                  float displacement) {
    return match([&execPol, dx, displacement](const auto &pars) -> GeneralIndexBuckets {
      using particles_t = remove_cvref_t<decltype(pars)>;
      using indexbuckets_t
          = IndexBuckets<particles_t::dim,
                         conditional_t<(sizeof(typename particles_t::size_type) > 4), i64, i32>,
                         int>;
      using vector_t = typename indexbuckets_t::vector_t;

      const auto &allocator = pars.get_allocator();

      indexbuckets_t indexBuckets{};
      indexBuckets._dx = dx;
      // table
      auto &table = indexBuckets._table;
      table = RM_CVREF_T(table){allocator, pars.size()};

      constexpr execspace_e space = RM_REF_T(execPol)::exec_tag::value;
      constexpr auto execTag = wrapv<space>{};
      execPol(range(table._tableSize), CleanSparsity{execTag, table});
      execPol(range(pars.size()),
              ComputeSparsity{execTag, dx, 1, table, pars.attrVector("x"), 0, displacement});
      /// counts, offsets, indices
      // counts
      auto &counts = indexBuckets._counts;
      auto numCells = table.size() + 1;

      counts = vector_t{allocator, (size_t)numCells};
      match([&counts](auto &&memTag) {
        memset(memTag, counts.data(), 0, sizeof(typename vector_t::value_type) * counts.size());
      })(counts.memoryLocation().getTag());

      auto tmp = counts;  // zero-ed array
      execPol(range(pars.size()),
              SpatiallyCount{execTag, dx, table, pars.attrVector("x"), counts, 1, 0, displacement});
      // offsets
      auto &offsets = indexBuckets._offsets;
      offsets = vector_t{allocator, (size_t)numCells};
      exclusive_scan(execPol, counts.begin(), counts.end(), offsets.begin());
      // fmt::print("scanned num particles: {}, num buckets: {} ({})\n", offsets[numCells - 1],
      //           numCells - 1, indexBuckets.numBuckets());
      // indices
      auto &indices = indexBuckets._indices;
      indices = vector_t{allocator, pars.size()};
      execPol(range(pars.size()), SpatiallyDistribute{execTag, dx, table, pars.attrVector("x"), tmp,
                                                      offsets, indices, 1, 0, displacement});
      return indexBuckets;
    })(particles);
  }

}  // namespace zs