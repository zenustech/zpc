#include "Query.hpp"

#include "zensim/container/HashTable.hpp"
#include "zensim/cuda/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/memory/MemoryResource.h"
#include "zensim/simulation/sparsity/SparsityOp.hpp"

namespace zs {

  GeneralIndexBuckets build_neighbor_list_impl(cuda_exec_tag, const GeneralParticles &particles,
                                               float dx) {
    return match([dx](const auto &pars) -> GeneralIndexBuckets {
      using particles_t = remove_cvref_t<decltype(pars)>;
      static constexpr int dim = particles_t::dim;
      using indexbuckets_t = IndexBuckets<dim, 32>;
      using vector_t = typename indexbuckets_t::vector_t;
      using table_t = typename indexbuckets_t::table_t;
      const auto memLoc = pars.space();
      const auto did = pars.devid();

      indexbuckets_t indexBuckets{};
      indexBuckets._dx = dx;
      // table
      auto &table = indexBuckets._table;
      table = {pars.size(), memLoc, did};

      if constexpr (true) {
        auto node = table.self().node();
        auto indices = node.chsrc;
        auto chmap = node.chmap;
        for (int i = 0; i < 3; ++i) fmt::print("{}\t", chmap[i]);
        fmt::print("done chmap\n");
        fmt::print("ptr size: {} vs {} vs {}\n", sizeof(void *), sizeof(uintptr_t),
                   sizeof(std::uintptr_t));
        fmt::print("alignment: {} ", node.alignment());
        for (int i = 0; i < 3; ++i) fmt::print(", {}", indices(i));
        fmt::print("\ntableSize: {}, totalBytes: {}, eleStride<1>: {}, eleSize: {}\n",
                   table._tableSize, node.size(), node.template element_stride<1>(),
                   node.element_size());

        {
          std::size_t offsets[3]
              = {node.template channel_offset<0>(), node.template channel_offset<1>(),
                 node.template channel_offset<2>()};
          fmt::print("channel offset: \n");
          for (int i = 0; i < 3; ++i) fmt::print("{}\t", offsets[i]);
          fmt::print("\n");
        }
        {
          std::size_t strides[3]
              = {node.template element_stride<0>(), node.template element_stride<1>(),
                 node.template element_stride<2>()};
          fmt::print("stride: \n");
          for (int i = 0; i < 3; ++i) fmt::print("{}\t", strides[i]);
          fmt::print("\n");
        }
      }

      auto cudaPol = cuda_exec().device(did).sync(true);
      cudaPol({table._tableSize}, CleanSparsity{exec_cuda, table});
      cudaPol({pars.size()},
              ComputeSparsity{exec_cuda, dx, 1, table, const_cast<particles_t &>(pars).X, -1});
      cudaPol({table.size()}, EnlargeSparsity{exec_cuda, table, table_t::key_t ::uniform(0),
                                              table_t::key_t ::uniform(3)});
      /// counts, offsets, indices
      // counts
      auto &counts = indexBuckets._counts;
      auto numCells = table.size();
      counts = vector_t{(std::size_t)numCells, memLoc, did};
      memset(mem_device, counts.data(), 0, sizeof(typename vector_t::value_type) * counts.size());
      auto tmp = counts;  // zero-ed array
      cudaPol({pars.size()},
              SpatiallyCount{exec_cuda, dx, table, const_cast<particles_t &>(pars).X, counts, 0});
      // offsets
      auto &offsets = indexBuckets._offsets;
      offsets = vector_t{(std::size_t)numCells, memLoc, did};
      exclusive_scan(cudaPol, counts.begin(), counts.end(), offsets.begin());
      // indices
      auto &indices = indexBuckets._indices;
      indices = vector_t{pars.size(), memLoc, did};
      cudaPol({pars.size()}, SpatiallyCount{exec_cuda, dx, table, const_cast<particles_t &>(pars).X,
                                            tmp, offsets, indices});
      return indexBuckets;
    })(particles);
  }

}  // namespace zs