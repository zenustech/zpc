#include "Query.hpp"

#include "zensim/container/HashTable.hpp"
#include "zensim/cuda/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/simulation/sparsity/SparsityOp.hpp"

namespace zs {

  GeneralIndexBuckets build_neighbor_list_impl(cuda_exec_tag, const GeneralParticles &particles,
                                               float dx) {
    return match([](const auto &pars) -> GeneralIndexBuckets {
      using particles_t = remove_cvref_t<decltype(pars)>;
      static constexpr int dim = particles_t::dim;
      const auto memLoc = pars.space();
      const auto did = pars.devid();
      IndexBuckets<dim, 32> indexBuckets{};
      // table
      auto &table = indexBuckets._table;
      table = HashTable<int, dim, i64>{pars.size(), memLoc, did};

      if constexpr (false) {
        auto node = table.self().node();
        auto indicies = node.chsrc;
        auto chmap = node.chmap;
        for (int i = 0; i < 3; ++i) fmt::print("{}\t", chmap[i]);
        fmt::print("done chmap\n");
        fmt::print("ptr size: {} vs {} vs {}\n", sizeof(void *), sizeof(uintptr_t),
                   sizeof(std::uintptr_t));
        fmt::print("alignment: {} ", node.alignment());
        for (int i = 0; i < 3; ++i) fmt::print(", {}", indicies(i));
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
      cudaPol({table._tableSize}, CleanSparsity{wrapv<execspace_e::cuda>{}, table});
      //
      // getchar();

      return indexBuckets;
    })(particles);
  }

}  // namespace zs