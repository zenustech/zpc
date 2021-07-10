#pragma once
#include <driver_types.h>

#include <string>

namespace zs {

  struct LaunchConfig {
    template <typename IndexType0, typename IndexType1> LaunchConfig(IndexType0 gs, IndexType1 bs)
        : dg{static_cast<unsigned int>(gs)},
          db{static_cast<unsigned int>(bs)},
          shmem{0},
          sid{cudaStreamDefault} {}
    template <typename IndexType0, typename IndexType1, typename IndexType2>
    LaunchConfig(IndexType0 gs, IndexType1 bs, IndexType2 mem)
        : dg{static_cast<unsigned int>(gs)},
          db{static_cast<unsigned int>(bs)},
          shmem{static_cast<std::size_t>(mem)},
          sid{cudaStreamDefault} {}
    template <typename IndexType0, typename IndexType1, typename IndexType2>
    LaunchConfig(IndexType0 gs, IndexType1 bs, IndexType2 mem, cudaStream_t stream)
        : dg{static_cast<unsigned int>(gs)},
          db{static_cast<unsigned int>(bs)},
          shmem{static_cast<std::size_t>(mem)},
          sid{stream} {}
    dim3 dg{};
    dim3 db{};
    std::size_t shmem{0};
    cudaStream_t sid{cudaStreamDefault};
  };

}  // namespace zs
