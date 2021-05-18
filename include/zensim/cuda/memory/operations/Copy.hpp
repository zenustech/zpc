#pragma once
#include "zensim/cuda/Cuda.h"
#include "zensim/memory/operations/Copy.hpp"

namespace zs {

  void *allocate(device_mem_tag, std::size_t size, std::size_t alignment);
  void deallocate(device_mem_tag, void *ptr, std::size_t size, std::size_t alignment);
  void copy(device_mem_tag, void *dst, void *src, std::size_t size);

  void *allocate(device_const_mem_tag, std::size_t size, std::size_t alignment);
  void deallocate(device_const_mem_tag, void *ptr, std::size_t size, std::size_t alignment);
  void copy(device_const_mem_tag, void *dst, void *src, std::size_t size);

  void *allocate(um_mem_tag, std::size_t size, std::size_t alignment);
  void deallocate(um_mem_tag, void *ptr, std::size_t size, std::size_t alignment);
  void copy(um_mem_tag, void *dst, void *src, std::size_t size);
  void advise(um_mem_tag, std::string advice, void *addr, std::size_t bytes, ProcID did);

  template <> struct mem_copy<execspace_e::cuda> {
    void operator()(MemoryEntity dst, MemoryEntity src, std::size_t size) const {
      cudaMemcpy(dst.ptr, src.ptr, size, cudaMemcpyDefault);
    }
  };

}  // namespace zs