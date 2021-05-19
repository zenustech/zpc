#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/memory/MemoryResource.h"

namespace zs {

  void *allocate(host_mem_tag, std::size_t size, std::size_t alignment);
  void deallocate(host_mem_tag, void *ptr, std::size_t size, std::size_t alignment);
  void memset(host_mem_tag, void *addr, int chval, std::size_t size);
  void copy(host_mem_tag, void *dst, void *src, std::size_t size);

}  // namespace zs