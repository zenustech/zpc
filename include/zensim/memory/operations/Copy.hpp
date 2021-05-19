#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/memory/MemoryResource.h"

namespace zs {

  void *allocate(host_mem_tag, std::size_t size, std::size_t alignment);
  void deallocate(host_mem_tag, void *ptr, std::size_t size, std::size_t alignment);
  void memset(host_mem_tag, void *addr, int chval, std::size_t size);
  void copy(host_mem_tag, void *dst, void *src, std::size_t size);

  template <execspace_e exec, typename = void> struct mem_copy {
    void operator()(MemoryEntity dst, MemoryEntity src, std::size_t size) const {
      throw std::runtime_error(fmt::format(
          "copy operation backend {} for [{}, {}, {}] -> [{}, {}, {}] not implemented\n",
          get_execution_space_tag(exec), src.descr.memSpaceName(), (int)src.descr.devid(),
          (std::uintptr_t)src.ptr, dst.descr.memSpaceName(), (int)dst.descr.devid(),
          (std::uintptr_t)dst.ptr));
    }
  };

  template <execspace_e exec> struct mem_copy<
      exec, void_t<std::enable_if_t<exec == execspace_e::host || exec == execspace_e::openmp>>> {
    void operator()(MemoryEntity dst, MemoryEntity src, std::size_t size) const {
      memcpy(dst.ptr, src.ptr, size);
    }
  };

}  // namespace zs