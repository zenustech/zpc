#pragma once
#include "zensim/memory/MemoryResource.h"
#include "zensim/types/SourceLocation.hpp"

namespace zs {

  void *allocate(device_mem_tag, std::size_t size, std::size_t alignment,
                 const source_location &loc = source_location::current());
  void deallocate(device_mem_tag, void *ptr, std::size_t size, std::size_t alignment,
                  const source_location &loc = source_location::current());
  void memset(device_mem_tag, void *addr, int chval, std::size_t size,
              const source_location &loc = source_location::current());
  void copy(device_mem_tag, void *dst, void *src, std::size_t size,
            const source_location &loc = source_location::current());

  void *allocate(um_mem_tag, std::size_t size, std::size_t alignment,
                 const source_location &loc = source_location::current());
  void deallocate(um_mem_tag, void *ptr, std::size_t size, std::size_t alignment,
                  const source_location &loc = source_location::current());
  void memset(um_mem_tag, void *addr, int chval, std::size_t size,
              const source_location &loc = source_location::current());
  void copy(um_mem_tag, void *dst, void *src, std::size_t size,
            const source_location &loc = source_location::current());
  void advise(um_mem_tag, std::string advice, void *addr, std::size_t bytes, ProcID did,
              const source_location &loc = source_location::current());

}  // namespace zs