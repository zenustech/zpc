#pragma once
#include "zensim/memory/MemoryResource.h"
#include "zensim/types/SourceLocation.hpp"

namespace zs {

  bool prepare_context(device_mem_tag, ProcID devid,
                       const source_location &loc = source_location::current());
  void *allocate(device_mem_tag, size_t size, size_t alignment,
                 const source_location &loc = source_location::current());
  void deallocate(device_mem_tag, void *ptr, size_t size, size_t alignment,
                  const source_location &loc = source_location::current());
  void memset(device_mem_tag, void *addr, int chval, size_t size,
              const source_location &loc = source_location::current());
  void copy(device_mem_tag, void *dst, void *src, size_t size,
            const source_location &loc = source_location::current());
  void copyHtoD(device_mem_tag, void *dst, void *src, size_t size,
                const source_location &loc = source_location::current());
  void copyDtoH(device_mem_tag, void *dst, void *src, size_t size,
                const source_location &loc = source_location::current());
  void copyDtoD(device_mem_tag, void *dst, void *src, size_t size,
                const source_location &loc = source_location::current());

  bool prepare_context(um_mem_tag, ProcID devid,
                       const source_location &loc = source_location::current());
  void *allocate(um_mem_tag, size_t size, size_t alignment,
                 const source_location &loc = source_location::current());
  void deallocate(um_mem_tag, void *ptr, size_t size, size_t alignment,
                  const source_location &loc = source_location::current());
  void memset(um_mem_tag, void *addr, int chval, size_t size,
              const source_location &loc = source_location::current());
  void copy(um_mem_tag, void *dst, void *src, size_t size,
            const source_location &loc = source_location::current());
  void copyHtoD(um_mem_tag, void *dst, void *src, size_t size,
                const source_location &loc = source_location::current());
  void copyDtoH(um_mem_tag, void *dst, void *src, size_t size,
                const source_location &loc = source_location::current());
  void copyDtoD(um_mem_tag, void *dst, void *src, size_t size,
                const source_location &loc = source_location::current());
  void advise(um_mem_tag, std::string advice, void *addr, size_t bytes, ProcID did,
              const source_location &loc = source_location::current());

}  // namespace zs