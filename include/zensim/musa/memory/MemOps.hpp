#pragma once
#include "zensim/memory/MemoryResource.h"
#include "zensim/types/SourceLocation.hpp"

namespace zs {

  ZPC_BACKEND_API bool prepare_context(device_mem_tag, ProcID devid,
                                       const source_location &loc = source_location::current());
  ZPC_BACKEND_API void *allocate(device_mem_tag, size_t size, size_t alignment,
                                 const source_location &loc = source_location::current());
  ZPC_BACKEND_API void deallocate(device_mem_tag, void *ptr, size_t size, size_t alignment,
                                  const source_location &loc = source_location::current());
  ZPC_BACKEND_API void memset(device_mem_tag, void *addr, int chval, size_t size,
                              const source_location &loc = source_location::current());
  ZPC_BACKEND_API void copy(device_mem_tag, void *dst, void *src, size_t size,
                            const source_location &loc = source_location::current());
  ZPC_BACKEND_API void copyHtoD(device_mem_tag, void *dst, void *src, size_t size,
                                const source_location &loc = source_location::current());
  ZPC_BACKEND_API void copyDtoH(device_mem_tag, void *dst, void *src, size_t size,
                                const source_location &loc = source_location::current());
  ZPC_BACKEND_API void copyDtoD(device_mem_tag, void *dst, void *src, size_t size,
                                const source_location &loc = source_location::current());

  ZPC_BACKEND_API bool prepare_context(um_mem_tag, ProcID devid,
                                       const source_location &loc = source_location::current());
  ZPC_BACKEND_API void *allocate(um_mem_tag, size_t size, size_t alignment,
                                 const source_location &loc = source_location::current());
  ZPC_BACKEND_API void deallocate(um_mem_tag, void *ptr, size_t size, size_t alignment,
                                  const source_location &loc = source_location::current());
  ZPC_BACKEND_API void memset(um_mem_tag, void *addr, int chval, size_t size,
                              const source_location &loc = source_location::current());
  ZPC_BACKEND_API void copy(um_mem_tag, void *dst, void *src, size_t size,
                            const source_location &loc = source_location::current());
  ZPC_BACKEND_API void copyHtoD(um_mem_tag, void *dst, void *src, size_t size,
                                const source_location &loc = source_location::current());
  ZPC_BACKEND_API void copyDtoH(um_mem_tag, void *dst, void *src, size_t size,
                                const source_location &loc = source_location::current());
  ZPC_BACKEND_API void copyDtoD(um_mem_tag, void *dst, void *src, size_t size,
                                const source_location &loc = source_location::current());
  ZPC_BACKEND_API void advise(um_mem_tag, std::string advice, void *addr, size_t bytes, ProcID did,
                              const source_location &loc = source_location::current());

}  // namespace zs