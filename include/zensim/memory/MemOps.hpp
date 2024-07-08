#pragma once
#include "zensim/memory/MemoryResource.h"
#include "zensim/types/SourceLocation.hpp"
#if ZS_ENABLE_CUDA
#  include "zensim/cuda/memory/MemOps.hpp"
#endif

namespace zs {

  ZPC_CORE_API void *allocate(host_mem_tag, size_t size, size_t alignment,
                              const source_location &loc = source_location::current());
  ZPC_CORE_API void deallocate(host_mem_tag, void *ptr, size_t size, size_t alignment,
                               const source_location &loc = source_location::current());
  ZPC_CORE_API void memset(host_mem_tag, void *addr, int chval, size_t size,
                           const source_location &loc = source_location::current());
  ZPC_CORE_API void copy(host_mem_tag, void *dst, void *src, size_t size,
                         const source_location &loc = source_location::current());

#if 0
  /// dispatch mem op calls
  void *allocate_dispatch(mem_tags tag, size_t size, size_t alignment);
  void deallocate_dispatch(mem_tags tag, void *ptr, size_t size, size_t alignment);
  void memset_dispatch(mem_tags tag, void *addr, int chval, size_t size);
  void copy_dispatch(mem_tags tag, void *dst, void *src, size_t size);
  void advise_dispatch(mem_tags tag, std::string advice, void *addr, size_t bytes, ProcID did);
#endif

  /// default memory operation implementations (fallback)
  template <typename MemTag> bool prepare_context(MemTag, ProcID) { return true; }
  template <typename MemTag> void *allocate(MemTag, size_t size, size_t alignment) {
    throw std::runtime_error(
        fmt::format("allocate(tag {}, size {}, alignment {}) not implemented\n",
                    get_memory_tag_name(MemTag{}), size, alignment));
  }
  template <typename MemTag> void deallocate(MemTag, void *ptr, size_t size, size_t alignment) {
    throw std::runtime_error(fmt::format(
        "deallocate(tag {}, ptr {}, size {}, alignment {}) not implemented\n",
        get_memory_tag_name(MemTag{}), reinterpret_cast<std::uintptr_t>(ptr), size, alignment));
  }
  template <typename MemTag> void memset(MemTag, void *addr, int chval, size_t size) {
    throw std::runtime_error(fmt::format(
        "memset(tag {}, ptr {}, charval {}, size {}) not implemented\n",
        get_memory_tag_name(MemTag{}), reinterpret_cast<std::uintptr_t>(addr), chval, size));
  }
  template <typename MemTag> void copy(MemTag, void *dst, void *src, size_t size) {
    throw std::runtime_error(fmt::format(
        "copy(tag {}, dst {}, src {}, size {}) not implemented\n", get_memory_tag_name(MemTag{}),
        reinterpret_cast<std::uintptr_t>(dst), reinterpret_cast<std::uintptr_t>(src), size));
  }
  template <typename MemTag> void copyHtoD(MemTag, void *dst, void *src, size_t size) {
    throw std::runtime_error(
        fmt::format("copyHtoD(tag {}, dst {}, src {}, size {}) not implemented\n",
                    get_memory_tag_name(MemTag{}), reinterpret_cast<std::uintptr_t>(dst),
                    reinterpret_cast<std::uintptr_t>(src), size));
  }
  template <typename MemTag> void copyDtoH(MemTag, void *dst, void *src, size_t size) {
    throw std::runtime_error(
        fmt::format("copyDtoH(tag {}, dst {}, src {}, size {}) not implemented\n",
                    get_memory_tag_name(MemTag{}), reinterpret_cast<std::uintptr_t>(dst),
                    reinterpret_cast<std::uintptr_t>(src), size));
  }
  template <typename MemTag> void copyDtoD(MemTag, void *dst, void *src, size_t size) {
    throw std::runtime_error(
        fmt::format("copyDtoD(tag {}, dst {}, src {}, size {}) not implemented\n",
                    get_memory_tag_name(MemTag{}), reinterpret_cast<std::uintptr_t>(dst),
                    reinterpret_cast<std::uintptr_t>(src), size));
  }
  template <typename MemTag, typename... Args>
  void advise(MemTag, std::string advice, void *addr, Args...) {
    throw std::runtime_error(
        fmt::format("advise(tag {}, advise {}, addr {}) with {} args not implemented\n",
                    get_memory_tag_name(MemTag{}), advice, reinterpret_cast<std::uintptr_t>(addr),
                    sizeof...(Args)));
  }

}  // namespace zs