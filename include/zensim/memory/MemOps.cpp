#include "MemOps.hpp"

namespace zs {

  void *allocate(host_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    ret = std::malloc(size);
    return ret;
  }
  void deallocate(host_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    std::free(ptr);
  }
  void memset(host_mem_tag, void *addr, int chval, std::size_t size) {
    std::memset(addr, chval, size);
  }
  void copy(host_mem_tag, void *dst, void *src, std::size_t size) { std::memcpy(dst, src, size); }

}  // namespace zs