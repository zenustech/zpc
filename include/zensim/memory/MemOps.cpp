#include "MemOps.hpp"

namespace zs {

  void *allocate(host_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    ret = std::aligned_alloc(alignment, size);
    return ret;
  }
  void deallocate(host_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    std::free(ptr);
  }
  void memset(host_mem_tag, void *addr, int chval, std::size_t size) {
    std::memset(addr, chval, size);
  }
  void copy(host_mem_tag, void *dst, void *src, std::size_t size) { std::memcpy(dst, src, size); }

#if 0
  void *allocate_dispatch(mem_tags tag, std::size_t size, std::size_t alignment) {
    return match([size, alignment](auto &tag) { return allocate(tag, size, alignment); })(tag);
  }
  void deallocate_dispatch(mem_tags tag, void *ptr, std::size_t size, std::size_t alignment) {
    match([ptr, size, alignment](auto &tag) { deallocate(tag, ptr, size, alignment); })(tag);
  }
  void memset_dispatch(mem_tags tag, void *addr, int chval, std::size_t size) {
    match([addr, chval, size](auto &tag) { memset(tag, addr, chval, size); })(tag);
  }
  void copy_dispatch(mem_tags tag, void *dst, void *src, std::size_t size) {
    match([dst, src, size](auto &tag) { copy(tag, dst, src, size); })(tag);
  }
  void advise_dispatch(mem_tags tag, std::string advice, void *addr, std::size_t bytes,
                       ProcID did) {
    match([&advice, addr, bytes, did](auto &tag) { advise(tag, advice, addr, bytes, did); })(tag);
  }
#endif

}  // namespace zs