#include "Copy.hpp"

#include "zensim/cuda/Cuda.h"

namespace zs {

  void *allocate(device_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    Cuda::instance().malloc(&ret, size);
    return ret;
  }

  void deallocate(device_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    Cuda::instance().free(ptr);
  }

  void copy(device_mem_tag, void *dst, void *src, std::size_t size) {
    Cuda::instance().memcpy(dst, src, size);
  }

  void *allocate(device_const_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    Cuda::instance().malloc(&ret, size);
    return ret;
  }
  void deallocate(device_const_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    Cuda::instance().free(ptr);
  }
  void copy(device_const_mem_tag, void *dst, void *src, std::size_t size);

  void *allocate(um_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    Cuda::instance().vmalloc(&ret, size);
    return ret;
  }
  void deallocate(um_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    Cuda::instance().free(ptr);
  }
  void copy(um_mem_tag, void *dst, void *src, std::size_t size) {
    Cuda::instance().memcpy(dst, src, size);
  }

}  // namespace zs