#include <zensim/tpls/fmt/color.h>
#include <zensim/tpls/fmt/core.h>
#include <zensim/tpls/fmt/format.h>

#include "Allocators.cuh"
#include "Cuda.h"
#include "CudaConstants.inc"

namespace zs {

  /// memory resources
  void *device_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment) {
    void *ret;
    Cuda::driver().malloc(&ret, bytes);
    return ret;
  }
  void device_memory_resource::do_deallocate(void *p, std::size_t bytes, std::size_t alignment) {
    Cuda::driver().free(p);
  }
  bool device_memory_resource::do_is_equal(const mr_t &other) const noexcept {
    return this == dynamic_cast<device_memory_resource *>(const_cast<mr_t *>(&other));
  }

  void *unified_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment) {
    void *ret;
    Cuda::driver().vmalloc(&ret, bytes, CU_MEM_ATTACH_GLOBAL);
    return ret;
  }
  void unified_memory_resource::do_deallocate(void *p, std::size_t bytes, std::size_t alignment) {
    Cuda::driver().free(p);
  }
  bool unified_memory_resource::do_is_equal(const mr_t &other) const noexcept {
    return this == dynamic_cast<unified_memory_resource *>(const_cast<mr_t *>(&other));
  }

  void *dedicated_unified_allocator::allocate(std::size_t bytes, std::size_t align) {
    void *ptr = resource()->allocate(bytes, align);
/// avoid page faults as much as possible
#if 1
    Cuda::driver().memAdvise(ptr, bytes, (uint32_t)CU_MEM_ADVISE_SET_PREFERRED_LOCATION, devid);
#else
    Cuda::driver().memAdvise(ptr, bytes, (uint32_t)CU_MEM_ADVISE_SET_ACCESSED_BY, devid);
#endif
    return ptr;
  }

  /// customized memory allocators
  MonotonicAllocator::MonotonicAllocator(std::size_t totalMemBytes, std::size_t textureAlignBytes)
      : stack_allocator{&device_memory_resource::instance(), totalMemBytes, textureAlignBytes} {
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::dark_sea_green),
               "device memory allocator alignment (Bytes): {}\tsize (MB): {}\n", textureAlignBytes,
               totalMemBytes / 1024.0 / 1024.0);
  }
  auto MonotonicAllocator::borrow(std::size_t bytes) -> void * { return allocate(bytes); }
  void MonotonicAllocator::reset() {
    std::size_t usedBytes = _head - _data;
    std::size_t totalBytes = _tail - _data;
    if (usedBytes >= totalBytes * 3 / 4) {
      this->resource()->deallocate((void *)this->_data, totalBytes);
      std::size_t totalMemBytes = totalBytes * 3 / 2;
      this->_data = (char *)(this->resource()->allocate(totalMemBytes));
      this->_head = this->_data;
      this->_tail = this->_data + totalMemBytes;
    } else {
      this->_head = this->_data;
    }
  }

  MonotonicVirtualAllocator::MonotonicVirtualAllocator(int devId, std::size_t totalMemBytes,
                                                       std::size_t textureAlignBytes)
      : stack_allocator{&unified_memory_resource::instance(), totalMemBytes, textureAlignBytes} {
    Cuda::driver().memAdvise(_data, totalMemBytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, devId);
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::dark_sea_green),
               "unified memory allocator alignment: {} bytes\tsize: {} MB\n", textureAlignBytes,
               totalMemBytes / 1024.0 / 1024.0);
  }
  auto MonotonicVirtualAllocator::borrow(std::size_t bytes) -> void * { return allocate(bytes); }
  void MonotonicVirtualAllocator::reset() {
    std::size_t usedBytes = _head - _data;
    std::size_t totalBytes = _tail - _data;
    if (usedBytes >= totalBytes * 3 / 4) {
      this->resource()->deallocate((void *)this->_data, totalBytes);
      std::size_t totalMemBytes = totalBytes * 3 / 2;
      this->_data = (char *)(this->resource()->allocate(totalMemBytes));
      this->_head = this->_data;
      this->_tail = this->_data + totalMemBytes;
    } else {
      this->_head = this->_data;
    }
  }

}  // namespace zs
