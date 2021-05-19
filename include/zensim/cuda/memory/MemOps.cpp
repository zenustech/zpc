#include "MemOps.hpp"

#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/CudaConstants.inc"

namespace zs {

  void *allocate(device_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    Cuda::instance().malloc(&ret, size);
    return ret;
  }

  void deallocate(device_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    Cuda::instance().free(ptr);
  }

  void memset(device_mem_tag, void *addr, int chval, std::size_t size) {
    Cuda::instance().memset(addr, chval, size);
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
  void memset(device_const_mem_tag, void *addr, int chval, std::size_t size) {
    Cuda::instance().memset(addr, chval, size);
  }
  void copy(device_const_mem_tag, void *dst, void *src, std::size_t size);

  void *allocate(um_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    Cuda::instance().vmalloc(&ret, size, CU_MEM_ATTACH_GLOBAL);
    return ret;
  }
  void deallocate(um_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    Cuda::instance().free(ptr);
  }
  void memset(um_mem_tag, void *addr, int chval, std::size_t size) {
    Cuda::instance().memset(addr, chval, size);
  }
  void copy(um_mem_tag, void *dst, void *src, std::size_t size) {
    Cuda::instance().memcpy(dst, src, size);
  }
  void advise(um_mem_tag, std::string advice, void *addr, std::size_t bytes, ProcID did) {
    uint32_t option;
    if (advice == "ACCESSED_BY")
      option = CU_MEM_ADVISE_SET_ACCESSED_BY;
    else if (advice == "PREFERRED_LOCATION")
      option = CU_MEM_ADVISE_SET_ACCESSED_BY;
    else if (advice == "READ_MOSTLY")
      option = CU_MEM_ADVISE_SET_READ_MOSTLY;
    else
      throw std::runtime_error(
          fmt::format("advise(tag um_mem_tag, advice {}, addr {}, bytes {}, devid {})\n", advice,
                      reinterpret_cast<std::uintptr_t>(addr), bytes, (int)did));
    Cuda::driver().memAdvise(addr, bytes, option, (int)did);
  }

}  // namespace zs