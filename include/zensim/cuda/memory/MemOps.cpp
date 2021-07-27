#include "MemOps.hpp"

#include "zensim/cuda/Cuda.h"

namespace zs {

  void *allocate(device_mem_tag, std::size_t size, std::size_t alignment,
                 const source_location &loc) {
    void *ret{nullptr};
    cudri::malloc(&ret, size, loc);
    return ret;
  }

  void deallocate(device_mem_tag, void *ptr, std::size_t size, std::size_t alignment,
                  const source_location &loc) {
    cudri::free{ptr, loc};
  }

  void memset(device_mem_tag, void *addr, int chval, std::size_t size, const source_location &loc) {
    cudri::memset(addr, chval, size, loc);
  }
  void copy(device_mem_tag, void *dst, void *src, std::size_t size, const source_location &loc) {
    cudri::memcpy(dst, src, size, loc);
  }

  void *allocate(um_mem_tag, std::size_t size, std::size_t alignment, const source_location &loc) {
    void *ret{nullptr};
    cudri::umalloc(&ret, size, 0x1, loc);  //(unsigned int)CU_MEM_ATTACH_GLOBAL);
    return ret;
  }
  void deallocate(um_mem_tag, void *ptr, std::size_t size, std::size_t alignment,
                  const source_location &loc) {
    cudri::free{ptr, loc};
  }
  void memset(um_mem_tag, void *addr, int chval, std::size_t size, const source_location &loc) {
    cudri::memset(addr, chval, size, loc);
  }
  void copy(um_mem_tag, void *dst, void *src, std::size_t size, const source_location &loc) {
    cudri::memcpy(dst, src, size, loc);  //, cudaMemcpyDefault);
  }
  void advise(um_mem_tag, std::string advice, void *addr, std::size_t bytes, ProcID did,
              const source_location &loc) {
    unsigned int option{};  // CUmem_advise
    if (advice == "ACCESSED_BY")
      option = 5;  // CU_MEM_ADVISE_SET_ACCESSED_BY;
    else if (advice == "PREFERRED_LOCATION")
      option = 3;  // CU_MEM_ADVISE_SET_PREFERRED_LOCATION;
    else if (advice == "READ_MOSTLY")
      option = 1;  // CU_MEM_ADVISE_SET_READ_MOSTLY;
    else
      throw std::runtime_error(
          fmt::format("advise(tag um_mem_tag, advice {}, addr {}, bytes {}, devid {})\n", advice,
                      addr, bytes, (int)did));
    if (Cuda::context(did).supportConcurrentUmAccess)
      cudri::memAdvise(addr, bytes, (unsigned int)option, (int)did, loc);
  }

}  // namespace zs