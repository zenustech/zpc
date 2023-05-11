#include "MemOps.hpp"

#include <cuda.h>

#include "zensim/Logger.hpp"
#include "zensim/cuda/Cuda.h"

namespace zs {

  bool prepare_context(device_mem_tag, ProcID did) {
    int devid;
    cuCtxGetDevice(&devid);
    if (devid != did) {
      ZS_WARN(fmt::format("context switching during (de)allocation of [tag [{}] @ device [{}]]",
                          get_memory_tag_name(device_mem_tag{}), (int)did));
      if (did < Cuda::device_count() && did >= 0)
        Cuda::context(did).setContext();
      else
        throw std::runtime_error(
            fmt::format("current binding device [{}] does not match the expected [{}] and failed "
                        "to switch context.",
                        devid, (int)did));
    }
    return true;
  }
  void *allocate(device_mem_tag, size_t size, size_t alignment,
                 const source_location &loc) {
    void *ret{nullptr};
    cuMemAlloc((CUdeviceptr *)&ret, size);
    return ret;
  }

  void deallocate(device_mem_tag, void *ptr, size_t size, size_t alignment,
                  const source_location &loc) {
    cuMemFree((CUdeviceptr)ptr);
  }

  void memset(device_mem_tag, void *addr, int chval, size_t size, const source_location &loc) {
    /// @note this is asynchronous with respect to host, cuz of using 'per-thread' default stream!!!
    cuMemsetD8((CUdeviceptr)addr, chval, size);
    // cudaDeviceSynchronize();
  }
  void copy(device_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    /// @note
    /// https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
    cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, size);
    // cudaDeviceSynchronize();
  }
  void copyHtoD(device_mem_tag, void *dst, void *src, size_t size,
                const source_location &loc) {
    cuMemcpyHtoD((CUdeviceptr)dst, (void *)src, size);
    // cudaDeviceSynchronize();
  }
  void copyDtoH(device_mem_tag, void *dst, void *src, size_t size,
                const source_location &loc) {
    cuMemcpyDtoH((void *)dst, (CUdeviceptr)src, size);
    // cudaDeviceSynchronize();
  }
  void copyDtoD(device_mem_tag, void *dst, void *src, size_t size,
                const source_location &loc) {
    cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, size);
    // cudaDeviceSynchronize();
  }

  bool prepare_context(um_mem_tag, ProcID did) {
    int devid;
    cuCtxGetDevice(&devid);
    if (devid != did) {
      ZS_WARN(fmt::format("context switching during (de)allocation of [tag [{}] @ device [{}]]",
                          get_memory_tag_name(um_mem_tag{}), (int)did));
      if (did < Cuda::device_count() && did >= 0)
        Cuda::context(did).setContext();
      else
        throw std::runtime_error(
            fmt::format("current binding device [{}] does not match the expected [{}] and failed "
                        "to switch context.",
                        devid, (int)did));
    }
    return true;
  }
  void *allocate(um_mem_tag, size_t size, size_t alignment, const source_location &loc) {
    void *ret{nullptr};
    // cudri::umalloc(&ret, size, 0x1, loc);  //(unsigned int)CU_MEM_ATTACH_GLOBAL);
    cuMemAllocManaged((CUdeviceptr *)&ret, size, CU_MEM_ATTACH_GLOBAL);
    return ret;
  }
  void deallocate(um_mem_tag, void *ptr, size_t size, size_t alignment,
                  const source_location &loc) {
    cuMemFree((CUdeviceptr)ptr);
  }
  void memset(um_mem_tag, void *addr, int chval, size_t size, const source_location &loc) {
    cuMemsetD8((CUdeviceptr)addr, chval, size);
  }
  void copy(um_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, size);
  }
  void copyHtoD(um_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    cuMemcpyHtoD((CUdeviceptr)dst, (void *)src, size);
  }
  void copyDtoH(um_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    cuMemcpyDtoH((void *)dst, (CUdeviceptr)src, size);
  }
  void copyDtoD(um_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, size);
  }
  void advise(um_mem_tag, std::string advice, void *addr, size_t bytes, ProcID did,
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
      if (bytes > 0)
        // cudri::memAdvise(addr, bytes, (unsigned int)option, (int)did);
        cuMemAdvise((CUdeviceptr)addr, bytes, (CUmem_advise)option, (CUdevice)did);
  }

}  // namespace zs