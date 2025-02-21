#include "MemOps.hpp"

#include <musa.h>
#include <musa_runtime.h>

#include <iostream>

#include "zensim/Logger.hpp"
#include "zensim/musa/Musa.h"

namespace zs {

  bool prepare_context(device_mem_tag, ProcID did, const source_location &loc) {
    MUcontext ctx = nullptr;
    auto ec = muCtxGetCurrent(&ctx);
    if (ec != MUSA_SUCCESS) {
      const char *errString = nullptr;
      muGetErrorString(ec, &errString);
      checkMuApiError((u32)ec, loc, "[muCtxGetCurrent]", errString);
      return false;
    } else {
      if (did < 0) did = Musa::get_default_device();
      int devid = did;
      if (ctx != NULL) {
        auto ec = muCtxGetDevice(&devid);
        if (ec != MUSA_SUCCESS) {
          const char *errString = nullptr;
          muGetErrorString(ec, &errString);
          checkMuApiError((u32)ec, loc, "[muCtxGetDevice]", errString);
          return false;
        }
      }  // otherwise, no context has been initialized yet.

      if (ctx == NULL || devid != did) {
        ZS_WARN(fmt::format("context switching during (de)allocation of [tag [{}] @ device [{}]]",
                            get_memory_tag_name(device_mem_tag{}), (int)did));
        if (did < Musa::device_count() && did >= 0)
          Musa::context(did).setContext();
        else
          throw std::runtime_error(
              fmt::format("current binding device [{}] does not match the expected [{}] and failed "
                          "to switch context.",
                          devid, (int)did));
      }
    }
    return true;
  }
  void *allocate(device_mem_tag, size_t size, size_t alignment, const source_location &loc) {
    void *ret{nullptr};
    [[maybe_unused]] auto ec = muMemAlloc((MUdeviceptr *)&ret, size);
#if ZS_ENABLE_OFB_ACCESS_CHECK
    musaDeviceSynchronize();
    if (ec != MUSA_SUCCESS) {
      const char *errString = nullptr;
      muGetErrorString(ec, &errString);
      const auto fileInfo = fmt::format("# File: \"{:<50}\"", loc.file_name());
      const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
      const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());
      int devid;
      muCtxGetDevice(&devid);
      std::cerr << fmt::format(
          "\nMusa Error on Device {}: muMemAlloc failed: {} (size: {} bytes, alignment: {} "
          "bytes)\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
          devid, errString, size, alignment, " musa driver api error location ", fileInfo, locInfo,
          funcInfo, "=");
    }
#endif
    return ret;
  }

  void deallocate(device_mem_tag, void *ptr, size_t size, size_t alignment,
                  const source_location &loc) {
    muMemFree((MUdeviceptr)ptr);
  }

  void memset(device_mem_tag, void *addr, int chval, size_t size, const source_location &loc) {
    /// @note this is asynchronous with respect to host, cuz of using 'per-thread' default stream!!!
    muMemsetD8((MUdeviceptr)addr, chval, size);
  }
  void copy(device_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    muMemcpy((MUdeviceptr)dst, (MUdeviceptr)src, size);
  }
  void copyHtoD(device_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    muMemcpyHtoD((MUdeviceptr)dst, (void *)src, size);
  }
  void copyDtoH(device_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    muMemcpyDtoH((void *)dst, (MUdeviceptr)src, size);
  }
  void copyDtoD(device_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    muMemcpyDtoD((MUdeviceptr)dst, (MUdeviceptr)src, size);
  }

  bool prepare_context(um_mem_tag, ProcID did, const source_location &loc) {
    MUcontext ctx = nullptr;
    auto ec = muCtxGetCurrent(&ctx);
    if (ec != MUSA_SUCCESS) {
      const char *errString = nullptr;
      muGetErrorString(ec, &errString);
      checkMuApiError((u32)ec, loc, "[muCtxGetCurrent]", errString);
      return false;
    } else {
      if (did < 0) did = Musa::get_default_device();
      int devid = did;
      if (ctx != NULL) {
        auto ec = muCtxGetDevice(&devid);
        if (ec != MUSA_SUCCESS) {
          const char *errString = nullptr;
          muGetErrorString(ec, &errString);
          checkMuApiError((u32)ec, loc, "[muCtxGetDevice]", errString);
          return false;
        }
      }  // otherwise, no context has been initialized yet.

      if (ctx == NULL || devid != did) {
        ZS_WARN(fmt::format("context switching during (de)allocation of [tag [{}] @ device [{}]]",
                            get_memory_tag_name(um_mem_tag{}), (int)did));
        if (did < Musa::device_count() && did >= 0)
          Musa::context(did).setContext();
        else
          throw std::runtime_error(
              fmt::format("current binding device [{}] does not match the expected [{}] and failed "
                          "to switch context.",
                          devid, (int)did));
      }
    }
    return true;
  }
  void *allocate(um_mem_tag, size_t size, size_t alignment, const source_location &loc) {
    void *ret{nullptr};
    [[maybe_unused]] auto ec = muMemAllocManaged((MUdeviceptr *)&ret, size, MU_MEM_ATTACH_GLOBAL);
#if ZS_ENABLE_OFB_ACCESS_CHECK
    musaDeviceSynchronize();
    if (ec != MUSA_SUCCESS) {
      const char *errString = nullptr;
      muGetErrorString(ec, &errString);
      const auto fileInfo = fmt::format("# File: \"{:<50}\"", loc.file_name());
      const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
      const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());
      int devid;
      muCtxGetDevice(&devid);
      std::cerr << fmt::format(
          "\nMusa Error on Device {}: muMemAllocManaged failed: {} (size: {} bytes, alignment: {} "
          "bytes)\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
          devid, errString, size, alignment, " musa driver api error location ", fileInfo, locInfo,
          funcInfo, "=");
    }
#endif
    return ret;
  }
  void deallocate(um_mem_tag, void *ptr, size_t size, size_t alignment,
                  const source_location &loc) {
    muMemFree((MUdeviceptr)ptr);
  }
  void memset(um_mem_tag, void *addr, int chval, size_t size, const source_location &loc) {
    muMemsetD8((MUdeviceptr)addr, chval, size);
  }
  void copy(um_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    muMemcpy((MUdeviceptr)dst, (MUdeviceptr)src, size);
  }
  void copyHtoD(um_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    muMemcpyHtoD((MUdeviceptr)dst, (void *)src, size);
  }
  void copyDtoH(um_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    muMemcpyDtoH((void *)dst, (MUdeviceptr)src, size);
  }
  void copyDtoD(um_mem_tag, void *dst, void *src, size_t size, const source_location &loc) {
    muMemcpyDtoD((MUdeviceptr)dst, (MUdeviceptr)src, size);
  }
  void advise(um_mem_tag, std::string advice, void *addr, size_t bytes, ProcID did,
              const source_location &loc) {
    unsigned int option{};  // MUmem_advise
    if (advice == "ACCESSED_BY")
      option = 5;  // MU_MEM_ADVISE_SET_ACCESSED_BY;
    else if (advice == "PREFERRED_LOCATION")
      option = 3;  // MU_MEM_ADVISE_SET_PREFERRED_LOCATION;
    else if (advice == "READ_MOSTLY")
      option = 1;  // MU_MEM_ADVISE_SET_READ_MOSTLY;
    else
      throw std::runtime_error(
          fmt::format("advise(tag um_mem_tag, advice {}, addr {}, bytes {}, devid {})\n", advice,
                      addr, bytes, (int)did));
    if (Musa::context(did).supportConcurrentUmAccess)
      if (bytes > 0) muMemAdvise((MUdeviceptr)addr, bytes, (MUmem_advise)option, (MUdevice)did);
  }

}  // namespace zs