#include "Musa.h"
//
#include <musa.h>
#include <musa_runtime.h>

#include "zensim/musa/memory/MemOps.hpp"

namespace zs {

  // error handling
  u32 Musa::get_last_musa_rt_error() { return (u32)musaPeekAtLastError(); }

  std::string_view Musa::get_musa_rt_error_string(u32 errorCode) {
    return musaGetErrorString((musaError_t)errorCode);
  }
  void Musa::check_musa_rt_error(u32 errorCode, ProcID did, const source_location &loc) {
    if (errorCode != 0) {
      if (did >= 0) {
        auto &context = Musa::context(did);
        if (context.errorStatus) return;  // there already exists a preceding musa error
        context.errorStatus = true;
      }
      const auto fileInfo = fmt::format("# File: \"{:<50}\"", loc.file_name());
      const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
      const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());

      std::cerr << fmt::format("\nMusa Error on Device {}: {}\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
                               did >= 0 ? std::to_string(did) : "unknown",
                               get_musa_rt_error_string(errorCode), " musa api error location ",
                               fileInfo, locInfo, funcInfo, "=");
    }
  }

  // kernel launch
  u32 Musa::launchKernel(const void *f, unsigned int gx, unsigned int gy, unsigned int gz,
                         unsigned int bx, unsigned int by, unsigned int bz, void **args,
                         size_t shmem, void *stream) {
    return muLaunchKernel((MUfunction)f, gx, gy, gz, bx, by, bz, shmem,
                            (MUstream)stream, args, nullptr);
  }
  u32 Musa::launchCallback(void *stream, void *f, void *data) {
    return (u32)muLaunchHostFunc((MUstream)stream, (MUhostFn)f, data);
  }

  //
  Musa::ContextGuard::ContextGuard(void *context, bool restore, const source_location &loc)
      : needRestore(false), loc(loc) {
    if (context) {
      if (restore)
        if (checkMuApiError(muCtxGetCurrent((MUcontext *)(&prevContext)), loc,
                            "[muCtxGetCurrent]")) {
          if (context != prevContext)
            needRestore
                = checkMuApiError(muCtxSetCurrent((MUcontext)context), loc, "[muCtxGetCurrent]");
        }
    }
  }
  Musa::ContextGuard::~ContextGuard() {
    if (needRestore)
      if (MUresult ec = muCtxSetCurrent((MUcontext)prevContext); ec != MUSA_SUCCESS) {
        const char *errString = nullptr;
        muGetErrorString(ec, &errString);
        checkMuApiError((u32)ec, loc, fmt::format("on restoring context {}", prevContext),
                        errString);
      }
  }

  // musa context
  void Musa::MusaContext::checkError(u32 errorCode, const source_location &loc) const {
    Musa::check_musa_rt_error(errorCode, getDevId(), loc);
  }

  void Musa::MusaContext::setContext(const source_location &loc) const {
    const char *errString = nullptr;
    auto ec = muCtxSetCurrent((MUcontext)getContext());
    if (ec != MUSA_SUCCESS) {
      muGetErrorString(ec, &errString);
      checkMuApiError((u32)ec, loc, "[Musa::MusaContext::setContext]", errString);
    }
  }
  // stream ordered memory allocator
  void *Musa::MusaContext::streamMemAlloc(size_t size, void *stream, const source_location &loc) {
    void *ptr;
/// TODO:
#if 1
    muMemAlloc((MUdeviceptr *)&ptr, size);
#else
    muMemAllocAsync((MUdeviceptr *)&ptr, size, (MUstream)stream);
#endif
    // ::musaMallocAsync((MUdeviceptr *)&ptr, size, (MUstream)stream);
    return ptr;
  }
  void Musa::MusaContext::streamMemFree(void *ptr, void *stream, const source_location &loc) {
/// TODO:
#if 1
    muStreamSynchronize((MUstream)stream);
    muMemFree((MUdeviceptr)ptr);
#else
    muMemFreeAsync((MUdeviceptr)ptr, (MUstream)stream);
#endif
    // ::musaFreeAsync((MUdeviceptr)ptr, (MUstream)stream);
  }

  bool Musa::set_default_device(int dev, const source_location &loc) {
    auto &inst = driver();
    if (dev == inst.defaultDevice || dev >= inst.numTotalDevice || dev < 0) return false;
    inst.defaultDevice = dev;
    return prepare_context(mem_device, dev, loc);
  }
  int Musa::get_default_device() noexcept { return driver().defaultDevice; }

  Musa::Musa() {
    ;
    ;
  }
  Musa::~Musa() {
    ;
    ;
  }

}  // namespace zs