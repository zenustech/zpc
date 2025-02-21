#pragma once
#include <iostream>
#include <vector>

#include "zensim/Platform.hpp"

#if !ZS_ENABLE_MUSA
#  error "ZS_ENABLE_MUSA* was not enabled, but Musa.hpp was included anyway."
#endif

#include "zensim/Reflection.h"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/zpc_tpls/fmt/format.h"

namespace zs {

  struct Musa {
  private:
    Musa();

  public:
    ZPC_BACKEND_API static Musa &instance() {
      static Musa s_instance{};
      return s_instance;
    }
    ~Musa();

    class ZPC_BACKEND_API ContextGuard {
    public:
      // default policy for restoring contexts
      explicit ContextGuard(void *context, bool restore = false,
                            const source_location &loc = source_location::current());
      ~ContextGuard();

    private:
      source_location loc;
      void *prevContext;
      bool needRestore;
    };

    static auto &driver() noexcept { return instance(); }
    static auto &context(int devid) { return driver().contexts[devid]; }
    static auto alignment() noexcept { return driver().textureAlignment; }
    static auto device_count() noexcept { return driver().numTotalDevice; }

    ZPC_BACKEND_API static bool set_default_device(int dev, const source_location &loc
                                                            = source_location::current());
    ZPC_BACKEND_API static int get_default_device() noexcept;

    /// error handling
    ZPC_BACKEND_API static u32 get_last_musa_rt_error();
    ZPC_BACKEND_API static std::string_view get_musa_rt_error_string(u32 errorCode);
    ZPC_BACKEND_API static void check_musa_rt_error(u32 errorCode, ProcID did = -1,
                                                    const source_location &loc
                                                    = source_location::current());

    /// kernel launch
    ZPC_BACKEND_API static u32 launchKernel(const void *f, unsigned int gx, unsigned int gy,
                                            unsigned int gz, unsigned int bx, unsigned int by,
                                            unsigned int bz, void **args, size_t shmem,
                                            void *stream);
    ZPC_BACKEND_API static u32 launchCallback(void *stream, void *f, void *data);

    struct ZPC_BACKEND_API MusaContext {
      auto getDevId() const noexcept { return devid; }
      auto getDevice() const noexcept { return dev; }
      auto getContext() const noexcept { return context; }
      void checkError(u32 errorCode, const source_location &loc = source_location::current()) const;
      void setContext(const source_location &loc = source_location::current()) const;

      // stream ordered memory allocator
      void *streamMemAlloc(size_t size, void *stream,
                           const source_location &loc = source_location::current());
      void streamMemFree(void *ptr, void *stream,
                         const source_location &loc = source_location::current());

      int devid{0};
      int dev{0};              ///< MUdevice (4 bytes)
      void *context{nullptr};  ///< MUcontext
      bool supportConcurrentUmAccess;

      mutable bool errorStatus{false};
    };

    void *_defaultCtx;

  private:
    int numTotalDevice;
    std::vector<MusaContext> contexts;  ///< generally one per device
    int textureAlignment;
    int defaultDevice;
  };

  [[maybe_unused]] inline bool checkMuApiError(u32 error, const source_location &loc,
                                               std::string_view msg,
                                               std::string_view errorString) noexcept {
    if (error != 0) {
      const auto fileInfo = fmt::format("# File: \"{:<50}\"", loc.file_name());
      const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
      const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());

      std::cerr << fmt::format("\nMusa Driver Api Error {}: {}\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
                               msg, errorString, " error location ", fileInfo, locInfo, funcInfo,
                               "=");
      return false;
    }
    return true;
  }
  [[maybe_unused]] inline bool checkMuApiError(u32 error, const source_location &loc,
                                               std::string_view msg) noexcept {
    if (error != 0) {
      const auto fileInfo = fmt::format("# File: \"{:<50}\"", loc.file_name());
      const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
      const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());

      std::cerr << fmt::format("\nMusa Driver Api Error {}\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n", msg,
                               " error location ", fileInfo, locInfo, funcInfo, "=");
      return false;
    }
    return true;
  }
  [[maybe_unused]] inline bool checkMuApiError(u32 error, std::string_view msg = "",
                                               const source_location &loc
                                               = source_location::current()) noexcept {
    return checkMuApiError(error, loc, msg);
  }

  inline void checkMuKernelLaunchError(u32 error, const Musa::MusaContext &ctx,
                                       std::string_view streamInfo,
                                       const source_location &loc) noexcept {
    if (error != 0) {
      const auto fileInfo = fmt::format("# File: \"{:<50}\"", loc.file_name());
      const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
      const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());
      if (ctx.errorStatus) return;  // there already exists a preceding musa error
      ctx.errorStatus = true;

      std::cerr << fmt::format(
          "\nMusa Error on Device {}, Stream {}: {}\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
          ctx.getDevId(), streamInfo, Musa::get_musa_rt_error_string(error),
          " kernel error location ", fileInfo, locInfo, funcInfo, "=");
    }
  }

}  // namespace zs