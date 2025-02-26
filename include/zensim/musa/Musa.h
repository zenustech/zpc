#pragma once
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

#include "zensim/Platform.hpp"

#if !defined(ZS_ENABLE_MUSA) || (defined(ZS_ENABLE_MUSA) && !ZS_ENABLE_MUSA)
#  error "ZS_ENABLE_MUSA* was not enabled, but Musa.hpp was included anyway."
#endif

#include "MusaLaunchConfig.h"
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

    enum class StreamIndex { Compute = 0, H2DCopy, D2HCopy, D2DCopy, Spare, Total = 32 };
    enum class EventIndex {
      Compute = 0,
      H2DCopy,
      D2HCopy,
      D2DCopy,
      Spare,
      Default = 32,
      Total = Default + 1
    };
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

      /// stream & event
      // stream
      /// @note
      /// https://stackoverflow.com/questions/31458016/in-cuda-is-it-guaranteed-that-the-default-stream-equals-nullptr
      template <StreamIndex sid> auto stream() const { return streams[static_cast<StreamID>(sid)]; }
      void *stream(StreamID sid) const {
        if (sid >= 0)
          return streams[sid];
        else
          return nullptr;
      }
      auto streamCompute() const { return streams[static_cast<StreamID>(StreamIndex::Compute)]; }
      void *streamSpare(StreamID sid = 0) const {
        if (sid >= 0)
          return streams[static_cast<StreamID>(StreamIndex::Spare) + sid];
        else
          return nullptr;
      }

      // event
      void *eventCompute() const { return events[static_cast<StreamID>(EventIndex::Compute)]; }
      void *eventSpare(StreamID eid = 0) const {
        if (eid >= 0)
          return events[static_cast<StreamID>(EventIndex::Spare) + eid];
        else
          return events[static_cast<StreamID>(EventIndex::Default)];
      }

      // record
      void recordEventCompute(const source_location &loc = source_location::current());
      void recordEventSpare(StreamID id = 0,
                            const source_location &loc = source_location::current());
      // sync
      void syncStream(StreamID sid, const source_location &loc = source_location::current()) const;
      void syncCompute(const source_location &loc = source_location::current()) const;
      template <StreamIndex sid> void syncStream() const { syncStream(stream<sid>()); }
      void syncStreamSpare(StreamID sid = 0,
                           const source_location &loc = source_location::current()) const;
      // stream-event sync
      void computeStreamWaitForEvent(void *event,
                                     const source_location &loc = source_location::current());
      void spareStreamWaitForEvent(StreamID sid, void *event,
                                   const source_location &loc = source_location::current());
      // stream ordered memory allocator
      void *streamMemAlloc(size_t size, void *stream,
                           const source_location &loc = source_location::current());
      void streamMemFree(void *ptr, void *stream,
                         const source_location &loc = source_location::current());

      struct StreamExecutionTimer {
        StreamExecutionTimer(MusaContext *ctx, void *stream, const source_location &loc)
            : ctx{ctx}, stream{stream} {
          msg = fmt::format("[Musa Exec [Device {}, Stream {}] | File {}, Ln {}, Col {}]",
                            ctx->getDevId(), stream, loc.file_name(), loc.line(), loc.column());
          timer.tick();
        }
        ~StreamExecutionTimer() { timer.tock(msg); }

        MusaContext *ctx;
        void *stream;
        CppTimer timer;
        std::string msg;
      };
      static void recycle_timer(void *streamExecTimer) {
        auto timer = (StreamExecutionTimer *)streamExecTimer;
        timer->ctx->timers.erase(timer);
        delete timer;
      }

      [[nodiscard]] StreamExecutionTimer *tick(void *stream, const source_location &loc
                                                             = source_location::current());
      void tock(StreamExecutionTimer *timer,
                const source_location &loc = source_location::current());

      struct Config {
        int optBlockSize{0};
      };

    public:
      int devid{0};
      int dev{0};                   ///< MUdevice (4 bytes)
      void *context{nullptr};       ///< MUcontext
      std::vector<void *> streams;  ///< MUstream
      std::vector<void *> events;   ///< MUevents
      std::set<StreamExecutionTimer *> timers;
      int maxThreadsPerBlock, maxThreadsPerMultiprocessor, regsPerMultiprocessor, numMultiprocessor,
          sharedMemPerMultiprocessor, sharedMemPerBlock, maxBlocksPerMultiprocessor;
      bool supportConcurrentUmAccess;

      mutable std::unordered_map<void *, Config> funcLaunchConfigs;

      mutable bool errorStatus{false};
    };

    ZPC_BACKEND_API static int deduce_block_size(const source_location &loc, const MusaContext &ctx,
                                                 void *f, function<size_t(int)>,
                                                 std::string_view = "");

    mutable bool errorStatus;
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

  template <typename... Args> struct musa_safe_launch {
    operator u32() const { return errorCode; }
#define CHECK_LAUNCH_CONFIG                                                                 \
  if (lc.autoConfigEnabled()) {                                                             \
    auto nwork = lc.db.x;                                                                   \
    lc.db.x = Musa::deduce_block_size(loc, ctx, (void *)f,                                  \
                                      [shmem = lc.shmem](int) -> size_t { return shmem; }); \
    lc.dg.x = (nwork + lc.db.x - 1) / lc.db.x;                                              \
    lc.shmem = (lc.shmem + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)         \
               * sizeof(std::max_align_t);                                                  \
  }
    explicit musa_safe_launch(const source_location &loc, const Musa::MusaContext &ctx,
                              LaunchConfig &&lc, void (*f)(remove_cvref_t<Args>...),
                              const Args &...args) {
      if (lc.valid()) {
        void *kernelArgs[] = {(void *)&args...};
        CHECK_LAUNCH_CONFIG
        errorCode = Musa::launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y,
                                       lc.db.z, kernelArgs, lc.shmem, ctx.streamCompute());
        // checkMuKernelLaunchError(error, ctx, "Compute", loc);
      }
    }
    explicit musa_safe_launch(const Musa::MusaContext &ctx, LaunchConfig &&lc,
                              void (*f)(remove_cvref_t<Args>...), const Args &...args,
                              const source_location &loc = source_location::current()) {
      if (lc.valid()) {
        void *kernelArgs[] = {(void *)&args...};
        CHECK_LAUNCH_CONFIG
        errorCode = Musa::launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y,
                                       lc.db.z, kernelArgs, lc.shmem, ctx.streamCompute());
        // checkMuKernelLaunchError(error, ctx, "Compute", loc);
      }
    }

    explicit musa_safe_launch(const source_location &loc, const Musa::MusaContext &ctx,
                              StreamID sid, LaunchConfig &&lc, void (*f)(remove_cvref_t<Args>...),
                              const Args &...args) {
      if (lc.valid()) {
        void *kernelArgs[] = {(void *)&args...};
        CHECK_LAUNCH_CONFIG
        errorCode = Musa::launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y,
                                       lc.db.z, kernelArgs, lc.shmem, ctx.streamSpare(sid));
        // checkMuKernelLaunchError(error, ctx, fmt::format("Spare [{}]", sid), loc);
      }
    }
    explicit musa_safe_launch(const Musa::MusaContext &ctx, StreamID sid, LaunchConfig &&lc,
                              void (*f)(remove_cvref_t<Args>...), const Args &...args,
                              const source_location &loc = source_location::current()) {
      if (lc.valid()) {
        void *kernelArgs[] = {(void *)&args...};
        CHECK_LAUNCH_CONFIG
        errorCode = Musa::launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y,
                                       lc.db.z, kernelArgs, lc.shmem, ctx.streamSpare(sid));
        // checkMuKernelLaunchError(error, ctx, fmt::format("Spare [{}]", sid), loc);
      }
    }
    u32 errorCode{0};
  };  // namespace zs
  template <typename... Args>
  musa_safe_launch(const source_location &, const Musa::MusaContext &, StreamID, LaunchConfig &&,
                   void (*f)(remove_cvref_t<Args>...), const Args &...)
      -> musa_safe_launch<Args...>;
  template <typename... Args> musa_safe_launch(const Musa::MusaContext &, StreamID, LaunchConfig &&,
                                               void (*f)(remove_cvref_t<Args>...), const Args &...)
      -> musa_safe_launch<Args...>;
  template <typename... Args> musa_safe_launch(const source_location &, const Musa::MusaContext &,
                                               LaunchConfig &&, void (*f)(remove_cvref_t<Args>...),
                                               const Args &...) -> musa_safe_launch<Args...>;
  template <typename... Args> musa_safe_launch(const Musa::MusaContext &, LaunchConfig &&,
                                               void (*f)(remove_cvref_t<Args>...), const Args &...)
      -> musa_safe_launch<Args...>;

}  // namespace zs