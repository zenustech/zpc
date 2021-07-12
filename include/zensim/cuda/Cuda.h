#pragma once

/// kokkos/core/src/setup/Kokkos_Setup_Cuda.hpp
#if !ZS_ENABLE_CUDA
#  error "ZS_ENABLE_CUDA was not enabled, but Cuda.h was included anyway."
#endif

#if ZS_ENABLE_CUDA && !defined(__CUDACC__)
#  error "ZS_ENABLE_CUDA defined but the compiler is not defining the __CUDACC__ macro as expected"
// Some tooling environments will still function better if we do this here.
#  define __CUDACC__
#endif

// #include <driver_types.h>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../DynamicLoader.h"
#include "Allocators.cuh"
#include "CudaFunction.cuh"
#include "CudaLaunchConfig.cuh"
#include "zensim/Reflection.h"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/types/Tuple.h"

namespace zs {

  class Cuda : public Singleton<Cuda> {
  public:
    Cuda();
    ~Cuda();

    /// kernel launching
    enum class StreamIndex { Compute = 0, H2DCopy, D2HCopy, D2DCopy, Spare, Total = 32 };
    enum class EventIndex { Compute = 0, H2DCopy, D2HCopy, D2DCopy, Spare, Total = 32 };

    static auto &driver() { return instance(); }
    static auto &context(int devid) { return driver().contexts[devid]; }
    static auto alignment() { return driver().textureAlignment; }
    static void init_constant_cache(void *ptr, std::size_t size);
    static void init_constant_cache(void *ptr, std::size_t size, void *stream);

    /// error handling
    static u32 getLastCudaError();
    static std::string_view getCudaErrorString(u32 errorCode);
    static void checkError(u32 errorCode, ProcID did = -1,
                           const source_location &loc = source_location::current());
    /// kernel launch
    static u32 launchKernel(const void *f, unsigned int gx, unsigned int gy, unsigned int gz,
                            unsigned int bx, unsigned int by, unsigned int bz, void **args,
                            std::size_t shmem, void *stream);
    static u32 launchCooperativeKernel(const void *f, unsigned int gx, unsigned int gy,
                                       unsigned int gz, unsigned int bx, unsigned int by,
                                       unsigned int bz, void **args, std::size_t shmem,
                                       void *stream);
    static u32 launchCallback(void *stream, void *f, void *data);

    struct CudaContext {
      auto &driver() const noexcept { return Cuda::driver(); }
      CudaContext(int devId = 0, int device = 0, void *contextIn = nullptr)
          : devid{devId}, dev{device}, context{contextIn}, errorStatus{false} {}
      auto getDevId() const noexcept { return devid; }
      auto getDevice() const noexcept { return dev; }
      auto getContext() const noexcept { return context; }
      void checkError() const;
      void checkError(u32 errorCode, const source_location &loc = source_location::current()) const;

      /// only use after Cuda system initialization
      void setContext(const source_location &loc = source_location::current()) const;
      /// stream & event
      // stream
      template <StreamIndex sid> auto stream() const {
        return streams[static_cast<unsigned int>(sid)];
      }
      auto stream(unsigned sid) const { return streams[sid]; }
      auto streamCompute() const {
        return streams[static_cast<unsigned int>(StreamIndex::Compute)];
      }
      auto streamSpare(unsigned sid = 0) const {
        return streams[static_cast<unsigned int>(StreamIndex::Spare) + sid];
      }

      // event
      auto eventCompute() const { return events[static_cast<unsigned int>(EventIndex::Compute)]; }
      auto eventSpare(unsigned eid = 0) const {
        return events[static_cast<unsigned int>(EventIndex::Spare) + eid];
      }

      // record
      void recordEventCompute(const source_location &loc = source_location::current());
      void recordEventSpare(unsigned id = 0,
                            const source_location &loc = source_location::current());
      // sync
      void syncStream(unsigned sid, const source_location &loc = source_location::current()) const;
      void syncCompute(const source_location &loc = source_location::current()) const;
      template <StreamIndex sid> void syncStream() const { syncStream(stream<sid>()); }
      void syncStreamSpare(unsigned sid = 0,
                           const source_location &loc = source_location::current()) const;
      // stream-event sync
      void computeStreamWaitForEvent(void *event,
                                     const source_location &loc = source_location::current());
      void spareStreamWaitForEvent(unsigned sid, void *event,
                                   const source_location &loc = source_location::current());
      // stream ordered memory allocator
      void *streamMemAlloc(std::size_t size, void *stream,
                           const source_location &loc = source_location::current());
      void streamMemFree(void *ptr, void *stream,
                         const source_location &loc = source_location::current());

      struct StreamExecutionTimer {
        StreamExecutionTimer(CudaContext *ctx, void *stream, const source_location &loc)
            : ctx{ctx}, stream{stream} {
          msg = fmt::format("[Cuda Exec [Device {}, Stream {}] | File {}, Ln {}, Col {}]",
                            ctx->getDevId(), stream, loc.file_name(), loc.line(), loc.column());
          timer.tick();
        }
        ~StreamExecutionTimer() { timer.tock(msg); }

        CudaContext *ctx;
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

      /// kernel launch
      template <typename... Arguments> [[deprecated("use cuda_safe_launch")]] void launchCompute(
          LaunchConfig &&lc, void (*f)(remove_vref_t<Arguments>...), const Arguments &...args) {
        if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
          void *kernelArgs[] = {(void *)&args...};
          // driver().launch((void *)f, lc.dg, lc.db, kernelArgs, lc.shmem, streamCompute());
          launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y, lc.db.z, kernelArgs,
                       lc.shmem, streamCompute());
          checkError();
        }
      }

      // https://docs.nvidia.com/cuda/archive/10.2/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER
      template <typename... Arguments> [[deprecated("use cuda_safe_launch")]] void launchSpare(
          StreamID sid, LaunchConfig &&lc, void (*f)(remove_vref_t<Arguments>...),
          const Arguments &...args) {
        if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
          void *kernelArgs[] = {(void *)&args...};
#if 0
          // driver api
          driver().launchCuKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y, lc.db.z, lc.shmem,
                                       streamSpare(sid), kernelArgs, nullptr);
#else
          // f<<<lc.dg, lc.db, lc.shmem, (cudaStream_t)streamSpare(sid)>>>(args...);
          launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y, lc.db.z, kernelArgs,
                       lc.shmem, streamSpare(sid));
#endif
          checkError();
        }
      }

      template <typename... Arguments>
      [[deprecated("use cuda_safe_launch")]] void launch(void *stream, LaunchConfig &&lc,
                                                         void (*f)(remove_vref_t<Arguments>...),
                                                         const Arguments &...args) {
        if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
          // f<<<lc.dg, lc.db, lc.shmem, (cudaStream_t)stream>>>(args...);
          void *kernelArgs[] = {(void *)&args...};
          launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y, lc.db.z, kernelArgs,
                       lc.shmem, stream);
          checkError();
        }
      }

      /// allocator initialization on use
      [[deprecated]] void initDeviceMemory();
      [[deprecated]] void initUnifiedMemory();

      [[deprecated]] auto borrow(std::size_t bytes) -> void *;
      [[deprecated]] void resetMem();

      [[deprecated]] auto borrowVirtual(std::size_t bytes) -> void *;
      [[deprecated]] void resetVirtualMem();

      struct Config {
        int optBlockSize{0};
      };

    public:
      int devid;
      int dev;                      ///< CUdevice (4 bytes)
      void *context;                ///< CUcontext
      std::vector<void *> streams;  ///< CUstream
      std::vector<void *> events;   ///< CUevents
      std::set<StreamExecutionTimer *> timers;
      int maxThreadsPerBlock, maxThreadsPerMultiprocessor, regsPerMultiprocessor,
          sharedMemPerMultiprocessor, sharedMemPerBlock, maxBlocksPerMultiprocessor;
      bool supportConcurrentUmAccess;

      mutable std::unordered_map<void *, Config> funcLaunchConfigs;

      mutable bool errorStatus;
      [[deprecated]] std::unique_ptr<MonotonicAllocator> deviceMem;
      [[deprecated]] std::unique_ptr<MonotonicVirtualAllocator> unifiedMem;
    };  //< [end] struct CudaContext

#define PER_CUDA_FUNCTION(name, symbol_name, ...) CudaDriverApi<__VA_ARGS__> name;
#include "cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION

#if 0
#  define PER_CUDA_FUNCTION(name, symbol_name, ...) CudaRuntimeApi<__VA_ARGS__> name;
#  include "cuda_runtime_functions.inc.h"
#  undef PER_CUDA_FUNCTION
#endif
    void (*get_cu_error_name)(uint32_t, const char **);
    void (*get_cu_error_string)(uint32_t, const char **);
    // const char *(*get_cuda_error_name)(uint32_t);
    // const char *(*get_cuda_error_string)(uint32_t);

    /// other utilities
    /// reference: kokkos/core/src/Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp, Ln 101
    static int deduce_block_size(const CudaContext &ctx, void *f, std::function<std::size_t(int)>,
                                 std::string_view = "");

  private:
    int numTotalDevice;

    std::vector<CudaContext> contexts;  ///< generally one per device
    int textureAlignment;
    std::unique_ptr<DynamicLoader> driverLoader, runtimeLoader;

    int _iDevID;  ///< need changing
  };

  template <typename... Args> struct cuda_safe_launch {
    void checkKernelLaunchError(u32 error, const Cuda::CudaContext &ctx,
                                std::string_view streamInfo, const source_location &loc) noexcept {
      if (error != 0) {
        const auto fileInfo = fmt::format("# File: \"{}\"", loc.file_name());
        const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
        const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());
        if (ctx.errorStatus) return;  // there already exists a preceding cuda error
        ctx.errorStatus = true;
        fmt::print(fg(fmt::color::crimson) | fmt::emphasis::italic | fmt::emphasis::bold,
                   "\nCuda Error on Device {}, Stream {}: {}\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
                   ctx.getDevId(), streamInfo, Cuda::getCudaErrorString(error),
                   " kernel error location ", fileInfo, locInfo, funcInfo, "=");
      }
    }
#define CHECK_LAUNCH_CONFIG                                                                       \
  if (lc.enableAutoConfig()) {                                                                    \
    auto nwork = lc.db.x;                                                                         \
    lc.db.x = Cuda::deduce_block_size(ctx, (void *)f, [shmem = lc.shmem](int) { return shmem; }); \
    lc.dg.x = (nwork + lc.db.x - 1) / lc.db.x;                                                    \
  }
    explicit cuda_safe_launch(const source_location &loc, const Cuda::CudaContext &ctx,
                              LaunchConfig &&lc, void (*f)(remove_cvref_t<Args>...),
                              const Args &...args) {
      if (lc.valid()) {
        void *kernelArgs[] = {(void *)&args...};
        CHECK_LAUNCH_CONFIG
        auto error = Cuda::launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y,
                                        lc.db.z, kernelArgs, lc.shmem, ctx.streamCompute());
        checkKernelLaunchError(error, ctx, "Compute", loc);
      }
    }
    explicit cuda_safe_launch(const Cuda::CudaContext &ctx, LaunchConfig &&lc,
                              void (*f)(remove_cvref_t<Args>...), const Args &...args,
                              const source_location &loc = source_location::current()) {
      if (lc.valid()) {
        void *kernelArgs[] = {(void *)&args...};
        CHECK_LAUNCH_CONFIG
        auto error = Cuda::launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y,
                                        lc.db.z, kernelArgs, lc.shmem, ctx.streamCompute());
        checkKernelLaunchError(error, ctx, "Compute", loc);
      }
    }

    explicit cuda_safe_launch(const source_location &loc, const Cuda::CudaContext &ctx,
                              StreamID sid, LaunchConfig &&lc, void (*f)(remove_cvref_t<Args>...),
                              const Args &...args) {
      if (lc.valid()) {
        void *kernelArgs[] = {(void *)&args...};
        CHECK_LAUNCH_CONFIG
        auto error = Cuda::launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y,
                                        lc.db.z, kernelArgs, lc.shmem, ctx.streamSpare(sid));
        checkKernelLaunchError(error, ctx, fmt::format("Spare [{}]", sid), loc);
      }
    }
    explicit cuda_safe_launch(const Cuda::CudaContext &ctx, StreamID sid, LaunchConfig &&lc,
                              void (*f)(remove_cvref_t<Args>...), const Args &...args,
                              const source_location &loc = source_location::current()) {
      if (lc.valid()) {
        void *kernelArgs[] = {(void *)&args...};
        CHECK_LAUNCH_CONFIG
        auto error = Cuda::launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x, lc.db.y,
                                        lc.db.z, kernelArgs, lc.shmem, ctx.streamSpare(sid));
        checkKernelLaunchError(error, ctx, fmt::format("Spare [{}]", sid), loc);
      }
    }
  };  // namespace zs
  template <typename... Args>
  cuda_safe_launch(const source_location &, const Cuda::CudaContext &, StreamID, LaunchConfig &&,
                   void (*f)(remove_cvref_t<Args>...), const Args &...)
      -> cuda_safe_launch<Args...>;
  template <typename... Args> cuda_safe_launch(const Cuda::CudaContext &, StreamID, LaunchConfig &&,
                                               void (*f)(remove_cvref_t<Args>...), const Args &...)
      -> cuda_safe_launch<Args...>;
  template <typename... Args> cuda_safe_launch(const source_location &, const Cuda::CudaContext &,
                                               LaunchConfig &&, void (*f)(remove_cvref_t<Args>...),
                                               const Args &...) -> cuda_safe_launch<Args...>;
  template <typename... Args> cuda_safe_launch(const Cuda::CudaContext &, LaunchConfig &&,
                                               void (*f)(remove_cvref_t<Args>...), const Args &...)
      -> cuda_safe_launch<Args...>;

}  // namespace zs
