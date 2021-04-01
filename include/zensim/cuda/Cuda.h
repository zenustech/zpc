#pragma once

#include <driver_types.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../DynamicLoader.h"
#include "Allocators.cuh"
#include "CudaFunction.cuh"
#include "CudaLaunchConfig.cuh"
#include "HostUtils.hpp"
#include "zensim/Reflection.h"
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

    struct CudaContext {
      auto &driver() const noexcept { return Cuda::driver(); }
      CudaContext(int devId, void *device, void *contextIn)
          : devid{devId}, dev{device}, context{contextIn} {}
      auto getDevId() const noexcept { return devid; }
      auto getDevice() const noexcept { return dev; }
      auto getContext() const noexcept { return context; }

      /// only use after Cuda system initialization
      void setContext() { driver().setContext(context); }
      /// stream & event
      // stream
      template <StreamIndex sid> auto stream() const {
        return streams[static_cast<unsigned int>(sid)];
      }
      auto stream(unsigned sid) const { return streams[sid]; }
      auto stream_compute() const {
        return streams[static_cast<unsigned int>(StreamIndex::Compute)];
      }
      auto stream_spare(unsigned sid = 0) const {
        return streams[static_cast<unsigned int>(StreamIndex::Spare) + sid];
      }
      // sync
      void syncCompute() const { driver().syncStream(stream_compute()); }
      template <StreamIndex sid> void syncStream() const { driver().syncStream(stream<sid>()); }
      void syncStream(unsigned sid) const { driver().syncStream(stream(sid)); }
      void syncStreamSpare(unsigned sid = 0) const { driver().syncStream(stream_spare(sid)); }

      // event
      auto event_compute() const { return events[static_cast<unsigned int>(EventIndex::Compute)]; }
      auto event_spare(unsigned eid = 0) const {
        return events[static_cast<unsigned int>(EventIndex::Spare) + eid];
      }
      //
      auto compute_event_record() { driver().recordEvent(event_compute(), stream_compute()); }
      auto spare_event_record(unsigned id = 0) {
        driver().recordEvent(event_spare(id), stream_spare(id));
      }
      void computeStreamWaitForEvent(void *event) {
        driver().streamWaitEvent(stream_compute(), event, 0);
      }
      void spareStreamWaitForEvent(unsigned sid, void *event) {
        driver().streamWaitEvent(stream_spare(sid), event, 0);
      }

      /// kernel launch
      template <typename... Args> std::tuple<bool, int> getKernelFunction(void (*func)(Args...)) {
        if (auto it = kernelUMap.find(reinterpret_cast<uintptr_t>(func)); it != kernelUMap.end())
          return std::tuple<bool, int>{false, it->second};
        else {
          int id = static_cast<int>(kernelUMap.size());
          kernelUMap.emplace(reinterpret_cast<uintptr_t>(func), id);
          if (id == 0)
            launchMem = std::make_unique<handle_resource>(&device_memory_resource::instance());
          return std::tuple<bool, int>{true, id};
        }
      }
      template <typename... Args, std::size_t... Is>
      void passKernelParameters(void **hdargs, void *stream, std::index_sequence<Is...>,
                                const Args &...args) {
        ((driver().memcpyAsync(hdargs[Is], (void *)&args, sizeof(args), stream)), ...);
      }
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#offline-compilation
      template <typename... Arguments> void compute_launch(LaunchConfig &&lc,
                                                           void (*f)(remove_vref_t<Arguments>...),
                                                           const Arguments &...args) {
        if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
#if 1
          f<<<lc.dg, lc.db, lc.shmem, (cudaStream_t)stream<StreamIndex::Compute>()>>>(args...);
#else
          void *hargs[] = {(void *)&args...};
          driver().launch((void *)f, lc.dg, lc.db, hargs, lc.shmem, stream<StreamIndex::Compute>());
#endif
          cudaError_t error = cudaGetLastError();
          if (error != cudaSuccess)
            printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n", devid,
                   cudaGetErrorString(error));
          return;
#if 0
        constexpr auto N = sizeof...(Arguments);
        void *hargs[N] = {(void *)(&args)...};
        //driver().launch((void *)f, lc.dg, lc.db, hargs, lc.shmem,
        //                stream<StreamIndex::Compute>());
        std::size_t argSizes[] = {sizeof(args)...};
        auto [tag, fnid] = getKernelFunction(f);
        if (tag) { ///< fill kernel launch config
          kernelLaunchConfigs.emplace_back(KernelLaunchParams{});
          auto &config = kernelLaunchConfigs.back();
          for (int i = 0; i < N; ++i)
            config.offsets.emplace_back(
                launchMem->acquire(argSizes[i], alignment()));
        }
        auto &config = kernelLaunchConfigs[fnid];
#  if 0
        for (int i = 0; i < 10; ++i) {
          getchar();
          fmt::print("{}-th round\n", i);
          driver().getFuncAttrib(&config.funcAttribs[i], i, (void *)f);
          getchar();
        }
        fmt::print("{}-th func: threadsPerBlock {}, staticSharedMem {}, "
                   "constMem {}, localMemPerThread {}, regsPerThread {}, "
                   "ptxVer {}, binaryVer {}, cacheMode {}, dynamicSharedMem "
                   "{}, sharedMemRatio {}\n",
                   fnid, config.funcAttribs[0], config.funcAttribs[1],
                   config.funcAttribs[2], config.funcAttribs[3],
                   config.funcAttribs[4], config.funcAttribs[5],
                   config.funcAttribs[6], config.funcAttribs[7],
                   config.funcAttribs[8], config.funcAttribs[9]);

        void *dargs[N];
        for (int i = 0; i < N; ++i)
          dargs[i] = launchMem->address(config.offsets[i]);
        passKernelParameters(dargs, stream<StreamIndex::Compute>(),
                             std::make_index_sequence<N>{}, args...);
        using tup = tuple<std::remove_reference_t<Arguments>...>;
        fmt::print("compute_launch: {}-th({}) {}({})\n", fnid, tag,
                   query_type_name(f), query_type_name<tup>());
        for (int i = 0; i < N; ++i)
          fmt::print("arg {} size {} args_addr {}\t", i, argSizes[i],
                     (uintptr_t)dargs[i]);
        fmt::print("\n");
#  endif
#  if 1
        if (true) {
          driver().launchKernel((void *)f, lc.dg.x, lc.dg.y, lc.dg.z, lc.db.x,
                                lc.db.y, lc.db.z, lc.shmem,
                                stream<StreamIndex::Compute>(), hargs, nullptr);
        } else {
          f<<<lc.dg, lc.db, lc.shmem,
              (CUstream_st *)stream<StreamIndex::Compute>()>>>(args...);
          cudaError_t error = cudaGetLastError();
          if (error != cudaSuccess)
            printf("[Dev %d] Kernel launch failure on [COMPUTE stream] %s\n",
                   devid, cudaGetErrorString(error));
        }
#  endif
#endif
        }
      }

      template <typename... Arguments> void spare_launch(unsigned sid, LaunchConfig &&lc,
                                                         void (*f)(remove_vref_t<Arguments>...),
                                                         const Arguments &...args) {
        if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
          f<<<lc.dg, lc.db, lc.shmem, (cudaStream_t)stream_spare(sid)>>>(args...);
          cudaError_t error = cudaGetLastError();
          if (error != cudaSuccess)
            printf("[Dev %d] Kernel launch failure on [SPARE stream] %s\n", devid,
                   cudaGetErrorString(error));
        }
      }

      template <typename... Arguments> void launch(void *stream, LaunchConfig &&lc,
                                                   void (*f)(remove_vref_t<Arguments>...),
                                                   const Arguments &...args) {
        if (lc.dg.x && lc.dg.y && lc.dg.z && lc.db.x && lc.db.y && lc.db.z) {
          f<<<lc.dg, lc.db, lc.shmem, (cudaStream_t)stream>>>(args...);
          cudaError_t error = cudaGetLastError();
          if (error != cudaSuccess)
            printf("[Dev %d] Kernel launch failure on [SPARE stream] %s\n", devid,
                   cudaGetErrorString(error));
        }
      }

      /// allocator initialization on use
      void initDeviceMemory();
      void initUnifiedMemory();

      auto borrow(std::size_t bytes) -> void *;
      void resetMem();

      auto borrowVirtual(std::size_t bytes) -> void *;
      void resetVirtualMem();

    public:
      int devid;
      void *dev;                    ///< CUdevice
      void *context;                ///< CUcontext
      std::vector<void *> streams;  ///< CUstream
      std::vector<void *> events;   ///< CUevents
      std::unique_ptr<MonotonicAllocator> deviceMem;
      std::unique_ptr<MonotonicVirtualAllocator> unifiedMem;
      std::unique_ptr<handle_resource> launchMem;
      std::unordered_map<uintptr_t, int> kernelUMap;  ///< function unordered_map
      struct KernelLaunchParams {
        int funcAttribs[10];
        std::vector<uintptr_t> offsets; /**< offset in bytes within launchMem */
      };
      std::vector<KernelLaunchParams> kernelLaunchConfigs;
    };  //< [end] struct CudaContext

    //< context ref
    auto &refCudaContext(int devId) noexcept { return contexts[devId]; }

    static auto ref_cuda_context(int devId) noexcept -> CudaContext & {
      return instance().contexts[devId];
    }

#define PER_CUDA_FUNCTION(name, symbol_name, ...) CudaDriverApi<__VA_ARGS__> name;
#include "cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION

#define PER_CUDA_FUNCTION(name, symbol_name, ...) CudaRuntimeApi<__VA_ARGS__> name;
#include "cuda_runtime_functions.inc.h"
#undef PER_CUDA_FUNCTION
    void (*get_cu_error_name)(uint32_t, const char **);
    void (*get_cu_error_string)(uint32_t, const char **);
    const char *(*get_cuda_error_name)(uint32_t);
    const char *(*get_cuda_error_string)(uint32_t);

  private:
    int numTotalDevice;

    /// driver apis
    std::vector<CudaContext> contexts;  ///< generally one per device
    int textureAlignment;
    std::unique_ptr<DynamicLoader> driverLoader, runtimeLoader;

    std::mutex lock;

    int _iDevID;  ///< need changing
  };

}  // namespace zs
