#include <cuda.h>

#include <utility>

#include "../Logger.hpp"
#include "../Platform.hpp"
#include "Cuda.h"
#include "zensim/cuda/memory/MemOps.hpp"
#include "zensim/execution/ConcurrencyPrimitive.hpp"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#define MEM_POOL_CTRL 3

#if 0
namespace {
  static zs::Mutex g_cudaMutex;
  static std::atomic<bool> g_isCudaInitialized = false;
  static zs::Cuda *g_cudaInstance = nullptr;
}  // namespace
#endif

namespace zs {

#if 0
  Cuda &Cuda::instance() {
    if (g_isCudaInitialized.load(std::memory_order_acquire)) return *g_cudaInstance;
    g_cudaMutex.lock();
    if (g_isCudaInitialized.load(std::memory_order_acquire)) return *g_cudaInstance;

    if (!g_cudaInstance) g_cudaInstance = new Cuda;

    g_isCudaInitialized.store(true, std::memory_order_release);
    g_cudaMutex.unlock();
    return *g_cudaInstance;
  }
#endif

  Cuda::ContextGuard::ContextGuard(void *context, bool restore, const source_location &loc)
      : needRestore(false), loc(loc) {
    if (context) {
      if (restore)
        if (checkCuApiError(cuCtxGetCurrent((CUcontext *)(&prevContext)), loc,
                            "[cuCtxGetCurrent]")) {
          if (context != prevContext)
            needRestore
                = checkCuApiError(cuCtxSetCurrent((CUcontext)context), loc, "[cuCtxGetCurrent]");
        }
    }
  }
  Cuda::ContextGuard::~ContextGuard() {
    if (needRestore)
      if (CUresult ec = cuCtxSetCurrent((CUcontext)prevContext); ec != CUDA_SUCCESS) {
        const char *errString = nullptr;
        if (cuGetErrorString) {
          cuGetErrorString(ec, &errString);
          checkCuApiError((u32)ec, loc, fmt::format("on restoring context {}", prevContext),
                          errString);
        } else
          checkCuApiError((u32)ec, loc, fmt::format("on restoring context {}", prevContext));
      }
  }
  /*
    __device__ __constant__ char g_cuda_constant_cache[8192];  // 1024 words

    void Cuda::init_constant_cache(void *ptr, size_t size) {
      cudaMemcpyToSymbol(g_cuda_constant_cache, ptr, size, 0, cudaMemcpyHostToDevice);
      // cudri::memcpyHtoD((void *)g_cuda_constant_cache, ptr, size);
    }
    void Cuda::init_constant_cache(void *ptr, size_t size, void *stream) {
      cudaMemcpyToSymbolAsync(g_cuda_constant_cache, ptr, size, 0, cudaMemcpyHostToDevice,
                              (cudaStream_t)stream);
    }
    */

  /// error handling
  u32 Cuda::get_last_cuda_rt_error() { return (u32)cudaPeekAtLastError(); }

  std::string_view Cuda::get_cuda_rt_error_string(u32 errorCode) {
    // return cudaGetErrorString((cudaError_t)errorCode);
    return cudaGetErrorString((cudaError_t)errorCode);
  }
  void Cuda::check_cuda_rt_error(u32 errorCode, ProcID did, const source_location &loc) {
    if (errorCode != 0) {
      if (did >= 0) {
        auto &context = Cuda::context(did);
        if (context.errorStatus) return;  // there already exists a preceding cuda error
        context.errorStatus = true;
      }
      const auto fileInfo = fmt::format("# File: \"{:<50}\"", loc.file_name());
      const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
      const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());
#if 0
      fmt::print(fg(fmt::color::crimson) | fmt::emphasis::italic | fmt::emphasis::bold,
                 "\nCuda Error on Device {}: {}\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
                 did >= 0 ? std::to_string(did) : "unknown", get_cuda_rt_error_string(errorCode),
                 " cuda api error location ", fileInfo, locInfo, funcInfo, "=");
#else
      std::cerr << fmt::format("\nCuda Error on Device {}: {}\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
                               did >= 0 ? std::to_string(did) : "unknown",
                               get_cuda_rt_error_string(errorCode), " cuda api error location ",
                               fileInfo, locInfo, funcInfo, "=");
#endif
    }
  }

  /// kernel launch
  u32 Cuda::launchKernel(const void *f, unsigned int gx, unsigned int gy, unsigned int gz,
                         unsigned int bx, unsigned int by, unsigned int bz, void **args,
                         size_t shmem, void *stream) {
    return cudaLaunchKernel(f, dim3{gx, gy, gz}, dim3{bx, by, bz}, args, shmem,
                            (cudaStream_t)stream);
    // return cudri::launchCuKernel(const_cast<void *>(f), gx, gy, gz, bx, by, bz, (unsigned
    // int)shmem,
    //                      stream, args, (void **)nullptr);
  }
  u32 Cuda::launchCooperativeKernel(const void *f, unsigned int gx, unsigned int gy,
                                    unsigned int gz, unsigned int bx, unsigned int by,
                                    unsigned int bz, void **args, size_t shmem, void *stream) {
    // return cudri::launchCuCooperativeKernel(const_cast<void *>(f), gx, gy, gz, bx, by, bz, shmem,
    // stream, args);
    return cudaLaunchCooperativeKernel(f, dim3{gx, gy, gz}, dim3{bx, by, bz}, args, shmem,
                                       (cudaStream_t)stream);
  }
  u32 Cuda::launchCallback(void *stream, void *f, void *data) {
    // return cudaLaunchHostFunc((cudaStream_t)stream, (cudaHostFn_t)f, data);
    return (u32)cuLaunchHostFunc((CUstream)stream, (CUhostFn)f, data);
  }

  void Cuda::CudaContext::checkError(u32 errorCode, const source_location &loc) const {
    /// only shows the first error message
    Cuda::check_cuda_rt_error(errorCode, getDevId(), loc);
  }

  // record
  void Cuda::CudaContext::recordEventCompute(const source_location &loc) {
    checkError(cudaEventRecord((cudaEvent_t)eventCompute(), (cudaStream_t)streamCompute()), loc);
    // cuEventRecord((CUevent)eventCompute(), (CUstream)streamCompute());
  }
  void Cuda::CudaContext::recordEventSpare(StreamID id, const source_location &loc) {
    checkError(cudaEventRecord((cudaEvent_t)eventSpare(id), (cudaStream_t)streamSpare(id)), loc);
    // cuEventRecord((CUevent)eventSpare(id), (CUstream)streamSpare(id));
  }
  // sync
  void Cuda::CudaContext::syncStream(StreamID sid, const source_location &loc) const {
    checkError(cudaStreamSynchronize((cudaStream_t)stream(sid)), loc);
    // cuStreamSynchronize((CUstream)stream(sid));
  }
  void Cuda::CudaContext::syncCompute(const source_location &loc) const {
    checkError(cudaStreamSynchronize((cudaStream_t)streamCompute()), loc);
    // cuStreamSynchronize((CUstream)streamCompute());
  }
  void Cuda::CudaContext::syncStreamSpare(StreamID sid, const source_location &loc) const {
    checkError(cudaStreamSynchronize((cudaStream_t)streamSpare(sid)), loc);
    // cuStreamSynchronize((CUstream)streamSpare(sid));
  }
  // stream-event sync
  void Cuda::CudaContext::computeStreamWaitForEvent(void *event, const source_location &loc) {
    checkError(cudaStreamWaitEvent((cudaStream_t)streamCompute(), (cudaEvent_t)event, 0), loc);
    // cuStreamWaitEvent((CUstream)streamCompute(), (CUevent)event, 0);
  }
  void Cuda::CudaContext::spareStreamWaitForEvent(StreamID sid, void *event,
                                                  const source_location &loc) {
    checkError(cudaStreamWaitEvent((cudaStream_t)streamSpare(sid), (cudaEvent_t)event, 0), loc);
    // cuStreamWaitEvent((CUstream)streamSpare(sid), (CUevent)event, 0);
  }
  void *Cuda::CudaContext::streamMemAlloc(size_t size, void *stream, const source_location &loc) {
    void *ptr;
    cuMemAllocAsync((CUdeviceptr *)&ptr, size, (CUstream)stream);
    return ptr;
  }
  void Cuda::CudaContext::streamMemFree(void *ptr, void *stream, const source_location &loc) {
    cuMemFreeAsync((CUdeviceptr)ptr, (CUstream)stream);
  }
  Cuda::CudaContext::StreamExecutionTimer *Cuda::CudaContext::tick(void *stream,
                                                                   const source_location &loc) {
    return new StreamExecutionTimer(this, stream, loc);
  }
  void Cuda::CudaContext::tock(Cuda::CudaContext::StreamExecutionTimer *timer,
                               const source_location &loc) {
    cuLaunchHostFunc((CUstream)timer->stream, (CUhostFn)recycle_timer, (void *)timer);
  }

  void Cuda::CudaContext::setContext(const source_location &loc) const {
    const char *errString = nullptr;
    auto ec = cuCtxSetCurrent((CUcontext)getContext());
    if (ec != CUDA_SUCCESS) {
      cuGetErrorString(ec, &errString);
      checkCuApiError((u32)ec, loc, "[Cuda::CudaContext::setContext]", errString);
    }
  }

  bool Cuda::set_default_device(int dev, const source_location &loc) {
    auto &inst = driver();
    if (dev == inst.defaultDevice || dev >= inst.numTotalDevice || dev < 0) return false;
    inst.defaultDevice = dev;
    return prepare_context(mem_device, dev, loc);
  }
  int Cuda::get_default_device() noexcept { return driver().defaultDevice; }

  Cuda::Cuda() {
    fmt::print("[Init -- Begin] Cuda\n");
    errorStatus = false;
    CUresult res = cuInit(0);

    numTotalDevice = 0;
    cuDeviceGetCount(&numTotalDevice);
    contexts.resize(numTotalDevice);
    if (numTotalDevice == 0)
      fmt::print(
          "\t[InitInfo -- DevNum] There are no available device(s) that "
          "support CUDA\n");
    else
      fmt::print("\t[InitInfo -- DevNum] Detected {} CUDA Capable device(s)\n", numTotalDevice);

    defaultDevice = 0;
    {
      CUcontext ctx = nullptr;
      auto ec = cuCtxGetCurrent(&ctx);
      if (ec != CUDA_SUCCESS) {
        const char *errString = nullptr;
        cuGetErrorString(ec, &errString);
        checkCuApiError((u32)ec, errString);
      } else {
        int devid = defaultDevice;
        if (ctx != NULL) {
          auto ec = cuCtxGetDevice(&devid);
          if (ec != CUDA_SUCCESS) {
            const char *errString = nullptr;
            cuGetErrorString(ec, &errString);
            checkCuApiError((u32)ec, errString);
          } else
            defaultDevice = devid;  // record for restore later
        }                           // otherwise, no context has been initialized yet.
      }
    }
    for (int i = 0; i < numTotalDevice; i++) {
      auto &context = contexts[i];
      int dev{};
      {
        void *ctx{nullptr};
        // checkError(cudaSetDevice(i), i);
        cuDeviceGet((CUdevice *)&dev, i);
        fmt::print("device ordinal {} has handle {}\n", i, dev);

        unsigned int ctxFlags, expectedFlags = CU_CTX_SCHED_AUTO;
        // unsigned int ctxFlags, expectedFlags = CU_CTX_SCHED_BLOCKING_SYNC;
        int isActive;
        cuDevicePrimaryCtxGetState((CUdevice)dev, &ctxFlags, &isActive);

        /// follow tensorflow's impl
        if (ctxFlags != expectedFlags) {
          if (isActive) {
            ZS_ERROR(
                fmt::format("The primary active context has flag [{}], but [{}] is expected.\n",
                            ctxFlags, expectedFlags)
                    .data());
          } else {
            cuDevicePrimaryCtxSetFlags((CUdevice)dev, expectedFlags);
          }
        }

        void *formerCtx;
        int formerDev;
        cuCtxGetCurrent((CUcontext *)&formerCtx);
        res = cuDevicePrimaryCtxRetain((CUcontext *)&ctx, (CUdevice)dev);
        if (formerCtx != nullptr) {
          cuCtxGetDevice(&formerDev);
          ZS_ERROR_IF(formerDev == dev,
                      fmt::format("setting device [{}], yet the current device handle is {}.", dev,
                                  formerDev));
          if (formerCtx == ctx) {
            ZS_INFO(fmt::format("The primary context [{}] for device {} exists.", formerCtx,
                                formerDev));
          } else {
            ZS_WARN(fmt::format(
                "A non-primary context [{}] for device {} exists. The primary context is now {}.",
                formerCtx, formerDev, ctx));
          }
        }
        cuCtxSetCurrent((CUcontext)ctx);  // not sure why this is meaningful
        if (res == CUDA_SUCCESS) {
          // add this new context
          context = CudaContext{i, dev, ctx};
        } else if (res == CUDA_ERROR_OUT_OF_MEMORY) {
          size_t nbs;
          cuDeviceTotalMem(&nbs, (CUdevice)dev);
          ZS_WARN(fmt::format("{} bytes in total for device {}.", nbs, dev));
        }
      }

      context.streams.resize((int)StreamIndex::Total);
      for (auto &stream : context.streams)
        cuStreamCreate((CUstream *)&stream, CU_STREAM_DEFAULT);  // safer to sync with stream 0
      /// @note event for default stream is the last
      context.events.resize((int)EventIndex::Total);
      for (auto &event : context.events) cuEventCreate((CUevent *)&event, CU_EVENT_BLOCKING_SYNC);

      {  ///< device properties
        int major, minor, multiGpuBoardGroupID, regsPerBlock;
        int supportUnifiedAddressing, supportUm, supportConcurrentUmAccess;
        cuDeviceGetAttribute(&regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
        cuDeviceGetAttribute(&multiGpuBoardGroupID, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
                             (CUdevice)dev);
        cuDeviceGetAttribute(&textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, (CUdevice)dev);
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, (CUdevice)dev);
        cuDeviceGetAttribute(&supportUnifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                             (CUdevice)dev);
        cuDeviceGetAttribute(&supportUm, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, (CUdevice)dev);
        cuDeviceGetAttribute(&supportConcurrentUmAccess,
                             CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, (CUdevice)dev);
        cuDeviceGetAttribute(&context.numMultiprocessor, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                             (CUdevice)dev);
        cuDeviceGetAttribute(&context.regsPerMultiprocessor,
                             CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, (CUdevice)dev);
        cuDeviceGetAttribute(&context.sharedMemPerMultiprocessor,
                             CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev);
        cuDeviceGetAttribute(&context.maxBlocksPerMultiprocessor,
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, (CUdevice)dev);
        cuDeviceGetAttribute(&context.sharedMemPerBlock,
                             CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, (CUdevice)dev);
        cuDeviceGetAttribute(&context.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             (CUdevice)dev);
        cuDeviceGetAttribute(&context.maxThreadsPerMultiprocessor,
                             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, (CUdevice)dev);

        context.supportConcurrentUmAccess = supportConcurrentUmAccess;

        fmt::print(
            "\t[InitInfo -- Dev Property] GPU device {} ({}-th group on "
            "board)\n\t\tshared memory per block: {} bytes,\n\t\tregisters per SM: "
            "{},\n\t\tMulti-Processor count: {},\n\t\tSM compute capabilities: "
            "{}.{}.\n\t\tTexture alignment: {} bytes\n\t\tUVM support: allocation({}), unified "
            "addressing({}), concurrent access({})\n",
            i, multiGpuBoardGroupID, context.sharedMemPerBlock, regsPerBlock,
            context.numMultiprocessor, major, minor, textureAlignment, supportUm,
            supportUnifiedAddressing, supportConcurrentUmAccess);
      }
    }

    /// enable peer access if feasible
    for (int i = 0; i < numTotalDevice; i++) {
      // checkError(cudaSetDevice(i), i);
      cuCtxSetCurrent((CUcontext)contexts[i].getContext());
      for (int j = 0; j < numTotalDevice; j++) {
        if (i != j) {
          int iCanAccessPeer = 0;
          cuDeviceCanAccessPeer(&iCanAccessPeer, contexts[i].getDevice(), contexts[j].getDevice());
          if (iCanAccessPeer) cuCtxEnablePeerAccess((CUcontext)contexts[j].getContext(), 0);
          fmt::print("\t[InitInfo -- Peer Access] Peer access status {} -> {}: {}\n", i, j,
                     iCanAccessPeer ? "Inactive" : "Active");
        }
      }
    }
    // select gpu 0 by default
    cuCtxSetCurrent((CUcontext)contexts[defaultDevice].getContext());
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
    /* GPUs with SM architecture 6.x or higher (Pascal class or newer) provide additional
    Unified Memory features such as on-demand page migration and GPU memory oversubscription
    that are outlined throughout this document. Note that currently these features are only
    supported on Linux operating systems. Applications running on Windows (whether in TCC
    or WDDM mode) will use the basic Unified Memory model as on pre-6.x architectures even
    when they are running on hardware with compute capability 6.x or higher. */

    fmt::print("\n[Init -- End] == Finished \'Cuda\' initialization\n\n");
  }

  Cuda::~Cuda() {
    // do not do anything, let driver recycle resources
  }

  /// reference: kokkos/core/src/Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp, Ln 101
  int Cuda::deduce_block_size(const source_location &loc, const Cuda::CudaContext &ctx,
                              void *kernelFunc, function<size_t(int)> block_size_to_dynamic_shmem,
                              std::string_view kernelName) {
    if (auto it = ctx.funcLaunchConfigs.find(kernelFunc); it != ctx.funcLaunchConfigs.end())
      return it->second.optBlockSize;
    cudaFuncAttributes funcAttribs;
    ctx.checkError(cudaFuncGetAttributes(&funcAttribs, kernelFunc), loc);
    int optBlockSize{0};

    auto cuda_max_active_blocks_per_sm = [&](int block_size, int dynamic_shmem) {
      // Limits due do registers/SM
      int const regs_per_sm = ctx.regsPerMultiprocessor;
      int const regs_per_thread = funcAttribs.numRegs;
      int const max_blocks_regs = regs_per_sm / (regs_per_thread * block_size);

      // Limits due to shared memory/SM
      size_t const shmem_per_sm = ctx.sharedMemPerMultiprocessor;
      size_t const shmem_per_block = ctx.sharedMemPerBlock;
      size_t const static_shmem = funcAttribs.sharedSizeBytes;
      size_t const dynamic_shmem_per_block = funcAttribs.maxDynamicSharedSizeBytes;
      size_t const total_shmem = static_shmem + dynamic_shmem;

      int const max_blocks_shmem
          = total_shmem > shmem_per_block || dynamic_shmem > dynamic_shmem_per_block
                ? 0
                : (total_shmem > 0 ? (int)shmem_per_sm / total_shmem : max_blocks_regs);

      // Limits due to blocks/SM
      int const max_blocks_per_sm = ctx.maxBlocksPerMultiprocessor;

      // Overall occupancy in blocks
      return std::min({max_blocks_regs, max_blocks_shmem, max_blocks_per_sm});
    };
    auto deduce_opt_block_size = [&]() {
      // Limits
      int const max_threads_per_sm = ctx.maxThreadsPerMultiprocessor;
      // unsure if I need to do that or if this is already accounted for in the functor attributes
      int const min_blocks_per_sm = 1;
      int const max_threads_per_block
          = std::min((int)ctx.maxThreadsPerBlock, funcAttribs.maxThreadsPerBlock);

      // Recorded maximum
      int opt_block_size = 0;
      int opt_threads_per_sm = 0;

      /// iterate all optional blocksize setup
      for (int block_size = max_threads_per_block; block_size > 0; block_size -= 32) {
        size_t const dynamic_shmem = block_size_to_dynamic_shmem(block_size);

        int blocks_per_sm = cuda_max_active_blocks_per_sm(block_size, dynamic_shmem);

        int threads_per_sm = blocks_per_sm * block_size;

        if (threads_per_sm > max_threads_per_sm) {
          blocks_per_sm = max_threads_per_sm / block_size;
          threads_per_sm = blocks_per_sm * block_size;
        }

        // update if higher occupancy (more threads per streaming multiprocessor)
        if (blocks_per_sm >= min_blocks_per_sm) {
          if (threads_per_sm >= opt_threads_per_sm) {
            opt_block_size = block_size;
            opt_threads_per_sm = threads_per_sm;
          }
        }

        // fmt::print("current blocks_sm: {}, threads_sm: {}, size: {}\n", blocks_per_sm,
        //           threads_per_sm, block_size);
        // if (blocks_per_sm != 0) break; // this enabled when querying for maximum block size
      }
      return opt_block_size;
    };

    optBlockSize = deduce_opt_block_size();
    fmt::print(
        fg(fmt::color::lime_green) | fmt::emphasis::bold,
        "{:=^60}\nnumRegs: {}\t\tmaxThreadsPerBlock: {}\nsharedSizeBytes: {}\t"
        "maxDynamicSharedSizeBytes: {}.\n",
        fmt::format(" cuda kernel [{}] optBlockSize [{}] ",
                    kernelName.empty() ? std::to_string((std::uintptr_t)kernelFunc) : kernelName,
                    optBlockSize),
        funcAttribs.numRegs, funcAttribs.maxThreadsPerBlock, funcAttribs.sharedSizeBytes,
        funcAttribs.maxDynamicSharedSizeBytes);
    ctx.funcLaunchConfigs.emplace(kernelFunc, typename Cuda::CudaContext::Config{optBlockSize});
    return optBlockSize;
  }

}  // namespace zs
