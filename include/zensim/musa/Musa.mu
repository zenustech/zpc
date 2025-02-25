#include "Musa.h"
//
#include <musa.h>
#include <musa_runtime.h>

#include "../Logger.hpp"
#include "zensim/musa/memory/MemOps.hpp"
#include "zensim/zpc_tpls/fmt/color.h"

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
#if 0
    // NOT working!
    return muLaunchKernel((MUfunction)f, gx, gy, gz, bx, by, bz, shmem, (MUstream)stream, args,
                          nullptr);
#else
    return musaLaunchKernel(f, dim3{gx, gy, gz}, dim3{bx, by, bz}, args, shmem,
                            (musaStream_t)stream);
#endif
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

  // record
  void Musa::MusaContext::recordEventCompute(const source_location &loc) {
    checkError(musaEventRecord((musaEvent_t)eventCompute(), (musaStream_t)streamCompute()), loc);
    // muEventRecord((MUevent)eventCompute(), (MUstream)streamCompute());
  }
  void Musa::MusaContext::recordEventSpare(StreamID id, const source_location &loc) {
    checkError(musaEventRecord((musaEvent_t)eventSpare(id), (musaStream_t)streamSpare(id)), loc);
    // muEventRecord((MUevent)eventSpare(id), (MUstream)streamSpare(id));
  }
  // sync
  void Musa::MusaContext::syncStream(StreamID sid, const source_location &loc) const {
    checkError(musaStreamSynchronize((musaStream_t)stream(sid)), loc);
    // muStreamSynchronize((MUstream)stream(sid));
  }
  void Musa::MusaContext::syncCompute(const source_location &loc) const {
    checkError(musaStreamSynchronize((musaStream_t)streamCompute()), loc);
    // muStreamSynchronize((MUstream)streamCompute());
  }
  void Musa::MusaContext::syncStreamSpare(StreamID sid, const source_location &loc) const {
    checkError(musaStreamSynchronize((musaStream_t)streamSpare(sid)), loc);
    // muStreamSynchronize((MUstream)streamSpare(sid));
  }
  // stream-event sync
  void Musa::MusaContext::computeStreamWaitForEvent(void *event, const source_location &loc) {
    checkError(musaStreamWaitEvent((musaStream_t)streamCompute(), (musaEvent_t)event, 0), loc);
    // muStreamWaitEvent((MUstream)streamCompute(), (MUevent)event, 0);
  }
  void Musa::MusaContext::spareStreamWaitForEvent(StreamID sid, void *event,
                                                  const source_location &loc) {
    checkError(musaStreamWaitEvent((musaStream_t)streamSpare(sid), (musaEvent_t)event, 0), loc);
    // muStreamWaitEvent((MUstream)streamSpare(sid), (MUevent)event, 0);
  }

  // stream ordered memory allocator
  void *Musa::MusaContext::streamMemAlloc(size_t size, void *stream, const source_location &loc) {
    void *ptr;
/// TODO:
#if 1
    muMemAlloc((MUdeviceptr *)&ptr, size);
#elif 1
    musaMallocAsync((void **)&ptr, size, (musaStream_t)stream);
#else
#endif
    return ptr;
  }
  void Musa::MusaContext::streamMemFree(void *ptr, void *stream, const source_location &loc) {
/// TODO:
#if 1
    muStreamSynchronize((MUstream)stream);
    muMemFree((MUdeviceptr)ptr);
#elif 0
    musaFreeAsync(ptr, (musaStream_t)stream);
#else
    muMemFreeAsync((MUdeviceptr)ptr, (MUstream)stream);
#endif
  }
  Musa::MusaContext::StreamExecutionTimer *Musa::MusaContext::tick(void *stream,
                                                                   const source_location &loc) {
    return new StreamExecutionTimer(this, stream, loc);
  }
  void Musa::MusaContext::tock(Musa::MusaContext::StreamExecutionTimer *timer,
                               const source_location &loc) {
    muLaunchHostFunc((MUstream)timer->stream, (MUhostFn)recycle_timer, (void *)timer);
  }

  bool Musa::set_default_device(int dev, const source_location &loc) {
    auto &inst = driver();
    if (dev == inst.defaultDevice || dev >= inst.numTotalDevice || dev < 0) return false;
    inst.defaultDevice = dev;
    return prepare_context(mem_device, dev, loc);
  }
  int Musa::get_default_device() noexcept { return driver().defaultDevice; }

  Musa::Musa() {
    fmt::print("[Init -- Begin] Musa\n");
    errorStatus = false;
    MUresult res = muInit(0);

    numTotalDevice = 0;
    muDeviceGetCount(&numTotalDevice);
    contexts.resize(numTotalDevice);
    if (numTotalDevice == 0)
      fmt::print(
          "\t[InitInfo -- DevNum] There are no available device(s) that "
          "support MUSA\n");
    else
      fmt::print("\t[InitInfo -- DevNum] Detected {} MUSA Capable device(s)\n", numTotalDevice);

    defaultDevice = 0;
    {
      MUcontext ctx = nullptr;
      auto ec = muCtxGetCurrent(&ctx);
      if (ec != MUSA_SUCCESS) {
        const char *errString = nullptr;
        muGetErrorString(ec, &errString);
        checkMuApiError((u32)ec, errString);
      } else {
        int devid = defaultDevice;
        if (ctx != NULL) {
          auto ec = muCtxGetDevice(&devid);
          if (ec != MUSA_SUCCESS) {
            const char *errString = nullptr;
            muGetErrorString(ec, &errString);
            checkMuApiError((u32)ec, errString);
          } else
            defaultDevice = devid;  // record for restore later
        }  // otherwise, no context has been initialized yet.
      }
    }

    for (int i = 0; i < numTotalDevice; i++) {
      auto &context = contexts[i];
      int dev{};
      {
        void *ctx{nullptr};
        // checkError(musaSetDevice(i), i);
        muDeviceGet((MUdevice *)&dev, i);
        fmt::print("device ordinal {} has handle {}\n", i, dev);

        unsigned int ctxFlags, expectedFlags = MU_CTX_SCHED_AUTO;
        // unsigned int ctxFlags, expectedFlags = MU_CTX_SCHED_BLOCKING_SYNC;
        int isActive;
        muDevicePrimaryCtxGetState((MUdevice)dev, &ctxFlags, &isActive);

        /// follow tensorflow's impl
        if (ctxFlags != expectedFlags) {
          if (isActive) {
            ZS_ERROR(
                fmt::format("The primary active context has flag [{}], but [{}] is expected.\n",
                            ctxFlags, expectedFlags)
                    .data());
          } else {
            muDevicePrimaryCtxSetFlags((MUdevice)dev, expectedFlags);
          }
        }

        void *formerCtx;
        int formerDev;
        muCtxGetCurrent((MUcontext *)&formerCtx);
        res = muDevicePrimaryCtxRetain((MUcontext *)&ctx, (MUdevice)dev);
        if (formerCtx != nullptr) {
          muCtxGetDevice(&formerDev);
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
        muCtxSetCurrent((MUcontext)ctx);  // not sure why this is meaningful
        if (res == MUSA_SUCCESS) {
          // add this new context
          context = MusaContext{i, dev, ctx};
        } else if (res == MUSA_ERROR_OUT_OF_MEMORY) {
          size_t nbs;
          muDeviceTotalMem(&nbs, (MUdevice)dev);
          ZS_WARN(fmt::format("{} bytes in total for device {}.", nbs, dev));
        }
      }

      context.streams.resize((int)StreamIndex::Total);
      for (auto &stream : context.streams)
        muStreamCreate((MUstream *)&stream, MU_STREAM_DEFAULT);  // safer to sync with stream 0
      /// @note event for default stream is the last
      context.events.resize((int)EventIndex::Total);
      for (auto &event : context.events) muEventCreate((MUevent *)&event, MU_EVENT_BLOCKING_SYNC);

      {  ///< device properties
        int major, minor, multiGpuBoardGroupID, regsPerBlock;
        int supportUnifiedAddressing, supportUm, supportConcurrentUmAccess;
        muDeviceGetAttribute(&regsPerBlock, MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
        muDeviceGetAttribute(&multiGpuBoardGroupID, MU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
                             (MUdevice)dev);
        muDeviceGetAttribute(&textureAlignment, MU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, dev);
        muDeviceGetAttribute(&minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, (MUdevice)dev);
        muDeviceGetAttribute(&major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, (MUdevice)dev);
        muDeviceGetAttribute(&supportUnifiedAddressing, MU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                             (MUdevice)dev);
        muDeviceGetAttribute(&supportUm, MU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, (MUdevice)dev);
        muDeviceGetAttribute(&supportConcurrentUmAccess,
                             MU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, (MUdevice)dev);
        muDeviceGetAttribute(&context.numMultiprocessor, MU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                             (MUdevice)dev);
        muDeviceGetAttribute(&context.regsPerMultiprocessor,
                             MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, (MUdevice)dev);
        muDeviceGetAttribute(&context.sharedMemPerMultiprocessor,
                             MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev);
        muDeviceGetAttribute(&context.maxBlocksPerMultiprocessor,
                             MU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, (MUdevice)dev);
        muDeviceGetAttribute(&context.sharedMemPerBlock,
                             MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, (MUdevice)dev);
        muDeviceGetAttribute(&context.maxThreadsPerBlock, MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             (MUdevice)dev);
        muDeviceGetAttribute(&context.maxThreadsPerMultiprocessor,
                             MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, (MUdevice)dev);

        context.supportConcurrentUmAccess = supportConcurrentUmAccess;

        fmt::print(
            "\t[InitInfo -- Dev Property] MUSA device {} ({}-th group on "
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
      // checkError(musaSetDevice(i), i);
      muCtxSetCurrent((MUcontext)contexts[i].getContext());
      for (int j = 0; j < numTotalDevice; j++) {
        if (i != j) {
          int iCanAccessPeer = 0;
          muDeviceCanAccessPeer(&iCanAccessPeer, contexts[i].getDevice(), contexts[j].getDevice());
          if (iCanAccessPeer) muCtxEnablePeerAccess((MUcontext)contexts[j].getContext(), 0);
          fmt::print("\t[InitInfo -- Peer Access] Peer access status {} -> {}: {}\n", i, j,
                     iCanAccessPeer ? "Inactive" : "Active");
        }
      }
    }
    // select gpu 0 by default
    muCtxSetCurrent((MUcontext)contexts[defaultDevice].getContext());

    fmt::print("\n[Init -- End] == Finished \'Musa\' initialization\n\n");
  }
  Musa::~Musa() {}

  int Musa::deduce_block_size(const source_location &loc, const Musa::MusaContext &ctx,
                              void *kernelFunc, function<size_t(int)> block_size_to_dynamic_shmem,
                              std::string_view kernelName) {
    if (auto it = ctx.funcLaunchConfigs.find(kernelFunc); it != ctx.funcLaunchConfigs.end())
      return it->second.optBlockSize;
    musaFuncAttributes funcAttribs;
    ctx.checkError(musaFuncGetAttributes(&funcAttribs, kernelFunc), loc);
    int optBlockSize{0};

    // printf("numregs: %d, numregs sm: %d, sharedSizeBytes: %d, maxDynamicSharedSizeBytes: %d,
    // threads: %d\n",
    //        (int)funcAttribs.numRegs, (int)ctx.regsPerMultiprocessor,
    //        (int)funcAttribs.sharedSizeBytes, (int)funcAttribs.maxDynamicSharedSizeBytes,
    //        (int)funcAttribs.maxThreadsPerBlock);

    auto musa_max_active_blocks_per_sm = [&](int block_size, int dynamic_shmem) {
      // Limits due do registers/SM
      int const regs_per_sm = ctx.regsPerMultiprocessor;
      int const regs_per_thread = std::max(funcAttribs.numRegs, 1);
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

        int blocks_per_sm = musa_max_active_blocks_per_sm(block_size, dynamic_shmem);

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
        fmt::format(" musa kernel [{}] optBlockSize [{}] ",
                    kernelName.empty() ? std::to_string((std::uintptr_t)kernelFunc) : kernelName,
                    optBlockSize),
        funcAttribs.numRegs, funcAttribs.maxThreadsPerBlock, funcAttribs.sharedSizeBytes,
        funcAttribs.maxDynamicSharedSizeBytes);
    ctx.funcLaunchConfigs.emplace(kernelFunc, typename Musa::MusaContext::Config{optBlockSize});
    return optBlockSize;
  }

}  // namespace zs