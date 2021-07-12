#include <utility>

#include "../Platform.hpp"
#include "Cuda.h"
#include "CudaConstants.inc"
#include "zensim/tpls/fmt/format.h"
#include "zensim/types/SourceLocation.hpp"

#define MEM_POOL_CTRL 3

namespace zs {

  __device__ __constant__ char g_cuda_constant_cache[8192];  // 1024 words

  void Cuda::init_constant_cache(void *ptr, std::size_t size) {
    cudaMemcpyToSymbol(g_cuda_constant_cache, ptr, size, 0, cudaMemcpyHostToDevice);
  }
  void Cuda::init_constant_cache(void *ptr, std::size_t size, void *stream) {
    cudaMemcpyToSymbolAsync(g_cuda_constant_cache, ptr, size, 0, cudaMemcpyHostToDevice,
                            (cudaStream_t)stream);
  }

  std::string get_cu_error_message(uint32_t err) {
    const char *err_name_ptr;
    const char *err_string_ptr;
    Cuda::instance().get_cu_error_name(err, &err_name_ptr);
    Cuda::instance().get_cu_error_string(err, &err_string_ptr);
    return fmt::format("CUDA Driver Error {}: {}", err_name_ptr, err_string_ptr);
  }

#if 0
  std::string get_cuda_error_message(uint32_t err) {
    return fmt::format("CUDA Runtime Error {}: {}", Cuda::instance().get_cuda_error_name(err),
                       Cuda::instance().get_cuda_error_string(err));
  }
#endif

  /// error handling
  u32 Cuda::getLastCudaError() { return (u32)cudaGetLastError(); }

  std::string_view Cuda::getCudaErrorString(u32 errorCode) {
    return cudaGetErrorString((cudaError_t)errorCode);
  }
  void Cuda::checkError(u32 errorCode, ProcID did, const source_location &loc) {
    if (errorCode != 0) {
      if (did >= 0) {
        auto &context = Cuda::context(did);
        if (context.errorStatus) return;  // there already exists a preceding cuda error
        context.errorStatus = true;
      }
      const auto fileInfo = fmt::format("# File: \"{}\"", loc.file_name());
      const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
      const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());
      fmt::print(fg(fmt::color::crimson) | fmt::emphasis::italic | fmt::emphasis::bold,
                 "\nCuda Error on Device {}: {}\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
                 did >= 0 ? std::to_string(did) : "unknown", Cuda::getCudaErrorString(errorCode),
                 " cuda api error location ", fileInfo, locInfo, funcInfo, "=");
    }
  }

  /// kernel launch
  u32 Cuda::launchKernel(const void *f, unsigned int gx, unsigned int gy, unsigned int gz,
                         unsigned int bx, unsigned int by, unsigned int bz, void **args,
                         std::size_t shmem, void *stream) {
    return cudaLaunchKernel(f, dim3{gx, gy, gz}, dim3{bx, by, bz}, args, shmem,
                            (cudaStream_t)stream);
  }
  u32 Cuda::launchCooperativeKernel(const void *f, unsigned int gx, unsigned int gy,
                                    unsigned int gz, unsigned int bx, unsigned int by,
                                    unsigned int bz, void **args, std::size_t shmem, void *stream) {
    return cudaLaunchCooperativeKernel(f, dim3{gx, gy, gz}, dim3{bx, by, bz}, args, shmem,
                                       (cudaStream_t)stream);
  }
  u32 Cuda::launchCallback(void *stream, void *f, void *data) {
    return cudaLaunchHostFunc((cudaStream_t)stream, (cudaHostFn_t)f, data);
  }

  void Cuda::CudaContext::checkError() const {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
      fmt::print("Last Error on [Dev {}]: {}\n", devid, cudaGetErrorString(error));
  }

  void Cuda::CudaContext::checkError(u32 errorCode, const source_location &loc) const {
    /// only shows the first error message
    Cuda::checkError(errorCode, getDevId(), loc);
  }

  // record
  void Cuda::CudaContext::recordEventCompute(const source_location &loc) {
    checkError(cudaEventRecord((cudaEvent_t)eventCompute(), (cudaStream_t)streamCompute()), loc);
  }
  void Cuda::CudaContext::recordEventSpare(unsigned id, const source_location &loc) {
    checkError(cudaEventRecord((cudaEvent_t)eventSpare(id), (cudaStream_t)streamSpare(id)), loc);
  }
  // sync
  void Cuda::CudaContext::syncStream(unsigned sid, const source_location &loc) const {
    checkError(cudaStreamSynchronize((cudaStream_t)stream(sid)), loc);
  }
  void Cuda::CudaContext::syncCompute(const source_location &loc) const {
    checkError(cudaStreamSynchronize((cudaStream_t)streamCompute()), loc);
  }
  void Cuda::CudaContext::syncStreamSpare(unsigned sid, const source_location &loc) const {
    checkError(cudaStreamSynchronize((cudaStream_t)streamSpare(sid)), loc);
  }
  // stream-event sync
  void Cuda::CudaContext::computeStreamWaitForEvent(void *event, const source_location &loc) {
    checkError(cudaStreamWaitEvent((cudaStream_t)streamCompute(), (cudaEvent_t)event, 0), loc);
  }
  void Cuda::CudaContext::spareStreamWaitForEvent(unsigned sid, void *event,
                                                  const source_location &loc) {
    checkError(cudaStreamWaitEvent((cudaStream_t)streamSpare(sid), (cudaEvent_t)event, 0), loc);
  }
  void *Cuda::CudaContext::streamMemAlloc(std::size_t size, void *stream,
                                          const source_location &loc) {
    void *ptr;
    checkError(cudaMallocAsync(&ptr, size, (cudaStream_t)stream), loc);
    return ptr;
  }
  void Cuda::CudaContext::streamMemFree(void *ptr, void *stream, const source_location &loc) {
    checkError(cudaFreeAsync(ptr, (cudaStream_t)stream), loc);
  }
  Cuda::CudaContext::StreamExecutionTimer *Cuda::CudaContext::tick(void *stream,
                                                                   const source_location &loc) {
    return new StreamExecutionTimer(this, stream, loc);
  }
  void Cuda::CudaContext::tock(Cuda::CudaContext::StreamExecutionTimer *timer,
                               const source_location &loc) {
    checkError(launchCallback(timer->stream, (void *)recycle_timer, (void *)timer), loc);
  }

  void Cuda::CudaContext::setContext(const source_location &loc) const {
    checkError(cudaSetDevice(devid), loc);
  }

  Cuda::Cuda() {
    fmt::print("[Init -- Begin] Cuda\n");

    {  // cuda driver api (for JIT)
#if defined(ZS_PLATFORM_LINUX)
      driverLoader = std::make_unique<DynamicLoader>("libcuda.so.1");
#elif defined(ZS_PLATFORM_WINDOWS)
      driverLoader = std::make_unique<DynamicLoader>("nvcuda.dll");
#else
      static_assert(false, "CUDA driver supports only Windows and Linux.");
#endif
      driverLoader->load_function("cuGetErrorName", get_cu_error_name);
      driverLoader->load_function("cuGetErrorString", get_cu_error_string);

#define PER_CUDA_FUNCTION(name, symbol_name, ...)      \
  name.set(driverLoader->load_function(#symbol_name)); \
  name.set_names(#name, #symbol_name);
#include "cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION
    }

    init(0);  // cuInit(0);

#if 0
    { // cuda runtime api
#  if defined(ZS_PLATFORM_LINUX)
      runtimeLoader.reset(new DynamicLoader("libcudart.so"));
#  elif defined(ZS_PLATFORM_WINDOWS)
      int version{0};
      getDriverVersion(&version);
      auto suf = std::to_string(version / 100);
      auto cudaDllName = std::string("cudart64_") + suf + ".dll";
      fmt::print("loading cuda runtime dll: {}\n", cudaDllName);
      runtimeLoader.reset(new DynamicLoader(cudaDllName.c_str())); //nvcudart.dll"));
#  else
      static_assert(false, "CUDA library supports only Windows and Linux.");
#  endif
      runtimeLoader->load_function("cudaGetErrorName", get_cuda_error_name);
      runtimeLoader->load_function("cudaGetErrorString", get_cuda_error_string);

#  define PER_CUDA_FUNCTION(name, symbol_name, ...)       \
    name.set(runtimeLoader->load_function(#symbol_name)); \
    name.set_names(#name, #symbol_name);
#  include "cuda_runtime_functions.inc.h"
#  undef PER_CUDA_FUNCTION
    }
#endif

    numTotalDevice = 0;
    getDeviceCount(&numTotalDevice);
    contexts.resize(numTotalDevice);
    if (numTotalDevice == 0)
      fmt::print(
          "\t[InitInfo -- DevNum] There are no available device(s) that "
          "support CUDA\n");
    else
      fmt::print("\t[InitInfo -- DevNum] Detected {} CUDA Capable device(s)\n", numTotalDevice);

    for (int i = 0; i < numTotalDevice; i++) {
      auto &context = contexts[i];
      int dev{};
      {
        void *c{nullptr};
        checkError(cudaSetDevice(i), i);
        getDevice(&dev, i);
        // fmt::print("device ordinal {} is {}\n", i, dev);

        // getContext(&c);
        retainPrimaryCtx(&c, dev);
        // createContext(&c, 4, dev); // CU_CTX_SCHED_BLOCKING_SYNC (0x04) | CU_CTX_SCHED_SPIN
        // (0x01)
        context = CudaContext{i, dev, c};
        // setContext(context.getContext());
      }

      context.streams.resize((int)StreamIndex::Total);
      for (auto &stream : context.streams)
        checkError(cudaStreamCreateWithFlags((cudaStream_t *)&stream, cudaStreamNonBlocking), i);
      context.events.resize((int)EventIndex::Total);
      for (auto &event : context.events)
        checkError(cudaEventCreateWithFlags((cudaEvent_t *)&event, cudaEventBlockingSync), i);

      /// device properties
      int major, minor, multiGpuBoardGroupID, multiProcessorCount, regsPerBlock;
      int supportUnifiedAddressing, supportUm, supportConcurrentUmAccess;
      getDeviceAttribute(&regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
      getDeviceAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
      getDeviceAttribute(&multiGpuBoardGroupID, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, dev);
      getDeviceAttribute(&textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, dev);
      getDeviceAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
      getDeviceAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
      getDeviceAttribute(&supportUnifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
      getDeviceAttribute(&supportUm, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev);
      getDeviceAttribute(&supportConcurrentUmAccess, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
                         dev);
      getDeviceAttribute(&context.regsPerMultiprocessor,
                         CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, dev);
      getDeviceAttribute(&context.sharedMemPerMultiprocessor,
                         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev);
      getDeviceAttribute(&context.maxBlocksPerMultiprocessor,
                         CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, dev);
      getDeviceAttribute(&context.sharedMemPerBlock,
                         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
      getDeviceAttribute(&context.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                         dev);
      getDeviceAttribute(&context.maxThreadsPerMultiprocessor,
                         CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);

      context.supportConcurrentUmAccess = supportConcurrentUmAccess;

      fmt::print(
          "\t[InitInfo -- Dev Property] GPU device {} ({}-th group on "
          "board)\n\t\tshared memory per block: {} bytes,\n\t\tregisters per SM: "
          "{},\n\t\tMulti-Processor count: {},\n\t\tSM compute capabilities: "
          "{}.{}.\n\t\tTexture alignment: {} bytes\n\t\tUVM support: allocation({}), unified "
          "addressing({}), concurrent access({})\n",
          i, multiGpuBoardGroupID, context.sharedMemPerBlock, regsPerBlock, multiProcessorCount,
          major, minor, textureAlignment, supportUm, supportUnifiedAddressing,
          supportConcurrentUmAccess);
    }

    /// enable peer access if feasible
    for (int i = 0; i < numTotalDevice; i++) {
      // setContext(contexts[i].getContext());
      checkError(cudaSetDevice(i), i);
      for (int j = 0; j < numTotalDevice; j++) {
        if (i != j) {
          int iCanAccessPeer = 0;
          canAccessPeer(&iCanAccessPeer, contexts[i].getDevice(), contexts[j].getDevice());
          if (iCanAccessPeer) enablePeerAccess(contexts[j].getContext(), 0);
          fmt::print("\t[InitInfo -- Peer Access] Peer access status {} -> {}: {}\n", i, j,
                     iCanAccessPeer ? "Inactive" : "Active");
        }
      }
    }
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
    for (int i = 0; i < numTotalDevice; i++) {
      auto &context = contexts[i];
      context.setContext();
      checkError(cudaDeviceSynchronize(), i);
      for (auto stream : context.streams) checkError(cudaStreamDestroy((cudaStream_t)stream), i);
      for (auto event : context.events) checkError(cudaEventDestroy((cudaEvent_t)event), i);
      context.deviceMem.reset(nullptr);
      context.unifiedMem.reset(nullptr);

      // destroyContext(context.getContext());
      checkError(cudaDeviceReset(), i);
    }
    fmt::print("  Finished \'Cuda\' termination\n");
  }

  void Cuda::CudaContext::initDeviceMemory() {
    /// memory
    std::size_t free_byte, total_byte;
    checkError(cudaMemGetInfo(&free_byte, &total_byte));
    deviceMem = std::make_unique<MonotonicAllocator>(free_byte >> MEM_POOL_CTRL,
                                                     driver().textureAlignment);
    fmt::print(
        "\t[InitInfo -- memory] device {}\n\t\tfree bytes/total bytes: "
        "{}/{},\n\t\tpre-allocated device memory: {} bytes\n\n",
        getDevId(), free_byte, total_byte, (free_byte >> MEM_POOL_CTRL));
  }
  void Cuda::CudaContext::initUnifiedMemory() {
#if defined(_WIN32)
    throw std::runtime_error("unified virtual memory manually disabled on windows!");
    return;
#endif
    std::size_t free_byte, total_byte;
    checkError(cudaMemGetInfo(&free_byte, &total_byte));
    unifiedMem = std::make_unique<MonotonicVirtualAllocator>(getDevId(), total_byte * 4,
                                                             driver().textureAlignment);
    fmt::print(
        "\t[InitInfo -- memory] device {}\n\t\tfree bytes/total bytes: "
        "{}/{},\n\t\tpre-allocated unified memory: {} bytes\n\n",
        getDevId(), free_byte, total_byte, total_byte * 4);
  }

  auto Cuda::CudaContext::borrow(std::size_t bytes) -> void * {
    if (!deviceMem) initDeviceMemory();
    return deviceMem->borrow(bytes);
  }
  void Cuda::CudaContext::resetMem() {
    if (!deviceMem) initDeviceMemory();
    deviceMem->reset();
  }

  auto Cuda::CudaContext::borrowVirtual(std::size_t bytes) -> void * {
#if defined(_WIN32)
    throw std::runtime_error("unified virtual memory manually disabled on windows!");
    return nullptr;
#endif
    if (!unifiedMem) initUnifiedMemory();
    return unifiedMem->borrow(bytes);
  }
  void Cuda::CudaContext::resetVirtualMem() {
    if (!unifiedMem) initUnifiedMemory();
    unifiedMem->reset();
  }

  /// reference: kokkos/core/src/Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp, Ln 101
  int Cuda::deduce_block_size(const Cuda::CudaContext &ctx, void *kernelFunc,
                              std::function<std::size_t(int)> block_size_to_dynamic_shmem,
                              std::string_view kernelName) {
    if (auto it = ctx.funcLaunchConfigs.find(kernelFunc); it != ctx.funcLaunchConfigs.end())
      return it->second.optBlockSize;
    cudaFuncAttributes funcAttribs;
    ctx.checkError(cudaFuncGetAttributes(&funcAttribs, kernelFunc));
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
