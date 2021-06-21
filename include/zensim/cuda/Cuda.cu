#include <utility>

#include "../Platform.hpp"
#include "Cuda.h"
#include "CudaConstants.inc"
#include "zensim/tpls/fmt/core.h"


#define MEM_POOL_CTRL 3

namespace zs {

  std::string get_cu_error_message(uint32_t err) {
    const char *err_name_ptr;
    const char *err_string_ptr;
    Cuda::instance().get_cu_error_name(err, &err_name_ptr);
    Cuda::instance().get_cu_error_string(err, &err_string_ptr);
    return fmt::format("CUDA Driver Error {}: {}", err_name_ptr, err_string_ptr);
  }

  std::string get_cuda_error_message(uint32_t err) {
    return fmt::format("CUDA Runtime Error {}: {}", Cuda::instance().get_cuda_error_name(err),
                       Cuda::instance().get_cuda_error_string(err));
  }

  Cuda::Cuda() {
    fmt::print("[Init -- Begin] Cuda\n");

    {
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
  name.set_lock(&lock);                                \
  name.set_names(#name, #symbol_name);
#include "cuda_driver_functions.inc.h"
#undef PER_CUDA_FUNCTION
    }
    {
#if defined(ZS_PLATFORM_LINUX)
      runtimeLoader.reset(new DynamicLoader("libcudart.so"));
#elif defined(ZS_PLATFORM_WINDOWS)
      int version{0};
      getDriverVersion(&version);
      auto suf = std::to_string(version / 100);
      auto cudaDllName = std::string("cudart64_") + suf + ".dll";
      fmt::print("loading cuda dll: {}\n", cudaDllName);
      runtimeLoader.reset(new DynamicLoader(cudaDllName.c_str())); //nvcudart.dll"));
#else
      static_assert(false, "CUDA library supports only Windows and Linux.");
#endif
      runtimeLoader->load_function("cudaGetErrorName", get_cuda_error_name);
      runtimeLoader->load_function("cudaGetErrorString", get_cuda_error_string);

#define PER_CUDA_FUNCTION(name, symbol_name, ...)       \
  name.set(runtimeLoader->load_function(#symbol_name)); \
  name.set_lock(&lock);                                 \
  name.set_names(#name, #symbol_name);
#include "cuda_runtime_functions.inc.h"
#undef PER_CUDA_FUNCTION
    }

    int version;
    getDriverVersion(&version);

    ZS_TRACE("CUDA driver API (v{}.{}) loaded.", version / 1000, version % 1000 / 10);

    init(0);
    numTotalDevice = 0;
    getDeviceCount(&numTotalDevice);
    if (numTotalDevice == 0)
      fmt::print(
          "\t[InitInfo -- DevNum] There are no available device(s) that "
          "support CUDA\n");
    else
      fmt::print("\t[InitInfo -- DevNum] Detected {} CUDA Capable device(s)\n", numTotalDevice);

    for (int i = 0; i < numTotalDevice; i++) {
      cudaSetDevice(i);
      void *dev{nullptr};
      {
        void *context;
        getDevice(&dev, i);
        createContext(&context, 0, dev);
        contexts.emplace_back(i, dev, context);  //< set device when construct
        setContext(context);
      }
      auto &context = contexts.back();
      context.streams.resize((int)StreamIndex::Total);
      for (auto &stream : context.streams) createStream(&stream);  ///< CU_STREAM_DEFAULT
      context.events.resize((int)EventIndex::Total);
      for (auto &event : context.events) createEvent(&event, CU_EVENT_DISABLE_TIMING);

      /// device properties
      int major, minor, multiGpuBoardGroupID, multiProcessorCount, sharedMemPerBlock, regsPerBlock;
      getDeviceAttribute(&sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
      getDeviceAttribute(&regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
      getDeviceAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
      getDeviceAttribute(&multiGpuBoardGroupID, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, dev);
      getDeviceAttribute(&textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, dev);
      getDeviceAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
      getDeviceAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);

      fmt::print(
          "\t[InitInfo -- Dev Property] GPU device {} ({}-th group on "
          "board)\n\t\tshared memory per block: {} bytes,\n\t\tregisters per SM: "
          "{},\n\t\tMulti-Processor count: {},\n\t\tSM compute capabilities: "
          "{}.{}.\n\t\tTexture alignment: {} bytes\n",
          i, multiGpuBoardGroupID, sharedMemPerBlock, regsPerBlock, multiProcessorCount, major,
          minor, textureAlignment);
    }

    /// enable peer access if feasible
    for (int i = 0; i < numTotalDevice; i++) {
      setContext(contexts[i].getContext());
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

    fmt::print("\n[Init -- End] == Finished \'Cuda\' initialization\n\n");
  }

  Cuda::~Cuda() {
    for (int i = 0; i < numTotalDevice; i++) {
      auto &context = contexts[i];
      context.setContext();
      for (auto &stream : context.streams) destroyStream(stream);
      for (auto &event : context.events) destroyEvent(event);
      context.deviceMem.reset(nullptr);
      context.unifiedMem.reset(nullptr);
      context.launchMem.reset(nullptr);

      destroyContext(context.getContext());
    }
    fmt::print("  Finished \'Cuda\' termination\n");
  }

  void Cuda::CudaContext::initDeviceMemory() {
    /// memory
    std::size_t free_byte, total_byte;
    driver().memInfo(&free_byte, &total_byte);
    deviceMem = std::make_unique<MonotonicAllocator>(free_byte >> MEM_POOL_CTRL,
                                                     driver().textureAlignment);
    fmt::print(
        "\t[InitInfo -- memory] device {}\n\t\tfree bytes/total bytes: "
        "{}/{},\n\t\tpre-allocated device memory: {} bytes\n\n",
        getDevId(), free_byte, total_byte, (free_byte >> MEM_POOL_CTRL));
  }
  void Cuda::CudaContext::initUnifiedMemory() {
    std::size_t free_byte, total_byte;
    driver().memInfo(&free_byte, &total_byte);
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
    if (!unifiedMem) initUnifiedMemory();
    return unifiedMem->borrow(bytes);
  }
  void Cuda::CudaContext::resetVirtualMem() {
    if (!unifiedMem) initUnifiedMemory();
    unifiedMem->reset();
  }

}  // namespace zs
