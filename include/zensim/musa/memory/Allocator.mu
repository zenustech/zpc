#include "Allocator.h"

#include <musa.h>

#include "zensim/Logger.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/memory/Allocator.h"
#include "zensim/musa/Musa.h"
#include "zensim/types/Iterator.h"

#if 0
namespace {
  static zs::raw_memory_resource<zs::device_mem_tag> *g_rawMusaMemInstance = nullptr;
  static zs::raw_memory_resource<zs::um_mem_tag> *g_rawMusaUmInstance = nullptr;
}  // namespace
#endif

namespace zs {

#if 0
  raw_memory_resource<device_mem_tag> &raw_memory_resource<device_mem_tag>::instance() {
    if (!g_rawMusaMemInstance) g_rawMusaMemInstance = new raw_memory_resource<device_mem_tag>;
    return *g_rawMusaMemInstance;
  }
  raw_memory_resource<um_mem_tag> &raw_memory_resource<um_mem_tag>::instance() {
    if (!g_rawMusaUmInstance) g_rawMusaUmInstance = new raw_memory_resource<um_mem_tag>;
    return *g_rawMusaUmInstance;
  }
#endif
  /// @ref
  /// https://stackoverflow.com/questions/6380326/c-symbol-export-for-class-extending-template-class
  raw_memory_resource<device_mem_tag>::raw_memory_resource() { (void)Musa::instance(); }

  raw_memory_resource<device_mem_tag> &raw_memory_resource<device_mem_tag>::instance() {
    static raw_memory_resource s_instance{};
    return s_instance;
  }

  void *raw_memory_resource<device_mem_tag>::do_allocate(size_t bytes, size_t alignment) {
    if (bytes) {
      auto ret = zs::allocate(mem_device, bytes, alignment);
      // record_allocation(MemTag{}, ret, demangle(*this), bytes, alignment);
      return ret;
    }
    return nullptr;
  }
  void raw_memory_resource<device_mem_tag>::do_deallocate(void *ptr, size_t bytes,
                                                          size_t alignment) {
    if (bytes) {
      zs::deallocate(mem_device, ptr, bytes, alignment);
      // erase_allocation(ptr);
    }
  }

  raw_memory_resource<um_mem_tag>::raw_memory_resource() { (void)Musa::instance(); }

  raw_memory_resource<um_mem_tag> &raw_memory_resource<um_mem_tag>::instance() {
    static raw_memory_resource s_instance{};
    return s_instance;
  }

  void *temporary_memory_resource<device_mem_tag>::do_allocate(std::size_t bytes,
                                                               std::size_t alignment) {
    if (bytes) {
      auto ret = ((Musa::MusaContext *)context)->streamMemAlloc(bytes, stream);
      /// @note make the allocation available to use instantly
      muStreamSynchronize((MUstream)stream);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (ret == nullptr) {
        auto loc = source_location::current();
        const auto fileInfo = fmt::format("# File: \"{:<50}\"", loc.file_name());
        const auto locInfo = fmt::format("# Ln {}, Col {}", loc.line(), loc.column());
        const auto funcInfo = fmt::format("# Func: \"{}\"", loc.function_name());
        int devid;
        muCtxGetDevice(&devid);
        std::cerr << fmt::format(
            "\nMusa Error on Device {}, Context [{}], Stream [{}]: muMemAllocAsync failed (size: "
            "{} bytes, alignment: {} "
            "bytes)\n{:=^60}\n{}\n{}\n{}\n{:=^60}\n\n",
            devid, context, stream, bytes, alignment, " musa driver api error location ", fileInfo,
            locInfo, funcInfo, "=");
      }
#endif
      return ret;
    }
    return nullptr;
  }
  void temporary_memory_resource<device_mem_tag>::do_deallocate(void *ptr, std::size_t bytes,
                                                                std::size_t alignment) {
    if (bytes) {
      ((Musa::MusaContext *)context)->streamMemFree(ptr, stream);
    }
  }

  template struct ZPC_BACKEND_TEMPLATE_EXPORT advisor_memory_resource<device_mem_tag>;

  stack_virtual_memory_resource<device_mem_tag>::stack_virtual_memory_resource(ProcID did,
                                                                               size_t size)
      : _allocHandles{}, _allocRanges{}, _allocatedSpace{0}, _did{did} {
    MUmemAllocationProp allocProp{};
    allocProp.type = MU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = (int)did;
    allocProp.win32HandleMetaData = NULL;
    _allocProp = allocProp;

    // _accessDescr
    MUmemAccessDesc accessDescr;
    accessDescr.location = allocProp.location;
    accessDescr.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    _accessDescr = accessDescr;

    // _granularity
    auto status = muMemGetAllocationGranularity(&_granularity, &allocProp,
                                                MU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (status != MUSA_SUCCESS) throw std::runtime_error("alloc granularity retrieval failed.");
    if (s_chunk_granularity % _granularity != 0)
      throw std::runtime_error("chunk granularity not a multiple of alloc granularity.");

    // _reservedSpace
    _reservedSpace = round_up(size, s_chunk_granularity);

    // _addr
    status = muMemAddressReserve((MUdeviceptr *)&_addr, _reservedSpace, (size_t)0 /*alignment*/,
                                 (MUdeviceptr)0ull, 0ull /*flag*/);
    if (status != MUSA_SUCCESS)
      throw std::runtime_error("fails to reserve a device virtual range.");
  }

  stack_virtual_memory_resource<device_mem_tag>::~stack_virtual_memory_resource() {
    muMemUnmap((MUdeviceptr)_addr, _reservedSpace);
    muMemAddressFree((MUdeviceptr)_addr, _reservedSpace);
    for (auto &&handle : _allocHandles) muMemRelease(handle);
  }

  bool stack_virtual_memory_resource<device_mem_tag>::do_check_residency(size_t offset,
                                                                         size_t bytes) const {
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;
    return ed <= _allocatedSpace;
  }
  bool stack_virtual_memory_resource<device_mem_tag>::do_commit(size_t offset, size_t bytes) {
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;

    auto &allocProp = std::any_cast<MUmemAllocationProp &>(_allocProp);
    auto &accessDescr = std::any_cast<MUmemAccessDesc &>(_accessDescr);

    if (ed <= _allocatedSpace) return true;

    unsigned long long handle{};  // MUmemGenericAllocationHandle
    const auto allocationBytes = ed - _allocatedSpace;
    auto status = muMemCreate(&handle, allocationBytes, &allocProp, 0ull);
    if (status != MUSA_SUCCESS) return false;

    if ((status
         = muMemMap((MUdeviceptr)_addr + _allocatedSpace, allocationBytes, (size_t)0, handle, 0ull))
        == MUSA_SUCCESS) {
      if ((status = muMemSetAccess((MUdeviceptr)_addr + _allocatedSpace, (size_t)allocationBytes,
                                   &accessDescr, 1ull))
          == MUSA_SUCCESS) {
        _allocRanges.push_back(std::make_pair(_allocatedSpace, allocationBytes));
        _allocHandles.push_back(handle);
      }
      if (status != MUSA_SUCCESS)
        (void)muMemUnmap((MUdeviceptr)_addr + _allocatedSpace, allocationBytes);
    }
    if (status != MUSA_SUCCESS) {
      (void)muMemRelease(handle);
      return false;
    }
    _allocatedSpace += allocationBytes;
    return true;
  }

  bool stack_virtual_memory_resource<device_mem_tag>::do_evict(size_t offset, size_t bytes) {
    ZS_WARN_IF(round_down(offset + bytes, s_chunk_granularity) < _allocatedSpace,
               "will evict more bytes (till the end) than asking");
    size_t st = round_up(offset, s_chunk_granularity);
    sint_t i = _allocRanges.size() - 1;
    for (; i >= 0 && _allocRanges[i].first >= st; --i) {
      if (muMemUnmap((MUdeviceptr)_addr + _allocRanges[i].first, _allocRanges[i].second)
          != MUSA_SUCCESS)
        return false;
      if (muMemRelease(_allocHandles[i]) != MUSA_SUCCESS) return false;
      _allocatedSpace = _allocRanges[i].first;
    }
    return true;
  }

  void *stack_virtual_memory_resource<device_mem_tag>::do_allocate(size_t bytes, size_t alignment) {
    return nullptr;
  }

  void stack_virtual_memory_resource<device_mem_tag>::do_deallocate(void *ptr, size_t bytes,
                                                                    size_t alignment) {
    return;
  }

  arena_virtual_memory_resource<device_mem_tag>::arena_virtual_memory_resource(ProcID did,
                                                                               size_t space)
      : _did{did}, _reservedSpace{round_up(space, s_chunk_granularity)} {
    if (did < 0)
      throw std::runtime_error(
          fmt::format("devicevm target device index [{}] is negative", (int)did));
    MUmemAllocationProp allocProp{};
    allocProp.type = MU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = (int)did;
    allocProp.win32HandleMetaData = NULL;
    _allocProp = allocProp;
    MUmemAccessDesc accessDescr;
    accessDescr.location = allocProp.location;
    accessDescr.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    _accessDescr = accessDescr;

    // _granularity
    auto status = muMemGetAllocationGranularity(&_granularity, &allocProp,
                                                MU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (status != MUSA_SUCCESS) throw std::runtime_error("alloc granularity retrieval failed.");
    if (s_chunk_granularity % _granularity != 0)
      throw std::runtime_error("chunk granularity not a multiple of alloc granularity.");
    // _addr
    status = muMemAddressReserve((MUdeviceptr *)&_addr, _reservedSpace, (size_t)0 /*alignment*/,
                                 (MUdeviceptr)0, 0ull /*flag*/);
    if (status != MUSA_SUCCESS)
      throw std::runtime_error("virtual address range reservation failed.");
    // active chunk masks
    _activeChunkMasks.resize((_reservedSpace / s_chunk_granularity + 63) / 64, (u64)0);
  }

  arena_virtual_memory_resource<device_mem_tag>::~arena_virtual_memory_resource() {
    muMemUnmap((MUdeviceptr)_addr, _reservedSpace);
    muMemAddressFree((MUdeviceptr)_addr, _reservedSpace);
    for (auto &&[offset, handle] : _allocations) muMemRelease(handle);
  }

  bool arena_virtual_memory_resource<device_mem_tag>::do_check_residency(size_t offset,
                                                                         size_t bytes) const {
    size_t st = round_down(offset, s_chunk_granularity);
    if (st >= _reservedSpace) return false;
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;
    for (st >>= s_chunk_granularity_bits, ed >>= s_chunk_granularity_bits; st != ed; ++st)
      if ((_activeChunkMasks[st >> 6] & ((u64)1 << (st & 63))) == 0) return false;
    return true;
  }
  bool arena_virtual_memory_resource<device_mem_tag>::do_commit(size_t offset, size_t bytes) {
    size_t st = round_down(offset, s_chunk_granularity);
    if (st >= _reservedSpace) return false;
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;

    auto &allocProp = std::any_cast<MUmemAllocationProp &>(_allocProp);
    auto &accessDescr = std::any_cast<MUmemAccessDesc &>(_accessDescr);

    for (; st != ed; st += s_chunk_granularity) {
      unsigned long long handle{};  // MUmemGenericAllocationHandle
      auto status = muMemCreate(&handle, s_chunk_granularity, &allocProp, 0ull);
      if (status != MUSA_SUCCESS) return false;

      if ((status = muMemMap((MUdeviceptr)_addr + st, s_chunk_granularity, (size_t)0, handle, 0ull))
          == MUSA_SUCCESS) {
        if ((status = muMemSetAccess((MUdeviceptr)_addr + st, (size_t)s_chunk_granularity,
                                     &accessDescr, 1ull))
            == MUSA_SUCCESS) {
          auto chunkid = st >> s_chunk_granularity_bits;
          _activeChunkMasks[chunkid >> 6] |= ((u64)1 << (chunkid & 63));
          _allocations.emplace(st, handle);
        }
        if (status != MUSA_SUCCESS) {
          (void)muMemUnmap((MUdeviceptr)_addr + st, s_chunk_granularity);
          return false;
        }
      }
      if (status != MUSA_SUCCESS) {
        (void)muMemRelease(handle);
        return false;
      }
    }
    return true;
  }

  bool arena_virtual_memory_resource<device_mem_tag>::do_evict(size_t offset, size_t bytes) {
    size_t st = round_up(offset, s_chunk_granularity);
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_down(offset, s_chunk_granularity) : _reservedSpace;
    if (st >= ed) return false;

    for (; st != ed; st += s_chunk_granularity) {
      if (muMemUnmap((MUdeviceptr)_addr + st, s_chunk_granularity) != MUSA_SUCCESS) return false;
      if (muMemRelease(_allocations[st]) != MUSA_SUCCESS) return false;

      _allocations.erase(st);
      auto chunkid = st >> s_chunk_granularity_bits;
      _activeChunkMasks[chunkid >> 6] &= ~((u64)1 << (chunkid & 63));
    }
    return true;
  }

}  // namespace zs