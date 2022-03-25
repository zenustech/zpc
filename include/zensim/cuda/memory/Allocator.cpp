#include "Allocator.h"

#include <cuda.h>

#include "zensim/Logger.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/memory/Allocator.h"
#include "zensim/types/Iterator.h"

namespace zs {

  template <>
  raw_memory_resource<device_mem_tag> raw_memory_resource<device_mem_tag>::s_rawMemResource{};
  template <> raw_memory_resource<um_mem_tag> raw_memory_resource<um_mem_tag>::s_rawMemResource{};

  template <>
  raw_memory_resource<device_mem_tag> &raw_memory_resource<device_mem_tag>::instance() noexcept {
    return s_rawMemResource;
  }
  template <>
  raw_memory_resource<um_mem_tag> &raw_memory_resource<um_mem_tag>::instance() noexcept {
    return s_rawMemResource;
  }

  template struct ZPC_API raw_memory_resource<device_mem_tag>;
  template struct ZPC_API raw_memory_resource<um_mem_tag>;

#if 0
  stack_virtual_memory_resource<device_mem_tag>::stack_virtual_memory_resource(
      ProcID did, std::string_view type)
      : _vaRanges{},
        _allocHandles{},
        _allocationRanges{},
        _type{type},
        _addr{nullptr},
        _offset{0},
        _reservedSpace{0},
        _allocatedSpace{0},
        _did{did} {
    CUmemAllocationProp allocProp{};
    if (type != "DEVICE_PINNED")
      throw std::runtime_error(
          fmt::format("currently cudavm does not support allocation type {}", type));
    allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = (int)did;
    allocProp.win32HandleMetaData = NULL;
    _allocProp = allocProp;

    // _accessDescr
    CUmemAccessDesc accessDescr;
    accessDescr.location = allocProp.location;
    accessDescr.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    _accessDescr = accessDescr;

    // _granularity
    auto status = cuMemGetAllocationGranularity(&_granularity, &allocProp,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (status != CUDA_SUCCESS) throw std::runtime_error("alloc granularity retrieval failed.");
  }

  stack_virtual_memory_resource<device_mem_tag>::~stack_virtual_memory_resource() {
    cuMemUnmap((CUdeviceptr)_addr, _reservedSpace);
    for (auto &&varange : _vaRanges) cuMemAddressFree(varange.first, varange.second);
    for (auto &&handle : _allocHandles) cuMemRelease(handle);
  }

  bool stack_virtual_memory_resource<device_mem_tag>::reserve(std::size_t desiredSpace) {
    if (desiredSpace <= _reservedSpace) return true;
    auto newSpace = (desiredSpace + _granularity - 1) / _granularity * _granularity;
    CUdeviceptr ptr = 0ull;
    auto status = cuMemAddressReserve(&ptr, newSpace - _reservedSpace, (size_t)0,
                                      ((CUdeviceptr)_addr + _reservedSpace), 0ull);
    if (status != CUDA_SUCCESS || ptr != (CUdeviceptr)_addr + _reservedSpace) {
      if (ptr != 0ull) (void)cuMemAddressFree(ptr, newSpace - _reservedSpace);
      // request a whole new virtual range
      status = cuMemAddressReserve(&ptr, newSpace, (size_t)0 /*alignment*/, (CUdeviceptr)0ull,
                                   0ull /*flag*/);
      if (status != CUDA_SUCCESS) return false;
      // remap previous allocations
      if (_addr != nullptr) {
        // unmap previous allocations // not sure if iterating allocations is needed
        if ((status = cuMemUnmap((CUdeviceptr)_addr, _reservedSpace)) != CUDA_SUCCESS) {
          (void)cuMemAddressFree(ptr, newSpace);
          ZS_WARN("failed to unmap previous reserved virtual range\n");
          return false;
        }
        auto newPtr = ptr;
        auto &accessDescr = std::any_cast<CUmemAccessDesc &>(_accessDescr);
        for (auto &&[handle, r] : zip(_allocHandles, _allocationRanges)) {
          auto &sz = r.second;
          if ((status = cuMemMap(newPtr, sz, 0ull, handle, 0ull)) != CUDA_SUCCESS) break;
          if ((status = cuMemSetAccess(newPtr, sz, &accessDescr, 1ull)) != CUDA_SUCCESS) break;
          newPtr += sz;
        }
        if (status != CUDA_SUCCESS) {
          cuMemUnmap(ptr, newSpace);
          cuMemAddressFree(ptr, newSpace);
          ZS_WARN("failed to map previous allocations to the new virtual range\n");
          return false;
        }
        for (auto &&varange : _vaRanges) (void)cuMemAddressFree(varange.first, varange.second);
        _vaRanges.clear();
      }
      _vaRanges.push_back(std::make_pair(ptr, newSpace));
      _addr = (void *)ptr;
    } else {
      _vaRanges.push_back(std::make_pair(ptr, newSpace - _reservedSpace));
      if (_addr == nullptr) _addr = (void *)ptr;
    }
    _reservedSpace = newSpace;
    return true;
  }

  void *stack_virtual_memory_resource<device_mem_tag>::do_allocate(std::size_t bytes,
                                                                   std::size_t alignment) {
    auto &allocProp = std::any_cast<CUmemAllocationProp &>(_allocProp);
    unsigned long long handle{};  // CUmemGenericAllocationHandle
    _offset = (_offset + alignment - 1) / alignment * alignment;

    if (!reserve(_offset + bytes)) return nullptr;

    if (_offset + bytes <= _allocatedSpace) {
      void *ret = (void *)((char *)_addr + _offset);
      _offset += bytes;
      return ret;
    }

    auto allocationBytes = (bytes + _granularity - 1) / _granularity * _granularity;
    // cudri::vcreate(&handle, allocationBytes, &allocProp, (unsigned long long)0);
    auto status = cuMemCreate(&handle, allocationBytes, &allocProp, 0ull);
    if (status != CUDA_SUCCESS) return nullptr;
    // ZS_WARN(fmt::format("alloc handle is {}, bytes {}\n", handle, allocationBytes));

    void *base = (char *)_addr + _offset;
    // cudri::mmap(base, allocationBytes, (size_t)0, handle, (unsigned long long)0);
    if ((status = cuMemMap((CUdeviceptr)base, allocationBytes, (size_t)0, handle, 0ull))
        == CUDA_SUCCESS) {
      auto &accessDescr = std::any_cast<CUmemAccessDesc &>(_accessDescr);
      // cudri::setMemAccess(base, allocationBytes, &accessDescr, (size_t)1);
      if ((status = cuMemSetAccess((CUdeviceptr)base, (size_t)allocationBytes, &accessDescr, 1ull))
          == CUDA_SUCCESS) {
        _offset += bytes;  // notice: not allocationBytes!
        _allocatedSpace += allocationBytes;
        _allocHandles.push_back(handle);
        _allocationRanges.push_back(std::make_pair(base, allocationBytes));
      }
      if (status != CUDA_SUCCESS) {
        (void)cuMemUnmap((CUdeviceptr)base, allocationBytes);
        return nullptr;
      }
    }
    if (status != CUDA_SUCCESS) {
      (void)cuMemRelease(handle);
      return nullptr;
    }
    return (void *)base;
  }

  void stack_virtual_memory_resource<device_mem_tag>::do_deallocate(void *ptr, std::size_t bytes,
                                                                    std::size_t alignment) {
    std::size_t i = _allocationRanges.size();
    for (; i != 0 && (std::uintptr_t)_allocationRanges[i - 1].first >= (std::uintptr_t)ptr;) {
      --i;
      cuMemUnmap((CUdeviceptr)_allocationRanges[i].first, _allocationRanges[i].second);
      cuMemRelease(_allocHandles[i]);
      _allocatedSpace -= _allocationRanges[i].second;
      _allocationRanges.pop_back();
      _allocHandles.pop_back();
    }
    if (_offset > _allocatedSpace) _offset = _allocatedSpace;
  }

#else
  stack_virtual_memory_resource<device_mem_tag>::stack_virtual_memory_resource(ProcID did,
                                                                               std::size_t size)
      : _allocHandles{}, _allocRanges{}, _allocatedSpace{0}, _did{did} {
    CUmemAllocationProp allocProp{};
    allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = (int)did;
    allocProp.win32HandleMetaData = NULL;
    _allocProp = allocProp;

    // _accessDescr
    CUmemAccessDesc accessDescr;
    accessDescr.location = allocProp.location;
    accessDescr.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    _accessDescr = accessDescr;

    // _granularity
    auto status = cuMemGetAllocationGranularity(&_granularity, &allocProp,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (status != CUDA_SUCCESS) throw std::runtime_error("alloc granularity retrieval failed.");
    if (s_chunk_granularity % _granularity != 0)
      throw std::runtime_error("chunk granularity not a multiple of alloc granularity.");

    // _reservedSpace
    _reservedSpace = round_up(size, s_chunk_granularity);

    // _addr
    status = cuMemAddressReserve((CUdeviceptr *)&_addr, _reservedSpace, (size_t)0 /*alignment*/,
                                 (CUdeviceptr)0ull, 0ull /*flag*/);
    if (status != CUDA_SUCCESS)
      throw std::runtime_error("fails to reserve a device virtual range.");
  }

  stack_virtual_memory_resource<device_mem_tag>::~stack_virtual_memory_resource() {
    cuMemUnmap((CUdeviceptr)_addr, _reservedSpace);
    cuMemAddressFree((CUdeviceptr)_addr, _reservedSpace);
    for (auto &&handle : _allocHandles) cuMemRelease(handle);
  }

  bool stack_virtual_memory_resource<device_mem_tag>::do_check_residency(std::size_t offset,
                                                                         std::size_t bytes) const {
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;
    return ed <= _allocatedSpace;
  }
  bool stack_virtual_memory_resource<device_mem_tag>::do_commit(std::size_t offset,
                                                                std::size_t bytes) {
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;

    auto &allocProp = std::any_cast<CUmemAllocationProp &>(_allocProp);
    auto &accessDescr = std::any_cast<CUmemAccessDesc &>(_accessDescr);

    if (ed <= _allocatedSpace) return true;

    unsigned long long handle{};  // CUmemGenericAllocationHandle
    const auto allocationBytes = ed - _allocatedSpace;
    auto status = cuMemCreate(&handle, allocationBytes, &allocProp, 0ull);
    if (status != CUDA_SUCCESS) return false;

    if ((status
         = cuMemMap((CUdeviceptr)_addr + _allocatedSpace, allocationBytes, (size_t)0, handle, 0ull))
        == CUDA_SUCCESS) {
      if ((status = cuMemSetAccess((CUdeviceptr)_addr + _allocatedSpace, (size_t)allocationBytes,
                                   &accessDescr, 1ull))
          == CUDA_SUCCESS) {
        _allocRanges.push_back(std::make_pair(_allocatedSpace, allocationBytes));
        _allocHandles.push_back(handle);
      }
      if (status != CUDA_SUCCESS)
        (void)cuMemUnmap((CUdeviceptr)_addr + _allocatedSpace, allocationBytes);
    }
    if (status != CUDA_SUCCESS) {
      (void)cuMemRelease(handle);
      return false;
    }
    _allocatedSpace += allocationBytes;
    return true;
  }

  bool stack_virtual_memory_resource<device_mem_tag>::do_evict(std::size_t offset,
                                                               std::size_t bytes) {
    ZS_WARN_IF(round_down(offset + bytes, s_chunk_granularity) < _allocatedSpace,
               "will evict more bytes (till the end) than asking");
    size_t st = round_up(offset, s_chunk_granularity);
    sint_t i = _allocRanges.size() - 1;
    for (; i >= 0 && _allocRanges[i].first >= st; --i) {
      if (cuMemUnmap((CUdeviceptr)_addr + _allocRanges[i].first, _allocRanges[i].second)
          != CUDA_SUCCESS)
        return false;
      if (cuMemRelease(_allocHandles[i]) != CUDA_SUCCESS) return false;
      _allocatedSpace = _allocRanges[i].first;
    }
    return true;
  }

  void *stack_virtual_memory_resource<device_mem_tag>::do_allocate(std::size_t bytes,
                                                                   std::size_t alignment) {
    return nullptr;
  }

  void stack_virtual_memory_resource<device_mem_tag>::do_deallocate(void *ptr, std::size_t bytes,
                                                                    std::size_t alignment) {
    return;
  }

#endif

  arena_virtual_memory_resource<device_mem_tag>::arena_virtual_memory_resource(ProcID did,
                                                                               size_t space)
      : _did{did}, _reservedSpace{round_up(space, s_chunk_granularity)} {
    if (did < 0)
      throw std::runtime_error(
          fmt::format("devicevm target device index [{}] is negative", (int)did));
    CUmemAllocationProp allocProp{};
    allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = (int)did;
    allocProp.win32HandleMetaData = NULL;
    _allocProp = allocProp;
    CUmemAccessDesc accessDescr;
    accessDescr.location = allocProp.location;
    accessDescr.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    _accessDescr = accessDescr;

    // _granularity
    auto status = cuMemGetAllocationGranularity(&_granularity, &allocProp,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (status != CUDA_SUCCESS) throw std::runtime_error("alloc granularity retrieval failed.");
    if (s_chunk_granularity % _granularity != 0)
      throw std::runtime_error("chunk granularity not a multiple of alloc granularity.");
    // _addr
    status = cuMemAddressReserve((CUdeviceptr *)&_addr, _reservedSpace, (size_t)0 /*alignment*/,
                                 (CUdeviceptr)0, 0ull /*flag*/);
    if (status != CUDA_SUCCESS)
      throw std::runtime_error("virtual address range reservation failed.");
    // active chunk masks
    _activeChunkMasks.resize((_reservedSpace / s_chunk_granularity + 63) / 64, (u64)0);
  }

  arena_virtual_memory_resource<device_mem_tag>::~arena_virtual_memory_resource() {
    cuMemUnmap((CUdeviceptr)_addr, _reservedSpace);
    cuMemAddressFree((CUdeviceptr)_addr, _reservedSpace);
    for (auto &&[offset, handle] : _allocations) cuMemRelease(handle);
  }

  bool arena_virtual_memory_resource<device_mem_tag>::do_check_residency(std::size_t offset,
                                                                         std::size_t bytes) const {
    size_t st = round_down(offset, s_chunk_granularity);
    if (st >= _reservedSpace) return false;
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;
    for (st >>= s_chunk_granularity_bits, ed >>= s_chunk_granularity_bits; st != ed; ++st)
      if ((_activeChunkMasks[st >> 6] & ((u64)1 << (st & 63))) == 0) return false;
    return true;
  }
  bool arena_virtual_memory_resource<device_mem_tag>::do_commit(std::size_t offset,
                                                                std::size_t bytes) {
    size_t st = round_down(offset, s_chunk_granularity);
    if (st >= _reservedSpace) return false;
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;

    auto &allocProp = std::any_cast<CUmemAllocationProp &>(_allocProp);
    auto &accessDescr = std::any_cast<CUmemAccessDesc &>(_accessDescr);

    for (; st != ed; st += s_chunk_granularity) {
      unsigned long long handle{};  // CUmemGenericAllocationHandle
      auto status = cuMemCreate(&handle, s_chunk_granularity, &allocProp, 0ull);
      if (status != CUDA_SUCCESS) return false;

      if ((status = cuMemMap((CUdeviceptr)_addr + st, s_chunk_granularity, (size_t)0, handle, 0ull))
          == CUDA_SUCCESS) {
        if ((status = cuMemSetAccess((CUdeviceptr)_addr + st, (size_t)s_chunk_granularity,
                                     &accessDescr, 1ull))
            == CUDA_SUCCESS) {
          auto chunkid = st >> s_chunk_granularity_bits;
          _activeChunkMasks[chunkid >> 6] |= ((u64)1 << (chunkid & 63));
          _allocations.emplace(st, handle);
        }
        if (status != CUDA_SUCCESS) {
          (void)cuMemUnmap((CUdeviceptr)_addr + st, s_chunk_granularity);
          return false;
        }
      }
      if (status != CUDA_SUCCESS) {
        (void)cuMemRelease(handle);
        return false;
      }
    }
    return true;
  }

  bool arena_virtual_memory_resource<device_mem_tag>::do_evict(std::size_t offset,
                                                               std::size_t bytes) {
    size_t st = round_up(offset, s_chunk_granularity);
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_down(offset, s_chunk_granularity) : _reservedSpace;
    if (st >= ed) return false;

    for (; st != ed; st += s_chunk_granularity) {
      if (cuMemUnmap((CUdeviceptr)_addr + st, s_chunk_granularity) != CUDA_SUCCESS) return false;
      if (cuMemRelease(_allocations[st]) != CUDA_SUCCESS) return false;

      _allocations.erase(st);
      auto chunkid = st >> s_chunk_granularity_bits;
      _activeChunkMasks[chunkid >> 6] &= ~((u64)1 << (chunkid & 63));
    }
    return true;
  }

}  // namespace zs