#include "Allocator.h"

#include <cuda.h>

#include "zensim/Logger.hpp"
#include "zensim/types/Iterator.h"

namespace zs {

  monotonic_virtual_memory_resource<device_mem_tag>::monotonic_virtual_memory_resource(
      ProcID did, std::string_view type) {
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

  monotonic_virtual_memory_resource<device_mem_tag>::~monotonic_virtual_memory_resource() {
    cuMemUnmap((CUdeviceptr)_addr, _reservedSpace);
    for (auto &&varange : _vaRanges) cuMemAddressFree(varange.first, varange.second);
    for (auto &&handle : _allocHandles) cuMemRelease(handle);
  }

  bool monotonic_virtual_memory_resource<device_mem_tag>::reserve(std::size_t desiredSpace) {
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

  void *monotonic_virtual_memory_resource<device_mem_tag>::do_allocate(std::size_t bytes,
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
    ZS_WARN(fmt::format("alloc handle is {}, bytes {}\n", handle, allocationBytes));

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

  void monotonic_virtual_memory_resource<device_mem_tag>::do_deallocate(void *ptr,
                                                                        std::size_t bytes,
                                                                        std::size_t alignment) {}

}  // namespace zs