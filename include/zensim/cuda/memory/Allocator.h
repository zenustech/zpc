#pragma once
#include <any>
#include <string_view>
#include <unordered_map>

#include "zensim/memory/MemoryResource.h"
#include "zensim/types/Property.h"

namespace zs {

  template <typename MemTag> struct stack_virtual_memory_resource;
  template <typename MemTag> struct arena_virtual_memory_resource;

  template <> struct stack_virtual_memory_resource<device_mem_tag> : mr_t {
    stack_virtual_memory_resource(ProcID did = 0, std::string_view type = "DEVICE_PINNED");
    ~stack_virtual_memory_resource();
    void *do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void *ptr, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }

    bool reserve(std::size_t desiredSpace);

    std::vector<std::pair<unsigned long long, size_t>> _vaRanges;
    std::vector<unsigned long long> _allocHandles;
    std::vector<std::pair<void *, size_t>> _allocationRanges;
    std::string _type;
    std::any _allocProp;
    std::any _accessDescr;
    size_t _granularity;
    void *_addr;
    size_t _offset, _reservedSpace, _allocatedSpace;
    ProcID _did;
  };

  template <> struct arena_virtual_memory_resource<device_mem_tag>
      : vmr_t {  // default impl falls back to
    /// 2MB chunk granularity
    static constexpr size_t s_chunk_granularity_bits = vmr_t::s_chunk_granularity_bits;
    static constexpr size_t s_chunk_granularity = vmr_t::s_chunk_granularity;

    arena_virtual_memory_resource(ProcID did = -1, size_t space = s_chunk_granularity);
    ~arena_virtual_memory_resource();
    bool do_check_residency(std::size_t offset, std::size_t bytes) const override;
    bool do_commit(std::size_t offset, std::size_t bytes) override;
    bool do_evict(std::size_t offset, std::size_t bytes) override;
    void *do_address(std::size_t offset) const override {
      return static_cast<void *>(static_cast<char *>(_addr) + offset);
    }

    void *do_allocate(std::size_t bytes, std::size_t alignment) override { return _addr; }

    std::any _allocProp;
    std::any _accessDescr;
    size_t _granularity;
    const size_t _reservedSpace;
    void *_addr;
    std::vector<u64> _activeChunkMasks;
    std::unordered_map<size_t, unsigned long long> _allocations;  // <offset, handle>
    ProcID _did;
  };

}  // namespace zs