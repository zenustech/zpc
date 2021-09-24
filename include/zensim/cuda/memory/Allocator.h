#pragma once
#include <any>
#include <string_view>

#include "zensim/memory/MemoryResource.h"

namespace zs {

  template <typename MemTag> struct monotonic_virtual_memory_resource;

  template <> struct monotonic_virtual_memory_resource<device_mem_tag> : mr_t {
    monotonic_virtual_memory_resource(ProcID did = 0, std::string_view type = "DEVICE_PINNED");
    ~monotonic_virtual_memory_resource();
    void *do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void *ptr, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }

    bool reserve(std::size_t desiredSpace);

  private:
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

}  // namespace zs