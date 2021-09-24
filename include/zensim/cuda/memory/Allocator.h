#pragma once
#include <any>
#include <string_view>

#include "zensim/memory/MemoryResource.h"
#include "zensim/types/Property.h"

namespace zs {

  template <typename MemTag> struct stack_virtual_memory_resource;

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

  template <> struct stack_virtual_memory_resource<um_mem_tag> : mr_t {
    stack_virtual_memory_resource(ProcID did = 0, std::string_view type = "DEVICE_PINNED")
        : type{type}, did{did} {}
    ~stack_virtual_memory_resource() = default;
    void *do_allocate(std::size_t bytes, std::size_t alignment) override {
      throw std::runtime_error(
          fmt::format("virtual_memory_resource[{}], type [{}]: \"allocate\" not implemented\n",
                      get_memory_tag_name(um_mem_tag::value), type));
      return nullptr;
    }
    void do_deallocate(void *ptr, std::size_t bytes, std::size_t alignment) override {
      throw std::runtime_error(
          fmt::format("virtual_memory_resource[{}], type [{}]: \"deallocate\" not implemented\n",
                      get_memory_tag_name(um_mem_tag::value), type));
    }
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }

  private:
    std::string type;
    ProcID did;
  };

}  // namespace zs