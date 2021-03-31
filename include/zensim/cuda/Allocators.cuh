#pragma once
#include <zensim/Singleton.h>
#include <zensim/memory/Allocator.h>

namespace zs {

  struct device_memory_resource : Singleton<device_memory_resource>,
                                  mr_t,
                                  Inherit<Object, device_memory_resource> {
    void *do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const mr_t &other) const noexcept override;
  };
  struct unified_memory_resource : Singleton<unified_memory_resource>,
                                   mr_t,
                                   Inherit<Object, unified_memory_resource> {
    void *do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const mr_t &other) const noexcept override;
  };

  /// memory allocators
  struct device_allocator : general_allocator {
    device_allocator() : general_allocator{&device_memory_resource::instance()} {}
  };

  struct unified_allocator : general_allocator {
    unified_allocator() : general_allocator{&unified_memory_resource::instance()} {}
  };

  struct dedicated_unified_allocator : general_allocator {
    dedicated_unified_allocator(int devid_)
        : general_allocator{&unified_memory_resource::instance()}, devid{devid_} {}

    void *allocate(std::size_t bytes, std::size_t align = alignof(std::max_align_t));

    int devid;
  };

  struct MonotonicAllocator : stack_allocator {
    MonotonicAllocator(std::size_t totalMemBytes, std::size_t alignment);
    auto borrow(std::size_t bytes) -> void *;
    void reset();
  };
  struct MonotonicVirtualAllocator : stack_allocator {
    MonotonicVirtualAllocator(int devid, std::size_t totalMemBytes, std::size_t alignment);
    auto borrow(std::size_t bytes) -> void *;
    void reset();
  };

}  // namespace zs
