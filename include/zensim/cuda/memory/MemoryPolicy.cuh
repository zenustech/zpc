#pragma once
#include <zensim/memory/Allocator.h>
#include <zensim/types/Polymorphism.h>

#include <zensim/cuda/Allocators.cuh>

namespace zs {

  /// memory space
  // space: heap/ device/ uvm
  // location: cpu/ gpu id

  /// memory policy (traits)
  // action: copy, move
  // scope: global, object space
  // lifetime: intermediate

  /// memory resource
  // level 0 - space - heap/ device/ uvm
  enum struct mem_source_e { custom, device, uvm, heap, stack };
  using MemSource = variant<mr_t, device_memory_resource, unified_memory_resource,
                            heap_memory_source, stack_memory_source>;
  // level 1 - alloc mechanism - monotonic/ pool/ null
  enum struct mem_allocator_e { handle, monotonic, sync_pool, unsync_pool };
  using MemAllocator = variant<mr_t, handle_resource>;
  // level 2 - behavior - noll/ printing/ logging/ registering/ event
  // mr_t *operator|(mr_t *a, mr_t *b) { ; }
  struct MemoryResource : mr_t, Inherit<Object, MemoryResource> {
  public:
    explicit MemoryResource(mr_t *source) { _allocator = std::make_unique<memory_pools>(source); }
    MemoryResource &setAllocator(mr_t *allocator) { _allocator.reset(allocator); }

  protected:
    void *do_allocate(std::size_t bytes, std::size_t alignment) override {
      for (auto &precb : _prehooks) precb.alloc(bytes, alignment);
      _allocator->allocate(bytes, alignment);
      for (auto &postcb : _posthooks) postcb.alloc(bytes, alignment);
    }
    void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override {
      for (auto &precb : _prehooks) precb.dealloc(p, bytes, alignment);
      _allocator->deallocate(p, bytes, alignment);
      for (auto &postcb : _posthooks) postcb.dealloc(p, bytes, alignment);
    }
    bool do_is_equal(const mr_t &other) const noexcept override {
      for (auto &precb : _prehooks) precb.dealloc(p, bytes, alignment);
      _allocator->deallocate(p, bytes, alignment);
      for (auto &postcb : _posthooks) postcb.dealloc(p, bytes, alignment);
    }

  private:
    std::unique_ptr<mr_t> _allocator{};
    std::vector<mr_callback> _prehooks{};
    std::vector<mr_callback> _posthooks{};
  };

  /// exception safety -> bad_alloc

}  // namespace zs