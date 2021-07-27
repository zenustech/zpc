#pragma once

#include <atomic>
#include <stdexcept>

#include "zensim/Reflection.h"
#include "zensim/Singleton.h"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/memory/Allocator.h"
#include "zensim/memory/MemOps.hpp"
#include "zensim/memory/MemoryResource.h"

namespace zs {

  template <typename T = std::byte> struct ZSPmrAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    /// this is different from std::polymorphic_allocator
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    static void free_mem_resource(mr_t *) noexcept {}

    ZSPmrAllocator() = default;
    ZSPmrAllocator(mr_t *mr) : res{mr} { /*res.reset(mr);*/
    }
    ZSPmrAllocator(const SharedHolder<mr_t> &mr) noexcept : res{mr} {}

    friend void swap(ZSPmrAllocator &a, ZSPmrAllocator &b) { std::swap(a.res, b.res); }

    constexpr mr_t *resource() noexcept { return res.get(); }
    [[nodiscard]] void *allocate(std::size_t bytes,
                                 std::size_t alignment = alignof(std::max_align_t)) {
      return res->allocate(bytes, alignment);
    }
    void deallocate(void *p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
      res->deallocate(p, bytes, alignment);
    }
    bool is_equal(const mr_t &other) const noexcept { return res.get() == &other; }
    bool is_equal(const ZSPmrAllocator &other) const noexcept {
      return res.get() == other.res.get();
    }

    /// non-owning upstream should specify deleter
    template <template <typename Tag> class ResourceT, typename... Args, std::size_t... Is>
    void setNonOwningUpstream(mem_tags tag, std::tuple<Args &&...> args, index_seq<Is...>) {
      match([&](auto t) {
        res = std::shared_ptr<ResourceT<decltype(t)>>(
            new ResourceT<decltype(t)>(std::get<Is>(args)...), free_mem_resource);
      })(tag);
    }
    template <template <typename Tag> class ResourceT, typename MemTag, typename... Args>
    void setNonOwningUpstream(MemTag tag, Args &&...args) {
      if constexpr (is_same_v<MemTag, mem_tags>)
        setNonOwningUpstream<ResourceT>(tag, std::forward_as_tuple(FWD(args)...),
                                        std::index_sequence_for<Args...>{});
      else
        res = std::shared_ptr<ResourceT<MemTag>>(new ResourceT<MemTag>(FWD(args)...),
                                                 free_mem_resource);
    }
    /// owning upstream should specify deleter
    template <template <typename Tag> class ResourceT, typename... Args, std::size_t... Is>
    void setOwningUpstream(mem_tags tag, std::tuple<Args &&...> args, index_seq<Is...>) {
      match([&](auto t) { res = std::make_shared<ResourceT<decltype(t)>>(std::get<Is>(args)...); })(
          tag);
    }
    template <template <typename Tag> class ResourceT, typename MemTag, typename... Args>
    void setOwningUpstream(MemTag tag, Args &&...args) {
      if constexpr (is_same_v<MemTag, mem_tags>)
        setOwningUpstream<ResourceT>(tag, std::forward_as_tuple(FWD(args)...),
                                     std::index_sequence_for<Args...>{});
      else
        res = std::make_shared<ResourceT<MemTag>>(FWD(args)...);
    }

    SharedHolder<mr_t> res{};
  };

  /// global free function
  void record_allocation(mem_tags, void *, std::string_view, std::size_t = 0, std::size_t = 0);
  void erase_allocation(void *);
  void copy(MemoryEntity dst, MemoryEntity src, std::size_t size);

  ZSPmrAllocator<> get_memory_source(memsrc_e mre, ProcID devid,
                                     std::string_view advice = std::string_view{});

  struct Resource : Singleton<Resource> {
    static std::atomic_ullong &counter() noexcept { return instance()._counter; }

    struct AllocationRecord {
      mem_tags tag{};
      std::size_t size{0}, alignment{0};
      std::string allocatorType{};
    };
    Resource() = default;
    ~Resource();

    void record(mem_tags tag, void *ptr, std::string_view name, std::size_t size,
                std::size_t alignment);
    void erase(void *ptr);

    void deallocate(void *ptr);

  private:
    mutable std::atomic_ullong _counter{0};
  };

  Resource &get_resource_manager() noexcept;

}  // namespace zs
