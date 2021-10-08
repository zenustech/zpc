#pragma once

#include <atomic>
#include <stdexcept>
#include <vector>

#include "zensim/Reflection.h"
#include "zensim/Singleton.h"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/memory/Allocator.h"
#include "zensim/memory/MemOps.hpp"
#include "zensim/memory/MemoryResource.h"
#include "zensim/types/Pointers.hpp"
#include "zensim/types/SmallVector.hpp"
#include "zensim/types/Tuple.h"

namespace zs {

  template <bool is_virtual_ = false, typename T = std::byte> struct ZSPmrAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    /// this is different from std::polymorphic_allocator
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_virtual = wrapv<is_virtual_>;
    using resource_type = conditional_t<is_virtual::value, vmr_t, mr_t>;

    ZSPmrAllocator() = default;
    ZSPmrAllocator(ZSPmrAllocator &&) = default;
    ZSPmrAllocator &operator=(ZSPmrAllocator &&) = default;
    ZSPmrAllocator(const ZSPmrAllocator &o) { *this = o.select_on_container_copy_construction(); }
    ZSPmrAllocator &operator=(const ZSPmrAllocator &o) {
      *this = o.select_on_container_copy_construction();
      return *this;
    }

    friend void swap(ZSPmrAllocator &a, ZSPmrAllocator &b) {
      std::swap(a.res, b.res);
      std::swap(a.location, b.location);
    }

    constexpr resource_type *resource() noexcept { return res.get(); }
    [[nodiscard]] void *allocate(std::size_t bytes,
                                 std::size_t alignment = alignof(std::max_align_t)) {
      return res->allocate(bytes, alignment);
    }
    void deallocate(void *p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
      res->deallocate(p, bytes, alignment);
    }
    bool is_equal(const ZSPmrAllocator &other) const noexcept {
      return res.get() == other.res.get() && location == other.location;
    }
    template <bool V = is_virtual::value>
    std::enable_if_t<V, bool> commit(std::size_t offset,
                                     std::size_t bytes = resource_type::s_chunk_granularity) {
      return res->commit(offset, bytes);
    }
    template <bool V = is_virtual::value>
    std::enable_if_t<V, bool> evict(std::size_t offset,
                                    std::size_t bytes = resource_type::s_chunk_granularity) {
      return res->evict(offset, bytes);
    }
    template <bool V = is_virtual::value> std::enable_if_t<V, bool> check_residency(
        std::size_t offset, std::size_t bytes = resource_type::s_chunk_granularity) const {
      return res->check_residency(offset, bytes);
    }
    template <bool V = is_virtual::value>
    std::enable_if_t<V, void *> address(std::size_t offset = 0) const {
      return res->address(offset);
    }

    ZSPmrAllocator select_on_container_copy_construction() const {
      ZSPmrAllocator ret{};
      ret.cloner = this->cloner;
      ret.res = this->cloner();
      ret.location = this->location;
      return ret;
    }

    /// owning upstream should specify deleter
    template <template <typename Tag> class ResourceT, typename... Args, std::size_t... Is>
    void setOwningUpstream(mem_tags tag, ProcID devid, std::tuple<Args &&...> args,
                           index_seq<Is...>) {
      match([&](auto t) {
        res = std::make_unique<ResourceT<decltype(t)>>(devid, std::get<Is>(args)...);
        location = MemoryLocation{t.value, devid};
        cloner = [devid, args]() -> std::unique_ptr<resource_type> {
          std::unique_ptr<resource_type> ret{};
          std::apply(
              [&ret](auto &&...ctorArgs) {
                ret = std::make_unique<ResourceT<decltype(t)>>(FWD(ctorArgs)...);
              },
              std::tuple_cat(std::make_tuple(devid), args));
          return ret;
        };
      })(tag);
    }
    template <template <typename Tag> class ResourceT, typename MemTag, typename... Args>
    void setOwningUpstream(MemTag tag, ProcID devid, Args &&...args) {
      if constexpr (is_same_v<MemTag, mem_tags>)
        setOwningUpstream<ResourceT>(tag, devid, std::forward_as_tuple(FWD(args)...),
                                     std::index_sequence_for<Args...>{});
      else {
        res = std::make_unique<ResourceT<MemTag>>(devid, FWD(args)...);
        location = MemoryLocation{MemTag::value, devid};
        cloner = [devid, args = std::make_tuple(args...)]() -> std::unique_ptr<resource_type> {
          std::unique_ptr<resource_type> ret{};
          std::apply(
              [&ret](auto &&...ctorArgs) {
                ret = std::make_unique<ResourceT<MemTag>>(FWD(ctorArgs)...);
              },
              std::tuple_cat(std::make_tuple(devid), args));
          return ret;
        };
      }
    }

    std::function<std::unique_ptr<resource_type>()> cloner{};
    std::unique_ptr<resource_type> res{};
    MemoryLocation location{memsrc_e::host, -1};
  };

  /// global free function
  void record_allocation(mem_tags, void *, std::string_view, std::size_t = 0, std::size_t = 0);
  void erase_allocation(void *);
  void copy(MemoryEntity dst, MemoryEntity src, std::size_t size);

  ZSPmrAllocator<> get_memory_source(memsrc_e mre, ProcID devid,
                                     std::string_view advice = std::string_view{});
  ZSPmrAllocator<true> get_virtual_memory_source(memsrc_e mre, ProcID devid, std::size_t bytes,
                                                 std::string_view option = "STACK");

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

  /// property tag
  struct PropertyTag {
    SmallString name;
    int numChannels;
  };

  inline auto select_properties(const std::vector<PropertyTag> &props,
                                const std::vector<SmallString> &names) {
    std::vector<PropertyTag> ret(0);
    for (auto &&name : names)
      for (auto &&prop : props)
        if (prop.name == name) {
          ret.push_back(prop);
          break;
        }
    return ret;
  }

}  // namespace zs
