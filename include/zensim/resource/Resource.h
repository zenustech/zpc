#pragma once

#include <atomic>

#include "zensim/Memory.hpp"
#include "zensim/Reflection.h"
#include "zensim/Singleton.h"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/memory/MemoryResource.h"
#include "zensim/memory/operations/Copy.hpp"
#include "zensim/tpls/magic_enum.hpp"
#include "zensim/tpls/umpire/ResourceManager.hpp"

namespace zs {

  struct GeneralAllocator;
  struct Resource;

  Resource &get_resource_manager() noexcept;

  void copy(host_exec_tag, void *dst, void *src, std::size_t size);
  void copy(MemoryEntity dst, MemoryEntity src, std::size_t size);

  template <typename MemTag> void *allocate(MemTag, std::size_t size, std::size_t alignment) {
    throw std::runtime_error(
        fmt::format("allocate(tag {}, size {}, alignment {}) not implemented\n", get_memory_source_tag(MemTag{}),
                    size, alignment));
  }
  template <typename MemTag>
  void deallocate(MemTag, void *ptr, std::size_t size, std::size_t alignment) {
    throw std::runtime_error(
        fmt::format("deallocate(tag {}, ptr {}, size {}, alignment {}) not implemented\n",
                    get_memory_source_tag(MemTag{}), reinterpret_cast<std::uintptr_t>(ptr), size, alignment));
  }
  template <typename MemTag> void memset(MemTag, void *addr, int chval, std::size_t size) {
    throw std::runtime_error(
        fmt::format("memset(tag {}, ptr {}, charval {}, size {}) not implemented\n",
                    get_memory_source_tag(MemTag{}), reinterpret_cast<std::uintptr_t>(addr), chval, size));
  }
  template <typename MemTag> void copy(MemTag, void *dst, void *src, std::size_t size) {
    throw std::runtime_error(fmt::format("copy(tag {}, dst {}, src {}, size {}) not implemented\n",
                                         get_memory_source_tag(MemTag{}), reinterpret_cast<std::uintptr_t>(dst),
                                         reinterpret_cast<std::uintptr_t>(src), size));
  }
  template <typename MemTag, typename... Args>
  void advise(MemTag, std::string advice, void *addr, Args...) {
    throw std::runtime_error(fmt::format(
        "advise(tag {}, advise {}, addr {}) with {} args not implemented\n", get_memory_source_tag(MemTag{}),
        advice, reinterpret_cast<std::uintptr_t>(addr), sizeof...(Args)));
  }

  GeneralAllocator get_memory_source(memsrc_e mre, ProcID devid, char *const advice = nullptr);

  struct Resource : std::reference_wrapper<umpire::ResourceManager>, Singleton<Resource> {
    Resource();
    GeneralAllocator source(memsrc_e mre) noexcept;
    GeneralAllocator source(std::string tag) noexcept;

    static std::atomic_ullong &counter() noexcept { return instance()._counter; }

  private:
    mutable std::atomic_ullong _counter{0};
  };

  struct GeneralAllocator : umpire::Allocator {
    umpire::Allocator &self() noexcept { return static_cast<umpire::Allocator &>(*this); }
    const umpire::Allocator &self() const noexcept {
      return static_cast<const umpire::Allocator &>(*this);
    }
    GeneralAllocator advisor(const std::string &advice_operation, int dev_id = 0);
    template <typename Strategy, bool introspection = true, typename... Args>
    GeneralAllocator allocator(Args &&...args) {
      auto &rm = get_resource_manager().get();
      auto name = this->getName() + demangle<Strategy>();
      if (rm.isAllocator(name)) return GeneralAllocator{rm.getAllocator(name)};
      return GeneralAllocator{
          rm.makeAllocator<Strategy>(name, self(), std::forward<Args>(args)...)};
    }
  };

}  // namespace zs
