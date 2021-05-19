#pragma once

#include <atomic>

#include "zensim/Memory.hpp"
#include "zensim/Reflection.h"
#include "zensim/Singleton.h"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/memory/MemoryResource.h"
#include "zensim/memory/MemOps.hpp"
#include "zensim/tpls/magic_enum.hpp"
#include "zensim/tpls/umpire/ResourceManager.hpp"

namespace zs {

  struct GeneralAllocator;
  struct Resource;

  Resource &get_resource_manager() noexcept;

  void copy(MemoryEntity dst, MemoryEntity src, std::size_t size);

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
