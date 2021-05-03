#include "Resource.h"

#include "zensim/memory/operations/Copy.hpp"
#include "zensim/tpls/umpire/strategy/AllocationAdvisor.hpp"
#if ZS_ENABLE_CUDA
#  include "zensim/cuda/memory/operations/Copy.hpp"
#endif

namespace zs {

  void copy(MemoryEntity dst, MemoryEntity src, std::size_t size) {
    if (dst.descr.onHost() && src.descr.onHost())
      mem_copy<execspace_e::host>{}(dst, src, size);
    else
      mem_copy<execspace_e::cuda>{}(dst, src, size);
  }

  Resource &get_resource_manager() noexcept { return Resource::instance(); }

  Resource::Resource()
      : std::reference_wrapper<umpire::ResourceManager>{umpire::ResourceManager::getInstance()},
        _counter{0} {}

  /// Resource
  // HOST, DEVICE, DEVICE_CONST, UM, PINNED, FILE
  GeneralAllocator Resource::source(memsrc_e mre) noexcept {
    return GeneralAllocator{get_resource_manager().get().getAllocator(
        memory_source_tag[magic_enum::enum_integer(mre)])};
  }
  GeneralAllocator Resource::source(std::string tag) noexcept {
    return GeneralAllocator{get_resource_manager().get().getAllocator(tag)};
  }

  /// GeneralAllocator
  GeneralAllocator GeneralAllocator::advisor(const std::string &advice_operation, int dev_id) {
    auto &rm = get_resource_manager().get();
    auto name = this->getName() + advice_operation;
    if (rm.isAllocator(name)) return GeneralAllocator{rm.getAllocator(name)};
    return GeneralAllocator{rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
        name, self(), advice_operation, dev_id)};
  }

}  // namespace zs
