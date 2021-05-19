#include "Resource.h"

#include "zensim/tpls/umpire/strategy/AllocationAdvisor.hpp"

namespace zs {

  void copy(MemoryEntity dst, MemoryEntity src, std::size_t size) {
    if (dst.descr.onHost() && src.descr.onHost())
      copy(mem_host, dst.ptr, src.ptr, size);
    else
      copy(mem_device, dst.ptr, src.ptr, size);
  }

  GeneralAllocator get_memory_source(memsrc_e mre, ProcID devid, char *const advice) {
    auto memorySource = get_resource_manager().source(mre);
    if (advice == nullptr) {
      if (mre == memsrc_e::um) {
        if (devid < -1)
          memorySource = memorySource.advisor("READ_MOSTLY", devid);
        else
          memorySource = memorySource.advisor("PREFERRED_LOCATION", devid);
      }
    } else
      memorySource = memorySource.advisor(advice, devid);
    return memorySource;
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
