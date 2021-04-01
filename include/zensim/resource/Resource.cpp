#include "Resource.h"

#include "zensim/tpls/umpire/strategy/AllocationAdvisor.hpp"

namespace zs {

  Resource &get_resource_manager() noexcept { return Resource::instance(); }

  Resource::Resource()
      : std::reference_wrapper<umpire::ResourceManager>{umpire::ResourceManager::getInstance()},
        _counter{0} {}

  /// Resource
  umpire::ResourceManager &Resource::self() noexcept { return *this; }
  const umpire::ResourceManager &Resource::self() const noexcept { return *this; }
  // HOST, DEVICE, DEVICE_CONST, UM, PINNED, FILE
  GeneralAllocator Resource::source(memsrc_e mre) noexcept {
    return GeneralAllocator{get_resource_manager().self().getAllocator(
        memory_source_tag[magic_enum::enum_integer(mre)])};
  }
  GeneralAllocator Resource::source(std::string tag) noexcept {
    return GeneralAllocator{get_resource_manager().self().getAllocator(tag)};
  }

  /// GeneralAllocator
  GeneralAllocator GeneralAllocator::advisor(const std::string &advice_operation, int dev_id) {
    auto &rm = get_resource_manager().self();
    auto name = this->getName() + advice_operation;
    if (rm.isAllocator(name)) return GeneralAllocator{rm.getAllocator(name)};
    return GeneralAllocator{rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
        name, self(), advice_operation, dev_id)};
  }

}  // namespace zs
