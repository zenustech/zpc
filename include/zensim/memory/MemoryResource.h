#pragma once
#include <memory>
#include <memory_resource>

#include "zensim/types/Function.h"
#include "zensim/types/Object.h"

namespace zs {

  namespace pmr = std::pmr;
  using mr_t = pmr::memory_resource;
  using unsynchronized_pool_resource = pmr::unsynchronized_pool_resource;
  using synchronized_pool_resource = pmr::synchronized_pool_resource;
  template <typename T> using object_allocator = pmr::polymorphic_allocator<T>;

  struct mr_callback : Inherit<Object, mr_callback> {
    std::function<void(std::size_t, std::size_t)> alloc{};
    std::function<void(void *, std::size_t, std::size_t)> dealloc{};
    std::function<void(const mr_t &)> is_equal{};
  };

  inline mr_callback default_mr_callback() {
    mr_callback tmp{};
    tmp.alloc = [](std::size_t bytes, std::size_t alignment) {};
    tmp.dealloc = [](void *p, std::size_t bytes, std::size_t alignment) {};
    tmp.is_equal = [](const mr_t &o) {};
    return tmp;
  }

  // HOST, DEVICE, DEVICE_CONST, UM, PINNED, FILE
  enum struct memsrc_e : char { host = 0, device, device_const, um, pinned, file };
  constexpr const char *memory_source_tag[]
      = {"HOST", "DEVICE", "DEVICE_CONST", "UM", "PINNED", "FILE"};

  struct MemoryHandle {
    constexpr MemoryHandle(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : _memsrc{mre}, _devid{devid} {}
    constexpr ProcID devid() const noexcept { return _devid; }
    constexpr memsrc_e memspace() const noexcept { return _memsrc; }

    void swap(MemoryHandle &o) noexcept {
      std::swap(_devid, o._devid);
      std::swap(_memsrc, o._memsrc);
    }

  protected:
    memsrc_e _memsrc{memsrc_e::host};  // memory source
    ProcID _devid{-1};                 // cpu id
  };

}  // namespace zs
