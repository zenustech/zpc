#pragma once

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "MemOps.hpp"
#include "MemoryResource.h"
#include "zensim/math/bit/Bits.h"
#include "zensim/memory/MemOps.hpp"

namespace zs {

  template <typename MemTag> struct raw_memory_resource : mr_t {
    static raw_memory_resource &instance() {
      static raw_memory_resource s_instance{};
      return s_instance;
    }
    
    using value_type = std::byte;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = true_type;
    using propagate_on_container_copy_assignment = true_type;
    using propagate_on_container_swap = true_type;

    void *do_allocate(size_t bytes, size_t alignment) override {
      if (bytes) {
        auto ret = zs::allocate(MemTag{}, bytes, alignment);
        // record_allocation(MemTag{}, ret, demangle(*this), bytes, alignment);
        return ret;
      }
      return nullptr;
    }
    void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
      if (bytes) {
        zs::deallocate(MemTag{}, ptr, bytes, alignment);
        // erase_allocation(ptr);
      }
    }
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }
  };

  template <> struct raw_memory_resource<host_mem_tag> : mr_t {
    ZPC_CORE_API static raw_memory_resource &instance() {
      static raw_memory_resource s_instance{};
      return s_instance;
    }

    using value_type = std::byte;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = true_type;
    using propagate_on_container_copy_assignment = true_type;
    using propagate_on_container_swap = true_type;

    void *do_allocate(size_t bytes, size_t alignment) override {
      if (bytes) {
        auto ret = zs::allocate(mem_host, bytes, alignment);
        return ret;
      }
      return nullptr;
    }
    void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
      if (bytes) {
        zs::deallocate(mem_host, ptr, bytes, alignment);
      }
    }
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }
  };

  template <typename MemTag> struct temporary_memory_resource : raw_memory_resource<MemTag> {
    using base_t = raw_memory_resource<MemTag>;
    using value_type = typename base_t::value_type;
    using size_type = typename base_t::size_type;
    using difference_type = typename base_t::difference_type;
    using propagate_on_container_move_assignment =
        typename base_t::propagate_on_container_move_assignment;
    using propagate_on_container_copy_assignment =
        typename base_t::propagate_on_container_copy_assignment;
    using propagate_on_container_swap = typename base_t::propagate_on_container_swap;
  };

  template <typename MemTag> struct default_memory_resource : mr_t {
    default_memory_resource(ProcID did = 0, mr_t *up = &raw_memory_resource<MemTag>::instance())
        : upstream{up}, did{did} {}
    ~default_memory_resource() = default;
    void *do_allocate(size_t bytes, size_t alignment) override {
      // if (!prepare_context(MemTag{}, did)) return nullptr;
      void *ret = upstream->allocate(bytes, alignment);
      return ret;
    }
    void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
      // if (!prepare_context(MemTag{}, did)) return;
      upstream->deallocate(ptr, bytes, alignment);
    }
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }

  private:
    mr_t *upstream;
    ProcID did;
  };

  template <typename MemTag> struct advisor_memory_resource : mr_t {
    advisor_memory_resource(ProcID did = 0, std::string_view option = "PREFERRED_LOCATION",
                            mr_t *up = &raw_memory_resource<MemTag>::instance())
        : upstream{up}, option{option}, did{did} {}
    ~advisor_memory_resource() = default;
    void *do_allocate(size_t bytes, size_t alignment) override {
      void *ret = upstream->allocate(bytes, alignment);
      advise(MemTag{}, option, ret, bytes, did);
      return ret;
    }
    void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
      upstream->deallocate(ptr, bytes, alignment);
    }
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }

  private:
    mr_t *upstream;
    std::string option;
    ProcID did;
  };

  extern template struct ZPC_CORE_TEMPLATE_IMPORT advisor_memory_resource<host_mem_tag>;

  template <typename MemTag> struct stack_virtual_memory_resource
      : vmr_t {  // default impl falls back to
    template <typename... Args> stack_virtual_memory_resource(Args...) {
      throw std::runtime_error("stack virtual memory allocator not implemented!");
    }
    ~stack_virtual_memory_resource() = default;

    bool do_check_residency([[maybe_unused]] size_t offset,
                            [[maybe_unused]] size_t bytes) const override {
      throw std::runtime_error("stack virtual memory allocator not implemented!");
      return false;
    }
    bool do_commit([[maybe_unused]] size_t offset, [[maybe_unused]] size_t bytes) override {
      throw std::runtime_error("stack virtual memory allocator not implemented!");
      return false;
    }
    bool do_evict([[maybe_unused]] size_t offset, [[maybe_unused]] size_t bytes) override {
      throw std::runtime_error("stack virtual memory allocator not implemented!");
      return false;
    }
    void *do_address([[maybe_unused]] size_t offset) const override {
      throw std::runtime_error("stack virtual memory allocator not implemented!");
      return nullptr;
    }

    void *do_allocate([[maybe_unused]] size_t bytes, [[maybe_unused]] size_t alignment) override {
      throw std::runtime_error("stack virtual memory allocator not implemented!");
      return nullptr;
    }

    void do_deallocate([[maybe_unused]] void *ptr, [[maybe_unused]] size_t bytes,
                       [[maybe_unused]] size_t alignment) override {
      throw std::runtime_error("stack virtual memory allocator not implemented!");
    }
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }
  };

  template <typename MemTag> struct arena_virtual_memory_resource : vmr_t {
    template <typename... Args> arena_virtual_memory_resource(Args...) {
      throw std::runtime_error("arena virtual memory allocator not implemented!");
    }
    ~arena_virtual_memory_resource() = default;

    bool do_check_residency([[maybe_unused]] size_t offset,
                            [[maybe_unused]] size_t bytes) const override {
      throw std::runtime_error("arena virtual memory allocator not implemented!");
      return false;
    }
    bool do_commit([[maybe_unused]] size_t offset, [[maybe_unused]] size_t bytes) override {
      throw std::runtime_error("arena virtual memory allocator not implemented!");
      return false;
    }
    bool do_evict([[maybe_unused]] size_t offset, [[maybe_unused]] size_t bytes) override {
      throw std::runtime_error("arena virtual memory allocator not implemented!");
      return false;
    }
    void *do_address([[maybe_unused]] size_t offset) const override {
      throw std::runtime_error("arena virtual memory allocator not implemented!");
      return nullptr;
    }

    void *do_allocate([[maybe_unused]] size_t bytes, [[maybe_unused]] size_t alignment) override {
      throw std::runtime_error("arena virtual memory allocator not implemented!");
      return nullptr;
    }

    void do_deallocate([[maybe_unused]] void *ptr, [[maybe_unused]] size_t bytes,
                       [[maybe_unused]] size_t alignment) override {
      throw std::runtime_error("arena virtual memory allocator not implemented!");
    }
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }
  };

#if 0
  template <> struct stack_virtual_memory_resource<host_mem_tag>
      : vmr_t {  // default impl falls back to
    stack_virtual_memory_resource(ProcID did = -1, std::string_view type = "HOST_VIRTUAL");
    ~stack_virtual_memory_resource();
    void *do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void *ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }

    bool reserve(size_t desiredSpace);

    std::string _type;
    size_t _granularity;
    void *_addr;
    size_t _offset, _allocatedSpace, _reservedSpace;
    ProcID _did;
  };
#else
  template <> struct stack_virtual_memory_resource<host_mem_tag>
      : vmr_t {  // default impl falls back to
    ZPC_CORE_API stack_virtual_memory_resource(ProcID did, size_t size);
    ZPC_CORE_API ~stack_virtual_memory_resource();
    ZPC_CORE_API bool do_check_residency(size_t offset, size_t bytes) const override;
    ZPC_CORE_API bool do_commit(size_t offset, size_t bytes) override;
    ZPC_CORE_API bool do_evict(size_t offset, size_t bytes) override;
    void *do_address(size_t offset) const override {
      return static_cast<void *>(static_cast<char *>(_addr) + offset);
    }

    ZPC_CORE_API void *do_allocate(size_t bytes, size_t alignment) override;
    ZPC_CORE_API void do_deallocate(void *ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }

    size_t _granularity;
    void *_addr;
    size_t _allocatedSpace, _reservedSpace;
    ProcID _did;
  };

#endif

#ifdef ZS_PLATFORM_WINDOWS
#elif defined(ZS_PLATFORM_UNIX)

  template <> struct arena_virtual_memory_resource<host_mem_tag>
      : vmr_t {  // default impl falls back to
    /// 2MB chunk granularity
    static constexpr size_t s_chunk_granularity_bits = vmr_t::s_chunk_granularity_bits;
    static constexpr size_t s_chunk_granularity = vmr_t::s_chunk_granularity;

    ZPC_CORE_API arena_virtual_memory_resource(ProcID did, size_t space);
    ZPC_CORE_API ~arena_virtual_memory_resource();
    ZPC_CORE_API bool do_check_residency(size_t offset, size_t bytes) const override;
    ZPC_CORE_API bool do_commit(size_t offset, size_t bytes) override;
    ZPC_CORE_API bool do_evict(size_t offset, size_t bytes) override;
    void *do_address(size_t offset) const override {
      return static_cast<void *>(static_cast<char *>(_addr) + offset);
    }

    void *do_allocate(size_t /*bytes*/, size_t /*alignment*/) override { return _addr; }

    size_t _granularity;
    const size_t _reservedSpace;
    void *_addr;
    std::vector<u64> _activeChunkMasks;
    ProcID _did;
  };
#endif

  class handle_resource : mr_t {
  public:
    explicit handle_resource(mr_t *upstream) noexcept;
    handle_resource(size_t initSize, mr_t *upstream) noexcept;
    handle_resource() noexcept;
    ~handle_resource() override;

    mr_t *upstream_resource() const noexcept { return _upstream; }

    void *handle() const noexcept { return _handle; }
    void *address(uintptr_t offset) const noexcept { return (_handle + offset); }
    uintptr_t acquire(size_t bytes, size_t alignment) {
      char *ret = (char *)this->do_allocate(bytes, alignment);
      return ret - _handle;
    }

  protected:
    void *do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void *p, size_t bytes, size_t alignment) override;
    bool do_is_equal(const mr_t &other) const noexcept override;

  private:
    size_t _bufSize{128 * sizeof(void *)}, _align;
    mr_t *const _upstream{nullptr};
    char *_handle{nullptr}, *_head{nullptr};
  };

  /// https://en.cppreference.com/w/cpp/named_req/Allocator#Allocator_completeness_requirements
  // An allocator type X for type T additionally satisfies the allocator
  // completeness requirements if both of the following are true regardless of
  // whether T is a complete type: X is a complete type Except for value_type, all
  // the member types of std::allocator_traits<X> are complete types.

#if 0
  /// for automatic dynamic memory management
  struct memory_pools : mr_t {
    /// https://stackoverflow.com/questions/46509152/why-in-x86-64-the-virtual-address-are-4-bits-shorter-than-physical-48-bits-vs

    using poolid = unsigned char;
    static constexpr poolid nPools = 4;
    /// 9-bit per page-level: 512, 4K, 2M, 1G
    static constexpr size_t block_bits[nPools] = {9, 12, 21, 30};
    static constexpr size_t block_sizes(poolid pid) noexcept {
      return static_cast<size_t>(1) << block_bits[pid];
    }
    static constexpr poolid pool_index(size_t bytes) noexcept {
      const poolid nbits = bit_count(bytes);
      for (poolid i = 0; i < nPools; ++i)
        if (block_bits[i] > nbits) return i;
      return nPools - 1;
    }

    memory_pools(mr_t *source) {
      for (char i = 0; i < nPools; ++i) {
        pmr::pool_options opt{/*.max_blocks_per_chunk = */ 0,
                              /*.largest_required_pool_block = */ block_sizes(i)};
        /// thread-safe version
        _pools[i] = std::make_unique<synchronized_pool_resource>(opt, source);
      }
    }

  protected:
    void *do_allocate(size_t bytes, size_t alignment) override {
      const poolid pid = pool_index(bytes);
      return _pools[pid]->allocate(bytes, alignment);
    }

    void do_deallocate(void *p, size_t bytes, size_t alignment) override {
      const poolid pid = pool_index(bytes);
      _pools[pid]->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const mr_t &other) const noexcept override {
      return this == dynamic_cast<memory_pools *>(const_cast<mr_t *>(&other));
    }

  private:
    std::array<std::unique_ptr<mr_t>, nPools> _pools;
  };

  template <size_t... Ns> struct static_memory_pools : mr_t {
    using poolid = char;
    static constexpr poolid nPools = sizeof...(Ns);
    static constexpr size_t block_bits[nPools] = {Ns...};
    static constexpr size_t block_sizes(poolid pid) noexcept {
      return static_cast<size_t>(1) << block_bits[pid];
    }
    static constexpr poolid pool_index(size_t bytes) noexcept {
      const poolid nbits = bit_count(bytes);
      for (poolid i = 0; i < nPools; ++i)
        if (block_bits[i] > nbits) return i;
      return nPools - 1;
    }

    static_memory_pools(mr_t *source) {
      for (char i = 0; i < nPools; ++i) {
        pmr::pool_options opt{/*.max_blocks_per_chunk = */ 0,
                              /*.largest_required_pool_block = */ block_sizes(i)};
        _pools[i] = std::make_unique<unsynchronized_pool_resource>(opt, source);
      }
    }

  protected:
    void *do_allocate(size_t bytes, size_t alignment) override {
      const poolid pid = pool_index(bytes);
      return _pools[pid]->allocate(bytes, alignment);
    }

    void do_deallocate(void *p, size_t bytes, size_t alignment) override {
      const poolid pid = pool_index(bytes);
      _pools[pid]->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const mr_t &other) const noexcept override {
      return this == dynamic_cast<static_memory_pools *>(const_cast<mr_t *>(&other));
    }

  private:
    std::array<std::unique_ptr<mr_t>, nPools> _pools;
  };
#endif

  struct ZPC_CORE_API general_allocator {
    general_allocator() noexcept : _mr{&raw_memory_resource<host_mem_tag>::instance()} {};
    general_allocator(const general_allocator &other) : _mr{other.resource()} {}
    general_allocator(mr_t *r) noexcept : _mr{r} {}

    mr_t *resource() const { return _mr; }

    void *allocate(size_t bytes, size_t align = alignof(std::max_align_t)) {
      return resource()->allocate(bytes, align);
    }
    void deallocate(void *p, size_t bytes, size_t align = alignof(std::max_align_t)) {
      resource()->deallocate(p, bytes, align);
    }

  private:
    mr_t *_mr{nullptr};
  };

  struct ZPC_CORE_API heap_allocator : general_allocator {
    heap_allocator() : general_allocator{&raw_memory_resource<host_mem_tag>::instance()} {}
  };

  struct ZPC_CORE_API stack_allocator {
    explicit stack_allocator(mr_t *mr, size_t totalMemBytes, size_t alignBytes);
    stack_allocator() = delete;
    ~stack_allocator();

    mr_t *resource() const noexcept { return _mr; }

    /// from taichi
    void *allocate(size_t bytes);
    void deallocate(void *p, size_t);
    void reset() { _head = _data; }

    char *_data, *_head, *_tail;
    size_t _align;

  private:
    mr_t *_mr{nullptr};
  };

}  // namespace zs
