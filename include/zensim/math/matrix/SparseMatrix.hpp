#pragma once
#include "zensim/container/Vector.hpp"
#include "zensim/container/Bcht.hpp"

namespace zs {

  template <typename T = float, bool RowMajor = true, typename Ti = int, typename Tn = int,
            typename AllocatorT = ZSPmrAllocator<>>
  struct SparseMatrix {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "SparseMatrix only works with zspmrallocator for now.");
    static_assert(std::is_default_constructible_v<T> && std::is_trivially_copyable_v<T>,
                  "element is not default-constructible or trivially-copyable!");

    static constexpr bool is_row_major = RowMajor;
    using value_type = T;
    using allocator_type = AllocatorT;
    using size_type = std::make_unsigned_t<Tn>;
    using difference_type = std::make_signed_t<size_type>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    using index_type = std::make_signed_t<Ti>;
    using subscript_type = zs::vec<index_type, 2>;
    using table_type
        = zs::bcht<subscript_type, sint_t, true, zs::universal_hash<subscript_type>, 16>;

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    decltype(auto) memoryLocation() const noexcept { return _ptrs.get_allocator().location; }
    ProcID devid() const noexcept { return memoryLocation().devid(); }
    memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) get_allocator() const noexcept { return _ptrs.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (std::size_t)1 << (std::size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }
    pointer allocate(std::size_t bytes) {
      /// virtual memory way
      auto &allocator = _ptrs;
      if constexpr (is_virtual_zs_allocator<allocator_type>::value) {
        allocator.commit(0, bytes);
        return (pointer)allocator.address(0);
      }
      /// conventional way
      else
        return (pointer)allocator.allocate(bytes, std::alignment_of_v<value_type>);
    }

    /// allocator-aware
    SparseMatrix(const allocator_type &allocator, Ti ni, Ti nj)
        : _nrows{ni}, _ncols{nj}, _ptrs{allocator, 1}, _inds{allocator, 0}, _vals{allocator, 0} {
		_ptrs.setVal(0);
	}
    explicit SparseMatrix(Ti ni, Ti nj, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : SparseMatrix{get_default_allocator(mre, devid), ni, nj} {}
    SparseMatrix(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : SparseMatrix{get_default_allocator(mre, devid), 0, 0} {}

    Ti _nrows = 0, _ncols = 0;  // for square matrix, nrows = ncols
    zs::Vector<Tn, allocator_type> _ptrs{};
    zs::Vector<Ti, allocator_type> _inds{};
    zs::Vector<value_type, allocator_type> _vals{};
  };

}  // namespace zs