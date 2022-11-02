#pragma once
#include "SparseGrid.hpp"

namespace zs {

  /// @brief stores all leaf nodes of an adaptive octree
  template <int dim_, typename ValueT = f32, int SideLength = 8,
            typename AllocatorT = ZSPmrAllocator<>, typename IntegerCoordT = i32>
  struct AdaptiveGrid : SparseGrid<dim_, ValueT, SideLength, AllocatorT, IntegerCoordT> {
    using base_t = SparseGrid<dim_, ValueT, SideLength, AllocatorT, IntegerCoordT>;
    using value_type = typename base_t::value_type;
    using allocator_type = typename base_t::allocator_type;
    using index_type = typename base_t::index_type;
    using size_type = typename base_t::size_type;

    using integer_coord_component_type = typename base_t::integer_coord_component_type;
    using coord_component_type = typename base_t::coord_component_type;

    static constexpr int dim = base_t::dim;
    static constexpr integer_coord_component_type side_length = base_t::side_length;
    static constexpr integer_coord_component_type block_size = base_t::block_size;

    using coord_type = typename base_t::coord_type;
    using integer_coord_type = typename base_t::integer_coord_type;

    using packed_value_type = typename base_t::packed_value_type;
    using grid_storage_type = typename base_t::grid_storage_type;
    using transform_type = typename base_t::transform_type;
    using table_type = typename base_t::table_type;

    /// mask
    static constexpr integer_coord_component_type block_word_size = block_size / 64;
    static_assert(block_size % 64 == 0, "block size should be a multiple of 64");
    using mask_storage_type = TileVector<u64, block_word_size, allocator_type>;

    constexpr MemoryLocation memoryLocation() const noexcept { return base_t::memoryLocation(); }
    constexpr ProcID devid() const noexcept { return base_t::devid(); }
    constexpr memsrc_e memspace() const noexcept { return base_t::memspace(); }
    constexpr auto size() const noexcept { return base_t::size(); }
    decltype(auto) get_allocator() const noexcept { return base_t::get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      return base_t::get_default_allocator(mre, devid);
    }

    // active mask (whether a voxel is active, use background value if inactive)
    // signed mask (indicates inside / outside status according to sdf)
    mask_storage_type _mask;
  };

  // special construct for blockwise access (with halos)

}  // namespace zs