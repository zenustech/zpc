#pragma once
#include "zensim/math/Vec.h"
#include "zensim/tpls/magic_enum.hpp"
#include "zensim/types/Iterator.h"

namespace zs {

  enum struct kernel_e { quadratic = 3, cubic = 4 };

  template <int dim_, kernel_e kt = kernel_e::quadratic, typename T = f32, typename Ti = int>
  struct LocalArena {
    using value_type = T;
    using index_type = Ti;
    static constexpr int dim = dim_;
    using TV = vec<value_type, dim>;
    using IV = vec<index_type, dim>;

    constexpr LocalArena(wrapv<kt>, value_type dx, TV pos)
        : dx{dx}, localPos{pos}, iter{ndrange<dim>(magic_enum::enum_integer(kt))} {
      ;
    }

    constexpr IV currentOffset() { return make_vec<index_type>(*iter); }  // tuple

    TV localPos;
    collapse_t<Ti, dim> iter;
    const f32 dx;
  };

  // template <kernel_e kt, typename T, typename Ti> LocalArena();

}  // namespace zs