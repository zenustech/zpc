#pragma once
#include "zensim/math/VecInterface.hpp"

namespace zs {

  struct Printer {
    template <typename VecT, enable_if_t<is_arithmetic_v<typename VecT::value_type>> = 0>
    constexpr void operator()(const VecInterface<VecT>& v) const {
      using index_type = typename VecT::index_type;
      using value_type = typename VecT::value_type;
      if constexpr (is_floating_point_v<value_type>)
        for (index_type i = 0; i != VecT::extent; ++i) printf("%f ", (float)v.val(i));
      else
        for (index_type i = 0; i != VecT::extent; ++i) printf("%d ", (int)v.val(i));
      printf("\n");
    }
  };

}  // namespace zs