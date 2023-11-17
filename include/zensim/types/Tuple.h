#pragma once

#include <tuple>

#include "zensim/ZpcTuple.hpp"

namespace zs {

  template <typename> struct is_std_tuple : false_type {};
  template <typename... Ts> struct is_std_tuple<std::tuple<Ts...>> : true_type {};
  template <typename T> static constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

}  // namespace zs
