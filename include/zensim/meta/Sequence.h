#pragma once

#include <type_traits>
#include <utility>

#include "../TypeAlias.hpp"
#include "Functional.h"
#include "Meta.h"

namespace zs {

  namespace type_impl {
    template <typename T, T... Is, typename... Ts>
    struct indexed_types<std::integer_sequence<T, Is...>, Ts...> : indexed_type<Is, Ts>... {};
  }  // namespace type_impl

  /// type_seq

}  // namespace zs
