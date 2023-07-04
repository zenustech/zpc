#pragma once

#include "zensim/ZpcFunctional.hpp"
#include "zensim/ZpcIterator.hpp"
#include "zensim/ZpcMathUtils.hpp"
#include "zensim/ZpcReflection.hpp"
#include "zensim/ZpcTuple.hpp"
#include "zensim/math/Tensor.hpp"
#include "zensim/math/Vec.h"
//
#include "zensim/py_interop/TileVectorView.hpp"
#include "zensim/py_interop/VectorView.hpp"
//
#include "zensim/execution/Atomics.hpp"
#include "zensim/types/Function.h"
#include "zensim/types/Property.h"
#include "zensim/types/SmallVector.hpp"
// #include "zensim/types/SourceLocation.hpp"

namespace zs {
  template<class T, class=int>
  struct printf_target; 

  template<class T>
  struct printf_target<T, enable_if_t<is_floating_point_v<T>>>
  {
    using type = float; 
    constexpr static char placeholder[] = "%f"; 
  }; 

  template<class T>
  struct printf_target<T, enable_if_t<is_integral_v<T>>>
  {
    using type = int; 
    constexpr static char placeholder[] = "%d"; 
  }; 

  template<class T>
  struct printf_target<T, enable_if_t<is_same_v<decay_t<T>, char*>>>
  {
    using type = const char *; 
    constexpr static char placeholder[] = "%s"; 
  }; 

  // only int, float, double for now 
  template <class... Types> ZS_FUNCTION void print_internal(Types &&...args) {
    auto formatStr = ((zs::SmallString{printf_target<remove_cvref_t<Types>>::placeholder} + zs::SmallString{" "}) + ...);
    auto formatChars = (formatStr).asChars(); 
    printf(formatChars, static_cast<typename printf_target<remove_cvref_t<Types>>::type>(args)...); 
  }

  template <class... Types> ZS_FUNCTION void print(Types &&...args) { 
    print_internal(FWD(args)..., "\n"); 
  }
}