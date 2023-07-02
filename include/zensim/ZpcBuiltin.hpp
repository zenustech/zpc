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

namespace zs 
{
template<class T, enable_if_t<is_floating_point_v<T>> = 0>
constexpr void print(T t)
{
    printf("%f ", (float)t); 
}

template<class T, enable_if_t<is_integral_v<T>> = 0>
constexpr void print(T t)
{
    printf("%d ", (int)t); 
}

constexpr void print(const SmallString& s)
{
    printf("%s ", s.asChars()); 
}

template<class T, class... Types>
constexpr void print(T t, Types... args)
{
    print(t)
    print(...args); 
}
}