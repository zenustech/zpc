#include <random>

#include "zensim/math/MathUtils.h"
#include "zensim/math/Rotation.hpp"
#include "zensim/math/Tensor.hpp"
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/math/matrix/SVD.hpp"
#include "zensim/math/matrix/Transform.hpp"
#include "zensim/physics/constitutive_models/NeoHookean.hpp"
#include "zensim/resource/Resource.h"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/core.h"

struct Dummy {
  using T = float;
  constexpr Dummy() : n{zs::detail::deduce_numeric_lowest<int>()} {}
  int n = zs::detail::deduce_numeric_lowest<int>();
  operator int() { return n; }
};

int main() {
  using namespace zs;
  {
    using namespace index_literals;

    static_assert(is_same_v<decltype(0_th), index_t<0>>, "index literal error!");

    tuple a{1};
    const tuple b{2.f};
    const Dummy bb{};
    auto bbb = zs::forward_as_tuple(std::move(bb));
    fmt::print("get(b): [{}], get(bb): [{}]\n", get_var_type_str(get<0>(b)),
               get_var_type_str(get<0>(bbb)));
    using TT = std::add_const_t<Dummy>;  // with & will fail
    fmt::print("{}::T: {}\n", get_type_str<TT>(), get_type_str<typename TT::T>());
    tuple c{'a', 3u};
    tuple d{c};
    tuple<> e{};
    tuple<> f{};
    auto ff = make_tuple([]() { return 0; });
    auto x = tuple_cat(e, b, a, f, f, d);
    auto xx = tuple_cat(-1, b, c, f, f, 33ul, f);

    // auto xxx = -1 + b + d + f + f + 33ul + f + ff;
    // auto xxx = -1 + b + d + f + f + zs::forward_as_tuple(33ul) + f + ff;
    Dummy dummy{};
    const auto dum = zs::forward_as_tuple(dummy);
    auto xxx = -1 + b + d + f + f + std::cref(dummy) + f + ff;
    // auto xxx = -1 + b + d + f + f + zs::forward_as_tuple([](){return Dummy{};}()) + f + ff;
    // auto xxx = -1 + b + d + f + f + zs::forward_as_tuple(std::move(dummy)) + f + ff;
    // auto xxx = -1 + b + d + f + f + zs::forward_as_tuple(dummy) + f + ff;

    // auto xxx = tuple_cat(e, -1);
    auto xxxx = tuple<int&&, char&&>{777, 'z'};
    fmt::print("[{}, {}]\n", get_var_type_str(get<0>(xxxx)), get_var_type_str(xxxx.get<1>()));

    // fmt::print("x {}\n", x);
    {
#if 1
      dummy.n = -100;
      fmt::print("xxx [{}]: {}, {}, {}, {}, {}, {}\n", get_var_type_str(xxx), zs::get<0>(xxx),
                 xxx.get<1>(), xxx.get<2>(), xxx.get<3>(), xxx.get<4>().n, xxx.get<5>()());
#endif
    }

#if 1
    fmt::print("xx [{}]: {}, {}, {}, {}, {}\n", get_var_type_str(xx), zs::get<0>(xx), xx.get<1>(),
               xx.get<2>(), xx.get<3>(), xx.get<4>());

    auto [_1, _2, _3, _4] = x;
    fmt::print("x [{}]: ({}, {}, {}, {}) ({}, {}, {}, {})\n", get_var_type_str(x), zs::get<0>(x),
               x.get<1>(), x.get<2>(), x.get<3>(), _1, _2, _3, _4);
    auto y = detail::concatenation_op{}(type_seq<>{});
    fmt::print("y: {}\n", get_var_type_str(y));
    // auto y = tuple_cat(e, f);
    static_assert(tuple_size<tuple<>>::value == 0, "wtf");
    fmt::print("d: {}\n", get_var_type_str(d));
    auto z = tuple_cat(a, b, c, d);
    fmt::print("a+b+c+d: {}\n", get_var_type_str(z));
#endif
  }
  return 0;
}