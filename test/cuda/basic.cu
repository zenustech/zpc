#include <random>

#include "zensim/container/DenseField.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/Predicates.hpp"
#include "zensim/math/Vec.h"
#if ZS_ENABLE_OPENMP
#  include "zensim/omp/execution/ExecutionPolicy.hpp"
#endif
#include "zensim/zpc_tpls/fmt/color.h"

template <typename VectorView> struct SomeFunc {
  constexpr void operator()(int i) { vals[i] = i; }
  VectorView vals;
};
template <typename T> SomeFunc(T) -> SomeFunc<T>;  // CTAD

#if 0
struct YourStruct{
	TileVector<int> a;
	TileVector<float> b;
};
template <typename >
struct YourStructView {
	constexpr void doSomething() {
		;
	}
	TileVectorView<>a;
	TileVectorView<>b;
};

template <zs::execspace_e space>
auto view(YourStruct &container) {
	return zs::make_tuple(view<space>(container.a), view<space>(container.b));
}
#  if 0
template <typename Pol, auto space = zs::remove_ref_t<Pol>::exec_tag::value>
void func(Pol &&pol) {
	pol(range(..), []ZS_LAMBDA() mutable {
		;
		;
	});
}
#  endif
#endif

int main() {
  using namespace zs;
  auto cudaPol = cuda_exec().sync(true);
  constexpr auto cuspace = execspace_e::cuda;
#if ZS_ENABLE_OPENMP
  auto pol = omp_exec();
  constexpr auto space = execspace_e::openmp;
#else
  auto pol = seq_exec();
  constexpr auto space = execspace_e::host;
#endif

  constexpr size_t N = 10000;
  Vector<int> dvals{N, memsrc_e::device};
  Vector<int> hvals{N, memsrc_e::host /*default device index is -1 indicating cpu*/};

  /// @note expressions in the comments could be used in substitution for the one ahead,
  /// unless stated otherwise

  /// @note three host-side initializations produce the same outcome
#if 1
  pol(Collapse{hvals.size()} /*range(hvals.size())*/,
      [hvals = view<space>(hvals) /*proxy<space>(hvals)*/](int i) mutable { hvals[i] = i; });
#elif 1
  // range-based for
  pol(enumerate(hvals), [](int i /*auto i*/, int &v /*auto &v*/) { v = i; });
#else
  {
    int i = 0;
    for (auto &v : hvals) v = i++;
  }
#endif

  /// @note three device-side implementations have equivalent effects
#if 1
  cudaPol(
      Collapse{dvals.size()} /*range(dvals.size())*/,
      [dvals = proxy<cuspace>(
           dvals) /*view<cuspace>(dvals)*/] __device__ /*this is required for cuda extended lambda*/
      (int i) mutable /*make sure the captured view of the container is writable*/ {
        dvals[i] = i;
      });
#elif 1
  cudaPol(enumerate(dvals /*'range(dvals)', 'dvals' are also valid here*/),
          [] __device__(auto id, auto &v) { v = id; });
#else
  // custom functors are also supported
  cudaPol(Collapse{vals.size()}, SomeFunc{view<cuspace>(vals)});
#endif

  auto tmp = dvals.clone({memsrc_e::host, -1});
  for (int i = 0; i != tmp.size(); ++i)
    if (tmp[i] != hvals[i]) {
      throw std::runtime_error(
          fmt::format("mismatch at [{}]: device size: {}, host size: {}\n", i, tmp[i], hvals[i]));
    }

  ///
  constexpr std::size_t numPars = 2000;
  // 32 is the tile size (num elements per tile)
  // memsrc_e::um is accessible from both host and cuda device
  TileVector<float, 32> pars{{{"m", 1}, {"x", 3}}, numPars, memsrc_e::um, 0};
  // inserting two additional properties, e.g. "v" for velocity, "F" for deformation gradient.
  pars.append_channels(cudaPol, {{"v", 3}, {"F", 9}});
  // tilevector can also be 'resized' like zs::Vector
  pars.resize(N);
  // init
  cudaPol(Collapse{pars.size()},
          [pars = view<cuspace>({}, pars)/*this overload allows property/channel access through names (i.e. string), the alternate unmaed view is constructed by 'view<cuspace>(pars)'*/] __device__(int pi) mutable {
        	  // initialize "m" property with a float
            pars("m", pi) = 1;
	          // for properties with more than one channel
	          // directly read/write a certain component is allowed
            pars("x", 0, pi) = 0;
            pars("x", 1, pi) = pi;
            pars("x", 2, pi) = -pars("x", 1, pi);

            {
              auto F = pars.pack(zs::dim_c<3, 3>, "F", pi);
	            static_assert(is_same_v<RM_CVREF_T(F), vec<float, 3, 3>>, "???");
            }

            // multi-channel property can also be initialized altogether
            pars.tuple(zs::dim_c<3, 3>, "F", pi) = zs::vec<float, 3, 3>::identity();
	          static_assert(is_same_v<RM_CVREF_T(pars.tuple(zs::dim_c<3, 3>, "F", pi)), zs::tuple<float&, float&, float&, float&, float&, float&, float&, float&, float&>>, "???");

            pars("v", 1, pi) = pi * 2;
          });

  /// @note zs::TileVector contents are generally accessed through its view (like many other
  /// containers except zs::Vector)
  auto tmp1 = proxy<space>({}, pars);
  /// @note get the property channel offset ahead is preferred
  const int xOffset = pars.getPropertyOffset("x");
  for (int i = 0; i != tmp1.size() /*i.e. N for now*/ && i != 10; ++i)
    fmt::print("par[{}]: mass {}, pos {}, {}, {}; v[y]: {}\n", i, tmp1("m", i),
               tmp1(xOffset + 0, i), tmp1(xOffset + 1, i), tmp1(xOffset + 2, i), tmp1("v", 1, i));

  DenseField<int> a{{3, 6, 2}, memsrc_e::host, -1};
  DenseField<u32> b{a.get_allocator(), {2, 3, 1}};
  pol(enumerate(a), [](int i, int &v) { v = -i; });
  pol(enumerate(b), [](int i, u32 &v) { v = i; });

  int id = 0;
  auto av = view<space>(a, false_c);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < 2; ++k, --id)
        if (a(i, j, k) != id || av(i, j, k) != id || av[-id] != id)
          throw std::runtime_error(
              fmt::format("densefield a failed! ({}, {}, {}) read {}, should be {}\n", i, j, k,
                          a(i, j, k), id));
  return 0;
}