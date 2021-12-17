#include "Wrangler.hpp"
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/types/Tuple.h"

__device__ void zfx_wrangle_func(float *globals, float const *params);

extern "C" __global__ void zpc_particle_wrangle_kernel(std::size_t npars,
                                                       zs::ConstitutiveModelConfig config,
                                                       zs::f32 const *params, int nchns,
                                                       zs::AccessorAoSoA *accessors) {
  auto pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= npars) return;

  zs::f32 globals[64];
  /// assign channels
  for (int i = 0; i < nchns; ++i) globals[i] = *(zs::f32 *)accessors[i](pid);

  /// execute
  zfx_wrangle_func((float *)globals, (float const *)params);

  /// write back
  for (int i = 0; i < nchns; ++i) *(zs::f32 *)accessors[i](pid) = globals[i];
}

extern "C" __global__ void zpc_particle_wrangler_kernel(std::size_t npars, zs::f32 const *params,
                                                        int nchns, zs::AccessorAoSoA *accessors) {
  auto pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= npars) return;

  zs::f32 globals[64];
  /// assign channels
  for (int i = 0; i < nchns; ++i) globals[i] = *(zs::f32 *)accessors[i](pid);

  /// execute
  zfx_wrangle_func((float *)globals, (float const *)params);

  /// write back
  for (int i = 0; i < nchns; ++i) *(zs::f32 *)accessors[i](pid) = globals[i];
}

// todo: templatize indexbucket parameter type
extern "C" __global__ void zpc_particle_neighbor_wrangle_kernel(
    std::size_t npars,
    zs::VectorView<zs::execspace_e::cuda, const zs::Vector<zs::vec<zs::f32, 3>>> pos,
    zs::VectorView<zs::execspace_e::cuda, const zs::Vector<zs::vec<zs::f32, 3>>> neighborPos,
    zs::IndexBucketsView<zs::execspace_e::cuda, zs::IndexBuckets<3>> ibs, zs::f32 const *params,
    int nchns, zs::AccessorAoSoA *accessors) {
  auto pi = blockIdx.x * blockDim.x + threadIdx.x;
  if (pi >= npars) return;

  using T = zs::f32;
  static_assert(zs::is_same_v<T, float>, "wtf");
  T globals[64];

  /// assign target particle channels
  for (int i = 0; i < nchns; ++i)
    if (!accessors[i].aux) globals[i] = *(T *)accessors[i](pi);

  /// execute
  auto xi = pos(pi);
  auto coord = ibs.bucketCoord(xi) - 1;

  for (auto &&iter : zs::ndrange<3>(3)) {  // 3x3x3 neighbor
    auto offset = coord + zs::make_vec<int>(iter);
    auto bucketNo = ibs.bucketNo(offset);
    if (bucketNo == -1) continue;  // skip this cell

    auto bucketOffset = ibs.offsets[bucketNo];
    auto bucketSize = ibs.counts[bucketNo];
    for (std::size_t j = 0; j < bucketSize; ++j) {
      auto pj = ibs.indices[bucketOffset + j];
      if (pj == pi) continue;  // skip myself
      auto xj = neighborPos(pj);
      auto disSqr = (xi - xj).l2NormSqr();
      if (disSqr < ibs.dx * ibs.dx) {
        /// assign neighbor particle channels
        for (int i = 0; i < nchns; ++i)
          if (accessors[i].aux) globals[i] = *(T *)accessors[i](pj);
        zfx_wrangle_func((float *)globals, (float const *)params);
      }
    }
  }

  /// write back
  /// assign target particle channels
  for (int i = 0; i < nchns; ++i)
    if (!accessors[i].aux) *(T *)accessors[i](pi) = globals[i];
}
