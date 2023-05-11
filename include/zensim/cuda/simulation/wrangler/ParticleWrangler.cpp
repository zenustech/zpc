#include "Wrangler.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/types/Tuple.h"

// zs::TileVector<float, 32, zs::ZSPmrAllocator<false>>;
using ZenoParticlesType = zs::TileVector<float, 32, zs::ZSPmrAllocator<false>>;

using ZenoParticlesView = zs::TileVectorView<zs::execspace_e::cuda, ZenoParticlesType, false>;
using ConstZenoParticlesView
    = zs::TileVectorView<zs::execspace_e::cuda, const ZenoParticlesType, false>;

using ZenoIndexBucketsType
    = zs::IndexBuckets<3, int, int, zs::grid_e::collocated, zs::ZSPmrAllocator<false>>;
using ZenoIndexBucketsView = zs::IndexBucketsView<zs::execspace_e::cuda, ZenoIndexBucketsType>;
using ConstZenoIndexBucketsView
    = zs::IndexBucketsView<zs::execspace_e::cuda, const ZenoIndexBucketsType>;

using ZenoLBvh = zs::LBvh<3, int, zs::f32, zs::ZSPmrAllocator<false>>;
using ZenoLBvhView = zs::LBvhView<zs::execspace_e::cuda, ZenoLBvh>;
using ConstZenoLBvhView = zs::LBvhView<zs::execspace_e::cuda, const ZenoLBvh>;

__device__ void zfx_wrangle_func(float *globals, float const *params);

extern "C" __global__ void zpc_particle_wrangle_kernel(size_t npars,
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

extern "C" __global__ void zpc_particle_wrangler_kernel(size_t npars, zs::f32 const *params,
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
    size_t npars, int isBox, float radius,
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
    for (size_t j = 0; j < bucketSize; ++j) {
      auto pj = ibs.indices[bucketOffset + j];
      if (pj == pi) continue;  // skip myself
      auto xj = neighborPos(pj);
      bool inRange = false;
      if (!isBox)
        inRange = (xi - xj).l2NormSqr() < radius * radius;
      else
        inRange = (xi - xj).abs().max() < radius;
      if (inRange) {
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

extern "C" __global__ void zpc_particle_neighbor_wrangler_kernel(
    size_t npars, int isBox, float radius, ZenoParticlesView pars,
    ConstZenoParticlesView neighborPars, ConstZenoIndexBucketsView ibs, zs::f32 const *params,
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
  auto xi = pars.pack<3>("x", pi);
  auto coord = ibs.bucketCoord(xi) - 1;

  for (auto &&iter : zs::ndrange<3>(3)) {
    auto offset = coord + zs::make_vec<int>(iter);
    auto bucketNo = ibs.bucketNo(offset);
    if (bucketNo == -1) continue;  // skip inactive cell

    auto bucketOffset = ibs.offsets[bucketNo];
    auto bucketSize = ibs.counts[bucketNo];
    for (size_t j = 0; j != bucketSize; ++j) {
      auto pj = ibs.indices[bucketOffset + j];
      if (pj == pi) continue;  // skip myself
      auto xj = neighborPars.pack<3>("x", pj);
      bool inRange = false;
      if (!isBox)
        inRange = (xi - xj).l2NormSqr() < radius * radius;
      else
        inRange = (xi - xj).abs().max() < radius;
      if (inRange) {
        /// assign neighbor particle channels
        for (int i = 0; i != nchns; ++i)
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

extern "C" __global__ void zpc_particle_neighbor_bvh_wrangle_kernel(
    size_t npars, int isBox, float radius2,
    zs::VectorView<zs::execspace_e::cuda, const zs::Vector<zs::vec<zs::f32, 3>>> pos,
    zs::VectorView<zs::execspace_e::cuda, const zs::Vector<zs::vec<zs::f32, 3>>> neighborPos,
    ConstZenoLBvhView lbvh, zs::f32 const *params, int nchns, zs::AccessorAoSoA *accessors) {
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
  lbvh.iter_neighbors(xi, [&](int pid) {
    auto xj = neighborPos(pid);
    if (!isBox)
      if ((xi - xj).l2NormSqr() > radius2) return;
    {
      /// assign neighbor particle channels
      for (int i = 0; i < nchns; ++i)
        if (accessors[i].aux) globals[i] = *(T *)accessors[i](pid);
      zfx_wrangle_func((float *)globals, (float const *)params);
    }
  });

  /// write back
  /// assign target particle channels
  for (int i = 0; i < nchns; ++i)
    if (!accessors[i].aux) *(T *)accessors[i](pi) = globals[i];
}

extern "C" __global__ void zpc_particle_neighbor_bvh_wrangler_kernel(
    size_t npars, int isBox, float radius2, ZenoParticlesView pars,
    ConstZenoParticlesView neighborPars, ConstZenoLBvhView lbvh, zs::f32 const *params, int nchns,
    zs::AccessorAoSoA *accessors) {
  auto pi = blockIdx.x * blockDim.x + threadIdx.x;
  if (pi >= npars) return;
  using T = zs::f32;
  static_assert(zs::is_same_v<T, float>, "wtf");
  T globals[64];

  /// assign target particle channels
  for (int i = 0; i < nchns; ++i)
    if (!accessors[i].aux) globals[i] = *(T *)accessors[i](pi);

  /// execute
  auto xi = pars.template pack<3>("x", pi);
  lbvh.iter_neighbors(xi, [&](int pid) {
    auto xj = neighborPars.pack<3>("x", pid);
    if (!isBox)
      if ((xi - xj).l2NormSqr() > radius2) return;
    {
      /// assign neighbor particle channels
      for (int i = 0; i < nchns; ++i)
        if (accessors[i].aux) globals[i] = *(T *)accessors[i](pid);
      zfx_wrangle_func((float *)globals, (float const *)params);
    }
  });

  /// write back
  /// assign target particle channels
  for (int i = 0; i < nchns; ++i)
    if (!accessors[i].aux) *(T *)accessors[i](pi) = globals[i];
}

extern "C" __global__ void zpc_particle_particle_wrangler_kernel(size_t npars,
                                                                 size_t nNeighborPars,
                                                                 zs::f32 const *params, int nchns,
                                                                 zs::AccessorAoSoA *accessors) {
  auto pi = blockIdx.x * blockDim.x + threadIdx.x;
  if (pi >= npars) return;

  using T = zs::f32;
  static_assert(zs::is_same_v<T, float>, "wtf");
  T globals[64];

  /// assign target particle channels
  for (int i = 0; i < nchns; ++i)
    if (!accessors[i].aux) globals[i] = *(T *)accessors[i](pi);

  /// assign neighbor particle channels
  for (size_t pj = 0; pj != nNeighborPars; ++pj) {
    for (int i = 0; i != nchns; ++i)
      if (accessors[i].aux) globals[i] = *(T *)accessors[i](pj);
    zfx_wrangle_func((float *)globals, (float const *)params);
  }

  /// write back
  /// assign target particle channels
  for (int i = 0; i < nchns; ++i)
    if (!accessors[i].aux) *(T *)accessors[i](pi) = globals[i];
}
