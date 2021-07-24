#include "Wrangler.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/types/Tuple.h"

extern __device__ void zfx_wrangle_func(float *globals, float const *params);

extern "C" __global__ void zpc_particle_wrangle_kernel(std::size_t npars,
                                                       zs::ConstitutiveModelConfig config,
                                                       zs::f32 const *params, int nchns,
                                                       zs::AccessorAoSoA *accessors) {
  auto pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= npars) return;

  zs::f32 globals[30];  // avoid frequent DRAM access
  // if (sizeof(zs::f32) == accessors[0].unitBytes)
  /// init globals
  for (int i = 0; i < nchns; ++i) globals[i] = *(zs::f32 *)accessors[i](pid);

  /// execute
  zfx_wrangle_func((float *)globals, (float const *)params);

  /// write back
  for (int i = 0; i < nchns; ++i) *(zs::f32 *)accessors[i](pid) = globals[i];
}
