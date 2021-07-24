#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/types/Tuple.h"

extern __device__ void zfx_wrangle_func(float *globals, float const *params);

extern "C" __global__ void zpc_particle_wrangle_kernel(
    std::size_t npars, zs::f32 const *params,
    zs::ParticlesProxy<zs::execspace_e::cuda, zs::Particles<zs::f32, 3>> particles,
    zs::ConstitutiveModelConfig config) {
  auto pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= npars) return;

  zs::f32 globals[30];  // avoid frequent DRAM access
  zs::u8 offset = 0;
  constexpr int dim = particles.dim;
  /// init globals
  globals[offset++] = particles.mass(pid);
  for (int d = 0; d < dim; ++d) globals[offset++] = particles.pos(pid)(d);
  for (int d = 0; d < dim; ++d) globals[offset++] = particles.vel(pid)(d);
  for (int d = 0; d < dim * dim; ++d) globals[offset++] = particles.C(pid)(d);

  if (std::holds_alternative<zs::FixedCorotatedConfig>(config))
    for (int d = 0; d < dim * dim; ++d) globals[offset++] = particles.F(pid)(d);
  else if (std::holds_alternative<zs::EquationOfStateConfig>(config))
    globals[offset++] = particles.J(pid);

  /// execute
  zfx_wrangle_func((float *)globals, (float const *)params);

  /// write back
  if (std::holds_alternative<zs::FixedCorotatedConfig>(config)) {
    for (int d = dim * dim - 1; d >= 0; --d) particles.F(pid)[d] = globals[--offset];
  } else if (std::holds_alternative<zs::EquationOfStateConfig>(config))
    particles.J(pid) = globals[--offset];

  for (int d = dim * dim - 1; d >= 0; --d) particles.C(pid)[d] = globals[--offset];
  for (int d = dim - 1; d >= 0; --d) particles.vel(pid)[d] = globals[--offset];
  for (int d = dim - 1; d >= 0; --d) particles.pos(pid)[d] = globals[--offset];
  particles.mass(pid) = globals[--offset];
}
