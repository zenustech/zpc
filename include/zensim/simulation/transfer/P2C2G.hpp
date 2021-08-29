#pragma once
#include "Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel_Vol_dP.hpp"
#include "zensim/simulation/Utils.hpp"

namespace zs {

  template <transfer_scheme_e, typename ConstitutiveModel, typename BucketsT, typename ParticlesT,
            typename TableT, typename GridT>
  struct P2C2GTransfer;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename BucketsT,
            typename ParticlesT, typename TableT, typename T, int d, auto l>
  P2C2GTransfer(wrapv<space>, wrapv<scheme>, float, Model, BucketsT, ParticlesT, TableT,
                Grids<T, d, l>)
      -> P2C2GTransfer<scheme, Model, IndexBucketsView<space, BucketsT>,
                       ParticlesView<space, ParticlesT>, HashTableView<space, TableT>,
                       GridsView<space, Grids<T, d, l>>>;

  template <execspace_e space, transfer_scheme_e scheme, typename ModelT, typename BucketsT,
            typename ParticlesT, typename TableT, typename GridsT>
  struct P2C2GTransfer<scheme, ModelT, IndexBucketsView<space, BucketsT>,
                       ParticlesView<space, ParticlesT>, HashTableView<space, TableT>,
                       GridsView<space, GridsT>> {
    using model_t = ModelT;  ///< constitutive model
    using buckets_t = IndexBucketsView<space, BucketsT>;
    using particles_t = ParticlesView<space, ParticlesT>;
    using partition_t = HashTableView<space, TableT>;
    using grids_t = GridsView<space, GridsT>;
    using value_type = typename grids_t::value_type;
    static constexpr int dim = particles_t::dim;
    static_assert(particles_t::dim == partition_t::dim && particles_t::dim == GridsT::dim,
                  "[particle-partition-grid] dimension mismatch");

    explicit P2C2GTransfer(wrapv<space>, wrapv<scheme>, float dt, const ModelT& model,
                           BucketsT& buckets, ParticlesT& particles, TableT& table, GridsT& grids)
        : model{model},
          buckets{proxy<space>(buckets)},
          particles{proxy<space>(particles)},
          partition{proxy<space>(table)},
          grids{proxy<space>(grids)},
          dt{dt} {}

    constexpr float dxinv() const { return static_cast<decltype(grids._dx)>(1.0) / grids._dx; }

    constexpr void operator()(typename grids_t::size_type blockid,
                              typename grids_t::cell_index_type cellid) noexcept {
      value_type const dx = grids._dx;
      value_type const dx_inv = dxinv();
      if constexpr (dim == 3) {
        // value_type const D_inv = 4.f * dx_inv * dx_inv;
        using TV = vec<value_type, dim>;
        using TM = vec<value_type, dim * dim>;

        value_type m_c{(value_type)0};
        TV mv_c{TV::zeros()};
        TM QDinv_c{TM::zeros()};
        TV QDinvXp_c{TV::zeros()};

        /// stage 1 (p -> c)
        // auto coord = buckets.coord(bucketno);
        auto coord = partition._activeKeys[blockid] * (typename partition_t::Tn)grids_t::side_length
                     + grids_t::cellid_to_coord(cellid).template cast<typename partition_t::Tn>();
        auto posc = (coord + (value_type)0.5) * dx;
        auto checkInKernelRange = [&posc, dx](auto&& posp) -> bool {
          for (int d = 0; d != dim; ++d)
            if (gcem::abs(posp[d] - posc[d]) > dx) return false;
          return true;
        };
        coord = coord - 1;  /// move to base coord
        for (auto&& iter : ndrange<dim>(3)) {
          auto bucketno = buckets.bucketNo(coord + make_vec<typename partition_t::Tn>(iter));
          if (bucketno >= 0) {
            for (int st = buckets.offsets(bucketno), ed = buckets.offsets(bucketno + 1); st != ed;
                 ++st) {
              auto parid = buckets.indices(st);
              auto posp = particles.pos(parid);
              if (checkInKernelRange(posp)) {
                TV Dinv{};
                for (int d = 0; d != dim; ++d) {
                  Dinv[d] = gcem::fmod(posp[d], dx * (value_type)0.5);
                  Dinv[d] = ((value_type)2 / (dx * dx - 2 * Dinv[d] * Dinv[d]));
                }
                auto vel = particles.vel(parid);
                auto mass = particles.mass(parid);
                auto C = particles.B(parid);
                for (int d = 0; d != dim * dim; ++d) C[d] *= Dinv[d / dim];
                RM_CVREF_T(C) contrib{};
                if constexpr (is_same_v<model_t, EquationOfStateConfig>) {
                  float J = particles.J(parid);
                  float vol = model.volume * J;
                  float pressure = model.bulk;
                  {
                    float J2 = J * J;
                    float J4 = J2 * J2;
                    // pressure = pressure * (powf(J, -model.gamma) - 1.f);
                    pressure = pressure * (1 / (J * J2 * J4) - 1);  // from Bow
                  }
                  contrib[0] = ((C[0] + C[0]) * model.viscosity - pressure) * vol;
                  contrib[1] = (C[1] + C[3]) * model.viscosity * vol;
                  contrib[2] = (C[2] + C[6]) * model.viscosity * vol;

                  contrib[3] = (C[3] + C[1]) * model.viscosity * vol;
                  contrib[4] = ((C[4] + C[4]) * model.viscosity - pressure) * vol;
                  contrib[5] = (C[5] + C[7]) * model.viscosity * vol;

                  contrib[6] = (C[6] + C[2]) * model.viscosity * vol;
                  contrib[7] = (C[7] + C[5]) * model.viscosity * vol;
                  contrib[8] = ((C[8] + C[8]) * model.viscosity - pressure) * vol;

                } else {
                  const auto [mu, lambda] = lame_parameters(model.E, model.nu);
                  RM_CVREF_T(C) F{particles.F(parid)};
                  if constexpr (is_same_v<model_t, FixedCorotatedConfig>) {
                    compute_stress_fixedcorotated(model.volume, mu, lambda, F, contrib);
                  } else if constexpr (is_same_v<model_t, VonMisesFixedCorotatedConfig>) {
                    compute_stress_vonmisesfixedcorotated(model.volume, mu, lambda,
                                                          model.yieldStress, F, contrib);
                  } else {
                    /// with plasticity additionally
                    float logJp = particles.logJp(parid);
                    if constexpr (is_same_v<model_t, DruckerPragerConfig>) {
                      compute_stress_sand(model.volume, mu, lambda, model.cohesion, model.beta,
                                          model.yieldSurface, model.volumeCorrection, logJp, F,
                                          contrib);
                    } else if constexpr (is_same_v<model_t, NACCConfig>) {
                      compute_stress_nacc(model.volume, mu, lambda, model.bulk(), model.xi,
                                          model.beta, model.Msqr(), model.hardeningOn, logJp, F,
                                          contrib);
                    }
                    particles.logJp(parid) = logJp;
                  }
                }

                for (int d = 0; d != dim * dim; ++d) contrib[d] *= Dinv[d / dim];
                contrib = C * mass - contrib * dt;

                auto xcxp = posc - posp;
                value_type Wpc = 1.f;
                auto diff = xcxp * dx_inv;
                for (int d = 0; d != dim; ++d) {
                  const auto xabs = gcem::abs(diff[d]);
                  if (xabs <= 1)
                    Wpc *= ((value_type)1. - xabs);
                  else
                    Wpc *= 0.f;
                }
                m_c += mass * Wpc;
                for (int d = 0; d != dim; ++d) {
                  mv_c[d] += mass * vel[d] * Wpc;
                  QDinvXp_c[d] += (contrib[d] * posp[0] + contrib[3 + d] * posp[1]
                                   + contrib[6 + d] * posp[2])
                                  * Wpc;
                }
                for (int d = 0; d != dim * dim; ++d) QDinv_c[d] += contrib[d] * Wpc;
              }  // check in range
            }    // iterate in the bucket
          }
        }  // iterate buckets within neighbor

        /// stage 2 (c -> i)
        auto grid = grids.grid(collocated_v);
        coord = coord + 1;  /// move to base coord
        for (auto&& iter : ndrange<dim>(2)) {
          const auto coordi = coord + make_vec<typename partition_t::Tn>(iter);
          const auto posi = coordi * dx;
          auto [blockno, local_index]
              = unpack_coord_in_grid(coordi, grids_t::side_length, partition);
          if (blockno >= 0) {
            auto block = grid.block(blockno);
            constexpr value_type Wci = 1. / 8;  // 3d
            const auto cellid = grids_t::coord_to_cellid(local_index);
            atomic_add(wrapv<space>{}, &block(0, cellid), m_c * Wci);
            for (int d = 0; d != dim; ++d)
              atomic_add(
                  wrapv<space>{}, &block(1 + d, cellid),
                  (mv_c[d]
                   + ((QDinv_c[d] * posi[0] + QDinv_c[3 + d] * posi[1] + QDinv_c[6 + d] * posi[2])
                      - QDinvXp_c[d]))
                      * Wci);
          }
        }
      }  // dim == 3
    }

    model_t model;
    buckets_t buckets;
    particles_t particles;
    partition_t partition;
    grids_t grids;
    float dt;
  };

}  // namespace zs