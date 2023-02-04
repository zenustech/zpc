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

#if 1
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
        auto coord = partition._activeKeys[blockid] * (typename partition_t::Tn)grids_t::side_length
                     + grids_t::cellid_to_coord(cellid).template cast<typename partition_t::Tn>();
        auto posc = (coord + (value_type)0.5) * dx;
        auto checkInKernelRange = [&posc, dx](auto&& posp) -> bool {
          for (int d = 0; d != dim; ++d)
            if (zs::abs(posp[d] - posc[d]) > dx) return false;
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
                  // Dinv[d] = zs::fmod(posp[d], dx * (value_type)0.5);
                  Dinv[d] = posp[d] - lower_trunc(posp[d] * dx_inv + (value_type)0.5) * dx;
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

                for (int d = 0; d != dim * dim; ++d) contrib[d] *= Dinv[d / dim] * -dt;

                contrib += C * mass;

                auto xcxp = posc - posp;
                value_type Wpc = 1.f;
                auto diff = xcxp * dx_inv;
                for (int d = 0; d != dim; ++d) Wpc *= ((value_type)1. - zs::abs(diff[d]));

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
        auto grid = grids.grid(collocated_c);
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
#else
    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      float const dx = grids._dx;
      float const dx_inv = dxinv();
      if constexpr (dim == 3) {
        float const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, dim>;
        using vec3 = vec<float, dim>;
        using vec9 = vec<float, dim * dim>;
        using vec3x3 = vec<float, dim, dim>;

        vec3 local_pos{particles.pos(parid)};
        vec3 vel{particles.vel(parid)};
        float mass = particles.mass(parid);
        vec9 contrib{}, C{particles.B(parid)};

        vec3 Xrel{}, Dinv{};
        for (int d = 0; d != dim; ++d) {
          Xrel[d] = local_pos[d] - lower_trunc(local_pos[d] * dx_inv + (value_type)0.5) * dx;
          // Xrel[d] = zs::fmod(local_pos[d], dx * (value_type)0.5);
          Dinv[d] = ((value_type)2 / (dx * dx - 2 * Xrel[d] * Xrel[d]));
        }
        for (int d = 0; d != dim * dim; ++d) C[d] *= Dinv[d / dim];
        auto Wpi = vec<value_type, 3, 3, 3>::zeros();
        vec<value_type, dim, 2> Wpc{};
        for (int d = 0; d != dim; ++d) {
          Wpc(d, 0) = 0.5 - Xrel[d] * dx_inv;
          Wpc(d, 1) = 0.5 + Xrel[d] * dx_inv;
        }
        auto weight_pc = [&Wpc](int i, int j, int k) { return Wpc(0, i) * Wpc(1, j) * Wpc(2, k); };

        for (int cx = 0; cx != 2; cx++)
          for (int cy = 0; cy != 2; cy++)
            for (int cz = 0; cz != 2; cz++)
              for (int x = 0; x != 2; x++)
                for (int y = 0; y != 2; y++)
                  for (int z = 0; z != 2; z++) {
                    Wpi(cx + x, cy + y, cz + z) += weight_pc(cx, cy, cz) * (1. / 8);
                  }

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
          vec9 F{particles.F(parid)};
          if constexpr (is_same_v<model_t, FixedCorotatedConfig>) {
            compute_stress_fixedcorotated(model.volume, mu, lambda, F, contrib);
          } else if constexpr (is_same_v<model_t, VonMisesFixedCorotatedConfig>) {
            compute_stress_vonmisesfixedcorotated(model.volume, mu, lambda, model.yieldStress, F,
                                                  contrib);
          } else {
            /// with plasticity additionally
            float logJp = particles.logJp(parid);
            if constexpr (is_same_v<model_t, DruckerPragerConfig>) {
              compute_stress_sand(model.volume, mu, lambda, model.cohesion, model.beta,
                                  model.yieldSurface, model.volumeCorrection, logJp, F, contrib);
            } else if constexpr (is_same_v<model_t, NACCConfig>) {
              compute_stress_nacc(model.volume, mu, lambda, model.bulk(), model.xi, model.beta,
                                  model.Msqr(), model.hardeningOn, logJp, F, contrib);
            }
            particles.logJp(parid) = logJp;
          }
        }

        for (int d = 0; d != dim * dim; ++d) contrib[d] *= Dinv[d / dim] * -dt;
        contrib += C * mass;

        using VT = typename grids_t::value_type;
        auto arena = make_local_arena((VT)dx, local_pos);
        for (auto loc : arena.range()) {
          auto [grid_block, local_index]
              = unpack_coord_in_grid(arena.coord(loc), grids_t::side_length, partition, grids);
          auto xixp = arena.diff(loc);
          value_type W = Wpi(zs::get<0>(loc), zs::get<1>(loc), zs::get<2>(loc));
          const auto cellid = grids_t::coord_to_cellid(local_index);
          atomic_add(wrapv<space>{}, &grid_block(0, cellid), mass * W);
          for (int d = 0; d != particles_t::dim; ++d) {
            // vi: W m v + W m C (xi - xp)
            atomic_add(wrapv<space>{}, &grid_block(1 + d, cellid),
                       W
                           * (mass * vel[d]
                              + (contrib[d] * xixp[0] + contrib[3 + d] * xixp[1]
                                 + contrib[6 + d] * xixp[2])));
          }
        }
      }
    }
#endif

    model_t model;
    buckets_t buckets;
    particles_t particles;
    partition_t partition;
    grids_t grids;
    float dt;
  };

  template <transfer_scheme_e, typename ConstitutiveModel, typename BucketsT, typename ParticlesT,
            typename TableT, typename GridT>
  struct P2C2GTransferMomentum;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename BucketsT,
            typename ParticlesT, typename TableT, typename T, int d, auto l>
  P2C2GTransferMomentum(wrapv<space>, wrapv<scheme>, float, Model, BucketsT, ParticlesT, TableT,
                        Grids<T, d, l>)
      -> P2C2GTransferMomentum<scheme, Model, IndexBucketsView<space, BucketsT>,
                               ParticlesView<space, ParticlesT>, HashTableView<space, TableT>,
                               GridsView<space, Grids<T, d, l>>>;

  template <execspace_e space, transfer_scheme_e scheme, typename ModelT, typename BucketsT,
            typename ParticlesT, typename TableT, typename GridsT>
  struct P2C2GTransferMomentum<scheme, ModelT, IndexBucketsView<space, BucketsT>,
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

    explicit P2C2GTransferMomentum(wrapv<space>, wrapv<scheme>, float dt, const ModelT& model,
                                   BucketsT& buckets, ParticlesT& particles, TableT& table,
                                   GridsT& grids)
        : model{model},
          buckets{proxy<space>(buckets)},
          particles{proxy<space>(particles)},
          partition{proxy<space>(table)},
          grids{proxy<space>(grids)},
          dt{dt} {}

    constexpr float dxinv() const { return static_cast<decltype(grids._dx)>(1.0) / grids._dx; }

#if 1
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
            if (zs::abs(posp[d] - posc[d]) > dx) return false;
          return true;
        };
        coord = coord - 1;  /// move to base coord
        for (auto&& iter : ndrange<dim>(3)) {
          auto bucketno
              = buckets.bucketNo(coord + make_vec<typename RM_CVREF_T(coord)::value_type>(iter));
          if (bucketno >= 0) {
            for (int st = buckets.offsets(bucketno), ed = buckets.offsets(bucketno + 1); st != ed;
                 ++st) {
              auto parid = buckets.indices(st);
              auto posp = particles.pos(parid);
              if (checkInKernelRange(posp)) {
                TV Dinv{};
                for (int d = 0; d != dim; ++d) {
                  // Dinv[d] = zs::fmod(posp[d], dx * (value_type)0.5);
                  Dinv[d] = posp[d] - lower_trunc(posp[d] * dx_inv + (value_type)0.5) * dx;
                  Dinv[d] = ((value_type)2 / (dx * dx - 2 * Dinv[d] * Dinv[d]));
                }
                auto vel = particles.vel(parid);
                auto mass = particles.mass(parid);
                auto C = particles.B(parid);
                for (int d = 0; d != dim * dim; ++d) C[d] *= Dinv[d / dim];
                auto contrib = C * mass;

                auto xcxp = posc - posp;
                value_type Wpc = 1.f;
                auto diff = xcxp * dx_inv;
                for (int d = 0; d != dim; ++d) {
                  const auto xabs = zs::abs(diff[d]);
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
        auto grid = grids.grid(collocated_c);
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
#else
    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      float const dx = grids._dx;
      float const dx_inv = dxinv();
      if constexpr (dim == 3) {
        float const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, dim>;
        using vec3 = vec<float, dim>;
        using vec9 = vec<float, dim * dim>;
        using vec3x3 = vec<float, dim, dim>;

        vec3 local_pos{particles.pos(parid)};
        vec3 vel{particles.vel(parid)};
        float mass = particles.mass(parid);
        vec9 contrib{}, C{particles.B(parid)};

        vec3 Xrel{}, Dinv{};
        for (int d = 0; d != dim; ++d) {
          Xrel[d] = local_pos[d] - lower_trunc(local_pos[d] * dx_inv + (value_type)0.5) * dx;
          // Xrel[d] = zs::fmod(local_pos[d], dx * (value_type)0.5);
          Dinv[d] = ((value_type)2 / (dx * dx - 2 * Xrel[d] * Xrel[d]));
        }
        for (int d = 0; d != dim * dim; ++d) C[d] *= Dinv[d / dim];
        auto Wpi = vec<value_type, 3, 3, 3>::zeros();
        vec<value_type, dim, 2> Wpc{};
        for (int d = 0; d != dim; ++d) {
          Wpc(d, 0) = 0.5 - Xrel[d] * dx_inv;
          Wpc(d, 1) = 0.5 + Xrel[d] * dx_inv;
        }
        auto weight_pc = [&Wpc](int i, int j, int k) { return Wpc(0, i) * Wpc(1, j) * Wpc(2, k); };

        for (int cx = 0; cx != 2; cx++)
          for (int cy = 0; cy != 2; cy++)
            for (int cz = 0; cz != 2; cz++)
              for (int x = 0; x != 2; x++)
                for (int y = 0; y != 2; y++)
                  for (int z = 0; z != 2; z++) {
                    Wpi(cx + x, cy + y, cz + z) += weight_pc(cx, cy, cz) * (1. / 8);
                  }

        using VT = typename grids_t::value_type;
        auto arena = make_local_arena((VT)dx, local_pos);
        for (auto loc : arena.range()) {
          auto [grid_block, local_index]
              = unpack_coord_in_grid(arena.coord(loc), grids_t::side_length, partition, grids);
          auto xixp = arena.diff(loc);
          value_type W = Wpi(zs::get<0>(loc), zs::get<1>(loc), zs::get<2>(loc));
          const auto cellid = grids_t::coord_to_cellid(local_index);
          atomic_add(wrapv<space>{}, &grid_block(0, cellid), mass * W);
          for (int d = 0; d != particles_t::dim; ++d) {
            // vi: W m v + W m C (xi - xp)
            atomic_add(
                wrapv<space>{}, &grid_block(1 + d, cellid),
                W * mass * (vel[d] + (C[d] * xixp[0] + C[3 + d] * xixp[1] + C[6 + d] * xixp[2])));
          }
        }
      }
    }
#endif

    model_t model;
    buckets_t buckets;
    particles_t particles;
    partition_t partition;
    grids_t grids;
    float dt;
  };

  template <transfer_scheme_e, typename ConstitutiveModel, typename BucketsT, typename ParticlesT,
            typename TableT, typename GridT>
  struct P2C2GTransferForce;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename BucketsT,
            typename ParticlesT, typename TableT, typename T, int d, auto l>
  P2C2GTransferForce(wrapv<space>, wrapv<scheme>, float, Model, BucketsT, ParticlesT, TableT,
                     Grids<T, d, l>)
      -> P2C2GTransferForce<scheme, Model, IndexBucketsView<space, BucketsT>,
                            ParticlesView<space, ParticlesT>, HashTableView<space, TableT>,
                            GridsView<space, Grids<T, d, l>>>;

  template <execspace_e space, transfer_scheme_e scheme, typename ModelT, typename BucketsT,
            typename ParticlesT, typename TableT, typename GridsT>
  struct P2C2GTransferForce<scheme, ModelT, IndexBucketsView<space, BucketsT>,
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

    explicit P2C2GTransferForce(wrapv<space>, wrapv<scheme>, float dt, const ModelT& model,
                                BucketsT& buckets, ParticlesT& particles, TableT& table,
                                GridsT& grids)
        : model{model},
          buckets{proxy<space>(buckets)},
          particles{proxy<space>(particles)},
          partition{proxy<space>(table)},
          grids{proxy<space>(grids)},
          dt{dt} {}

    constexpr float dxinv() const { return static_cast<decltype(grids._dx)>(1.0) / grids._dx; }

#if 1
    constexpr void operator()(typename grids_t::size_type blockid,
                              typename grids_t::cell_index_type cellid) noexcept {
      value_type const dx = grids._dx;
      value_type const dx_inv = dxinv();
      if constexpr (dim == 3) {
        // value_type const D_inv = 4.f * dx_inv * dx_inv;
        using TV = vec<value_type, dim>;
        using TM = vec<value_type, dim * dim>;

        TM QDinv_c{TM::zeros()};
        TV QDinvXp_c{TV::zeros()};

        /// stage 1 (p -> c)
        // auto coord = buckets.coord(bucketno);
        auto coord = partition._activeKeys[blockid] * (typename partition_t::Tn)grids_t::side_length
                     + grids_t::cellid_to_coord(cellid).template cast<typename partition_t::Tn>();
        auto posc = (coord + (value_type)0.5) * dx;
        auto checkInKernelRange = [&posc, dx](auto&& posp) -> bool {
          for (int d = 0; d != dim; ++d)
            if (zs::abs(posp[d] - posc[d]) > dx) return false;
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
                  Dinv[d] = posp[d] - lower_trunc(posp[d] * dx_inv + (value_type)0.5) * dx;
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

                for (int d = 0; d != dim * dim; ++d) contrib[d] *= Dinv[d / dim] * -dt;
                // contrib = contrib * -dt;

                auto xcxp = posc - posp;
                value_type Wpc = 1.f;
                auto diff = xcxp * dx_inv;
                for (int d = 0; d != dim; ++d) {
                  const auto xabs = zs::abs(diff[d]);
                  if (xabs <= 1)
                    Wpc *= ((value_type)1. - xabs);
                  else
                    Wpc *= 0.f;
                }
                for (int d = 0; d != dim; ++d)
                  QDinvXp_c[d] += (contrib[d] * posp[0] + contrib[3 + d] * posp[1]
                                   + contrib[6 + d] * posp[2])
                                  * Wpc;
                for (int d = 0; d != dim * dim; ++d) QDinv_c[d] += contrib[d] * Wpc;
              }  // check in range
            }    // iterate in the bucket
          }
        }  // iterate buckets within neighbor

        /// stage 2 (c -> i)
        auto grid = grids.grid(collocated_c);
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
            for (int d = 0; d != dim; ++d)
              atomic_add(
                  wrapv<space>{}, &block(1 + d, cellid),
                  (((QDinv_c[d] * posi[0] + QDinv_c[3 + d] * posi[1] + QDinv_c[6 + d] * posi[2])
                    - QDinvXp_c[d]))
                      * Wci);
          }
        }
      }  // dim == 3
    }
#else
    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      float const dx = grids._dx;
      float const dx_inv = dxinv();
      if constexpr (particles_t::dim == 3) {
        float const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, particles_t::dim>;
        using vec3 = vec<float, particles_t::dim>;
        using vec9 = vec<float, particles_t::dim * particles_t::dim>;
        using vec3x3 = vec<float, particles_t::dim, particles_t::dim>;

        vec3 local_pos{particles.pos(parid)};
        vec3 vel{particles.vel(parid)};
        float mass = particles.mass(parid);
        vec9 contrib{}, C{particles.B(parid)};

        vec3 Xrel{}, Dinv{};
        for (int d = 0; d != dim; ++d) {
          // Xrel[d] = zs::fmod(local_pos[d], dx * (value_type)0.5);
          Xrel[d] = local_pos[d] - lower_trunc(local_pos[d] * dx_inv + (value_type)0.5) * dx;
          Dinv[d] = ((value_type)2 / (dx * dx - 2 * Xrel[d] * Xrel[d]));
        }
        for (int d = 0; d != dim * dim; ++d) C[d] *= Dinv[d / dim];

        auto Wpi = vec<value_type, 3, 3, 3>::zeros();
        vec<value_type, dim, 2> Wpc{};
        for (int d = 0; d != dim; ++d) {
          Wpc(d, 0) = 0.5 - Xrel[d] * dx_inv;
          Wpc(d, 1) = 0.5 + Xrel[d] * dx_inv;
        }
        auto weight_pc = [&Wpc](int i, int j, int k) { return Wpc(0, i) * Wpc(1, j) * Wpc(2, k); };

        for (int cx = 0; cx != 2; cx++)
          for (int cy = 0; cy != 2; cy++)
            for (int cz = 0; cz != 2; cz++)
              for (int x = 0; x != 2; x++)
                for (int y = 0; y != 2; y++)
                  for (int z = 0; z != 2; z++) {
                    Wpi(cx + x, cy + y, cz + z) += weight_pc(cx, cy, cz) * (1. / 8);
                  }

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
          vec9 F{particles.F(parid)};
          if constexpr (is_same_v<model_t, FixedCorotatedConfig>) {
            compute_stress_fixedcorotated(model.volume, mu, lambda, F, contrib);
          } else if constexpr (is_same_v<model_t, VonMisesFixedCorotatedConfig>) {
            compute_stress_vonmisesfixedcorotated(model.volume, mu, lambda, model.yieldStress, F,
                                                  contrib);
          } else {
            /// with plasticity additionally
            float logJp = particles.logJp(parid);
            if constexpr (is_same_v<model_t, DruckerPragerConfig>) {
              compute_stress_sand(model.volume, mu, lambda, model.cohesion, model.beta,
                                  model.yieldSurface, model.volumeCorrection, logJp, F, contrib);
            } else if constexpr (is_same_v<model_t, NACCConfig>) {
              compute_stress_nacc(model.volume, mu, lambda, model.bulk(), model.xi, model.beta,
                                  model.Msqr(), model.hardeningOn, logJp, F, contrib);
            }
            particles.logJp(parid) = logJp;
          }
        }

        for (int d = 0; d != dim * dim; ++d) contrib[d] *= Dinv[d / dim] * -dt;

        using VT = typename grids_t::value_type;
        auto arena = make_local_arena((VT)dx, local_pos);
        for (auto loc : arena.range()) {
          auto [grid_block, local_index]
              = unpack_coord_in_grid(arena.coord(loc), grids_t::side_length, partition, grids);
          auto xixp = arena.diff(loc);
          value_type W = Wpi(zs::get<0>(loc), zs::get<1>(loc), zs::get<2>(loc));
          const auto cellid = grids_t::coord_to_cellid(local_index);
          for (int d = 0; d != particles_t::dim; ++d) {
            atomic_add(
                wrapv<space>{}, &grid_block(1 + d, cellid),
                (contrib[d] * xixp[0] + contrib[3 + d] * xixp[1] + contrib[6 + d] * xixp[2]) * W);
          }
        }
      }
    }
#endif

    model_t model;
    buckets_t buckets;
    particles_t particles;
    partition_t partition;
    grids_t grids;
    float dt;
  };

}  // namespace zs