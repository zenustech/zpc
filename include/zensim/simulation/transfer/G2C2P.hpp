#pragma once
#include "Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/math/matrix/MatrixUtils.h"
#include "zensim/physics/ConstitutiveModel.hpp"

namespace zs {

  template <transfer_scheme_e, typename ConstitutiveModel, typename BucketsT, typename GridsT,
            typename TableT, typename ParticlesT>
  struct G2C2PTransfer;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename BucketsT,
            typename T, int d, auto l, typename TableT, typename ParticlesT>
  G2C2PTransfer(wrapv<space>, wrapv<scheme>, float, Model, BucketsT, Grids<T, d, l>, TableT,
                ParticlesT)
      -> G2C2PTransfer<scheme, Model, IndexBucketsView<space, BucketsT>,
                       GridsView<space, Grids<T, d, l>>, HashTableView<space, TableT>,
                       ParticlesView<space, ParticlesT>>;

  template <typename ParticlesT> struct PreG2C2PTransfer;
  template <transfer_scheme_e, typename ConstitutiveModel, typename ParticlesT>
  struct PostG2C2PTransfer;

  template <execspace_e space, typename ParticlesT> PreG2C2PTransfer(wrapv<space>, ParticlesT)
      -> PreG2C2PTransfer<ParticlesView<space, ParticlesT>>;
  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename ParticlesT>
  PostG2C2PTransfer(wrapv<space>, wrapv<scheme>, float, float, Model, ParticlesT)
      -> PostG2C2PTransfer<scheme, Model, ParticlesView<space, ParticlesT>>;

  template <execspace_e space, transfer_scheme_e scheme, typename ModelT, typename BucketsT,
            typename GridsT, typename TableT, typename ParticlesT>
  struct G2C2PTransfer<scheme, ModelT, IndexBucketsView<space, BucketsT>, GridsView<space, GridsT>,
                       HashTableView<space, TableT>, ParticlesView<space, ParticlesT>> {
    using model_t = ModelT;  ///< constitutive model
    using buckets_t = IndexBucketsView<space, BucketsT>;
    using grids_t = GridsView<space, GridsT>;
    using value_type = typename grids_t::value_type;
    using partition_t = HashTableView<space, TableT>;
    using particles_t = ParticlesView<space, ParticlesT>;
    static constexpr int dim = particles_t::dim;
    static_assert(particles_t::dim == partition_t::dim && particles_t::dim == grids_t::dim,
                  "[particle-partition-grid] dimension mismatch");

    explicit G2C2PTransfer(wrapv<space>, wrapv<scheme>, float dt, const ModelT& model,
                           BucketsT& buckets, GridsT& grids, TableT& table, ParticlesT& particles)
        : model{model},
          buckets{proxy<space>(buckets)},
          grids{proxy<space>(grids)},
          partition{proxy<space>(table)},
          particles{proxy<space>(particles)},
          dt{dt} {}

    constexpr void operator()(typename grids_t::size_type blockid,
                              typename grids_t::cell_index_type cellid) noexcept {
      value_type const dx = grids._dx;
      value_type const dx_inv = (value_type)1 / dx;
      if constexpr (grids_t::dim == 3) {
        using TV = vec<value_type, grids_t::dim>;
        using TM = vec<value_type, grids_t::dim * grids_t::dim>;

        // auto coord = buckets.coord(bucketno);
        auto coord = partition._activeKeys[blockid] * (typename partition_t::Tn)grids_t::side_length
                     + grids_t::cellid_to_coord(cellid).template cast<typename partition_t::Tn>();
        TV v_c{TV::zeros()};
        TM v_cross_x_c{TM::zeros()};

        auto grid = grids.grid(collocated_v);
        for (auto&& iter : ndrange<grids_t::dim>(2)) {
          const auto coordi = coord + make_vec<typename partition_t::Tn>(iter);
          const auto posi = coordi * dx;

          // auto [block, local_index]
          //    = unpack_coord_in_grid(coordi, grids_t::side_length, partition, grid);
          auto [blockno, local_index]
              = unpack_coord_in_grid(coordi, grids_t::side_length, partition);
          if (blockno >= 0) {
            auto block = grid.block(blockno);
            value_type W = 1. / 8;  // 3d
            const auto cellid = grids_t::coord_to_cellid(local_index);
            auto v_i = block.pack<grids_t::dim>(1, grids_t::coord_to_cellid(local_index));
            v_c += v_i * W;
            for (int d = 0; d < grids_t::dim * grids_t::dim; ++d)
              v_cross_x_c[d] += W * v_i(d % 3) * posi(d / 3);
          }
        }

        auto posc = (coord + (value_type)0.5) * dx;
        auto checkInKernelRange = [&posc, dx](auto&& posp) -> bool {
          for (int d = 0; d != grids_t::dim; ++d)
            if (gcem::abs(posp[d] - posc[d]) > dx) return false;
          return true;
        };

        coord = coord - 1;  /// move to base coord
        for (auto&& iter : ndrange<grids_t::dim>(3)) {
          auto bucketno = buckets.bucketNo(coord + make_vec<typename partition_t::Tn>(iter));
          if (bucketno >= 0) {
            for (int st = buckets.offsets(bucketno), ed = buckets.offsets(bucketno + 1); st != ed;
                 ++st) {
              auto parid = buckets.indices(st);
              auto posp = particles.pos(parid);
              if (checkInKernelRange(posp)) {
                auto xcxp = posc - posp;
                value_type W = 1.f;
                auto diff = xcxp * dx_inv;
                for (int d = 0; d != dim; ++d) {
                  const auto xabs = gcem::abs(diff[d]);
                  if (xabs <= 1)
                    W *= ((value_type)1. - xabs);
                  else
                    W *= 0.f;
                }
                auto& vp = particles.vel(parid);
                auto& Bp = particles.B(parid);

                // v_p
                for (int d = 0; d != dim; ++d) atomic_add(wrapv<space>{}, &vp[d], v_c[d] * W);
                // B_p
                for (int d = 0; d != dim * dim; ++d)
                  atomic_add(wrapv<space>{}, &Bp[d],
                             W * (v_cross_x_c[d] - v_c(d % dim) * posp(d / dim)));
              }
            }
          }
        }
      }
    }

    model_t model;
    buckets_t buckets;
    grids_t grids;
    partition_t partition;
    particles_t particles;
    float dt;
  };

  template <execspace_e space, typename ParticlesT>
  struct PreG2C2PTransfer<ParticlesView<space, ParticlesT>> {
    using particles_t = ParticlesView<space, ParticlesT>;
    using value_type = typename particles_t::T;

    explicit PreG2C2PTransfer(wrapv<space>, ParticlesT& particles)
        : particles{proxy<space>(particles)} {}

    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      particles.vel(parid) = particles_t::TV::zeros();
      particles.C(parid) = particles_t::TM::zeros();
    }

    particles_t particles;
  };

  template <execspace_e space, transfer_scheme_e scheme, typename ModelT, typename ParticlesT>
  struct PostG2C2PTransfer<scheme, ModelT, ParticlesView<space, ParticlesT>> {
    using model_t = ModelT;  ///< constitutive model
    using particles_t = ParticlesView<space, ParticlesT>;
    using value_type = typename particles_t::T;
    static constexpr int dim = particles_t::dim;
    using TV = vec<value_type, dim>;

    explicit PostG2C2PTransfer(wrapv<space>, wrapv<scheme>, float dt, float dx, const ModelT& model,
                               ParticlesT& particles)
        : model{model}, particles{proxy<space>(particles)}, dt{dt}, dx{dx} {}

    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      if constexpr (particles_t::dim == 3) {
        using ivec3 = vec<int, particles_t::dim>;
        using vec3 = vec<value_type, particles_t::dim>;
        using vec9 = vec<value_type, particles_t::dim * particles_t::dim>;
        using vec3x3 = vec<value_type, particles_t::dim, particles_t::dim>;
        vec3 pos{particles.pos(parid)};
        vec3 vel{particles.vel(parid)};

        vec9 C{particles.B(parid)};
        TV Dinv{};

        for (int d = 0; d != dim; ++d) {
          Dinv[d] = gcem::fmod(pos[d], dx * (value_type)0.5);
          Dinv[d] = ((value_type)2 / (dx * dx - 2 * Dinv[d] * Dinv[d]));
        }
        for (int d = 0; d != dim * dim; ++d) C[d] *= Dinv[d / dim];

        if constexpr (is_same_v<model_t, EquationOfStateConfig>) {
          float J = particles.J(parid);
          J = (1 + (C[0] + C[4] + C[8]) * dt) * J;
          // if (J < 0.1) J = 0.1;
          particles.J(parid) = J;
        } else {
          vec9 oldF{particles.F(parid)}, tmp{}, F{};
          for (int d = 0; d < 9; ++d) tmp(d) = C[d] * dt + ((d & 0x3) ? 0.f : 1.f);
          matrixMatrixMultiplication3d(tmp.data(), oldF.data(), F.data());
          particles.F(parid) = F;
        }

        pos += vel * dt;
        particles.pos(parid) = pos;
      }
    }

    model_t model;
    particles_t particles;
    float dt, dx;
  };

}  // namespace zs