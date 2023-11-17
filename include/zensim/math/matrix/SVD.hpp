#pragma once

#include "QRSVD.hpp"
#include "zensim/math/VecInterface.hpp"
#include "zensim/math/matrix/Utility.h"

namespace zs {
  namespace math {

    union un_fu {
      float f;
      unsigned int ui;
    };

    template <typename T> constexpr void svd_3d(T a11, T a12, T a13, T a21, T a22, T a23, T a31,
                                                T a32, T a33,  // input A
                                                T& u11, T& u12, T& u13, T& u21, T& u22, T& u23,
                                                T& u31, T& u32, T& u33,  // output U
                                                T& s11,
                                                // float &s12, float &s13, float &s21,
                                                T& s22,
                                                // float &s23, float &s31, float &s32,
                                                T& s33,  // output S
                                                T& v11, T& v12, T& v13, T& v21, T& v22, T& v23,
                                                T& v31, T& v32, T& v33  // output V
                                                ) noexcept {
      constexpr auto gone = 1065353216;
      constexpr auto gsine_pi_over_eight = 1053028117;
      constexpr auto gcosine_pi_over_eight = 1064076127;
      constexpr auto gsmall_number = 1.e-12f;
      constexpr auto gtiny_number = 1.e-20f;
      constexpr auto gfour_gamma_squared = 5.8284273147583007813f;
      un_fu Sa11{}, Sa21{}, Sa31{}, Sa12{}, Sa22{}, Sa32{}, Sa13{}, Sa23{}, Sa33{};
      un_fu Su11{}, Su21{}, Su31{}, Su12{}, Su22{}, Su32{}, Su13{}, Su23{}, Su33{};
      un_fu Sv11{}, Sv21{}, Sv31{}, Sv12{}, Sv22{}, Sv32{}, Sv13{}, Sv23{}, Sv33{};
      un_fu Sc{}, Ss{}, Sch{}, Ssh{};
      un_fu Stmp1{}, Stmp2{}, Stmp3{}, Stmp4{}, Stmp5{};
      un_fu Ss11{}, Ss21{}, Ss31{}, Ss22{}, Ss32{}, Ss33{};
      un_fu Sqvs{}, Sqvvx{}, Sqvvy{}, Sqvvz{};

      Sa11.f = a11;
      Sa12.f = a12;
      Sa13.f = a13;
      Sa21.f = a21;
      Sa22.f = a22;
      Sa23.f = a23;
      Sa31.f = a31;
      Sa32.f = a32;
      Sa33.f = a33;

      //###########################################################
      // Compute normal equations matrix
      //###########################################################

      Ss11.f = Sa11.f * Sa11.f;
      Stmp1.f = Sa21.f * Sa21.f;
      Ss11.f = (Stmp1.f + Ss11.f);
      Stmp1.f = Sa31.f * Sa31.f;
      Ss11.f = (Stmp1.f + Ss11.f);

      Ss21.f = Sa12.f * Sa11.f;
      Stmp1.f = Sa22.f * Sa21.f;
      Ss21.f = (Stmp1.f + Ss21.f);
      Stmp1.f = Sa32.f * Sa31.f;
      Ss21.f = (Stmp1.f + Ss21.f);

      Ss31.f = Sa13.f * Sa11.f;
      Stmp1.f = Sa23.f * Sa21.f;
      Ss31.f = (Stmp1.f + Ss31.f);
      Stmp1.f = Sa33.f * Sa31.f;
      Ss31.f = (Stmp1.f + Ss31.f);

      Ss22.f = Sa12.f * Sa12.f;
      Stmp1.f = Sa22.f * Sa22.f;
      Ss22.f = (Stmp1.f + Ss22.f);
      Stmp1.f = Sa32.f * Sa32.f;
      Ss22.f = (Stmp1.f + Ss22.f);

      Ss32.f = Sa13.f * Sa12.f;
      Stmp1.f = Sa23.f * Sa22.f;
      Ss32.f = (Stmp1.f + Ss32.f);
      Stmp1.f = Sa33.f * Sa32.f;
      Ss32.f = (Stmp1.f + Ss32.f);

      Ss33.f = Sa13.f * Sa13.f;
      Stmp1.f = Sa23.f * Sa23.f;
      Ss33.f = (Stmp1.f + Ss33.f);
      Stmp1.f = Sa33.f * Sa33.f;
      Ss33.f = (Stmp1.f + Ss33.f);

      Sqvs.f = 1.f;
      Sqvvx.f = 0.f;
      Sqvvy.f = 0.f;
      Sqvvz.f = 0.f;

      //###########################################################
      // Solve symmetric eigenproblem using Jacobi iteration
      //###########################################################
      for (int i = 0; i < 4; i++) {
        Ssh.f = Ss21.f * 0.5f;
        Stmp5.f = (Ss11.f - Ss22.f);

        Stmp2.f = Ssh.f * Ssh.f;
        Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
        Ssh.ui = Stmp1.ui & Ssh.ui;
        Sch.ui = Stmp1.ui & Stmp5.ui;
        Stmp2.ui = ~Stmp1.ui & gone;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Stmp3.f = (Stmp1.f + Stmp2.f);
        Stmp4.f = zs::rsqrt(Stmp3.f);  // __frsqrt_rn(Stmp3.f);

        Ssh.f = Stmp4.f * Ssh.f;
        Sch.f = Stmp4.f * Sch.f;
        Stmp1.f = gfour_gamma_squared * Stmp1.f;
        Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

        Stmp2.ui = gsine_pi_over_eight & Stmp1.ui;
        Ssh.ui = ~Stmp1.ui & Ssh.ui;
        Ssh.ui = Ssh.ui | Stmp2.ui;
        Stmp2.ui = gcosine_pi_over_eight & Stmp1.ui;
        Sch.ui = ~Stmp1.ui & Sch.ui;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Sc.f = (Stmp2.f - Stmp1.f);
        Ss.f = Sch.f * Ssh.f;
        Ss.f = (Ss.f + Ss.f);

        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        Stmp3.f = (Stmp1.f + Stmp2.f);
        Ss33.f = Ss33.f * Stmp3.f;
        Ss31.f = Ss31.f * Stmp3.f;
        Ss32.f = Ss32.f * Stmp3.f;
        Ss33.f = Ss33.f * Stmp3.f;

        Stmp1.f = Ss.f * Ss31.f;
        Stmp2.f = Ss.f * Ss32.f;
        Ss31.f = Sc.f * Ss31.f;
        Ss32.f = Sc.f * Ss32.f;
        Ss31.f = (Stmp2.f + Ss31.f);
        Ss32.f = (Ss32.f - Stmp1.f);

        Stmp2.f = Ss.f * Ss.f;
        Stmp1.f = Ss22.f * Stmp2.f;
        Stmp3.f = Ss11.f * Stmp2.f;
        Stmp4.f = Sc.f * Sc.f;
        Ss11.f = Ss11.f * Stmp4.f;
        Ss22.f = Ss22.f * Stmp4.f;
        Ss11.f = (Ss11.f + Stmp1.f);
        Ss22.f = (Ss22.f + Stmp3.f);
        Stmp4.f = (Stmp4.f - Stmp2.f);
        Stmp2.f = (Ss21.f + Ss21.f);
        Ss21.f = Ss21.f * Stmp4.f;
        Stmp4.f = Sc.f * Ss.f;
        Stmp2.f = Stmp2.f * Stmp4.f;
        Stmp5.f = Stmp5.f * Stmp4.f;
        Ss11.f = (Ss11.f + Stmp2.f);
        Ss21.f = (Ss21.f - Stmp5.f);
        Ss22.f = (Ss22.f - Stmp2.f);

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        Stmp1.f = Ssh.f * Sqvvx.f;
        Stmp2.f = Ssh.f * Sqvvy.f;
        Stmp3.f = Ssh.f * Sqvvz.f;
        Ssh.f = Ssh.f * Sqvs.f;

        Sqvs.f = Sch.f * Sqvs.f;
        Sqvvx.f = Sch.f * Sqvvx.f;
        Sqvvy.f = Sch.f * Sqvvy.f;
        Sqvvz.f = Sch.f * Sqvvz.f;

        Sqvvz.f = (Sqvvz.f + Ssh.f);
        Sqvs.f = (Sqvs.f - Stmp3.f);
        Sqvvx.f = (Sqvvx.f + Stmp2.f);
        Sqvvy.f = (Sqvvy.f - Stmp1.f);

        //////////////////////////////////////////////////////////////////////////
        // (1->3)
        //////////////////////////////////////////////////////////////////////////
        Ssh.f = Ss32.f * 0.5f;
        Stmp5.f = (Ss22.f - Ss33.f);

        Stmp2.f = Ssh.f * Ssh.f;
        Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
        Ssh.ui = Stmp1.ui & Ssh.ui;
        Sch.ui = Stmp1.ui & Stmp5.ui;
        Stmp2.ui = ~Stmp1.ui & gone;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Stmp3.f = (Stmp1.f + Stmp2.f);
        Stmp4.f = zs::rsqrt(Stmp3.f);  // __frsqrt_rn(Stmp3.f);

        Ssh.f = Stmp4.f * Ssh.f;
        Sch.f = Stmp4.f * Sch.f;
        Stmp1.f = gfour_gamma_squared * Stmp1.f;
        Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

        Stmp2.ui = gsine_pi_over_eight & Stmp1.ui;
        Ssh.ui = ~Stmp1.ui & Ssh.ui;
        Ssh.ui = Ssh.ui | Stmp2.ui;
        Stmp2.ui = gcosine_pi_over_eight & Stmp1.ui;
        Sch.ui = ~Stmp1.ui & Sch.ui;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Sc.f = (Stmp2.f - Stmp1.f);
        Ss.f = Sch.f * Ssh.f;
        Ss.f = (Ss.f + Ss.f);

        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        Stmp3.f = (Stmp1.f + Stmp2.f);
        Ss11.f = Ss11.f * Stmp3.f;
        Ss21.f = Ss21.f * Stmp3.f;
        Ss31.f = Ss31.f * Stmp3.f;
        Ss11.f = Ss11.f * Stmp3.f;

        Stmp1.f = Ss.f * Ss21.f;
        Stmp2.f = Ss.f * Ss31.f;
        Ss21.f = Sc.f * Ss21.f;
        Ss31.f = Sc.f * Ss31.f;
        Ss21.f = (Stmp2.f + Ss21.f);
        Ss31.f = (Ss31.f - Stmp1.f);

        Stmp2.f = Ss.f * Ss.f;
        Stmp1.f = Ss33.f * Stmp2.f;
        Stmp3.f = Ss22.f * Stmp2.f;
        Stmp4.f = Sc.f * Sc.f;
        Ss22.f = Ss22.f * Stmp4.f;
        Ss33.f = Ss33.f * Stmp4.f;
        Ss22.f = (Ss22.f + Stmp1.f);
        Ss33.f = (Ss33.f + Stmp3.f);
        Stmp4.f = (Stmp4.f - Stmp2.f);
        Stmp2.f = (Ss32.f + Ss32.f);
        Ss32.f = Ss32.f * Stmp4.f;
        Stmp4.f = Sc.f * Ss.f;
        Stmp2.f = Stmp2.f * Stmp4.f;
        Stmp5.f = Stmp5.f * Stmp4.f;
        Ss22.f = (Ss22.f + Stmp2.f);
        Ss32.f = (Ss32.f - Stmp5.f);
        Ss33.f = (Ss33.f - Stmp2.f);

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        Stmp1.f = Ssh.f * Sqvvx.f;
        Stmp2.f = Ssh.f * Sqvvy.f;
        Stmp3.f = Ssh.f * Sqvvz.f;
        Ssh.f = Ssh.f * Sqvs.f;

        Sqvs.f = Sch.f * Sqvs.f;
        Sqvvx.f = Sch.f * Sqvvx.f;
        Sqvvy.f = Sch.f * Sqvvy.f;
        Sqvvz.f = Sch.f * Sqvvz.f;

        Sqvvx.f = (Sqvvx.f + Ssh.f);
        Sqvs.f = (Sqvs.f - Stmp1.f);
        Sqvvy.f = (Sqvvy.f + Stmp3.f);
        Sqvvz.f = (Sqvvz.f - Stmp2.f);

#if 1
        //////////////////////////////////////////////////////////////////////////
        // 1 -> 2
        //////////////////////////////////////////////////////////////////////////

        Ssh.f = Ss31.f * 0.5f;
        Stmp5.f = (Ss33.f - Ss11.f);

        Stmp2.f = Ssh.f * Ssh.f;
        Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
        Ssh.ui = Stmp1.ui & Ssh.ui;
        Sch.ui = Stmp1.ui & Stmp5.ui;
        Stmp2.ui = ~Stmp1.ui & gone;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Stmp3.f = (Stmp1.f + Stmp2.f);
        Stmp4.f = zs::rsqrt(Stmp3.f);  //__frsqrt_rn(Stmp3.f);

        Ssh.f = Stmp4.f * Ssh.f;
        Sch.f = Stmp4.f * Sch.f;
        Stmp1.f = gfour_gamma_squared * Stmp1.f;
        Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

        Stmp2.ui = gsine_pi_over_eight & Stmp1.ui;
        Ssh.ui = ~Stmp1.ui & Ssh.ui;
        Ssh.ui = Ssh.ui | Stmp2.ui;
        Stmp2.ui = gcosine_pi_over_eight & Stmp1.ui;
        Sch.ui = ~Stmp1.ui & Sch.ui;
        Sch.ui = Sch.ui | Stmp2.ui;

        Stmp1.f = Ssh.f * Ssh.f;
        Stmp2.f = Sch.f * Sch.f;
        Sc.f = (Stmp2.f - Stmp1.f);
        Ss.f = Sch.f * Ssh.f;
        Ss.f = (Ss.f + Ss.f);

#  ifdef DEBUG_JACOBI_CONJUGATE
        printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f, Sch.f);
#  endif

        //###########################################################
        // Perform the actual Givens conjugation
        //###########################################################

        Stmp3.f = (Stmp1.f + Stmp2.f);
        Ss22.f = Ss22.f * Stmp3.f;
        Ss32.f = Ss32.f * Stmp3.f;
        Ss21.f = Ss21.f * Stmp3.f;
        Ss22.f = Ss22.f * Stmp3.f;

        Stmp1.f = Ss.f * Ss32.f;
        Stmp2.f = Ss.f * Ss21.f;
        Ss32.f = Sc.f * Ss32.f;
        Ss21.f = Sc.f * Ss21.f;
        Ss32.f = (Stmp2.f + Ss32.f);
        Ss21.f = (Ss21.f - Stmp1.f);

        Stmp2.f = Ss.f * Ss.f;
        Stmp1.f = Ss11.f * Stmp2.f;
        Stmp3.f = Ss33.f * Stmp2.f;
        Stmp4.f = Sc.f * Sc.f;
        Ss33.f = Ss33.f * Stmp4.f;
        Ss11.f = Ss11.f * Stmp4.f;
        Ss33.f = (Ss33.f + Stmp1.f);
        Ss11.f = (Ss11.f + Stmp3.f);
        Stmp4.f = (Stmp4.f - Stmp2.f);
        Stmp2.f = (Ss31.f + Ss31.f);
        Ss31.f = Ss31.f * Stmp4.f;
        Stmp4.f = Sc.f * Ss.f;
        Stmp2.f = Stmp2.f * Stmp4.f;
        Stmp5.f = Stmp5.f * Stmp4.f;
        Ss33.f = (Ss33.f + Stmp2.f);
        Ss31.f = (Ss31.f - Stmp5.f);
        Ss11.f = (Ss11.f - Stmp2.f);

        //###########################################################
        // Compute the cumulative rotation, in quaternion form
        //###########################################################

        Stmp1.f = Ssh.f * Sqvvx.f;
        Stmp2.f = Ssh.f * Sqvvy.f;
        Stmp3.f = Ssh.f * Sqvvz.f;
        Ssh.f = Ssh.f * Sqvs.f;

        Sqvs.f = Sch.f * Sqvs.f;
        Sqvvx.f = Sch.f * Sqvvx.f;
        Sqvvy.f = Sch.f * Sqvvy.f;
        Sqvvz.f = Sch.f * Sqvvz.f;

        Sqvvy.f = (Sqvvy.f + Ssh.f);
        Sqvs.f = (Sqvs.f - Stmp2.f);
        Sqvvz.f = (Sqvvz.f + Stmp1.f);
        Sqvvx.f = (Sqvvx.f - Stmp3.f);
#endif
      }

      //###########################################################
      // Normalize quaternion for matrix V
      //###########################################################

      Stmp2.f = Sqvs.f * Sqvs.f;
      Stmp1.f = Sqvvx.f * Sqvvx.f;
      Stmp2.f = (Stmp1.f + Stmp2.f);
      Stmp1.f = Sqvvy.f * Sqvvy.f;
      Stmp2.f = (Stmp1.f + Stmp2.f);
      Stmp1.f = Sqvvz.f * Sqvvz.f;
      Stmp2.f = (Stmp1.f + Stmp2.f);

      Stmp1.f = zs::rsqrt(Stmp2.f);  //__frsqrt_rn(Stmp2.f);
      Stmp4.f = Stmp1.f * 0.5f;
      Stmp3.f = Stmp1.f * Stmp4.f;
      Stmp3.f = Stmp1.f * Stmp3.f;
      Stmp3.f = Stmp2.f * Stmp3.f;
      Stmp1.f = (Stmp1.f + Stmp4.f);
      Stmp1.f = (Stmp1.f - Stmp3.f);

      Sqvs.f = Sqvs.f * Stmp1.f;
      Sqvvx.f = Sqvvx.f * Stmp1.f;
      Sqvvy.f = Sqvvy.f * Stmp1.f;
      Sqvvz.f = Sqvvz.f * Stmp1.f;

      //###########################################################
      // Transform quaternion to matrix V
      //###########################################################

      Stmp1.f = Sqvvx.f * Sqvvx.f;
      Stmp2.f = Sqvvy.f * Sqvvy.f;
      Stmp3.f = Sqvvz.f * Sqvvz.f;
      Sv11.f = Sqvs.f * Sqvs.f;
      Sv22.f = (Sv11.f - Stmp1.f);
      Sv33.f = (Sv22.f - Stmp2.f);
      Sv33.f = (Sv33.f + Stmp3.f);
      Sv22.f = (Sv22.f + Stmp2.f);
      Sv22.f = (Sv22.f - Stmp3.f);
      Sv11.f = (Sv11.f + Stmp1.f);
      Sv11.f = (Sv11.f - Stmp2.f);
      Sv11.f = (Sv11.f - Stmp3.f);
      Stmp1.f = (Sqvvx.f + Sqvvx.f);
      Stmp2.f = (Sqvvy.f + Sqvvy.f);
      Stmp3.f = (Sqvvz.f + Sqvvz.f);
      Sv32.f = Sqvs.f * Stmp1.f;
      Sv13.f = Sqvs.f * Stmp2.f;
      Sv21.f = Sqvs.f * Stmp3.f;
      Stmp1.f = Sqvvy.f * Stmp1.f;
      Stmp2.f = Sqvvz.f * Stmp2.f;
      Stmp3.f = Sqvvx.f * Stmp3.f;
      Sv12.f = (Stmp1.f - Sv21.f);
      Sv23.f = (Stmp2.f - Sv32.f);
      Sv31.f = (Stmp3.f - Sv13.f);
      Sv21.f = (Stmp1.f + Sv21.f);
      Sv32.f = (Stmp2.f + Sv32.f);
      Sv13.f = (Stmp3.f + Sv13.f);

      ///###########################################################
      // Multiply (from the right) with V
      //###########################################################

      Stmp2.f = Sa12.f;
      Stmp3.f = Sa13.f;
      Sa12.f = Sv12.f * Sa11.f;
      Sa13.f = Sv13.f * Sa11.f;
      Sa11.f = Sv11.f * Sa11.f;
      Stmp1.f = Sv21.f * Stmp2.f;
      Sa11.f = (Sa11.f + Stmp1.f);
      Stmp1.f = Sv31.f * Stmp3.f;
      Sa11.f = (Sa11.f + Stmp1.f);
      Stmp1.f = Sv22.f * Stmp2.f;
      Sa12.f = (Sa12.f + Stmp1.f);
      Stmp1.f = Sv32.f * Stmp3.f;
      Sa12.f = (Sa12.f + Stmp1.f);
      Stmp1.f = Sv23.f * Stmp2.f;
      Sa13.f = (Sa13.f + Stmp1.f);
      Stmp1.f = Sv33.f * Stmp3.f;
      Sa13.f = (Sa13.f + Stmp1.f);

      Stmp2.f = Sa22.f;
      Stmp3.f = Sa23.f;
      Sa22.f = Sv12.f * Sa21.f;
      Sa23.f = Sv13.f * Sa21.f;
      Sa21.f = Sv11.f * Sa21.f;
      Stmp1.f = Sv21.f * Stmp2.f;
      Sa21.f = (Sa21.f + Stmp1.f);
      Stmp1.f = Sv31.f * Stmp3.f;
      Sa21.f = (Sa21.f + Stmp1.f);
      Stmp1.f = Sv22.f * Stmp2.f;
      Sa22.f = (Sa22.f + Stmp1.f);
      Stmp1.f = Sv32.f * Stmp3.f;
      Sa22.f = (Sa22.f + Stmp1.f);
      Stmp1.f = Sv23.f * Stmp2.f;
      Sa23.f = (Sa23.f + Stmp1.f);
      Stmp1.f = Sv33.f * Stmp3.f;
      Sa23.f = (Sa23.f + Stmp1.f);

      Stmp2.f = Sa32.f;
      Stmp3.f = Sa33.f;
      Sa32.f = Sv12.f * Sa31.f;
      Sa33.f = Sv13.f * Sa31.f;
      Sa31.f = Sv11.f * Sa31.f;
      Stmp1.f = Sv21.f * Stmp2.f;
      Sa31.f = (Sa31.f + Stmp1.f);
      Stmp1.f = Sv31.f * Stmp3.f;
      Sa31.f = (Sa31.f + Stmp1.f);
      Stmp1.f = Sv22.f * Stmp2.f;
      Sa32.f = (Sa32.f + Stmp1.f);
      Stmp1.f = Sv32.f * Stmp3.f;
      Sa32.f = (Sa32.f + Stmp1.f);
      Stmp1.f = Sv23.f * Stmp2.f;
      Sa33.f = (Sa33.f + Stmp1.f);
      Stmp1.f = Sv33.f * Stmp3.f;
      Sa33.f = (Sa33.f + Stmp1.f);

      //###########################################################
      // Permute columns such that the singular values are sorted
      //###########################################################

      Stmp1.f = Sa11.f * Sa11.f;
      Stmp4.f = Sa21.f * Sa21.f;
      Stmp1.f = (Stmp1.f + Stmp4.f);
      Stmp4.f = Sa31.f * Sa31.f;
      Stmp1.f = (Stmp1.f + Stmp4.f);

      Stmp2.f = Sa12.f * Sa12.f;
      Stmp4.f = Sa22.f * Sa22.f;
      Stmp2.f = (Stmp2.f + Stmp4.f);
      Stmp4.f = Sa32.f * Sa32.f;
      Stmp2.f = (Stmp2.f + Stmp4.f);

      Stmp3.f = Sa13.f * Sa13.f;
      Stmp4.f = Sa23.f * Sa23.f;
      Stmp3.f = (Stmp3.f + Stmp4.f);
      Stmp4.f = Sa33.f * Sa33.f;
      Stmp3.f = (Stmp3.f + Stmp4.f);

      // Swap columns 1-2 if necessary

      Stmp4.ui = (Stmp1.f < Stmp2.f) ? 0xffffffff : 0;
      Stmp5.ui = Sa11.ui ^ Sa12.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sa11.ui = Sa11.ui ^ Stmp5.ui;
      Sa12.ui = Sa12.ui ^ Stmp5.ui;

      Stmp5.ui = Sa21.ui ^ Sa22.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sa21.ui = Sa21.ui ^ Stmp5.ui;
      Sa22.ui = Sa22.ui ^ Stmp5.ui;

      Stmp5.ui = Sa31.ui ^ Sa32.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sa31.ui = Sa31.ui ^ Stmp5.ui;
      Sa32.ui = Sa32.ui ^ Stmp5.ui;

      Stmp5.ui = Sv11.ui ^ Sv12.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sv11.ui = Sv11.ui ^ Stmp5.ui;
      Sv12.ui = Sv12.ui ^ Stmp5.ui;

      Stmp5.ui = Sv21.ui ^ Sv22.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sv21.ui = Sv21.ui ^ Stmp5.ui;
      Sv22.ui = Sv22.ui ^ Stmp5.ui;

      Stmp5.ui = Sv31.ui ^ Sv32.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sv31.ui = Sv31.ui ^ Stmp5.ui;
      Sv32.ui = Sv32.ui ^ Stmp5.ui;

      Stmp5.ui = Stmp1.ui ^ Stmp2.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Stmp1.ui = Stmp1.ui ^ Stmp5.ui;
      Stmp2.ui = Stmp2.ui ^ Stmp5.ui;

      // If columns 1-2 have been swapped, negate 2nd column of A and V so that V is still a
      // rotation

      Stmp5.f = -2.f;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Stmp4.f = 1.f;
      Stmp4.f = (Stmp4.f + Stmp5.f);

      Sa12.f = Sa12.f * Stmp4.f;
      Sa22.f = Sa22.f * Stmp4.f;
      Sa32.f = Sa32.f * Stmp4.f;

      Sv12.f = Sv12.f * Stmp4.f;
      Sv22.f = Sv22.f * Stmp4.f;
      Sv32.f = Sv32.f * Stmp4.f;

      // Swap columns 1-3 if necessary

      Stmp4.ui = (Stmp1.f < Stmp3.f) ? 0xffffffff : 0;
      Stmp5.ui = Sa11.ui ^ Sa13.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sa11.ui = Sa11.ui ^ Stmp5.ui;
      Sa13.ui = Sa13.ui ^ Stmp5.ui;

      Stmp5.ui = Sa21.ui ^ Sa23.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sa21.ui = Sa21.ui ^ Stmp5.ui;
      Sa23.ui = Sa23.ui ^ Stmp5.ui;

      Stmp5.ui = Sa31.ui ^ Sa33.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sa31.ui = Sa31.ui ^ Stmp5.ui;
      Sa33.ui = Sa33.ui ^ Stmp5.ui;

      Stmp5.ui = Sv11.ui ^ Sv13.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sv11.ui = Sv11.ui ^ Stmp5.ui;
      Sv13.ui = Sv13.ui ^ Stmp5.ui;

      Stmp5.ui = Sv21.ui ^ Sv23.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sv21.ui = Sv21.ui ^ Stmp5.ui;
      Sv23.ui = Sv23.ui ^ Stmp5.ui;

      Stmp5.ui = Sv31.ui ^ Sv33.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sv31.ui = Sv31.ui ^ Stmp5.ui;
      Sv33.ui = Sv33.ui ^ Stmp5.ui;

      Stmp5.ui = Stmp1.ui ^ Stmp3.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Stmp1.ui = Stmp1.ui ^ Stmp5.ui;
      Stmp3.ui = Stmp3.ui ^ Stmp5.ui;

      // If columns 1-3 have been swapped, negate 1st column of A and V so that V is still a
      // rotation

      Stmp5.f = -2.f;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Stmp4.f = 1.f;
      Stmp4.f = (Stmp4.f + Stmp5.f);

      Sa11.f = Sa11.f * Stmp4.f;
      Sa21.f = Sa21.f * Stmp4.f;
      Sa31.f = Sa31.f * Stmp4.f;

      Sv11.f = Sv11.f * Stmp4.f;
      Sv21.f = Sv21.f * Stmp4.f;
      Sv31.f = Sv31.f * Stmp4.f;

      // Swap columns 2-3 if necessary

      Stmp4.ui = (Stmp2.f < Stmp3.f) ? 0xffffffff : 0;
      Stmp5.ui = Sa12.ui ^ Sa13.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sa12.ui = Sa12.ui ^ Stmp5.ui;
      Sa13.ui = Sa13.ui ^ Stmp5.ui;

      Stmp5.ui = Sa22.ui ^ Sa23.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sa22.ui = Sa22.ui ^ Stmp5.ui;
      Sa23.ui = Sa23.ui ^ Stmp5.ui;

      Stmp5.ui = Sa32.ui ^ Sa33.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sa32.ui = Sa32.ui ^ Stmp5.ui;
      Sa33.ui = Sa33.ui ^ Stmp5.ui;

      Stmp5.ui = Sv12.ui ^ Sv13.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sv12.ui = Sv12.ui ^ Stmp5.ui;
      Sv13.ui = Sv13.ui ^ Stmp5.ui;

      Stmp5.ui = Sv22.ui ^ Sv23.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sv22.ui = Sv22.ui ^ Stmp5.ui;
      Sv23.ui = Sv23.ui ^ Stmp5.ui;

      Stmp5.ui = Sv32.ui ^ Sv33.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Sv32.ui = Sv32.ui ^ Stmp5.ui;
      Sv33.ui = Sv33.ui ^ Stmp5.ui;

      Stmp5.ui = Stmp2.ui ^ Stmp3.ui;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Stmp2.ui = Stmp2.ui ^ Stmp5.ui;
      Stmp3.ui = Stmp3.ui ^ Stmp5.ui;

      // If columns 2-3 have been swapped, negate 3rd column of A and V so that V is still a
      // rotation

      Stmp5.f = -2.f;
      Stmp5.ui = Stmp5.ui & Stmp4.ui;
      Stmp4.f = 1.f;
      Stmp4.f = (Stmp4.f + Stmp5.f);

      Sa13.f = Sa13.f * Stmp4.f;
      Sa23.f = Sa23.f * Stmp4.f;
      Sa33.f = Sa33.f * Stmp4.f;

      Sv13.f = Sv13.f * Stmp4.f;
      Sv23.f = Sv23.f * Stmp4.f;
      Sv33.f = Sv33.f * Stmp4.f;

      //###########################################################
      // Construct QR factorization of A*V (=U*D) using Givens rotations
      //###########################################################

      Su11.f = 1.f;
      Su12.f = 0.f;
      Su13.f = 0.f;
      Su21.f = 0.f;
      Su22.f = 1.f;
      Su23.f = 0.f;
      Su31.f = 0.f;
      Su32.f = 0.f;
      Su33.f = 1.f;

      Ssh.f = Sa21.f * Sa21.f;
      Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
      Ssh.ui = Ssh.ui & Sa21.ui;

      Stmp5.f = 0.f;
      Sch.f = (Stmp5.f - Sa11.f);
      Sch.f = math::max(Sch.f, Sa11.f);
      Sch.f = math::max(Sch.f, gsmall_number);
      Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

      Stmp1.f = Sch.f * Sch.f;
      Stmp2.f = Ssh.f * Ssh.f;
      Stmp2.f = (Stmp1.f + Stmp2.f);
      Stmp1.f = zs::rsqrt(Stmp2.f);  //__frsqrt_rn(Stmp2.f);

      Stmp4.f = Stmp1.f * 0.5f;
      Stmp3.f = Stmp1.f * Stmp4.f;
      Stmp3.f = Stmp1.f * Stmp3.f;
      Stmp3.f = Stmp2.f * Stmp3.f;
      Stmp1.f = (Stmp1.f + Stmp4.f);
      Stmp1.f = (Stmp1.f - Stmp3.f);
      Stmp1.f = Stmp1.f * Stmp2.f;

      Sch.f = (Sch.f + Stmp1.f);

      Stmp1.ui = ~Stmp5.ui & Ssh.ui;
      Stmp2.ui = ~Stmp5.ui & Sch.ui;
      Sch.ui = Stmp5.ui & Sch.ui;
      Ssh.ui = Stmp5.ui & Ssh.ui;
      Sch.ui = Sch.ui | Stmp1.ui;
      Ssh.ui = Ssh.ui | Stmp2.ui;

      Stmp1.f = Sch.f * Sch.f;
      Stmp2.f = Ssh.f * Ssh.f;
      Stmp2.f = (Stmp1.f + Stmp2.f);
      Stmp1.f = zs::rsqrt(Stmp2.f);  //__frsqrt_rn(Stmp2.f);

      Stmp4.f = Stmp1.f * 0.5f;
      Stmp3.f = Stmp1.f * Stmp4.f;
      Stmp3.f = Stmp1.f * Stmp3.f;
      Stmp3.f = Stmp2.f * Stmp3.f;
      Stmp1.f = (Stmp1.f + Stmp4.f);
      Stmp1.f = (Stmp1.f - Stmp3.f);

      Sch.f = Sch.f * Stmp1.f;
      Ssh.f = Ssh.f * Stmp1.f;

      Sc.f = Sch.f * Sch.f;
      Ss.f = Ssh.f * Ssh.f;
      Sc.f = (Sc.f - Ss.f);
      Ss.f = Ssh.f * Sch.f;
      Ss.f = (Ss.f + Ss.f);

      //###########################################################
      // Rotate matrix A
      //###########################################################

      Stmp1.f = Ss.f * Sa11.f;
      Stmp2.f = Ss.f * Sa21.f;
      Sa11.f = Sc.f * Sa11.f;
      Sa21.f = Sc.f * Sa21.f;
      Sa11.f = (Sa11.f + Stmp2.f);
      Sa21.f = (Sa21.f - Stmp1.f);

      Stmp1.f = Ss.f * Sa12.f;
      Stmp2.f = Ss.f * Sa22.f;
      Sa12.f = Sc.f * Sa12.f;
      Sa22.f = Sc.f * Sa22.f;
      Sa12.f = (Sa12.f + Stmp2.f);
      Sa22.f = (Sa22.f - Stmp1.f);

      Stmp1.f = Ss.f * Sa13.f;
      Stmp2.f = Ss.f * Sa23.f;
      Sa13.f = Sc.f * Sa13.f;
      Sa23.f = Sc.f * Sa23.f;
      Sa13.f = (Sa13.f + Stmp2.f);
      Sa23.f = (Sa23.f - Stmp1.f);

      //###########################################################
      // Update matrix U
      //###########################################################

      Stmp1.f = Ss.f * Su11.f;
      Stmp2.f = Ss.f * Su12.f;
      Su11.f = Sc.f * Su11.f;
      Su12.f = Sc.f * Su12.f;
      Su11.f = (Su11.f + Stmp2.f);
      Su12.f = (Su12.f - Stmp1.f);

      Stmp1.f = Ss.f * Su21.f;
      Stmp2.f = Ss.f * Su22.f;
      Su21.f = Sc.f * Su21.f;
      Su22.f = Sc.f * Su22.f;
      Su21.f = (Su21.f + Stmp2.f);
      Su22.f = (Su22.f - Stmp1.f);

      Stmp1.f = Ss.f * Su31.f;
      Stmp2.f = Ss.f * Su32.f;
      Su31.f = Sc.f * Su31.f;
      Su32.f = Sc.f * Su32.f;
      Su31.f = (Su31.f + Stmp2.f);
      Su32.f = (Su32.f - Stmp1.f);

      // Second Givens rotation

      Ssh.f = Sa31.f * Sa31.f;
      Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
      Ssh.ui = Ssh.ui & Sa31.ui;

      Stmp5.f = 0.f;
      Sch.f = (Stmp5.f - Sa11.f);
      Sch.f = math::max(Sch.f, Sa11.f);
      Sch.f = math::max(Sch.f, gsmall_number);
      Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

      Stmp1.f = Sch.f * Sch.f;
      Stmp2.f = Ssh.f * Ssh.f;
      Stmp2.f = (Stmp1.f + Stmp2.f);
      Stmp1.f = zs::rsqrt(Stmp2.f);  //__frsqrt_rn(Stmp2.f);

      Stmp4.f = Stmp1.f * 0.5;
      Stmp3.f = Stmp1.f * Stmp4.f;
      Stmp3.f = Stmp1.f * Stmp3.f;
      Stmp3.f = Stmp2.f * Stmp3.f;
      Stmp1.f = (Stmp1.f + Stmp4.f);
      Stmp1.f = (Stmp1.f - Stmp3.f);
      Stmp1.f = Stmp1.f * Stmp2.f;

      Sch.f = (Sch.f + Stmp1.f);

      Stmp1.ui = ~Stmp5.ui & Ssh.ui;
      Stmp2.ui = ~Stmp5.ui & Sch.ui;
      Sch.ui = Stmp5.ui & Sch.ui;
      Ssh.ui = Stmp5.ui & Ssh.ui;
      Sch.ui = Sch.ui | Stmp1.ui;
      Ssh.ui = Ssh.ui | Stmp2.ui;

      Stmp1.f = Sch.f * Sch.f;
      Stmp2.f = Ssh.f * Ssh.f;
      Stmp2.f = (Stmp1.f + Stmp2.f);
      Stmp1.f = zs::rsqrt(Stmp2.f);  //__frsqrt_rn(Stmp2.f);

      Stmp4.f = Stmp1.f * 0.5f;
      Stmp3.f = Stmp1.f * Stmp4.f;
      Stmp3.f = Stmp1.f * Stmp3.f;
      Stmp3.f = Stmp2.f * Stmp3.f;
      Stmp1.f = (Stmp1.f + Stmp4.f);
      Stmp1.f = (Stmp1.f - Stmp3.f);

      Sch.f = Sch.f * Stmp1.f;
      Ssh.f = Ssh.f * Stmp1.f;

      Sc.f = Sch.f * Sch.f;
      Ss.f = Ssh.f * Ssh.f;
      Sc.f = (Sc.f - Ss.f);
      Ss.f = Ssh.f * Sch.f;
      Ss.f = (Ss.f + Ss.f);

      //###########################################################
      // Rotate matrix A
      //###########################################################

      Stmp1.f = Ss.f * Sa11.f;
      Stmp2.f = Ss.f * Sa31.f;
      Sa11.f = Sc.f * Sa11.f;
      Sa31.f = Sc.f * Sa31.f;
      Sa11.f = (Sa11.f + Stmp2.f);
      Sa31.f = (Sa31.f - Stmp1.f);

      Stmp1.f = Ss.f * Sa12.f;
      Stmp2.f = Ss.f * Sa32.f;
      Sa12.f = Sc.f * Sa12.f;
      Sa32.f = Sc.f * Sa32.f;
      Sa12.f = (Sa12.f + Stmp2.f);
      Sa32.f = (Sa32.f - Stmp1.f);

      Stmp1.f = Ss.f * Sa13.f;
      Stmp2.f = Ss.f * Sa33.f;
      Sa13.f = Sc.f * Sa13.f;
      Sa33.f = Sc.f * Sa33.f;
      Sa13.f = (Sa13.f + Stmp2.f);
      Sa33.f = (Sa33.f - Stmp1.f);

      //###########################################################
      // Update matrix U
      //###########################################################

      Stmp1.f = Ss.f * Su11.f;
      Stmp2.f = Ss.f * Su13.f;
      Su11.f = Sc.f * Su11.f;
      Su13.f = Sc.f * Su13.f;
      Su11.f = (Su11.f + Stmp2.f);
      Su13.f = (Su13.f - Stmp1.f);

      Stmp1.f = Ss.f * Su21.f;
      Stmp2.f = Ss.f * Su23.f;
      Su21.f = Sc.f * Su21.f;
      Su23.f = Sc.f * Su23.f;
      Su21.f = (Su21.f + Stmp2.f);
      Su23.f = (Su23.f - Stmp1.f);

      Stmp1.f = Ss.f * Su31.f;
      Stmp2.f = Ss.f * Su33.f;
      Su31.f = Sc.f * Su31.f;
      Su33.f = Sc.f * Su33.f;
      Su31.f = (Su31.f + Stmp2.f);
      Su33.f = (Su33.f - Stmp1.f);

      // Third Givens Rotation

      Ssh.f = Sa32.f * Sa32.f;
      Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
      Ssh.ui = Ssh.ui & Sa32.ui;

      Stmp5.f = 0.f;
      Sch.f = (Stmp5.f - Sa22.f);
      Sch.f = math::max(Sch.f, Sa22.f);
      Sch.f = math::max(Sch.f, gsmall_number);
      Stmp5.ui = (Sa22.f >= Stmp5.f) ? 0xffffffff : 0;

      Stmp1.f = Sch.f * Sch.f;
      Stmp2.f = Ssh.f * Ssh.f;
      Stmp2.f = (Stmp1.f + Stmp2.f);
      Stmp1.f = zs::rsqrt(Stmp2.f);  //__frsqrt_rn(Stmp2.f);

      Stmp4.f = Stmp1.f * 0.5f;
      Stmp3.f = Stmp1.f * Stmp4.f;
      Stmp3.f = Stmp1.f * Stmp3.f;
      Stmp3.f = Stmp2.f * Stmp3.f;
      Stmp1.f = (Stmp1.f + Stmp4.f);
      Stmp1.f = (Stmp1.f - Stmp3.f);
      Stmp1.f = Stmp1.f * Stmp2.f;

      Sch.f = (Sch.f + Stmp1.f);

      Stmp1.ui = ~Stmp5.ui & Ssh.ui;
      Stmp2.ui = ~Stmp5.ui & Sch.ui;
      Sch.ui = Stmp5.ui & Sch.ui;
      Ssh.ui = Stmp5.ui & Ssh.ui;
      Sch.ui = Sch.ui | Stmp1.ui;
      Ssh.ui = Ssh.ui | Stmp2.ui;

      Stmp1.f = Sch.f * Sch.f;
      Stmp2.f = Ssh.f * Ssh.f;
      Stmp2.f = (Stmp1.f + Stmp2.f);
      Stmp1.f = zs::rsqrt(Stmp2.f);  //__frsqrt_rn(Stmp2.f);

      Stmp4.f = Stmp1.f * 0.5f;
      Stmp3.f = Stmp1.f * Stmp4.f;
      Stmp3.f = Stmp1.f * Stmp3.f;
      Stmp3.f = Stmp2.f * Stmp3.f;
      Stmp1.f = (Stmp1.f + Stmp4.f);
      Stmp1.f = (Stmp1.f - Stmp3.f);

      Sch.f = Sch.f * Stmp1.f;
      Ssh.f = Ssh.f * Stmp1.f;

      Sc.f = Sch.f * Sch.f;
      Ss.f = Ssh.f * Ssh.f;
      Sc.f = (Sc.f - Ss.f);
      Ss.f = Ssh.f * Sch.f;
      Ss.f = (Ss.f + Ss.f);

      //###########################################################
      // Rotate matrix A
      //###########################################################

      Stmp1.f = Ss.f * Sa21.f;
      Stmp2.f = Ss.f * Sa31.f;
      Sa21.f = Sc.f * Sa21.f;
      Sa31.f = Sc.f * Sa31.f;
      Sa21.f = (Sa21.f + Stmp2.f);
      Sa31.f = (Sa31.f - Stmp1.f);

      Stmp1.f = Ss.f * Sa22.f;
      Stmp2.f = Ss.f * Sa32.f;
      Sa22.f = Sc.f * Sa22.f;
      Sa32.f = Sc.f * Sa32.f;
      Sa22.f = (Sa22.f + Stmp2.f);
      Sa32.f = (Sa32.f - Stmp1.f);

      Stmp1.f = Ss.f * Sa23.f;
      Stmp2.f = Ss.f * Sa33.f;
      Sa23.f = Sc.f * Sa23.f;
      Sa33.f = Sc.f * Sa33.f;
      Sa23.f = (Sa23.f + Stmp2.f);
      Sa33.f = (Sa33.f - Stmp1.f);

      //###########################################################
      // Update matrix U
      //###########################################################

      Stmp1.f = Ss.f * Su12.f;
      Stmp2.f = Ss.f * Su13.f;
      Su12.f = Sc.f * Su12.f;
      Su13.f = Sc.f * Su13.f;
      Su12.f = (Su12.f + Stmp2.f);
      Su13.f = (Su13.f - Stmp1.f);

      Stmp1.f = Ss.f * Su22.f;
      Stmp2.f = Ss.f * Su23.f;
      Su22.f = Sc.f * Su22.f;
      Su23.f = Sc.f * Su23.f;
      Su22.f = (Su22.f + Stmp2.f);
      Su23.f = (Su23.f - Stmp1.f);

      Stmp1.f = Ss.f * Su32.f;
      Stmp2.f = Ss.f * Su33.f;
      Su32.f = Sc.f * Su32.f;
      Su33.f = Sc.f * Su33.f;
      Su32.f = (Su32.f + Stmp2.f);
      Su33.f = (Su33.f - Stmp1.f);

      v11 = Sv11.f;
      v12 = Sv12.f;
      v13 = Sv13.f;
      v21 = Sv21.f;
      v22 = Sv22.f;
      v23 = Sv23.f;
      v31 = Sv31.f;
      v32 = Sv32.f;
      v33 = Sv33.f;

      u11 = Su11.f;
      u12 = Su12.f;
      u13 = Su13.f;
      u21 = Su21.f;
      u22 = Su22.f;
      u23 = Su23.f;
      u31 = Su31.f;
      u32 = Su32.f;
      u33 = Su33.f;

      s11 = Sa11.f;
      // s12 = Sa12.f; s13 = Sa13.f; s21 = Sa21.f;
      s22 = Sa22.f;
      // s23 = Sa23.f; s31 = Sa31.f; s32 = Sa32.f;
      s33 = Sa33.f;
    }

    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value <= 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto svd(const VecInterface<VecT>& F) noexcept {
      // F = U S V^T
      typename VecT::template variant_vec<typename VecT::value_type, typename VecT::extents> U{},
          V{};
      typename VecT::template variant_vec<
          typename VecT::value_type,
          integer_sequence<typename VecT::index_type, VecT::template range_t<0>::value>>
          S{};
      if constexpr (is_same_v<typename VecT::dims, index_sequence<3, 3>>) {
        if constexpr (is_same_v<typename VecT::value_type, float>) {
          svd_3d(F(0, 0), F(0, 1), F(0, 2), F(1, 0), F(1, 1), F(1, 2), F(2, 0), F(2, 1), F(2, 2),
                 U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2),
                 S(0), S(1), S(2), V(0, 0), V(0, 1), V(0, 2), V(1, 0), V(1, 1), V(1, 2), V(2, 0),
                 V(2, 1), V(2, 2));
          return zs::make_tuple(U, S, V);
        } else
          return qr_svd(F);
      } else if constexpr (is_same_v<typename VecT::dims, index_sequence<2, 2>>) {
        return qr_svd(F);
      } else if constexpr (is_same_v<typename VecT::dims, index_sequence<1, 1>>) {
        U(0, 0) = (typename VecT::value_type)1;
        V(0, 0) = (typename VecT::value_type)1;
        S(0) = F(0, 0);
        return zs::make_tuple(U, S, V);
      } else {
      }
    }

  }  // namespace math

}  // namespace zs