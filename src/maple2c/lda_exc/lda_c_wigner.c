/*
  This file was generated automatically with scripts/maple2c.py.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2022 (X86 64 LINUX)
  Maple source      : ./maple/lda_exc/lda_c_wigner.mpl
  Type of functional: lda_exc
*/

#define maple2c_order 4
#define MAPLE2C_FLAGS (XC_FLAGS_I_HAVE_EXC | XC_FLAGS_I_HAVE_VXC | XC_FLAGS_I_HAVE_FXC | XC_FLAGS_I_HAVE_KXC | XC_FLAGS_I_HAVE_LXC)


#ifndef XC_DONT_COMPILE_EXC
GPU_DEVICE_FUNCTION static inline void
func_exc_unpol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t6, t7, t8;
  double t12, tzk0;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = M_CBRT3;
  t2 = 0.1e1 / M_PI;
  t3 = POW_1_3(t2);
  t4 = t1 * t3;
  t5 = M_CBRT4;
  t6 = t5 * t5;
  t7 = POW_1_3(rho[0]);
  t8 = 0.1e1 / t7;
  t12 = params->b + t4 * t6 * t8 / 0.4e1;
  tzk0 = params->a / t12;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

}

#endif


#ifndef XC_DONT_COMPILE_VXC
GPU_DEVICE_FUNCTION static inline void
func_vxc_unpol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t6, t7, t8;
  double t12, tzk0;

  double t15, t16, tvrho0;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = M_CBRT3;
  t2 = 0.1e1 / M_PI;
  t3 = POW_1_3(t2);
  t4 = t1 * t3;
  t5 = M_CBRT4;
  t6 = t5 * t5;
  t7 = POW_1_3(rho[0]);
  t8 = 0.1e1 / t7;
  t12 = params->b + t4 * t6 * t8 / 0.4e1;
  tzk0 = params->a / t12;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

  t15 = t12 * t12;
  t16 = 0.1e1 / t15;
  tvrho0 = tzk0 + t8 * params->a * t16 * t4 * t6 / 0.12e2;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 0] += tvrho0;

}

#endif


#ifndef XC_DONT_COMPILE_FXC
GPU_DEVICE_FUNCTION static inline void
func_fxc_unpol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t6, t7, t8;
  double t12, tzk0;

  double t15, t16, tvrho0;

  double t22, t23, t28, t33, t35, t36, tv2rho20;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = M_CBRT3;
  t2 = 0.1e1 / M_PI;
  t3 = POW_1_3(t2);
  t4 = t1 * t3;
  t5 = M_CBRT4;
  t6 = t5 * t5;
  t7 = POW_1_3(rho[0]);
  t8 = 0.1e1 / t7;
  t12 = params->b + t4 * t6 * t8 / 0.4e1;
  tzk0 = params->a / t12;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

  t15 = t12 * t12;
  t16 = 0.1e1 / t15;
  tvrho0 = tzk0 + t8 * params->a * t16 * t4 * t6 / 0.12e2;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 0] += tvrho0;

  t22 = params->a * t16 * t1;
  t23 = t3 * t6;
  t28 = t7 * t7;
  t33 = 0.1e1 / t15 / t12;
  t35 = t1 * t1;
  t36 = t3 * t3;
  tv2rho20 = t22 * t23 / t7 / rho[0] / 0.18e2 + 0.1e1 / t28 / rho[0] * params->a * t33 * t35 * t36 * t5 / 0.18e2;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 0] += tv2rho20;

}

#endif


#ifndef XC_DONT_COMPILE_KXC
GPU_DEVICE_FUNCTION static inline void
func_kxc_unpol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t6, t7, t8;
  double t12, tzk0;

  double t15, t16, tvrho0;

  double t22, t23, t28, t33, t35, t36, tv2rho20;

  double t42, t43, t44, t55, t58, t59, tv3rho30;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = M_CBRT3;
  t2 = 0.1e1 / M_PI;
  t3 = POW_1_3(t2);
  t4 = t1 * t3;
  t5 = M_CBRT4;
  t6 = t5 * t5;
  t7 = POW_1_3(rho[0]);
  t8 = 0.1e1 / t7;
  t12 = params->b + t4 * t6 * t8 / 0.4e1;
  tzk0 = params->a / t12;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

  t15 = t12 * t12;
  t16 = 0.1e1 / t15;
  tvrho0 = tzk0 + t8 * params->a * t16 * t4 * t6 / 0.12e2;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 0] += tvrho0;

  t22 = params->a * t16 * t1;
  t23 = t3 * t6;
  t28 = t7 * t7;
  t33 = 0.1e1 / t15 / t12;
  t35 = t1 * t1;
  t36 = t3 * t3;
  tv2rho20 = t22 * t23 / t7 / rho[0] / 0.18e2 + 0.1e1 / t28 / rho[0] * params->a * t33 * t35 * t36 * t5 / 0.18e2;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 0] += tv2rho20;

  t42 = params->a * t33 * t35;
  t43 = t36 * t5;
  t44 = rho[0] * rho[0];
  t55 = t44 * rho[0];
  t58 = t15 * t15;
  t59 = 0.1e1 / t58;
  tv3rho30 = -t42 * t43 / t28 / t44 / 0.18e2 - 0.2e1 / 0.27e2 * t22 * t23 / t7 / t44 + 0.1e1 / t55 * params->a * t59 * t2 / 0.6e1;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 0] += tv3rho30;

}

#endif


#ifndef XC_DONT_COMPILE_LXC
GPU_DEVICE_FUNCTION static inline void
func_lxc_unpol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t6, t7, t8;
  double t12, tzk0;

  double t15, t16, tvrho0;

  double t22, t23, t28, t33, t35, t36, tv2rho20;

  double t42, t43, t44, t55, t58, t59, tv3rho30;

  double t64, tv4rho40;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = M_CBRT3;
  t2 = 0.1e1 / M_PI;
  t3 = POW_1_3(t2);
  t4 = t1 * t3;
  t5 = M_CBRT4;
  t6 = t5 * t5;
  t7 = POW_1_3(rho[0]);
  t8 = 0.1e1 / t7;
  t12 = params->b + t4 * t6 * t8 / 0.4e1;
  tzk0 = params->a / t12;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

  t15 = t12 * t12;
  t16 = 0.1e1 / t15;
  tvrho0 = tzk0 + t8 * params->a * t16 * t4 * t6 / 0.12e2;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 0] += tvrho0;

  t22 = params->a * t16 * t1;
  t23 = t3 * t6;
  t28 = t7 * t7;
  t33 = 0.1e1 / t15 / t12;
  t35 = t1 * t1;
  t36 = t3 * t3;
  tv2rho20 = t22 * t23 / t7 / rho[0] / 0.18e2 + 0.1e1 / t28 / rho[0] * params->a * t33 * t35 * t36 * t5 / 0.18e2;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 0] += tv2rho20;

  t42 = params->a * t33 * t35;
  t43 = t36 * t5;
  t44 = rho[0] * rho[0];
  t55 = t44 * rho[0];
  t58 = t15 * t15;
  t59 = 0.1e1 / t58;
  tv3rho30 = -t42 * t43 / t28 / t44 / 0.18e2 - 0.2e1 / 0.27e2 * t22 * t23 / t7 / t44 + 0.1e1 / t55 * params->a * t59 * t2 / 0.6e1;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 0] += tv3rho30;

  t64 = t44 * t44;
  tv4rho40 = -0.2e1 / 0.3e1 * params->a * t59 * t2 / t64 + 0.8e1 / 0.81e2 * t42 * t43 / t28 / t55 + 0.14e2 / 0.81e2 * t22 * t23 / t7 / t55 + 0.1e1 / t7 / t64 * params->a / t58 / t12 * t2 * t1 * t23 / 0.18e2;

  if(out->v4rho4 != NULL && (p->info->flags & XC_FLAGS_HAVE_LXC))
    out->v4rho4[ip*p->dim.v4rho4 + 0] += tv4rho40;

}

#endif


#ifndef XC_DONT_COMPILE_EXC
GPU_DEVICE_FUNCTION static inline void
func_exc_pol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t7, t8, t9;
  double t10, t11, t12, t13, t14, t15, t16, t20;
  double t21, tzk0;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = rho[0] - rho[1];
  t2 = t1 * t1;
  t3 = rho[0] + rho[1];
  t4 = t3 * t3;
  t5 = 0.1e1 / t4;
  t7 = -t2 * t5 + 0.1e1;
  t8 = t7 * params->a;
  t9 = M_CBRT3;
  t10 = 0.1e1 / M_PI;
  t11 = POW_1_3(t10);
  t12 = t9 * t11;
  t13 = M_CBRT4;
  t14 = t13 * t13;
  t15 = POW_1_3(t3);
  t16 = 0.1e1 / t15;
  t20 = params->b + t12 * t14 * t16 / 0.4e1;
  t21 = 0.1e1 / t20;
  tzk0 = t8 * t21;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

}

#endif


#ifndef XC_DONT_COMPILE_VXC
GPU_DEVICE_FUNCTION static inline void
func_vxc_pol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t7, t8, t9;
  double t10, t11, t12, t13, t14, t15, t16, t20;
  double t21, tzk0;

  double t22, t23, t24, t25, t27, t29, t33, t34;
  double t36, t37, t39, tvrho0, t41, tvrho1;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = rho[0] - rho[1];
  t2 = t1 * t1;
  t3 = rho[0] + rho[1];
  t4 = t3 * t3;
  t5 = 0.1e1 / t4;
  t7 = -t2 * t5 + 0.1e1;
  t8 = t7 * params->a;
  t9 = M_CBRT3;
  t10 = 0.1e1 / M_PI;
  t11 = POW_1_3(t10);
  t12 = t9 * t11;
  t13 = M_CBRT4;
  t14 = t13 * t13;
  t15 = POW_1_3(t3);
  t16 = 0.1e1 / t15;
  t20 = params->b + t12 * t14 * t16 / 0.4e1;
  t21 = 0.1e1 / t20;
  tzk0 = t8 * t21;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

  t22 = t1 * t5;
  t23 = t4 * t3;
  t24 = 0.1e1 / t23;
  t25 = t2 * t24;
  t27 = -0.2e1 * t22 + 0.2e1 * t25;
  t29 = params->a * t21;
  t33 = t20 * t20;
  t34 = 0.1e1 / t33;
  t36 = t11 * t14;
  t37 = t34 * t9 * t36;
  t39 = t16 * t7 * params->a * t37 / 0.12e2;
  tvrho0 = t3 * t27 * t29 + t39 + tzk0;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 0] += tvrho0;

  t41 = 0.2e1 * t22 + 0.2e1 * t25;
  tvrho1 = t3 * t41 * t29 + t39 + tzk0;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 1] += tvrho1;

}

#endif


#ifndef XC_DONT_COMPILE_FXC
GPU_DEVICE_FUNCTION static inline void
func_fxc_pol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t7, t8, t9;
  double t10, t11, t12, t13, t14, t15, t16, t20;
  double t21, tzk0;

  double t22, t23, t24, t25, t27, t29, t33, t34;
  double t36, t37, t39, tvrho0, t41, tvrho1;

  double t44, t45, t47, t51, t53, t54, t56, t57;
  double t58, t60, t61, t66, t68, t70, t74, t75;
  double t77, t79, t81, tv2rho20, t82, t83, t84, t89;
  double tv2rho21, t93, tv2rho22;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = rho[0] - rho[1];
  t2 = t1 * t1;
  t3 = rho[0] + rho[1];
  t4 = t3 * t3;
  t5 = 0.1e1 / t4;
  t7 = -t2 * t5 + 0.1e1;
  t8 = t7 * params->a;
  t9 = M_CBRT3;
  t10 = 0.1e1 / M_PI;
  t11 = POW_1_3(t10);
  t12 = t9 * t11;
  t13 = M_CBRT4;
  t14 = t13 * t13;
  t15 = POW_1_3(t3);
  t16 = 0.1e1 / t15;
  t20 = params->b + t12 * t14 * t16 / 0.4e1;
  t21 = 0.1e1 / t20;
  tzk0 = t8 * t21;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

  t22 = t1 * t5;
  t23 = t4 * t3;
  t24 = 0.1e1 / t23;
  t25 = t2 * t24;
  t27 = -0.2e1 * t22 + 0.2e1 * t25;
  t29 = params->a * t21;
  t33 = t20 * t20;
  t34 = 0.1e1 / t33;
  t36 = t11 * t14;
  t37 = t34 * t9 * t36;
  t39 = t16 * t7 * params->a * t37 / 0.12e2;
  tvrho0 = t3 * t27 * t29 + t39 + tzk0;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 0] += tvrho0;

  t41 = 0.2e1 * t22 + 0.2e1 * t25;
  tvrho1 = t3 * t41 * t29 + t39 + tzk0;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 1] += tvrho1;

  t44 = t27 * params->a;
  t45 = t44 * t21;
  t47 = t8 * t34;
  t51 = t12 * t14 / t15 / t3;
  t53 = t47 * t51 / 0.18e2;
  t54 = 0.2e1 * t5;
  t56 = 0.8e1 * t1 * t24;
  t57 = t4 * t4;
  t58 = 0.1e1 / t57;
  t60 = 0.6e1 * t2 * t58;
  t61 = -t54 + t56 - t60;
  t66 = t16 * t27 * params->a * t37;
  t68 = t15 * t15;
  t70 = 0.1e1 / t68 / t3;
  t74 = 0.1e1 / t33 / t20;
  t75 = t9 * t9;
  t77 = t11 * t11;
  t79 = t74 * t75 * t77 * t13;
  t81 = t70 * t7 * params->a * t79 / 0.18e2;
  tv2rho20 = 0.2e1 * t45 + t53 + t3 * t61 * t29 + t66 / 0.6e1 + t81;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 0] += tv2rho20;

  t82 = t41 * params->a;
  t83 = t82 * t21;
  t84 = t54 - t60;
  t89 = t16 * t41 * params->a * t37;
  tv2rho21 = t45 + t53 + t83 + t3 * t84 * t29 + t89 / 0.12e2 + t66 / 0.12e2 + t81;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 1] += tv2rho21;

  t93 = -t54 - t56 - t60;
  tv2rho22 = 0.2e1 * t83 + t53 + t3 * t93 * t29 + t89 / 0.6e1 + t81;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 2] += tv2rho22;

}

#endif


#ifndef XC_DONT_COMPILE_KXC
GPU_DEVICE_FUNCTION static inline void
func_kxc_pol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t7, t8, t9;
  double t10, t11, t12, t13, t14, t15, t16, t20;
  double t21, tzk0;

  double t22, t23, t24, t25, t27, t29, t33, t34;
  double t36, t37, t39, tvrho0, t41, tvrho1;

  double t44, t45, t47, t51, t53, t54, t56, t57;
  double t58, t60, t61, t66, t68, t70, t74, t75;
  double t77, t79, t81, tv2rho20, t82, t83, t84, t89;
  double tv2rho21, t93, tv2rho22;

  double t97, t98, t100, t101, t103, t104, t108, t110;
  double t114, t116, t117, t118, t119, t121, t123, t124;
  double t129, t133, t136, t137, t139, t141, tv3rho30, t143;
  double t145, t146, t147, t149, t150, t151, t157, t160;
  double tv3rho31, t166, t167, t168, t173, tv3rho32, t179, tv3rho33;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = rho[0] - rho[1];
  t2 = t1 * t1;
  t3 = rho[0] + rho[1];
  t4 = t3 * t3;
  t5 = 0.1e1 / t4;
  t7 = -t2 * t5 + 0.1e1;
  t8 = t7 * params->a;
  t9 = M_CBRT3;
  t10 = 0.1e1 / M_PI;
  t11 = POW_1_3(t10);
  t12 = t9 * t11;
  t13 = M_CBRT4;
  t14 = t13 * t13;
  t15 = POW_1_3(t3);
  t16 = 0.1e1 / t15;
  t20 = params->b + t12 * t14 * t16 / 0.4e1;
  t21 = 0.1e1 / t20;
  tzk0 = t8 * t21;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

  t22 = t1 * t5;
  t23 = t4 * t3;
  t24 = 0.1e1 / t23;
  t25 = t2 * t24;
  t27 = -0.2e1 * t22 + 0.2e1 * t25;
  t29 = params->a * t21;
  t33 = t20 * t20;
  t34 = 0.1e1 / t33;
  t36 = t11 * t14;
  t37 = t34 * t9 * t36;
  t39 = t16 * t7 * params->a * t37 / 0.12e2;
  tvrho0 = t3 * t27 * t29 + t39 + tzk0;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 0] += tvrho0;

  t41 = 0.2e1 * t22 + 0.2e1 * t25;
  tvrho1 = t3 * t41 * t29 + t39 + tzk0;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 1] += tvrho1;

  t44 = t27 * params->a;
  t45 = t44 * t21;
  t47 = t8 * t34;
  t51 = t12 * t14 / t15 / t3;
  t53 = t47 * t51 / 0.18e2;
  t54 = 0.2e1 * t5;
  t56 = 0.8e1 * t1 * t24;
  t57 = t4 * t4;
  t58 = 0.1e1 / t57;
  t60 = 0.6e1 * t2 * t58;
  t61 = -t54 + t56 - t60;
  t66 = t16 * t27 * params->a * t37;
  t68 = t15 * t15;
  t70 = 0.1e1 / t68 / t3;
  t74 = 0.1e1 / t33 / t20;
  t75 = t9 * t9;
  t77 = t11 * t11;
  t79 = t74 * t75 * t77 * t13;
  t81 = t70 * t7 * params->a * t79 / 0.18e2;
  tv2rho20 = 0.2e1 * t45 + t53 + t3 * t61 * t29 + t66 / 0.6e1 + t81;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 0] += tv2rho20;

  t82 = t41 * params->a;
  t83 = t82 * t21;
  t84 = t54 - t60;
  t89 = t16 * t41 * params->a * t37;
  tv2rho21 = t45 + t53 + t83 + t3 * t84 * t29 + t89 / 0.12e2 + t66 / 0.12e2 + t81;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 1] += tv2rho21;

  t93 = -t54 - t56 - t60;
  tv2rho22 = 0.2e1 * t83 + t53 + t3 * t93 * t29 + t89 / 0.6e1 + t81;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 2] += tv2rho22;

  t97 = t61 * params->a;
  t98 = t97 * t21;
  t100 = t44 * t34;
  t101 = t100 * t51;
  t103 = t8 * t74;
  t104 = t75 * t77;
  t108 = t104 * t13 / t68 / t4;
  t110 = t103 * t108 / 0.18e2;
  t114 = t12 * t14 / t15 / t4;
  t116 = 0.2e1 / 0.27e2 * t47 * t114;
  t117 = 0.12e2 * t24;
  t118 = t1 * t58;
  t119 = 0.36e2 * t118;
  t121 = 0.1e1 / t57 / t3;
  t123 = 0.24e2 * t2 * t121;
  t124 = t117 - t119 + t123;
  t129 = t16 * t61 * params->a * t37;
  t133 = t70 * t27 * params->a * t79;
  t136 = t33 * t33;
  t137 = 0.1e1 / t136;
  t139 = params->a * t137 * t10;
  t141 = t24 * t7 * t139 / 0.6e1;
  tv3rho30 = 0.3e1 * t98 + t101 / 0.6e1 - t110 - t116 + t3 * t124 * t29 + t129 / 0.4e1 + t133 / 0.6e1 + t141;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 0] += tv3rho30;

  t143 = t84 * params->a;
  t145 = 0.2e1 * t143 * t21;
  t146 = t82 * t34;
  t147 = t146 * t51;
  t149 = 0.4e1 * t24;
  t150 = 0.12e2 * t118;
  t151 = -t149 - t150 + t123;
  t157 = t16 * t84 * params->a * t37 / 0.6e1;
  t160 = t70 * t41 * params->a * t79;
  tv3rho31 = t98 + t101 / 0.9e1 - t110 - t116 + t145 + t147 / 0.18e2 + t3 * t151 * t29 + t157 + t160 / 0.18e2 + t129 / 0.12e2 + t133 / 0.9e1 + t141;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 1] += tv3rho31;

  t166 = t93 * params->a;
  t167 = t166 * t21;
  t168 = -t149 + t150 + t123;
  t173 = t16 * t93 * params->a * t37;
  tv3rho32 = t145 + t147 / 0.9e1 + t101 / 0.18e2 - t110 - t116 + t167 + t3 * t168 * t29 + t173 / 0.12e2 + t157 + t160 / 0.9e1 + t133 / 0.18e2 + t141;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 2] += tv3rho32;

  t179 = t117 + t119 + t123;
  tv3rho33 = 0.3e1 * t167 + t147 / 0.6e1 - t110 - t116 + t3 * t179 * t29 + t173 / 0.4e1 + t160 / 0.6e1 + t141;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 3] += tv3rho33;

}

#endif


#ifndef XC_DONT_COMPILE_LXC
GPU_DEVICE_FUNCTION static inline void
func_lxc_pol(const xc_func_type *p, size_t ip, const double *rho, xc_lda_out_params *out)
{
  double t1, t2, t3, t4, t5, t7, t8, t9;
  double t10, t11, t12, t13, t14, t15, t16, t20;
  double t21, tzk0;

  double t22, t23, t24, t25, t27, t29, t33, t34;
  double t36, t37, t39, tvrho0, t41, tvrho1;

  double t44, t45, t47, t51, t53, t54, t56, t57;
  double t58, t60, t61, t66, t68, t70, t74, t75;
  double t77, t79, t81, tv2rho20, t82, t83, t84, t89;
  double tv2rho21, t93, tv2rho22;

  double t97, t98, t100, t101, t103, t104, t108, t110;
  double t114, t116, t117, t118, t119, t121, t123, t124;
  double t129, t133, t136, t137, t139, t141, tv3rho30, t143;
  double t145, t146, t147, t149, t150, t151, t157, t160;
  double tv3rho31, t166, t167, t168, t173, tv3rho32, t179, tv3rho33;

  double t185, t188, t191, t193, t198, t204, t210, t211;
  double t212, t213, t217, t223, t227, t230, t242, tv4rho40;
  double t245, t247, t252, t255, t262, t263, t265, t269;
  double t273, t274, tv4rho41, t286, t296, t300, t304, t306;
  double tv4rho42, t314, t320, tv4rho43, tv4rho44;

  lda_c_wigner_params *params;

  assert(p->params != NULL);
  params = (lda_c_wigner_params * )(p->params);

  t1 = rho[0] - rho[1];
  t2 = t1 * t1;
  t3 = rho[0] + rho[1];
  t4 = t3 * t3;
  t5 = 0.1e1 / t4;
  t7 = -t2 * t5 + 0.1e1;
  t8 = t7 * params->a;
  t9 = M_CBRT3;
  t10 = 0.1e1 / M_PI;
  t11 = POW_1_3(t10);
  t12 = t9 * t11;
  t13 = M_CBRT4;
  t14 = t13 * t13;
  t15 = POW_1_3(t3);
  t16 = 0.1e1 / t15;
  t20 = params->b + t12 * t14 * t16 / 0.4e1;
  t21 = 0.1e1 / t20;
  tzk0 = t8 * t21;

  if(out->zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    out->zk[ip*p->dim.zk + 0] += tzk0;

  t22 = t1 * t5;
  t23 = t4 * t3;
  t24 = 0.1e1 / t23;
  t25 = t2 * t24;
  t27 = -0.2e1 * t22 + 0.2e1 * t25;
  t29 = params->a * t21;
  t33 = t20 * t20;
  t34 = 0.1e1 / t33;
  t36 = t11 * t14;
  t37 = t34 * t9 * t36;
  t39 = t16 * t7 * params->a * t37 / 0.12e2;
  tvrho0 = t3 * t27 * t29 + t39 + tzk0;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 0] += tvrho0;

  t41 = 0.2e1 * t22 + 0.2e1 * t25;
  tvrho1 = t3 * t41 * t29 + t39 + tzk0;

  if(out->vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    out->vrho[ip*p->dim.vrho + 1] += tvrho1;

  t44 = t27 * params->a;
  t45 = t44 * t21;
  t47 = t8 * t34;
  t51 = t12 * t14 / t15 / t3;
  t53 = t47 * t51 / 0.18e2;
  t54 = 0.2e1 * t5;
  t56 = 0.8e1 * t1 * t24;
  t57 = t4 * t4;
  t58 = 0.1e1 / t57;
  t60 = 0.6e1 * t2 * t58;
  t61 = -t54 + t56 - t60;
  t66 = t16 * t27 * params->a * t37;
  t68 = t15 * t15;
  t70 = 0.1e1 / t68 / t3;
  t74 = 0.1e1 / t33 / t20;
  t75 = t9 * t9;
  t77 = t11 * t11;
  t79 = t74 * t75 * t77 * t13;
  t81 = t70 * t7 * params->a * t79 / 0.18e2;
  tv2rho20 = 0.2e1 * t45 + t53 + t3 * t61 * t29 + t66 / 0.6e1 + t81;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 0] += tv2rho20;

  t82 = t41 * params->a;
  t83 = t82 * t21;
  t84 = t54 - t60;
  t89 = t16 * t41 * params->a * t37;
  tv2rho21 = t45 + t53 + t83 + t3 * t84 * t29 + t89 / 0.12e2 + t66 / 0.12e2 + t81;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 1] += tv2rho21;

  t93 = -t54 - t56 - t60;
  tv2rho22 = 0.2e1 * t83 + t53 + t3 * t93 * t29 + t89 / 0.6e1 + t81;

  if(out->v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    out->v2rho2[ip*p->dim.v2rho2 + 2] += tv2rho22;

  t97 = t61 * params->a;
  t98 = t97 * t21;
  t100 = t44 * t34;
  t101 = t100 * t51;
  t103 = t8 * t74;
  t104 = t75 * t77;
  t108 = t104 * t13 / t68 / t4;
  t110 = t103 * t108 / 0.18e2;
  t114 = t12 * t14 / t15 / t4;
  t116 = 0.2e1 / 0.27e2 * t47 * t114;
  t117 = 0.12e2 * t24;
  t118 = t1 * t58;
  t119 = 0.36e2 * t118;
  t121 = 0.1e1 / t57 / t3;
  t123 = 0.24e2 * t2 * t121;
  t124 = t117 - t119 + t123;
  t129 = t16 * t61 * params->a * t37;
  t133 = t70 * t27 * params->a * t79;
  t136 = t33 * t33;
  t137 = 0.1e1 / t136;
  t139 = params->a * t137 * t10;
  t141 = t24 * t7 * t139 / 0.6e1;
  tv3rho30 = 0.3e1 * t98 + t101 / 0.6e1 - t110 - t116 + t3 * t124 * t29 + t129 / 0.4e1 + t133 / 0.6e1 + t141;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 0] += tv3rho30;

  t143 = t84 * params->a;
  t145 = 0.2e1 * t143 * t21;
  t146 = t82 * t34;
  t147 = t146 * t51;
  t149 = 0.4e1 * t24;
  t150 = 0.12e2 * t118;
  t151 = -t149 - t150 + t123;
  t157 = t16 * t84 * params->a * t37 / 0.6e1;
  t160 = t70 * t41 * params->a * t79;
  tv3rho31 = t98 + t101 / 0.9e1 - t110 - t116 + t145 + t147 / 0.18e2 + t3 * t151 * t29 + t157 + t160 / 0.18e2 + t129 / 0.12e2 + t133 / 0.9e1 + t141;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 1] += tv3rho31;

  t166 = t93 * params->a;
  t167 = t166 * t21;
  t168 = -t149 + t150 + t123;
  t173 = t16 * t93 * params->a * t37;
  tv3rho32 = t145 + t147 / 0.9e1 + t101 / 0.18e2 - t110 - t116 + t167 + t3 * t168 * t29 + t173 / 0.12e2 + t157 + t160 / 0.9e1 + t133 / 0.18e2 + t141;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 2] += tv3rho32;

  t179 = t117 + t119 + t123;
  tv3rho33 = 0.3e1 * t167 + t147 / 0.6e1 - t110 - t116 + t3 * t179 * t29 + t173 / 0.4e1 + t160 / 0.6e1 + t141;

  if(out->v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    out->v3rho3[ip*p->dim.v3rho3 + 3] += tv3rho33;

  t185 = t124 * params->a * t21;
  t188 = t97 * t34 * t51;
  t191 = t44 * t74 * t108;
  t193 = t100 * t114;
  t198 = 0.2e1 / 0.3e1 * t8 * t137 * t10 * t58;
  t204 = 0.8e1 / 0.81e2 * t103 * t104 * t13 / t68 / t23;
  t210 = 0.14e2 / 0.81e2 * t47 * t12 * t14 / t15 / t23;
  t211 = 0.72e2 * t58;
  t212 = t1 * t121;
  t213 = 0.192e3 * t212;
  t217 = 0.12e3 * t2 / t57 / t4;
  t223 = t16 * t124 * params->a * t37;
  t227 = t70 * t61 * params->a * t79;
  t230 = t24 * t27 * t139;
  t242 = 0.1e1 / t15 / t57 * t7 * params->a / t136 / t20 * t10 * t9 * t36 / 0.18e2;
  tv4rho40 = 0.4e1 * t185 + t188 / 0.3e1 - 0.2e1 / 0.9e1 * t191 - 0.8e1 / 0.27e2 * t193 - t198 + t204 + t210 + t3 * (-t211 + t213 - t217) * t29 + t223 / 0.3e1 + t227 / 0.3e1 + 0.2e1 / 0.3e1 * t230 + t242;

  if(out->v4rho4 != NULL && (p->info->flags & XC_FLAGS_HAVE_LXC))
    out->v4rho4[ip*p->dim.v4rho4 + 0] += tv4rho40;

  t245 = t24 * t41 * t139;
  t247 = 0.96e2 * t212;
  t252 = t146 * t114;
  t255 = t151 * params->a * t21;
  t262 = t143 * t34 * t51;
  t263 = t262 / 0.6e1;
  t265 = t82 * t74 * t108;
  t269 = t16 * t151 * params->a * t37;
  t273 = t70 * t84 * params->a * t79;
  t274 = t273 / 0.6e1;
  tv4rho41 = -t198 + t230 / 0.2e1 + t245 / 0.6e1 + t3 * (t247 - t217) * t29 - 0.2e1 / 0.9e1 * t193 + t204 + t210 + t242 - 0.2e1 / 0.27e2 * t252 + t185 + 0.3e1 * t255 + t188 / 0.6e1 - t191 / 0.6e1 + t223 / 0.12e2 + t227 / 0.6e1 + t263 - t265 / 0.18e2 + t269 / 0.4e1 + t274;

  if(out->v4rho4 != NULL && (p->info->flags & XC_FLAGS_HAVE_LXC))
    out->v4rho4[ip*p->dim.v4rho4 + 1] += tv4rho41;

  t286 = t168 * params->a * t21;
  t296 = t166 * t34 * t51;
  t300 = t16 * t168 * params->a * t37;
  t304 = t70 * t93 * params->a * t79;
  t306 = 0.2e1 * t286 + t188 / 0.18e2 - t191 / 0.9e1 + t227 / 0.18e2 + 0.2e1 / 0.9e1 * t262 - t265 / 0.9e1 + t269 / 0.6e1 + 0.2e1 / 0.9e1 * t273 + t296 / 0.18e2 + t300 / 0.6e1 + t304 / 0.18e2;
  tv4rho42 = -t198 + t230 / 0.3e1 + t245 / 0.3e1 + t3 * (0.24e2 * t58 - t217) * t29 - 0.4e1 / 0.27e2 * t193 + t204 + t210 + t242 - 0.4e1 / 0.27e2 * t252 + 0.2e1 * t255 + t306;

  if(out->v4rho4 != NULL && (p->info->flags & XC_FLAGS_HAVE_LXC))
    out->v4rho4[ip*p->dim.v4rho4 + 2] += tv4rho42;

  t314 = t179 * params->a * t21;
  t320 = t16 * t179 * params->a * t37;
  tv4rho43 = 0.3e1 * t286 + t296 / 0.6e1 + t263 - t265 / 0.6e1 - 0.2e1 / 0.9e1 * t252 - t191 / 0.18e2 - t198 + t204 - 0.2e1 / 0.27e2 * t193 + t210 + t314 + t3 * (-t247 - t217) * t29 + t320 / 0.12e2 + t300 / 0.4e1 + t304 / 0.6e1 + t274 + t245 / 0.2e1 + t230 / 0.6e1 + t242;

  if(out->v4rho4 != NULL && (p->info->flags & XC_FLAGS_HAVE_LXC))
    out->v4rho4[ip*p->dim.v4rho4 + 3] += tv4rho43;

  tv4rho44 = 0.4e1 * t314 + t296 / 0.3e1 - 0.2e1 / 0.9e1 * t265 - 0.8e1 / 0.27e2 * t252 - t198 + t204 + t210 + t3 * (-t211 - t213 - t217) * t29 + t320 / 0.3e1 + t304 / 0.3e1 + 0.2e1 / 0.3e1 * t245 + t242;

  if(out->v4rho4 != NULL && (p->info->flags & XC_FLAGS_HAVE_LXC))
    out->v4rho4[ip*p->dim.v4rho4 + 4] += tv4rho44;

}

#endif

