/* 
  This file was generated automatically with ./scripts/maple2c_new.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ./maple/lda_exc/lda_c_hl.mpl
  Type of functional: lda_exc
*/

#define maple2c_order 3

static inline void
func_unpol(const xc_func_type *p, int order, const double *rho, double *zk, double *vrho, double *v2rho2, double *v3rho3)
{
  double t1, t2, t5, t6, t7, t8, t11, t12;
  double t13, t14, t15, t16, t17, t18, t23, t24;
  double t26, t27, t28, t30, t31, t35, t36, t39;
  double t45, t46, t47, t49, t53, t54, t55, t61;
  double t67, t71, t75, t81, t85, t92, t93, t94;
  double t95, t96, t100, t106, t110, t114, t120, t126;

  lda_c_hl_params *params;

  assert(p->params != NULL);
  params = (lda_c_hl_params * )(p->params);

  t1 = params->c[0];
  t2 = 0.1e1 / M_PI;
  t5 = params->r[0];
  t6 = t5 * t5;
  t7 = t6 * t5;
  t8 = 0.1e1 / t7;
  t11 = 0.1e1 + 0.3e1 / 0.4e1 * t2 / rho[0] * t8;
  t12 = M_CBRT3;
  t13 = t12 * t12;
  t14 = POW_1_3(t2);
  t15 = 0.1e1 / t14;
  t16 = t13 * t15;
  t17 = M_CBRT4;
  t18 = POW_1_3(rho[0]);
  t23 = 0.1e1 + t16 * t17 * t18 * t5 / 0.3e1;
  t24 = log(t23);
  t26 = t14 * t14;
  t27 = t13 * t26;
  t28 = t18 * t18;
  t30 = t17 / t28;
  t31 = 0.1e1 / t6;
  t35 = t12 * t14;
  t36 = t17 * t17;
  t39 = 0.1e1 / t5;
  if(zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    *zk = -t1 * (t11 * t24 - t27 * t30 * t31 / 0.4e1 + t35 * t36 / t18 * t39 / 0.8e1 - 0.1e1 / 0.3e1);

#ifndef XC_DONT_COMPILE_VXC

  if(order < 1) return;


  t45 = rho[0] * t1;
  t46 = rho[0] * rho[0];
  t47 = 0.1e1 / t46;
  t49 = t8 * t24;
  t53 = t11 * t13 * t15;
  t54 = 0.1e1 / t23;
  t55 = t5 * t54;
  t61 = t17 / t28 / rho[0];
  t67 = t36 / t18 / rho[0];
  t71 = -0.3e1 / 0.4e1 * t2 * t47 * t49 + t53 * t30 * t55 / 0.9e1 + t27 * t61 * t31 / 0.6e1 - t35 * t67 * t39 / 0.24e2;
  if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    vrho[0] = -t45 * t71 + (-t1 * (t11 * t24 - t27 * t30 * t31 / 0.4e1 + t35 * t36 / t18 * t39 / 0.8e1 - 0.1e1 / 0.3e1));

#ifndef XC_DONT_COMPILE_FXC

  if(order < 2) return;


  t75 = t46 * rho[0];
  t81 = 0.1e1 / t28 / t46;
  t85 = t16 * t17 * t54;
  t92 = 0.1e1 / t26;
  t93 = t11 * t12 * t92;
  t94 = t23 * t23;
  t95 = 0.1e1 / t94;
  t96 = t6 * t95;
  t100 = t17 * t81;
  t106 = t36 / t18 / t46;
  t110 = 0.3e1 / 0.2e1 * t2 / t75 * t49 - t2 * t81 * t31 * t85 / 0.6e1 - 0.2e1 / 0.27e2 * t53 * t61 * t55 - t93 * t67 * t96 / 0.27e2 - 0.5e1 / 0.18e2 * t27 * t100 * t31 + t35 * t106 * t39 / 0.18e2;
  if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    v2rho2[0] = -0.2e1 * t1 * t71 - t45 * t110;

#ifndef XC_DONT_COMPILE_KXC

  if(order < 3) return;


  t114 = t46 * t46;
  t120 = 0.1e1 / t28 / t75;
  t126 = 0.1e1 / t18 / t75;
  if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    v3rho3[0] = -0.3e1 * t1 * t110 - t45 * (-0.9e1 / 0.2e1 * t2 / t114 * t49 + 0.2e1 / 0.3e1 * t2 * t120 * t31 * t85 + t2 * t126 * t39 * t12 * t92 * t36 * t95 / 0.12e2 + 0.10e2 / 0.81e2 * t53 * t100 * t55 + 0.2e1 / 0.27e2 * t93 * t106 * t96 + 0.8e1 / 0.81e2 * t11 * M_PI * t47 * t7 / t94 / t23 + 0.20e2 / 0.27e2 * t27 * t17 * t120 * t31 - 0.7e1 / 0.54e2 * t35 * t36 * t126 * t39);

#ifndef XC_DONT_COMPILE_LXC

  if(order < 4) return;


#endif

#endif

#endif

#endif


}


static inline void
func_ferr(const xc_func_type *p, int order, const double *rho, double *zk, double *vrho, double *v2rho2, double *v3rho3)
{
  double t1, t2, t5, t6, t7, t8, t11, t12;
  double t13, t14, t15, t16, t17, t18, t23, t24;
  double t26, t27, t28, t30, t31, t35, t36, t39;
  double t45, t46, t47, t49, t53, t54, t55, t61;
  double t67, t71, t75, t81, t85, t92, t93, t94;
  double t95, t96, t100, t106, t110, t114, t120, t126;

  lda_c_hl_params *params;

  assert(p->params != NULL);
  params = (lda_c_hl_params * )(p->params);

  t1 = params->c[1];
  t2 = 0.1e1 / M_PI;
  t5 = params->r[1];
  t6 = t5 * t5;
  t7 = t6 * t5;
  t8 = 0.1e1 / t7;
  t11 = 0.1e1 + 0.3e1 / 0.4e1 * t2 / rho[0] * t8;
  t12 = M_CBRT3;
  t13 = t12 * t12;
  t14 = POW_1_3(t2);
  t15 = 0.1e1 / t14;
  t16 = t13 * t15;
  t17 = M_CBRT4;
  t18 = POW_1_3(rho[0]);
  t23 = 0.1e1 + t16 * t17 * t18 * t5 / 0.3e1;
  t24 = log(t23);
  t26 = t14 * t14;
  t27 = t13 * t26;
  t28 = t18 * t18;
  t30 = t17 / t28;
  t31 = 0.1e1 / t6;
  t35 = t12 * t14;
  t36 = t17 * t17;
  t39 = 0.1e1 / t5;
  if(zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    *zk = -t1 * (t11 * t24 - t27 * t30 * t31 / 0.4e1 + t35 * t36 / t18 * t39 / 0.8e1 - 0.1e1 / 0.3e1);

#ifndef XC_DONT_COMPILE_VXC

  if(order < 1) return;


  t45 = rho[0] * t1;
  t46 = rho[0] * rho[0];
  t47 = 0.1e1 / t46;
  t49 = t8 * t24;
  t53 = t11 * t13 * t15;
  t54 = 0.1e1 / t23;
  t55 = t5 * t54;
  t61 = t17 / t28 / rho[0];
  t67 = t36 / t18 / rho[0];
  t71 = -0.3e1 / 0.4e1 * t2 * t47 * t49 + t53 * t30 * t55 / 0.9e1 + t27 * t61 * t31 / 0.6e1 - t35 * t67 * t39 / 0.24e2;
  if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    vrho[0] = -t45 * t71 + (-t1 * (t11 * t24 - t27 * t30 * t31 / 0.4e1 + t35 * t36 / t18 * t39 / 0.8e1 - 0.1e1 / 0.3e1));

#ifndef XC_DONT_COMPILE_FXC

  if(order < 2) return;


  t75 = t46 * rho[0];
  t81 = 0.1e1 / t28 / t46;
  t85 = t16 * t17 * t54;
  t92 = 0.1e1 / t26;
  t93 = t11 * t12 * t92;
  t94 = t23 * t23;
  t95 = 0.1e1 / t94;
  t96 = t6 * t95;
  t100 = t17 * t81;
  t106 = t36 / t18 / t46;
  t110 = 0.3e1 / 0.2e1 * t2 / t75 * t49 - t2 * t81 * t31 * t85 / 0.6e1 - 0.2e1 / 0.27e2 * t53 * t61 * t55 - t93 * t67 * t96 / 0.27e2 - 0.5e1 / 0.18e2 * t27 * t100 * t31 + t35 * t106 * t39 / 0.18e2;
  if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    v2rho2[0] = -0.2e1 * t1 * t71 - t45 * t110;

#ifndef XC_DONT_COMPILE_KXC

  if(order < 3) return;


  t114 = t46 * t46;
  t120 = 0.1e1 / t28 / t75;
  t126 = 0.1e1 / t18 / t75;
  if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    v3rho3[0] = -0.3e1 * t1 * t110 - t45 * (-0.9e1 / 0.2e1 * t2 / t114 * t49 + 0.2e1 / 0.3e1 * t2 * t120 * t31 * t85 + t2 * t126 * t39 * t12 * t92 * t36 * t95 / 0.12e2 + 0.10e2 / 0.81e2 * t53 * t100 * t55 + 0.2e1 / 0.27e2 * t93 * t106 * t96 + 0.8e1 / 0.81e2 * t11 * M_PI * t47 * t7 / t94 / t23 + 0.20e2 / 0.27e2 * t27 * t17 * t120 * t31 - 0.7e1 / 0.54e2 * t35 * t36 * t126 * t39);

#ifndef XC_DONT_COMPILE_LXC

  if(order < 4) return;


#endif

#endif

#endif

#endif


}


static inline void
func_pol(const xc_func_type *p, int order, const double *rho, double *zk, double *vrho, double *v2rho2, double *v3rho3)
{
  double t1, t2, t3, t4, t5, t6, t7, t8;
  double t9, t12, t13, t14, t15, t16, t17, t18;
  double t19, t20, t24, t25, t27, t28, t29, t31;
  double t32, t36, t37, t39, t40, t45, t46, t47;
  double t48, t49, t51, t52, t55, t58, t59, t60;
  double t61, t62, t63, t64, t67, t71, t72, t74;
  double t78, t84, t85, t86, t87, t88, t89, t93;
  double t94, t95, t101, t107, t112, t113, t114, t116;
  double t120, t121, t122, t126, t127, t128, t140, t141;
  double t144, t146, t150, t151, t154, t156, t157, t158;
  double t159, t163, t164, t167, t174, t175, t176, t177;
  double t178, t182, t188, t193, t194, t195, t196, t199;
  double t201, t204, t205, t206, t209, t213, t214, t215;
  double t216, t221, t228, t229, t230, t231, t243, t244;
  double t247, t250, t253, t256, t260, t261, t262, t266;
  double t270, t273, t276, t280, t281, t282, t285, t288;
  double t289, t290, t291, t295, t296, t301, t302, t304;
  double t322, t326, t331, t333, t340, t342, t346, t359;
  double t361, t395, t399, t430, t431, t446, t463, t475;

  lda_c_hl_params *params;

  assert(p->params != NULL);
  params = (lda_c_hl_params * )(p->params);

  t1 = params->c[0];
  t2 = 0.1e1 / M_PI;
  t3 = rho[0] + rho[1];
  t4 = 0.1e1 / t3;
  t5 = t2 * t4;
  t6 = params->r[0];
  t7 = t6 * t6;
  t8 = t7 * t6;
  t9 = 0.1e1 / t8;
  t12 = 0.1e1 + 0.3e1 / 0.4e1 * t5 * t9;
  t13 = M_CBRT3;
  t14 = t13 * t13;
  t15 = POW_1_3(t2);
  t16 = 0.1e1 / t15;
  t17 = t14 * t16;
  t18 = M_CBRT4;
  t19 = POW_1_3(t3);
  t20 = t18 * t19;
  t24 = 0.1e1 + t17 * t20 * t6 / 0.3e1;
  t25 = log(t24);
  t27 = t15 * t15;
  t28 = t14 * t27;
  t29 = t19 * t19;
  t31 = t18 / t29;
  t32 = 0.1e1 / t7;
  t36 = t13 * t15;
  t37 = t18 * t18;
  t39 = t37 / t19;
  t40 = 0.1e1 / t6;
  t45 = t1 * (t12 * t25 - t28 * t31 * t32 / 0.4e1 + t36 * t39 * t40 / 0.8e1 - 0.1e1 / 0.3e1);
  t46 = rho[0] - rho[1];
  t47 = t46 * t4;
  t48 = 0.1e1 + t47;
  t49 = POW_1_3(t48);
  t51 = 0.1e1 - t47;
  t52 = POW_1_3(t51);
  t55 = M_CBRT2;
  t58 = 0.1e1 / (0.2e1 * t55 - 0.2e1);
  t59 = (t49 * t48 + t52 * t51 - 0.2e1) * t58;
  t60 = params->c[1];
  t61 = params->r[1];
  t62 = t61 * t61;
  t63 = t62 * t61;
  t64 = 0.1e1 / t63;
  t67 = 0.1e1 + 0.3e1 / 0.4e1 * t5 * t64;
  t71 = 0.1e1 + t17 * t20 * t61 / 0.3e1;
  t72 = log(t71);
  t74 = 0.1e1 / t62;
  t78 = 0.1e1 / t61;
  t84 = -t60 * (t67 * t72 - t28 * t31 * t74 / 0.4e1 + t36 * t39 * t78 / 0.8e1 - 0.1e1 / 0.3e1) + t45;
  t85 = t59 * t84;
  if(zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
    *zk = -t45 + t85;

#ifndef XC_DONT_COMPILE_VXC

  if(order < 1) return;


  t86 = t3 * t3;
  t87 = 0.1e1 / t86;
  t88 = t2 * t87;
  t89 = t9 * t25;
  t93 = t12 * t14 * t16;
  t94 = 0.1e1 / t24;
  t95 = t6 * t94;
  t101 = t18 / t29 / t3;
  t107 = t37 / t19 / t3;
  t112 = t1 * (-0.3e1 / 0.4e1 * t88 * t89 + t93 * t31 * t95 / 0.9e1 + t28 * t101 * t32 / 0.6e1 - t36 * t107 * t40 / 0.24e2);
  t113 = t46 * t87;
  t114 = t4 - t113;
  t116 = -t114;
  t120 = (0.4e1 / 0.3e1 * t49 * t114 + 0.4e1 / 0.3e1 * t52 * t116) * t58;
  t121 = t120 * t84;
  t122 = t64 * t72;
  t126 = t67 * t14 * t16;
  t127 = 0.1e1 / t71;
  t128 = t61 * t127;
  t140 = -t60 * (-0.3e1 / 0.4e1 * t88 * t122 + t126 * t31 * t128 / 0.9e1 + t28 * t101 * t74 / 0.6e1 - t36 * t107 * t78 / 0.24e2) + t112;
  t141 = t59 * t140;
  if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    vrho[0] = -t45 + t85 + t3 * (-t112 + t121 + t141);

  t144 = -t4 - t113;
  t146 = -t144;
  t150 = (0.4e1 / 0.3e1 * t49 * t144 + 0.4e1 / 0.3e1 * t52 * t146) * t58;
  t151 = t150 * t84;
  if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC))
    vrho[1] = -t45 + t85 + t3 * (-t112 + t151 + t141);

#ifndef XC_DONT_COMPILE_FXC

  if(order < 2) return;


  t154 = 0.2e1 * t112;
  t156 = 0.2e1 * t141;
  t157 = t86 * t3;
  t158 = 0.1e1 / t157;
  t159 = t2 * t158;
  t163 = 0.1e1 / t29 / t86;
  t164 = t2 * t163;
  t167 = t17 * t18 * t94;
  t174 = 0.1e1 / t27;
  t175 = t12 * t13 * t174;
  t176 = t24 * t24;
  t177 = 0.1e1 / t176;
  t178 = t7 * t177;
  t182 = t18 * t163;
  t188 = t37 / t19 / t86;
  t193 = t1 * (0.3e1 / 0.2e1 * t159 * t89 - t164 * t32 * t167 / 0.6e1 - 0.2e1 / 0.27e2 * t93 * t101 * t95 - t175 * t107 * t178 / 0.27e2 - 0.5e1 / 0.18e2 * t28 * t182 * t32 + t36 * t188 * t40 / 0.18e2);
  t194 = t49 * t49;
  t195 = 0.1e1 / t194;
  t196 = t114 * t114;
  t199 = t46 * t158;
  t201 = -0.2e1 * t87 + 0.2e1 * t199;
  t204 = t52 * t52;
  t205 = 0.1e1 / t204;
  t206 = t116 * t116;
  t209 = -t201;
  t213 = (0.4e1 / 0.9e1 * t195 * t196 + 0.4e1 / 0.3e1 * t49 * t201 + 0.4e1 / 0.9e1 * t205 * t206 + 0.4e1 / 0.3e1 * t52 * t209) * t58;
  t214 = t213 * t84;
  t215 = t120 * t140;
  t216 = 0.2e1 * t215;
  t221 = t17 * t18 * t127;
  t228 = t67 * t13 * t174;
  t229 = t71 * t71;
  t230 = 0.1e1 / t229;
  t231 = t62 * t230;
  t243 = -t60 * (0.3e1 / 0.2e1 * t159 * t122 - t164 * t74 * t221 / 0.6e1 - 0.2e1 / 0.27e2 * t126 * t101 * t128 - t228 * t107 * t231 / 0.27e2 - 0.5e1 / 0.18e2 * t28 * t182 * t74 + t36 * t188 * t78 / 0.18e2) + t193;
  t244 = t59 * t243;
  if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    v2rho2[0] = -t154 + 0.2e1 * t121 + t156 + t3 * (-t193 + t214 + t216 + t244);

  t247 = t195 * t144;
  t250 = t49 * t46;
  t253 = t205 * t146;
  t256 = t52 * t46;
  t260 = (0.4e1 / 0.9e1 * t247 * t114 + 0.8e1 / 0.3e1 * t250 * t158 + 0.4e1 / 0.9e1 * t253 * t116 - 0.8e1 / 0.3e1 * t256 * t158) * t58;
  t261 = t260 * t84;
  t262 = t150 * t140;
  if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    v2rho2[1] = -t154 + t121 + t156 + t151 + t3 * (-t193 + t261 + t262 + t215 + t244);

  t266 = t144 * t144;
  t270 = 0.2e1 * t87 + 0.2e1 * t199;
  t273 = t146 * t146;
  t276 = -t270;
  t280 = (0.4e1 / 0.9e1 * t195 * t266 + 0.4e1 / 0.3e1 * t49 * t270 + 0.4e1 / 0.9e1 * t205 * t273 + 0.4e1 / 0.3e1 * t52 * t276) * t58;
  t281 = t280 * t84;
  t282 = 0.2e1 * t262;
  if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC))
    v2rho2[2] = -t154 + 0.2e1 * t151 + t156 + t3 * (-t193 + t281 + t282 + t244);

#ifndef XC_DONT_COMPILE_KXC

  if(order < 3) return;


  t285 = 0.3e1 * t193;
  t288 = 0.3e1 * t244;
  t289 = t86 * t86;
  t290 = 0.1e1 / t289;
  t291 = t2 * t290;
  t295 = 0.1e1 / t29 / t157;
  t296 = t2 * t295;
  t301 = 0.1e1 / t19 / t157;
  t302 = t2 * t301;
  t304 = t13 * t174;
  t322 = t18 * t295;
  t326 = t37 * t301;
  t331 = t1 * (-0.9e1 / 0.2e1 * t291 * t89 + 0.2e1 / 0.3e1 * t296 * t32 * t167 + t302 * t40 * t304 * t37 * t177 / 0.12e2 + 0.10e2 / 0.81e2 * t93 * t182 * t95 + 0.2e1 / 0.27e2 * t175 * t188 * t178 + 0.8e1 / 0.81e2 * t12 * M_PI * t87 * t8 / t176 / t24 + 0.20e2 / 0.27e2 * t28 * t322 * t32 - 0.7e1 / 0.54e2 * t36 * t326 * t40);
  t333 = 0.1e1 / t194 / t48;
  t340 = t46 * t290;
  t342 = 0.6e1 * t158 - 0.6e1 * t340;
  t346 = 0.1e1 / t204 / t51;
  t359 = t213 * t140;
  t361 = t120 * t243;
  t395 = t59 * (-t60 * (-0.9e1 / 0.2e1 * t291 * t122 + 0.2e1 / 0.3e1 * t296 * t74 * t221 + t302 * t78 * t304 * t37 * t230 / 0.12e2 + 0.10e2 / 0.81e2 * t126 * t182 * t128 + 0.2e1 / 0.27e2 * t228 * t188 * t231 + 0.8e1 / 0.81e2 * t67 * M_PI * t87 * t63 / t229 / t71 + 0.20e2 / 0.27e2 * t28 * t322 * t74 - 0.7e1 / 0.54e2 * t36 * t326 * t78) + t331);
  if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    v3rho3[0] = -t285 + 0.3e1 * t214 + 0.6e1 * t215 + t288 + t3 * (-t331 + (-0.8e1 / 0.27e2 * t333 * t196 * t114 + 0.4e1 / 0.3e1 * t195 * t114 * t201 + 0.4e1 / 0.3e1 * t49 * t342 - 0.8e1 / 0.27e2 * t346 * t206 * t116 + 0.4e1 / 0.3e1 * t205 * t116 * t209 - 0.4e1 / 0.3e1 * t52 * t342) * t58 * t84 + 0.3e1 * t359 + 0.3e1 * t361 + t395);

  t399 = 0.2e1 * t261;
  t430 = 0.2e1 * t260 * t140;
  t431 = t150 * t243;
  if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    v3rho3[1] = -t285 + t214 + 0.4e1 * t215 + t288 + t399 + t282 + t3 * (-t331 + (-0.8e1 / 0.27e2 * t333 * t144 * t196 + 0.16e2 / 0.9e1 * t195 * t46 * t158 * t114 + 0.4e1 / 0.9e1 * t247 * t201 + 0.8e1 / 0.3e1 * t49 * t158 - 0.8e1 * t250 * t290 - 0.8e1 / 0.27e2 * t346 * t146 * t206 - 0.16e2 / 0.9e1 * t205 * t46 * t158 * t116 + 0.4e1 / 0.9e1 * t253 * t209 - 0.8e1 / 0.3e1 * t52 * t158 + 0.8e1 * t256 * t290) * t58 * t84 + t430 + t431 + t359 + 0.2e1 * t361 + t395);

  t446 = -0.2e1 * t158 - 0.6e1 * t340;
  t463 = t280 * t140;
  if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    v3rho3[2] = -t285 + t399 + 0.4e1 * t262 + t216 + t288 + t281 + t3 * (-t331 + (-0.8e1 / 0.27e2 * t333 * t266 * t114 + 0.16e2 / 0.9e1 * t247 * t199 + 0.4e1 / 0.9e1 * t195 * t270 * t114 + 0.4e1 / 0.3e1 * t49 * t446 - 0.8e1 / 0.27e2 * t346 * t273 * t116 - 0.16e2 / 0.9e1 * t253 * t199 + 0.4e1 / 0.9e1 * t205 * t276 * t116 - 0.4e1 / 0.3e1 * t52 * t446) * t58 * t84 + t463 + t430 + 0.2e1 * t431 + t361 + t395);

  t475 = -0.6e1 * t158 - 0.6e1 * t340;
  if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC))
    v3rho3[3] = -t285 + 0.3e1 * t281 + 0.6e1 * t262 + t288 + t3 * (-t331 + (-0.8e1 / 0.27e2 * t333 * t266 * t144 + 0.4e1 / 0.3e1 * t247 * t270 + 0.4e1 / 0.3e1 * t49 * t475 - 0.8e1 / 0.27e2 * t346 * t273 * t146 + 0.4e1 / 0.3e1 * t253 * t276 - 0.4e1 / 0.3e1 * t52 * t475) * t58 * t84 + 0.3e1 * t463 + 0.3e1 * t431 + t395);

#ifndef XC_DONT_COMPILE_LXC

  if(order < 4) return;


#endif

#endif

#endif

#endif


}
