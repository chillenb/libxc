/*
 Copyright (C) 2006-2008 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"
#include "xc_funcs.h"

#define XC_MGGA_X_EDMGGA          686 /* Tao 2001 */
#define XC_HYB_MGGA_XC_EDMGGAH    695 /* Tao 2001 hybrid */

#include "maple2c/mgga_exc/mgga_x_edmgga.c"
#include "work_mgga.c"

#ifdef __cplusplus
extern "C"
#endif
const xc_func_info_type xc_func_info_mgga_x_edmgga = {
  XC_MGGA_X_EDMGGA,
  XC_EXCHANGE,
  "Tao 2001",
  XC_FAMILY_MGGA,
  {&xc_ref_Tao2001_3519, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_NEEDS_LAPLACIAN | MAPLE2C_FLAGS | XC_FLAGS_DEVELOPMENT,
  1e-14,
  {0, NULL, NULL, NULL, NULL},
  NULL, NULL,
  NULL, NULL, &work_mgga,
};

static void
hyb_mgga_xc_edmggah_init(xc_func_type *p)
{
  static int   funcs_id  [2] = {XC_MGGA_X_EDMGGA, XC_MGGA_C_CS};
  static double funcs_coef[2] = {0.78, 1.0};

  xc_mix_init(p, 2, funcs_id, funcs_coef);
  xc_hyb_init_hybrid(p, 0.22);
}

#ifdef __cplusplus
extern "C"
#endif
const xc_func_info_type xc_func_info_hyb_mgga_xc_edmggah = {
  XC_HYB_MGGA_XC_EDMGGAH,
  XC_EXCHANGE_CORRELATION,
  "EDMGGA hybrid",
  XC_FAMILY_HYB_MGGA,
  {&xc_ref_Tao2002_2335, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_NEEDS_TAU | XC_FLAGS_NEEDS_LAPLACIAN | XC_FLAGS_I_HAVE_ALL | XC_FLAGS_DEVELOPMENT,
  1e-14,
  {0, NULL, NULL, NULL, NULL},
  hyb_mgga_xc_edmggah_init, NULL,
  NULL, NULL, NULL
};
