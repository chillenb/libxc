#include "util.h"

extern xc_func_info_type xc_func_info_hyb_mgga_x_dldf;
extern xc_func_info_type xc_func_info_hyb_mgga_x_ms2h;
extern xc_func_info_type xc_func_info_hyb_mgga_x_mn12_sx;
extern xc_func_info_type xc_func_info_hyb_mgga_x_scan0;
extern xc_func_info_type xc_func_info_hyb_mgga_x_mn15;
extern xc_func_info_type xc_func_info_hyb_mgga_x_bmk;
extern xc_func_info_type xc_func_info_hyb_mgga_x_tau_hcth;
extern xc_func_info_type xc_func_info_hyb_mgga_x_m08_hx;
extern xc_func_info_type xc_func_info_hyb_mgga_x_m08_so;
extern xc_func_info_type xc_func_info_hyb_mgga_x_m11;
extern xc_func_info_type xc_func_info_hyb_mgga_x_revm11;
extern xc_func_info_type xc_func_info_hyb_mgga_x_revm06;
extern xc_func_info_type xc_func_info_hyb_mgga_x_m06_sx;
extern xc_func_info_type xc_func_info_hyb_mgga_x_cf22d;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_br3p86;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_tpss0;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_b94_hyb;
extern xc_func_info_type xc_func_info_hyb_mgga_x_m05;
extern xc_func_info_type xc_func_info_hyb_mgga_x_m05_2x;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_b88b95;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_b86b95;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_pw86b95;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_bb1k;
extern xc_func_info_type xc_func_info_hyb_mgga_x_m06_hf;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_mpw1b95;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_mpwb1k;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_x1b95;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_xb1k;
extern xc_func_info_type xc_func_info_hyb_mgga_x_m06;
extern xc_func_info_type xc_func_info_hyb_mgga_x_m06_2x;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_pw6b95;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_pwb6k;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_tpssh;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_revtpssh;
extern xc_func_info_type xc_func_info_hyb_mgga_x_mvsh;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_wb97m_v;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_b0kcis;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_mpw1kcis;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_mpwkcis1k;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_pbe1kcis;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_tpss1kcis;
extern xc_func_info_type xc_func_info_hyb_mgga_x_revscan0;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_b98;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_gas22;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_r2scanh;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_r2scan0;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_r2scan50;
extern xc_func_info_type xc_func_info_hyb_mgga_x_wr2scan;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_edmggah;
extern xc_func_info_type xc_func_info_hyb_mgga_x_js18;
extern xc_func_info_type xc_func_info_hyb_mgga_x_pjs18;
extern xc_func_info_type xc_func_info_hyb_mgga_xc_lc_tmlyp;

const xc_func_info_type *xc_hyb_mgga_known_funct[] = {
  &xc_func_info_hyb_mgga_x_dldf,
  &xc_func_info_hyb_mgga_x_ms2h,
  &xc_func_info_hyb_mgga_x_mn12_sx,
  &xc_func_info_hyb_mgga_x_scan0,
  &xc_func_info_hyb_mgga_x_mn15,
  &xc_func_info_hyb_mgga_x_bmk,
  &xc_func_info_hyb_mgga_x_tau_hcth,
  &xc_func_info_hyb_mgga_x_m08_hx,
  &xc_func_info_hyb_mgga_x_m08_so,
  &xc_func_info_hyb_mgga_x_m11,
  &xc_func_info_hyb_mgga_x_revm11,
  &xc_func_info_hyb_mgga_x_revm06,
  &xc_func_info_hyb_mgga_x_m06_sx,
  &xc_func_info_hyb_mgga_x_cf22d,
  &xc_func_info_hyb_mgga_xc_br3p86,
  &xc_func_info_hyb_mgga_xc_tpss0,
  &xc_func_info_hyb_mgga_xc_b94_hyb,
  &xc_func_info_hyb_mgga_x_m05,
  &xc_func_info_hyb_mgga_x_m05_2x,
  &xc_func_info_hyb_mgga_xc_b88b95,
  &xc_func_info_hyb_mgga_xc_b86b95,
  &xc_func_info_hyb_mgga_xc_pw86b95,
  &xc_func_info_hyb_mgga_xc_bb1k,
  &xc_func_info_hyb_mgga_x_m06_hf,
  &xc_func_info_hyb_mgga_xc_mpw1b95,
  &xc_func_info_hyb_mgga_xc_mpwb1k,
  &xc_func_info_hyb_mgga_xc_x1b95,
  &xc_func_info_hyb_mgga_xc_xb1k,
  &xc_func_info_hyb_mgga_x_m06,
  &xc_func_info_hyb_mgga_x_m06_2x,
  &xc_func_info_hyb_mgga_xc_pw6b95,
  &xc_func_info_hyb_mgga_xc_pwb6k,
  &xc_func_info_hyb_mgga_xc_tpssh,
  &xc_func_info_hyb_mgga_xc_revtpssh,
  &xc_func_info_hyb_mgga_x_mvsh,
  &xc_func_info_hyb_mgga_xc_wb97m_v,
  &xc_func_info_hyb_mgga_xc_b0kcis,
  &xc_func_info_hyb_mgga_xc_mpw1kcis,
  &xc_func_info_hyb_mgga_xc_mpwkcis1k,
  &xc_func_info_hyb_mgga_xc_pbe1kcis,
  &xc_func_info_hyb_mgga_xc_tpss1kcis,
  &xc_func_info_hyb_mgga_x_revscan0,
  &xc_func_info_hyb_mgga_xc_b98,
  &xc_func_info_hyb_mgga_xc_gas22,
  &xc_func_info_hyb_mgga_xc_r2scanh,
  &xc_func_info_hyb_mgga_xc_r2scan0,
  &xc_func_info_hyb_mgga_xc_r2scan50,
  &xc_func_info_hyb_mgga_x_wr2scan,
  &xc_func_info_hyb_mgga_xc_edmggah,
  &xc_func_info_hyb_mgga_x_js18,
  &xc_func_info_hyb_mgga_x_pjs18,
  &xc_func_info_hyb_mgga_xc_lc_tmlyp,
  NULL
};
