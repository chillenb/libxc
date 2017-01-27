/* 
  This file was generated automatically with ../scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  Maple source      : ../maple/lda_k_zlp.mpl
  Type of functional: work_lda
*/

static void
func0(const XC(func_type) *p, XC(lda_work_t) *r)
{
  double t1, t2, t3, t5, t6, t9, t12, t17;
  double t20, t23, t24, t33, t34, t37;


  t1 = r->rs * r->rs;
  t2 = 0.1e1 / t1;
  t3 = 0.1e1 / r->rs;
  t5 = 0.10e1 + 0.82244487449819879626e3 * r->rs;
  t6 = log(t5);
  t9 = 0.10e1 - 0.12158869621628240327e-2 * t3 * t6;
  r->e = 0.20081986091395377340e1 * t2 * t9;

  if(r->order < 1) return;

  t12 = 0.1e1 / t1 / r->rs;
  t17 = 0.1e1 / t5;
  t20 = 0.12158869621628240327e-2 * t2 * t6 - 0.10000000000000000000e1 * t3 * t17;
  r->dedrs = -0.40163972182790754680e1 * t12 * t9 + 0.20081986091395377340e1 * t2 * t20;
  r->dedz = 0;

  if(r->order < 2) return;

  t23 = t1 * t1;
  t24 = 0.1e1 / t23;
  t33 = t5 * t5;
  t34 = 0.1e1 / t33;
  t37 = -0.24317739243256480654e-2 * t12 * t6 + 0.20000000000000000000e1 * t2 * t17 + 0.82244487449819879626e3 * t3 * t34;
  r->d2edrs2 = 0.12049191654837226404e2 * t24 * t9 - 0.80327944365581509360e1 * t12 * t20 + 0.20081986091395377340e1 * t2 * t37;
  r->d2edz2 = 0;
  r->d2edrsz = 0;

  if(r->order < 3) return;

  r->d3edrs3 = -0.48196766619348905616e2 / t23 / r->rs * t9 + 0.36147574964511679212e2 * t24 * t20 - 0.12049191654837226404e2 * t12 * t37 + 0.20081986091395377340e1 * t2 * (0.72953217729769441962e-2 * t24 * t6 - 0.60000000000000000000e1 * t12 * t17 - 0.24673346234945963888e4 * t2 * t34 - 0.13528311431767159373e7 * t3 / t33 / t5);
  r->d3edz3 = 0;
  r->d3edrs2z = 0;
  r->d3edrsz2 = 0;

  if(r->order < 4) return;


}

static void
func1(const XC(func_type) *p, XC(lda_work_t) *r)
{
  double t1, t2, t4, t5, t7, t8, t9, t10;
  double t11, t13, t14, t17, t20, t21, t26, t29;
  double t32, t34, t36, t37, t39, t40, t41, t50;
  double t51, t54, t57, t59, t61, t62, t64, t91;
  double t93;


  t1 = 0.10e1 + r->zeta;
  t2 = pow(t1, 0.16666666666666666667e1);
  t4 = 0.10e1 - r->zeta;
  t5 = pow(t4, 0.16666666666666666667e1);
  t7 = 0.5e0 * t2 + 0.5e0 * t5;
  t8 = r->rs * r->rs;
  t9 = 0.1e1 / t8;
  t10 = t7 * t9;
  t11 = 0.1e1 / r->rs;
  t13 = 0.10e1 + 0.82244487449819879626e3 * r->rs;
  t14 = log(t13);
  t17 = 0.10e1 - 0.12158869621628240327e-2 * t11 * t14;
  r->e = 0.20081986091395377340e1 * t10 * t17;

  if(r->order < 1) return;

  t20 = 0.1e1 / t8 / r->rs;
  t21 = t7 * t20;
  t26 = 0.1e1 / t13;
  t29 = 0.12158869621628240327e-2 * t9 * t14 - 0.10000000000000000000e1 * t11 * t26;
  r->dedrs = -0.40163972182790754680e1 * t21 * t17 + 0.20081986091395377340e1 * t10 * t29;
  t32 = pow(t1, 0.6666666666666666667e0);
  t34 = pow(t4, 0.6666666666666666667e0);
  t36 = 0.83333333333333333335e0 * t32 - 0.83333333333333333335e0 * t34;
  t37 = t36 * t9;
  r->dedz = 0.20081986091395377340e1 * t37 * t17;

  if(r->order < 2) return;

  t39 = t8 * t8;
  t40 = 0.1e1 / t39;
  t41 = t7 * t40;
  t50 = t13 * t13;
  t51 = 0.1e1 / t50;
  t54 = -0.24317739243256480654e-2 * t20 * t14 + 0.20000000000000000000e1 * t9 * t26 + 0.82244487449819879626e3 * t11 * t51;
  r->d2edrs2 = 0.12049191654837226404e2 * t41 * t17 - 0.80327944365581509360e1 * t21 * t29 + 0.20081986091395377340e1 * t10 * t54;
  t57 = pow(t1, -0.3333333333333333333e0);
  t59 = pow(t4, -0.3333333333333333333e0);
  t61 = 0.55555555555555555559e0 * t57 + 0.55555555555555555559e0 * t59;
  t62 = t61 * t9;
  r->d2edz2 = 0.20081986091395377340e1 * t62 * t17;
  t64 = t36 * t20;
  r->d2edrsz = -0.40163972182790754680e1 * t64 * t17 + 0.20081986091395377340e1 * t37 * t29;

  if(r->order < 3) return;

  r->d3edrs3 = -0.48196766619348905616e2 * t7 / t39 / r->rs * t17 + 0.36147574964511679212e2 * t41 * t29 - 0.12049191654837226404e2 * t21 * t54 + 0.20081986091395377340e1 * t10 * (0.72953217729769441962e-2 * t40 * t14 - 0.60000000000000000000e1 * t20 * t26 - 0.24673346234945963888e4 * t9 * t51 - 0.13528311431767159373e7 * t11 / t50 / t13);
  t91 = pow(t1, -0.13333333333333333333e1);
  t93 = pow(t4, -0.13333333333333333333e1);
  r->d3edz3 = 0.20081986091395377340e1 * (-0.18518518518518518518e0 * t91 + 0.18518518518518518518e0 * t93) * t9 * t17;
  r->d3edrs2z = 0.12049191654837226404e2 * t36 * t40 * t17 - 0.80327944365581509360e1 * t64 * t29 + 0.20081986091395377340e1 * t37 * t54;
  r->d3edrsz2 = -0.40163972182790754680e1 * t61 * t20 * t17 + 0.20081986091395377340e1 * t62 * t29;

  if(r->order < 4) return;


}

void 
XC(lda_k_zlp_func)(const XC(func_type) *p, XC(lda_work_t) *r)
{
  if(p->nspin == XC_UNPOLARIZED)
    func0(p, r);
  else
    func1(p, r);
}

#define maple2c_order 3
#define maple2c_func  XC(lda_k_zlp_func)