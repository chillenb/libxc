
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_wr2scan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.787649248285103e+00, -1.135110465472151e+00, -1.253678360915237e-01, -5.592084963356717e-02, -3.949182087187397e-03, -1.346114894032680e-05, -1.174062389005189e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_wr2scan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.557612177695022e+00, -2.560141943918290e+00, -1.751386469113795e+00, -1.752921209447163e+00, -2.014692520299001e-01, -2.017025490591771e-01, -9.824592464096442e-02, 7.811112269811108e-04, -8.173147758548056e-03, 8.658149221838142e-09, 5.140436284548227e-04, 8.880089012308674e-04, 1.154231585023271e-10, -8.746432730659797e-28])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_wr2scan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.183873126964053e-04, 0.000000000000000e+00, -3.171890035804726e-04, -1.553348338514188e-03, 0.000000000000000e+00, -1.547851545208648e-03, -1.016245888849399e-02, 0.000000000000000e+00, -1.045613131656336e-02, -1.430849316794007e+00, 0.000000000000000e+00, -2.094112687216602e+01, -2.202392188555768e+00, 0.000000000000000e+00, -1.826576034146201e+01, -2.427657189454197e-01, 0.000000000000000e+00, -2.039654610279709e+01, -5.523164769620911e-06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_wr2scan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.648266235307624e-02, 1.646509810398581e-02, 2.685253970162495e-02, 2.682623198237942e-02, 1.272769075393999e-03, 1.341957166551537e-03, 5.484511463752743e-02, 2.683684734677235e-04, 4.559281028302647e-03, 7.470072144365486e-09, 3.618198544192821e-06, 2.973696326329880e-04, 6.733104799976543e-16, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
