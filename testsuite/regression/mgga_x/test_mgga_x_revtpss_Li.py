
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.996937807965659e+00, -1.384502134830239e+00, -3.792841441119765e-01, -1.790679594718642e-01, -7.440283338789914e-02, -2.053745905397894e-02, -3.838585904368181e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.367330115186916e+00, -2.369512271564766e+00, -1.677081468194108e+00, -1.678860212433046e+00, -3.432593159251480e-01, -3.444704928196132e-01, -2.108756853903869e-01, -2.609111414186119e-02, -7.784769457351819e-02, -8.296413305157013e-04, -2.742838671951816e-02, -2.723266425828722e-02, -5.541550226732571e-04, -3.939539840817354e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.348428667702541e-03, 0.000000000000000e+00, -1.346772679972561e-03, -2.470157653534429e-03, 0.000000000000000e+00, -2.465504712070893e-03, -8.427081021987697e-02, 0.000000000000000e+00, -8.345826559156282e-02, -3.565092330136976e+01, 0.000000000000000e+00, -4.352781648750262e-01, -5.952615223686899e+01, 0.000000000000000e+00, -2.785411237651398e+00, -4.423051982552123e-01, 0.000000000000000e+00, -4.130531457693812e-01, -2.027677897781295e+00, 0.000000000000000e+00, -2.902412778350461e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtpss_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.009449335261402e-02, 7.023470702401029e-02, 4.398859901501104e-02, 4.408835565868383e-02, 2.079331839875053e-03, 2.038314604166997e-03, 1.289206977889731e+00, 1.181165623758641e-10, 4.895459026038992e-02, 6.325059422019447e-17, 3.822514513723293e-18, 1.354661166472653e-10, 1.446194674483104e-41, 1.053045407132662e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
