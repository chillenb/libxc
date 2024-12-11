
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbrxh_bg_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.636698309847121e+00, -1.158258420656382e+00, -4.135811922332200e-01, -1.492431042285663e-01, -7.659750752034372e-02, -1.169705956211836e+02, -4.575387235745384e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbrxh_bg_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.133755391382619e+00, -2.135818402880848e+00, -1.464617705380649e+00, -1.465970478879583e+00, -4.081003954672450e-01, -4.063426206604275e-01, -1.935396662460682e-01, 7.175649620374539e+00, -7.721781975841160e-02, 5.996070715040565e+01, 2.527536336797810e+02, 7.089268974173035e+00, 7.692992218630224e+06, -2.339466723001917e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxh_bg_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.978646314443272e-04, 0.000000000000000e+00, -1.966797756771733e-04, -1.012276662783806e-03, 0.000000000000000e+00, -1.007986748326784e-03, -1.660639675932469e-01, 0.000000000000000e+00, -1.690055392041013e-01, -2.758231270940342e+00, 0.000000000000000e+00, -1.574741330882202e+05, -1.324526633814139e+02, 0.000000000000000e+00, -9.823028623418918e+10, -9.406232341520305e+04, 0.000000000000000e+00, -1.333737039764843e+05, -2.756126430647408e+11, 0.000000000000000e+00, -4.965033232489231e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxh_bg_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxh_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.746768627781796e-03, -6.735459199793098e-03, -1.157074028699787e-02, -1.155114968659372e-02, -2.631663201260100e-02, -2.674863591790316e-02, -6.969103165992450e-02, -6.207624491035486e-05, -2.084904207345092e-01, -3.138844825588677e-06, -9.830154186259436e-07, -6.350077341847037e-05, -7.910577345819194e-12, -1.423763502291663e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
