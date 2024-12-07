
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wc04_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.349131171531903e-02, -7.426745911653466e-02, -7.955316956076744e-02, -8.792726061497368e-02, -6.092060717103381e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wc04_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.699648601626254e-02, -2.403478463420415e-01, -3.179577675865020e-02, -2.306465675700976e-01, -1.854501499304303e-02, -1.855220611517822e-01, -1.962237200921450e-02, -9.583933576451271e-02, -1.647718380769689e-02, -8.766873230328768e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wc04_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.090090623565381e-02, 2.924820117334720e-06, 2.192774932900085e-06, -2.542433776652689e-02, 4.646917650590301e-06, 3.480665078312204e-06, -1.687196424019187e-01, 3.986771514585075e-05, 2.989990296175629e-05, -1.035097200501987e+01, 1.702787037828197e-03, 1.277088117642038e-03, -5.126345441843766e+04, 7.157809872575427e-22, 5.368349375325281e-22]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
