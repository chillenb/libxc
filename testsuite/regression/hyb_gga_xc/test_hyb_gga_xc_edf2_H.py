
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_edf2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_edf2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.376829174744965e-01, -4.956097246677423e-01, -3.046570639485623e-01, -1.121522075912153e-01, -3.540392216896278e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_edf2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_edf2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.140561584083500e-01, -2.219675306467162e-01, -6.273793350319800e-01, -2.263726676624917e-01, -3.604444163617764e-01, -1.776285112139514e-01, -9.663457735465347e-02, -4.960796325585599e-02, -1.076421523202303e-02, -3.564513347716613e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_edf2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_edf2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.205508481386539e-02, 1.733484377217513e-02, 1.299615339216251e-02, -1.461950794036089e-02, 2.754123022163692e-02, 2.062911492180864e-02, -9.674095114447449e-02, 2.362774091064861e-01, 1.772028213428826e-01, -5.886583115636444e+00, 1.008859501631528e+01, 7.566433460447121e+00, -2.909825386909339e+04, 4.186344149728749e-18, 3.139753414610310e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
