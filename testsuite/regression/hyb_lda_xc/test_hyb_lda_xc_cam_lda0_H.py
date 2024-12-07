
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_lda_xc_cam_lda0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_cam_lda0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.567379698383960e-01, -4.078389001936600e-01, -2.321939535280164e-01, -5.925503695667884e-02, -3.620873636283225e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_lda_xc_cam_lda0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_cam_lda0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.144498297367278e-01, -2.627091453975081e-01, -5.492351718889261e-01, -2.506001728069931e-01, -3.141880455821421e-01, -1.951180632676960e-01, -7.841014821202993e-02, -9.051731111311005e-02, -4.736642508545393e-03, -7.071279902180486e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
