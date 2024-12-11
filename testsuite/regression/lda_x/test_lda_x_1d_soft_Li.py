
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_1d_soft_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_soft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.921999316705460e-01, -4.766921111978855e-01, -9.886402608717515e-02, -2.454874623654805e-02, -2.359591844514422e-03, -2.384826658485566e-05, -3.127158572054876e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_1d_soft_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_1d_soft", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.999999999999995e-01, -5.000000000000001e-01, -4.999998764756783e-01, -4.999998808474733e-01, -1.679777828906765e-01, -1.678390789632371e-01, -4.431711767921458e-02, -4.031315023382314e-05, -4.420250661618633e-03, -2.340566518354759e-09, -4.631682799325226e-05, -4.539310751353402e-05, -7.342292053394597e-10, -2.749624753767792e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
