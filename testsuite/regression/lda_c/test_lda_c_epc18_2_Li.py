
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_epc18_2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.321169665695134e-02, -8.142397593830238e-02, -3.975052013990836e-03, -4.095209882501757e-07, -1.305933400649539e-11, -2.353941451669200e-07, -1.028612419795515e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_epc18_2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_epc18_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.997956576259179e-03, 3.034333687214344e-03, 5.390054609603777e-03, 5.519321088944608e-03, -8.064092755729998e-03, -8.074339108091466e-03, -4.099451977476893e-07, -1.231345569061668e-03, -1.305984227547001e-11, -7.665287899464066e-05, -4.656898738256364e-07, -4.760004848660794e-07, -1.398179377195331e-12, -3.891540203248744e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
