
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pg1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pg1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.774452290925365e+01, 9.644112101748789e+00, 3.221303991474779e+00, 1.392081311793968e-01, 7.412979614079540e-02, 3.084472388730704e+00, 1.356897369842305e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pg1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pg1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.405648986445933e+01, 2.410529394711826e+01, 9.885341149917316e+00, 9.906690343153331e+00, -3.136592885678377e+00, -3.147233968341859e+00, 2.061308264470824e-01, -3.057065625943076e+00, -5.364132960991030e-02, -1.210227402337044e+00, -3.032148342681577e+00, -3.137954916182146e+00, -1.418447087986350e+00, -1.185586587718588e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pg1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pg1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [9.414829790764422e-03, 0.000000000000000e+00, 9.386567666590347e-03, 3.254112616593233e-02, 0.000000000000000e+00, 3.244875204635927e-02, 4.121841622612577e+00, 0.000000000000000e+00, 4.127630515613161e+00, 1.180678149657673e+01, 0.000000000000000e+00, 7.829811571772351e+04, 3.877344338852023e+02, 0.000000000000000e+00, 2.454376727176486e+09, 6.733474755479385e+04, 0.000000000000000e+00, 6.882557364072798e+04, 8.236143114997760e+09, 0.000000000000000e+00, 2.292358375079883e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
