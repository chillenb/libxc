
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_wc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.220345560888795e-01, -5.762934998000241e-01, -3.546050312177794e-01, -1.306329010455363e-01, -7.394690417341574e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_wc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.282835685056642e-01, -5.518218033960693e-17, -7.230249718991756e-01, -2.118271600608641e-16, -4.144216770061510e-01, 5.123547211748484e-17, -1.220357691419573e-01, -4.042472762902272e-17, -9.847742865271488e-03, -5.622971707680543e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_wc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.680875200637492e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.107221754394992e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.275090305477260e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.869037188729136e+00, 0.000000000000000e+00, 0.000000000000000e+00, -9.464660508847437e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
