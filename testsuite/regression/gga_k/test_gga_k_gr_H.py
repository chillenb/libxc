
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_gr_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_gr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [3.459134155603210e-01, 7.394403078910295e-01, 5.914436380854666e-01, 5.150347187400954e-01, 6.850131189790659e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_gr_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_gr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [5.491869270095150e-01, 7.428748290711585e-17, -1.980248668519217e-02, 8.020551416779838e-17, -3.452531280190876e-01, 7.262466708037583e-17, -4.974764971951229e-01, 3.270927122899548e-16, -6.849741582501029e-01, -1.823248711654674e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_gr_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_gr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.192899183611812e-01, 0.000000000000000e+00, 0.000000000000000e+00, 5.816356732758560e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.906799273274511e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.526164046858387e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.459646400083933e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
