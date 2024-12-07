
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_golden_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_golden", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.036967732571142e+00, 1.770944532390716e+00, 7.036287743957075e-01, 1.867845846584076e-01, 1.979770088860917e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_golden_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_golden", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.387049318299694e+00, 2.242306166957847e-16, 2.589826687364988e+00, 4.252533805309527e-16, 7.882056723758313e-01, 1.392031193076916e-16, -8.038747081899134e-02, 8.584691452040105e-17, -1.977408733763043e-01, -4.987742710155899e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_golden_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_golden", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.211281986376746e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.680280833908028e-01, 0.000000000000000e+00, 0.000000000000000e+00, 8.397420122793032e-01, 0.000000000000000e+00, 0.000000000000000e+00, 4.408918357590894e+01, 0.000000000000000e+00, 0.000000000000000e+00, 4.216756266909140e+05, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
