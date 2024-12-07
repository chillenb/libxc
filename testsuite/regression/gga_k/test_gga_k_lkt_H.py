
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_lkt_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lkt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.039071075321319e+00, 1.892614918827589e+00, 8.752188079798854e-01, 5.106406924814834e-01, 6.849985057062444e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_lkt_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lkt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.384916588072852e+00, 1.441848605086531e-16, 2.406586356657456e+00, 3.267001678233153e-16, 4.808014901086985e-01, 5.505224134548764e-17, -4.943177086892100e-01, 3.158456639586135e-16, -6.849985137050892e-01, -1.231706895681280e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_lkt_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lkt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.076123977444765e-01, 0.000000000000000e+00, 0.000000000000000e+00, 3.473324840136910e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.135661057387644e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.514365326131209e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.459646400083933e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
