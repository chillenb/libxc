
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pbe4_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.990876641957163e+00, 1.485336007965899e+00, 9.779480797653136e-01, 9.912306202159374e-03, -7.742438556093697e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pbe4_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.429565757253352e+00, 1.367739915489132e-16, 5.948666323712557e-01, 3.321208839449418e-16, 4.825079240152501e-01, 6.713137710192779e-17, 1.115619222251284e-01, 5.446473627608954e-18, -1.284986420886391e-04, 2.014584118780478e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pbe4_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.709314055968929e+00, 0.000000000000000e+00, 0.000000000000000e+00, 8.735631211319095e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.505857496700714e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.069785690085989e+01, 0.000000000000000e+00, 0.000000000000000e+00, -4.331020147110639e-01, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
