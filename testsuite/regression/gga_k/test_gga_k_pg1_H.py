
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pg1_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pg1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.038115998514603e+00, 1.846054913758006e+00, 8.266658953912521e-01, 5.084694539479426e-01, 6.849985057062444e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pg1_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pg1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.385876305047580e+00, 4.234395827170800e-16, 2.456968414562818e+00, 3.330621934883938e-16, 5.143513604370739e-01, 1.172058854717086e-16, -5.080298115540047e-01, 2.490702208550987e-16, -6.849985137050892e-01, -1.231706895681280e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pg1_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pg1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.684755330910875e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.878861040408532e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.885663141650345e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.525726420931515e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.459646400083933e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
