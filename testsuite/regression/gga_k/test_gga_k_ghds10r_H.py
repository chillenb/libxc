
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ghds10r_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.102520468728049e+00, 2.161121827003441e+00, 1.104950926973866e+00, 5.704532184636711e-01, 6.509891770980747e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ghds10r_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.444154857252501e+00, 3.431340090031606e-16, 2.318294082064694e+00, 4.266471366911257e-16, 4.857968450456146e-01, 1.567135275488237e-16, -4.137165743979689e-01, 2.620175777940279e-16, -7.128171082549924e-01, -1.334476409064946e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ghds10r_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.192899183611812e-01, 0.000000000000000e+00, 0.000000000000000e+00, 5.816356732758560e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.906799273274511e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.526164046858387e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.459646400083933e+06, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
