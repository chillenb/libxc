
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_q1d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q1d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.219720452348458e-01, -5.568923726862042e-01, -2.523676432341918e-01, -9.880434764251670e-04, -5.766153267001310e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_q1d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q1d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.283792460925208e-01, -3.814324973123412e-17, -7.990330327226695e-01, -1.560880415751814e-16, -7.416857925070109e-01, -4.788996386282176e-18, -5.084448357229474e-03, -5.897074516071625e-19, -2.306465275786354e-07, -1.564576053043424e-24])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_q1d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q1d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.406273572292835e-02, 0.000000000000000e+00, 0.000000000000000e+00, 2.624826468940886e-02, 0.000000000000000e+00, 0.000000000000000e+00, 8.849202817332600e-01, 0.000000000000000e+00, 0.000000000000000e+00, 4.240197669062681e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.228698510442470e-01, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
