
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bpccac_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bpccac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.220351725775085e-01, -5.778255937586186e-01, -3.609673708829545e-01, -1.374260206463532e-01, -6.501527958482659e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bpccac_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bpccac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.282811073616763e-01, -7.367109913847475e-17, -7.175317719964434e-01, -2.340522506217145e-16, -3.962098069310196e-01, -2.142202163848187e-17, -1.604241656419363e-01, -6.122937954010836e-17, -2.649799673180514e-04, -1.154022923816420e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bpccac_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bpccac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.685911092533160e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.457260980671195e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.858091148872574e-01, 0.000000000000000e+00, 0.000000000000000e+00, -2.567552032840541e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.424699964546575e+02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
