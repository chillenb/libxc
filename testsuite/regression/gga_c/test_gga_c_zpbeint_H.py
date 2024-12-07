
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zpbeint_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.215289033948523e-02, -2.283324530398973e-02, -1.747840860800438e-02, -1.319122040057002e-02, -1.569853905510373e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zpbeint_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.686276796221159e-02, 1.189680404667983e+00, -3.189201273439594e-02, 1.150462577923147e+02, -1.121859537167206e-02, 1.105375135921113e+02, -1.419112211725487e-02, 1.788749709996894e+00, -2.001690237919110e-03, -6.965489951221249e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zpbeint_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.286132450868928e-02, 2.572264901737855e-02, 1.286132450868928e-02, 2.773452024075620e-03, 5.546904048151241e-03, 2.773452024075620e-03, -2.368058938017989e-02, -4.736117876035979e-02, -2.368058938017989e-02, -1.839452251849211e-01, -3.678904503698422e-01, -1.839452251849211e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
