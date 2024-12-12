
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_gea4_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.200318989953135e+01, -7.399311523005765e-01, 6.396530258522020e-01, 5.468122050749300e-01, 2.267048738971982e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_gea4_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.256514700072928e+00, -6.080522432613743e-16, 2.478979058144304e+00, -1.452157330110609e-16, 8.589904779232780e-01, 1.464826178547030e-16, 1.652421380518767e-02, 2.878178628430878e-16, 2.382799863466397e+01, -2.199435666898680e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea4_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.506810465480464e-01, 0.000000000000000e+00, 0.000000000000000e+00, 9.557040925309569e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.813333421613371e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.204831733342464e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.281730135739212e+08, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea4_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([1.114989336118692e-01, 0.000000000000000e+00, 1.549207939936373e-01, 0.000000000000000e+00, 1.624440797820432e-01, 0.000000000000000e+00, 1.772234144061925e-01, 0.000000000000000e+00, 2.177410183692441e+01, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
