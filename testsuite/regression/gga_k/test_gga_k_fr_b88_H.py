
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_fr_b88_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.035715766564646e+00, 1.702565241727822e+00, 6.197717621296135e-01, 6.610890872480123e-02, 1.195981228875085e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_fr_b88_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.388323143051290e+00, -9.520897219055956e-18, 2.677795866973262e+00, 1.786671911961991e-16, 9.035239754004092e-01, 9.702575217891280e-17, 7.090613187347247e-02, 3.251571540704970e-17, 7.123450727170421e-04, -3.236348352159035e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_fr_b88_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_fr_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.958350734060988e-02, 0.000000000000000e+00, 0.000000000000000e+00, 7.423146711079362e-02, 0.000000000000000e+00, 0.000000000000000e+00, 2.826642555531943e-01, 0.000000000000000e+00, 0.000000000000000e+00, 4.420835246041031e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.023584851490311e+03, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
