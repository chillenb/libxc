
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_apbeint_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.035146234792744e+00, 1.688593789099421e+00, 6.147405898076006e-01, 6.206326420553575e-02, 1.596956661147757e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_apbeint_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.388868537932807e+00, 5.017213053614917e-16, 2.671490551892117e+00, 2.200731157305936e-16, 8.869900596373090e-01, 8.265327524938523e-17, 8.488947812599995e-02, 2.697444276361183e-17, 2.660220266870649e-04, -2.268988289811803e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_apbeint_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.665806806653500e-02, 0.000000000000000e+00, 0.000000000000000e+00, 6.634422419108763e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.004603191176445e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.087907844207348e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.098092868868377e-01, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
