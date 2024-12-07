
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_thakkar_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_thakkar", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.009462568425104e+00, 1.683042357687904e+00, 6.214300170062054e-01, 7.240652484608734e-02, 1.538257863879364e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_thakkar_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_thakkar", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.358687612337365e+00, 3.693231333548008e-16, 2.613540288892281e+00, 3.169467318928483e-16, 8.752665967799660e-01, 6.046700395579455e-17, 6.952187125500139e-02, 4.297208545765148e-17, 8.825393135957120e-04, -2.678610960027914e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_thakkar_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_thakkar", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.469958220674154e-01, 0.000000000000000e+00, 0.000000000000000e+00, 8.896389740885426e-02, 0.000000000000000e+00, 0.000000000000000e+00, 3.504123578704131e-01, 0.000000000000000e+00, 0.000000000000000e+00, 5.758079852946545e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.343429356924493e+03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
