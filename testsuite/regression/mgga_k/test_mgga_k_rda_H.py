
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_rda_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([3.524662768739421e+00, 8.804733365049194e-01, 4.381027344138282e-01, 3.120920678304049e-01, 6.841317580086449e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_rda_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.069721982377233e+00, 6.613453915846553e-16, 1.112226298554701e+00, 2.833713247244599e-16, 1.234244550933540e-01, 6.649263103481548e-17, -5.288427202688892e-01, 1.371786329543777e-16, -6.864399801208210e-01, -3.180148336711019e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_rda_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.077797118923748e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.650004122050497e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.325094644850962e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.180749597198087e+02, 0.000000000000000e+00, 0.000000000000000e+00, 1.459643912475783e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_rda_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([2.514736071620535e-01, 0.000000000000000e+00, 1.193688097026266e-01, 0.000000000000000e+00, 1.029565079469868e-01, 0.000000000000000e+00, 1.792876152831596e-02, 0.000000000000000e+00, 1.236950747506703e-07, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
