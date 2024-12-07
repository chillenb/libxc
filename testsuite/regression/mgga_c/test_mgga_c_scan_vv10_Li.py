
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_scan_vv10_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.565506030176055e-02, -2.438854625799005e-02, -1.481541165534285e-02, -2.012273169483131e-04, -3.855018850928288e-08, -1.067648954226573e-03, -5.788917075476808e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_scan_vv10_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.481283186180276e-02, -3.474003143061156e-02, -3.782393253332390e-02, -3.775854944483058e-02, -4.863339842092504e-02, -4.865336083502241e-02, -1.592931098268111e-03, -1.591262284201089e-01, -1.347771748322780e-02, -9.398266853483453e-02, -2.008555855529216e-03, -2.023114096014518e-03, -1.075222067322012e-05, -1.347139221366846e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scan_vv10_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.683820280357966e-05, 3.367640560715931e-05, 1.683820280357966e-05, 7.435796006136708e-05, 1.487159201227341e-04, 7.435796006136708e-05, 1.798254952914109e-02, 3.596509905828217e-02, 1.798254952914109e-02, 2.110748210649237e+00, 4.221496421298474e+00, 2.110748210649237e+00, 7.864623676175758e+01, 1.572924735235152e+02, 7.864623676175758e+01, 2.991368446931750e+00, 5.982736893863499e+00, 2.991368446931750e+00, 6.472757637041947e+03, 1.294551527408389e+04, 6.472757637041947e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scan_vv10_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scan_vv10_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.563675097218774e-03, -1.563675097218773e-03, -2.253199918151447e-03, -2.253199918151446e-03, -7.581468418189831e-03, -7.581468418189825e-03, -8.022330382421826e-02, -8.022330382420054e-02, -1.880805053924900e-01, -1.880805052405384e-01, -3.154606704576700e-11, -3.154606704576701e-11, -2.986261399710777e-23, -2.986261399710778e-23]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
