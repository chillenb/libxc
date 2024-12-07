
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_absp4_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_absp4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [8.766111411548855e-01, 9.026009143856025e-01, 4.539280078687358e-01, 2.351412106991085e-01, 2.937766112299787e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_absp4_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_absp4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.449296685119059e+00, 2.372325802484566e-16, 9.673695915130696e-01, 2.270716671403752e-16, 1.857951682768605e-01, 7.744028166814340e-17, -1.895163762913147e-01, 1.213354509445857e-16, -2.936753548741808e-01, -3.171689988992421e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_absp4_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_absp4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.797984212526506e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.494149542349296e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.246483393342415e+00, 0.000000000000000e+00, 0.000000000000000e+00, 6.544442739529961e+01, 0.000000000000000e+00, 0.000000000000000e+00, 6.259204116997998e+05, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
