
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_lypr_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lypr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.899060944251295e-15, -7.883368760435867e-15, -2.229670632964285e-14, -2.263270598693193e-13, -1.925789854690417e-68])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_lypr_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lypr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.509618393298178e-15, -1.760773302872894e-01, -1.412171595065345e-15, -1.694595597828856e-01, -9.907869135579169e-15, -9.589118163789837e-02, -1.362280697170434e-13, -1.853718638088999e-02, -1.693715544880295e-66, -1.649209041986737e-61])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_lypr_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_lypr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.749408192300297e-16, 1.584589538161744e-02, 1.187986979993522e-02, -4.839848549775119e-16, 2.305856579466211e-02, 1.727147988242239e-02, 1.131019884468112e-15, 9.754494499905084e-02, 7.315654732685858e-02, 1.868765919527869e-14, 2.252154681560530e-04, 1.689113153225914e-04, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
