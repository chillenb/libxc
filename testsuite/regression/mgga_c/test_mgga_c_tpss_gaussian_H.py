
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpss_gaussian_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.659977450494912e-16, -3.118079983566956e-02, -2.518518753711948e-02, -1.327196219152886e-02, -1.569790621002125e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpss_gaussian_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.412737069683964e-02, -6.412810237148274e-02, -3.508811108098497e-02, -2.506001728069859e-01, -2.872211520383755e-02, -1.951180632676889e-01, -1.571598845292718e-02, -9.051731111310296e-02, -2.001663505771958e-03, -7.071279902173426e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.623014187042609e+00, 5.246028374085217e+00, 2.623014187042609e+00, 2.611147685667045e-02, 5.222295371334090e-02, 2.611147685667045e-02, 2.231088409632864e-01, 4.462176819265727e-01, 2.231088409632864e-01, 4.386300201909211e+01, 8.772600403818421e+01, 4.386300201909211e+01, 8.904902355503965e+06, 1.780980471100793e+07, 8.904902355503965e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.255848452771801e+00, -6.182507926244350e+00, -1.764588031562576e-79, -1.735823258748167e-79, -7.403799837903862e-77, -7.376898574840270e-77, -4.487301961655919e-59, -4.487185716430095e-59, -7.264645195613155e-44, -7.264645252252261e-44]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
