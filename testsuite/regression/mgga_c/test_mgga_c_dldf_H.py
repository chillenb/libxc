
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_dldf_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.483657391384560e-15, -3.118079983567378e-02, -2.518518753713570e-02, -1.327196218908067e-02, -1.569788767680283e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_dldf_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.156022009873050e-02, -2.557890552045181e-01, -3.508811108098573e-02, -2.430006092179204e-01, -2.872211520384063e-02, -1.892604446596727e-01, -1.571598845239644e-02, 3.494584370852801e-01, -2.001662928803568e-03, 2.726812051007818e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_dldf_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.290913757504731e+00, 0.000000000000000e+00, -3.230690123770064e+21, 1.016100811207673e-01, 0.000000000000000e+00, -3.124099416533992e+21, 8.899062714412654e-01, 0.000000000000000e+00, -2.416869846886508e+21, 2.923147664627514e+02, 0.000000000000000e+00, 1.070456859532401e+17, 1.469665910782340e+08, 0.000000000000000e+00, 8.358638748022629e+15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_dldf_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.078809437036652e+00, 0.000000000000000e+00, -2.797610855251388e-39, 0.000000000000000e+00, -4.995414211320001e-38, 0.000000000000000e+00, -6.992788755972477e-35, 0.000000000000000e+00, -4.168384509940232e-28, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
