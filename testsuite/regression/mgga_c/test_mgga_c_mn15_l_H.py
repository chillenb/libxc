
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn15_l_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.670687225770537e-01, -3.369643014136574e-02, -3.205887466412884e-02, -1.989972471405878e-01, -3.709489064576812e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn15_l_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.591886635117581e-01, 4.587466200886846e+02, -4.053546577785505e-02, -2.734347810000222e-01, 2.615479310882906e-02, -1.856548998463993e-01, -9.724410483499611e-02, -1.218800742151262e+00, -4.724988988358848e-02, -1.670473095678985e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_l_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.066632888948589e-01, 8.133265777897178e-01, 4.066632888948589e-01, -4.947065551540751e-02, -9.894131103081502e-02, -4.947065551540751e-02, 4.764024466590610e-01, 9.528048933181219e-01, 4.764024466590610e-01, 6.790434634773574e+02, 1.358086926954715e+03, 6.790434634773574e+02, 1.868822873545835e+08, 3.737645747091671e+08, 1.868822873545835e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_l_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.764472915153422e+00, 2.763079826325087e+00, 3.343211422917785e-03, 3.320809089481372e-03, -7.539145473963096e-02, -7.523995235822835e-02, -1.633178839223604e-01, -1.633169056524887e-01, -4.411116672265024e-05, -4.411116697575049e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
